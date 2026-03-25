/**
 * platform.c -- ISA detection and vendor BLAS dispatch initialisation.
 *
 * Detects the best available SIMD ISA at runtime (AVX2 on x86_64, NEON on
 * aarch64) for jlinalg_isa_name() introspection, then calls
 * blas_dispatch_init() to discover and wire vendor BLAS/LAPACK.
 *
 * jlinalg_init() is idempotent: repeated calls are safe and cheap
 * (guarded by a static flag).
 */

#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> /* npy_intp */
#include "jlinalg.h"

/* Initialisation guard */
static int _initialized = 0;

/* Active ISA name -- set during jlinalg_init() */
static const char *_isa_name = "generic";

/* Thread count -- exposed via get/set API */
static int _n_threads = 1;

/* ---------------------------------------------------------------------------
 * x86_64 AVX2 detection
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__) || defined(_M_X64)

#include <cpuid.h>

/**
 * _xgetbv_asm -- Read extended control register via inline assembly.
 *
 * We use inline ASM instead of the _xgetbv() intrinsic because platform.c
 * is compiled without -mavx2/-mxsave (baseline source), and the intrinsic
 * requires <immintrin.h> + -mxsave.  The XGETBV instruction is available
 * on any CPU that advertises OSXSAVE (CPUID leaf 1, ECX bit 27), which we
 * check before calling this.
 */
static unsigned long long _xgetbv_asm(unsigned int index) {
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((unsigned long long)edx << 32) | eax;
}

/**
 * _detect_avx2 -- Return 1 if AVX2 is available and enabled by the OS.
 *
 * Checks:
 *   1. CPUID leaf 7, subleaf 0, EBX bit 5 = AVX2
 *   2. CPUID leaf 1, ECX bit 27 = OSXSAVE (OS has enabled XSAVE)
 *   3. XGETBV(0) bits 1:2 = YMM state saved by OS (required for AVX/AVX2)
 */
static int _detect_avx2(void) {
    unsigned int eax, ebx, ecx, edx;

    /* Check OSXSAVE first (leaf 1) */
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
    /* ECX bit 27 = OSXSAVE */
    if (!((ecx >> 27) & 1)) return 0;
    /* Check YMM state: XCR0 bits 1:2 (SSE + AVX state) */
    unsigned long long xcr0 = _xgetbv_asm(0);
    if ((xcr0 & 0x6) != 0x6) return 0;

    /* Check AVX2: leaf 7, subleaf 0, EBX bit 5 */
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx >> 5) & 1;
}

#endif /* __x86_64__ */

/* ---------------------------------------------------------------------------
 * aarch64 NEON detection
 * ---------------------------------------------------------------------------
 */
#if defined(__aarch64__)

#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>

static int _detect_neon(void) {
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & HWCAP_ASIMD) != 0;
}

#elif defined(__APPLE__)
/* NEON (ASIMD) is architecturally mandatory on Apple Silicon */
static int _detect_neon(void) {
    return 1;
}
#else
/* Conservative fallback: NEON is mandatory per AArch64 spec */
static int _detect_neon(void) {
    return 1;
}
#endif /* __linux__ / __APPLE__ */

#endif /* __aarch64__ */

/* ---------------------------------------------------------------------------
 * jlinalg_init -- Detect ISA and initialise vendor BLAS dispatch.
 *
 * Thread safety: called from PyInit__jlinalg under the GIL during module import.
 * No additional synchronization is needed; fork() children inherit the
 * already-initialized state.
 * ---------------------------------------------------------------------------
 */
int jlinalg_init(void) {
    if (_initialized) return 0;

    /* Detect ISA for jlinalg_isa_name() */
#if defined(__x86_64__) || defined(_M_X64)
    if (_detect_avx2()) {
        _isa_name = "AVX2";
    } else {
        _isa_name = "generic";
    }
#elif defined(__aarch64__)
    if (_detect_neon()) {
        _isa_name = "NEON";
    } else {
        _isa_name = "generic";
    }
#else
    _isa_name = "generic";
#endif

    /* Determine default thread count from environment */
    const char *omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        int t = atoi(omp_threads);
        if (t > 0) _n_threads = t;
    }

    /* Discover and wire vendor BLAS/LAPACK dispatch.
     * blas_dispatch_init() may wire external dgemm, dsyrk, dsyevd, etc.
     * Falls through to numpy-fallback on failure. */
    blas_dispatch_init();

    _initialized = 1;
    return 0;
}

/* ---------------------------------------------------------------------------
 * jlinalg_isa_name -- Return active ISA string
 * ---------------------------------------------------------------------------
 */
const char *jlinalg_isa_name(void) {
    return _isa_name;
}

/* ---------------------------------------------------------------------------
 * Thread control API
 * ---------------------------------------------------------------------------
 */

int jlinalg_get_n_threads(void) {
    return __atomic_load_n(&_n_threads, __ATOMIC_RELAXED);
}

int jlinalg_set_n_threads(int n) {
    if (n < 1) return -1;
    int old = __atomic_load_n(&_n_threads, __ATOMIC_RELAXED);
    __atomic_store_n(&_n_threads, n, __ATOMIC_RELAXED);
    return old;
}
