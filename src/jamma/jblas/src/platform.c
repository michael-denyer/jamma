/**
 * platform.c — ISA detection and dispatch table initialisation for jblas.
 *
 * Detects the best available SIMD ISA at runtime (AVX2 on x86_64, NEON on
 * aarch64) and populates the global jblas_dispatch table with the appropriate
 * microkernel function pointers.  jblas_init() is idempotent: repeated calls
 * are safe and cheap (guarded by a static flag).
 *
 * Current status:
 *   - AVX2 path: fully wired (ddot, dnrm2, daxpy, dscal, dgemv, dgemm 6x8)
 *   - NEON path: dgemm 8x4 wired; level 1/2 still dispatch to generic
 *   - dgemm: three-level Goto/BLIS blocking loop, ISA microkernel dispatched
 */

#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  /* npy_intp */
#include "jblas.h"

/* Global dispatch table — set once by jblas_init(), then read-only. */
jblas_dispatch_t jblas_dispatch;

/* Initialisation guard */
static int _initialized = 0;

/* Active ISA name — set during jblas_init() */
static const char *_isa_name = "generic";

/* Maximum thread count from init time — jblas_set_n_threads clamps to this
 * to prevent packed_A OOB access (Pitfall 5). */
static int _init_max_threads = 0;

/* ---------------------------------------------------------------------------
 * x86_64 AVX2 detection
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__) || defined(_M_X64)

#include <cpuid.h>

/**
 * _xgetbv_asm — Read extended control register via inline assembly.
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
 * _detect_avx2 — Return 1 if AVX2 is available and enabled by the OS.
 *
 * Checks:
 *   1. CPUID leaf 7, subleaf 0, EBX bit 5 = AVX2
 *   2. CPUID leaf 1, ECX bit 27 = OSXSAVE (OS has enabled XSAVE)
 *   3. XGETBV(0) bits 1:2 = YMM state saved by OS (required for AVX/AVX2)
 */
static int _detect_avx2(void) {
    unsigned int eax, ebx, ecx, edx;

    /* Check OSXSAVE first (leaf 1) */
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return 0;
    /* ECX bit 27 = OSXSAVE */
    if (!((ecx >> 27) & 1))
        return 0;
    /* Check YMM state: XCR0 bits 1:2 (SSE + AVX state) */
    unsigned long long xcr0 = _xgetbv_asm(0);
    if ((xcr0 & 0x6) != 0x6)
        return 0;

    /* Check AVX2: leaf 7, subleaf 0, EBX bit 5 */
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return 0;
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
 * jblas_init — Detect ISA and populate dispatch table.
 *
 * Thread safety: called from PyInit__jblas under the GIL during module import.
 * No additional synchronization is needed; fork() children inherit the
 * already-initialized state.
 * ---------------------------------------------------------------------------
 */
int jblas_init(void) {
    if (_initialized)
        return 0;

    /* Cache ISA detection result once — avoid redundant CPUID/hwcap calls */
#if defined(__x86_64__) || defined(_M_X64)
    int has_simd = _detect_avx2();
#elif defined(__aarch64__)
    int has_simd = _detect_neon();
#else
    int has_simd = 0;
#endif

    /* Wire Level 1/2 dispatch table */
#if defined(__x86_64__) || defined(_M_X64)
    if (has_simd) {
        _isa_name = "AVX2";
        jblas_dispatch.ddot  = jblas_ddot_avx2;
        jblas_dispatch.dnrm2 = jblas_dnrm2_avx2;
        jblas_dispatch.daxpy = jblas_daxpy_avx2;
        jblas_dispatch.dscal = jblas_dscal_avx2;
        jblas_dispatch.dgemv = jblas_dgemv_avx2;
    } else {
        _isa_name = "generic";
        jblas_dispatch.ddot  = jblas_ddot_generic;
        jblas_dispatch.dnrm2 = jblas_dnrm2_generic;
        jblas_dispatch.daxpy = jblas_daxpy_generic;
        jblas_dispatch.dscal = jblas_dscal_generic;
        jblas_dispatch.dgemv = jblas_dgemv_generic;
    }

#elif defined(__aarch64__)
    if (has_simd) {
        /* NEON has a dgemm 8x4 microkernel (Phase 78).
         * Level 1/2 NEON microkernels are planned for a future phase;
         * those dispatch slots still use generic. */
        _isa_name = "NEON";
    } else {
        _isa_name = "generic";
    }
    /* Level 1/2: always use generic on aarch64 until NEON L1/L2 implemented */
    jblas_dispatch.ddot  = jblas_ddot_generic;
    jblas_dispatch.dnrm2 = jblas_dnrm2_generic;
    jblas_dispatch.daxpy = jblas_daxpy_generic;
    jblas_dispatch.dscal = jblas_dscal_generic;
    jblas_dispatch.dgemv = jblas_dgemv_generic;

#else
    _isa_name = "generic";
    jblas_dispatch.ddot  = jblas_ddot_generic;
    jblas_dispatch.dnrm2 = jblas_dnrm2_generic;
    jblas_dispatch.daxpy = jblas_daxpy_generic;
    jblas_dispatch.dscal = jblas_dscal_generic;
    jblas_dispatch.dgemv = jblas_dgemv_generic;
#endif

    /* Set ISA-specific dgemm blocking parameters.  If no SIMD ISA was
     * detected, dgemm.c generic defaults (MR=4, NR=4, etc.) apply. */
    if (has_simd) {
#if defined(__x86_64__) || defined(_M_X64)
        /* AVX2 blocking: MR=6, NR=8, KC=256, MC=72, NC=4096 */
        JBLAS_MR = 6; JBLAS_NR = 8;
        JBLAS_KC = 256; JBLAS_MC = 72; JBLAS_NC = 4096;
#elif defined(__aarch64__)
        /* NEON blocking: MR=8, NR=4, KC=256, MC=64, NC=4096 */
        JBLAS_MR = 8; JBLAS_NR = 4;
        JBLAS_KC = 256; JBLAS_MC = 64; JBLAS_NC = 4096;
#endif
    }

    /* Initialise dgemm workspace (allocates packed_A/B using blocking params) */
    if (jblas_dgemm_init() != 0) {
        /* Allocation failure makes dgemm unusable.  Fail init so
         * __init__.py falls back to NumPy for ALL operations. */
        return -1;
    }

    /* Wire ISA-specific dgemm microkernel (jblas_dgemm_init set generic) */
    if (has_simd) {
#if defined(__x86_64__) || defined(_M_X64)
        jblas_dgemm_microkernel = jblas_dgemm_micro_avx2;
#elif defined(__aarch64__)
        jblas_dgemm_microkernel = jblas_dgemm_micro_neon;
#endif
    }

    /* Wire blocking dispatch wrapper into the dispatch table */
    jblas_dispatch.dgemm = jblas_dgemm_dispatch_fn;

    /* Try to upgrade dgemm to system BLAS / bundled BLIS.
     * blas_dispatch_init() may replace jblas_dispatch.dgemm with an
     * external wrapper.  Falls through to jblas own dgemm on failure. */
    blas_dispatch_init();

    /* Record init-time thread count for clamping in jblas_set_n_threads */
    _init_max_threads = jblas_n_threads;

    _initialized = 1;
    return 0;
}

/* ---------------------------------------------------------------------------
 * jblas_isa_name — Return active ISA string
 * ---------------------------------------------------------------------------
 */
const char *jblas_isa_name(void) {
    return _isa_name;
}

/* ---------------------------------------------------------------------------
 * Thread control API
 * ---------------------------------------------------------------------------
 */

int jblas_get_n_threads(void) {
    return jblas_n_threads;
}

int jblas_set_n_threads(int n) {
    if (n < 1) return -1;
    int old = jblas_n_threads;
    /* Clamp to init-time allocation to prevent packed_A OOB (Pitfall 5) */
    jblas_n_threads = (n > _init_max_threads) ? _init_max_threads : n;
    return old;
}
