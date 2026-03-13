/**
 * platform.c — ISA detection and dispatch table initialisation for jblas.
 *
 * Detects the best available SIMD ISA at runtime (AVX2 on x86_64, NEON on
 * aarch64) and populates the global jblas_dispatch table with the appropriate
 * microkernel function pointers.  jblas_init() is idempotent: repeated calls
 * are safe and cheap (guarded by a static flag).
 *
 * Current status:
 *   - AVX2 path: fully wired (ddot, dnrm2, daxpy, dscal, dgemv)
 *   - NEON path: detected but dispatches to generic (microkernels not yet implemented)
 *   - dgemm: stub (aborts with fatal error if called from C; Python layer uses NumPy fallback)
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

/* ---------------------------------------------------------------------------
 * x86_64 AVX2 detection
 * ---------------------------------------------------------------------------
 */
#if defined(__x86_64__) || defined(_M_X64)

#include <cpuid.h>

/**
 * _detect_avx2 — Return 1 if AVX2 is available and enabled by the OS.
 *
 * Checks:
 *   1. CPUID leaf 7, subleaf 0, EBX bit 5 = AVX2
 *   2. CPUID leaf 1, ECX bit 27 = OSXSAVE (OS has enabled XSAVE)
 *   3. _xgetbv(0) bits 1:2 = YMM state saved by OS (required for AVX/AVX2)
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
    unsigned long long xcr0 = _xgetbv(0);
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
 * dgemm stub — safe trap until C implementation is added
 * ---------------------------------------------------------------------------
 */
static void _dgemm_stub(
    npy_intp m, npy_intp n, npy_intp k,
    const double *A, const double *B, double *C)
{
    (void)m; (void)n; (void)k; (void)A; (void)B; (void)C;
    /* This should never be called — pymodule.c does not expose dgemm yet.
     * If it is called, the caller has bypassed the Python layer. */
    fprintf(stderr, "FATAL: jblas_dispatch.dgemm called but not implemented\n");
    fflush(stderr);
    abort();
}

/* ---------------------------------------------------------------------------
 * jblas_init — Detect ISA and populate dispatch table
 * ---------------------------------------------------------------------------
 */
int jblas_init(void) {
    if (_initialized)
        return 0;

#if defined(__x86_64__) || defined(_M_X64)
    if (_detect_avx2()) {
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
    if (_detect_neon()) {
        /* NEON detected but microkernels not yet implemented; report generic
         * to avoid misleading callers into thinking SIMD is active. */
        _isa_name = "generic";
    } else {
        _isa_name = "generic";
    }
    /* All dispatch slots use generic until NEON microkernels are added */
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

    /* dgemm not yet implemented — stub traps accidental calls */
    jblas_dispatch.dgemm = _dgemm_stub;

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
