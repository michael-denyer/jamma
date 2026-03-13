/**
 * jblas.h — Public C API for the JAMMA BLAS compute layer.
 *
 * Declares the ISA dispatch table, jblas_init(), and function signatures for
 * Level 1/2 BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv) plus Level 3
 * (dgemm with three-level Goto/BLIS blocking and ISA-dispatched microkernels).
 *
 * ABI version bump required if any function signature or struct layout changes.
 */

#pragma once

#include <stddef.h>             /* size_t */
#include <numpy/arrayobject.h>  /* npy_intp */

/* Bump this constant whenever the public ABI changes (new fields in
 * jblas_dispatch_t, changed function signatures, etc.). pymodule.c exposes
 * this as a Python-level integer so callers can guard against ABI mismatches. */
#define JBLAS_ABI_VERSION 2

/* ---------------------------------------------------------------------------
 * Function-pointer typedefs for ISA-dispatched microkernels
 * ---------------------------------------------------------------------------
 * incx / incy are stride arguments (always 1 in the current Python API;
 * retained for BLAS compatibility and future strided-array support).
 */

typedef double (*jblas_ddot_fn)(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

typedef double (*jblas_dnrm2_fn)(
    npy_intp n,
    const double *x, int incx);

typedef void (*jblas_daxpy_fn)(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

typedef void (*jblas_dscal_fn)(
    npy_intp n,
    double alpha,
    double *x, int incx);

/* dgemv: y = A*x (no alpha/beta/transpose for the internal primitive).
 * A is row-major (m x n), x is length-n, y is length-m output. */
typedef void (*jblas_dgemv_fn)(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

/* dgemm: C = A*B.  A is (m x k), B is (k x n), C is (m x n). Row-major. */
typedef void (*jblas_dgemm_fn)(
    npy_intp m, npy_intp n, npy_intp k,
    const double *A,
    const double *B,
    double       *C);

/* ---------------------------------------------------------------------------
 * Global dispatch table (set once by jblas_init, then read-only)
 * ---------------------------------------------------------------------------
 */

typedef struct {
    jblas_ddot_fn  ddot;
    jblas_dnrm2_fn dnrm2;
    jblas_daxpy_fn daxpy;
    jblas_dscal_fn dscal;
    jblas_dgemv_fn dgemv;
    jblas_dgemm_fn dgemm;
} jblas_dispatch_t;

extern jblas_dispatch_t jblas_dispatch;

/* ---------------------------------------------------------------------------
 * dgemm microkernel function pointer
 * ---------------------------------------------------------------------------
 * Operates on packed data (see jblas_pack_A / jblas_pack_B).  Different from
 * the dispatch table's jblas_dgemm_fn which takes raw (M, N, K) pointers.
 *
 * Updates C_tile[MR x NR] += packed_A[MR x kc] * packed_B[kc x NR].
 * C is in row-major layout with leading dimension ldc.
 */
typedef void (*jblas_dgemm_micro_fn)(
    npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc);

extern jblas_dgemm_micro_fn jblas_dgemm_microkernel;

/* ---------------------------------------------------------------------------
 * dgemm blocking parameters (ISA-dependent, set by jblas_init() in platform.c)
 * ---------------------------------------------------------------------------
 * AVX2:    MR=6, NR=8,  KC=256, MC=72,  NC=4096
 * NEON:    MR=8, NR=4,  KC=256, MC=64,  NC=4096
 * Generic: MR=4, NR=4,  KC=128, MC=32,  NC=1024
 */
extern int JBLAS_MR, JBLAS_NR, JBLAS_KC, JBLAS_MC, JBLAS_NC;

/* ---------------------------------------------------------------------------
 * dgemm packing workspace buffers (allocated by jblas_dgemm_init())
 * ---------------------------------------------------------------------------
 * packed_A: per-thread (n_threads * MC * KC doubles), 64-byte aligned.
 * packed_B: shared    (KC * NC doubles), 64-byte aligned.
 */
extern double *jblas_packed_A;
extern double *jblas_packed_B;
extern int     jblas_n_threads; /* Thread count at init time */

/* ---------------------------------------------------------------------------
 * dgemm function declarations
 * ---------------------------------------------------------------------------
 */

/* Workspace allocation — called from jblas_init() after ISA detection. */
int  jblas_dgemm_init(void);
void jblas_dgemm_cleanup(void);

/* Dispatch-table wrapper: assigned to jblas_dispatch.dgemm by platform.c.
 * Matches jblas_dgemm_fn signature; calls jblas_dgemm_c with transa=transb=0. */
extern jblas_dgemm_fn jblas_dgemm_dispatch_fn;

/* Full dgemm with transpose support — the implementation behind
 * jblas_dispatch.dgemm and the Python-facing wrapper.
 * transa/transb: 0 = no transpose, 1 = transpose. */
void jblas_dgemm_c(npy_intp M, npy_intp N, npy_intp K,
                   const double *A, npy_intp lda,
                   const double *B, npy_intp ldb,
                   double *C, npy_intp ldc,
                   int transa, int transb);

/* Generic (scalar) microkernel — always available as fallback. */
void jblas_dgemm_micro_generic(npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc);

/* Packing helpers — copy A/B panels into packed format for cache-friendly
 * microkernel access.  trans=0 for no transpose, trans=1 for transpose. */
void jblas_pack_A(const double *A, npy_intp lda,
                  npy_intp mc, npy_intp kc, double *packed,
                  int mr, int trans);
void jblas_pack_B(const double *B, npy_intp ldb,
                  npy_intp kc, npy_intp nr, double *packed,
                  int nr_param, int trans);

/* ---------------------------------------------------------------------------
 * Initialisation and introspection
 * ---------------------------------------------------------------------------
 */

/**
 * jblas_init — Detect ISA and populate jblas_dispatch with the best available
 * microkernel pointers.  Guard with a static flag so it is idempotent across
 * repeated calls (e.g. multiprocessing child-process re-import).
 *
 * Returns: 0 on success, -1 on failure (e.g. workspace allocation for dgemm
 * failed).
 */
int jblas_init(void);

/**
 * jblas_isa_name — Return the active ISA as a C string.
 *
 * Returns: "AVX2", "NEON", or "generic" (never NULL).
 */
const char *jblas_isa_name(void);

/* ---------------------------------------------------------------------------
 * Generic (portable C) microkernel declarations
 * ---------------------------------------------------------------------------
 * These are always compiled and linked; they are used when no SIMD path is
 * available.  Tail handling uses the ISA-specific microkernel + scratch buffer.
 */

double jblas_ddot_generic(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

double jblas_dnrm2_generic(
    npy_intp n,
    const double *x, int incx);

void jblas_daxpy_generic(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

void jblas_dscal_generic(
    npy_intp n,
    double alpha,
    double *x, int incx);

void jblas_dgemv_generic(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

/* ---------------------------------------------------------------------------
 * x86-64 AVX2 microkernel declarations
 * ---------------------------------------------------------------------------
 * These translation units are compiled with -mavx2 -mfma.  They must only be
 * called after jblas_init() has confirmed AVX2 support via CPUID.
 */

#if defined(__x86_64__)

double jblas_ddot_avx2(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

double jblas_dnrm2_avx2(
    npy_intp n,
    const double *x, int incx);

void jblas_daxpy_avx2(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

void jblas_dscal_avx2(
    npy_intp n,
    double alpha,
    double *x, int incx);

void jblas_dgemv_avx2(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

void jblas_dgemm_micro_avx2(npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc);

#endif /* __x86_64__ */

/* ---------------------------------------------------------------------------
 * AArch64 NEON microkernel declarations
 * ---------------------------------------------------------------------------
 * Level 1/2 NEON microkernels are planned for a future phase; until then,
 * aarch64 dispatches to the generic (portable C) kernels for ddot/dnrm2/etc.
 * The dgemm NEON microkernel declaration is active now (wired in Phase 78).
 */

#if defined(__aarch64__)

/* Level 1/2 NEON declarations (uncomment when .c implementations exist):
double jblas_ddot_neon(npy_intp n, const double *x, int incx,
                       const double *y, int incy);
double jblas_dnrm2_neon(npy_intp n, const double *x, int incx);
void   jblas_daxpy_neon(npy_intp n, double alpha,
                         const double *x, int incx, double *y, int incy);
void   jblas_dscal_neon(npy_intp n, double alpha, double *x, int incx);
void   jblas_dgemv_neon(npy_intp m, npy_intp n, const double *A,
                         const double *x, double *y);
*/

void jblas_dgemm_micro_neon(npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc);

#endif /* __aarch64__ */
