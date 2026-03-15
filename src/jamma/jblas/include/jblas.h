/**
 * jblas.h — Public C API for the JAMMA BLAS compute layer.
 *
 * Declares the ISA dispatch table, jblas_init(), and function signatures for
 * Level 1/2 BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv) plus Level 3
 * (dgemm, dsyrk, dsyr2k with three-level Goto/BLIS blocking and
 * ISA-dispatched microkernels).
 *
 * ABI version bump required if any function signature or struct layout changes.
 */

#pragma once

#include <stddef.h>             /* size_t */
#include <pthread.h>            /* pthread_mutex_t — shared across dgemm/dsyrk/dsyr2k */
#include <numpy/arrayobject.h>  /* npy_intp */

/* Bump this constant whenever the public ABI changes (new fields in
 * jblas_dispatch_t, changed function signatures, etc.). pymodule.c exposes
 * this as a Python-level integer so callers can guard against ABI mismatches. */
#define JBLAS_ABI_VERSION 6

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
 * External BLAS dispatch (system BLAS / bundled BLIS discovery)
 * ---------------------------------------------------------------------------
 */

/* Fortran-style dgemm function pointer types for dlopen'd BLAS */
typedef void (*jblas_dgemm_lp64_fn)(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const double *alpha, const double *a, const int *lda,
    const double *b, const int *ldb,
    const double *beta, double *c, const int *ldc);

typedef void (*jblas_dgemm_ilp64_fn)(
    const char *transa, const char *transb,
    const long long *m, const long long *n, const long long *k,
    const double *alpha, const double *a, const long long *lda,
    const double *b, const long long *ldb,
    const double *beta, double *c, const long long *ldc);

/* CBLAS C-interface dgemm: handles row-major natively (no A/B swap needed).
 * Preferred over Fortran interface when available — Accelerate/MKL can
 * choose optimal algorithm for the memory layout. */
enum { JBLAS_CblasRowMajor = 101, JBLAS_CblasNoTrans = 111, JBLAS_CblasTrans = 112 };
typedef void (*jblas_cblas_dgemm_fn)(
    int order, int transa, int transb,
    int m, int n, int k,
    double alpha, const double *a, int lda,
    const double *b, int ldb,
    double beta, double *c, int ldc);

/* Initialise external BLAS dispatch: system BLAS -> bundled BLIS -> own kernels.
 * Called from jblas_init() after ISA detection and dgemm_init().
 * If an external dgemm is found, replaces jblas_dispatch.dgemm with a wrapper.
 * Returns 0 always (discovery failure is not fatal -- falls back to own dgemm). */
int blas_dispatch_init(void);

/* Returns a string identifying the active dgemm backend:
 *   "MKL-ILP64", "MKL-LP64", "OpenBLAS-ILP64", "OpenBLAS-LP64",
 *   "Accelerate", "BLIS", "jblas-own", "system-BLAS-ILP64", "system-BLAS-LP64"
 * Never returns NULL. */
const char *blas_backend_name(void);

/* Returns 1 if the external dgemm uses ILP64 (64-bit integer) parameters,
 * 0 if LP64 (32-bit integer) or no external dgemm was found. */
int blas_is_ilp64(void);

/* Returns 1 if an external dgemm (system BLAS or BLIS) was discovered. */
int blas_has_external(void);

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

/* Mutex shared by dgemm, dsyrk, and dsyr2k — all use jblas_packed_B. */
extern pthread_mutex_t jblas_dgemm_mutex;

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
 * Workspace struct for mutex-free GEMM (caller-managed buffers)
 * ---------------------------------------------------------------------------
 * Allows concurrent DGEMM calls (e.g. DSTEDC recursion) without mutex
 * serialisation.  Each workspace owns its own packed_A and packed_B buffers.
 */
typedef struct {
    double  *packed_B;   /* KC * NC doubles, 64-byte aligned */
    double  *packed_A;   /* n_threads * MC * KC doubles, 64-byte aligned */
    int      n_threads;
} jblas_workspace_t;

int  jblas_workspace_alloc(jblas_workspace_t *ws, int n_threads);
void jblas_workspace_free(jblas_workspace_t *ws);

/* ---------------------------------------------------------------------------
 * Accumulate GEMM: C = alpha * op(A) * op(B) + beta * C
 * ---------------------------------------------------------------------------
 * beta=0: zeroes C before accumulation (same as jblas_dgemm_c).
 * beta=1: accumulates into existing C (for rank-k updates, DORMTR, etc.).
 * alpha: scales the product; common values are 1.0 and -1.0.
 *
 * Uses the global mutex + global packed_A/B workspace.
 */
void jblas_dgemm_accum_c(npy_intp M, npy_intp N, npy_intp K,
                          const double *A, npy_intp lda,
                          const double *B, npy_intp ldb,
                          double *C, npy_intp ldc,
                          int transa, int transb,
                          double alpha, double beta);

/* ---------------------------------------------------------------------------
 * Workspace-explicit GEMM: C = alpha * op(A) * op(B) + beta * C
 * ---------------------------------------------------------------------------
 * Same as jblas_dgemm_accum_c but uses a caller-owned workspace instead
 * of the global packed_A/B + mutex.  No locking — safe for concurrent use
 * (e.g. inside DSTEDC recursive D&C).
 */
void jblas_dgemm_ws(npy_intp M, npy_intp N, npy_intp K,
                    const double *A, npy_intp lda,
                    const double *B, npy_intp ldb,
                    double *C, npy_intp ldc,
                    int transa, int transb,
                    double alpha, double beta,
                    jblas_workspace_t *ws);

/* ---------------------------------------------------------------------------
 * Full-signature dispatch: external BLAS when available, jblas-own otherwise.
 * ---------------------------------------------------------------------------
 * These are the correct entry points for callers that need transpose flags,
 * custom leading dimensions, or alpha/beta.  The simplified dispatch table
 * (jblas_dispatch.dgemm) only handles the NN natural-stride case.
 *
 * Row-major convention: C(M x N) = alpha * op(A)(M x K) * op(B)(K x N) + beta * C
 * transa/transb: 0 = no transpose, 1 = transpose.
 */

/* C = op(A) * op(B), zeroes C first.  Uses global workspace + mutex. */
void jblas_dgemm_ext(npy_intp M, npy_intp N, npy_intp K,
                     const double *A, npy_intp lda,
                     const double *B, npy_intp ldb,
                     double *C, npy_intp ldc,
                     int transa, int transb);

/* C = alpha * op(A) * op(B) + beta * C.  Uses caller-owned workspace (no mutex).
 * Falls back to jblas_dgemm_ws when no external BLAS.  When external BLAS is
 * active, ws is ignored (external BLAS manages its own threading/memory). */
void jblas_dgemm_ext_ws(npy_intp M, npy_intp N, npy_intp K,
                        const double *A, npy_intp lda,
                        const double *B, npy_intp ldb,
                        double *C, npy_intp ldc,
                        int transa, int transb,
                        double alpha, double beta,
                        jblas_workspace_t *ws);

/* ---------------------------------------------------------------------------
 * Thread control API
 * ---------------------------------------------------------------------------
 * jblas_get_n_threads: returns current thread count.
 * jblas_set_n_threads: sets thread count, clamped to init-time maximum
 *   (prevents packed_A OOB access).  Returns previous count, or -1 on error.
 */
int  jblas_get_n_threads(void);
int  jblas_set_n_threads(int n);

/* ---------------------------------------------------------------------------
 * dsyrk and dsyr2k function declarations
 * ---------------------------------------------------------------------------
 */

/**
 * jblas_dsyrk_c — Symmetric rank-k update: C = X @ X.T (lower triangle, then mirror).
 *
 * Computes the N×N symmetric matrix C = X @ X.T using lower-triangle tile
 * skipping (both A and B panels are packed from the same source matrix X).
 * After all tiles are accumulated, mirrors the lower triangle to fill the
 * upper triangle (BL3-06).
 *
 * C is zeroed before accumulation; the caller provides a pre-allocated N×N
 * output buffer (zero-initialised is fine, the function overwrites it).
 *
 * N    : number of rows/columns of C and rows of X.
 * K    : number of columns of X.
 * X    : input matrix, row-major, shape (N, K), leading dimension ldx.
 * ldx  : leading dimension of X (>= K).
 * C    : output matrix, row-major, shape (N, N), leading dimension ldc.
 * ldc  : leading dimension of C (>= N).
 */
void jblas_dsyrk_c(npy_intp N, npy_intp K,
                   const double *X, npy_intp ldx,
                   double *C, npy_intp ldc);

/**
 * jblas_dsyrk_lower_c — Symmetric rank-k update: C = X @ X.T (lower only).
 *
 * Identical to jblas_dsyrk_c but:
 *   1. Only zeroes the lower triangle of C (not the full matrix).
 *   2. Skips the mirror step — upper triangle is NOT filled.
 *
 * Saves O(N^2) wasted writes for callers that only read the lower triangle
 * (e.g. eigensolver-internal paths, kinship computation).
 */
void jblas_dsyrk_lower_c(npy_intp N, npy_intp K,
                          const double *X, npy_intp ldx,
                          double *C, npy_intp ldc);

/**
 * jblas_dsyr2k_c — Symmetric rank-2k update: C -= A @ B.T + B @ A.T.
 *
 * Applies two half-product subtractions to all elements of C (full-matrix
 * update).  Both triangles are updated for correctness when C is not
 * symmetric, matching the NumPy fallback contract.
 *
 * N    : number of rows/columns of C, and rows of A and B.
 * K    : number of columns of A and B.
 * A    : first factor, row-major, shape (N, K), leading dimension lda.
 * lda  : leading dimension of A (>= K).
 * B    : second factor, row-major, shape (N, K), leading dimension ldb.
 * ldb  : leading dimension of B (>= K).
 * C    : in/out matrix, row-major, shape (N, N), leading dimension ldc.
 * ldc  : leading dimension of C (>= N).
 */
void jblas_dsyr2k_c(npy_intp N, npy_intp K,
                    const double *A, npy_intp lda,
                    const double *B, npy_intp ldb,
                    double *C, npy_intp ldc);

/* ---------------------------------------------------------------------------
 * eigh function declarations (LAPACK eigendecomposition)
 * ---------------------------------------------------------------------------
 */

/**
 * jblas_eigh_c — compute all eigenvalues and eigenvectors of symmetric K.
 *
 * K is N x N, row-major, lower triangle used. K is overwritten as scratch.
 * eigenvalues: caller-allocated N doubles (ascending order on return).
 * eigenvectors: caller-allocated N x N doubles, row-major. U[:,j] is the
 *               eigenvector for eigenvalues[j].
 *
 * Returns 0 on success, -1 on allocation failure, positive i if the
 * D&C secular solver failed to converge for eigenvalue i.
 */
int jblas_eigh_c(npy_intp N,
                 double *K, npy_intp ldk,
                 double *eigenvalues,
                 double *eigenvectors, npy_intp ldz);

/* Internal LAPACK-layer functions (called by jblas_eigh_c, not Python-facing) */
int jblas_dsytrd_c(npy_intp N, double *A, npy_intp lda,
                   double *d, double *e, double *tau);
int jblas_dstedc_c(npy_intp N, double *d, double *e,
                   double *Z, npy_intp ldz,
                   jblas_workspace_t *ws);
int jblas_dormtr_c(npy_intp N, npy_intp M,
                   const double *A, npy_intp lda, const double *tau,
                   double *C, npy_intp ldc);

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
