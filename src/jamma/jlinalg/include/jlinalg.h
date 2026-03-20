/**
 * jlinalg.h — Public C API for the JAMMA BLAS and LAPACK compute layer.
 *
 * Declares the ISA dispatch table, jlinalg_init(), and function signatures for
 * Level 1/2 BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv), Level 3
 * (dgemm, dsyrk, dsyr2k with three-level Goto/BLIS blocking and
 * ISA-dispatched microkernels), and LAPACK eigendecomposition (eigh via
 * DSYTRD + DSTEDC + DORMTR).
 *
 * ABI version bump required if any function signature or struct layout changes.
 */

#pragma once

#include <stddef.h>             /* size_t */
#include <pthread.h>            /* pthread_mutex_t — shared across dgemm/dsyrk/dsyr2k */
#include <numpy/arrayobject.h>  /* npy_intp */

/* Bump this constant whenever the public ABI changes (new fields in
 * jlinalg_dispatch_t, changed function signatures, etc.). pymodule.c exposes
 * this as a Python-level integer so callers can guard against ABI mismatches. */
#define JLINALG_ABI_VERSION 11

/* ---------------------------------------------------------------------------
 * Function-pointer typedefs for ISA-dispatched microkernels
 * ---------------------------------------------------------------------------
 * incx / incy are stride arguments (always 1 in the current Python API;
 * retained for BLAS compatibility and future strided-array support).
 */

typedef double (*jlinalg_ddot_fn)(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

typedef double (*jlinalg_dnrm2_fn)(
    npy_intp n,
    const double *x, int incx);

typedef void (*jlinalg_daxpy_fn)(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

typedef void (*jlinalg_dscal_fn)(
    npy_intp n,
    double alpha,
    double *x, int incx);

/* dgemv: y = A*x (no alpha/beta/transpose for the internal primitive).
 * A is row-major (m x n), x is length-n, y is length-m output. */
typedef void (*jlinalg_dgemv_fn)(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

/* dgemm: C = A*B.  A is (m x k), B is (k x n), C is (m x n). Row-major. */
typedef void (*jlinalg_dgemm_fn)(
    npy_intp m, npy_intp n, npy_intp k,
    const double *A,
    const double *B,
    double       *C);

/* ---------------------------------------------------------------------------
 * Global dispatch table (set once by jlinalg_init, then read-only)
 * ---------------------------------------------------------------------------
 */

typedef struct {
    jlinalg_ddot_fn  ddot;
    jlinalg_dnrm2_fn dnrm2;
    jlinalg_daxpy_fn daxpy;
    jlinalg_dscal_fn dscal;
    jlinalg_dgemv_fn dgemv;
    jlinalg_dgemm_fn dgemm;
} jlinalg_dispatch_t;

extern jlinalg_dispatch_t jlinalg_dispatch;

/* ---------------------------------------------------------------------------
 * External BLAS dispatch (system BLAS / bundled BLIS discovery)
 * ---------------------------------------------------------------------------
 */

/* Fortran-style dgemm function pointer types for dlopen'd BLAS */
typedef void (*jlinalg_dgemm_lp64_fn)(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const double *alpha, const double *a, const int *lda,
    const double *b, const int *ldb,
    const double *beta, double *c, const int *ldc);

typedef void (*jlinalg_dgemm_ilp64_fn)(
    const char *transa, const char *transb,
    const long long *m, const long long *n, const long long *k,
    const double *alpha, const double *a, const long long *lda,
    const double *b, const long long *ldb,
    const double *beta, double *c, const long long *ldc);

/* CBLAS C-interface dgemm: handles row-major natively (no A/B swap needed).
 * Preferred over Fortran interface when available — Accelerate/MKL can
 * choose optimal algorithm for the memory layout.
 *
 * LP64 CBLAS uses int (32-bit) for dimensions.  ILP64 CBLAS (e.g.
 * Accelerate $NEWLAPACK$ILP64) uses long (64-bit on LP64 platforms
 * like macOS arm64 and Linux x86_64).  We use separate typedefs. */
enum { JLINALG_CblasRowMajor = 101, JLINALG_CblasNoTrans = 111, JLINALG_CblasTrans = 112 };
enum { JLINALG_CblasUpper = 121, JLINALG_CblasLower = 122 };
typedef void (*jlinalg_cblas_dgemm_fn)(
    int order, int transa, int transb,
    int m, int n, int k,
    double alpha, const double *a, int lda,
    const double *b, int ldb,
    double beta, double *c, int ldc);
typedef void (*jlinalg_cblas_dgemm_ilp64_fn)(
    int order, int transa, int transb,
    long m, long n, long k,
    double alpha, const double *a, long lda,
    const double *b, long ldb,
    double beta, double *c, long ldc);

/* Fortran dsyrk: dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc) */
typedef void (*jlinalg_dsyrk_lp64_fn)(
    const char *uplo, const char *trans,
    const int *n, const int *k,
    const double *alpha, const double *a, const int *lda,
    const double *beta, double *c, const int *ldc);

typedef void (*jlinalg_dsyrk_ilp64_fn)(
    const char *uplo, const char *trans,
    const long long *n, const long long *k,
    const double *alpha, const double *a, const long long *lda,
    const double *beta, double *c, const long long *ldc);

/* CBLAS dsyrk: cblas_dsyrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc) */
typedef void (*jlinalg_cblas_dsyrk_fn)(
    int order, int uplo, int trans,
    int n, int k,
    double alpha, const double *a, int lda,
    double beta, double *c, int ldc);

typedef void (*jlinalg_cblas_dsyrk_ilp64_fn)(
    int order, int uplo, int trans,
    long n, long k,
    double alpha, const double *a, long lda,
    double beta, double *c, long ldc);

/* LAPACK dsyevd (Fortran): dsyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info) */
typedef void (*jlinalg_dsyevd_lp64_fn)(
    const char *jobz, const char *uplo,
    const int *n, double *a, const int *lda,
    double *w, double *work, const int *lwork,
    int *iwork, const int *liwork, int *info);

typedef void (*jlinalg_dsyevd_ilp64_fn)(
    const char *jobz, const char *uplo,
    const long long *n, double *a, const long long *lda,
    double *w, double *work, const long long *lwork,
    long long *iwork, const long long *liwork, long long *info);

/* LAPACKE dsyevd (C interface): handles row-major natively, no manual transpose.
 * lapack_int is int for LP64 builds, long long for MKL ILP64 (int64_t for
 * OpenBLAS ILP64).  Used as fallback when Fortran ILP64 symbols are unavailable
 * — Fortran is preferred because its suffixed symbol names (dsyevd_64_,
 * dsyevd$NEWLAPACK$ILP64) are unambiguous for LP64/ILP64. */
enum { JLINALG_LAPACK_ROW_MAJOR = 101, JLINALG_LAPACK_COL_MAJOR = 102 };
typedef int (*jlinalg_lapacke_dsyevd_lp64_fn)(
    int matrix_layout, char jobz, char uplo,
    int n, double *a, int lda, double *w);
typedef long long (*jlinalg_lapacke_dsyevd_ilp64_fn)(
    int matrix_layout, char jobz, char uplo,
    long long n, double *a, long long lda, double *w);

/* LAPACK dsyevr (Fortran): dsyevr_(jobz, range, uplo, n, a, lda, vl, vu, il, iu,
 *   abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info) */
typedef void (*jlinalg_dsyevr_lp64_fn)(
    const char *jobz, const char *range, const char *uplo,
    const int *n, double *a, const int *lda,
    const double *vl, const double *vu, const int *il, const int *iu,
    const double *abstol, int *m, double *w, double *z, const int *ldz,
    int *isuppz, double *work, const int *lwork,
    int *iwork, const int *liwork, int *info);

typedef void (*jlinalg_dsyevr_ilp64_fn)(
    const char *jobz, const char *range, const char *uplo,
    const long long *n, double *a, const long long *lda,
    const double *vl, const double *vu, const long long *il, const long long *iu,
    const double *abstol, long long *m, double *w, double *z, const long long *ldz,
    long long *isuppz, double *work, const long long *lwork,
    long long *iwork, const long long *liwork, long long *info);

/* LAPACK dgeqrf (Fortran): dgeqrf_(m, n, a, lda, tau, work, lwork, info) */
typedef void (*jlinalg_dgeqrf_lp64_fn)(
    const int *m, const int *n,
    double *a, const int *lda,
    double *tau, double *work, const int *lwork,
    int *info);
typedef void (*jlinalg_dgeqrf_ilp64_fn)(
    const long long *m, const long long *n,
    double *a, const long long *lda,
    double *tau, double *work, const long long *lwork,
    long long *info);

/* LAPACK dorgqr (Fortran): dorgqr_(m, n, k, a, lda, tau, work, lwork, info) */
typedef void (*jlinalg_dorgqr_lp64_fn)(
    const int *m, const int *n, const int *k,
    double *a, const int *lda,
    const double *tau, double *work, const int *lwork,
    int *info);
typedef void (*jlinalg_dorgqr_ilp64_fn)(
    const long long *m, const long long *n, const long long *k,
    double *a, const long long *lda,
    const double *tau, double *work, const long long *lwork,
    long long *info);

/* LAPACK dgesvd (Fortran): dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info) */
typedef void (*jlinalg_dgesvd_lp64_fn)(
    const char *jobu, const char *jobvt,
    const int *m, const int *n,
    double *a, const int *lda,
    double *s, double *u, const int *ldu,
    double *vt, const int *ldvt,
    double *work, const int *lwork,
    int *info);
typedef void (*jlinalg_dgesvd_ilp64_fn)(
    const char *jobu, const char *jobvt,
    const long long *m, const long long *n,
    double *a, const long long *lda,
    double *s, double *u, const long long *ldu,
    double *vt, const long long *ldvt,
    double *work, const long long *lwork,
    long long *info);

/* Initialise external BLAS dispatch: system BLAS -> bundled BLIS -> own kernels.
 * Called from jlinalg_init() after ISA detection and dgemm_init().
 * If an external dgemm is found, replaces jlinalg_dispatch.dgemm with a wrapper.
 * Returns 0 always (discovery failure is not fatal -- falls back to own dgemm). */
int blas_dispatch_init(void);

/* Returns a string identifying the active dgemm backend:
 *   "MKL-ILP64", "MKL-LP64", "OpenBLAS-ILP64", "OpenBLAS-LP64",
 *   "Accelerate", "Accelerate-ILP64", "BLIS", "BLIS-ILP64", "jlinalg-own",
 *   "system-BLAS-ILP64", "system-BLAS-LP64"
 * Never returns NULL. */
const char *blas_backend_name(void);

/* Returns 1 if the external dgemm uses ILP64 (64-bit integer) parameters,
 * 0 if LP64 (32-bit integer) or no external dgemm was found. */
int blas_is_ilp64(void);

/* Returns 1 if an external dgemm (system BLAS or BLIS) was discovered. */
int blas_has_external(void);

/* LP64 overflow tracking: incremented when dimensions exceed LP64_DIM_MAX
 * and the fallback to jlinalg-own dgemm is used.  py_eigh resets before the
 * computation and checks after to issue a Python warning. */
int  blas_dispatch_lp64_overflow_count(void);
void blas_dispatch_reset_lp64_overflow(void);

/* ---------------------------------------------------------------------------
 * dgemm microkernel function pointer
 * ---------------------------------------------------------------------------
 * Operates on packed data (see jlinalg_pack_A / jlinalg_pack_B).  Different from
 * the dispatch table's jlinalg_dgemm_fn which takes raw (M, N, K) pointers.
 *
 * Updates C_tile[MR x NR] += packed_A[MR x kc] * packed_B[kc x NR].
 * C is in row-major layout with leading dimension ldc.
 */
typedef void (*jlinalg_dgemm_micro_fn)(
    npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc);

extern jlinalg_dgemm_micro_fn jlinalg_dgemm_microkernel;

/* ---------------------------------------------------------------------------
 * dgemm blocking parameters (ISA-dependent, set by jlinalg_init() in platform.c)
 * ---------------------------------------------------------------------------
 * AVX2:    MR=6, NR=8,  KC=256, MC=72,  NC=4096
 * NEON:    MR=8, NR=4,  KC=256, MC=64,  NC=4096
 * Generic: MR=4, NR=4,  KC=128, MC=32,  NC=1024
 */
extern int JLINALG_MR, JLINALG_NR, JLINALG_KC, JLINALG_MC, JLINALG_NC;

/* ---------------------------------------------------------------------------
 * dgemm packing workspace buffers (allocated by jlinalg_dgemm_init())
 * ---------------------------------------------------------------------------
 * packed_A: per-thread (n_threads * MC * KC doubles), 64-byte aligned.
 * packed_B: shared    (KC * NC doubles), 64-byte aligned.
 */
extern double *jlinalg_packed_A;
extern double *jlinalg_packed_B;
extern int     jlinalg_n_threads; /* Thread count at init time */

/* Mutex shared by dgemm, dsyrk, and dsyr2k — all use jlinalg_packed_B. */
extern pthread_mutex_t jlinalg_dgemm_mutex;

/* ---------------------------------------------------------------------------
 * dgemm function declarations
 * ---------------------------------------------------------------------------
 */

/* Workspace allocation — called from jlinalg_init() after ISA detection. */
int  jlinalg_dgemm_init(void);
void jlinalg_dgemm_cleanup(void);

/* Dispatch-table wrapper: assigned to jlinalg_dispatch.dgemm by platform.c.
 * Matches jlinalg_dgemm_fn signature; calls jlinalg_dgemm_c with transa=transb=0. */
extern jlinalg_dgemm_fn jlinalg_dgemm_dispatch_fn;

/* Full dgemm with transpose support — the implementation behind
 * jlinalg_dispatch.dgemm and the Python-facing wrapper.
 * transa/transb: 0 = no transpose, 1 = transpose. */
void jlinalg_dgemm_c(npy_intp M, npy_intp N, npy_intp K,
                   const double *A, npy_intp lda,
                   const double *B, npy_intp ldb,
                   double *C, npy_intp ldc,
                   int transa, int transb);

/* Generic (scalar) microkernel — always available as fallback. */
void jlinalg_dgemm_micro_generic(npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc);

/* Packing helpers — copy A/B panels into packed format for cache-friendly
 * microkernel access.  trans=0 for no transpose, trans=1 for transpose. */
void jlinalg_pack_A(const double *A, npy_intp lda,
                  npy_intp mc, npy_intp kc, double *packed,
                  int mr, int trans);
void jlinalg_pack_B(const double *B, npy_intp ldb,
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
} jlinalg_workspace_t;

int  jlinalg_workspace_alloc(jlinalg_workspace_t *ws, int n_threads);
void jlinalg_workspace_free(jlinalg_workspace_t *ws);

/* ---------------------------------------------------------------------------
 * Accumulate GEMM: C = alpha * op(A) * op(B) + beta * C
 * ---------------------------------------------------------------------------
 * beta=0: zeroes C before accumulation (same as jlinalg_dgemm_c).
 * beta=1: accumulates into existing C (for rank-k updates, DORMTR, etc.).
 * alpha: scales the product; common values are 1.0 and -1.0.
 *
 * Uses the global mutex + global packed_A/B workspace.
 */
void jlinalg_dgemm_accum_c(npy_intp M, npy_intp N, npy_intp K,
                          const double *A, npy_intp lda,
                          const double *B, npy_intp ldb,
                          double *C, npy_intp ldc,
                          int transa, int transb,
                          double alpha, double beta);

/* ---------------------------------------------------------------------------
 * Workspace-explicit GEMM: C = alpha * op(A) * op(B) + beta * C
 * ---------------------------------------------------------------------------
 * Same as jlinalg_dgemm_accum_c but uses a caller-owned workspace instead
 * of the global packed_A/B + mutex.  No locking — safe for concurrent use
 * (e.g. inside DSTEDC recursive D&C).
 */
void jlinalg_dgemm_ws(npy_intp M, npy_intp N, npy_intp K,
                    const double *A, npy_intp lda,
                    const double *B, npy_intp ldb,
                    double *C, npy_intp ldc,
                    int transa, int transb,
                    double alpha, double beta,
                    jlinalg_workspace_t *ws);

/* ---------------------------------------------------------------------------
 * Full-signature dispatch: external BLAS when available, jlinalg-own otherwise.
 * ---------------------------------------------------------------------------
 * These are the correct entry points for callers that need transpose flags,
 * custom leading dimensions, or alpha/beta.  The simplified dispatch table
 * (jlinalg_dispatch.dgemm) only handles the NN natural-stride case.
 *
 * Row-major convention: C(M x N) = alpha * op(A)(M x K) * op(B)(K x N) + beta * C
 * transa/transb: 0 = no transpose, 1 = transpose.
 */

/* C = op(A) * op(B), zeroes C first.  Uses global workspace + mutex. */
void jlinalg_dgemm_ext(npy_intp M, npy_intp N, npy_intp K,
                     const double *A, npy_intp lda,
                     const double *B, npy_intp ldb,
                     double *C, npy_intp ldc,
                     int transa, int transb);

/* C = alpha * op(A) * op(B) + beta * C.  Uses caller-owned workspace (no mutex).
 * Falls back to jlinalg_dgemm_ws when no external BLAS.  When external BLAS is
 * active, ws is ignored (external BLAS manages its own threading/memory). */
void jlinalg_dgemm_ext_ws(npy_intp M, npy_intp N, npy_intp K,
                        const double *A, npy_intp lda,
                        const double *B, npy_intp ldb,
                        double *C, npy_intp ldc,
                        int transa, int transb,
                        double alpha, double beta,
                        jlinalg_workspace_t *ws);

/* ---------------------------------------------------------------------------
 * Vendor-dispatch dsyrk / dsyevd API
 * ---------------------------------------------------------------------------
 */

/* Vendor-dispatch dsyrk: C = X @ X.T (lower triangle + mirror).
 * Routes to vendor cblas_dsyrk when available, else jlinalg_dsyrk_c. */
void jlinalg_dsyrk_ext(npy_intp N, npy_intp K,
                     const double *X, npy_intp ldx,
                     double *C, npy_intp ldc);

/* Returns 1 if vendor dsyrk is available (cblas_dsyrk resolved), 0 otherwise. */
int blas_has_dsyrk(void);

/* Returns 1 if vendor dsyevd is available, 0 otherwise. */
int blas_has_dsyevd(void);

/* Returns 1 if vendor dsyevr is available, 0 otherwise. */
int blas_has_dsyevr(void);

/* Returns 1 if LAPACKE C interface for dsyevd is available (MKL).
 * When true, jlinalg_dsyevd_ext uses row-major LAPACKE — no transpose needed. */
int blas_has_lapacke_dsyevd(void);

/* Returns 1 if vendor dgeqrf + dorgqr are available, 0 otherwise. */
int blas_has_dgeqrf(void);

/* Returns 1 if vendor dgesvd is available, 0 otherwise. */
int blas_has_dgesvd(void);

/* Return codes for jlinalg_dsyevd_ext (and future vendor-dispatch functions). */
#define JLINALG_EXT_SUCCESS         0   /* Operation succeeded */
#define JLINALG_EXT_ALLOC_FAIL     -1   /* Workspace allocation failed */
#define JLINALG_EXT_UNAVAILABLE    -2   /* No vendor routine available — use jlinalg pipeline */
#define JLINALG_EXT_COUNT_MISMATCH -3   /* DSYEVR returned fewer eigenvalues than expected (ABI mismatch) */
#define JLINALG_EXT_INTERNAL_ERROR -4   /* Internal logic error (e.g. unsupported dgemm parameters) */
#define JLINALG_EXT_INPLACE_UNSUPPORTED -5  /* inplace=True requires vendor LAPACK (DSYEVD/DSYEVR) */

/* Vendor-dispatch dsyevd for eigh.
 * Routes to vendor dsyevd when available, else returns JLINALG_EXT_UNAVAILABLE.
 * K: N x N row-major symmetric matrix (overwritten with eigenvectors in row-major on success).
 * eigenvalues: N doubles, ascending on success.
 * Returns JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, JLINALG_EXT_ALLOC_FAIL,
 * or positive LAPACK info on convergence/argument error. */
int jlinalg_dsyevd_ext(npy_intp N, double *K, npy_intp ldk,
                     double *eigenvalues);

/* Vendor-dispatch dsyevr for eigh (memory-pressure fallback).
 * DSYEVR uses O(N) workspace vs O(N^2) for DSYEVD.  Writes eigenvectors
 * into a separate Z buffer (does not overwrite K).
 * K: N x N row-major symmetric matrix (lower triangle used, overwritten).
 * eigenvalues: N doubles, ascending on success.
 * eigenvectors: N x N row-major output, eigenvectors as columns.
 * Returns JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, JLINALG_EXT_ALLOC_FAIL,
 * or positive LAPACK info on convergence/argument error. */
int jlinalg_dsyevr_ext(npy_intp N, double *K, npy_intp ldk,
                     double *eigenvalues,
                     double *eigenvectors, npy_intp ldz);

/* Vendor-dispatch QR factorization.
 * A: m x n input (column-major, overwritten). tau: min(m,n) Householder scalars.
 * Returns JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, or JLINALG_EXT_ALLOC_FAIL. */
int jlinalg_dgeqrf_ext(npy_intp m, npy_intp n, double *A_col, npy_intp lda, double *tau);

/* Vendor-dispatch generate Q from Householder vectors (after dgeqrf).
 * A: m x n input/output (column-major). tau: n Householder scalars from dgeqrf.
 * Returns JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, or JLINALG_EXT_ALLOC_FAIL. */
int jlinalg_dorgqr_ext(npy_intp m, npy_intp n, double *A_col, npy_intp lda, const double *tau);

/* Vendor-dispatch SVD: A = U * diag(s) * Vt.
 * A_col: m x n column-major input (overwritten).
 * s: min(m,n) singular values (descending).
 * U_col: m x min(m,n) column-major output (NULL if compute_uv=0).
 * Vt_col: min(m,n) x n column-major output (NULL if compute_uv=0).
 * compute_uv: 1 = compute U and Vt, 0 = singular values only.
 * Returns JLINALG_EXT_SUCCESS, JLINALG_EXT_UNAVAILABLE, JLINALG_EXT_ALLOC_FAIL, or positive info. */
int jlinalg_dgesvd_ext(npy_intp m, npy_intp n,
                       double *A_col, npy_intp lda,
                       double *s,
                       double *U_col, npy_intp ldu,
                       double *Vt_col, npy_intp ldvt,
                       int compute_uv);

/* ---------------------------------------------------------------------------
 * Thread control API
 * ---------------------------------------------------------------------------
 * jlinalg_get_n_threads: returns current thread count.
 * jlinalg_set_n_threads: sets thread count, clamped to init-time maximum
 *   (prevents packed_A OOB access).  Returns previous count, or -1 on error.
 */
int  jlinalg_get_n_threads(void);
int  jlinalg_set_n_threads(int n);

/* ---------------------------------------------------------------------------
 * dsyrk and dsyr2k function declarations
 * ---------------------------------------------------------------------------
 */

/**
 * jlinalg_dsyrk_c — Symmetric rank-k update: C = X @ X.T (lower triangle, then mirror).
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
void jlinalg_dsyrk_c(npy_intp N, npy_intp K,
                   const double *X, npy_intp ldx,
                   double *C, npy_intp ldc);

/**
 * jlinalg_dsyrk_lower_c — Symmetric rank-k update: C = X @ X.T (lower only).
 *
 * Identical to jlinalg_dsyrk_c but:
 *   1. Only zeroes the lower triangle of C (not the full matrix).
 *   2. Skips the mirror step — upper triangle is NOT filled.
 *
 * Saves O(N^2) wasted writes for callers that only read the lower triangle
 * (e.g. eigensolver-internal paths, kinship computation).
 */
void jlinalg_dsyrk_lower_c(npy_intp N, npy_intp K,
                          const double *X, npy_intp ldx,
                          double *C, npy_intp ldc);

/**
 * jlinalg_dsyr2k_c — Symmetric rank-2k update: C -= A @ B.T + B @ A.T.
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
void jlinalg_dsyr2k_c(npy_intp N, npy_intp K,
                    const double *A, npy_intp lda,
                    const double *B, npy_intp ldb,
                    double *C, npy_intp ldc);

/**
 * Workspace-explicit symmetric BLAS variants (no mutex).
 *
 * Same algorithms as the mutex-based _c variants but use caller-owned
 * packed_A/packed_B workspace.  Safe for concurrent use from within the
 * eigensolver (dsytrd trailing update, etc.).
 */
void jlinalg_dsyrk_ws(npy_intp N, npy_intp K,
                     const double *X, npy_intp ldx,
                     double *C, npy_intp ldc,
                     jlinalg_workspace_t *ws);

void jlinalg_dsyrk_lower_ws(npy_intp N, npy_intp K,
                            const double *X, npy_intp ldx,
                            double *C, npy_intp ldc,
                            jlinalg_workspace_t *ws);

void jlinalg_dsyr2k_ws(npy_intp N, npy_intp K,
                      const double *A, npy_intp lda,
                      const double *B, npy_intp ldb,
                      double *C, npy_intp ldc,
                      jlinalg_workspace_t *ws);

/* ---------------------------------------------------------------------------
 * eigh status struct (populated during eigh, checked by py_eigh for warnings)
 * ---------------------------------------------------------------------------
 */
typedef struct {
    int dstedc_ws_fallback;      /* 1 if dstedc workspace alloc failed (global mutex path) */
    int dsytrd_mirror_fallback;  /* 1 if dsytrd mirror buffer alloc failed (scalar dsymv) */
    int secular_failures;        /* count of secular equation non-convergences */
    int qr_fallback;             /* 1 if QR fallback was used */
    int vendor_lapack_skipped;   /* 1 if vendor LAPACK (dsyevd/dsyevr) alloc failed */
} jlinalg_eigh_status_t;

/* ---------------------------------------------------------------------------
 * eigh function declarations (LAPACK eigendecomposition)
 * ---------------------------------------------------------------------------
 */

/**
 * jlinalg_eigh_c — compute all eigenvalues and eigenvectors of symmetric K.
 *
 * K is N x N, row-major, lower triangle used. K is overwritten as scratch.
 * eigenvalues: caller-allocated N doubles (ascending order on return).
 * eigenvectors: caller-allocated N x N doubles, row-major. U[:,j] is the
 *               eigenvector for eigenvalues[j].
 * status: if non-NULL, populated with diagnostic flags (fallbacks, failures).
 *
 * Returns 0 on success, -1 on allocation failure, positive i if the
 * D&C secular solver failed to converge for eigenvalue i.
 */
int jlinalg_eigh_c(npy_intp N,
                 double *K, npy_intp ldk,
                 double *eigenvalues,
                 double *eigenvectors, npy_intp ldz,
                 jlinalg_eigh_status_t *status);

/* jlinalg_eigh_factored_c — factored eigendecomposition (no eigenvector matrix).
 *
 * Runs dsytrd + dstedc but NOT dormtr. Returns:
 *   - eigenvalues[N]: ascending eigenvalues
 *   - K: overwritten with Householder vectors in lower triangle (from dsytrd)
 *   - tau[N-1]: Householder scalars (caller-allocated)
 *   - V[N x N]: tridiagonal eigenvectors (caller-allocated, row-major)
 *
 * Only works on jlinalg D&C pipeline. Returns JLINALG_EXT_UNAVAILABLE (-2) when
 * jlinalg_packed_A is NULL (workspace not initialized).
 */
int jlinalg_eigh_factored_c(npy_intp N,
                 double *K, npy_intp ldk,
                 double *eigenvalues,
                 double *tau,
                 double *V, npy_intp ldv,
                 jlinalg_eigh_status_t *status);

/* Internal LAPACK-layer functions (called by jlinalg_eigh_c, not Python-facing) */
int jlinalg_dsytrd_c(npy_intp N, double *A, npy_intp lda,
                   double *d, double *e, double *tau,
                   jlinalg_workspace_t *ws,
                   jlinalg_eigh_status_t *status);
int jlinalg_dstedc_c(npy_intp N, double *d, double *e,
                   double *Z, npy_intp ldz,
                   jlinalg_workspace_t *ws,
                   jlinalg_eigh_status_t *status);
int jlinalg_dormtr_c(npy_intp N, npy_intp M,
                   const double *A, npy_intp lda, const double *tau,
                   double *C, npy_intp ldc,
                   jlinalg_workspace_t *ws);
int jlinalg_dormtr_transpose_c(npy_intp N, npy_intp M,
                   const double *A, npy_intp lda, const double *tau,
                   double *C, npy_intp ldc,
                   jlinalg_workspace_t *ws);

/* ---------------------------------------------------------------------------
 * Initialisation and introspection
 * ---------------------------------------------------------------------------
 */

/**
 * jlinalg_init — Detect ISA and populate jlinalg_dispatch with the best available
 * microkernel pointers.  Guard with a static flag so it is idempotent across
 * repeated calls (e.g. multiprocessing child-process re-import).
 *
 * Returns: 0 on success, -1 on failure (e.g. workspace allocation for dgemm
 * failed).
 */
int jlinalg_init(void);

/**
 * jlinalg_isa_name — Return the active ISA as a C string.
 *
 * Returns: "AVX2", "NEON", or "generic" (never NULL).
 */
const char *jlinalg_isa_name(void);

/* ---------------------------------------------------------------------------
 * Generic (portable C) microkernel declarations
 * ---------------------------------------------------------------------------
 * These are always compiled and linked; they are used when no SIMD path is
 * available.  Tail handling uses the ISA-specific microkernel + scratch buffer.
 */

double jlinalg_ddot_generic(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

double jlinalg_dnrm2_generic(
    npy_intp n,
    const double *x, int incx);

void jlinalg_daxpy_generic(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

void jlinalg_dscal_generic(
    npy_intp n,
    double alpha,
    double *x, int incx);

void jlinalg_dgemv_generic(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

/* ---------------------------------------------------------------------------
 * x86-64 AVX2 microkernel declarations
 * ---------------------------------------------------------------------------
 * These translation units are compiled with -mavx2 -mfma.  They must only be
 * called after jlinalg_init() has confirmed AVX2 support via CPUID.
 */

#if defined(__x86_64__)

double jlinalg_ddot_avx2(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

double jlinalg_dnrm2_avx2(
    npy_intp n,
    const double *x, int incx);

void jlinalg_daxpy_avx2(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

void jlinalg_dscal_avx2(
    npy_intp n,
    double alpha,
    double *x, int incx);

void jlinalg_dgemv_avx2(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

void jlinalg_dgemm_micro_avx2(npy_intp kc,
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
double jlinalg_ddot_neon(npy_intp n, const double *x, int incx,
                       const double *y, int incy);
double jlinalg_dnrm2_neon(npy_intp n, const double *x, int incx);
void   jlinalg_daxpy_neon(npy_intp n, double alpha,
                         const double *x, int incx, double *y, int incy);
void   jlinalg_dscal_neon(npy_intp n, double alpha, double *x, int incx);
void   jlinalg_dgemv_neon(npy_intp m, npy_intp n, const double *A,
                         const double *x, double *y);
*/

void jlinalg_dgemm_micro_neon(npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc);

#endif /* __aarch64__ */
