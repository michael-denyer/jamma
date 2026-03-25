/**
 * jlinalg.h -- Public C API for the JAMMA vendor BLAS/LAPACK dispatch layer.
 *
 * Declares blas_dispatch_init(), vendor-dispatch functions (dgemm_ext,
 * dsyrk_ext, dsyevd_ext, dsyevr_ext, dgeqrf_ext, dorgqr_ext, dgesvd_ext),
 * eigh driver, SNP statistics, ISA detection, and thread control.
 *
 * The C layer is a thin vendor-dispatch shim; all computation is handled
 * by vendor BLAS/LAPACK or NumPy.
 *
 * ABI version bump required if any function signature or struct layout changes.
 */

#pragma once

#include <stddef.h>             /* size_t */
#include <stdint.h>             /* int32_t, int64_t -- snp_stats output types */
#include <numpy/arrayobject.h>  /* npy_intp */

/* Bump this constant whenever the public ABI changes (new fields in
 * structs, changed function signatures, etc.). pymodule.c exposes
 * this as a Python-level integer so callers can guard against ABI mismatches. */
#define JLINALG_ABI_VERSION 12

/* ---------------------------------------------------------------------------
 * External BLAS dispatch (vendor BLAS / LAPACK discovery)
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
 * Preferred over Fortran interface when available -- Accelerate/MKL can
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
 * -- Fortran is preferred because its suffixed symbol names (dsyevd_64_,
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

/* Initialise external BLAS dispatch: discovers system BLAS, pip MKL, and
 * bundled BLIS, then selects the best candidate.
 * Called from jlinalg_init() after ISA detection.
 * Returns 0 always (discovery failure is not fatal -- falls back to numpy). */
int blas_dispatch_init(void);

/* Returns a string identifying the active dgemm backend:
 *   "MKL-ILP64", "MKL-LP64", "OpenBLAS-ILP64", "OpenBLAS-LP64",
 *   "Accelerate", "Accelerate-ILP64", "BLIS", "BLIS-ILP64",
 *   "numpy-fallback", "system-BLAS-ILP64", "system-BLAS-LP64"
 * Never returns NULL. */
const char *blas_backend_name(void);

/* Returns 1 if the external dgemm uses ILP64 (64-bit integer) parameters,
 * 0 if LP64 (32-bit integer) or no external dgemm was found. */
int blas_is_ilp64(void);

/* Returns 1 if an external dgemm was discovered (vendor BLAS), 0 otherwise. */
int blas_has_external(void);

/* LP64 overflow tracking: incremented when dimensions exceed LP64_DIM_MAX.
 * py_eigh resets before the computation and checks after to issue a Python
 * warning. */
int  blas_dispatch_lp64_overflow_count(void);
void blas_dispatch_reset_lp64_overflow(void);

/* ---------------------------------------------------------------------------
 * Vendor-dispatch dsyrk / dsyevd / dsyevr API
 * ---------------------------------------------------------------------------
 */

/* Vendor-dispatch dsyrk: C = X @ X.T (lower triangle + mirror).
 * Routes to vendor cblas_dsyrk when available, else returns without computing
 * (caller must use numpy fallback). */
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
 * When true, jlinalg_dsyevd_ext uses row-major LAPACKE -- no transpose needed. */
int blas_has_lapacke_dsyevd(void);

/* Returns 1 if vendor dgeqrf + dorgqr are available, 0 otherwise. */
int blas_has_dgeqrf(void);

/* Returns 1 if vendor dgesvd is available, 0 otherwise. */
int blas_has_dgesvd(void);

/* Return codes for vendor-dispatch functions. */
#define JLINALG_EXT_SUCCESS         0   /* Operation succeeded */
#define JLINALG_EXT_ALLOC_FAIL     -1   /* Workspace allocation failed */
#define JLINALG_EXT_UNAVAILABLE    -2   /* No vendor routine available -- use numpy fallback */
#define JLINALG_EXT_COUNT_MISMATCH -3   /* DSYEVR returned fewer eigenvalues than expected (ABI mismatch) */
#define JLINALG_EXT_INTERNAL_ERROR -4   /* Internal logic error */
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
 * Full-signature vendor dgemm dispatch
 * ---------------------------------------------------------------------------
 * These are the correct entry points for callers that need transpose flags,
 * custom leading dimensions, or alpha/beta.
 *
 * Row-major convention: C(M x N) = alpha * op(A)(M x K) * op(B)(K x N) + beta * C
 * transa/transb: 0 = no transpose, 1 = transpose.
 *
 * When no vendor BLAS is available, these functions return without computing
 * (caller should check blas_has_external() and use numpy fallback).
 */

/* C = op(A) * op(B), zeroes C first. */
void jlinalg_dgemm_ext(npy_intp M, npy_intp N, npy_intp K,
                     const double *A, npy_intp lda,
                     const double *B, npy_intp ldb,
                     double *C, npy_intp ldc,
                     int transa, int transb);

/* C = alpha * op(A) * op(B) + beta * C. */
void jlinalg_dgemm_ext_ws(npy_intp M, npy_intp N, npy_intp K,
                        const double *A, npy_intp lda,
                        const double *B, npy_intp ldb,
                        double *C, npy_intp ldc,
                        int transa, int transb,
                        double alpha, double beta);

/* ---------------------------------------------------------------------------
 * Thread control API
 * ---------------------------------------------------------------------------
 * jlinalg_get_n_threads: returns current thread count.
 * jlinalg_set_n_threads: sets thread count.  Returns previous count, or -1 on error.
 */
int  jlinalg_get_n_threads(void);
int  jlinalg_set_n_threads(int n);

/* ---------------------------------------------------------------------------
 * eigh status struct (populated during eigh, checked by py_eigh for warnings)
 * ---------------------------------------------------------------------------
 */
typedef struct {
    int dstedc_ws_fallback;      /* legacy field, always zero — retained for ABI stability */
    int dsytrd_mirror_fallback;  /* legacy field, always zero — retained for ABI stability */
    int secular_failures;        /* legacy field, always zero — retained for ABI stability */
    int qr_fallback;             /* legacy field, always zero — retained for ABI stability */
    int vendor_lapack_skipped;   /* 1 if vendor LAPACK (dsyevd/dsyevr) alloc failed */
} jlinalg_eigh_status_t;

/* ---------------------------------------------------------------------------
 * eigh function declaration (vendor LAPACK eigendecomposition)
 * ---------------------------------------------------------------------------
 */

/**
 * jlinalg_eigh_c -- compute all eigenvalues and eigenvectors of symmetric K.
 *
 * K is N x N, row-major, lower triangle used. K is overwritten as scratch.
 * eigenvalues: caller-allocated N doubles (ascending order on return).
 * eigenvectors: caller-allocated N x N doubles, row-major. U[:,j] is the
 *               eigenvector for eigenvalues[j].
 * status: if non-NULL, populated with diagnostic flags.
 *
 * Returns 0 on success, JLINALG_EXT_UNAVAILABLE if no vendor LAPACK,
 * JLINALG_EXT_ALLOC_FAIL on allocation failure, positive i on convergence
 * failure.
 */
int jlinalg_eigh_c(npy_intp N,
                 double *K, npy_intp ldk,
                 double *eigenvalues,
                 double *eigenvectors, npy_intp ldz,
                 jlinalg_eigh_status_t *status);

/* ---------------------------------------------------------------------------
 * snp_stats: single-pass per-SNP statistics (mean, variance, miss, HWE)
 * ---------------------------------------------------------------------------
 */
void snp_stats_chunk_f32(const float *data, npy_intp n_samples, npy_intp n_snps_chunk,
                         double *means, npy_intp *miss_counts, double *variances,
                         int64_t *n_aa, int64_t *n_ab, int64_t *n_bb, int compute_hwe);
void snp_stats_chunk_f64(const double *data, npy_intp n_samples, npy_intp n_snps_chunk,
                         double *means, npy_intp *miss_counts, double *variances,
                         int64_t *n_aa, int64_t *n_ab, int64_t *n_bb, int compute_hwe);

/* ---------------------------------------------------------------------------
 * Initialisation and introspection
 * ---------------------------------------------------------------------------
 */

/**
 * jlinalg_init -- Detect ISA and initialise vendor BLAS dispatch.
 * Idempotent (guarded by a static flag).
 *
 * Returns: 0 on success.
 */
int jlinalg_init(void);

/**
 * jlinalg_isa_name -- Return the active ISA as a C string.
 *
 * Returns: "AVX2", "NEON", or "generic" (never NULL).
 */
const char *jlinalg_isa_name(void);
