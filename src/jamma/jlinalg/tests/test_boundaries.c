/**
 * test_boundaries.c -- C-level boundary tests for jlinalg via Unity framework.
 *
 * VALID-01: Tests dgemm, dsyrk, and eigh at blocking boundary sizes directly
 * from C, without Python overhead.  Boundary sizes are derived from the runtime
 * blocking parameters (MR, NR, MC, NC, KC) set by jlinalg_init().
 *
 * Compiled by _compile_jlinalg.py:compile_test_harness() into a standalone
 * binary; invoked by tests/test_jlinalg_unity.py via subprocess.
 */

#include <Python.h>
#include "../include/jlinalg.h"
#include "unity/unity.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Simple LCG for reproducible random doubles in [-1, 1]. */
static unsigned _lcg_state;

static void seed_lcg(unsigned s) { _lcg_state = s; }

static double rand_double(void) {
    _lcg_state = _lcg_state * 1103515245u + 12345u;
    return ((double)(_lcg_state & 0x7FFFFFFFu) / (double)0x7FFFFFFFu) * 2.0 - 1.0;
}

static void fill_random(double *A, int n, unsigned seed) {
    seed_lcg(seed);
    for (int i = 0; i < n; i++)
        A[i] = rand_double();
}

/* Naive triple-loop dgemm reference: C = A(m x k) * B(k x n), row-major. */
static void naive_dgemm(int m, int n, int k,
                        const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)m * n * sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int p = 0; p < k; p++)
                C[i * n + j] += A[i * k + p] * B[p * n + j];
}

/* Max relative error between two arrays (skip near-zero ref). */
static double max_rel_diff(const double *a, const double *b, int n) {
    double max_rd = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(a[i] - b[i]);
        double scale = fabs(b[i]);
        if (scale < 1e-15) scale = 1.0;  /* avoid div-by-zero for near-zero ref */
        double rd = diff / scale;
        if (rd > max_rd) max_rd = rd;
    }
    return max_rd;
}

/* Make symmetric positive definite: K = A * A^T + eps * I */
static void make_spd(double *K, int n, unsigned seed) {
    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    if (!A) { TEST_FAIL_MESSAGE("malloc failed in make_spd"); return; }
    fill_random(A, n * n, seed);
    /* K = A * A^T (manual transpose — naive_dgemm has no transpose mode) */
    memset(K, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int p = 0; p < n; p++)
                K[i * n + j] += A[i * n + p] * A[j * n + p];
    /* Add eps * I for positive definiteness */
    for (int i = 0; i < n; i++)
        K[i * n + i] += 1.0;
    free(A);
}

/* Frobenius norm of an n x n matrix. */
static double frobenius_norm(const double *A, int n) {
    double sum = 0.0;
    for (int i = 0; i < n * n; i++)
        sum += A[i] * A[i];
    return sqrt(sum);
}

/* -------------------------------------------------------------------------
 * dgemm boundary tests
 * ------------------------------------------------------------------------- */

static void _test_dgemm_size(int m, int n, int k, unsigned seed) {
    double *A   = (double *)calloc((size_t)m * k, sizeof(double));
    double *B   = (double *)calloc((size_t)k * n, sizeof(double));
    double *C   = (double *)calloc((size_t)m * n, sizeof(double));
    double *ref = (double *)calloc((size_t)m * n, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(A, "malloc A");
    TEST_ASSERT_NOT_NULL_MESSAGE(B, "malloc B");
    TEST_ASSERT_NOT_NULL_MESSAGE(C, "malloc C");
    TEST_ASSERT_NOT_NULL_MESSAGE(ref, "malloc ref");

    fill_random(A, m * k, seed);
    fill_random(B, k * n, seed + 1);

    /* jlinalg dgemm */
    jlinalg_dispatch.dgemm((npy_intp)m, (npy_intp)n, (npy_intp)k, A, B, C);

    /* Reference naive dgemm */
    naive_dgemm(m, n, k, A, B, ref);

    double rd = max_rel_diff(C, ref, m * n);
    char msg[256];
    snprintf(msg, sizeof(msg),
             "dgemm(%d,%d,%d): max_rel_diff=%.3e (limit 1e-10)", m, n, k, rd);
    TEST_ASSERT_LESS_THAN_DOUBLE_MESSAGE(1e-10, rd, msg);

    free(A); free(B); free(C); free(ref);
}

/* Square M=N=K tests at specific boundaries */
void test_dgemm_1x1(void)        { _test_dgemm_size(1, 1, 1, 100); }
void test_dgemm_mr_minus_1(void) { _test_dgemm_size(JLINALG_MR - 1, JLINALG_MR - 1, JLINALG_MR - 1, 101); }
void test_dgemm_mr(void)         { _test_dgemm_size(JLINALG_MR, JLINALG_MR, JLINALG_MR, 102); }
void test_dgemm_mr_plus_1(void)  { _test_dgemm_size(JLINALG_MR + 1, JLINALG_MR + 1, JLINALG_MR + 1, 103); }
void test_dgemm_mc_minus_1(void) { _test_dgemm_size(JLINALG_MC - 1, JLINALG_NR, JLINALG_KC, 104); }
void test_dgemm_mc(void)         { _test_dgemm_size(JLINALG_MC, JLINALG_NR, JLINALG_KC, 105); }
void test_dgemm_mc_plus_1(void)  { _test_dgemm_size(JLINALG_MC + 1, JLINALG_NR, JLINALG_KC, 106); }
void test_dgemm_kc_minus_1(void) { _test_dgemm_size(JLINALG_MR, JLINALG_NR, JLINALG_KC - 1, 107); }
void test_dgemm_kc(void)         { _test_dgemm_size(JLINALG_MR, JLINALG_NR, JLINALG_KC, 108); }
void test_dgemm_kc_plus_1(void)  { _test_dgemm_size(JLINALG_MR, JLINALG_NR, JLINALG_KC + 1, 109); }

/* -------------------------------------------------------------------------
 * dsyrk boundary tests
 * ------------------------------------------------------------------------- */

static void _test_dsyrk_size(int n, int k, unsigned seed) {
    double *X   = (double *)calloc((size_t)n * k, sizeof(double));
    double *C   = (double *)calloc((size_t)n * n, sizeof(double));
    double *ref = (double *)calloc((size_t)n * n, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(X, "malloc X");
    TEST_ASSERT_NOT_NULL_MESSAGE(C, "malloc C");
    TEST_ASSERT_NOT_NULL_MESSAGE(ref, "malloc ref");

    fill_random(X, n * k, seed);

    /* jlinalg dsyrk: C = X @ X^T */
    jlinalg_dsyrk_c((npy_intp)n, (npy_intp)k, X, (npy_intp)k, C, (npy_intp)n);

    /* Reference: ref = X @ X^T via naive */
    /* ref[i][j] = sum_p X[i][p] * X[j][p] */
    memset(ref, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int p = 0; p < k; p++)
                ref[i * n + j] += X[i * k + p] * X[j * k + p];

    double rd = max_rel_diff(C, ref, n * n);
    char msg[256];
    snprintf(msg, sizeof(msg),
             "dsyrk(%d,%d): max_rel_diff=%.3e (limit 1e-10)", n, k, rd);
    TEST_ASSERT_LESS_THAN_DOUBLE_MESSAGE(1e-10, rd, msg);

    free(X); free(C); free(ref);
}

void test_dsyrk_1x1(void)        { _test_dsyrk_size(1, 1, 200); }
void test_dsyrk_mr_minus_1(void) { _test_dsyrk_size(JLINALG_MR - 1, 64, 201); }
void test_dsyrk_mr(void)         { _test_dsyrk_size(JLINALG_MR, 64, 202); }
void test_dsyrk_mr_plus_1(void)  { _test_dsyrk_size(JLINALG_MR + 1, 64, 203); }
void test_dsyrk_mc(void)         { _test_dsyrk_size(JLINALG_MC, 64, 204); }

/* -------------------------------------------------------------------------
 * eigh boundary tests
 * ------------------------------------------------------------------------- */

static void _test_eigh_size(int n, double recon_tol, double ortho_tol,
                            unsigned seed) {
    double *K   = (double *)calloc((size_t)n * n, sizeof(double));
    double *K_orig = (double *)calloc((size_t)n * n, sizeof(double));
    double *eigenvalues = (double *)calloc((size_t)n, sizeof(double));
    double *eigenvectors = (double *)calloc((size_t)n * n, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(K, "malloc K");
    TEST_ASSERT_NOT_NULL_MESSAGE(K_orig, "malloc K_orig");
    TEST_ASSERT_NOT_NULL_MESSAGE(eigenvalues, "malloc eigenvalues");
    TEST_ASSERT_NOT_NULL_MESSAGE(eigenvectors, "malloc eigenvectors");

    make_spd(K, n, seed);
    memcpy(K_orig, K, (size_t)n * n * sizeof(double));

    jlinalg_eigh_status_t status;
    memset(&status, 0, sizeof(status));
    int info = jlinalg_eigh_c((npy_intp)n, K, (npy_intp)n,
                              eigenvalues, eigenvectors, (npy_intp)n, &status);

    char msg[256];
    snprintf(msg, sizeof(msg), "eigh(%d): jlinalg_eigh_c returned %d", n, info);
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, info, msg);

    /* Check reconstruction: ||K_orig - V diag(w) V^T||_F / ||K_orig||_F */
    double *reconstructed = (double *)calloc((size_t)n * n, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(reconstructed, "malloc reconstructed");
    /* reconstructed = V * diag(w) * V^T */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double val = 0.0;
            for (int k = 0; k < n; k++)
                val += eigenvectors[i * n + k] * eigenvalues[k] * eigenvectors[j * n + k];
            reconstructed[i * n + j] = val;
        }
    /* diff = reconstructed - K_orig */
    for (int i = 0; i < n * n; i++)
        reconstructed[i] -= K_orig[i];
    double recon_err = frobenius_norm(reconstructed, n) / frobenius_norm(K_orig, n);
    snprintf(msg, sizeof(msg),
             "eigh(%d): reconstruction error=%.3e (limit %.0e)", n, recon_err, recon_tol);
    TEST_ASSERT_LESS_THAN_DOUBLE_MESSAGE(recon_tol, recon_err, msg);

    /* Check orthogonality: ||V^T V - I||_F */
    double *VtV = (double *)calloc((size_t)n * n, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(VtV, "malloc VtV");
    /* VtV = V^T * V (V is row-major: V[i][j] = eigenvectors[i*n+j]) */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double val = 0.0;
            for (int k = 0; k < n; k++)
                val += eigenvectors[k * n + i] * eigenvectors[k * n + j];
            VtV[i * n + j] = val;
        }
    /* Subtract identity */
    for (int i = 0; i < n; i++)
        VtV[i * n + i] -= 1.0;
    double ortho_err = frobenius_norm(VtV, n);
    snprintf(msg, sizeof(msg),
             "eigh(%d): orthogonality error=%.3e (limit %.0e)", n, ortho_err, ortho_tol);
    TEST_ASSERT_LESS_THAN_DOUBLE_MESSAGE(ortho_tol, ortho_err, msg);

    free(K); free(K_orig); free(eigenvalues); free(eigenvectors);
    free(reconstructed); free(VtV);
}

/* DSTEDC_BASE = 64 in the current implementation (base case for QR iteration).
 * NB_DSYTRD = 64 (block size for tridiagonal reduction).
 *
 * N <= 64 uses QR iteration (base case) → machine-precision accuracy (1e-13).
 * N = 65 triggers the D&C recursive path with secular equation solving,
 * which introduces conditioning-dependent error → relaxed to 1e-8. */
void test_eigh_small(void)               { _test_eigh_size(5,  1e-14, 1e-14, 300); }
void test_eigh_dstedc_base(void)         { _test_eigh_size(64, 1e-13, 1e-13, 301); }
void test_eigh_dstedc_base_plus_1(void)  { _test_eigh_size(65, 1e-8,  1e-8,  302); }
void test_eigh_nb_dsytrd(void)           { _test_eigh_size(64, 1e-13, 1e-13, 303); }
void test_eigh_nb_dsytrd_plus_1(void)    { _test_eigh_size(65, 1e-8,  1e-8,  304); }

/* -------------------------------------------------------------------------
 * Unity main
 * ------------------------------------------------------------------------- */

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    /* Python must be initialized before jlinalg_init() because
     * blas_dispatch.c uses Python C API to discover numpy's BLAS. */
    Py_Initialize();

    int init_rc = jlinalg_init();
    if (init_rc != 0) {
        fprintf(stderr, "jlinalg_init() failed with %d\n", init_rc);
        Py_Finalize();
        return 1;
    }

    printf("jlinalg ISA: %s\n", jlinalg_isa_name());
    printf("Blocking: MR=%d NR=%d KC=%d MC=%d NC=%d\n",
           JLINALG_MR, JLINALG_NR, JLINALG_KC, JLINALG_MC, JLINALG_NC);

    UNITY_BEGIN();

    /* dgemm boundary tests */
    RUN_TEST(test_dgemm_1x1);
    RUN_TEST(test_dgemm_mr_minus_1);
    RUN_TEST(test_dgemm_mr);
    RUN_TEST(test_dgemm_mr_plus_1);
    RUN_TEST(test_dgemm_mc_minus_1);
    RUN_TEST(test_dgemm_mc);
    RUN_TEST(test_dgemm_mc_plus_1);
    RUN_TEST(test_dgemm_kc_minus_1);
    RUN_TEST(test_dgemm_kc);
    RUN_TEST(test_dgemm_kc_plus_1);

    /* dsyrk boundary tests */
    RUN_TEST(test_dsyrk_1x1);
    RUN_TEST(test_dsyrk_mr_minus_1);
    RUN_TEST(test_dsyrk_mr);
    RUN_TEST(test_dsyrk_mr_plus_1);
    RUN_TEST(test_dsyrk_mc);

    /* eigh boundary tests */
    RUN_TEST(test_eigh_small);
    RUN_TEST(test_eigh_dstedc_base);
    RUN_TEST(test_eigh_dstedc_base_plus_1);
    RUN_TEST(test_eigh_nb_dsytrd);
    RUN_TEST(test_eigh_nb_dsytrd_plus_1);

    int result = UNITY_END();
    Py_Finalize();
    return result;
}
