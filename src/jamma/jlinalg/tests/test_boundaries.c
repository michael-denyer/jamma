/**
 * test_boundaries.c -- C-level boundary tests for jlinalg via Unity framework.
 *
 * Tests vendor-dispatched eigh at various matrix sizes directly from C,
 * without Python overhead.
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

/* Make symmetric positive definite: K = A * A^T + eps * I */
static void make_spd(double *K, int n, unsigned seed) {
    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    if (!A) { TEST_FAIL_MESSAGE("malloc failed in make_spd"); return; }
    fill_random(A, n * n, seed);
    memset(K, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int p = 0; p < n; p++)
                K[i * n + j] += A[i * n + p] * A[j * n + p];
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

/* Vendor DSYEVD/DSYEVR boundary tests at various matrix sizes.
 * Tolerances are conservative (1e-13 for small, 1e-8 for larger).
 *
 * The 64 and 65 sizes appear twice each, differing only in seed.  Vendor
 * LAPACK switches blocking around 64, so a defect that only shows up either
 * side of that step is worth two independent draws rather than one. */
void test_eigh_small(void)          { _test_eigh_size(5,  1e-14, 1e-14, 300); }
void test_eigh_64(void)             { _test_eigh_size(64, 1e-13, 1e-13, 301); }
void test_eigh_65(void)             { _test_eigh_size(65, 1e-8,  1e-8,  302); }
void test_eigh_64_alt_seed(void)    { _test_eigh_size(64, 1e-13, 1e-13, 303); }
void test_eigh_65_alt_seed(void)    { _test_eigh_size(65, 1e-8,  1e-8,  304); }

/* The wrappers pass raw LAPACK info (info = -i for illegal argument i) straight
 * back to eigh.c and pymodule.c, which both compare it against the sentinels.
 * If any sentinel fell inside LAPACK's [-JLINALG_LAPACK_MAX_ARG, -1] argument
 * range, a genuine argument error would read as that sentinel and be swallowed
 * (fall through to DSYEVR, or a spurious NumPy-fallback / MemoryError). This
 * asserts the bands stay disjoint; the _Static_assert in jlinalg.h guards it at
 * compile time, and this test surfaces the same invariant through the harness. */
void test_ext_sentinels_disjoint_from_lapack_arg_range(void) {
    const int sentinels[] = {
        JLINALG_EXT_ALLOC_FAIL, JLINALG_EXT_UNAVAILABLE, JLINALG_EXT_COUNT_MISMATCH,
        JLINALG_EXT_INTERNAL_ERROR, JLINALG_EXT_INPLACE_UNSUPPORTED,
    };
    for (size_t i = 0; i < sizeof(sentinels) / sizeof(sentinels[0]); i++) {
        TEST_ASSERT_LESS_THAN_INT_MESSAGE(
            -JLINALG_LAPACK_MAX_ARG, sentinels[i],
            "JLINALG_EXT_* sentinel overlaps the LAPACK info argument range");
    }
}

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

    UNITY_BEGIN();

    /* eigh boundary tests */
    RUN_TEST(test_eigh_small);
    RUN_TEST(test_eigh_64);
    RUN_TEST(test_eigh_65);
    RUN_TEST(test_eigh_64_alt_seed);
    RUN_TEST(test_eigh_65_alt_seed);
    RUN_TEST(test_ext_sentinels_disjoint_from_lapack_arg_range);

    int result = UNITY_END();
    Py_Finalize();
    return result;
}
