/*
 * _lmm_accel.c — C extension implementing per-SNP REML/MLE pipelines
 * for Wald, Score, and LRT tests (n_cvt=1 and general n_cvt).
 *
 * Exported functions: compute_lmm_batch_c, compute_lmm_batch_split_c,
 *                     create_workspace_split_c, compute_lmm_chunk_split_c,
 *                     create_workspace_general_c, compute_lmm_chunk_general_c,
 *                     compute_score_batch_c, compute_lrt_batch_c,
 *                     create_workspace_mode4_split_c, compute_mode4_chunk_split_c
 *
 * Translates the Python/NumPy golden-section REML/MLE optimizer + Wald/Score/LRT
 * test pipelines (likelihood_numpy.py) to C with optional OpenMP parallelism.
 *
 * Performance optimizations over the naive per-call approach (n_cvt=1 path;
 * the general n_cvt path uses table-driven recursion with cached invariant
 * dot products — see "GENERAL n_cvt support" section below):
 *   1. Fused Pab: single pass over n_samples accumulates all 6 dot products
 *   2. Thread-local hi_eval: one malloc per worker thread, reused across SNPs
 *   3. Pre-computed logdet_iab: lambda-independent log(iab) terms computed once
 *   4. Pre-built lambda grid: avoids exp() in every SNP's coarse search loop
 *   5. Hoisted constants: REML normalizer + betainc lgamma terms computed once
 *   6. C-side betainc: Lentz CF for F->p-value avoids Python round-trip
 *   7. Cached coarse-grid hi_eval: hi_eval[g][i] and logdet_h[g] precomputed
 *      once across all SNPs — eliminates n_snps * n_grid redundant hi_eval passes
 *   8. restrict + SIMD hints: helps compiler vectorize hot inner loops
 *      (#pragma omp simd is used without #ifdef _OPENMP guards — unknown
 *      pragmas are silently ignored per the C standard, so these are safe
 *      on non-OpenMP compilers and act purely as vectorization hints)
 *
 * Pab indexing (n_cvt=1, build_index_table(1)):
 *   n_index = 6
 *   col 0: ww = GetabIndex(1,1,1) = 0
 *   col 1: wx = GetabIndex(1,2,1) = 1
 *   col 2: wy = GetabIndex(1,3,1) = 2
 *   col 3: xx = GetabIndex(2,2,1) = 3
 *   col 4: xy = GetabIndex(2,3,1) = 4
 *   col 5: yy = GetabIndex(3,3,1) = 5
 *
 *   Row 0: dot products (all 6 columns)
 *   Row 1 (project W):
 *     Pab[1][3] = Pab[0][3] - Pab[0][1]*Pab[0][1] / Pab[0][0]  (xx)
 *     Pab[1][4] = Pab[0][4] - Pab[0][1]*Pab[0][2] / Pab[0][0]  (xy)
 *     Pab[1][5] = Pab[0][5] - Pab[0][2]*Pab[0][2] / Pab[0][0]  (yy)
 *   Row 2 (project X):
 *     Pab[2][5] = Pab[1][5] - Pab[1][4]*Pab[1][4] / Pab[1][3]  (yy)
 *
 *   logdet_diag_indices: [(0, 0), (1, 3)]
 *   idx_xx = 3, idx_xy = 4, idx_yy = 5
 *   nc_total = n_cvt + 1 = 2 (Pab row for Px_YY)
 *   df = n_samples - 2 (n_cvt=1)
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Minimum P_yy guard — matches _P_YY_MIN in likelihood.py */
#define P_YY_MIN 1e-8

/* ABI version: bump when function signatures or array layout expectations change.
 * The Python side checks this at import time to detect stale .so files. */
#define ABI_VERSION 6  /* v6: Fused mode-4 kernel (Wald/Score/LRT single pass from SoA split) */

/* Betainc continued fraction constants — matches special.py */
#define CF_TINY     1.0e-30
#define CF_STOP     1.0e-14
#define CF_MAX_ITER 200

/* REML sentinel: replaces NaN log-likelihood from degenerate P_yy.
 * reml_finish returns NaN when P_yy < 0; the golden section callers
 * map NaN -> REML_SENTINEL so the > comparison skips degenerate points
 * without needing an isnan() guard on every iteration.
 * Matches the Python path's np.where(isnan, -inf, logl). */
#define REML_SENTINEL (-INFINITY)

/* ---------------------------------------------------------------------------
 * alloc_aligned_doubles — allocate n doubles with 32-byte alignment (AVX2).
 *
 * Uses aligned_alloc (C11). Size is rounded up to a 32-byte multiple as
 * required by the C11 spec.
 * Returns NULL on failure or if n == 0 (caller checks).
 * ------------------------------------------------------------------------- */
static double *alloc_aligned_doubles(size_t n)
{
    if (n == 0) return NULL;
    size_t raw = n * sizeof(double);
    if (raw / sizeof(double) != n) return NULL;  /* overflow check */
    size_t bytes = (raw + 31) & ~(size_t)31;
    return (double *)aligned_alloc(32, bytes);
}

/* ---------------------------------------------------------------------------
 * validate_eigenvalues — reject NaN/Inf before entering the compute loop.
 *
 * Without this check, non-finite eigenvalues silently propagate through the
 * entire REML pipeline, producing garbage lambda/beta/SE/p-value results
 * with no error or warning.  O(n_samples) scan — negligible vs O(n*m*k).
 * ------------------------------------------------------------------------- */
static int validate_eigenvalues(const double *data, int n_samples)
{
    for (int i = 0; i < n_samples; i++) {
        if (!isfinite(data[i])) {
            /* PyErr_Format doesn't support %g — use snprintf + %s instead */
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", data[i]);
            PyErr_Format(PyExc_ValueError,
                "eigenvalues[%d] = %s is not finite. "
                "Check kinship matrix and eigendecomposition quality.", i, buf);
            return -1;
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * warn_betainc_convergence — post-compute scan for NaN p-values where beta
 * is finite (i.e., stats computed fine but betainc CF didn't converge).
 * Used for Wald and Score p-values.
 *
 * Called after Py_END_ALLOW_THREADS so Python API is available.
 * Returns 0 on success, -1 if the warning was promoted to an exception
 * (e.g. user has simplefilter("error")).
 * ------------------------------------------------------------------------- */
static int warn_betainc_convergence(
    const double *betas, const double *pvalues, int n_snps)
{
    int n_betainc_nan = 0;
    for (int i = 0; i < n_snps; i++) {
        if (isfinite(betas[i]) && !isfinite(pvalues[i]))
            n_betainc_nan++;
    }
    if (n_betainc_nan > 0) {
        if (PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                "%d SNPs have NaN p-values despite finite beta/SE — "
                "betainc continued fraction did not converge "
                "(extreme F-statistics). Consider checking these SNPs manually.",
                n_betainc_nan) < 0) {
            return -1;  /* warning promoted to exception */
        }
    }
    return 0;
}

/* =========================================================================
 * Shared helpers — eliminate duplication across full/split paths
 * ========================================================================= */

/* Output array bundle — passed by pointer to avoid repeating 5 Py_DECREF chains */
typedef struct {
    PyArrayObject *lambdas;
    PyArrayObject *logls;
    PyArrayObject *betas;
    PyArrayObject *ses;
    PyArrayObject *pwalds;
} output_arrays_t;

/* Allocate 5 output arrays of shape (n_snps,). Returns 0 on success, -1 on failure. */
static int alloc_output_arrays(output_arrays_t *out, npy_intp n_snps)
{
    npy_intp dims[1] = { n_snps };
    out->lambdas = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->logls   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->betas   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->ses     = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->pwalds  = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!out->lambdas || !out->logls || !out->betas || !out->ses || !out->pwalds) {
        Py_XDECREF(out->lambdas);
        Py_XDECREF(out->logls);
        Py_XDECREF(out->betas);
        Py_XDECREF(out->ses);
        Py_XDECREF(out->pwalds);
        return -1;
    }
    return 0;
}

static void decref_output_arrays(output_arrays_t *out)
{
    Py_DECREF(out->lambdas);
    Py_DECREF(out->logls);
    Py_DECREF(out->betas);
    Py_DECREF(out->ses);
    Py_DECREF(out->pwalds);
}

/* Build a result dict from 5 output arrays. Returns new dict, or NULL on error.
 * On success, releases the caller's references to the output arrays
 * (dict holds its own references via PyDict_SetItemString).
 * On failure, decrefs the output arrays and returns NULL. */
static PyObject *build_result_dict(output_arrays_t *out)
{
    PyObject *result = PyDict_New();
    if (!result) {
        decref_output_arrays(out);
        return NULL;
    }

    if (PyDict_SetItemString(result, "lambdas",  (PyObject *)out->lambdas)  < 0 ||
        PyDict_SetItemString(result, "logls",    (PyObject *)out->logls)    < 0 ||
        PyDict_SetItemString(result, "betas",    (PyObject *)out->betas)    < 0 ||
        PyDict_SetItemString(result, "ses",      (PyObject *)out->ses)      < 0 ||
        PyDict_SetItemString(result, "pwalds",   (PyObject *)out->pwalds)   < 0) {
        Py_DECREF(result);
        decref_output_arrays(out);
        return NULL;
    }

    decref_output_arrays(out);
    return result;
}

/* Validate shared batch parameters. Returns 0 on success, -1 with PyErr set. */
static int validate_batch_params(int n_samples, double l_min, double l_max,
                                 int n_grid, int n_refine)
{
    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return -1;
    }
    if (!(l_min > 0.0 && l_max > l_min)) {
        PyErr_SetString(PyExc_ValueError, "Require 0 < l_min < l_max");
        return -1;
    }
    if (n_grid < 2) {
        PyErr_SetString(PyExc_ValueError, "n_grid must be >= 2");
        return -1;
    }
    if (n_refine < 1) {
        PyErr_SetString(PyExc_ValueError, "n_refine must be >= 1");
        return -1;
    }
    return 0;
}

/* REML log-likelihood tail: logdet_pab + P_yy guard + REML formula.
 * Shared by reml_logl_ncvt1, reml_logl_ncvt1_cached, reml_logl_ncvt1_split. */
static inline double reml_finish(
    const double pab[3][6],
    double logdet_h,
    double logdet_iab,
    int df,
    double reml_const
)
{
    double logdet_pab = 0.0;
    if (pab[0][0] > 0.0) logdet_pab += log(pab[0][0]);
    if (pab[1][3] > 0.0) logdet_pab += log(pab[1][3]);
    double logdet_hiw = logdet_pab - logdet_iab;

    double P_yy = pab[2][5];
    if (P_yy < 0.0) {
        P_yy = (double)NAN;
    } else if (P_yy < P_YY_MIN) {
        P_yy = P_YY_MIN;
    }

    return reml_const - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * log(P_yy);
}

/* Wald statistics from a populated pab array.
 * Shared by golden_section_lambda_ncvt1 and golden_section_lambda_ncvt1_split.
 *
 * Returns 1 if the SNP is valid (P_XX > 0), 0 if degenerate (P_XX <= 0).
 * Degenerate SNPs get beta = se = f_stat = NaN.
 *
 * The return value (not isnan(beta)) is used for validity checks — this is
 * more robust than relying on NaN propagation through comparisons. */
static inline int wald_from_pab(
    const double pab[3][6],
    int df,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    double P_XX  = pab[1][3];
    double P_XY  = pab[1][4];
    double P_YY  = pab[1][5];
    double Px_YY = pab[2][5];

    if (Px_YY < 0.0) {
        /* Schur complement went negative — degenerate SNP. Without this
         * guard, small negative Px_YY passes through variance_safe's fabs
         * branch and produces a fabricated positive SE with is_valid=1. */
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;  /* degenerate */
    }
    if (Px_YY < P_YY_MIN) {
        Px_YY = P_YY_MIN;
    }

    if (P_XX <= 0.0) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;  /* degenerate */
    }

    double beta = P_XY / P_XX;

    /* SE via JAMMA's corrected safe_sqrt (see GEMMA_DIVERGENCES.md section 1):
     * if |var| < 0.001, use fabs(var) instead of var to avoid sqrt of tiny
     * negative FP rounding artifacts. */
    double tau = (double)df / Px_YY;
    double variance_beta = 1.0 / (tau * P_XX);
    double variance_safe = (fabs(variance_beta) < 0.001)
                            ? fabs(variance_beta)
                            : variance_beta;
    double se = sqrt(variance_safe);

    double f_stat = (P_YY - Px_YY) * tau;

    *beta_out   = beta;
    *se_out     = se;
    *f_stat_out = f_stat;

    /* Guard against non-finite results from pathological Px_YY / tau.
     * Without this, NaN f_stat passes is_valid=1 to f_to_pvalue, which
     * clamps NaN to 1e-10 and returns a bogus near-1 p-value. */
    if (!isfinite(f_stat) || !isfinite(beta) || !isfinite(se))
        return 0;

    return 1;  /* valid */
}

/* -------------------------------------------------------------------------
 * score_from_pab
 *
 * Score test statistics from a populated pab array (n_cvt=1).
 *
 * Reads from two pab levels:
 *   - Level 1 (pab[1]): P_yy, P_xx, P_xy for Score F-statistic
 *   - Level 2 (pab[2]): Px_yy for beta/SE computation (same as Wald)
 *
 * Key differences from Wald:
 *   - F = n_samples * P_xy^2 / (P_yy * P_xx) — uses n_samples, not df
 *   - No per-SNP lambda optimization (uses null-model Hi_eval)
 *
 * Returns 1 if valid, 0 if degenerate (P_xx <= 0 or P_yy < 0 or Px_yy < 0
 * or any output non-finite).
 * ------------------------------------------------------------------------- */
static inline int score_from_pab(
    const double pab[3][6],
    int n_samples,
    int df,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    /* Score extracts at level n_cvt=1 (row 1), NOT n_cvt+1=2 */
    double P_yy = pab[1][5];
    double P_xx = pab[1][3];
    double P_xy = pab[1][4];
    /* Px_yy at level n_cvt+1=2 for beta/se computation */
    double Px_yy = pab[2][5];

    if (P_xx <= 0.0 || P_yy < 0.0 || Px_yy < 0.0) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;  /* degenerate */
    }

    /* Clamp P_yy for F-stat denominator */
    if (P_yy < P_YY_MIN) P_yy = P_YY_MIN;
    /* Clamp Px_yy for beta/se */
    if (Px_yy < P_YY_MIN) Px_yy = P_YY_MIN;

    *beta_out = P_xy / P_xx;

    double tau = (double)df / Px_yy;
    double variance_beta = 1.0 / (tau * P_xx);
    double variance_safe = (fabs(variance_beta) < 0.001)
                            ? fabs(variance_beta)
                            : variance_beta;
    *se_out = sqrt(variance_safe);

    /* Score F-statistic: uses n_samples (not df) in numerator */
    *f_stat_out = (double)n_samples * (P_xy * P_xy) / (P_yy * P_xx);

    if (!isfinite(*f_stat_out) || !isfinite(*beta_out) || !isfinite(*se_out)) {
        *beta_out   = (double)NAN;
        *se_out     = (double)NAN;
        *f_stat_out = (double)NAN;
        return 0;
    }

    return 1;  /* valid */
}

/* -------------------------------------------------------------------------
 * betainc_cf
 *
 * Lentz continued fraction for regularized incomplete beta I_x(a, b).
 * Based on special.py _betainc_cf / codeplea incbeta (zlib license).
 * Differs: takes precomputed lbeta_ab to avoid per-call lgamma;
 * returns NaN (not exception) on non-convergence.
 * Caller guarantees x < (a+1)/(a+b+2) (symmetry threshold).
 * ------------------------------------------------------------------------- */
static double betainc_cf(double a, double b, double x, double lbeta_ab)
{
    double front = exp(log(x) * a + log(1.0 - x) * b - lbeta_ab) / a;

    double f = 1.0, c = 1.0, d = 0.0;

    for (int i = 0; i <= CF_MAX_ITER; i++) {
        int m = i / 2;
        double numerator;
        if (i == 0) {
            numerator = 1.0;
        } else if (i % 2 == 0) {
            double mf = (double)m;
            numerator = (mf * (b - mf) * x) /
                        ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        } else {
            double mf = (double)m;
            numerator = -((a + mf) * (a + b + mf) * x) /
                         ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        }

        d = 1.0 + numerator * d;
        if (fabs(d) < CF_TINY) d = CF_TINY;
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if (fabs(c) < CF_TINY) c = CF_TINY;

        double cd = c * d;
        f *= cd;

        if (fabs(1.0 - cd) < CF_STOP) {
            return front * (f - 1.0);
        }
    }
    return (double)NAN;  /* non-convergence */
}

/* -------------------------------------------------------------------------
 * betainc
 *
 * Regularized incomplete beta I_z(a, b) with symmetry relation.
 * Matches special.py betainc() scalar interface.
 *
 * complement_z is the algebraically exact 1-z, used for precision near z=1.
 * ------------------------------------------------------------------------- */
static double betainc(
    double a,
    double b,
    double z,
    double complement_z,
    double lbeta_ab
)
{
    if (z <= 0.0) return 0.0;
    if (z >= 1.0) return 1.0;

    double threshold = (a + 1.0) / (a + b + 2.0);
    if (z <= threshold) {
        return betainc_cf(a, b, z, lbeta_ab);
    } else {
        return 1.0 - betainc_cf(b, a, complement_z, lbeta_ab);
    }
}

/* -------------------------------------------------------------------------
 * f_to_pvalue
 *
 * Convert F-statistic to p-value via regularized incomplete beta.
 * Matches _f_to_pvalue in likelihood_numpy.py.
 * Returns NaN if is_valid is false (degenerate SNP).
 * ------------------------------------------------------------------------- */
static double f_to_pvalue(
    double f_stat,
    int df,
    int is_valid,
    double a,
    double b,
    double lbeta_ab
)
{
    if (!is_valid) return (double)NAN;
    if (f_stat <= 0.0) return 1.0;

    double f_safe = (f_stat > 1e-10) ? f_stat : 1e-10;
    double denom = (double)df + f_safe;
    double z = (double)df / denom;
    double complement_z = f_safe / denom;  /* algebraically exact 1-z */

    if (z < 0.0) z = 0.0;
    if (z > 1.0) z = 1.0;

    double p = betainc(a, b, z, complement_z, lbeta_ab);
    /* Clamp to [0, 1] — continued fraction FP accumulation can overshoot. */
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

/* -------------------------------------------------------------------------
 * calc_pab_ncvt1
 *
 * Compute all three Pab rows for n_cvt=1.
 * Single fused pass over n_samples accumulates all 6 dot products,
 * then computes the two projection rows from the accumulators.
 *
 * uab:      (n_samples, 6) C-contiguous, row-major
 * hi_eval:  (n_samples,) = 1 / (lambda * eigenvalues + 1)
 * n_samples: number of samples
 * pab:      output 3x6 array (caller-allocated)
 * ------------------------------------------------------------------------- */
static void calc_pab_ncvt1(
    const double * restrict uab,
    const double * restrict hi_eval,
    int n_samples,
    double pab[3][6]
)
{
    /* Fused single pass: accumulate all 6 dot products simultaneously.
     * uab is row-major (n_samples, 6), so row i starts at uab[i*6].
     * Accessing 6 consecutive doubles per row is cache-friendly. */
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
    #pragma omp simd reduction(+:s0,s1,s2,s3,s4,s5)
    for (int i = 0; i < n_samples; i++) {
        double h = hi_eval[i];
        const double *row = uab + i * 6;
        s0 += h * row[0];
        s1 += h * row[1];
        s2 += h * row[2];
        s3 += h * row[3];
        s4 += h * row[4];
        s5 += h * row[5];
    }
    pab[0][0] = s0;
    pab[0][1] = s1;
    pab[0][2] = s2;
    pab[0][3] = s3;
    pab[0][4] = s4;
    pab[0][5] = s5;

    /* Row 1: project out W (column index 0 = ww) */
    {
        double inv_ww = (s0 != 0.0) ? 1.0 / s0 : 0.0;
        pab[1][3] = s3 - s1 * s1 * inv_ww;  /* xx */
        pab[1][4] = s4 - s1 * s2 * inv_ww;  /* xy */
        pab[1][5] = s5 - s2 * s2 * inv_ww;  /* yy */
    }

    /* Row 2: project out X (column index 3 = xx at level 1) */
    {
        double ps_xx = pab[1][3];
        double inv_xx = (ps_xx != 0.0) ? 1.0 / ps_xx : 0.0;
        pab[2][5] = pab[1][5] - pab[1][4] * pab[1][4] * inv_xx;  /* yy */
    }
}

/* -------------------------------------------------------------------------
 * reml_logl_ncvt1
 *
 * REML log-likelihood for one SNP at one lambda (n_cvt=1).
 * Used during golden section refinement where lambda is SNP-specific.
 *
 * Returns REML log-likelihood (positive = better; optimizer maximises).
 * ------------------------------------------------------------------------- */
static double reml_logl_ncvt1(
    const double * restrict uab,
    const double * restrict eigenvalues,
    double logdet_iab,
    int n_samples,
    double lambda,
    double reml_const,
    double * restrict hi_eval
)
{
    int df = n_samples - 2;

    double logdet_h = 0.0;
    #pragma omp simd reduction(+:logdet_h)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        hi_eval[i] = 1.0 / v;
        logdet_h += log(v);  /* v > 1.0: lambda > 0, eval >= 0 */
    }

    double pab[3][6];
    calc_pab_ncvt1(uab, hi_eval, n_samples, pab);

    return reml_finish(pab, logdet_h, logdet_iab, df, reml_const);
}

/* -------------------------------------------------------------------------
 * reml_logl_ncvt1_cached
 *
 * REML log-likelihood using precomputed hi_eval and logdet_h from the
 * shared coarse grid cache. Avoids recomputing 1/(lambda*eval+1) and
 * log(v) for every SNP at every grid point.
 *
 * Returns REML log-likelihood.
 * ------------------------------------------------------------------------- */
static double reml_logl_ncvt1_cached(
    const double * restrict uab,
    const double * restrict cached_hi_eval,
    double cached_logdet_h,
    double logdet_iab,
    int n_samples,
    int df,
    double reml_const
)
{
    double pab[3][6];
    calc_pab_ncvt1(uab, cached_hi_eval, n_samples, pab);

    return reml_finish(pab, cached_logdet_h, logdet_iab, df, reml_const);
}

/* -------------------------------------------------------------------------
 * compute_logdet_iab
 *
 * Pre-compute the lambda-independent log(iab) terms for one SNP.
 * logdet_diag_indices for n_cvt=1: [(0, 0), (1, 3)]
 *   iab layout: row r, col c -> iab[r*6 + c]
 *
 * This is called once per SNP before the optimization loop, avoiding
 * redundant log() calls across n_grid + n_refine + 3 logl evaluations per SNP.
 * ------------------------------------------------------------------------- */
static double compute_logdet_iab(const double *iab)
{
    double logdet = 0.0;
    /* (row=0, col=0) = ww */
    if (iab[0 * 6 + 0] > 0.0) logdet += log(iab[0 * 6 + 0]);
    /* (row=1, col=3) = xx at level 1 */
    if (iab[1 * 6 + 3] > 0.0) logdet += log(iab[1 * 6 + 3]);
    return logdet;
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_ncvt1
 *
 * Grid search + golden section refinement in log-lambda space.
 * Returns optimal lambda; writes optimal logl to *logl_out.
 *
 * Coarse search uses precomputed hi_eval_grid and logdet_h_grid (shared
 * across all SNPs). Refinement uses per-eval reml_logl_ncvt1 since
 * refinement lambdas are SNP-specific.
 * ------------------------------------------------------------------------- */
static double golden_section_lambda_ncvt1(
    const double * restrict uab,
    const double * restrict eigenvalues,
    double logdet_iab,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    double log_l_min, double step,
    int n_grid, int n_refine,
    int df, double reml_const,
    double * restrict hi_eval, double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
)
{
    const double phi = 0.6180339887498949;  /* golden ratio - 1 */

    /* Stage 1: coarse grid search using cached hi_eval and logdet_h.
     * Degenerate grid points return NaN from reml_finish (P_yy < 0);
     * map NaN → REML_SENTINEL so the > comparison skips them. */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = reml_logl_ncvt1_cached(
            uab,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g],
            logdet_iab,
            n_samples, df, reml_const
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    /* Every grid point produced NaN — fully degenerate SNP.
     * Without this, refinement proceeds on a meaningless bracket from
     * best_idx=0 and can produce finite but nonsensical results. */
    if (best_logl == REML_SENTINEL) {
        *logl_out    = (double)NAN;
        *beta_out    = (double)NAN;
        *se_out      = (double)NAN;
        *f_stat_out  = (double)NAN;
        *is_valid_out = 0;
        return lambda_grid[0];
    }

    /* Bracket around best grid point */
    int idx_low = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement (SNP-specific lambdas) */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                 n_samples, exp(c), reml_const, hi_eval);
    double fd = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                 n_samples, exp(d), reml_const, hi_eval);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            /* Maximum is in [a, d] — keep left */
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                  n_samples, exp(c), reml_const, hi_eval);
        } else {
            /* Maximum is in [c, b] — keep right */
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                  n_samples, exp(d), reml_const, hi_eval);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);
    *logl_out = reml_logl_ncvt1(uab, eigenvalues, logdet_iab,
                                 n_samples, lambda_opt, reml_const, hi_eval);

    /* hi_eval is now populated with 1/(lambda_opt*eval+1) — reuse for Wald stats
     * without another n_samples pass to recompute hi_eval. */
    {
        double pab[3][6];
        calc_pab_ncvt1(uab, hi_eval, n_samples, pab);
        *is_valid_out = wald_from_pab(pab, df, beta_out, se_out, f_stat_out);
    }

    return lambda_opt;
}

/* =========================================================================
 * SPLIT-Uab functions (SoA layout)
 *
 * These variants operate on separated varying/invariant Uab columns to
 * halve per-SNP DRAM traffic. The invariant columns (ww, wy, yy) are
 * identical across all SNPs and fit in L2 cache after the first SNP.
 *
 * SoA (Structure-of-Arrays) layout for SIMD:
 *   uab_var: (n_snps, 3, n_samples) — columns [wx, xx, xy] contiguous
 *   uab_inv: (3, n_samples)         — columns [ww, wy, yy] contiguous
 *
 * Each column is stride-1, enabling contiguous SIMD loads (vmovupd)
 * instead of stride-3 gather instructions (vgatherdpd).
 * ========================================================================= */

/* Pre-computed invariant dot products for one coarse grid point.
 * Memory: n_grid * sizeof(grid_invariant_t) ~ 50 * 48 = 2.4 KB (fits L1). */
typedef struct {
    double s_ww;       /* sum of hi * ww */
    double s_wy;       /* sum of hi * wy */
    double s_yy;       /* sum of hi * yy */
    double log_s_ww;   /* log(s_ww) if > 0, else 0 */
    double inv_s_ww;   /* 1/s_ww if != 0, else 0 */
    double pab1_5;     /* s_yy - s_wy^2/s_ww — completely SNP-invariant */
} grid_invariant_t;

/* -------------------------------------------------------------------------
 * calc_pab_ncvt1_split
 *
 * Compute Pab from separated varying + invariant dot product sums.
 * The caller provides the 6 pre-accumulated sums (3 varying + 3 invariant).
 * ------------------------------------------------------------------------- */
static void calc_pab_ncvt1_split(
    double s_ww, double s_wx, double s_wy,
    double s_xx, double s_xy, double s_yy,
    double pab[3][6]
)
{
    pab[0][0] = s_ww;
    pab[0][1] = s_wx;
    pab[0][2] = s_wy;
    pab[0][3] = s_xx;
    pab[0][4] = s_xy;
    pab[0][5] = s_yy;

    /* Row 1: project out W */
    double inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
    pab[1][3] = s_xx - s_wx * s_wx * inv_ww;
    pab[1][4] = s_xy - s_wx * s_wy * inv_ww;
    pab[1][5] = s_yy - s_wy * s_wy * inv_ww;

    /* Row 2: project out X */
    double ps_xx = pab[1][3];
    double inv_xx = (ps_xx != 0.0) ? 1.0 / ps_xx : 0.0;
    pab[2][5] = pab[1][5] - pab[1][4] * pab[1][4] * inv_xx;
}

/* -------------------------------------------------------------------------
 * reml_logl_ncvt1_cached_split
 *
 * Coarse grid REML log-likelihood using:
 *   - Precomputed hi_eval_grid and logdet_h_grid (per grid point)
 *   - Precomputed grid_invariant_t (invariant dot products per grid point)
 *   - Only 3 varying reductions from DRAM per SNP per grid eval
 *
 * This is the primary performance win: DRAM reads drop from 6 to 3
 * doubles per sample. Invariant sums come from L1-resident struct.
 *
 * SoA layout: var_wx/xx/xy are contiguous (stride-1) for SIMD.
 * ------------------------------------------------------------------------- */
static double reml_logl_ncvt1_cached_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict cached_hi_eval,
    double cached_logdet_h,
    double logdet_iab,
    const grid_invariant_t *ginv,
    int n_samples,
    int df,
    double reml_const
)
{
    /* Only 3 varying reductions — invariant sums precomputed per grid point */
    double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
    #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
    for (int i = 0; i < n_samples; i++) {
        double h = cached_hi_eval[i];
        s_wx += h * var_wx[i];
        s_xx += h * var_xx[i];
        s_xy += h * var_xy[i];
    }

    /* Combine with precomputed invariant sums */
    double s_ww = ginv->s_ww;
    double inv_ww = ginv->inv_s_ww;
    double s_wy = ginv->s_wy;

    /* Pab row 1 */
    double p1_xx = s_xx - s_wx * s_wx * inv_ww;
    double p1_xy = s_xy - s_wx * s_wy * inv_ww;
    /* p1_yy = ginv->pab1_5 is completely invariant */
    double p1_yy = ginv->pab1_5;

    /* Pab row 2 */
    double inv_xx = (p1_xx != 0.0) ? 1.0 / p1_xx : 0.0;
    double P_yy = p1_yy - p1_xy * p1_xy * inv_xx;

    /* logdet_hiw */
    double logdet_pab = ginv->log_s_ww;
    if (p1_xx > 0.0) logdet_pab += log(p1_xx);
    double logdet_hiw = logdet_pab - logdet_iab;

    /* P_yy guard */
    if (P_yy < 0.0) {
        P_yy = (double)NAN;
    } else if (P_yy < P_YY_MIN) {
        P_yy = P_YY_MIN;
    }

    return reml_const - 0.5 * cached_logdet_h - 0.5 * logdet_hiw
           - 0.5 * df * log(P_yy);
}

/* -------------------------------------------------------------------------
 * reml_logl_ncvt1_split
 *
 * Refinement path: fused hi_eval computation + all 6 dot products in
 * a single pass over n_samples. Eliminates the separate calc_pab call.
 *
 * Used during golden section where lambda is SNP-specific.
 *
 * SoA layout: varying and invariant columns are contiguous (stride-1),
 * enabling SIMD vectorized loads instead of stride-3 gathers.
 * ------------------------------------------------------------------------- */
static double reml_logl_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    double logdet_iab,
    int n_samples,
    double lambda,
    double reml_const
)
{
    int df = n_samples - 2;

    /* Fused: compute hi_eval + logdet_h + all 6 dot products in single pass.
     * v = lambda * eval[i] + 1.0 is always > 1.0 (eigenvalues >= 0, lambda > 0),
     * so fabs() is unnecessary and would block SIMD vectorization of log().
     * SoA layout gives stride-1 access for all 6 columns — enables contiguous
     * SIMD loads (vmovupd) instead of stride-3 gather instructions. */
    double logdet_h = 0.0;
    double s_ww = 0.0, s_wx = 0.0, s_wy = 0.0;
    double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;

    #pragma omp simd reduction(+:logdet_h,s_ww,s_wx,s_wy,s_xx,s_xy,s_yy)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        logdet_h += log(v);

        /* Varying (per-SNP, from DRAM) — stride-1 */
        s_wx += h * var_wx[i];
        s_xx += h * var_xx[i];
        s_xy += h * var_xy[i];

        /* Invariant (shared, from L2 cache) — stride-1 */
        s_ww += h * inv_ww[i];
        s_wy += h * inv_wy[i];
        s_yy += h * inv_yy[i];
    }

    /* Pab from sums */
    double pab[3][6];
    calc_pab_ncvt1_split(s_ww, s_wx, s_wy, s_xx, s_xy, s_yy, pab);

    return reml_finish(pab, logdet_h, logdet_iab, df, reml_const);
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_ncvt1_split
 *
 * Grid search + golden section refinement using split Uab arrays.
 * Coarse search uses cached_split (3 varying reductions + precomputed
 * invariants). Refinement uses fused reml_logl_ncvt1_split.
 *
 * SoA layout: var_wx/xx/xy and inv_ww/wy/yy are contiguous (stride-1).
 *
 * The final evaluation fuses REML logl + Wald stats in a single pass,
 * eliminating a redundant n_samples traversal per SNP.
 * ------------------------------------------------------------------------- */
static double golden_section_lambda_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    double logdet_iab,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const grid_invariant_t *grid_inv,
    double log_l_min, double step,
    int n_grid, int n_refine,
    int df, double reml_const,
    double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
)
{
    const double phi = 0.6180339887498949;

    /* Stage 1: coarse grid search using cached split.
     * Degenerate grid points return NaN from reml_logl_ncvt1_cached_split
     * (P_yy < 0); map NaN → REML_SENTINEL so > comparison skips them. */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = reml_logl_ncvt1_cached_split(
            var_wx, var_xx, var_xy,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g],
            logdet_iab,
            &grid_inv[g],
            n_samples, df, reml_const
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    /* Every grid point produced NaN — fully degenerate SNP. */
    if (best_logl == REML_SENTINEL) {
        *logl_out    = (double)NAN;
        *beta_out    = (double)NAN;
        *se_out      = (double)NAN;
        *f_stat_out  = (double)NAN;
        *is_valid_out = 0;
        return lambda_grid[0];
    }

    /* Bracket around best grid point */
    int idx_low = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement (fused single-pass) */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       logdet_iab, n_samples, exp(c),
                                       reml_const);
    double fd = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       logdet_iab, n_samples, exp(d),
                                       reml_const);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                        inv_ww, inv_wy, inv_yy, eigenvalues,
                                        logdet_iab, n_samples, exp(c),
                                        reml_const);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = reml_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                        inv_ww, inv_wy, inv_yy, eigenvalues,
                                        logdet_iab, n_samples, exp(d),
                                        reml_const);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);

    /* Final evaluation: fuse REML logl + Wald stats in single n_samples pass.
     * This eliminates the separate calc_rl_wald_ncvt1_split call that would
     * redundantly recompute the identical Pab sums. */
    {
        double logdet_h = 0.0;
        double s_ww = 0.0, s_wx = 0.0, s_wy = 0.0;
        double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;

        #pragma omp simd reduction(+:logdet_h,s_ww,s_wx,s_wy,s_xx,s_xy,s_yy)
        for (int i = 0; i < n_samples; i++) {
            double v = lambda_opt * eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            logdet_h += log(v);

            s_wx += h * var_wx[i];
            s_xx += h * var_xx[i];
            s_xy += h * var_xy[i];

            s_ww += h * inv_ww[i];
            s_wy += h * inv_wy[i];
            s_yy += h * inv_yy[i];
        }

        double pab[3][6];
        calc_pab_ncvt1_split(s_ww, s_wx, s_wy, s_xx, s_xy, s_yy, pab);

        *logl_out = reml_finish(pab, logdet_h, logdet_iab, df, reml_const);
        *is_valid_out = wald_from_pab(pab, df, beta_out, se_out, f_stat_out);
    }

    return lambda_opt;
}

/* =========================================================================
 * Workspace API — persistent cross-chunk state for split-Uab pipeline
 *
 * Eliminates per-chunk malloc + grid precomputation overhead:
 *   - lambda_grid, hi_eval_grid, logdet_h_grid built once per run
 *   - grid_inv (invariant dot products) built once per run
 *   - iab_s_ww / iab_log_ww precomputed from invariant column sums
 *   - Python arrays kept alive via Py_INCREF until workspace freed
 * ========================================================================= */

typedef struct {
    int n_samples;
    int n_grid;
    int n_refine;
    int df;
    double l_min, l_max, log_l_min, step;
    double reml_const;
    double beta_a, beta_b, lbeta_ab;
    /* Per-grid precomputed data (owned by workspace) */
    double *lambda_grid;      /* (n_grid,) */
    double *hi_eval_grid;     /* (n_grid * n_samples) */
    double *logdet_h_grid;    /* (n_grid,) */
    grid_invariant_t *grid_inv;  /* (n_grid,) */
    /* Invariant Iab scalars (lambda-independent) */
    double iab_s_ww;    /* sum(inv_ww) */
    double iab_inv_ww;  /* 1/iab_s_ww (or 0) */
    double iab_log_ww;  /* log(iab_s_ww) (or 0) */
    /* Borrowed pointers — kept alive via Py_INCREF */
    const double *eigenvalues;
    const double *inv_ww;   /* uab_invariant_soa row 0 */
    const double *inv_wy;   /* uab_invariant_soa row 1 */
    const double *inv_yy;   /* uab_invariant_soa row 2 */
    PyObject *eigenvalues_ref;  /* keeps eigenvalues array alive */
    PyObject *uab_inv_ref;      /* keeps uab_invariant_soa array alive */
    /* Mode-4 fused fields (only populated when mode=4) */
    int mode;                   /* 0=Wald-only (default from calloc zero-init), 4=fused mode-4 */
    double *hi_eval_null;       /* (n_samples,) null-model Hi_eval, owned */
    double logl_H0;             /* null MLE log-likelihood */
    double mle_const;           /* 0.5 * n * (log(n) - log(2*pi) - 1) */
    double null_s_ww;           /* invariant dot product under null Hi_eval */
    double null_s_wy;
    double null_s_yy;
    double null_inv_ww;         /* 1/null_s_ww */
} lmm_workspace_t;

/* PyCapsule destructor: free owned allocations, release Python array refs. */
static void lmm_workspace_destructor(PyObject *cap)
{
    lmm_workspace_t *ws =
        (lmm_workspace_t *)PyCapsule_GetPointer(cap, "lmm_workspace");
    if (!ws) return;
    free(ws->lambda_grid);
    free(ws->hi_eval_grid);
    free(ws->logdet_h_grid);
    free(ws->grid_inv);
    free(ws->hi_eval_null);
    Py_XDECREF(ws->eigenvalues_ref);
    Py_XDECREF(ws->uab_inv_ref);
    free(ws);
}

/* -------------------------------------------------------------------------
 * create_workspace_split_c
 *
 * Python signature:
 *   create_workspace_split_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (3, n_samples) float64 — SoA [ww, wy, yy]
 *       n_samples,        # int
 *       l_min,            # float
 *       l_max,            # float
 *       n_grid,           # int
 *       n_refine,         # int
 *       n_threads,        # int
 *   ) -> PyCapsule wrapping lmm_workspace_t
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_split_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "n_samples",
        "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOiddiii", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    lmm_workspace_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_invariant must be shape (3, n_samples)");
        goto err_input;
    }

    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }

    /* Fill scalar fields */
    ws->n_samples = n_samples;
    ws->n_grid    = n_grid;
    ws->n_refine  = n_refine;
    ws->l_min     = l_min;
    ws->l_max     = l_max;
    ws->df        = n_samples - 2;

    ws->beta_a   = (double)ws->df / 2.0;
    ws->beta_b   = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);

    ws->log_l_min   = log(l_min);
    double log_l_max = log(l_max);
    ws->step        = (log_l_max - ws->log_l_min) / (double)(n_grid - 1);
    ws->reml_const  = 0.5 * ws->df * (log((double)ws->df)
                       - log(2.0 * M_PI) - 1.0);

    /* Borrow pointers — arrays kept alive via Py_INCREF */
    Py_INCREF(eigenvalues_arr);
    Py_INCREF(uab_inv_arr);
    ws->eigenvalues_ref = (PyObject *)eigenvalues_arr;
    ws->uab_inv_ref     = (PyObject *)uab_inv_arr;

    ws->eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    ws->inv_ww = (const double *)PyArray_DATA(uab_inv_arr);
    ws->inv_wy = ws->inv_ww + (size_t)n_samples;
    ws->inv_yy = ws->inv_ww + (size_t)2 * n_samples;

    /* Compute invariant Iab scalar: sum(inv_ww) */
    {
        double s_ww = 0.0;
        for (int i = 0; i < n_samples; i++) s_ww += ws->inv_ww[i];
        ws->iab_s_ww   = s_ww;
        ws->iab_inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
        ws->iab_log_ww = (s_ww > 0.0)  ? log(s_ww)  : 0.0;
    }

    /* Allocate grid arrays */
    ws->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->grid_inv) {
        PyErr_NoMemory();
        goto err_ws;
    }

    /* Build lambda grid + invariant dot products (same logic as batch_split_c) */
    for (int g = 0; g < n_grid; g++) {
        ws->lambda_grid[g] = exp(ws->log_l_min + g * ws->step);
    }
    for (int g = 0; g < n_grid; g++) {
        double lam    = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double sw = 0.0, swy = 0.0, sy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
            sw  += h * ws->inv_ww[i];
            swy += h * ws->inv_wy[i];
            sy  += h * ws->inv_yy[i];
        }
        ws->logdet_h_grid[g] = logdet;

        ws->grid_inv[g].s_ww    = sw;
        ws->grid_inv[g].s_wy    = swy;
        ws->grid_inv[g].s_yy    = sy;
        ws->grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
        ws->grid_inv[g].inv_s_ww = (sw != 0.0) ? 1.0 / sw : 0.0;
        ws->grid_inv[g].pab1_5   = sy - swy * swy * ws->grid_inv[g].inv_s_ww;
    }

    /* Wrap in PyCapsule; destructor frees ws on GC */
    capsule = PyCapsule_New(ws, "lmm_workspace", lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    /* Release local refs — capsule now owns ws->*_ref via destructor */
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    return capsule;

err_ws:
    /* Destructor not yet registered — free manually */
    if (ws) {
        Py_XDECREF(ws->eigenvalues_ref);
        Py_XDECREF(ws->uab_inv_ref);
        free(ws->lambda_grid);
        free(ws->hi_eval_grid);
        free(ws->logdet_h_grid);
        free(ws->grid_inv);
        free(ws);
    }
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * create_workspace_mode4_split_c
 *
 * Create a mode-4 workspace: extends the standard split workspace with
 * null-model Hi_eval (for Score), MLE constant, and null logl (for LRT).
 *
 * Python signature:
 *   create_workspace_mode4_split_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (3, n_samples) float64 — SoA [ww, wy, yy]
 *       n_samples,        # int
 *       l_min,            # float
 *       l_max,            # float
 *       n_grid,           # int
 *       n_refine,         # int
 *       n_threads,        # int
 *       hi_eval_null,     # (n_samples,) float64 — null-model Hi_eval
 *       logl_H0,          # float — null MLE log-likelihood
 *   ) -> PyCapsule wrapping lmm_workspace_t (mode=4)
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_mode4_split_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "n_samples",
        "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "hi_eval_null", "logl_H0",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj, *hi_eval_null_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOiddiiiOd", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &hi_eval_null_obj, &logl_H0)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    PyArrayObject *eigenvalues_arr = NULL, *uab_inv_arr = NULL;
    PyArrayObject *hi_eval_null_arr = NULL;
    lmm_workspace_t *ws = NULL;
    PyObject *capsule = NULL;

    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_invariant must be shape (3, n_samples)");
        goto err_input;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "hi_eval_null must be shape (n_samples,)");
        goto err_input;
    }

    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0)
        goto err_input;

    /* Validate Hi_eval_null for NaN/Inf */
    {
        const double *hi_null = (const double *)PyArray_DATA(hi_eval_null_arr);
        for (int i = 0; i < n_samples; i++) {
            if (!isfinite(hi_null[i])) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%g", hi_null[i]);
                PyErr_Format(PyExc_ValueError,
                    "hi_eval_null[%d] = %s is not finite. "
                    "Null model optimization may have failed.", i, buf);
                goto err_input;
            }
        }
    }

    ws = (lmm_workspace_t *)calloc(1, sizeof(lmm_workspace_t));
    if (!ws) { PyErr_NoMemory(); goto err_input; }

    /* Fill scalar fields (same as create_workspace_split_c) */
    ws->n_samples = n_samples;
    ws->n_grid    = n_grid;
    ws->n_refine  = n_refine;
    ws->l_min     = l_min;
    ws->l_max     = l_max;
    ws->df        = n_samples - 2;

    ws->beta_a   = (double)ws->df / 2.0;
    ws->beta_b   = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);

    ws->log_l_min   = log(l_min);
    double log_l_max = log(l_max);
    ws->step        = (log_l_max - ws->log_l_min) / (double)(n_grid - 1);
    ws->reml_const  = 0.5 * ws->df * (log((double)ws->df)
                       - log(2.0 * M_PI) - 1.0);

    /* Borrow pointers — arrays kept alive via Py_INCREF */
    Py_INCREF(eigenvalues_arr);
    Py_INCREF(uab_inv_arr);
    ws->eigenvalues_ref = (PyObject *)eigenvalues_arr;
    ws->uab_inv_ref     = (PyObject *)uab_inv_arr;

    ws->eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    ws->inv_ww = (const double *)PyArray_DATA(uab_inv_arr);
    ws->inv_wy = ws->inv_ww + (size_t)n_samples;
    ws->inv_yy = ws->inv_ww + (size_t)2 * n_samples;

    /* Compute invariant Iab scalar: sum(inv_ww) */
    {
        double s_ww = 0.0;
        for (int i = 0; i < n_samples; i++) s_ww += ws->inv_ww[i];
        ws->iab_s_ww   = s_ww;
        ws->iab_inv_ww = (s_ww != 0.0) ? 1.0 / s_ww : 0.0;
        ws->iab_log_ww = (s_ww > 0.0)  ? log(s_ww)  : 0.0;
    }

    /* Allocate grid arrays */
    ws->lambda_grid   = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid  = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->grid_inv      = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->grid_inv) {
        PyErr_NoMemory();
        goto err_ws;
    }

    /* Build lambda grid + invariant dot products */
    for (int g = 0; g < n_grid; g++) {
        ws->lambda_grid[g] = exp(ws->log_l_min + g * ws->step);
    }
    for (int g = 0; g < n_grid; g++) {
        double lam    = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double sw = 0.0, swy = 0.0, sy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
            sw  += h * ws->inv_ww[i];
            swy += h * ws->inv_wy[i];
            sy  += h * ws->inv_yy[i];
        }
        ws->logdet_h_grid[g] = logdet;

        ws->grid_inv[g].s_ww    = sw;
        ws->grid_inv[g].s_wy    = swy;
        ws->grid_inv[g].s_yy    = sy;
        ws->grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
        ws->grid_inv[g].inv_s_ww = (sw != 0.0) ? 1.0 / sw : 0.0;
        ws->grid_inv[g].pab1_5   = sy - swy * swy * ws->grid_inv[g].inv_s_ww;
    }

    /* --- Mode-4 specific fields --- */
    ws->mode = 4;
    ws->logl_H0 = logl_H0;
    ws->mle_const = 0.5 * (double)n_samples
                    * (log((double)n_samples) - log(2.0 * M_PI) - 1.0);

    /* Copy hi_eval_null into workspace-owned buffer */
    ws->hi_eval_null = alloc_aligned_doubles((size_t)n_samples);
    if (!ws->hi_eval_null) {
        PyErr_NoMemory();
        goto err_ws;
    }
    {
        const double *src = (const double *)PyArray_DATA(hi_eval_null_arr);
        memcpy(ws->hi_eval_null, src, (size_t)n_samples * sizeof(double));
    }

    /* Precompute null-model invariant dot products under hi_eval_null */
    {
        double ns_ww = 0.0, ns_wy = 0.0, ns_yy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double h = ws->hi_eval_null[i];
            ns_ww += h * ws->inv_ww[i];
            ns_wy += h * ws->inv_wy[i];
            ns_yy += h * ws->inv_yy[i];
        }
        ws->null_s_ww   = ns_ww;
        ws->null_s_wy   = ns_wy;
        ws->null_s_yy   = ns_yy;
        ws->null_inv_ww  = (ns_ww != 0.0) ? 1.0 / ns_ww : 0.0;
    }

    /* Wrap in PyCapsule */
    capsule = PyCapsule_New(ws, "lmm_workspace", lmm_workspace_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(hi_eval_null_arr);
    return capsule;

err_ws:
    if (ws) {
        Py_XDECREF(ws->eigenvalues_ref);
        Py_XDECREF(ws->uab_inv_ref);
        free(ws->lambda_grid);
        free(ws->hi_eval_grid);
        free(ws->logdet_h_grid);
        free(ws->grid_inv);
        free(ws->hi_eval_null);
        free(ws);
    }
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(hi_eval_null_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_split_c
 *
 * Per-chunk compute using a pre-built workspace. No allocation, no grid
 * rebuild, no Python Iab — logdet_iab computed entirely in C.
 *
 * Python signature:
 *   compute_lmm_chunk_split_c(
 *       workspace,     # PyCapsule from create_workspace_split_c
 *       uab_varying,   # (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]
 *       n_threads,     # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}  each (n_snps,) float64
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_chunk_split_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "uab_varying", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *uab_var_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &uab_var_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_t *ws = (lmm_workspace_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace");
    if (!ws) return NULL;  /* PyCapsule_GetPointer sets TypeError */

    PyArrayObject *uab_var_arr = NULL;
    output_arrays_t out = {0};
    PyObject *result = NULL;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) return NULL;

    int n_samples = ws->n_samples;

    /* Validate shape */
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != 3 ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying must be shape (n_snps, 3, %d)", n_samples);
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);
    const double *inv_ww = ws->inv_ww;
    const double *inv_wy = ws->inv_wy;
    const double *inv_yy = ws->inv_yy;

    double *lambdas = (double *)PyArray_DATA(out.lambdas);
    double *logls   = (double *)PyArray_DATA(out.logls);
    double *betas   = (double *)PyArray_DATA(out.betas);
    double *ses     = (double *)PyArray_DATA(out.ses);
    double *pwalds  = (double *)PyArray_DATA(out.pwalds);

    int n_grid    = ws->n_grid;
    int n_refine  = ws->n_refine;
    int df        = ws->df;
    double reml_const = ws->reml_const;

    /* Clamp n_threads to n_snps */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    Py_BEGIN_ALLOW_THREADS

    /* Static schedule: SNP cost is uniform (same n_grid, n_refine, n_samples).
     * No atomic work-stealing overhead vs dynamic scheduling. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        const double *snp_base = uab_var_data + (size_t)snp * 3 * n_samples;
        const double *vwx = snp_base;
        const double *vxx = snp_base + (size_t)n_samples;
        const double *vxy = snp_base + (size_t)2 * n_samples;

        /* Compute logdet_iab internally from raw Uab column sums.
         *
         * logdet_iab = log(iab[0][0]) + log(iab[1][3]) when positive.
         *
         *   iab[0][0] = sum(inv_ww) = ws->iab_s_ww  (invariant, precomputed)
         *   iab[1][3] = sum(var_xx) - (sum(var_wx))^2 / sum(inv_ww)
         *
         * The varying sums require O(n_samples) reductions — ~100 cycles for
         * 1400 samples, negligible vs the existing 50-grid-point REML loop.
         */
        double iab_s_wx = 0.0, iab_s_xx = 0.0;
        #pragma omp simd reduction(+:iab_s_wx,iab_s_xx)
        for (int i = 0; i < n_samples; i++) {
            iab_s_wx += vwx[i];
            iab_s_xx += vxx[i];
        }

        double iab_p1_xx = iab_s_xx - iab_s_wx * iab_s_wx * ws->iab_inv_ww;
        double logdet_iab = ws->iab_log_ww
                            + ((iab_p1_xx > 0.0) ? log(iab_p1_xx) : 0.0);

        /* Golden section lambda optimization (reuses workspace grids) */
        double logl_opt, beta, se, f_stat;
        int is_valid;
        double lambda_opt = golden_section_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, logdet_iab,
            n_samples, ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->grid_inv, ws->log_l_min, ws->step, n_grid, n_refine,
            df, reml_const, &logl_opt, &beta, &se, &f_stat, &is_valid
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;
        betas[snp]   = beta;
        ses[snp]     = se;

        pwalds[snp] = f_to_pvalue(
            f_stat, df, is_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0)
        goto err_output;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(uab_var_arr);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    Py_XDECREF(uab_var_arr);
    return NULL;
}

/* =========================================================================
 * GENERAL n_cvt support — table-driven Pab recursion for arbitrary covariates
 *
 * Adds new workspace type (lmm_workspace_general_t) and entry points
 * (create_workspace_general_c, compute_lmm_chunk_general_c) that accept
 * n_cvt as a parameter. The existing n_cvt=1 code path is unchanged.
 *
 * Key design: Python builds the recursion table (via build_pab_table_for_c)
 * and passes flat int32 arrays. C code just walks the table — no index
 * computation in C.
 *
 * Memory: Per-SNP Pab uses stack buffers (MAX_N_CVT=20 -> MAX_PAB_SIZE=5566
 * doubles = ~44KB, well within 1MB+ thread stacks).
 * ========================================================================= */

#define MAX_N_CVT   20
#define MAX_N_INDEX ((MAX_N_CVT + 3) * (MAX_N_CVT + 2) / 2)  /* 253 for n_cvt=20 */
#define MAX_N_ROWS  (MAX_N_CVT + 2)                           /* 22 */
#define MAX_PAB_SIZE (MAX_N_ROWS * MAX_N_INDEX)                /* 5566 */

typedef struct {
    int index_ab, index_aw, index_bw, index_ww;
} pab_entry_t;

typedef struct {
    int n_cvt, n_index, n_rows, n_inv, n_var;
    int idx_xx, idx_xy, idx_yy;
    int df;  /* n_samples - n_cvt - 1 */
    int *invariant_indices;  /* (n_inv,) */
    int *varying_indices;    /* (n_var,) */
    int *logdet_diag_rows;   /* (n_cvt+1,) */
    int *logdet_diag_cols;   /* (n_cvt+1,) */
    int *level_offsets;      /* (n_rows,) — offset into entries per level */
    int *level_counts;       /* (n_rows,) — count per level */
    pab_entry_t *entries;    /* all entries concatenated */
    int n_entries;
} pab_table_t;

/* -------------------------------------------------------------------------
 * calc_pab_general — Table-driven Pab recursion for arbitrary n_cvt.
 *
 * Row 0 from row0 array (dot product sums), rows 1..n_rows-1 from entries.
 * Output in pab[n_rows * n_index], row-major.
 * ------------------------------------------------------------------------- */
static void calc_pab_general(
    const double *row0,
    const pab_table_t *t,
    double *pab
)
{
    int ni = t->n_index;
    /* Copy row 0 */
    for (int i = 0; i < ni; i++) pab[i] = row0[i];

    /* Recursive projection: rows 1..n_rows-1 */
    for (int p = 1; p < t->n_rows; p++) {
        int offset = t->level_offsets[p];
        int count  = t->level_counts[p];
        for (int e = 0; e < count; e++) {
            const pab_entry_t *re = &t->entries[offset + e];
            double ps_ww = pab[(p - 1) * ni + re->index_ww];
            /* Match n_cvt=1 paths: zero projection when divisor is zero,
             * so Px_YY < 0 guard in wald_from_pab catches degeneracy. */
            double inv_ww = (ps_ww != 0.0) ? 1.0 / ps_ww : 0.0;
            pab[p * ni + re->index_ab] =
                pab[(p - 1) * ni + re->index_ab]
                - pab[(p - 1) * ni + re->index_aw]
                * pab[(p - 1) * ni + re->index_bw]
                * inv_ww;
        }
    }
}

/* -------------------------------------------------------------------------
 * reml_finish_general — REML tail for general n_cvt.
 *
 * logdet_pab from logdet_diag entries, P_yy guard, return full REML formula
 * including logdet_h.
 * ------------------------------------------------------------------------- */
static double reml_finish_general(
    const double *pab,
    const pab_table_t *t,
    double logdet_h,
    double logdet_iab,
    double reml_const
)
{
    int ni = t->n_index;
    int df = t->df;

    /* logdet_pab from diagonal entries.  A non-positive diagonal means the
     * projected matrix is not positive-definite — return NaN so the REML
     * sentinel mechanism correctly flags this as degenerate. */
    double logdet_pab = 0.0;
    for (int d = 0; d < t->n_cvt + 1; d++) {
        double val = pab[t->logdet_diag_rows[d] * ni + t->logdet_diag_cols[d]];
        if (val <= 0.0) return (double)NAN;
        logdet_pab += log(val);
    }
    double logdet_hiw = logdet_pab - logdet_iab;

    /* P_yy guard */
    int nc_total = t->n_cvt + 1;
    double P_yy = pab[nc_total * ni + t->idx_yy];
    if (P_yy < 0.0) {
        P_yy = (double)NAN;
    } else if (P_yy < P_YY_MIN) {
        P_yy = P_YY_MIN;
    }

    return reml_const - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * log(P_yy);
}

/* -------------------------------------------------------------------------
 * reml_logl_general_cached — REML using cached grid hi_eval + invariant sums.
 *
 * For cached grid points: invariant sums already computed, just compute
 * varying dot products, reconstruct row0, calc_pab, reml_finish.
 * ------------------------------------------------------------------------- */
static double reml_logl_general_cached(
    const double *inv_sums_cached,
    const double *uab_var,
    const double *hi_eval,
    int n_samples,
    double logdet_h,
    double logdet_iab,
    double reml_const,
    const pab_table_t *t
)
{
    int ni = t->n_index;
    int n_var = t->n_var;

    /* Compute varying dot products */
    double var_sums[MAX_N_INDEX];
    for (int c = 0; c < n_var; c++) var_sums[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double h = hi_eval[i];
        for (int c = 0; c < n_var; c++)
            var_sums[c] += h * uab_var[c * n_samples + i];
    }

    /* Reconstruct row 0 */
    double row0[MAX_N_INDEX];
    for (int i = 0; i < ni; i++) row0[i] = 0.0;
    for (int c = 0; c < t->n_inv; c++)
        row0[t->invariant_indices[c]] = inv_sums_cached[c];
    for (int c = 0; c < n_var; c++)
        row0[t->varying_indices[c]] = var_sums[c];

    /* Full Pab via recursion */
    double pab[MAX_PAB_SIZE];
    calc_pab_general(row0, t, pab);

    return reml_finish_general(pab, t, logdet_h, logdet_iab, reml_const);
}

/* -------------------------------------------------------------------------
 * reml_logl_general_fresh — Full REML evaluation for a specific lambda.
 *
 * Computes hi_eval + logdet_h + all dot products in single n_samples pass
 * (fused loop), then calc_pab + reml_finish.
 * Used during golden section refinement where lambda is SNP-specific.
 * ------------------------------------------------------------------------- */
static double reml_logl_general_fresh(
    const double *uab_inv,
    const double *uab_var,
    const double *eigenvalues,
    int n_samples,
    double lambda,
    double logdet_iab,
    double reml_const,
    const pab_table_t *t
)
{
    int ni = t->n_index;
    int n_inv = t->n_inv;
    int n_var = t->n_var;

    double logdet_h = 0.0;
    double inv_sums[MAX_N_INDEX];
    double var_sums[MAX_N_INDEX];
    for (int c = 0; c < n_inv; c++) inv_sums[c] = 0.0;
    for (int c = 0; c < n_var; c++) var_sums[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        logdet_h += log(v);
        for (int c = 0; c < n_inv; c++)
            inv_sums[c] += h * uab_inv[c * n_samples + i];
        for (int c = 0; c < n_var; c++)
            var_sums[c] += h * uab_var[c * n_samples + i];
    }

    /* Reconstruct row 0 */
    double row0[MAX_N_INDEX];
    for (int i = 0; i < ni; i++) row0[i] = 0.0;
    for (int c = 0; c < n_inv; c++)
        row0[t->invariant_indices[c]] = inv_sums[c];
    for (int c = 0; c < n_var; c++)
        row0[t->varying_indices[c]] = var_sums[c];

    /* Full Pab via recursion */
    double pab[MAX_PAB_SIZE];
    calc_pab_general(row0, t, pab);

    return reml_finish_general(pab, t, logdet_h, logdet_iab, reml_const);
}

/* -------------------------------------------------------------------------
 * wald_from_pab_general — Extract Wald stats from general-n_cvt Pab.
 *
 * P_XX = Pab[n_cvt, idx_xx], P_XY = Pab[n_cvt, idx_xy],
 * P_YY = Pab[n_cvt, idx_yy] (pre-genotype-projection),
 * Px_YY = Pab[n_cvt+1, idx_yy] (fully projected).
 * Same Wald formula as existing wald_from_pab.
 * Returns 1 if valid, 0 if degenerate.
 * ------------------------------------------------------------------------- */
static int wald_from_pab_general(
    const double *pab,
    const pab_table_t *t,
    double *beta_out, double *se_out, double *f_stat_out
)
{
    int ni = t->n_index;
    int df = t->df;
    int nc = t->n_cvt;

    double P_XX  = pab[nc * ni + t->idx_xx];
    double P_XY  = pab[nc * ni + t->idx_xy];
    double P_YY  = pab[nc * ni + t->idx_yy];
    double Px_YY = pab[(nc + 1) * ni + t->idx_yy];

    if (Px_YY < 0.0) {
        *beta_out = *se_out = *f_stat_out = (double)NAN;
        return 0;
    }
    if (Px_YY < P_YY_MIN) Px_YY = P_YY_MIN;

    if (P_XX <= 0.0) {
        *beta_out = *se_out = *f_stat_out = (double)NAN;
        return 0;
    }

    double beta = P_XY / P_XX;
    double tau = (double)df / Px_YY;
    double variance_beta = 1.0 / (tau * P_XX);
    double variance_safe = (fabs(variance_beta) < 0.001)
                            ? fabs(variance_beta)
                            : variance_beta;
    double se = sqrt(variance_safe);
    double f_stat = (P_YY - Px_YY) * tau;

    *beta_out   = beta;
    *se_out     = se;
    *f_stat_out = f_stat;

    if (!isfinite(f_stat) || !isfinite(beta) || !isfinite(se))
        return 0;

    return 1;
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_general — Grid + golden section for general n_cvt.
 *
 * Mirrors golden_section_lambda_ncvt1_split() structure. Grid phase uses
 * precomputed hi_eval + invariant sums; refinement uses fresh evaluation.
 * At optimal lambda, computes full Pab and returns it + Wald stats.
 * ------------------------------------------------------------------------- */
static double golden_section_lambda_general(
    const double *uab_inv,
    const double *uab_var,
    const double *eigenvalues,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const double *inv_sums_grid,    /* (n_grid, n_inv) */
    double log_l_min, double step,
    int n_grid, int n_refine,
    double logdet_iab,
    double reml_const,
    const pab_table_t *t,
    double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
)
{
    const double phi = 0.6180339887498949;
    int n_inv = t->n_inv;

    /* Stage 1: coarse grid search using cached invariant sums */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = reml_logl_general_cached(
            inv_sums_grid + (size_t)g * n_inv,
            uab_var,
            hi_eval_grid + (size_t)g * n_samples,
            n_samples,
            logdet_h_grid[g],
            logdet_iab,
            reml_const,
            t
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    /* Fully degenerate SNP */
    if (best_logl == REML_SENTINEL) {
        *logl_out    = (double)NAN;
        *beta_out    = (double)NAN;
        *se_out      = (double)NAN;
        *f_stat_out  = (double)NAN;
        *is_valid_out = 0;
        return lambda_grid[0];
    }

    /* Bracket around best grid point */
    int idx_low  = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement (fresh evaluation) */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = reml_logl_general_fresh(
        uab_inv, uab_var, eigenvalues, n_samples, exp(c),
        logdet_iab, reml_const, t);
    double fd = reml_logl_general_fresh(
        uab_inv, uab_var, eigenvalues, n_samples, exp(d),
        logdet_iab, reml_const, t);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = reml_logl_general_fresh(
                uab_inv, uab_var, eigenvalues, n_samples, exp(c),
                logdet_iab, reml_const, t);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = reml_logl_general_fresh(
                uab_inv, uab_var, eigenvalues, n_samples, exp(d),
                logdet_iab, reml_const, t);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);

    /* Final: compute REML logl + Pab at optimal lambda for Wald extraction */
    {
        int ni = t->n_index;
        int n_var = t->n_var;

        double logdet_h = 0.0;
        double inv_sums_final[MAX_N_INDEX];
        double var_sums_final[MAX_N_INDEX];
        for (int cc = 0; cc < n_inv; cc++) inv_sums_final[cc] = 0.0;
        for (int cc = 0; cc < n_var; cc++) var_sums_final[cc] = 0.0;

        for (int i = 0; i < n_samples; i++) {
            double v = lambda_opt * eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            logdet_h += log(v);
            for (int cc = 0; cc < n_inv; cc++)
                inv_sums_final[cc] += h * uab_inv[cc * n_samples + i];
            for (int cc = 0; cc < n_var; cc++)
                var_sums_final[cc] += h * uab_var[cc * n_samples + i];
        }

        double row0[MAX_N_INDEX];
        for (int i = 0; i < ni; i++) row0[i] = 0.0;
        for (int cc = 0; cc < n_inv; cc++)
            row0[t->invariant_indices[cc]] = inv_sums_final[cc];
        for (int cc = 0; cc < n_var; cc++)
            row0[t->varying_indices[cc]] = var_sums_final[cc];

        double pab[MAX_PAB_SIZE];
        calc_pab_general(row0, t, pab);

        *logl_out = reml_finish_general(pab, t, logdet_h, logdet_iab, reml_const);
        *is_valid_out = wald_from_pab_general(
            pab, t, beta_out, se_out, f_stat_out);
    }

    return lambda_opt;
}

/* -------------------------------------------------------------------------
 * General workspace struct — persistent cross-chunk state for n_cvt >= 1
 * ------------------------------------------------------------------------- */
typedef struct {
    /* Grid precomputed */
    double *lambda_grid;    /* (n_grid,) */
    double *hi_eval_grid;   /* (n_grid * n_samples) */
    double *logdet_h_grid;  /* (n_grid,) */
    double *inv_sums_grid;  /* (n_grid * n_inv) — precomputed invariant dot products */
    /* Fixed params */
    double *eigenvalues;    /* (n_samples,) — owned copy */
    double reml_const;
    int n_samples, n_grid, n_refine;
    /* Table (owned copy of indices) */
    pab_table_t table;
    /* Iab: invariant identity sums (precomputed, reused per-SNP) */
    double *inv_identity_sums;  /* (n_inv,) — sum of each invariant column at identity */
    /* F-distribution */
    double lbeta_ab;
    double beta_a, beta_b;
    /* Invariant SoA (reference, not owned — Python holds the array) */
    const double *uab_inv;
    PyObject *uab_inv_ref;      /* keeps uab_invariant_soa array alive */
} lmm_workspace_general_t;

/* PyCapsule destructor for general workspace */
static void lmm_workspace_general_destructor(PyObject *cap)
{
    lmm_workspace_general_t *ws =
        (lmm_workspace_general_t *)PyCapsule_GetPointer(cap, "lmm_workspace_general");
    if (!ws) return;
    free(ws->lambda_grid);
    free(ws->hi_eval_grid);
    free(ws->logdet_h_grid);
    free(ws->inv_sums_grid);
    free(ws->eigenvalues);
    free(ws->inv_identity_sums);
    free(ws->table.invariant_indices);
    free(ws->table.varying_indices);
    free(ws->table.logdet_diag_rows);
    free(ws->table.logdet_diag_cols);
    free(ws->table.level_offsets);
    free(ws->table.level_counts);
    free(ws->table.entries);
    Py_XDECREF(ws->uab_inv_ref);
    free(ws);
}

/* Helper: parse int32 array from Python object, return malloced copy (caller frees) */
static int *parse_int32_array(PyObject *obj, int expected_len, const char *name)
{
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr) return NULL;
    if (PyArray_SIZE(arr) != expected_len) {
        PyErr_Format(PyExc_ValueError, "%s must have %d elements", name, expected_len);
        Py_DECREF(arr);
        return NULL;
    }
    int *copy = (int *)malloc((size_t)expected_len * sizeof(int));
    if (!copy) { Py_DECREF(arr); PyErr_NoMemory(); return NULL; }
    memcpy(copy, PyArray_DATA(arr), (size_t)expected_len * sizeof(int));
    Py_DECREF(arr);
    return copy;
}

/* -------------------------------------------------------------------------
 * create_workspace_general_c
 *
 * Python signature:
 *   create_workspace_general_c(
 *       eigenvalues,      # (n_samples,) float64
 *       uab_invariant,    # (n_inv, n_samples) float64 — SoA
 *       n_samples,        # int
 *       l_min, l_max,     # float
 *       n_grid, n_refine, n_threads,  # int
 *       n_cvt,            # int
 *       invariant_indices, varying_indices,    # (n_inv,) / (n_var,) int32
 *       logdet_diag_rows, logdet_diag_cols,    # (n_cvt+1,) int32
 *       level_offsets, level_counts,           # (n_rows,) int32
 *       entries,           # (n_entries * 4,) int32 stride-4 flat
 *       idx_xx, idx_xy, idx_yy,  # int
 *   ) -> PyCapsule wrapping lmm_workspace_general_t
 * ------------------------------------------------------------------------- */
static PyObject *create_workspace_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "n_samples",
        "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "n_cvt",
        "invariant_indices", "varying_indices",
        "logdet_diag_rows", "logdet_diag_cols",
        "level_offsets", "level_counts", "entries",
        "idx_xx", "idx_xy", "idx_yy",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_inv_obj;
    PyObject *inv_idx_obj, *var_idx_obj;
    PyObject *diag_rows_obj, *diag_cols_obj;
    PyObject *offsets_obj, *counts_obj, *entries_obj;
    int n_samples, n_grid, n_refine, n_threads, n_cvt;
    int idx_xx, idx_xy, idx_yy;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOiddiiiiOOOOOOOiii", (char **)kwlist,
            &eigenvalues_obj, &uab_inv_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads,
            &n_cvt,
            &inv_idx_obj, &var_idx_obj,
            &diag_rows_obj, &diag_cols_obj,
            &offsets_obj, &counts_obj, &entries_obj,
            &idx_xx, &idx_xy, &idx_yy)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError,
            "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return NULL;
    }

    int n_index = (n_cvt + 3) * (n_cvt + 2) / 2;
    int n_rows  = n_cvt + 2;

    /* Parse invariant_indices to determine n_inv */
    PyArrayObject *inv_idx_arr = (PyArrayObject *)PyArray_FROM_OTF(
        inv_idx_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!inv_idx_arr) return NULL;
    int n_inv = (int)PyArray_SIZE(inv_idx_arr);
    Py_DECREF(inv_idx_arr);

    PyArrayObject *var_idx_arr = (PyArrayObject *)PyArray_FROM_OTF(
        var_idx_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!var_idx_arr) return NULL;
    int n_var = (int)PyArray_SIZE(var_idx_arr);
    Py_DECREF(var_idx_arr);

    if (n_inv + n_var != n_index) {
        PyErr_Format(PyExc_ValueError,
            "n_inv (%d) + n_var (%d) != n_index (%d)", n_inv, n_var, n_index);
        return NULL;
    }

    /* Parse entries to get total count */
    PyArrayObject *entries_arr = (PyArrayObject *)PyArray_FROM_OTF(
        entries_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!entries_arr) return NULL;
    int entries_len = (int)PyArray_SIZE(entries_arr);
    Py_DECREF(entries_arr);
    if (entries_len % 4 != 0) {
        PyErr_Format(PyExc_ValueError,
            "entries length (%d) not a multiple of 4", entries_len);
        return NULL;
    }
    int n_entries = entries_len / 4;

    /* Convert eigenvalues and uab_invariant */
    PyArrayObject *eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError, "eigenvalues must be shape (n_samples,)");
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }
    if (validate_eigenvalues(
            (const double *)PyArray_DATA(eigenvalues_arr), n_samples) < 0) {
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }

    PyArrayObject *uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) { Py_DECREF(eigenvalues_arr); return NULL; }

    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != n_inv ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_invariant must be shape (%d, %d)", n_inv, n_samples);
        Py_DECREF(eigenvalues_arr);
        Py_DECREF(uab_inv_arr);
        return NULL;
    }

    /* Allocate workspace */
    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)calloc(
        1, sizeof(lmm_workspace_general_t));
    if (!ws) {
        PyErr_NoMemory();
        Py_DECREF(eigenvalues_arr);
        Py_DECREF(uab_inv_arr);
        return NULL;
    }

    ws->n_samples = n_samples;
    ws->n_grid = n_grid;
    ws->n_refine = n_refine;

    /* Fill table */
    ws->table.n_cvt = n_cvt;
    ws->table.n_index = n_index;
    ws->table.n_rows = n_rows;
    ws->table.n_inv = n_inv;
    ws->table.n_var = n_var;
    ws->table.idx_xx = idx_xx;
    ws->table.idx_xy = idx_xy;
    ws->table.idx_yy = idx_yy;
    ws->table.df = n_samples - n_cvt - 1;
    ws->table.n_entries = n_entries;

    /* Parse index arrays into owned copies — check each immediately to
     * avoid calling Python/C API with a live exception set. */
    ws->table.invariant_indices = parse_int32_array(inv_idx_obj, n_inv, "invariant_indices");
    if (!ws->table.invariant_indices) goto err_ws;
    ws->table.varying_indices   = parse_int32_array(var_idx_obj, n_var, "varying_indices");
    if (!ws->table.varying_indices) goto err_ws;
    ws->table.logdet_diag_rows  = parse_int32_array(diag_rows_obj, n_cvt + 1, "logdet_diag_rows");
    if (!ws->table.logdet_diag_rows) goto err_ws;
    ws->table.logdet_diag_cols  = parse_int32_array(diag_cols_obj, n_cvt + 1, "logdet_diag_cols");
    if (!ws->table.logdet_diag_cols) goto err_ws;
    ws->table.level_offsets     = parse_int32_array(offsets_obj, n_rows, "level_offsets");
    if (!ws->table.level_offsets) goto err_ws;
    ws->table.level_counts      = parse_int32_array(counts_obj, n_rows, "level_counts");
    if (!ws->table.level_counts) goto err_ws;

    /* Parse entries (stride-4) into pab_entry_t array */
    {
        int *raw_entries = parse_int32_array(entries_obj, n_entries * 4, "entries");
        if (!raw_entries) goto err_ws;
        ws->table.entries = (pab_entry_t *)malloc(
            (size_t)n_entries * sizeof(pab_entry_t));
        if (!ws->table.entries) {
            free(raw_entries);
            PyErr_NoMemory();
            goto err_ws;
        }
        for (int i = 0; i < n_entries; i++) {
            ws->table.entries[i].index_ab = raw_entries[i * 4 + 0];
            ws->table.entries[i].index_aw = raw_entries[i * 4 + 1];
            ws->table.entries[i].index_bw = raw_entries[i * 4 + 2];
            ws->table.entries[i].index_ww = raw_entries[i * 4 + 3];
        }
        free(raw_entries);
    }

    /* Validate all table indices are within [0, n_index) to prevent
     * out-of-bounds access in the OpenMP parallel loop. A bug in
     * build_pab_table_for_c() or a stale @lru_cache entry could produce
     * invalid indices that corrupt stack buffers silently. */
    for (int i = 0; i < n_inv; i++) {
        if (ws->table.invariant_indices[i] < 0 ||
            ws->table.invariant_indices[i] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "invariant_indices[%d] = %d out of range [0, %d)",
                i, ws->table.invariant_indices[i], n_index);
            goto err_ws;
        }
    }
    for (int i = 0; i < n_var; i++) {
        if (ws->table.varying_indices[i] < 0 ||
            ws->table.varying_indices[i] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "varying_indices[%d] = %d out of range [0, %d)",
                i, ws->table.varying_indices[i], n_index);
            goto err_ws;
        }
    }
    for (int d = 0; d < n_cvt + 1; d++) {
        if (ws->table.logdet_diag_rows[d] < 0 ||
            ws->table.logdet_diag_rows[d] >= n_rows) {
            PyErr_Format(PyExc_ValueError,
                "logdet_diag_rows[%d] = %d out of range [0, %d)",
                d, ws->table.logdet_diag_rows[d], n_rows);
            goto err_ws;
        }
        if (ws->table.logdet_diag_cols[d] < 0 ||
            ws->table.logdet_diag_cols[d] >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "logdet_diag_cols[%d] = %d out of range [0, %d)",
                d, ws->table.logdet_diag_cols[d], n_index);
            goto err_ws;
        }
    }
    /* Validate level_offsets + level_counts don't exceed n_entries */
    for (int p = 0; p < n_rows; p++) {
        if (ws->table.level_offsets[p] < 0 ||
            ws->table.level_counts[p] < 0 ||
            (int64_t)ws->table.level_offsets[p] + ws->table.level_counts[p] > n_entries) {
            PyErr_Format(PyExc_ValueError,
                "level_offsets[%d]=%d + level_counts[%d]=%d exceeds n_entries=%d",
                p, ws->table.level_offsets[p], p, ws->table.level_counts[p], n_entries);
            goto err_ws;
        }
    }
    if (idx_xx < 0 || idx_xx >= n_index ||
        idx_xy < 0 || idx_xy >= n_index ||
        idx_yy < 0 || idx_yy >= n_index) {
        PyErr_SetString(PyExc_ValueError, "idx_xx/xy/yy out of range [0, n_index)");
        goto err_ws;
    }
    for (int i = 0; i < n_entries; i++) {
        const pab_entry_t *e = &ws->table.entries[i];
        if (e->index_ab < 0 || e->index_ab >= n_index ||
            e->index_aw < 0 || e->index_aw >= n_index ||
            e->index_bw < 0 || e->index_bw >= n_index ||
            e->index_ww < 0 || e->index_ww >= n_index) {
            PyErr_Format(PyExc_ValueError,
                "entries[%d] has index out of range [0, %d)", i, n_index);
            goto err_ws;
        }
    }

    /* Copy eigenvalues (owned) */
    ws->eigenvalues = (double *)malloc((size_t)n_samples * sizeof(double));
    if (!ws->eigenvalues) { PyErr_NoMemory(); goto err_ws; }
    memcpy(ws->eigenvalues, PyArray_DATA(eigenvalues_arr),
           (size_t)n_samples * sizeof(double));

    /* Borrow invariant Uab pointer — keep alive via Py_INCREF */
    Py_INCREF(uab_inv_arr);
    ws->uab_inv_ref = (PyObject *)uab_inv_arr;
    ws->uab_inv = (const double *)PyArray_DATA(uab_inv_arr);

    /* Compute df, reml_const, beta params */
    int df = ws->table.df;
    ws->beta_a = (double)df / 2.0;
    ws->beta_b = 0.5;
    ws->lbeta_ab = lgamma(ws->beta_a) + lgamma(ws->beta_b)
                   - lgamma(ws->beta_a + ws->beta_b);
    ws->reml_const = 0.5 * df * (log((double)df) - log(2.0 * M_PI) - 1.0);

    /* Build lambda grid */
    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);

    ws->lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->hi_eval_grid = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    ws->logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    ws->inv_sums_grid = (double *)malloc(
        (size_t)n_grid * (size_t)n_inv * sizeof(double));

    if (!ws->lambda_grid || !ws->hi_eval_grid ||
        !ws->logdet_h_grid || !ws->inv_sums_grid) {
        PyErr_NoMemory();
        goto err_ws;
    }

    for (int g = 0; g < n_grid; g++)
        ws->lambda_grid[g] = exp(log_l_min + g * step);

    /* Precompute hi_eval_grid, logdet_h_grid, and invariant sums */
    for (int g = 0; g < n_grid; g++) {
        double lam = ws->lambda_grid[g];
        double *hi_row = ws->hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;

        /* First pass: compute hi_eval + logdet_h */
        for (int i = 0; i < n_samples; i++) {
            double v = lam * ws->eigenvalues[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);
        }
        ws->logdet_h_grid[g] = logdet;

        /* Compute invariant dot products for this grid point */
        double *inv_sums = ws->inv_sums_grid + (size_t)g * n_inv;
        for (int c = 0; c < n_inv; c++) {
            double s = 0.0;
            const double *col = ws->uab_inv + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += hi_row[i] * col[i];
            inv_sums[c] = s;
        }
    }

    /* Precompute invariant identity sums (sum of each invariant Uab column
     * at identity, i.e. lambda=0, hi=1). These are constant across SNPs
     * and reused in the per-SNP logdet_iab computation. The varying identity
     * sums are SNP-dependent (genotype cross-products), so logdet_iab must
     * be computed per-SNP in compute_lmm_chunk_general_c. */
    ws->inv_identity_sums = (double *)malloc((size_t)n_inv * sizeof(double));
    if (!ws->inv_identity_sums) { PyErr_NoMemory(); goto err_ws; }
    for (int c = 0; c < n_inv; c++) {
        double s = 0.0;
        const double *col = ws->uab_inv + (size_t)c * n_samples;
        for (int i = 0; i < n_samples; i++)
            s += col[i];
        ws->inv_identity_sums[c] = s;
    }
    /* Wrap in PyCapsule */
    PyObject *capsule = PyCapsule_New(
        ws, "lmm_workspace_general", lmm_workspace_general_destructor);
    if (!capsule) goto err_ws;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    return capsule;

err_ws:
    if (ws) {
        free(ws->lambda_grid);
        free(ws->hi_eval_grid);
        free(ws->logdet_h_grid);
        free(ws->inv_sums_grid);
        free(ws->eigenvalues);
        free(ws->inv_identity_sums);
        free(ws->table.invariant_indices);
        free(ws->table.varying_indices);
        free(ws->table.logdet_diag_rows);
        free(ws->table.logdet_diag_cols);
        free(ws->table.level_offsets);
        free(ws->table.level_counts);
        free(ws->table.entries);
        Py_XDECREF(ws->uab_inv_ref);
        free(ws);
    }
    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_inv_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lmm_chunk_general_c
 *
 * Per-chunk compute using a pre-built general workspace. OpenMP parallel
 * over SNPs. Each thread has its own stack-allocated Pab buffers.
 *
 * Python signature:
 *   compute_lmm_chunk_general_c(
 *       workspace,      # PyCapsule from create_workspace_general_c
 *       uab_varying,    # (n_snps, n_var, n_samples) float64 — SoA
 *       n_threads,      # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_chunk_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "uab_varying", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *uab_var_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &uab_var_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_general_t *ws = (lmm_workspace_general_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace_general");
    if (!ws) return NULL;

    PyArrayObject *uab_var_arr = NULL;
    output_arrays_t out = {0};
    PyObject *result = NULL;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) return NULL;

    int n_samples = ws->n_samples;
    int n_var = ws->table.n_var;
    int n_inv = ws->table.n_inv;

    /* Validate shape: (n_snps, n_var, n_samples) */
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != n_var ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying must be shape (n_snps, %d, %d)", n_var, n_samples);
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);

    double *lambdas = (double *)PyArray_DATA(out.lambdas);
    double *logls   = (double *)PyArray_DATA(out.logls);
    double *betas   = (double *)PyArray_DATA(out.betas);
    double *ses     = (double *)PyArray_DATA(out.ses);
    double *pwalds  = (double *)PyArray_DATA(out.pwalds);

    int n_grid = ws->n_grid;
    int n_refine = ws->n_refine;
    int df = ws->table.df;
    int n_index = ws->table.n_index;
    double reml_const = ws->reml_const;

    /* Compute log_l_min and step from lambda_grid */
    double log_l_min = log(ws->lambda_grid[0]);
    double step = (n_grid > 1)
        ? (log(ws->lambda_grid[n_grid - 1]) - log_l_min) / (double)(n_grid - 1)
        : 0.0;

    /* Clamp n_threads */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    /* Static schedule: per-SNP cost is uniform (same table, n_grid,
     * n_refine, n_samples).  Matches n_cvt=1 split path rationale. */
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        const double *snp_var = uab_var_data +
            (size_t)snp * n_var * n_samples;

        /* Compute per-SNP logdet_iab at identity (lambda=0, hi=1).
         * Row 0: identity-weighted sums = simple column sums.
         * Invariant sums are precomputed in workspace; only varying
         * sums (genotype-dependent) need per-SNP computation. */
        double iab_row0[MAX_N_INDEX];
        for (int i = 0; i < n_index; i++) iab_row0[i] = 0.0;

        /* Invariant identity sums from precomputed workspace */
        for (int c = 0; c < n_inv; c++)
            iab_row0[ws->table.invariant_indices[c]] = ws->inv_identity_sums[c];
        /* Varying identity sums from this SNP's uab_var */
        for (int c = 0; c < n_var; c++) {
            double s = 0.0;
            const double *col = snp_var + (size_t)c * n_samples;
            for (int i = 0; i < n_samples; i++)
                s += col[i];
            iab_row0[ws->table.varying_indices[c]] = s;
        }

        /* Compute full Iab Pab recursion */
        double iab_pab[MAX_PAB_SIZE];
        calc_pab_general(iab_row0, &ws->table, iab_pab);

        /* logdet_iab from diagonal entries.  Non-positive diagonal → NaN
         * propagates through golden section and REML sentinel. */
        double logdet_iab = 0.0;
        int iab_degenerate = 0;
        for (int d = 0; d < ws->table.n_cvt + 1; d++) {
            double val = iab_pab[ws->table.logdet_diag_rows[d] * n_index
                                 + ws->table.logdet_diag_cols[d]];
            if (val <= 0.0) { iab_degenerate = 1; break; }
            logdet_iab += log(val);
        }
        if (iab_degenerate) logdet_iab = (double)NAN;

        /* Golden section optimization */
        double logl_opt, beta, se, f_stat;
        int is_valid;
        double lambda_opt = golden_section_lambda_general(
            ws->uab_inv, snp_var, ws->eigenvalues,
            n_samples, ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->inv_sums_grid,
            log_l_min, step, n_grid, n_refine,
            logdet_iab, reml_const, &ws->table,
            &logl_opt, &beta, &se, &f_stat, &is_valid
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;
        betas[snp]   = beta;
        ses[snp]     = se;
        pwalds[snp]  = f_to_pvalue(
            f_stat, df, is_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0)
        goto err_output;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(uab_var_arr);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    Py_XDECREF(uab_var_arr);
    return NULL;
}


/* =========================================================================
 * Python entry points
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * compute_lmm_batch_split_c
 *
 * Python-callable entry point for the split-Uab path.
 *
 * Python signature:
 *   compute_lmm_batch_split_c(
 *       eigenvalues,   # (n_samples,) float64
 *       uab_varying,   # (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]
 *       uab_invariant, # (3, n_samples) float64 — SoA [ww, wy, yy]
 *       Iab_batch,     # (n_snps, 3, 6) float64
 *       n_samples,     # int
 *       l_min,         # float
 *       l_max,         # float
 *       n_grid,        # int
 *       n_refine,      # int
 *       n_threads,     # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}  each (n_snps,) float64
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_batch_split_c(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_varying", "uab_invariant", "Iab_batch",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_var_obj, *uab_inv_obj, *iab_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiii", (char **)kwlist,
            &eigenvalues_obj, &uab_var_obj, &uab_inv_obj, &iab_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    /* Locals for goto cleanup — must be declared before any goto target */
    PyArrayObject *eigenvalues_arr = NULL, *uab_var_arr = NULL;
    PyArrayObject *uab_inv_arr = NULL, *iab_arr = NULL;
    output_arrays_t out = {0};
    double *lambda_grid = NULL, *hi_eval_grid = NULL, *logdet_h_grid = NULL;
    grid_invariant_t *grid_inv = NULL;
    PyObject *result = NULL;

    /* Convert inputs to C-contiguous double arrays */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) goto err_input;

    uab_inv_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_inv_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_inv_arr) goto err_input;

    iab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        iab_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!iab_arr) goto err_input;

    /* Validate shapes — SoA layout: (n_snps, 3, n_samples) and (3, n_samples) */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != 3 ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_varying must be shape (n_snps, 3, n_samples)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_inv_arr) != 2 ||
        PyArray_DIM(uab_inv_arr, 0) != 3 ||
        PyArray_DIM(uab_inv_arr, 1) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "uab_invariant must be shape (3, n_samples)");
        goto err_input;
    }
    if (PyArray_NDIM(iab_arr) != 3 ||
        PyArray_DIM(iab_arr, 1) != 3 ||
        PyArray_DIM(iab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Iab_batch must be shape (n_snps, 3, 6)");
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX",
            n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (PyArray_DIM(iab_arr, 0) != n_snps) {
        PyErr_SetString(PyExc_ValueError,
            "Iab_batch.shape[0] must match uab_varying.shape[0]");
        goto err_input;
    }

    /* Allocate output arrays */
    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    /* Raw pointers — SoA layout */
    const double *eigenvalues_data = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_var_data     = (const double *)PyArray_DATA(uab_var_arr);
    const double *uab_inv_data     = (const double *)PyArray_DATA(uab_inv_arr);
    const double *iab_data         = (const double *)PyArray_DATA(iab_arr);

    /* SoA invariant column pointers: (3, n_samples) layout */
    const double *inv_ww = uab_inv_data;
    const double *inv_wy = uab_inv_data + (size_t)n_samples;
    const double *inv_yy = uab_inv_data + (size_t)2 * n_samples;

    /* Reject non-finite eigenvalues before entering compute loop */
    if (validate_eigenvalues(eigenvalues_data, n_samples) < 0)
        goto err_output;

    double *lambdas  = (double *)PyArray_DATA(out.lambdas);
    double *logls    = (double *)PyArray_DATA(out.logls);
    double *betas    = (double *)PyArray_DATA(out.betas);
    double *ses      = (double *)PyArray_DATA(out.ses);
    double *pwalds   = (double *)PyArray_DATA(out.pwalds);

    int df = n_samples - 2;
    double beta_a = (double)df / 2.0;
    double beta_b = 0.5;
    double lbeta_ab = lgamma(beta_a) + lgamma(beta_b) - lgamma(beta_a + beta_b);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);
    double reml_const = 0.5 * df * (log((double)df) - log(2.0 * M_PI) - 1.0);

    /* Pre-build lambda grid */
    lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!lambda_grid) { PyErr_NoMemory(); goto err_output; }
    for (int g = 0; g < n_grid; g++) {
        lambda_grid[g] = exp(log_l_min + g * step);
    }

    /* Precompute coarse-grid hi_eval, logdet_h, and invariant dot products */
    hi_eval_grid = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    grid_inv = (grid_invariant_t *)malloc(
        (size_t)n_grid * sizeof(grid_invariant_t));

    if (!hi_eval_grid || !logdet_h_grid || !grid_inv) {
        PyErr_NoMemory();
        goto err_output;
    }

    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi_row = hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        double sw = 0.0, swy = 0.0, sy = 0.0;

        for (int i = 0; i < n_samples; i++) {
            double v = lam * eigenvalues_data[i] + 1.0;
            double h = 1.0 / v;
            hi_row[i] = h;
            logdet += log(v);  /* v > 1.0: lambda > 0, eval >= 0 */

            sw  += h * inv_ww[i];
            swy += h * inv_wy[i];
            sy  += h * inv_yy[i];
        }
        logdet_h_grid[g] = logdet;

        grid_inv[g].s_ww = sw;
        grid_inv[g].s_wy = swy;
        grid_inv[g].s_yy = sy;
        grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
        grid_inv[g].inv_s_ww = (sw != 0.0) ? 1.0 / sw : 0.0;
        grid_inv[g].pab1_5 = sy - swy * swy * grid_inv[g].inv_s_ww;
    }

    /* Thread setup — no per-thread hi_eval buffers needed for split path
     * since reml_logl_ncvt1_split fuses hi_eval into the dot product loop */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    Py_BEGIN_ALLOW_THREADS

    /* Static schedule: uniform SNP cost — no work-stealing overhead */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        /* SoA: (n_snps, 3, n_samples) — 3 contiguous columns per SNP */
        const double *snp_base = uab_var_data + (size_t)snp * 3 * n_samples;
        const double *vwx = snp_base;
        const double *vxx = snp_base + (size_t)n_samples;
        const double *vxy = snp_base + (size_t)2 * n_samples;
        const double *iab = iab_data + (size_t)snp * 3 * 6;

        double logdet_iab = compute_logdet_iab(iab);

        double lambda_opt, logl_opt, beta, se, f_stat;
        int is_valid;
        lambda_opt = golden_section_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            eigenvalues_data, logdet_iab,
            n_samples, lambda_grid, hi_eval_grid, logdet_h_grid,
            grid_inv, log_l_min, step, n_grid, n_refine,
            df, reml_const, &logl_opt,
            &beta, &se, &f_stat, &is_valid
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;

        betas[snp] = beta;
        ses[snp]   = se;

        pwalds[snp] = f_to_pvalue(f_stat, df, is_valid,
                                   beta_a, beta_b, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0)
        goto err_output;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_var_arr);
    Py_DECREF(uab_inv_arr);
    Py_DECREF(iab_arr);
    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    free(grid_inv);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_var_arr);
    Py_XDECREF(uab_inv_arr);
    Py_XDECREF(iab_arr);
    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    free(grid_inv);
    return NULL;
}

/* -------------------------------------------------------------------------
 * compute_lmm_batch_c
 *
 * Python-callable entry point.
 *
 * Python signature:
 *   compute_lmm_batch_c(
 *       eigenvalues,   # (n_samples,) float64
 *       Uab_batch,     # (n_snps, n_samples, 6) float64
 *       Iab_batch,     # (n_snps, 3, 6) float64
 *       n_samples,     # int
 *       l_min,         # float
 *       l_max,         # float
 *       n_grid,        # int
 *       n_refine,      # int
 *       n_threads,     # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds}  each (n_snps,) float64
 *
 * p-values are computed C-side via betainc (Lentz CF) for full end-to-end
 * acceleration. The output key is 'pwalds' (not 'f_stats' as in v1).
 * ------------------------------------------------------------------------- */
static PyObject *compute_lmm_batch_c(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "Uab_batch", "Iab_batch",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        NULL
    };

    PyObject *eigenvalues_obj, *uab_obj, *iab_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOiddiii", (char **)kwlist,
            &eigenvalues_obj, &uab_obj, &iab_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine, &n_threads)) {
        return NULL;
    }

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    /* Locals for goto cleanup — must be declared before any goto target */
    PyArrayObject *eigenvalues_arr = NULL, *uab_arr = NULL, *iab_arr = NULL;
    output_arrays_t out = {0};
    double *lambda_grid = NULL, *hi_eval_grid = NULL, *logdet_h_grid = NULL;
    double **thread_bufs = NULL;
    int actual_threads = 1;
    PyObject *result = NULL;

    /* Convert inputs to C-contiguous double arrays */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    uab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_arr) goto err_input;

    iab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        iab_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!iab_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_arr) != 3 ||
        PyArray_DIM(uab_arr, 1) != n_samples ||
        PyArray_DIM(uab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Uab_batch must be shape (n_snps, n_samples, 6)");
        goto err_input;
    }
    if (PyArray_NDIM(iab_arr) != 3 ||
        PyArray_DIM(iab_arr, 1) != 3 ||
        PyArray_DIM(iab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Iab_batch must be shape (n_snps, 3, 6)");
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (PyArray_DIM(iab_arr, 0) != n_snps) {
        PyErr_SetString(PyExc_ValueError,
            "Iab_batch.shape[0] must match Uab_batch.shape[0] (n_snps)");
        goto err_input;
    }

    /* Allocate output arrays */
    if (alloc_output_arrays(&out, n_snps) < 0)
        goto err_input;

    /* Raw pointers */
    const double *eigenvalues_data = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_data         = (const double *)PyArray_DATA(uab_arr);
    const double *iab_data         = (const double *)PyArray_DATA(iab_arr);

    /* Reject non-finite eigenvalues before entering compute loop */
    if (validate_eigenvalues(eigenvalues_data, n_samples) < 0)
        goto err_output;

    double *lambdas  = (double *)PyArray_DATA(out.lambdas);
    double *logls    = (double *)PyArray_DATA(out.logls);
    double *betas    = (double *)PyArray_DATA(out.betas);
    double *ses      = (double *)PyArray_DATA(out.ses);
    double *pwalds   = (double *)PyArray_DATA(out.pwalds);

    int df = n_samples - 2;
    double beta_a = (double)df / 2.0;
    double beta_b = 0.5;
    double lbeta_ab = lgamma(beta_a) + lgamma(beta_b) - lgamma(beta_a + beta_b);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);
    double reml_const = 0.5 * df * (log((double)df) - log(2.0 * M_PI) - 1.0);

    /* Pre-build lambda grid */
    lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!lambda_grid) { PyErr_NoMemory(); goto err_output; }
    for (int g = 0; g < n_grid; g++) {
        lambda_grid[g] = exp(log_l_min + g * step);
    }

    /* Precompute coarse-grid hi_eval and logdet_h.
     *
     * hi_eval_grid: (n_grid * n_samples) — hi_eval[g][i] = 1/(lambda_grid[g]*eval[i]+1)
     * logdet_h_grid: (n_grid)            — sum of log(lambda_grid[g]*eval[i]+1) per grid point
     *
     * These are identical across all SNPs (eigenvalues are shared) so we compute
     * them once here instead of n_snps * n_grid times inside the parallel loop.
     * Memory: n_grid * n_samples * 8 bytes (e.g. 50 * 50k = 20 MB). */
    hi_eval_grid = alloc_aligned_doubles((size_t)n_grid * (size_t)n_samples);
    logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));

    if (!hi_eval_grid || !logdet_h_grid) { PyErr_NoMemory(); goto err_output; }

    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi_row = hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        #pragma omp simd reduction(+:logdet)
        for (int i = 0; i < n_samples; i++) {
            double v = lam * eigenvalues_data[i] + 1.0;
            hi_row[i] = 1.0 / v;
            logdet += log(v);  /* v > 1.0: lambda > 0, eval >= 0 */
        }
        logdet_h_grid[g] = logdet;
    }

    /* Pre-allocate per-thread hi_eval buffers OUTSIDE the parallel region.
     * Each thread reuses one buffer across all its SNPs, eliminating the
     * malloc/free-per-SNP that caused heap lock contention with 48 threads. */
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    thread_bufs = (double **)calloc((size_t)actual_threads, sizeof(double *));
    int alloc_ok = (thread_bufs != NULL);
    if (alloc_ok) {
        for (int t = 0; t < actual_threads; t++) {
            thread_bufs[t] = alloc_aligned_doubles((size_t)n_samples);
            if (!thread_bufs[t]) { alloc_ok = 0; break; }
        }
    }

    if (!alloc_ok) {
        PyErr_NoMemory();
        goto err_output;
    }

    Py_BEGIN_ALLOW_THREADS

    /* Static schedule: uniform SNP cost — no work-stealing overhead */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        /* Per-thread buffer — no malloc inside the hot loop */
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval = thread_bufs[tid];

        const double *uab = uab_data + (size_t)snp * n_samples * 6;
        const double *iab = iab_data + (size_t)snp * 3 * 6;

        double logdet_iab = compute_logdet_iab(iab);

        double lambda_opt, logl_opt;
        double beta, se, f_stat;
        int is_valid;
        lambda_opt = golden_section_lambda_ncvt1(
            uab, eigenvalues_data, logdet_iab, n_samples,
            lambda_grid, hi_eval_grid, logdet_h_grid,
            log_l_min, step, n_grid, n_refine,
            df, reml_const,
            hi_eval, &logl_opt,
            &beta, &se, &f_stat, &is_valid
        );

        lambdas[snp] = lambda_opt;
        logls[snp]   = logl_opt;
        betas[snp]   = beta;
        ses[snp]     = se;

        pwalds[snp] = f_to_pvalue(f_stat, df, is_valid, beta_a, beta_b, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(betas, pwalds, n_snps) < 0) {
        /* Free per-thread buffers before going to error path */
        for (int t = 0; t < actual_threads; t++) free(thread_bufs[t]);
        free(thread_bufs);
        thread_bufs = NULL;
        goto err_output;
    }

    /* Free per-thread buffers */
    for (int t = 0; t < actual_threads; t++) free(thread_bufs[t]);
    free(thread_bufs);
    thread_bufs = NULL;

    result = build_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(eigenvalues_arr);
    Py_DECREF(uab_arr);
    Py_DECREF(iab_arr);
    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    return result;

err_output:
    decref_output_arrays(&out);
err_input:
    if (thread_bufs) {
        for (int t = 0; t < actual_threads; t++) free(thread_bufs[t]);
        free(thread_bufs);
    }
    Py_XDECREF(eigenvalues_arr);
    Py_XDECREF(uab_arr);
    Py_XDECREF(iab_arr);
    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    return NULL;
}

/* =========================================================================
 * MLE (Maximum Likelihood Estimation) helpers for LRT
 *
 * Key differences from REML:
 *   - No Iab parameter or logdet_hiw computation
 *   - Uses n_samples instead of df = n_samples - n_cvt - 1
 *   - MLE constant: 0.5 * n * (log(n) - log(2*pi) - 1)
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * chi2_sf_c
 *
 * Chi-squared survival function for df=1: P(X > x) = erfc(sqrt(x/2)).
 * Uses C99 erfc() from math.h — no custom implementation needed.
 * Matches special.py chi2_sf exactly.
 * ------------------------------------------------------------------------- */
static inline double chi2_sf_c(double x)
{
    if (isnan(x)) return x;          /* NaN propagation */
    if (x <= 0.0) return 1.0;
    if (!isfinite(x)) return 0.0;   /* +inf → 0 */
    return erfc(sqrt(x / 2.0));
}

/* -------------------------------------------------------------------------
 * mle_finish
 *
 * MLE log-likelihood tail (simpler than REML — no logdet_hiw, no Iab).
 * logl = mle_const - 0.5 * logdet_h - 0.5 * n * log(P_yy)
 *
 * P_yy at level nc_total = n_cvt+1 = 2 (pab[2][5]) — same index as REML.
 * Uses n_samples (not df).
 * ------------------------------------------------------------------------- */
static inline double mle_finish(
    const double pab[3][6],
    double logdet_h,
    int n_samples,
    double mle_const
)
{
    double P_yy = pab[2][5];
    if (P_yy < 0.0) return (double)NAN;
    if (P_yy < P_YY_MIN) P_yy = P_YY_MIN;
    return mle_const - 0.5 * logdet_h - 0.5 * n_samples * log(P_yy);
}

/* -------------------------------------------------------------------------
 * mle_logl_ncvt1
 *
 * MLE log-likelihood for one SNP at one lambda (n_cvt=1).
 * Used during golden section refinement.
 * Returns MLE log-likelihood.
 * ------------------------------------------------------------------------- */
static double mle_logl_ncvt1(
    const double * restrict uab,
    const double * restrict eigenvalues,
    int n_samples,
    double lambda,
    double mle_const,
    double * restrict hi_eval
)
{
    double logdet_h = 0.0;
    #pragma omp simd reduction(+:logdet_h)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        hi_eval[i] = 1.0 / v;
        logdet_h += log(v);
    }

    double pab[3][6];
    calc_pab_ncvt1(uab, hi_eval, n_samples, pab);

    return mle_finish(pab, logdet_h, n_samples, mle_const);
}

/* -------------------------------------------------------------------------
 * mle_logl_ncvt1_cached
 *
 * MLE log-likelihood using precomputed hi_eval and logdet_h from
 * the shared coarse grid cache.
 * ------------------------------------------------------------------------- */
static double mle_logl_ncvt1_cached(
    const double * restrict uab,
    const double * restrict cached_hi_eval,
    double cached_logdet_h,
    int n_samples,
    double mle_const
)
{
    double pab[3][6];
    calc_pab_ncvt1(uab, cached_hi_eval, n_samples, pab);

    return mle_finish(pab, cached_logdet_h, n_samples, mle_const);
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_mle_ncvt1
 *
 * Grid search + golden section refinement for MLE lambda (n_cvt=1).
 * Structurally identical to golden_section_lambda_ncvt1 but:
 *   - Uses mle_finish instead of reml_finish
 *   - No Iab parameter or logdet_iab
 *   - Uses n_samples instead of df in the likelihood
 *
 * Returns optimal lambda; writes logl to *logl_out.
 * ------------------------------------------------------------------------- */
static double golden_section_lambda_mle_ncvt1(
    const double * restrict uab,
    const double * restrict eigenvalues,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    double log_l_min, double step,
    int n_grid, int n_refine,
    double mle_const,
    double * restrict hi_eval, double *logl_out
)
{
    const double phi = 0.6180339887498949;  /* golden ratio - 1 */

    /* Stage 1: coarse grid search using cached hi_eval and logdet_h */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = mle_logl_ncvt1_cached(
            uab,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g],
            n_samples, mle_const
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    /* Fully degenerate SNP — all grid evaluations returned NaN */
    if (best_logl == REML_SENTINEL) {
        *logl_out = (double)NAN;
        return (double)NAN;
    }

    /* Bracket around best grid point */
    int idx_low = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                exp(c), mle_const, hi_eval);
    double fd = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                exp(d), mle_const, hi_eval);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                 exp(c), mle_const, hi_eval);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                 exp(d), mle_const, hi_eval);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);
    *logl_out = mle_logl_ncvt1(uab, eigenvalues, n_samples,
                                lambda_opt, mle_const, hi_eval);

    return lambda_opt;
}

/* -------------------------------------------------------------------------
 * mle_logl_ncvt1_cached_split
 *
 * MLE log-likelihood from SoA split data using cached grid hi_eval.
 * Pattern-matches reml_logl_ncvt1_cached_split but:
 *   - No logdet_iab / logdet_hiw terms
 *   - Uses n_samples (not df)
 *   - Uses mle_const (not reml_const)
 * ------------------------------------------------------------------------- */
static double mle_logl_ncvt1_cached_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict cached_hi_eval,
    double cached_logdet_h,
    const grid_invariant_t *ginv,
    int n_samples,
    double mle_const
)
{
    double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
    #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
    for (int i = 0; i < n_samples; i++) {
        double h = cached_hi_eval[i];
        s_wx += h * var_wx[i];
        s_xx += h * var_xx[i];
        s_xy += h * var_xy[i];
    }

    /* Combine with precomputed invariant sums */
    double pab[3][6];
    calc_pab_ncvt1_split(ginv->s_ww, s_wx, ginv->s_wy,
                          s_xx, s_xy, ginv->s_yy, pab);

    return mle_finish(pab, cached_logdet_h, n_samples, mle_const);
}

/* -------------------------------------------------------------------------
 * mle_logl_ncvt1_split
 *
 * MLE log-likelihood from SoA split data at an arbitrary lambda.
 * Used during golden section refinement. Computes hi_eval from scratch,
 * accumulates all 6 dot products (3 invariant + 3 varying), builds Pab.
 *
 * hi_eval is a caller-provided scratch buffer of size (n_samples,).
 * ------------------------------------------------------------------------- */
static double mle_logl_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    int n_samples,
    double lambda,
    double mle_const,
    double * restrict hi_eval
)
{
    double logdet_h = 0.0;
    double s_ww = 0.0, s_wx = 0.0, s_wy = 0.0;
    double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;

    #pragma omp simd reduction(+:logdet_h,s_ww,s_wx,s_wy,s_xx,s_xy,s_yy)
    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        hi_eval[i] = h;
        logdet_h += log(v);

        s_wx += h * var_wx[i];
        s_xx += h * var_xx[i];
        s_xy += h * var_xy[i];

        s_ww += h * inv_ww[i];
        s_wy += h * inv_wy[i];
        s_yy += h * inv_yy[i];
    }

    double pab[3][6];
    calc_pab_ncvt1_split(s_ww, s_wx, s_wy, s_xx, s_xy, s_yy, pab);

    return mle_finish(pab, logdet_h, n_samples, mle_const);
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_mle_ncvt1_split
 *
 * Grid search + golden section refinement for MLE lambda using SoA split
 * data. Structurally identical to golden_section_lambda_mle_ncvt1 but
 * uses split cached/refinement evaluators.
 *
 * Returns optimal MLE lambda; writes log-likelihood to *logl_out.
 * hi_eval is a caller-provided scratch buffer of size (n_samples,).
 * ------------------------------------------------------------------------- */
static double golden_section_lambda_mle_ncvt1_split(
    const double * restrict var_wx,
    const double * restrict var_xx,
    const double * restrict var_xy,
    const double * restrict inv_ww,
    const double * restrict inv_wy,
    const double * restrict inv_yy,
    const double * restrict eigenvalues,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const grid_invariant_t *grid_inv,
    double log_l_min, double step,
    int n_grid, int n_refine,
    double mle_const,
    double * restrict hi_eval,
    double *logl_out
)
{
    const double phi = 0.6180339887498949;

    /* Stage 1: coarse grid search using cached split */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = mle_logl_ncvt1_cached_split(
            var_wx, var_xx, var_xy,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g],
            &grid_inv[g],
            n_samples, mle_const
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    /* Fully degenerate SNP */
    if (best_logl == REML_SENTINEL) {
        *logl_out = (double)NAN;
        return (double)NAN;
    }

    /* Bracket around best grid point */
    int idx_low = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                      inv_ww, inv_wy, inv_yy, eigenvalues,
                                      n_samples, exp(c), mle_const, hi_eval);
    double fd = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                      inv_ww, inv_wy, inv_yy, eigenvalues,
                                      n_samples, exp(d), mle_const, hi_eval);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       n_samples, exp(c), mle_const, hi_eval);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                       inv_ww, inv_wy, inv_yy, eigenvalues,
                                       n_samples, exp(d), mle_const, hi_eval);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);
    *logl_out = mle_logl_ncvt1_split(var_wx, var_xx, var_xy,
                                      inv_ww, inv_wy, inv_yy, eigenvalues,
                                      n_samples, lambda_opt, mle_const, hi_eval);

    return lambda_opt;
}

/* =========================================================================
 * LRT BATCH — compute_lrt_batch_c
 *
 * LRT test with per-SNP MLE optimization + chi2_sf p-value.
 * ========================================================================= */

/* LRT output: 2 arrays */
typedef struct {
    PyArrayObject *lambdas_mle;
    PyArrayObject *p_lrts;
} lrt_output_t;

static int alloc_lrt_output(lrt_output_t *out, npy_intp n_snps)
{
    npy_intp dims[1] = { n_snps };
    out->lambdas_mle = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->p_lrts      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!out->lambdas_mle || !out->p_lrts) {
        Py_XDECREF(out->lambdas_mle);
        Py_XDECREF(out->p_lrts);
        return -1;
    }
    return 0;
}

static void decref_lrt_output(lrt_output_t *out)
{
    Py_DECREF(out->lambdas_mle);
    Py_DECREF(out->p_lrts);
}

static PyObject *build_lrt_result_dict(lrt_output_t *out)
{
    PyObject *result = PyDict_New();
    if (!result) {
        decref_lrt_output(out);
        return NULL;
    }

    if (PyDict_SetItemString(result, "lambdas_mle", (PyObject *)out->lambdas_mle) < 0 ||
        PyDict_SetItemString(result, "p_lrts",      (PyObject *)out->p_lrts)      < 0) {
        Py_DECREF(result);
        decref_lrt_output(out);
        return NULL;
    }

    decref_lrt_output(out);
    return result;
}

/* -------------------------------------------------------------------------
 * compute_lrt_batch_c
 *
 * Batch LRT for n_cvt=1 with optional OpenMP.
 * Per-SNP MLE golden section optimization + chi2_sf for p-value.
 *
 * Args: eigenvalues (n_samples,), Uab_batch (n_snps, n_samples, 6),
 *       n_samples, l_min, l_max, n_grid, n_refine, logl_H0, n_threads
 * Returns: dict with keys lambdas_mle, p_lrts (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_lrt_batch_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_batch_obj;
    int n_samples, n_grid, n_refine, n_threads;
    double l_min, l_max, logl_H0;

    if (!PyArg_ParseTuple(args, "OOiddiidi",
            &eigenvalues_obj, &uab_batch_obj,
            &n_samples, &l_min, &l_max, &n_grid, &n_refine,
            &logl_H0, &n_threads))
        return NULL;

    if (validate_batch_params(n_samples, l_min, l_max, n_grid, n_refine) < 0)
        return NULL;

    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return NULL;
    }

    /* Convert inputs to C-contiguous double arrays */
    PyArrayObject *eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) return NULL;

    PyArrayObject *uab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_batch_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_arr) { Py_DECREF(eigenvalues_arr); return NULL; }

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }
    if (PyArray_NDIM(uab_arr) != 3 ||
        PyArray_DIM(uab_arr, 1) != n_samples ||
        PyArray_DIM(uab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Uab_batch must be shape (n_snps, n_samples, 6)");
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }

    const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_batch = (const double *)PyArray_DATA(uab_arr);

    if (validate_eigenvalues(eigenvalues, n_samples) < 0) {
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr); return NULL;
    }

    /* Allocate output arrays */
    lrt_output_t out;
    if (alloc_lrt_output(&out, (npy_intp)n_snps) < 0) {
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
        return PyErr_NoMemory();
    }

    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    /* Pre-compute MLE constant and grid */
    double n = (double)n_samples;
    double mle_const = 0.5 * n * (log(n) - log(2.0 * M_PI) - 1.0);

    double log_l_min = log(l_min);
    double log_l_max = log(l_max);
    double step = (log_l_max - log_l_min) / (double)(n_grid - 1);

    /* Build lambda grid */
    double *lambda_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!lambda_grid) {
        decref_lrt_output(&out);
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
        return PyErr_NoMemory();
    }
    for (int g = 0; g < n_grid; g++)
        lambda_grid[g] = exp(log_l_min + g * step);

    /* Pre-compute hi_eval_grid and logdet_h_grid (shared across SNPs) */
    double *hi_eval_grid = (double *)malloc(
        (size_t)n_grid * (size_t)n_samples * sizeof(double));
    double *logdet_h_grid = (double *)malloc((size_t)n_grid * sizeof(double));
    if (!hi_eval_grid || !logdet_h_grid) {
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        decref_lrt_output(&out);
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
        return PyErr_NoMemory();
    }
    for (int g = 0; g < n_grid; g++) {
        double lam = lambda_grid[g];
        double *hi = hi_eval_grid + (size_t)g * n_samples;
        double logdet = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double v = lam * eigenvalues[i] + 1.0;
            hi[i] = 1.0 / v;
            logdet += log(v);
        }
        logdet_h_grid[g] = logdet;
    }

    /* Pre-allocate per-thread hi_eval buffers OUTSIDE the parallel region.
     * Matches the Wald batch pattern — fail hard with PyErr_NoMemory rather
     * than silently producing NaN from inside an OpenMP parallel region. */
    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    double **thread_bufs = (double **)calloc((size_t)actual_threads, sizeof(double *));
    int alloc_ok = (thread_bufs != NULL);
    if (alloc_ok) {
        for (int t = 0; t < actual_threads; t++) {
            thread_bufs[t] = alloc_aligned_doubles((size_t)n_samples);
            if (!thread_bufs[t]) { alloc_ok = 0; break; }
        }
    }
    if (!alloc_ok) {
        if (thread_bufs) {
            for (int t = 0; t < actual_threads; t++) free(thread_bufs[t]);
            free(thread_bufs);
        }
        free(lambda_grid);
        free(hi_eval_grid);
        free(logdet_h_grid);
        decref_lrt_output(&out);
        Py_DECREF(uab_arr); Py_DECREF(eigenvalues_arr);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval_local = thread_bufs[tid];
        const double *uab = uab_batch + (size_t)s * n_samples * 6;

        double logl_H1;
        double lam_mle = golden_section_lambda_mle_ncvt1(
            uab, eigenvalues, n_samples,
            lambda_grid, hi_eval_grid, logdet_h_grid,
            log_l_min, step, n_grid, n_refine,
            mle_const, hi_eval_local, &logl_H1
        );
        out_lambdas_mle[s] = lam_mle;

        /* LRT stat = 2*(logl_H1 - logl_H0), clamp >= 0.
         * NaN logl_H1 (degenerate SNP): NaN-finite=NaN, NaN<0 is false
         * (IEEE 754), chi2_sf_c(NaN) returns NaN. Correct: degenerate
         * SNPs get NaN p-value. */
        double lrt_stat = 2.0 * (logl_H1 - logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[s] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    for (int t = 0; t < actual_threads; t++) free(thread_bufs[t]);
    free(thread_bufs);

    free(lambda_grid);
    free(hi_eval_grid);
    free(logdet_h_grid);
    Py_DECREF(uab_arr);
    Py_DECREF(eigenvalues_arr);

    return build_lrt_result_dict(&out);
}

/* =========================================================================
 * SCORE BATCH — compute_score_batch_c
 *
 * Score test with fixed null-model Hi_eval shared across all SNPs.
 * No per-SNP lambda optimization — just Pab computation + F-test.
 * ========================================================================= */

/* Score output: 3 arrays */
typedef struct {
    PyArrayObject *betas;
    PyArrayObject *ses;
    PyArrayObject *p_scores;
} score_output_t;

static int alloc_score_output(score_output_t *out, npy_intp n_snps)
{
    npy_intp dims[1] = { n_snps };
    out->betas    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->ses      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->p_scores = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!out->betas || !out->ses || !out->p_scores) {
        Py_XDECREF(out->betas);
        Py_XDECREF(out->ses);
        Py_XDECREF(out->p_scores);
        return -1;
    }
    return 0;
}

static void decref_score_output(score_output_t *out)
{
    Py_DECREF(out->betas);
    Py_DECREF(out->ses);
    Py_DECREF(out->p_scores);
}

static PyObject *build_score_result_dict(score_output_t *out)
{
    PyObject *result = PyDict_New();
    if (!result) {
        decref_score_output(out);
        return NULL;
    }

    if (PyDict_SetItemString(result, "betas",    (PyObject *)out->betas)    < 0 ||
        PyDict_SetItemString(result, "ses",      (PyObject *)out->ses)      < 0 ||
        PyDict_SetItemString(result, "p_scores", (PyObject *)out->p_scores) < 0) {
        Py_DECREF(result);
        decref_score_output(out);
        return NULL;
    }

    decref_score_output(out);
    return result;
}

/* -------------------------------------------------------------------------
 * compute_score_batch_c
 *
 * Batch Score test for n_cvt=1 with optional OpenMP.
 *
 * Args: eigenvalues (n_samples,), Uab_batch (n_snps, n_samples, 6),
 *       Hi_eval_null (n_samples,), n_samples, n_threads
 * Returns: dict with keys betas, ses, p_scores (each n_snps,)
 * ------------------------------------------------------------------------- */
static PyObject *compute_score_batch_c(PyObject *self, PyObject *args)
{
    PyObject *eigenvalues_obj, *uab_batch_obj, *hi_eval_null_obj;
    int n_samples, n_threads;
    PyArrayObject *eigenvalues_arr = NULL, *uab_arr = NULL, *hi_eval_null_arr = NULL;

    if (!PyArg_ParseTuple(args, "OOOii",
            &eigenvalues_obj, &uab_batch_obj, &hi_eval_null_obj,
            &n_samples, &n_threads))
        return NULL;

    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return NULL;
    }

    /* Convert inputs to C-contiguous double arrays */
    eigenvalues_arr = (PyArrayObject *)PyArray_FROM_OTF(
        eigenvalues_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!eigenvalues_arr) goto err_input;

    uab_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_batch_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_arr) goto err_input;

    hi_eval_null_arr = (PyArrayObject *)PyArray_FROM_OTF(
        hi_eval_null_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!hi_eval_null_arr) goto err_input;

    /* Validate shapes */
    if (PyArray_NDIM(eigenvalues_arr) != 1 ||
        PyArray_DIM(eigenvalues_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "eigenvalues must be shape (n_samples,)");
        goto err_input;
    }
    if (PyArray_NDIM(uab_arr) != 3 ||
        PyArray_DIM(uab_arr, 1) != n_samples ||
        PyArray_DIM(uab_arr, 2) != 6) {
        PyErr_SetString(PyExc_ValueError,
            "Uab_batch must be shape (n_snps, n_samples, 6)");
        goto err_input;
    }
    if (PyArray_NDIM(hi_eval_null_arr) != 1 ||
        PyArray_DIM(hi_eval_null_arr, 0) != n_samples) {
        PyErr_SetString(PyExc_ValueError,
            "Hi_eval_null must be shape (n_samples,)");
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX; split into smaller batches",
            n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;
    if (n_snps == 0) {
        PyErr_SetString(PyExc_ValueError, "n_snps must be > 0");
        goto err_input;
    }

    const double *eigenvalues = (const double *)PyArray_DATA(eigenvalues_arr);
    const double *uab_batch = (const double *)PyArray_DATA(uab_arr);
    const double *hi_eval_null = (const double *)PyArray_DATA(hi_eval_null_arr);

    if (validate_eigenvalues(eigenvalues, n_samples) < 0)
        goto err_input;

    /* Validate Hi_eval_null for NaN/Inf from failed null model */
    for (int i = 0; i < n_samples; i++) {
        if (!isfinite(hi_eval_null[i])) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", hi_eval_null[i]);
            PyErr_Format(PyExc_ValueError,
                "Hi_eval_null[%d] = %s is not finite. "
                "Null model optimization may have failed.", i, buf);
            goto err_input;
        }
    }

    /* Allocate output arrays */
    score_output_t out;
    if (alloc_score_output(&out, (npy_intp)n_snps) < 0) {
        PyErr_NoMemory();
        goto err_input;
    }

    double *out_betas    = (double *)PyArray_DATA(out.betas);
    double *out_ses      = (double *)PyArray_DATA(out.ses);
    double *out_p_scores = (double *)PyArray_DATA(out.p_scores);

    /* Pre-compute F-distribution constants (betainc) */
    int df = n_samples - 2;  /* n_cvt=1: df = n - n_cvt - 1 */
    double a = (double)df / 2.0;  /* same convention as Wald: a=df/2, b=0.5 */
    double b = 0.5;
    double lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);

    int actual_threads = 1;
#ifdef _OPENMP
    if (n_threads > 0) {
        actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    } else {
        actual_threads = omp_get_max_threads();
        if (actual_threads > n_snps) actual_threads = n_snps;
    }
    if (actual_threads < 1) actual_threads = 1;
#else
    (void)n_threads;
#endif

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int s = 0; s < n_snps; s++) {
        const double *uab = uab_batch + (size_t)s * n_samples * 6;

        double pab[3][6];
        calc_pab_ncvt1(uab, hi_eval_null, n_samples, pab);

        double beta, se, f_stat;
        int is_valid = score_from_pab(pab, n_samples, df, &beta, &se, &f_stat);

        out_betas[s] = beta;
        out_ses[s] = se;
        out_p_scores[s] = f_to_pvalue(f_stat, df, is_valid, a, b, lbeta_ab);
    }

    Py_END_ALLOW_THREADS

    if (warn_betainc_convergence(out_betas, out_p_scores, n_snps) < 0) {
        decref_score_output(&out);
        Py_DECREF(hi_eval_null_arr);
        Py_DECREF(uab_arr);
        Py_DECREF(eigenvalues_arr);
        return NULL;
    }

    Py_DECREF(hi_eval_null_arr);
    Py_DECREF(uab_arr);
    Py_DECREF(eigenvalues_arr);
    return build_score_result_dict(&out);

err_input:
    Py_XDECREF(hi_eval_null_arr);
    Py_XDECREF(uab_arr);
    Py_XDECREF(eigenvalues_arr);
    return NULL;
}

/* =========================================================================
 * FUSED MODE-4 — compute_mode4_chunk_split_c
 *
 * Single OpenMP loop computes Wald + Score + LRT from SoA split data.
 * Eliminates Uab reconstruction and redundant Pab computation.
 * ========================================================================= */

/* Mode-4 output: 8 arrays */
typedef struct {
    PyArrayObject *lambdas;      /* REML lambda */
    PyArrayObject *logls;        /* REML log-likelihood */
    PyArrayObject *betas;        /* Wald beta (REML-optimized) */
    PyArrayObject *ses;          /* Wald SE (REML-optimized) */
    PyArrayObject *pwalds;       /* Wald p-value */
    PyArrayObject *p_scores;     /* Score p-value */
    PyArrayObject *lambdas_mle;  /* MLE lambda */
    PyArrayObject *p_lrts;       /* LRT p-value */
} mode4_output_t;

static int alloc_mode4_output(mode4_output_t *out, npy_intp n_snps)
{
    npy_intp dims[1] = { n_snps };
    out->lambdas     = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->logls       = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->betas       = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->ses         = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->pwalds      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->p_scores    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->lambdas_mle = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->p_lrts      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!out->lambdas || !out->logls || !out->betas || !out->ses ||
        !out->pwalds || !out->p_scores || !out->lambdas_mle || !out->p_lrts) {
        Py_XDECREF(out->lambdas);
        Py_XDECREF(out->logls);
        Py_XDECREF(out->betas);
        Py_XDECREF(out->ses);
        Py_XDECREF(out->pwalds);
        Py_XDECREF(out->p_scores);
        Py_XDECREF(out->lambdas_mle);
        Py_XDECREF(out->p_lrts);
        return -1;
    }
    return 0;
}

static void decref_mode4_output(mode4_output_t *out)
{
    Py_XDECREF(out->lambdas);
    Py_XDECREF(out->logls);
    Py_XDECREF(out->betas);
    Py_XDECREF(out->ses);
    Py_XDECREF(out->pwalds);
    Py_XDECREF(out->p_scores);
    Py_XDECREF(out->lambdas_mle);
    Py_XDECREF(out->p_lrts);
}

static PyObject *build_mode4_result_dict(mode4_output_t *out)
{
    PyObject *result = PyDict_New();
    if (!result) {
        decref_mode4_output(out);
        return NULL;
    }

    if (PyDict_SetItemString(result, "lambdas",     (PyObject *)out->lambdas)     < 0 ||
        PyDict_SetItemString(result, "logls",       (PyObject *)out->logls)       < 0 ||
        PyDict_SetItemString(result, "betas",       (PyObject *)out->betas)       < 0 ||
        PyDict_SetItemString(result, "ses",         (PyObject *)out->ses)         < 0 ||
        PyDict_SetItemString(result, "pwalds",      (PyObject *)out->pwalds)      < 0 ||
        PyDict_SetItemString(result, "p_scores",    (PyObject *)out->p_scores)    < 0 ||
        PyDict_SetItemString(result, "lambdas_mle", (PyObject *)out->lambdas_mle) < 0 ||
        PyDict_SetItemString(result, "p_lrts",      (PyObject *)out->p_lrts)      < 0) {
        Py_DECREF(result);
        decref_mode4_output(out);
        return NULL;
    }

    decref_mode4_output(out);
    return result;
}

/* -------------------------------------------------------------------------
 * compute_mode4_chunk_split_c
 *
 * Fused per-chunk mode-4 compute: Score + Wald + LRT in a single OpenMP
 * parallel loop from SoA split data. Requires a mode-4 workspace.
 *
 * Python signature:
 *   compute_mode4_chunk_split_c(
 *       workspace,     # PyCapsule from create_workspace_mode4_split_c
 *       uab_varying,   # (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]
 *       n_threads,     # int
 *   ) -> dict {lambdas, logls, betas, ses, pwalds, p_scores, lambdas_mle, p_lrts}
 * ------------------------------------------------------------------------- */
static PyObject *compute_mode4_chunk_split_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "uab_varying", "n_threads", NULL};

    PyObject *capsule_obj;
    PyObject *uab_var_obj;
    int n_threads;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOi", (char **)kwlist,
            &capsule_obj, &uab_var_obj, &n_threads)) {
        return NULL;
    }

    lmm_workspace_t *ws = (lmm_workspace_t *)PyCapsule_GetPointer(
        capsule_obj, "lmm_workspace");
    if (!ws) return NULL;

    /* Validate workspace mode */
    if (ws->mode != 4) {
        PyErr_Format(PyExc_ValueError,
            "compute_mode4_chunk_split_c requires a mode-4 workspace "
            "(got mode=%d). Use create_workspace_mode4_split_c.", ws->mode);
        return NULL;
    }

    PyArrayObject *uab_var_arr = NULL;
    mode4_output_t out = {0};
    PyObject *result = NULL;

    uab_var_arr = (PyArrayObject *)PyArray_FROM_OTF(
        uab_var_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!uab_var_arr) return NULL;

    int n_samples = ws->n_samples;

    /* Validate shape */
    if (PyArray_NDIM(uab_var_arr) != 3 ||
        PyArray_DIM(uab_var_arr, 1) != 3 ||
        PyArray_DIM(uab_var_arr, 2) != n_samples) {
        PyErr_Format(PyExc_ValueError,
            "uab_varying must be shape (n_snps, 3, %d)", n_samples);
        goto err_input;
    }

    npy_intp n_snps_raw = PyArray_DIM(uab_var_arr, 0);
    if (n_snps_raw > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
            "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps_raw);
        goto err_input;
    }
    int n_snps = (int)n_snps_raw;

    if (alloc_mode4_output(&out, (npy_intp)n_snps) < 0) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        goto err_input;
    }

    const double *uab_var_data = (const double *)PyArray_DATA(uab_var_arr);
    const double *inv_ww = ws->inv_ww;
    const double *inv_wy = ws->inv_wy;
    const double *inv_yy = ws->inv_yy;

    double *out_lambdas     = (double *)PyArray_DATA(out.lambdas);
    double *out_logls       = (double *)PyArray_DATA(out.logls);
    double *out_betas       = (double *)PyArray_DATA(out.betas);
    double *out_ses         = (double *)PyArray_DATA(out.ses);
    double *out_pwalds      = (double *)PyArray_DATA(out.pwalds);
    double *out_p_scores    = (double *)PyArray_DATA(out.p_scores);
    double *out_lambdas_mle = (double *)PyArray_DATA(out.lambdas_mle);
    double *out_p_lrts      = (double *)PyArray_DATA(out.p_lrts);

    int n_grid    = ws->n_grid;
    int n_refine  = ws->n_refine;
    int df        = ws->df;
    double reml_const = ws->reml_const;

    /* Clamp n_threads to n_snps */
    int actual_threads = 1;
#ifdef _OPENMP
    actual_threads = (n_threads < n_snps) ? n_threads : n_snps;
    if (actual_threads < 1) actual_threads = 1;
#endif

    /* Per-thread scratch buffers for MLE golden section refinement */
    double **thread_bufs = (double **)calloc((size_t)actual_threads, sizeof(double *));
    int alloc_ok = (thread_bufs != NULL);
    if (alloc_ok) {
        for (int t = 0; t < actual_threads; t++) {
            thread_bufs[t] = alloc_aligned_doubles((size_t)n_samples);
            if (!thread_bufs[t]) { alloc_ok = 0; break; }
        }
    }
    if (!alloc_ok) {
        if (thread_bufs) {
            for (int t = 0; t < actual_threads; t++) free(thread_bufs[t]);
            free(thread_bufs);
        }
        decref_mode4_output(&out);
        PyErr_NoMemory();
        goto err_input;
    }

    Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(actual_threads)
#endif
    for (int snp = 0; snp < n_snps; snp++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *hi_eval_local = thread_bufs[tid];

        const double *snp_base = uab_var_data + (size_t)snp * 3 * n_samples;
        const double *vwx = snp_base;
        const double *vxx = snp_base + (size_t)n_samples;
        const double *vxy = snp_base + (size_t)2 * n_samples;

        /* ---- (a) Score: null-model Pab (single pass, no optimization) ---- */
        {
            double s_wx = 0.0, s_xx = 0.0, s_xy = 0.0;
            #pragma omp simd reduction(+:s_wx,s_xx,s_xy)
            for (int i = 0; i < n_samples; i++) {
                double h = ws->hi_eval_null[i];
                s_wx += h * vwx[i];
                s_xx += h * vxx[i];
                s_xy += h * vxy[i];
            }

            double pab_null[3][6];
            calc_pab_ncvt1_split(ws->null_s_ww, s_wx, ws->null_s_wy,
                                  s_xx, s_xy, ws->null_s_yy, pab_null);

            double score_beta, score_se, score_f;
            int score_valid = score_from_pab(pab_null, n_samples, df,
                                              &score_beta, &score_se, &score_f);

            out_p_scores[snp] = f_to_pvalue(
                score_f, df, score_valid,
                ws->beta_a, ws->beta_b, ws->lbeta_ab);
        }

        /* ---- (b) logdet_iab (same as compute_lmm_chunk_split_c) ---- */
        double iab_s_wx = 0.0, iab_s_xx = 0.0;
        #pragma omp simd reduction(+:iab_s_wx,iab_s_xx)
        for (int i = 0; i < n_samples; i++) {
            iab_s_wx += vwx[i];
            iab_s_xx += vxx[i];
        }

        double iab_p1_xx = iab_s_xx - iab_s_wx * iab_s_wx * ws->iab_inv_ww;
        double logdet_iab = ws->iab_log_ww
                            + ((iab_p1_xx > 0.0) ? log(iab_p1_xx) : 0.0);

        /* ---- (c) Wald: REML optimization ---- */
        double logl_reml, wald_beta, wald_se, wald_f;
        int wald_valid;
        double lambda_reml = golden_section_lambda_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, logdet_iab,
            n_samples, ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->grid_inv, ws->log_l_min, ws->step, n_grid, n_refine,
            df, reml_const, &logl_reml, &wald_beta, &wald_se, &wald_f,
            &wald_valid
        );

        out_lambdas[snp] = lambda_reml;
        out_logls[snp]   = logl_reml;
        out_betas[snp]   = wald_beta;
        out_ses[snp]     = wald_se;
        out_pwalds[snp]  = f_to_pvalue(
            wald_f, df, wald_valid,
            ws->beta_a, ws->beta_b, ws->lbeta_ab);

        /* ---- (d) LRT: MLE optimization ---- */
        double logl_H1;
        double lambda_mle = golden_section_lambda_mle_ncvt1_split(
            vwx, vxx, vxy, inv_ww, inv_wy, inv_yy,
            ws->eigenvalues, n_samples,
            ws->lambda_grid, ws->hi_eval_grid, ws->logdet_h_grid,
            ws->grid_inv, ws->log_l_min, ws->step, n_grid, n_refine,
            ws->mle_const, hi_eval_local, &logl_H1
        );

        out_lambdas_mle[snp] = lambda_mle;

        /* LRT stat = 2*(logl_H1 - logl_H0), clamp >= 0 */
        double lrt_stat = 2.0 * (logl_H1 - ws->logl_H0);
        if (lrt_stat < 0.0) lrt_stat = 0.0;
        out_p_lrts[snp] = chi2_sf_c(lrt_stat);
    }

    Py_END_ALLOW_THREADS

    /* Free per-thread scratch buffers before any Python calls that might
     * raise (warn_betainc_convergence can raise if warnings are errors).
     * Buffers are only used inside the GIL-released compute loop above. */
    for (int t = 0; t < actual_threads; t++) free(thread_bufs[t]);
    free(thread_bufs);
    thread_bufs = NULL;

    if (warn_betainc_convergence(out_betas, out_pwalds, n_snps) < 0)
        goto err_output;

    result = build_mode4_result_dict(&out);
    if (!result) goto err_input;

    Py_DECREF(uab_var_arr);
    return result;

err_output:
    decref_mode4_output(&out);
err_input:
    Py_XDECREF(uab_var_arr);
    return NULL;
}

/* -------------------------------------------------------------------------
 * _get_aligned_alloc_test_ptr
 *
 * Debug function: verify aligned_alloc returns 32-byte-aligned pointers.
 * Returns the pointer value as a Python int for assertion in tests.
 * ------------------------------------------------------------------------- */
static PyObject *_get_aligned_alloc_test_ptr(PyObject *self, PyObject *args)
{
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return NULL;
    }
    double *p = alloc_aligned_doubles((size_t)n);
    if (!p) return PyErr_NoMemory();
    uintptr_t addr = (uintptr_t)p;
    free(p);
    return PyLong_FromUnsignedLongLong((unsigned long long)addr);
}

/* -------------------------------------------------------------------------
 * Module definition
 * ------------------------------------------------------------------------- */
static PyMethodDef methods[] = {
    {
        "compute_lmm_batch_c",
        (PyCFunction)compute_lmm_batch_c,
        METH_VARARGS | METH_KEYWORDS,
        "Batch REML Wald pipeline for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Args:\n"
        "    eigenvalues: (n_samples,) float64 kinship eigenvalues\n"
        "    Uab_batch:   (n_snps, n_samples, 6) float64\n"
        "    Iab_batch:   (n_snps, 3, 6) float64 identity-weighted Pab\n"
        "    n_samples:   int\n"
        "    l_min:       float, minimum lambda\n"
        "    l_max:       float, maximum lambda\n"
        "    n_grid:      int, coarse grid points\n"
        "    n_refine:    int, golden section iterations (>= 1; caller typically passes >= 20)\n"
        "    n_threads:   int, OpenMP thread count\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds — each (n_snps,) float64\n"
    },
    {
        "compute_lmm_batch_split_c",
        (PyCFunction)compute_lmm_batch_split_c,
        METH_VARARGS | METH_KEYWORDS,
        "Split-Uab REML Wald pipeline for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Separates SNP-invariant Uab columns (ww, wy, yy) from varying\n"
        "columns (wx, xx, xy) to halve per-SNP DRAM traffic.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_varying:   (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    Iab_batch:     (n_snps, 3, 6) float64\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    n_threads:     int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds\n"
    },
    {
        "create_workspace_split_c",
        (PyCFunction)create_workspace_split_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create a persistent workspace for the split-Uab REML pipeline.\n"
        "\n"
        "Precomputes lambda_grid, hi_eval_grid, logdet_h_grid, grid_inv, and\n"
        "invariant Iab column sums once per run. The workspace is reused across\n"
        "all chunks — eliminating per-chunk C malloc and grid precomputation.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    n_threads:     int\n"
        "\n"
        "Returns:\n"
        "    PyCapsule wrapping lmm_workspace_t (opaque; pass to compute_lmm_chunk_split_c)\n"
    },
    {
        "compute_lmm_chunk_split_c",
        (PyCFunction)compute_lmm_chunk_split_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Per-chunk REML Wald using a pre-built workspace (no Iab_batch needed).\n"
        "\n"
        "Uses precomputed grids from create_workspace_split_c. logdet_iab is\n"
        "computed internally from raw Uab column sums. OpenMP schedule is static\n"
        "for uniform SNP cost. No per-chunk malloc.\n"
        "\n"
        "Args:\n"
        "    workspace:    PyCapsule from create_workspace_split_c\n"
        "    uab_varying:  (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]\n"
        "    n_threads:    int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds\n"
    },
    {
        "create_workspace_general_c",
        (PyCFunction)create_workspace_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create a persistent workspace for general n_cvt REML pipeline.\n"
        "\n"
        "Precomputes lambda_grid, hi_eval_grid, invariant column sums.\n"
        "Accepts recursion table arrays from build_pab_table_for_c().\n"
    },
    {
        "compute_lmm_chunk_general_c",
        (PyCFunction)compute_lmm_chunk_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Per-chunk REML Wald for general n_cvt using a pre-built workspace.\n"
        "\n"
        "OpenMP parallel over SNPs. Table-driven Pab recursion.\n"
    },
    {
        "compute_lrt_batch_c",
        (PyCFunction)compute_lrt_batch_c,
        METH_VARARGS,
        "Batch LRT for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Per-SNP MLE golden section optimization + chi2_sf p-value.\n"
        "\n"
        "Args:\n"
        "    eigenvalues: (n_samples,) float64\n"
        "    Uab_batch:   (n_snps, n_samples, 6) float64\n"
        "    n_samples:   int\n"
        "    l_min:       float\n"
        "    l_max:       float\n"
        "    n_grid:      int\n"
        "    n_refine:    int\n"
        "    logl_H0:     float, null model MLE log-likelihood\n"
        "    n_threads:   int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "compute_score_batch_c",
        (PyCFunction)compute_score_batch_c,
        METH_VARARGS,
        "Batch Score test for n_cvt=1 with optional OpenMP.\n"
        "\n"
        "Uses fixed null-model Hi_eval (no per-SNP optimization).\n"
        "\n"
        "Args:\n"
        "    eigenvalues:  (n_samples,) float64\n"
        "    Uab_batch:    (n_snps, n_samples, 6) float64\n"
        "    Hi_eval_null: (n_samples,) float64 — null-model weights\n"
        "    n_samples:    int\n"
        "    n_threads:    int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: betas, ses, p_scores — each (n_snps,) float64\n"
    },
    {
        "create_workspace_mode4_split_c",
        (PyCFunction)create_workspace_mode4_split_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create a mode-4 workspace for fused Wald/Score/LRT pipeline.\n"
        "\n"
        "Extends the standard split workspace with null-model Hi_eval,\n"
        "MLE constant, and null log-likelihood for LRT computation.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    n_threads:     int\n"
        "    hi_eval_null:  (n_samples,) float64 — null-model Hi_eval\n"
        "    logl_H0:       float — null MLE log-likelihood\n"
        "\n"
        "Returns:\n"
        "    PyCapsule wrapping lmm_workspace_t (mode=4)\n"
    },
    {
        "compute_mode4_chunk_split_c",
        (PyCFunction)compute_mode4_chunk_split_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Fused per-chunk mode-4 compute: Wald/Score/LRT from SoA split data.\n"
        "\n"
        "Single OpenMP parallel loop produces all 8 output arrays.\n"
        "Requires a mode-4 workspace from create_workspace_mode4_split_c.\n"
        "\n"
        "Args:\n"
        "    workspace:    PyCapsule from create_workspace_mode4_split_c\n"
        "    uab_varying:  (n_snps, 3, n_samples) float64 — SoA [wx, xx, xy]\n"
        "    n_threads:    int\n"
        "\n"
        "Returns:\n"
        "    dict with keys: lambdas, logls, betas, ses, pwalds, p_scores,\n"
        "                    lambdas_mle, p_lrts — each (n_snps,) float64\n"
    },
    {
        "_get_aligned_alloc_test_ptr",
        (PyCFunction)_get_aligned_alloc_test_ptr,
        METH_VARARGS,
        "Debug: return address of an aligned_alloc buffer for alignment testing."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_lmm_accel",
    "C extension: per-SNP REML/MLE pipelines (Wald, Score, LRT, fused mode-4) with OpenMP parallelism (n_cvt=1 + general n_cvt).",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__lmm_accel(void)
{
    import_array();  /* returns NULL on failure (NumPy Python 3 macro) */
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    /* ABI version — Python side checks this to detect stale .so files */
    if (PyModule_AddIntConstant(m, "ABI_VERSION", ABI_VERSION) < 0) { Py_DECREF(m); return NULL; }

    /* Expose whether this .so was compiled with OpenMP support */
#ifdef _OPENMP
    if (PyModule_AddIntConstant(m, "HAS_OPENMP", 1) < 0) { Py_DECREF(m); return NULL; }
#else
    if (PyModule_AddIntConstant(m, "HAS_OPENMP", 0) < 0) { Py_DECREF(m); return NULL; }
#endif

    return m;
}
