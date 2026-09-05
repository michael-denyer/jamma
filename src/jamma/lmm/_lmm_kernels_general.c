/*
 * _lmm_kernels_general.c — see _lmm_kernels_general.h.
 *
 * Pure arithmetic — see the header. Nothing here touches CPython, so it
 * needs none of _lmm_support.h's import_array() handling.
 */

#include "_lmm_kernels_general.h"

/* wald_from_pab_general lives with the other statistic extractors in
 * _lmm_stats.c; the fused-general REML path calls it once per SNP. */
#include "_lmm_stats.h"
/* logdet_h_lambda: the logdet(H) term every REML and MLE evaluation needs. */
#include "_lmm_logdet.h"

#include <math.h>
#include <float.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * calc_pab_general — Table-driven Pab recursion for arbitrary n_cvt.
 *
 * Row 0 from row0 array (dot product sums), rows 1..n_rows-1 from entries.
 * Output in pab[n_rows * n_index], row-major.
 * ------------------------------------------------------------------------- */
void calc_pab_general(
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
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
)
{
    int ni = t->n_index;
    int n_var = t->n_var;

    /* Compute varying dot products (reuse tail of row0 as temp) */
    double var_sums[MAX_N_INDEX];
    for (int c = 0; c < n_var; c++) var_sums[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double h = hi_eval[i];
        for (int c = 0; c < n_var; c++)
            var_sums[c] += h * uab_var[c * n_samples + i];
    }

    /* Reconstruct row 0 */
    for (int i = 0; i < ni; i++) row0[i] = 0.0;
    for (int c = 0; c < t->n_inv; c++)
        row0[t->invariant_indices[c]] = inv_sums_cached[c];
    for (int c = 0; c < n_var; c++)
        row0[t->varying_indices[c]] = var_sums[c];

    /* Full Pab via recursion */
    calc_pab_general(row0, t, pab_scratch);

    return reml_finish_general(pab_scratch, t, logdet_h, logdet_iab, reml_const);
}

/* -------------------------------------------------------------------------
 * reml_logl_general_fresh — Full REML evaluation for a specific lambda.
 *
 * Computes hi_eval + all dot products in a single n_samples pass (fused
 * loop), logdet_h in a second pass over the eigenvalues alone (see
 * _lmm_logdet.h), then calc_pab + reml_finish.
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
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
)
{
    int ni = t->n_index;
    int n_inv = t->n_inv;
    int n_var = t->n_var;

    double logdet_h = logdet_h_lambda(eigenvalues, n_samples, lambda);
    double inv_sums[MAX_N_INDEX];
    double var_sums[MAX_N_INDEX];
    for (int c = 0; c < n_inv; c++) inv_sums[c] = 0.0;
    for (int c = 0; c < n_var; c++) var_sums[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        for (int c = 0; c < n_inv; c++)
            inv_sums[c] += h * uab_inv[c * n_samples + i];
        for (int c = 0; c < n_var; c++)
            var_sums[c] += h * uab_var[c * n_samples + i];
    }

    /* Reconstruct row 0 */
    for (int i = 0; i < ni; i++) row0[i] = 0.0;
    for (int c = 0; c < n_inv; c++)
        row0[t->invariant_indices[c]] = inv_sums[c];
    for (int c = 0; c < n_var; c++)
        row0[t->varying_indices[c]] = var_sums[c];

    /* Full Pab via recursion */
    calc_pab_general(row0, t, pab_scratch);

    return reml_finish_general(pab_scratch, t, logdet_h, logdet_iab, reml_const);
}

static double reml_score_loglambda_general(
    const double *uab_inv, const double *uab_var, const double *eigenvalues,
    int n_samples, double lambda, const pab_table_t *t,
    double *row0, double *pab, double *dpab)
{
    int ni = t->n_index;
    double crow0[MAX_N_INDEX], cdrow0[MAX_N_INDEX];
    double *drow0 = dpab;
    memset(dpab, 0, (size_t)t->n_rows * ni * sizeof(double));
    for (int c = 0; c < ni; c++) {
        row0[c] = 0.0; drow0[c] = 0.0;
        crow0[c] = 0.0; cdrow0[c] = 0.0;
    }
    double trace = 0.0, ctrace = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double d = eigenvalues[i];
        double h = 1.0 / (1.0 + lambda * d);
        double dh = -lambda * d * h * h;
        double trace_value = lambda * d * h - ctrace;
        double trace_next = trace + trace_value;
        ctrace = (trace_next - trace) - trace_value;
        trace = trace_next;
        for (int c = 0; c < t->n_inv; c++) {
            int index = t->invariant_indices[c];
            double value = uab_inv[c * n_samples + i];
            double term = h * value - crow0[index];
            double next = row0[index] + term;
            crow0[index] = (next - row0[index]) - term;
            row0[index] = next;
            term = dh * value - cdrow0[index];
            next = drow0[index] + term;
            cdrow0[index] = (next - drow0[index]) - term;
            drow0[index] = next;
        }
        for (int c = 0; c < t->n_var; c++) {
            int index = t->varying_indices[c];
            double value = uab_var[c * n_samples + i];
            double term = h * value - crow0[index];
            double next = row0[index] + term;
            crow0[index] = (next - row0[index]) - term;
            row0[index] = next;
            term = dh * value - cdrow0[index];
            next = drow0[index] + term;
            cdrow0[index] = (next - drow0[index]) - term;
            drow0[index] = next;
        }
    }
    calc_pab_general(row0, t, pab);
    for (int p = 1; p < t->n_rows; p++) {
        int offset = t->level_offsets[p], count = t->level_counts[p];
        for (int e = 0; e < count; e++) {
            const pab_entry_t *re = &t->entries[offset + e];
            int prev = (p - 1) * ni, out = p * ni;
            double q = pab[prev + re->index_ww];
            if (q == 0.0) continue;
            double aw = pab[prev + re->index_aw];
            double bw = pab[prev + re->index_bw];
            dpab[out + re->index_ab] = dpab[prev + re->index_ab]
                - (dpab[prev + re->index_aw] * bw
                   + aw * dpab[prev + re->index_bw]) / q
                + aw * bw * dpab[prev + re->index_ww] / (q * q);
        }
    }
    double score = -0.5 * trace;
    for (int d = 0; d < t->n_cvt + 1; d++) {
        int index = t->logdet_diag_rows[d] * ni + t->logdet_diag_cols[d];
        if (!(pab[index] > 0.0)) return NAN;
        score -= 0.5 * dpab[index] / pab[index];
    }
    int yy = (t->n_cvt + 1) * ni + t->idx_yy;
    if (!(pab[yy] > P_YY_MIN)) return NAN;
    score -= 0.5 * t->df * dpab[yy] / pab[yy];
    return score;
}


/* -------------------------------------------------------------------------
 * golden_section_lambda_general — Grid + golden section for general n_cvt.
 *
 * Mirrors the coarse-grid plus refine_lambda_ncvt1_split pair. Grid phase uses
 * precomputed hi_eval + invariant sums; refinement uses fresh evaluation.
 * At optimal lambda, computes full Pab and returns it + Wald stats.
 * ------------------------------------------------------------------------- */
double golden_section_lambda_general(
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
    int *is_valid_out,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch,
    double *dpab_scratch
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
            t,
            row0, pab_scratch
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
    const double coarse_a = a, coarse_b = b;

    /* Stage 2: golden section refinement (fresh evaluation) */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = reml_logl_general_fresh(
        uab_inv, uab_var, eigenvalues, n_samples, exp(c),
        logdet_iab, reml_const, t, row0, pab_scratch);
    double fd = reml_logl_general_fresh(
        uab_inv, uab_var, eigenvalues, n_samples, exp(d),
        logdet_iab, reml_const, t, row0, pab_scratch);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = reml_logl_general_fresh(
                uab_inv, uab_var, eigenvalues, n_samples, exp(c),
                logdet_iab, reml_const, t, row0, pab_scratch);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = reml_logl_general_fresh(
                uab_inv, uab_var, eigenvalues, n_samples, exp(d),
                logdet_iab, reml_const, t, row0, pab_scratch);
        }
    }

    double log_opt = (a + b) / 2.0;
    /* Refine enclosed peaks independently of rounded objective ties. */
    if (a > coarse_a && b < coarse_b) {
        double delta = fmin(1e-3, 0.25 * (coarse_b - coarse_a));
        delta = fmin(delta, 0.5 * (log_opt - coarse_a));
        delta = fmin(delta, 0.5 * (coarse_b - log_opt));
        double score = reml_score_loglambda_general(
            uab_inv, uab_var, eigenvalues, n_samples, exp(log_opt), t,
            row0, pab_scratch, dpab_scratch);
        double sm = reml_score_loglambda_general(
            uab_inv, uab_var, eigenvalues, n_samples, exp(log_opt - delta), t,
            row0, pab_scratch, dpab_scratch);
        double sp = reml_score_loglambda_general(
            uab_inv, uab_var, eigenvalues, n_samples, exp(log_opt + delta), t,
            row0, pab_scratch, dpab_scratch);
        double curvature = (sp - sm) / (2.0 * delta);
        if (isfinite(delta) && delta > 0.0 && isfinite(score)
            && isfinite(curvature) && curvature < 0.0) {
            double candidate = log_opt - score / curvature;
            if (isfinite(candidate) && candidate >= coarse_a && candidate <= coarse_b) {
                double candidate_score = reml_score_loglambda_general(
                    uab_inv, uab_var, eigenvalues, n_samples, exp(candidate), t,
                    row0, pab_scratch, dpab_scratch);
                if (isfinite(candidate_score) && fabs(candidate_score) < fabs(score))
                    log_opt = candidate;
            }
        }
    }
    double lambda_opt = exp(log_opt);

    /* Final evaluation: reml_logl_general_fresh fills pab_scratch as a side
     * effect (the caller's own buffer), so the Wald extraction below reads
     * the same Pab the logl was computed from without a second pass. */
    *logl_out = reml_logl_general_fresh(
        uab_inv, uab_var, eigenvalues, n_samples, lambda_opt,
        logdet_iab, reml_const, t, row0, pab_scratch);
    *is_valid_out = wald_from_pab_general(
        pab_scratch, t, beta_out, se_out, f_stat_out);

    return lambda_opt;
}


/* -------------------------------------------------------------------------
 * mle_logl_general — MLE log-likelihood for one SNP at one lambda (general n_cvt).
 *
 * MLE formula: -0.5 * n * log(P_yy_full) - 0.5 * logdet_h + mle_const
 * where P_yy_full is at level n_cvt+1 (fully projected).
 *
 * Uses full Uab row (n_samples * n_index) in AoS layout.
 * ------------------------------------------------------------------------- */
static double mle_logl_general(
    const double *uab_snp,     /* (n_samples, n_index) row-major */
    const double *eigenvalues,
    int n_samples,
    double lambda,
    double mle_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
)
{
    int ni = t->n_index;

    double logdet_h = logdet_h_lambda(eigenvalues, n_samples, lambda);
    for (int c = 0; c < ni; c++) row0[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double v = lambda * eigenvalues[i] + 1.0;
        double h = 1.0 / v;
        for (int c = 0; c < ni; c++)
            row0[c] += h * uab_snp[i * ni + c];
    }

    calc_pab_general(row0, t, pab_scratch);

    /* P_yy_full at level n_cvt+1 (fully projected) */
    int nc = t->n_cvt;
    double P_yy = pab_scratch[(nc + 1) * ni + t->idx_yy];
    if (P_yy < 0.0) return (double)NAN;
    if (P_yy < P_YY_MIN) P_yy = P_YY_MIN;

    return mle_const - 0.5 * logdet_h - 0.5 * (double)n_samples * log(P_yy);
}

/* -------------------------------------------------------------------------
 * mle_logl_general_cached — MLE using cached hi_eval for coarse grid search.
 * ------------------------------------------------------------------------- */
static double mle_logl_general_cached(
    const double *uab_snp,
    const double *cached_hi_eval,
    double cached_logdet_h,
    int n_samples,
    double mle_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
)
{
    int ni = t->n_index;

    for (int c = 0; c < ni; c++) row0[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double h = cached_hi_eval[i];
        for (int c = 0; c < ni; c++)
            row0[c] += h * uab_snp[i * ni + c];
    }

    calc_pab_general(row0, t, pab_scratch);

    int nc = t->n_cvt;
    double P_yy = pab_scratch[(nc + 1) * ni + t->idx_yy];
    if (P_yy < 0.0) return (double)NAN;
    if (P_yy < P_YY_MIN) P_yy = P_YY_MIN;

    return mle_const - 0.5 * cached_logdet_h - 0.5 * (double)n_samples * log(P_yy);
}

/* -------------------------------------------------------------------------
 * golden_section_lambda_mle_general — Grid + golden section for MLE (general n_cvt).
 *
 * Mirrors golden_section_lambda_mle_ncvt1 but uses mle_logl_general.
 * Returns optimal lambda; writes logl to *logl_out.
 * ------------------------------------------------------------------------- */
double golden_section_lambda_mle_general(
    const double *uab_snp,
    const double *eigenvalues,
    int n_samples,
    const double *lambda_grid,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    double log_l_min, double step,
    int n_grid, int n_refine,
    double mle_const,
    const pab_table_t *t,
    double *logl_out,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
)
{
    const double phi = 0.6180339887498949;

    /* Stage 1: coarse grid search using cached hi_eval */
    double best_logl = REML_SENTINEL;
    int best_idx = 0;
    for (int g = 0; g < n_grid; g++) {
        double logl = mle_logl_general_cached(
            uab_snp,
            hi_eval_grid + (size_t)g * n_samples,
            logdet_h_grid[g],
            n_samples, mle_const, t,
            row0, pab_scratch
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }

    if (best_logl == REML_SENTINEL) {
        *logl_out = (double)NAN;
        return (double)NAN;
    }

    /* Bracket */
    int idx_low  = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < n_grid - 1) ? best_idx + 1 : n_grid - 1;
    double a = log_l_min + idx_low * step;
    double b = log_l_min + idx_high * step;

    /* Stage 2: golden section refinement */
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = mle_logl_general(uab_snp, eigenvalues, n_samples, exp(c), mle_const, t,
                                  row0, pab_scratch);
    double fd = mle_logl_general(uab_snp, eigenvalues, n_samples, exp(d), mle_const, t,
                                  row0, pab_scratch);

    for (int iter = 0; iter < n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = mle_logl_general(uab_snp, eigenvalues, n_samples, exp(c), mle_const, t,
                                   row0, pab_scratch);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = mle_logl_general(uab_snp, eigenvalues, n_samples, exp(d), mle_const, t,
                                   row0, pab_scratch);
        }
    }

    double log_opt = (a + b) / 2.0;
    double lambda_opt = exp(log_opt);
    *logl_out = mle_logl_general(uab_snp, eigenvalues, n_samples, lambda_opt, mle_const, t,
                                  row0, pab_scratch);

    return lambda_opt;
}
