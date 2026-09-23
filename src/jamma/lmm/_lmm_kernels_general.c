/*
 * _lmm_kernels_general.c — see _lmm_kernels_general.h.
 *
 * Pure arithmetic — see the header. Nothing here touches CPython, so it
 * needs none of _lmm_support.h's import_array() handling.
 */

#include "_lmm_kernels_general.h"

#include "_lmm_stats.h"
/* logdet_h_lambda: the logdet(H) term every REML and MLE evaluation needs. */
#include "_lmm_logdet.h"
#include "_lmm_lambda_search.h"

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
             * so Px_yy < 0 guard in wald_stats catches degeneracy. */
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

    int nc_total = t->n_cvt + 1;
    double P_yy = replace_zero_p_yy(pab[nc_total * ni + t->idx_yy]);
    if (P_yy < 0.0) P_yy = (double)NAN;

    return reml_const - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * log(P_yy);
}

/* Row 0 at a cached grid point: the grid's invariant sums, plus the varying
 * columns weighted by that point's Hi_eval. */
static void row0_general_cached(
    const double *inv_sums_cached,
    const double *uab_var,
    const double *hi_eval,
    int n_samples,
    const pab_table_t *t,
    double *row0
)
{
    int n_var = t->n_var;
    double var_sums[MAX_N_INDEX];
    for (int c = 0; c < n_var; c++) var_sums[c] = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double h = hi_eval[i];
        for (int c = 0; c < n_var; c++)
            var_sums[c] += h * uab_var[c * n_samples + i];
    }

    for (int i = 0; i < t->n_index; i++) row0[i] = 0.0;
    for (int c = 0; c < t->n_inv; c++)
        row0[t->invariant_indices[c]] = inv_sums_cached[c];
    for (int c = 0; c < n_var; c++)
        row0[t->varying_indices[c]] = var_sums[c];
}

/* Row 0 at an SNP-specific lambda: one fused pass over the samples sums the
 * invariant and varying columns into snp->row0. */
static void row0_general_fresh(const general_snp_t *snp, double lambda)
{
    const pab_table_t *t = snp->t;
    const double *uab_inv = snp->uab_inv;
    const double *uab_var = snp->uab_var;
    const double *eigenvalues = snp->eigenvalues;
    int n_samples = snp->n_samples;
    int n_inv = t->n_inv;
    int n_var = t->n_var;
    double *row0 = snp->row0;

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

    for (int i = 0; i < t->n_index; i++) row0[i] = 0.0;
    for (int c = 0; c < n_inv; c++)
        row0[t->invariant_indices[c]] = inv_sums[c];
    for (int c = 0; c < n_var; c++)
        row0[t->varying_indices[c]] = var_sums[c];
}

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
    row0_general_cached(inv_sums_cached, uab_var, hi_eval, n_samples, t, row0);
    calc_pab_general(row0, t, pab_scratch);
    return reml_finish_general(pab_scratch, t, logdet_h, logdet_iab, reml_const);
}

/* Full REML evaluation at an SNP-specific lambda, for golden-section
 * refinement. logdet_h is a second pass over the eigenvalues alone (see
 * _lmm_logdet.h). */
static double reml_logl_general_fresh(const void *ctx, double lambda)
{
    const general_snp_t *snp = (const general_snp_t *)ctx;
    double logdet_h = logdet_h_lambda(snp->eigenvalues, snp->n_samples, lambda);
    row0_general_fresh(snp, lambda);
    calc_pab_general(snp->row0, snp->t, snp->pab);
    return reml_finish_general(snp->pab, snp->t, logdet_h, snp->logdet_iab,
                               snp->reml_const);
}

static double reml_score_loglambda_general(const void *ctx, double lambda)
{
    const general_snp_t *snp = (const general_snp_t *)ctx;
    const pab_table_t *t = snp->t;
    const double *uab_inv = snp->uab_inv;
    const double *uab_var = snp->uab_var;
    const double *eigenvalues = snp->eigenvalues;
    int n_samples = snp->n_samples;
    double *row0 = snp->row0, *pab = snp->pab, *dpab = snp->dpab;
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
    if (!(pab[yy] > 0.0)) return NAN;
    score -= 0.5 * t->df * dpab[yy] / pab[yy];
    return score;
}


int coarse_grid_reml_general(
    const general_snp_t *snp,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const double *inv_sums_grid,
    int n_grid
)
{
    const pab_table_t *t = snp->t;
    double best_logl = REML_SENTINEL;
    int best_idx = -1;
    for (int g = 0; g < n_grid; g++) {
        double logl = reml_logl_general_cached(
            inv_sums_grid + (size_t)g * t->n_inv,
            snp->uab_var,
            hi_eval_grid + (size_t)g * snp->n_samples,
            snp->n_samples,
            logdet_h_grid[g],
            snp->logdet_iab,
            snp->reml_const,
            t,
            snp->row0, snp->pab
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }
    return best_idx;
}

double refine_lambda_general(
    const general_snp_t *snp,
    const lambda_search_t *search,
    int best_idx,
    double *logl_out,
    double *beta_out, double *se_out, double *f_stat_out,
    int *is_valid_out
)
{
    /* Fully degenerate SNP */
    if (best_idx < 0) {
        *logl_out    = (double)NAN;
        *beta_out    = (double)NAN;
        *se_out      = (double)NAN;
        *f_stat_out  = (double)NAN;
        *is_valid_out = 0;
        return search->lambda_grid[0];
    }

    double lambda_opt = exp(golden_section_log_lambda(
        reml_logl_general_fresh, reml_score_loglambda_general, snp,
        search, best_idx));

    /* Final evaluation: reml_logl_general_fresh fills snp->pab as a side
     * effect (the caller's own buffer), so the Wald extraction below reads
     * the same Pab the logl was computed from without a second pass. */
    *logl_out = reml_logl_general_fresh(snp, lambda_opt);
    *is_valid_out = wald_stats(pab_terms_general(snp->pab, snp->t),
                               snp->t->df, beta_out, se_out, f_stat_out);

    return lambda_opt;
}


/* MLE logl from a fully projected Pab:
 * mle_const - 0.5 * logdet_h - 0.5 * n * log(P_yy), with P_yy at level n_cvt+1. */
static double mle_finish_general(
    const double *pab, const pab_table_t *t, int n_samples,
    double logdet_h, double mle_const)
{
    double P_yy = replace_zero_p_yy(pab[(t->n_cvt + 1) * t->n_index + t->idx_yy]);
    if (P_yy < 0.0) return (double)NAN;
    return mle_const - 0.5 * logdet_h - 0.5 * (double)n_samples * log(P_yy);
}

static double mle_logl_general(const void *ctx, double lambda)
{
    const general_snp_t *snp = (const general_snp_t *)ctx;
    double logdet_h = logdet_h_lambda(snp->eigenvalues, snp->n_samples, lambda);
    row0_general_fresh(snp, lambda);
    calc_pab_general(snp->row0, snp->t, snp->pab);
    return mle_finish_general(snp->pab, snp->t, snp->n_samples, logdet_h,
                              snp->mle_const);
}

static double mle_logl_general_cached(
    const double *inv_sums_cached,
    const double *uab_var,
    const double *hi_eval,
    int n_samples,
    double logdet_h,
    double mle_const,
    const pab_table_t *t,
    double *row0,          /* caller-provided, at least n_index doubles */
    double *pab_scratch    /* caller-provided, at least n_rows * n_index doubles */
)
{
    row0_general_cached(inv_sums_cached, uab_var, hi_eval, n_samples, t, row0);
    calc_pab_general(row0, t, pab_scratch);
    return mle_finish_general(pab_scratch, t, n_samples, logdet_h, mle_const);
}

int coarse_grid_mle_general(
    const general_snp_t *snp,
    const double *hi_eval_grid,
    const double *logdet_h_grid,
    const double *inv_sums_grid,
    int n_grid
)
{
    double best_logl = REML_SENTINEL;
    int best_idx = -1;
    for (int g = 0; g < n_grid; g++) {
        double logl = mle_logl_general_cached(
            inv_sums_grid + (size_t)g * snp->t->n_inv,
            snp->uab_var,
            hi_eval_grid + (size_t)g * snp->n_samples,
            snp->n_samples,
            logdet_h_grid[g],
            snp->mle_const,
            snp->t,
            snp->row0, snp->pab
        );
        if (isnan(logl)) logl = REML_SENTINEL;
        if (logl > best_logl) {
            best_logl = logl;
            best_idx = g;
        }
    }
    return best_idx;
}

double refine_lambda_mle_general(
    const general_snp_t *snp,
    const lambda_search_t *search,
    int best_idx,
    double *logl_out
)
{
    if (best_idx < 0) {
        *logl_out = (double)NAN;
        return (double)NAN;
    }

    double lambda_opt = exp(golden_section_log_lambda(
        mle_logl_general, NULL, snp, search, best_idx));
    *logl_out = mle_logl_general(snp, lambda_opt);

    return lambda_opt;
}
