/*
 * _lmm_lambda_search.h. The one lambda optimiser every refiner drives.
 *
 * Each family's REML and MLE refiner picks a coarse-grid index, then hands
 * this driver an objective over lambda and an opaque context holding that
 * SNP's data. The bracket rule, the golden-section update, and the Newton
 * polish of an enclosed REML peak live here once.
 *
 * Header-only and static inline. Every refiner passes a constant pointer to
 * a static objective in its own translation unit, so the compiler can inline
 * the objective into the ~70 evaluations per SNP.
 */

#ifndef JAMMA_LMM_LAMBDA_SEARCH_H
#define JAMMA_LMM_LAMBDA_SEARCH_H

#include <math.h>

/* The coarse grid a search refines within: n_grid points evenly spaced in
 * log(lambda) from log_l_min, then n_refine golden-section iterations. */
typedef struct {
    double *lambda_grid;    /* (n_grid,) */
    double log_l_min, step;
    int n_grid, n_refine;
} lambda_search_t;

typedef double (*lambda_objective_fn)(const void *ctx, double lambda);

/* Log-lambda optimum of objective within the grid cells either side of
 * best_idx, which must be a valid index (callers handle a fully degenerate
 * SNP before calling). When score is non-NULL it is the analytic
 * d(objective)/d(log lambda), and up to three safeguarded Newton steps polish
 * a peak the golden section left strictly inside the coarse bracket. */
static inline double golden_section_log_lambda(
    lambda_objective_fn objective, lambda_objective_fn score, const void *ctx,
    const lambda_search_t *search, int best_idx)
{
    const double phi = 0.6180339887498949;
    int idx_low = (best_idx > 0) ? best_idx - 1 : 0;
    int idx_high = (best_idx < search->n_grid - 1) ? best_idx + 1 : search->n_grid - 1;
    double a = search->log_l_min + idx_low * search->step;
    double b = search->log_l_min + idx_high * search->step;
    const double coarse_a = a, coarse_b = b;

    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = objective(ctx, exp(c));
    double fd = objective(ctx, exp(d));
    for (int iter = 0; iter < search->n_refine; iter++) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = objective(ctx, exp(c));
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = objective(ctx, exp(d));
        }
    }

    double log_opt = (a + b) / 2.0;
    if (!score) return log_opt;
    /* Rounded objective ties can stall the bracket short of the peak. */
    for (int step = 0; step < 3 && a > coarse_a && b < coarse_b; step++) {
        double delta = fmin(1e-3, 0.25 * (coarse_b - coarse_a));
        delta = fmin(delta, 0.5 * (log_opt - coarse_a));
        delta = fmin(delta, 0.5 * (coarse_b - log_opt));
        double s = score(ctx, exp(log_opt));
        double sm = score(ctx, exp(log_opt - delta));
        double sp = score(ctx, exp(log_opt + delta));
        double curvature = (sp - sm) / (2.0 * delta);
        if (isfinite(delta) && delta > 0.0 && isfinite(s)
            && isfinite(curvature) && curvature < 0.0) {
            double candidate = log_opt - s / curvature;
            if (isfinite(candidate) && candidate >= coarse_a && candidate <= coarse_b) {
                double candidate_score = score(ctx, exp(candidate));
                if (isfinite(candidate_score) && fabs(candidate_score) < fabs(s)) {
                    log_opt = candidate;
                    /* Score magnitude must be scaled by the local curvature. */
                    if (fabs(candidate_score / curvature) > 1e-10)
                        continue;
                }
            }
        }
        break;
    }
    return log_opt;
}

#endif /* JAMMA_LMM_LAMBDA_SEARCH_H */
