"""Inputs for the historical Pab benchmark and its shared-input control."""

import numpy as np


def benchmark_inputs(n_samples=500, n_snps=2000, *, shared=False):
    """Retain the original RNG stream; share w/y only in the control."""
    rng = np.random.default_rng(42)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    w = np.abs(rng.standard_normal((n_snps, n_samples))) + 1.0
    x = np.abs(rng.standard_normal((n_snps, n_samples))) + 0.5
    y = rng.standard_normal((n_snps, n_samples))
    if shared:
        w[:] = w[0]
        y[:] = y[0]
    return eigenvalues, w, x, y


def gram_products(w, x, y):
    return np.stack([w * w, w * x, w * y, x * x, x * y, y * y], axis=2)


def numpy_routes(eigenvalues, w, x, y):
    from jamma.lmm.compute_numpy import _compute_wald_numpy
    from jamma.lmm.likelihood_numpy import golden_section_optimize_lambda_numpy
    from jamma.lmm.stats import batch_calc_wald_stats_from_pab_numpy
    from jamma.lmm.uab import batch_compute_iab_numpy

    uab = gram_products(w, x, y)
    iab = batch_compute_iab_numpy(1, uab)
    split = _compute_wald_numpy(
        1, eigenvalues, uab, len(eigenvalues), 1e-5, 1e5, 50, 20, Iab_batch=iab
    )
    lam, logl, pab = golden_section_optimize_lambda_numpy(1, eigenvalues, uab, iab)
    beta, se, p = batch_calc_wald_stats_from_pab_numpy(1, pab, len(eigenvalues))
    return split, {
        "lambdas": lam,
        "logls": logl,
        "betas": beta,
        "ses": se,
        "pwalds": p,
    }


def reduced_inputs():
    """Smallest identifiable n_cvt=1 model, two SNPs expose shared-input misuse."""
    ev, w, x, y = benchmark_inputs()
    rows = [0, 1, 498, 499]
    return ev[rows].copy(), w[:2, rows].copy(), x[:2, rows].copy(), y[:2, rows].copy()
