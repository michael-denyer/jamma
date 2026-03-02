"""Smoke test for the C extension wheel — used by cibuildwheel CIBW_TEST_COMMAND.

Verifies:
1. The compiled C extension imports successfully (ABI match)
2. compute_lmm_batch_c produces finite outputs on synthetic data
"""

import numpy as np

from jamma.lmm._lmm_accel import HAS_OPENMP, compute_lmm_batch_c

print(f"C extension OK, OpenMP={bool(HAS_OPENMP)}")

# Synthetic data: 50 samples, 4 SNPs (3 normal + 1 degenerate)
rng = np.random.default_rng(42)
n = 50
n_snps = 4
n_covariates = 1  # intercept-only

# Eigenvalues (sorted positive)
eigenvalues = np.sort(rng.uniform(0.1, 2.0, n))

# Uab: (n_snps, n_samples, ab_cols) — 6 columns for n_cvt=1
# ab_cols = (c+2)(c+3)/2 = 6 for c=1
ab_cols = 6
Uab = rng.standard_normal((n_snps, n, ab_cols))
# Column 0 (yy product) must be positive for valid REML
Uab[:, :, 0] = np.abs(Uab[:, :, 0]) + 0.1

# SNP 3: degenerate (constant genotype) — xx column = 0
# This tests the is_valid flag and make_quiet_nan() under -ffinite-math-only
Uab[3, :, 3] = 0.0  # xx = 0 -> P_XX <= 0 -> degenerate

# Iab: (n_snps, 3, ab_cols) — precomputed CalcPab invariants
# 3 levels: base sums, first elimination, second elimination
Iab = np.zeros((n_snps, 3, ab_cols))
# Level 0: column sums of Uab
Iab[:, 0, :] = Uab.sum(axis=1)
# Level 1: partial elimination (Pab hierarchy)
Iab[:, 1, 3] = Iab[:, 0, 3] - Iab[:, 0, 1] ** 2 / np.maximum(Iab[:, 0, 0], 1e-10)
Iab[:, 1, 4] = Iab[:, 0, 4] - Iab[:, 0, 1] * Iab[:, 0, 2] / np.maximum(
    Iab[:, 0, 0], 1e-10
)
Iab[:, 1, 5] = Iab[:, 0, 5] - Iab[:, 0, 2] ** 2 / np.maximum(Iab[:, 0, 0], 1e-10)
# Level 2: final elimination
Iab[:, 2, 5] = Iab[:, 1, 5] - Iab[:, 1, 4] ** 2 / np.maximum(Iab[:, 1, 3], 1e-10)

result = compute_lmm_batch_c(eigenvalues, Uab, Iab, n, 1e-5, 1e5, 50, 20, n_covariates)

# First 3 SNPs: lambdas should be finite
assert np.isfinite(result["lambdas"][:3]).all(), (
    f"Non-finite lambdas for normal SNPs: {result['lambdas'][:3]}"
)

# SNP 3 (degenerate): beta, se, pwald should be NaN
assert np.isnan(result["betas"][3]), (
    f"Degenerate SNP beta not NaN: {result['betas'][3]}"
)
assert np.isnan(result["ses"][3]), f"Degen SNP se not NaN: {result['ses'][3]}"
assert np.isnan(result["pwalds"][3]), (
    f"Degenerate SNP pwald not NaN: {result['pwalds'][3]}"
)

print(f"lambdas: {result['lambdas']}")
print("Degenerate SNP correctly produced NaN beta/se/pwald")
print("Numerical sanity check passed")
