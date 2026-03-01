"""Smoke test for the C extension wheel — used by cibuildwheel CIBW_TEST_COMMAND.

Verifies:
1. The compiled C extension imports successfully (ABI match)
2. compute_lmm_batch_c produces finite outputs on synthetic data
"""

import numpy as np

from jamma.lmm._lmm_accel import HAS_OPENMP, compute_lmm_batch_c

print(f"C extension OK, OpenMP={bool(HAS_OPENMP)}")

# Synthetic data: 50 samples, 3 SNPs (batch_size=3 via Uab first axis)
rng = np.random.default_rng(42)
n = 50
n_snps = 3
n_covariates = 1  # intercept-only

# Eigenvalues (sorted positive)
eigenvalues = np.sort(rng.uniform(0.1, 2.0, n))

# Uab: (n_snps, n_samples, ab_cols) — 6 columns for n_cvt=1
# ab_cols = (c+2)(c+3)/2 = 6 for c=1
ab_cols = 6
Uab = rng.standard_normal((n_snps, n, ab_cols))
# Column 0 (yy product) must be positive for valid REML
Uab[:, :, 0] = np.abs(Uab[:, :, 0]) + 0.1

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

# Lambdas should be finite (NaN indicates optimiser failure)
assert np.isfinite(result["lambdas"]).all(), f"Non-finite lambdas: {result['lambdas']}"
print(f"lambdas: {result['lambdas']}")
print("Numerical sanity check passed")
