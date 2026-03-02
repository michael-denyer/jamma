"""Smoke test for the C extension wheel — used by cibuildwheel CIBW_TEST_COMMAND.

Verifies:
1. The compiled C extension imports successfully (ABI match)
2. compute_lmm_batch_c produces finite outputs on synthetic data
3. Degenerate SNPs (P_XX <= 0) produce NaN beta/se/pwald via is_valid flag
"""

import sys

import numpy as np

try:
    from jamma.lmm._lmm_accel import HAS_OPENMP, compute_lmm_batch_c
except ImportError as exc:
    print(f"FAIL: C extension import failed (ABI mismatch?): {exc}", file=sys.stderr)
    sys.exit(1)

print(f"C extension OK, OpenMP={bool(HAS_OPENMP)}")

# Synthetic data: 50 samples, 4 SNPs (3 normal + 1 degenerate)
rng = np.random.default_rng(42)
n = 50
n_snps = 4
n_threads = 1  # single-threaded for CI smoke test

# Eigenvalues (sorted positive)
eigenvalues = np.sort(rng.uniform(0.1, 2.0, n))

# Uab: (n_snps, n_samples, ab_cols) — 6 columns for n_cvt=1
# ab_cols = (c+2)(c+3)/2 = 6 for c=1
ab_cols = 6
Uab = rng.standard_normal((n_snps, n, ab_cols))
# Column 0 (ww product) must be positive for valid Pab W-projection
Uab[:, :, 0] = np.abs(Uab[:, :, 0]) + 0.1

# SNP 3: degenerate (constant genotype) — xx column = 0
# This tests the is_valid flag: P_XX <= 0 -> beta/se/pwald = NaN
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

result = compute_lmm_batch_c(eigenvalues, Uab, Iab, n, 1e-5, 1e5, 50, 20, n_threads)

# First 3 SNPs: lambdas should be finite
if not np.isfinite(result["lambdas"][:3]).all():
    print(f"FAIL: Non-finite lambdas for normal SNPs: {result['lambdas'][:3]}")
    sys.exit(1)

# SNP 3 (degenerate): beta, se, pwald should be NaN
if not np.isnan(result["betas"][3]):
    print(f"FAIL: Degenerate SNP beta not NaN: {result['betas'][3]}")
    sys.exit(1)
if not np.isnan(result["ses"][3]):
    print(f"FAIL: Degenerate SNP se not NaN: {result['ses'][3]}")
    sys.exit(1)
if not np.isnan(result["pwalds"][3]):
    print(f"FAIL: Degenerate SNP pwald not NaN: {result['pwalds'][3]}")
    sys.exit(1)

print(f"lambdas: {result['lambdas']}")
print("Degenerate SNP correctly produced NaN beta/se/pwald")
print("Numerical sanity check passed")
