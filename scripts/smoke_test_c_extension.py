"""Smoke test for the C extension wheel — used by cibuildwheel CIBW_TEST_COMMAND.

Verifies:
1. The compiled C extension imports successfully (ABI match)
2. The fused Wald kernel produces finite outputs on synthetic data
3. Degenerate SNPs (P_XX <= 0) produce NaN beta/se/pwald via is_valid flag

Drives create_workspace_fused_c and compute_lmm_chunk_fused_c, which is what
DispatchPath.FUSED resolves to for n_cvt=1. It used to drive
compute_lmm_batch_c, an entry point no dispatch path selected, so a wheel could
have passed this while shipping a broken production kernel.
"""

import sys

import numpy as np

try:
    from jamma.lmm._lmm_accel import (
        HAS_OPENMP,
        compute_lmm_chunk_fused_c,
        create_workspace_fused_c,
    )
except ImportError as exc:
    print(f"FAIL: C extension import failed (ABI mismatch?): {exc}", file=sys.stderr)
    sys.exit(1)

print(f"C extension OK, OpenMP={bool(HAS_OPENMP)}")

# Synthetic data: 50 samples, 4 SNPs (3 normal + 1 degenerate)
rng = np.random.default_rng(42)
n = 50
n_snps = 4
n_threads = 1  # single-threaded for CI smoke test

eigenvalues = np.sort(rng.uniform(0.1, 2.0, n))

# The fused kernel builds Uab itself from the rotated covariate, phenotype and
# genotypes, so it takes those rather than a prebuilt Uab.
w = rng.standard_normal(n)
Uty = rng.standard_normal(n)
utg_t = rng.standard_normal((n_snps, n))

# SNP 3: constant genotype, which rotates to an all-zero UtG column and so
# drives P_XX to zero. This tests the is_valid flag.
utg_t[3, :] = 0.0

# Invariant SoA, rows [ww, wy, yy].
uab_inv_soa = np.empty((3, n), dtype=np.float64)
uab_inv_soa[0] = w * w
uab_inv_soa[1] = w * Uty
uab_inv_soa[2] = Uty * Uty

ws = create_workspace_fused_c(
    eigenvalues, uab_inv_soa, w, Uty, n, 1e-5, 1e5, 50, 20, n_threads
)
result = compute_lmm_chunk_fused_c(ws, utg_t, n_threads)

# First 3 SNPs: lambdas should be finite
if not np.isfinite(result["lambdas"][:3]).all():
    print(f"FAIL: Non-finite lambdas for normal SNPs: {result['lambdas'][:3]}")
    sys.exit(1)

# SNP 3 (degenerate): beta, se, pwald should be NaN
for key in ("betas", "ses", "pwalds"):
    if not np.isnan(result[key][3]):
        print(f"FAIL: Degenerate SNP {key} not NaN: {result[key][3]}")
        sys.exit(1)

print(f"lambdas: {result['lambdas']}")
print("Degenerate SNP correctly produced NaN beta/se/pwald")
print("Numerical sanity check passed")
