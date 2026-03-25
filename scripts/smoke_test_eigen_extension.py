"""Smoke test for jlinalg.eigh — used by cibuildwheel.

Verifies:
1. The compiled C extension imports successfully (ABI match)
2. eigh produces correct eigenvalues on a known matrix (if vendor LAPACK available)

In manylinux containers without vendor LAPACK, eigh raises RuntimeError.
This is expected — the wheel still works via numpy.linalg.eigh fallback at runtime.
"""

import sys

import numpy as np

from jamma.jlinalg._jlinalg import (
    ABI_VERSION,
    blas_backend,
    blas_has_dsyevd,
    blas_has_dsyevr,
    eigh,
)

print(f"_jlinalg OK, ABI={ABI_VERSION}, backend={blas_backend}")
print(f"LAPACK: DSYEVD={blas_has_dsyevd}, DSYEVR={blas_has_dsyevr}")

if not blas_has_dsyevd and not blas_has_dsyevr:
    print(
        "No vendor LAPACK available — skipping eigh numerical test (expected in manylinux)"
    )
    print("Eigen extension smoke test passed (import-only)")
    sys.exit(0)

# Test with identity matrix (eigenvalues should be all 1.0)
n = 100
K = np.eye(n, dtype=np.float64)
w, v = eigh(K)

assert w.shape == (n,), f"Expected ({n},) eigenvalues, got {w.shape}"
assert v.shape == (n, n), f"Expected ({n},{n}) eigenvectors, got {v.shape}"
np.testing.assert_allclose(w, np.ones(n), rtol=1e-14)
print(f"Identity eigenvalues OK: min={w.min():.15f}, max={w.max():.15f}")

# Test with random SPD matrix
rng = np.random.default_rng(42)
A = rng.standard_normal((50, 50))
K_spd = (A @ A.T) / 50
K_ref = K_spd.copy()
w2, v2 = eigh(K_spd)

# Reconstruction check
K_recon = v2 @ np.diag(w2) @ v2.T
max_err = np.max(np.abs(K_ref - K_recon))
assert max_err < 1e-10, f"Reconstruction error: {max_err}"
print(f"SPD reconstruction error: {max_err:.2e}")
print("Eigen extension smoke test passed")
