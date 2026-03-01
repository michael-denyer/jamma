"""Smoke test for the _eigen_accel C extension — used by cibuildwheel.

Verifies:
1. The compiled C extension imports successfully (ABI match)
2. LAPACK was discovered via dlopen (IS_ILP64 is set)
3. eigh_dsyevr produces correct eigenvalues on a known matrix
"""

import numpy as np

from jamma.lmm._eigen_accel import ABI_VERSION, IS_ILP64, eigh_dsyevr

print(f"_eigen_accel OK, ABI={ABI_VERSION}, IS_ILP64={IS_ILP64}")

# Test with identity matrix (eigenvalues should be all 1.0)
n = 100
K = np.eye(n, dtype=np.float64)
w, v = eigh_dsyevr(K)

assert w.shape == (n,), f"Expected ({n},) eigenvalues, got {w.shape}"
assert v.shape == (n, n), f"Expected ({n},{n}) eigenvectors, got {v.shape}"
np.testing.assert_allclose(w, np.ones(n), rtol=1e-14)
print(f"Identity eigenvalues OK: min={w.min():.15f}, max={w.max():.15f}")

# Test with random SPD matrix
rng = np.random.default_rng(42)
A = rng.standard_normal((50, 50))
K_spd = (A @ A.T) / 50
K_ref = K_spd.copy()
w2, v2 = eigh_dsyevr(K_spd)

# Reconstruction check
K_recon = v2 @ np.diag(w2) @ v2.T
max_err = np.max(np.abs(K_ref - K_recon))
assert max_err < 1e-10, f"Reconstruction error: {max_err}"
print(f"SPD reconstruction error: {max_err:.2e}")
print("Eigen extension smoke test passed")
