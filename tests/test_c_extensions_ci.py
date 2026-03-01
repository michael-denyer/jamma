"""CI smoke test: assert both C extensions compiled and loaded.

In CI (where the compile step runs before tests), these tests fail hard if
either extension can't be imported. This prevents the skipif guards in
test_lmm_accel.py and test_eigen_dsyevr.py from silently hiding breakage.

Locally (CI env var not set), these tests skip — developers who haven't
compiled the extensions aren't blocked.
"""

import os

import pytest

_IN_CI = os.environ.get("CI") == "true"


@pytest.mark.tier0
@pytest.mark.skipif(not _IN_CI, reason="C extension CI smoke test (skipped locally)")
class TestCExtensionsAvailable:
    """Assert C extensions are importable in CI."""

    def test_lmm_accel_available(self):
        """_lmm_accel C extension must be compiled and importable in CI."""
        from jamma.lmm.compute_numpy import _C_ACCEL_AVAILABLE

        assert _C_ACCEL_AVAILABLE, (
            "_lmm_accel C extension not available in CI. "
            "The 'Compile C extensions' step may have failed silently."
        )

    def test_eigen_accel_available(self):
        """_eigen_accel C extension must be compiled and importable in CI."""
        from jamma.lmm.eigen import _DSYEVR_AVAILABLE

        assert _DSYEVR_AVAILABLE, (
            "_eigen_accel C extension not available in CI. "
            "The 'Compile C extensions' step may have failed silently."
        )

    def test_lmm_accel_numerical_sanity(self):
        """_lmm_accel produces finite outputs on synthetic data."""
        import numpy as np

        from jamma.lmm._lmm_accel import compute_lmm_batch_c

        rng = np.random.default_rng(42)
        n, n_snps, n_cvt = 50, 3, 1
        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n))
        ab_cols = 6
        Uab = rng.standard_normal((n_snps, n, ab_cols))
        Uab[:, :, 0] = np.abs(Uab[:, :, 0]) + 0.1

        Iab = np.zeros((n_snps, 3, ab_cols))
        Iab[:, 0, :] = Uab.sum(axis=1)
        Iab[:, 1, 3] = Iab[:, 0, 3] - Iab[:, 0, 1] ** 2 / np.maximum(
            Iab[:, 0, 0], 1e-10
        )
        Iab[:, 1, 4] = Iab[:, 0, 4] - Iab[:, 0, 1] * Iab[:, 0, 2] / np.maximum(
            Iab[:, 0, 0], 1e-10
        )
        Iab[:, 1, 5] = Iab[:, 0, 5] - Iab[:, 0, 2] ** 2 / np.maximum(
            Iab[:, 0, 0], 1e-10
        )
        Iab[:, 2, 5] = Iab[:, 1, 5] - Iab[:, 1, 4] ** 2 / np.maximum(
            Iab[:, 1, 3], 1e-10
        )

        result = compute_lmm_batch_c(eigenvalues, Uab, Iab, n, 1e-5, 1e5, 50, 20, n_cvt)
        assert np.isfinite(result["lambdas"]).all()

    def test_eigen_accel_numerical_sanity(self):
        """_eigen_accel (DSYEVR) produces correct eigenvalues for identity."""
        import numpy as np

        from jamma.lmm._eigen_accel import eigh_dsyevr

        n = 50
        K = np.eye(n, dtype=np.float64)
        w, v = eigh_dsyevr(K)

        assert w.shape == (n,)
        assert v.shape == (n, n)
        np.testing.assert_allclose(w, np.ones(n), rtol=1e-14)
