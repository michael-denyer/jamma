"""CI smoke test: assert C extensions compiled and loaded.

In CI (where the compile step runs before tests), these tests fail hard if
the extension can't be imported. This prevents the skipif guards in
test_lmm_accel.py from silently hiding breakage.

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
        from jamma.lmm import compute_numpy

        assert compute_numpy._accel is not None, (
            "_lmm_accel C extension not available in CI. "
            "The 'Compile C extensions' step may have failed silently."
        )

    def test_lmm_accel_numerical_sanity(self):
        """The fused Wald kernel produces finite outputs on synthetic data.

        Drives what DispatchPath.FUSED resolves to for n_cvt=1. It used to drive
        compute_lmm_batch_c, which no dispatch path selected, so this could have
        passed on a build whose production kernel was broken.
        """
        import numpy as np

        from jamma.lmm._lmm_accel import (
            compute_lmm_chunk_fused_c,
            create_workspace_fused_c,
        )

        rng = np.random.default_rng(42)
        n, n_snps = 50, 3
        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n))
        w = rng.standard_normal(n)
        Uty = rng.standard_normal(n)
        utg_t = rng.standard_normal((n_snps, n))

        uab_inv_soa = np.empty((3, n), dtype=np.float64)
        uab_inv_soa[0] = w * w
        uab_inv_soa[1] = w * Uty
        uab_inv_soa[2] = Uty * Uty

        ws = create_workspace_fused_c(
            eigenvalues, uab_inv_soa, w, Uty, n, 1e-5, 1e5, 50, 20, 1
        )
        result = compute_lmm_chunk_fused_c(ws, utg_t, 1)
        assert np.isfinite(result["lambdas"]).all()
