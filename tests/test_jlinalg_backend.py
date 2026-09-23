"""Regression coverage for the BLAS backend name jlinalg resolves to."""

from __future__ import annotations

import pytest

from jamma.jlinalg import HAS_C_EXTENSION, blas_backend, blas_is_ilp64

pytestmark = pytest.mark.tier0


@pytest.mark.skipif(not HAS_C_EXTENSION, reason="jlinalg C extension not compiled")
class TestBackendDetectionUnchanged:
    """The active backend is whatever this machine's dispatch already resolved."""

    def test_blas_backend_known_value(self):
        known = {
            "MKL-ILP64",
            "OpenBLAS-ILP64",
            "Accelerate-ILP64",
            "system-BLAS-ILP64",
            "numpy-fallback",
        }
        assert blas_backend in known, f"Unknown blas_backend: {blas_backend}"

    def test_ilp64_backend_reports_ilp64(self):
        if blas_backend == "numpy-fallback":
            pytest.skip("no vendor backend resolved on this host")
        assert blas_is_ilp64 == 1
