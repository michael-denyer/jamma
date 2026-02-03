"""Tests for compute backend detection and dispatch."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.backend import (
    get_backend_info,
    get_compute_backend,
    is_rust_available,
)


class TestBackendDetection:
    """Tests for get_compute_backend function."""

    def setup_method(self):
        """Clear cache before each test."""
        get_compute_backend.cache_clear()

    def teardown_method(self):
        """Clean up environment after each test."""
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    def test_env_override_rust(self):
        """JAMMA_BACKEND=rust should force Rust backend."""
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()

        assert get_compute_backend() == "rust"

    def test_env_override_jax(self):
        """JAMMA_BACKEND=jax should force JAX backend."""
        os.environ["JAMMA_BACKEND"] = "jax"
        get_compute_backend.cache_clear()

        assert get_compute_backend() == "jax"

    def test_env_override_case_insensitive(self):
        """Environment override should be case-insensitive."""
        os.environ["JAMMA_BACKEND"] = "RUST"
        get_compute_backend.cache_clear()

        assert get_compute_backend() == "rust"

    def test_env_override_whitespace(self):
        """Environment override should handle whitespace."""
        os.environ["JAMMA_BACKEND"] = "  rust  "
        get_compute_backend.cache_clear()

        assert get_compute_backend() == "rust"

    def test_invalid_override_ignored(self):
        """Invalid JAMMA_BACKEND value should be ignored."""
        os.environ["JAMMA_BACKEND"] = "invalid"
        get_compute_backend.cache_clear()

        # Should fall through to auto-detection
        backend = get_compute_backend()
        assert backend in ("jax", "rust")

    def test_no_gpu_returns_rust(self):
        """Without GPU, should return 'rust'."""
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

        with patch("jamma.core.backend._has_gpu", return_value=False):
            get_compute_backend.cache_clear()
            assert get_compute_backend() == "rust"

    def test_with_gpu_returns_jax(self):
        """With GPU, should return 'jax'."""
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

        with patch("jamma.core.backend._has_gpu", return_value=True):
            get_compute_backend.cache_clear()
            assert get_compute_backend() == "jax"

    def test_caching(self):
        """Backend detection should be cached."""
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

        # First call
        backend1 = get_compute_backend()
        # Change env (shouldn't matter due to cache)
        os.environ["JAMMA_BACKEND"] = "jax" if backend1 == "rust" else "rust"
        # Second call should return cached value
        backend2 = get_compute_backend()

        assert backend1 == backend2


class TestRustAvailable:
    """Tests for is_rust_available function."""

    def test_returns_bool(self):
        """Should return a boolean."""
        result = is_rust_available()
        assert isinstance(result, bool)


class TestBackendInfo:
    """Tests for get_backend_info function."""

    def setup_method(self):
        get_compute_backend.cache_clear()

    def teardown_method(self):
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    def test_returns_dict(self):
        """Should return a dictionary with expected keys."""
        info = get_backend_info()

        assert isinstance(info, dict)
        assert "selected" in info
        assert "rust_available" in info
        assert "gpu_available" in info
        assert "override" in info

    def test_shows_override_when_set(self):
        """Should show override value when JAMMA_BACKEND is set."""
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()

        info = get_backend_info()
        assert info["override"] == "rust"


class TestEigendecompDispatch:
    """Tests for eigendecompose_kinship backend dispatch."""

    def setup_method(self):
        get_compute_backend.cache_clear()

    def teardown_method(self):
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not installed")
    def test_rust_backend_produces_valid_output(self):
        """Rust backend should produce valid eigendecomposition."""
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()

        from jamma.lmm.eigen import eigendecompose_kinship

        n = 100
        K = np.eye(n, dtype=np.float64)
        eigenvalues, eigenvectors = eigendecompose_kinship(K)

        assert eigenvalues.shape == (n,)
        assert eigenvectors.shape == (n, n)
        np.testing.assert_allclose(eigenvalues, np.ones(n), rtol=1e-10)

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not installed")
    def test_rust_scipy_parity(self):
        """Rust and scipy backends should produce numerically identical results."""
        from scipy import linalg

        np.random.seed(42)
        n = 100
        A = np.random.randn(n, n)
        K = (A + A.T) / 2

        # Rust backend
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()
        from jamma.lmm.eigen import eigendecompose_kinship

        rust_eigenvalues, _ = eigendecompose_kinship(K.copy())

        # scipy reference (direct call, not through jamma)
        scipy_eigenvalues, _ = linalg.eigh(K.copy())

        # Eigenvalues should match within tolerance
        np.testing.assert_allclose(
            rust_eigenvalues, scipy_eigenvalues, rtol=1e-10, atol=1e-14
        )

    def test_scipy_fallback_when_rust_unavailable(self):
        """Should fall back to scipy when jamma_core import fails."""
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()

        # Mock jamma_core import to fail
        import sys

        # Temporarily remove jamma_core if it exists
        original_module = sys.modules.get("jamma_core")
        sys.modules["jamma_core"] = None  # This will cause ImportError

        try:
            from jamma.lmm.eigen import eigendecompose_kinship

            n = 50
            K = np.eye(n, dtype=np.float64)
            eigenvalues, eigenvectors = eigendecompose_kinship(K)

            # Should still work via scipy fallback
            assert eigenvalues.shape == (n,)
        finally:
            # Restore original module
            if original_module is not None:
                sys.modules["jamma_core"] = original_module
            else:
                sys.modules.pop("jamma_core", None)
