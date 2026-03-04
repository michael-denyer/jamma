"""Tests for compute backend detection and information."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from jamma.core.backend import detect_backend, get_backend_info


@pytest.mark.tier0
class TestBackendInfo:
    """Tests for get_backend_info function."""

    def test_returns_dict(self):
        """Should return a dictionary with expected keys."""
        info = get_backend_info()

        assert isinstance(info, dict)
        assert "selected" in info
        assert "jax_available" in info
        assert set(info.keys()) == {"selected", "jax_available"}

    def test_selected_is_valid_backend(self):
        """Selected backend should be 'jax' or 'numpy'."""
        info = get_backend_info()
        assert info["selected"] in ("jax", "numpy")

    def test_jax_available_is_bool(self):
        """jax_available should be a boolean."""
        info = get_backend_info()
        assert isinstance(info["jax_available"], bool)


@pytest.mark.tier0
class TestDetectBackend:
    """Tests for detect_backend() function."""

    @pytest.mark.requires_jax
    def test_auto_returns_jax_when_available(self):
        """detect_backend('auto') returns 'jax' in dev env where JAX is installed."""
        result = detect_backend("auto")
        assert result == "jax"

    def test_numpy_always_returns_numpy(self):
        """detect_backend('numpy') always returns 'numpy', regardless of JAX."""
        result = detect_backend("numpy")
        assert result == "numpy"

    @pytest.mark.requires_jax
    def test_jax_returns_jax_when_available(self):
        """detect_backend('jax') returns 'jax' in dev env where JAX is installed."""
        result = detect_backend("jax")
        assert result == "jax"

    def test_invalid_backend_raises(self):
        """detect_backend with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            detect_backend("invalid")

    def test_env_var_overrides_requested(self, monkeypatch):
        """JAMMA_BACKEND=numpy overrides detect_backend('jax')."""
        monkeypatch.setenv("JAMMA_BACKEND", "numpy")
        result = detect_backend("jax")
        assert result == "numpy"

    @pytest.mark.requires_jax
    def test_env_var_jax_overrides_auto(self, monkeypatch):
        """JAMMA_BACKEND=jax overrides detect_backend('auto') explicitly."""
        monkeypatch.setenv("JAMMA_BACKEND", "jax")
        result = detect_backend("auto")
        assert result == "jax"

    def test_env_var_invalid_raises(self, monkeypatch):
        """JAMMA_BACKEND with invalid value raises ValueError."""
        monkeypatch.setenv("JAMMA_BACKEND", "spark")
        with pytest.raises(ValueError, match="Unknown backend"):
            detect_backend("auto")


@pytest.mark.tier0
class TestDetectBackendJaxAbsent:
    """T1: detect_backend('jax') raises ValueError when JAX is absent."""

    def test_jax_requested_when_absent_raises(self, monkeypatch):
        """detect_backend('jax') raises ValueError when JAX import fails."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jax":
                raise ImportError("mock: jax not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Clear the has_jax cache so detect_backend re-probes
        from jamma.core import backend

        backend.has_jax.cache_clear()
        try:
            with pytest.raises(ValueError, match="JAX is not installed"):
                detect_backend("jax")
        finally:
            backend.has_jax.cache_clear()


@pytest.mark.tier0
def test_import_jamma_succeeds():
    """Smoke test: importing jamma should not raise even if JAX is available."""
    import jamma

    assert hasattr(jamma, "__version__")


@pytest.mark.tier0
class TestAutoSelectBackend:
    """Tests for _auto_select_backend() memory-aware auto-selection in pipeline.py."""

    def _make_sufficient_estimate(self):
        """Build a MemoryBreakdown-like mock with sufficient=True."""
        m = MagicMock()
        m.sufficient = True
        m.total_gb = 1.0
        m.available_gb = 100.0
        return m

    def _make_insufficient_estimate(self):
        """Build a MemoryBreakdown-like mock with sufficient=False."""
        m = MagicMock()
        m.sufficient = False
        m.total_gb = 500.0
        m.available_gb = 10.0
        return m

    def test_small_dataset_c_ext_available_picks_numpy(self):
        """Small dataset + C extension available -> selects 'numpy'."""
        from jamma.pipeline import _auto_select_backend

        with (
            patch(
                "jamma.pipeline.estimate_lmm_memory",
                return_value=self._make_sufficient_estimate(),
            ),
            patch.dict(sys.modules, {"jamma.lmm._lmm_accel": MagicMock()}),
        ):
            result = _auto_select_backend(n_samples=100, n_snps=1000, use_gpu=False)
        assert result == "numpy"

    def test_large_dataset_no_memory_jax_available_picks_jax(self):
        """Large dataset + insufficient memory + JAX available -> selects 'jax'."""
        from jamma.pipeline import _auto_select_backend

        with (
            patch(
                "jamma.pipeline.estimate_lmm_memory",
                return_value=self._make_insufficient_estimate(),
            ),
            patch.dict(sys.modules, {"jamma.lmm._lmm_accel": MagicMock()}),
            patch("jamma.pipeline.has_jax", return_value=True),
        ):
            result = _auto_select_backend(
                n_samples=200_000, n_snps=100_000, use_gpu=False
            )
        assert result == "jax"

    def test_no_c_ext_jax_available_picks_jax(self):
        """No C extension + JAX available -> selects 'jax'."""
        from jamma.pipeline import _auto_select_backend

        # Remove _lmm_accel from sys.modules to simulate absence
        modules = {k: v for k, v in sys.modules.items() if "_lmm_accel" not in k}
        # Ensure _lmm_accel is absent
        modules.pop("jamma.lmm._lmm_accel", None)

        with (
            patch(
                "jamma.pipeline.estimate_lmm_memory",
                return_value=self._make_sufficient_estimate(),
            ),
            patch.dict(sys.modules, {"jamma.lmm._lmm_accel": None}),
            patch("jamma.pipeline.has_jax", return_value=True),
        ):
            result = _auto_select_backend(n_samples=100, n_snps=1000, use_gpu=False)
        assert result == "jax"

    def test_no_c_ext_no_jax_falls_back_to_numpy(self):
        """No C extension + no JAX -> falls back to 'numpy'."""
        from jamma.pipeline import _auto_select_backend

        with (
            patch(
                "jamma.pipeline.estimate_lmm_memory",
                return_value=self._make_sufficient_estimate(),
            ),
            patch.dict(sys.modules, {"jamma.lmm._lmm_accel": None}),
            patch("jamma.pipeline.has_jax", return_value=False),
        ):
            result = _auto_select_backend(n_samples=100, n_snps=1000, use_gpu=False)
        assert result == "numpy"

    def test_use_gpu_true_jax_available_picks_jax(self):
        """use_gpu=True + JAX available -> always 'jax'."""
        from jamma.pipeline import _auto_select_backend

        with (
            patch(
                "jamma.pipeline.estimate_lmm_memory",
                return_value=self._make_sufficient_estimate(),
            ),
            patch.dict(sys.modules, {"jamma.lmm._lmm_accel": MagicMock()}),
            patch("jamma.pipeline.has_jax", return_value=True),
        ):
            result = _auto_select_backend(n_samples=100, n_snps=1000, use_gpu=True)
        assert result == "jax"

    def test_explicit_numpy_not_using_auto_select(self):
        """Explicit backend='numpy' bypasses _auto_select_backend."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(bfile="/tmp/test", backend="numpy")
        assert config.backend == "numpy"

    def test_explicit_jax_not_using_auto_select(self):
        """Explicit backend='jax' bypasses _auto_select_backend."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(bfile="/tmp/test", backend="jax")
        assert config.backend == "jax"
