"""Tests for compute backend detection and information."""

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
        assert "gpu_available" in info
        # Should only have these two keys
        assert set(info.keys()) == {"selected", "gpu_available"}

    def test_selected_is_valid_backend(self):
        """Selected backend should be 'jax' or 'numpy'."""
        info = get_backend_info()
        assert info["selected"] in ("jax", "numpy")

    def test_gpu_available_is_bool(self):
        """gpu_available should be a boolean."""
        info = get_backend_info()
        assert isinstance(info["gpu_available"], bool)


@pytest.mark.tier0
class TestDetectBackend:
    """Tests for detect_backend() function."""

    def test_auto_returns_jax_when_available(self):
        """detect_backend('auto') returns 'jax' in dev env where JAX is installed."""
        result = detect_backend("auto")
        assert result == "jax"

    def test_numpy_always_returns_numpy(self):
        """detect_backend('numpy') always returns 'numpy', regardless of JAX."""
        result = detect_backend("numpy")
        assert result == "numpy"

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
def test_import_jamma_succeeds():
    """Smoke test: importing jamma should not raise even if JAX is available."""
    import jamma

    assert hasattr(jamma, "__version__")
