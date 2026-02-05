"""Tests for compute backend information."""

from jamma.core.backend import get_backend_info


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

    def test_selected_is_jax_numpy(self):
        """Selected backend should always be jax.numpy."""
        info = get_backend_info()
        assert info["selected"] == "jax.numpy"

    def test_gpu_available_is_bool(self):
        """gpu_available should be a boolean."""
        info = get_backend_info()
        assert isinstance(info["gpu_available"], bool)
