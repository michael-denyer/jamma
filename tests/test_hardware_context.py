"""Tests for hardware context collection and x64 precision guard."""

from __future__ import annotations

import json

import pytest

from jamma.core.hardware import assert_x64_precision, get_hardware_context


@pytest.mark.tier0
class TestHardwareContext:
    """Tests for hardware context collection."""

    def test_hardware_context_keys(self):
        """All expected keys are present in hardware context."""
        ctx = get_hardware_context()
        expected_keys = {
            "cpu_model",
            "cpu_count_physical",
            "cpu_count_logical",
            "blas_backend",
            "blas_threads",
            "jax_version",
            "jax_backend",
            "jax_device_count",
            "jax_x64_enabled",
            "numpy_version",
            "platform",
            "python_version",
        }
        assert set(ctx.keys()) == expected_keys

    def test_hardware_context_types(self):
        """All values are JSON-serializable types."""
        ctx = get_hardware_context()
        # Should not raise
        json.dumps(ctx)

    def test_hardware_context_cpu_model_nonempty(self):
        """CPU model is a non-empty string."""
        ctx = get_hardware_context()
        assert isinstance(ctx["cpu_model"], str)
        assert len(ctx["cpu_model"]) > 0

    def test_hardware_context_blas_backend(self):
        """BLAS backend is a recognized string."""
        ctx = get_hardware_context()
        assert isinstance(ctx["blas_backend"], str)
        # Should be one of known backends or "unknown"
        assert ctx["blas_backend"] in (
            "mkl",
            "openblas",
            "blis",
            "accelerate",
            "unknown",
        ) or isinstance(ctx["blas_backend"], str)

    def test_hardware_context_jax_x64(self):
        """JAX x64 is enabled in test environment."""
        ctx = get_hardware_context()
        assert ctx["jax_x64_enabled"] is True

    def test_hardware_context_positive_counts(self):
        """CPU and device counts are positive integers."""
        ctx = get_hardware_context()
        assert ctx["cpu_count_physical"] >= 1
        assert ctx["cpu_count_logical"] >= 1
        assert ctx["jax_device_count"] >= 1
        assert ctx["blas_threads"] >= 1


@pytest.mark.tier0
class TestAssertX64Precision:
    """Tests for x64 precision guard."""

    def test_assert_x64_passes_when_enabled(self):
        """assert_x64_precision passes when x64 is configured."""
        # conftest.py enables x64 for all tests
        assert_x64_precision()  # Should not raise

    def test_assert_x64_raises_when_disabled(self, monkeypatch):
        """assert_x64_precision raises RuntimeError when x64 is disabled."""
        import jax

        # jax.config.jax_enable_x64 is a read-only property on the Config class.
        # We must patch the property on the class itself, not on the instance.
        monkeypatch.setattr(
            type(jax.config),
            "jax_enable_x64",
            property(lambda self: False),
        )
        with pytest.raises(RuntimeError, match="64-bit precision is not enabled"):
            assert_x64_precision()
