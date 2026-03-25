"""Tests for hardware context collection."""

from __future__ import annotations

import json

import pytest

from jamma.core.hardware import get_hardware_context


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
        """BLAS backend is a non-empty string."""
        ctx = get_hardware_context()
        assert isinstance(ctx["blas_backend"], str)
        assert len(ctx["blas_backend"]) > 0

    def test_hardware_context_positive_counts(self):
        """CPU counts are positive integers."""
        ctx = get_hardware_context()
        assert ctx["cpu_count_physical"] >= 1
        assert ctx["cpu_count_logical"] >= 1
        assert ctx["blas_threads"] >= 1
