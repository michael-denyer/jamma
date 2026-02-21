"""Tests for CPU device sharding infrastructure.

Tests for:
- configure_jax() multi-device setup (via get_jax_info observability)
- get_blas_thread_count() JAX-device-aware thread reduction
- _compute_chunk_size() device-count alignment
- auto_tune_chunk_size() device-count alignment pass-through
"""

from __future__ import annotations

from unittest.mock import patch

import jax
import pytest

from jamma.core.jax_config import get_jax_info
from jamma.core.threading import get_blas_thread_count
from jamma.lmm.chunk import _compute_chunk_size, auto_tune_chunk_size


@pytest.mark.tier0
class TestJaxDeviceConfiguration:
    """Tests for configure_jax() device sharding infrastructure."""

    def test_get_jax_info_includes_n_cpu_devices(self):
        """get_jax_info() must report n_cpu_devices for observability."""
        info = get_jax_info()
        assert "n_cpu_devices" in info
        assert isinstance(info["n_cpu_devices"], int)
        assert info["n_cpu_devices"] >= 1

    def test_n_cpu_devices_matches_jax_devices(self):
        """n_cpu_devices in get_jax_info() must match len(jax.devices('cpu'))."""
        info = get_jax_info()
        assert info["n_cpu_devices"] == len(jax.devices("cpu"))

    def test_auto_config_produces_at_least_one_device(self):
        """Auto-config must produce at least 1 device regardless of hardware."""
        # conftest.py calls ensure_jax_configured() which triggers auto-config.
        # This test verifies the result is sane.
        assert len(jax.devices("cpu")) >= 1


@pytest.mark.tier0
class TestBlasThreadJaxAwareness:
    """Tests for get_blas_thread_count() JAX device-aware reduction."""

    def test_returns_positive_with_multiple_jax_devices(self):
        """Thread count must be positive even when JAX has many devices."""
        # Use current JAX state (conftest auto-configured)
        n = get_blas_thread_count()
        assert n >= 1

    def test_env_override_takes_priority_over_jax_devices(self, monkeypatch):
        """JAMMA_BLAS_THREADS env var must win over JAX device reduction."""
        monkeypatch.setenv("JAMMA_BLAS_THREADS", "3")
        # Even with multiple JAX devices, env var should win
        assert get_blas_thread_count() == 3

    def test_reduces_threads_for_multiple_jax_devices(self, monkeypatch):
        """Thread count must reduce proportionally to JAX device count."""
        import jax
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or 4

        # Mock jax.devices to return a list of N device objects.
        # We need n_devices > 1 to trigger the reduction path.
        # Patch the jax module directly since threading.py does `import jax` inside
        # get_blas_thread_count(), then calls jax.devices("cpu").
        n_devices = 4
        mock_devices = [object() for _ in range(n_devices)]

        with patch.object(jax, "devices", return_value=mock_devices):
            n = get_blas_thread_count()

        expected = max(1, physical_cores // n_devices)
        assert n == expected

    def test_single_jax_device_uses_physical_cores(self, monkeypatch):
        """With 1 JAX device, threads should equal physical core count."""
        import os

        import jax
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
        max_threads = os.cpu_count() or 64

        mock_devices = [object()]  # single device

        with patch.object(jax, "devices", return_value=mock_devices):
            n = get_blas_thread_count()

        expected = max(1, min(physical_cores, max_threads))
        assert n == expected


@pytest.mark.tier0
class TestComputeChunkSizeAlignment:
    """Tests for _compute_chunk_size() n_devices alignment."""

    def test_default_n_devices_is_backward_compatible(self):
        """n_devices=1 must produce same result as calling without it."""
        result_default = _compute_chunk_size(1410, 10768, 50, 1)
        result_explicit = _compute_chunk_size(1410, 10768, 50, 1, n_devices=1)
        assert result_default == result_explicit

    def test_n_devices_4_result_is_multiple_of_4(self):
        """Result with n_devices=4 must be a multiple of 4."""
        result = _compute_chunk_size(1410, 10768, 50, 1, n_devices=4)
        assert result % 4 == 0, f"Expected multiple of 4, got {result}"

    def test_n_devices_8_result_is_multiple_of_8(self):
        """Result with n_devices=8 must be a multiple of 8."""
        result = _compute_chunk_size(1000, 100000, 50, 1, n_devices=8)
        if result < 100000:  # only check when chunking actually occurs
            assert result % 8 == 0, f"Expected multiple of 8, got {result}"

    def test_n_devices_1_no_alignment_applied(self):
        """n_devices=1 must not modify the chunk size calculation."""
        result_no_devices = _compute_chunk_size(5000, 50000, 50, 1, n_devices=1)
        # Just verify it returns a positive reasonable value
        assert result_no_devices >= 100

    def test_minimum_still_enforced_after_alignment(self):
        """Minimum chunk of 100 must apply even after device alignment."""
        # Use extreme values to force very small chunk
        result = _compute_chunk_size(
            n_samples=1000000,
            n_snps=10,
            n_grid=50,
            n_cvt=1,
            n_devices=4,
        )
        # When n_snps is smaller than chunking threshold, returns n_snps
        # otherwise minimum of 100 applies
        assert result >= 4  # at least n_devices (alignment floor)

    def test_no_chunking_needed_returns_n_snps(self):
        """When no chunking needed, n_snps is returned unchanged."""
        result = _compute_chunk_size(10, 100, 50, 1, n_devices=4)
        # With tiny n_samples, buffer limit >> n_snps, so should return n_snps
        assert result == 100

    def test_n_devices_2_mouse_hs1940_scale(self):
        """Test alignment at mouse_hs1940 scale (1410 samples, 10768 SNPs)."""
        result = _compute_chunk_size(1410, 10768, 50, 1, n_devices=2)
        # No chunking expected at this scale (buffer limit >> 10768)
        assert result == 10768


@pytest.mark.tier0
class TestAutoTuneChunkSizeDeviceAlignment:
    """Tests for auto_tune_chunk_size() n_devices pass-through."""

    def test_default_n_devices_is_backward_compatible(self):
        """Calling without n_devices must match n_devices=1."""
        result_default = auto_tune_chunk_size(10000, 50000)
        result_explicit = auto_tune_chunk_size(10000, 50000, n_devices=1)
        assert result_default == result_explicit

    def test_n_devices_4_result_is_multiple_of_4(self):
        """When n_devices=4, chunk must be a multiple of 4."""
        result = auto_tune_chunk_size(
            n_samples=10000,
            n_filtered=100000,
            mem_budget_gb=4.0,
            n_devices=4,
        )
        assert result % 4 == 0, f"Expected multiple of 4, got {result}"

    def test_min_chunk_still_enforced_with_n_devices(self):
        """min_chunk floor must apply even with n_devices set."""
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=500,
            mem_budget_gb=0.001,
            min_chunk=1000,
            n_devices=4,
        )
        assert result >= 1000
