"""Tests for eigendecomposition memory pre-flight check."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.backend import get_compute_backend
from jamma.core.memory import estimate_eigendecomp_memory
from jamma.lmm.eigen import eigendecompose_kinship


class TestEigendecompMemoryEstimate:
    """Tests for memory estimation function."""

    def test_estimate_200k_samples(self):
        """200k samples should require approximately 640GB."""
        n_samples = 200_000
        estimate = estimate_eigendecomp_memory(n_samples)
        # K (320GB) + U (320GB) + workspace (~0.04GB) = ~640GB
        assert 635 < estimate < 645

    def test_estimate_100k_samples(self):
        """100k samples should require approximately 160GB."""
        n_samples = 100_000
        estimate = estimate_eigendecomp_memory(n_samples)
        # K (80GB) + U (80GB) + workspace (~0.02GB) = ~160GB
        assert 155 < estimate < 165

    def test_estimate_scales_quadratically(self):
        """Memory should scale quadratically with n_samples."""
        est_10k = estimate_eigendecomp_memory(10_000)
        est_20k = estimate_eigendecomp_memory(20_000)
        # 2x samples -> 4x memory (quadratic)
        ratio = est_20k / est_10k
        assert 3.9 < ratio < 4.1


class TestEigendecompPreflightCheck:
    """Tests for pre-flight memory check in eigendecompose_kinship.

    Note: These tests require the jax.scipy backend since the jax.rust backend
    doesn't use pre-flight memory checks (faer handles memory internally).
    """

    def setup_method(self):
        """Force jax.scipy backend for scipy memory checks."""
        get_compute_backend.cache_clear()
        os.environ["JAMMA_BACKEND"] = "jax.scipy"
        get_compute_backend.cache_clear()

    def teardown_method(self):
        """Clean up environment."""
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    def test_raises_memory_error_when_insufficient(self):
        """Should raise MemoryError before scipy call when memory insufficient."""
        # Create small matrix (won't actually be decomposed if check fails)
        K = np.eye(100, dtype=np.float64)

        # 100x100 matrix needs ~0.0002GB, mock available memory well below that
        # With 10% safety margin, need < required_gb * 1.1 = 0.0002 * 1.1 = 0.00022GB
        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_process,
        ):
            mock_vm.return_value.available = 100  # 100 bytes (way too small)
            mock_vm.return_value.total = 100
            mock_process.return_value.memory_info.return_value.rss = 10
            mock_process.return_value.memory_info.return_value.vms = 20

            with pytest.raises(MemoryError) as exc_info:
                eigendecompose_kinship(K)

            # Verify error message is informative
            error_msg = str(exc_info.value)
            assert "Insufficient memory" in error_msg
            assert "100" in error_msg or "100x100" in error_msg

    def test_succeeds_when_memory_sufficient(self):
        """Should proceed when sufficient memory available."""
        # Create small matrix
        K = np.eye(100, dtype=np.float64)

        # Mock psutil to report ample memory (1TB)
        # Need to mock both virtual_memory and Process for log_memory_snapshot
        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_process,
        ):
            mock_vm.return_value.available = 1e12  # 1TB
            mock_vm.return_value.total = 1e12
            mock_process.return_value.memory_info.return_value.rss = 1e9  # 1GB
            mock_process.return_value.memory_info.return_value.vms = 2e9  # 2GB

            eigenvalues, eigenvectors = eigendecompose_kinship(K)

            assert eigenvalues.shape == (100,)
            assert eigenvectors.shape == (100, 100)

    def test_error_message_includes_required_and_available(self):
        """Error message should include required GB, available GB."""
        K = np.eye(1000, dtype=np.float64)

        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_process,
        ):
            mock_vm.return_value.available = 1e6  # 1MB (way too small)
            mock_vm.return_value.total = 1e6
            mock_process.return_value.memory_info.return_value.rss = 1e5
            mock_process.return_value.memory_info.return_value.vms = 2e5

            with pytest.raises(MemoryError) as exc_info:
                eigendecompose_kinship(K)

            error_msg = str(exc_info.value)
            assert "GB" in error_msg  # Should mention GB
            has_need = "Need" in error_msg or "required" in error_msg.lower()
            assert has_need
