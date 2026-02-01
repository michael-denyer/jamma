"""Tests for memory estimation module."""

import numpy as np
import psutil
import pytest

from jamma.core import MemoryBreakdown, check_memory_available, estimate_workflow_memory


class TestMemoryEstimation:
    """Tests for estimate_workflow_memory function."""

    def test_memory_breakdown_200k(self):
        """Memory estimate for 200k samples - peak during eigendecomp ~640GB."""
        est = estimate_workflow_memory(200_000, 95_000)

        # Kinship: 200k^2 * 8 / 1e9 = 320GB
        assert (
            319 < est.kinship_gb < 321
        ), f"Expected ~320GB kinship, got {est.kinship_gb}"

        # Genotypes: 200k * 95k * 4 / 1e9 = 76GB
        assert (
            75 < est.genotypes_gb < 77
        ), f"Expected ~76GB genotypes, got {est.genotypes_gb}"

        # Eigenvectors: same as kinship = 320GB
        assert 319 < est.eigenvectors_gb < 321

        # Eigendecomp workspace: O(n) not O(n^2) - should be < 1GB
        assert (
            est.eigendecomp_workspace_gb < 1.0
        ), f"Workspace should be O(n) ~50MB, got {est.eigendecomp_workspace_gb:.2f}GB"

        # Peak is during eigendecomp: kinship + eigenvectors = ~640GB
        # (kinship matrix in + eigenvectors out simultaneously)
        assert (
            600 < est.total_gb < 700
        ), f"Expected ~640GB (eigendecomp peak), got {est.total_gb}"

    def test_memory_breakdown_10k(self):
        """Memory estimate for 10k samples should be reasonable."""
        est = estimate_workflow_memory(10_000, 100_000)

        # Total should be < 10GB for this scale
        assert est.total_gb < 10, f"10k scale should need <10GB, got {est.total_gb}"

    def test_memory_breakdown_has_all_fields(self):
        """MemoryBreakdown should have all expected fields."""
        est = estimate_workflow_memory(1_000, 1_000)

        assert isinstance(est, MemoryBreakdown)
        assert isinstance(est.kinship_gb, float)
        assert isinstance(est.genotypes_gb, float)
        assert isinstance(est.eigenvectors_gb, float)
        assert isinstance(est.eigendecomp_workspace_gb, float)
        assert isinstance(est.lmm_rotated_gb, float)
        assert isinstance(est.lmm_batch_gb, float)
        assert isinstance(est.total_gb, float)
        assert isinstance(est.available_gb, float)
        assert isinstance(est.sufficient, bool)

    def test_sufficient_flag_correct(self):
        """Sufficient flag should reflect available vs required."""
        # Tiny estimate should always be sufficient
        est = estimate_workflow_memory(100, 100)
        assert est.sufficient is True

        # 200k estimate will not be sufficient on most machines
        est = estimate_workflow_memory(200_000, 95_000)
        available = psutil.virtual_memory().available / 1e9
        expected_sufficient = est.total_gb * 1.1 < available
        assert est.sufficient == expected_sufficient


class TestCheckMemoryAvailable:
    """Tests for check_memory_available function."""

    def test_sufficient_memory_returns_true(self):
        """Tiny memory request should succeed."""
        result = check_memory_available(0.001, operation="test")
        assert result is True

    def test_insufficient_memory_raises(self):
        """Huge memory request should raise MemoryError."""
        with pytest.raises(MemoryError) as exc_info:
            check_memory_available(1_000_000, operation="test allocation")

        assert "Insufficient memory" in str(exc_info.value)
        assert "test allocation" in str(exc_info.value)

    def test_error_message_contains_details(self):
        """Error message should include required, available, and suggestion."""
        with pytest.raises(MemoryError) as exc_info:
            check_memory_available(1_000_000, safety_margin=0.1, operation="kinship")

        msg = str(exc_info.value)
        assert "1000000" in msg or "1e+06" in msg.lower()  # Required amount
        assert "GB available" in msg  # Available amount
        assert "kinship" in msg  # Operation name


class TestEigendecompMemory:
    """Tests for eigendecomposition memory usage."""

    @pytest.mark.slow
    def test_eigendecomp_memory_reasonable(self):
        """JAX eigh should use reasonable memory (not O(n^2) workspace)."""
        import jax.numpy as jnp

        # 5000x5000 symmetric matrix
        n = 5000
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        K = (A + A.T) / 2  # Symmetric

        # Measure memory before
        mem_before = psutil.Process().memory_info().rss / 1e9

        # Run eigendecomposition
        K_jax = jnp.array(K)
        eigenvalues, eigenvectors = jnp.linalg.eigh(K_jax)

        # Force computation to complete
        eigenvalues.block_until_ready()
        eigenvectors.block_until_ready()

        # Measure memory after
        mem_after = psutil.Process().memory_info().rss / 1e9
        mem_delta = mem_after - mem_before

        # Matrix itself is 5000^2 * 8 = 200MB
        # Eigenvectors are another 200MB
        # Eigenvalues are 5000 * 8 = 40KB
        # O(n^2) workspace would add another ~200-400MB
        # Total with O(n^2) would be ~600-800MB
        # With O(n) workspace, should be ~400-500MB

        # Allow up to 1GB (generous margin for JIT compilation overhead)
        assert (
            mem_delta < 1.0
        ), f"Eigendecomp used {mem_delta:.2f}GB - may have O(n^2) workspace"

        # Sanity check results
        assert eigenvalues.shape == (n,)
        assert eigenvectors.shape == (n, n)
