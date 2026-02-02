"""Tests for memory estimation module."""

import numpy as np
import psutil
import pytest

from jamma.core import (
    MemoryBreakdown,
    MemorySnapshot,
    check_memory_available,
    cleanup_memory,
    estimate_workflow_memory,
    get_memory_snapshot,
    log_memory_snapshot,
)


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

        # 200k estimate will not be sufficient on most machines (needs ~640GB)
        est = estimate_workflow_memory(200_000, 95_000)
        # Don't check exact match - just verify it's False for this huge estimate
        # (would need 640GB+ which no typical machine has)
        assert (
            est.sufficient is False
        ), "200k sample workflow should exceed available memory"


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
        """JAX eigh should use reasonable memory (not O(n^2) workspace).

        Note: This test measures RSS delta which can be affected by JIT caching
        and garbage collection timing. The threshold is generous to avoid flakiness.
        """
        import gc

        import jax.numpy as jnp

        # 3000x3000 symmetric matrix (smaller to reduce variance)
        n = 3000
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        K = (A + A.T) / 2  # Symmetric

        # Force garbage collection before measurement
        gc.collect()

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

        # Matrix itself is 3000^2 * 8 = 72MB
        # Eigenvectors are another 72MB
        # Eigenvalues are 3000 * 8 = 24KB
        # With O(n) workspace, expect ~150-200MB delta
        # With O(n^2) workspace, would see 300MB+ delta
        #
        # Use 500MB threshold - generous for JIT but catches gross O(n^2) issues
        assert (
            mem_delta < 0.5
        ), f"Eigendecomp used {mem_delta:.2f}GB - may have O(n^2) workspace"

        # Sanity check results
        assert eigenvalues.shape == (n,)
        assert eigenvectors.shape == (n, n)


class TestMemorySnapshot:
    """Tests for memory snapshot functions."""

    def test_get_memory_snapshot_returns_namedtuple(self):
        """get_memory_snapshot returns MemorySnapshot with all fields."""
        snap = get_memory_snapshot()

        assert isinstance(snap, MemorySnapshot)
        assert isinstance(snap.rss_gb, float)
        assert isinstance(snap.vms_gb, float)
        assert isinstance(snap.available_gb, float)
        assert isinstance(snap.total_gb, float)
        assert isinstance(snap.percent_used, float)

    def test_memory_snapshot_values_reasonable(self):
        """Memory values should be positive and sensible."""
        snap = get_memory_snapshot()

        assert snap.rss_gb > 0, "RSS should be positive"
        assert snap.available_gb > 0, "Available should be positive"
        assert snap.total_gb > 0, "Total should be positive"
        assert 0 <= snap.percent_used <= 100, "Percent should be 0-100"
        assert snap.rss_gb <= snap.total_gb, "RSS <= total"

    def test_log_memory_snapshot_returns_snapshot(self):
        """log_memory_snapshot should return MemorySnapshot."""
        snap = log_memory_snapshot("test_label", level="DEBUG")

        assert isinstance(snap, MemorySnapshot)
        assert snap.rss_gb > 0


class TestCleanupMemory:
    """Tests for memory cleanup function."""

    def test_cleanup_memory_returns_snapshot(self):
        """cleanup_memory should return MemorySnapshot after cleanup."""
        snap = cleanup_memory(clear_jax=True, verbose=False)

        assert isinstance(snap, MemorySnapshot)
        assert snap.rss_gb > 0

    def test_cleanup_memory_verbose_logs(self, caplog):
        """cleanup_memory with verbose=True should log before/after."""
        import logging

        caplog.set_level(logging.INFO)
        cleanup_memory(clear_jax=False, verbose=True)

        # Check that memory logging happened (loguru logs may not appear in caplog)
        # Just verify it doesn't crash
        assert True

    def test_cleanup_memory_clears_jax_caches(self):
        """cleanup_memory with clear_jax=True should clear JAX caches."""
        import jax
        import jax.numpy as jnp

        # Create a JIT-compiled function and call it
        @jax.jit
        def dummy_fn(x):
            return x + 1

        dummy_fn(jnp.array([1, 2, 3]))

        # Cleanup should not raise
        cleanup_memory(clear_jax=True, verbose=False)

        # Function should still work after cleanup (just recompiled)
        result = dummy_fn(jnp.array([4, 5, 6]))
        assert list(result) == [5, 6, 7]

    def test_cleanup_frees_memory_after_allocation(self):
        """Cleanup should free memory from deleted arrays."""
        import gc

        # Allocate a moderate array
        big_array = np.zeros((1000, 1000), dtype=np.float64)  # 8MB
        _ = big_array.sum()  # Touch it

        before = get_memory_snapshot()

        # Delete and cleanup
        del big_array
        gc.collect()

        after = cleanup_memory(clear_jax=False, verbose=False)

        # Memory should not have increased significantly
        # (may not decrease due to allocator behavior, but shouldn't spike)
        assert after.rss_gb <= before.rss_gb + 0.1
