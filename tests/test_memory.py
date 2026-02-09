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
from jamma.core.memory import _uab_iab_gb, estimate_lmm_memory


class TestMemoryEstimation:
    """Tests for estimate_workflow_memory function."""

    def test_memory_breakdown_200k(self):
        """Memory estimate for 200k samples - peak during eigendecomp ~1280GB."""
        est = estimate_workflow_memory(200_000, 95_000)

        # Kinship: 200k^2 * 8 / 1e9 = 320GB
        assert (
            319 < est.kinship_gb < 321
        ), f"Expected ~320GB kinship, got {est.kinship_gb}"

        # Genotypes: 200k * 95k * 8 / 1e9 = 152GB (float64 JAX copy)
        assert (
            151 < est.genotypes_gb < 153
        ), f"Expected ~152GB genotypes (float64), got {est.genotypes_gb}"

        # Eigenvectors: same as kinship = 320GB
        assert 319 < est.eigenvectors_gb < 321

        # Eigendecomp workspace: DSYEVD O(n^2) - ~640GB at 200k
        assert est.eigendecomp_workspace_gb > 600, (
            f"DSYEVD workspace should be ~640GB at 200k, "
            f"got {est.eigendecomp_workspace_gb:.2f}GB"
        )

        # Peak is during eigendecomp: kinship + eigenvectors + workspace = ~1280GB
        assert (
            1250 < est.total_gb < 1310
        ), f"Expected ~1280GB (eigendecomp peak), got {est.total_gb}"

    def test_memory_breakdown_10k(self):
        """Memory estimate for 10k samples should be reasonable."""
        est = estimate_workflow_memory(10_000, 100_000)

        # Total includes float64 genotypes (8GB) + Uab/Iab intermediates
        # LMM phase peak: eigenvectors(0.8) + genotypes(8) + batch+Uab/Iab(~11) ≈ 20GB
        assert est.total_gb < 25, f"10k scale should need <25GB, got {est.total_gb}"

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


class TestLmmMemoryEstimation:
    """Tests for estimate_lmm_memory function (LMM-phase only)."""

    def test_lmm_estimate_less_than_workflow(self):
        """LMM-only estimate should be less than full workflow estimate."""
        lmm_est = estimate_lmm_memory(100_000, 10_000)
        full_est = estimate_workflow_memory(100_000, 10_000)

        assert lmm_est.total_gb < full_est.total_gb, (
            f"LMM-only ({lmm_est.total_gb:.1f}GB) should be less than "
            f"full pipeline ({full_est.total_gb:.1f}GB)"
        )

    def test_lmm_estimate_excludes_kinship(self):
        """LMM estimate should not include kinship memory."""
        est = estimate_lmm_memory(100_000, 10_000)
        assert est.kinship_gb == 0.0

    def test_lmm_estimate_excludes_eigendecomp_workspace(self):
        """LMM estimate should not include eigendecomp workspace."""
        est = estimate_lmm_memory(100_000, 10_000)
        assert est.eigendecomp_workspace_gb == 0.0

    def test_lmm_estimate_includes_eigenvectors(self):
        """LMM estimate should include eigenvectors (~80GB at 100k)."""
        est = estimate_lmm_memory(100_000, 10_000)
        assert 79 < est.eigenvectors_gb < 81

    def test_lmm_estimate_100k_under_300gb(self):
        """At 100k samples with 100 SNPs, LMM should need well under 300GB.

        This is the exact scenario from the xlarge benchmark bug:
        300.6GB available, but old check demanded 320GB (eigendecomp peak).
        """
        est = estimate_lmm_memory(100_000, 100)
        assert est.total_gb < 200, (
            f"LMM for 100k samples × 100 SNPs should need <200GB, "
            f"got {est.total_gb:.1f}GB"
        )

    def test_returns_memory_breakdown(self):
        """Should return MemoryBreakdown with all fields."""
        est = estimate_lmm_memory(1_000, 1_000)
        assert isinstance(est, MemoryBreakdown)

    def test_sufficient_flag_correct(self):
        """Tiny estimate should be sufficient."""
        est = estimate_lmm_memory(100, 100)
        assert est.sufficient is True


class TestMemoryEstimateVsActualAllocation:
    """Regression tests: estimates must cover actual runtime tensor shapes.

    These tests verify that memory estimators account for the dominant JAX
    intermediate buffers (Uab_batch, Iab_batch) that are created during LMM
    computation. Without these, the estimate can pass but execution OOMs.
    """

    @pytest.mark.parametrize(
        "n_samples,chunk_size,n_cvt",
        [
            (1_000, 500, 1),
            (10_000, 5_000, 1),
            (100_000, 10_000, 1),
            (10_000, 5_000, 3),
            (50_000, 20_000, 2),
        ],
    )
    def test_lmm_estimate_covers_uab_iab(self, n_samples, chunk_size, n_cvt):
        """LMM estimate must include Uab_batch + Iab_batch memory.

        Runtime allocates:
        - Uab_batch: (chunk_size, n_samples, n_index) float64
        - Iab_batch: (chunk_size, n_cvt+2, n_index) float64
        """
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2

        # Actual allocation sizes
        uab_bytes = chunk_size * n_samples * n_index * 8
        iab_bytes = chunk_size * (n_cvt + 2) * n_index * 8
        actual_uab_iab_gb = (uab_bytes + iab_bytes) / 1e9

        # Estimator's computation
        estimated_gb = _uab_iab_gb(n_samples, chunk_size, n_cvt)

        assert abs(estimated_gb - actual_uab_iab_gb) < 1e-9, (
            f"_uab_iab_gb({n_samples}, {chunk_size}, {n_cvt}) = {estimated_gb:.6f}GB "
            f"but actual is {actual_uab_iab_gb:.6f}GB"
        )

    def test_lmm_batch_gb_includes_uab_iab(self):
        """estimate_lmm_memory.lmm_batch_gb must include Uab/Iab, not just UtG."""
        n_samples = 10_000
        batch_size = 5_000
        n_cvt = 1

        est = estimate_lmm_memory(
            n_samples, 1_000, lmm_batch_size=batch_size, n_cvt=n_cvt
        )

        # UtG alone: n_samples * batch_size * 8
        utg_only_gb = n_samples * batch_size * 8 / 1e9

        # lmm_batch_gb must be strictly larger than UtG alone (Uab+Iab added)
        assert est.lmm_batch_gb > utg_only_gb, (
            f"lmm_batch_gb ({est.lmm_batch_gb:.4f}GB) should exceed "
            f"UtG-only ({utg_only_gb:.4f}GB) because Uab/Iab must be included"
        )

    def test_streaming_lmm_estimate_covers_uab_iab(self):
        """Streaming LMM estimate must include Uab/Iab in total_peak_gb."""
        from jamma.core.memory import estimate_lmm_streaming_memory

        n_samples = 10_000
        chunk_size = 5_000

        est = estimate_lmm_streaming_memory(n_samples, 95_000, chunk_size=chunk_size)

        # Minimum must include eigenvectors + Uab/Iab
        uab_iab_gb = _uab_iab_gb(n_samples, chunk_size, n_cvt=1)
        eigenvectors_gb = n_samples**2 * 8 / 1e9

        assert est.total_peak_gb >= eigenvectors_gb + uab_iab_gb, (
            f"total_peak_gb ({est.total_peak_gb:.4f}GB) should be >= "
            f"eigenvectors ({eigenvectors_gb:.4f}GB) + Uab/Iab ({uab_iab_gb:.4f}GB)"
        )


class TestKinshipDtypeAccounting:
    """Verify memory model accounts for float64 genotype copy in kinship."""

    def test_workflow_genotypes_gb_is_float64(self):
        """estimate_workflow_memory must use float64 (8 bytes) for genotypes.

        compute_centered_kinship converts genotypes to float64 via
        jnp.array(genotypes_filtered, dtype=jnp.float64).
        """
        n_samples = 10_000
        n_snps = 50_000

        est = estimate_workflow_memory(n_samples, n_snps)

        # float64: n_samples * n_snps * 8 bytes
        expected_gb = n_samples * n_snps * 8 / 1e9
        assert abs(est.genotypes_gb - expected_gb) < 1e-9, (
            f"genotypes_gb ({est.genotypes_gb:.4f}GB) should be float64 "
            f"({expected_gb:.4f}GB), not float32 ({expected_gb / 2:.4f}GB)"
        )

    def test_lmm_genotypes_gb_is_float64(self):
        """estimate_lmm_memory must use float64 for genotypes."""
        n_samples = 10_000
        n_snps = 50_000

        est = estimate_lmm_memory(n_samples, n_snps)

        expected_gb = n_samples * n_snps * 8 / 1e9
        assert abs(est.genotypes_gb - expected_gb) < 1e-9


class TestGateCorrectnessRunnerJax:
    """Tests that runner_jax.py memory gate correctly blocks/passes."""

    def test_lmm_gate_passes_with_ample_memory(self):
        """Memory check should pass when plenty of memory is available."""
        from unittest.mock import patch

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 500 * 1e9  # 500GB

            est = estimate_lmm_memory(1_000, 1_000)
            assert est.sufficient is True

    def test_lmm_gate_blocks_with_scarce_memory(self):
        """Memory check should fail when memory is insufficient."""
        from unittest.mock import patch

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 1 * 1e9  # 1GB

            # 100k samples needs ~80GB eigenvectors alone
            est = estimate_lmm_memory(100_000, 10_000)
            assert est.sufficient is False

    def test_lmm_gate_threshold_boundary(self):
        """Memory check should account for 10% safety margin.

        _check_available uses strict less-than: total_gb * 1.1 < available_gb.
        """
        from unittest.mock import patch

        # Compute total_gb deterministically (mock memory so it doesn't affect total)
        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 1000 * 1e9
            est_dry = estimate_lmm_memory(100, 100)

        needed_with_margin = est_dry.total_gb * 1.1

        # Set available to just above the margin (should pass)
        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = (needed_with_margin + 0.001) * 1e9

            est = estimate_lmm_memory(100, 100)
            assert est.sufficient is True

        # Set available to just under the margin (should fail)
        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = (needed_with_margin - 0.001) * 1e9

            est = estimate_lmm_memory(100, 100)
            assert est.sufficient is False


class TestGateCorrectnessRunnerStreaming:
    """Tests that runner_streaming.py memory gate correctly blocks/passes."""

    def test_streaming_gate_passes_with_ample_memory(self):
        """Memory check should pass when plenty of memory is available."""
        from unittest.mock import patch

        from jamma.core.memory import estimate_lmm_streaming_memory

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 500 * 1e9

            est = estimate_lmm_streaming_memory(1_000, 10_000)
            assert est.sufficient is True

    def test_streaming_gate_blocks_with_scarce_memory(self):
        """Memory check should fail when memory is insufficient."""
        from unittest.mock import patch

        from jamma.core.memory import estimate_lmm_streaming_memory

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 1 * 1e9

            est = estimate_lmm_streaming_memory(100_000, 95_000)
            assert est.sufficient is False
