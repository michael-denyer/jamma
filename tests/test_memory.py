"""Tests for memory estimation module."""

from unittest.mock import MagicMock, patch

import numpy as np
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
from jamma.core.estimates import _format_duration
from jamma.core.memory import (
    _uab_iab_gb,
    estimate_lmm_memory,
)


@pytest.mark.tier0
class TestMemoryEstimation:
    """Tests for estimate_workflow_memory function."""

    def test_memory_breakdown_200k(self):
        """Memory estimate for 200k samples - peak during eigendecomp ~1280GB."""
        est = estimate_workflow_memory(200_000, 95_000)

        # Kinship: 200k^2 * 8 / 1e9 = 320GB
        assert 319 < est.kinship_gb < 321, (
            f"Expected ~320GB kinship, got {est.kinship_gb}"
        )

        # Genotypes: 200k * 95k * 8 / 1e9 = 152GB (float64)
        assert 151 < est.genotypes_gb < 153, (
            f"Expected ~152GB genotypes (float64), got {est.genotypes_gb}"
        )

        # Eigenvectors: same as kinship = 320GB (used in LMM phase)
        assert 319 < est.eigenvectors_gb < 321

        # Eigendecomp workspace: always DSYEVD O(n^2) ~640GB at 200k
        assert est.eigendecomp_workspace_gb > 600, (
            f"DSYEVD workspace should be ~640GB at 200k, "
            f"got {est.eigendecomp_workspace_gb:.2f}GB"
        )
        # Peak: K + U + DSYEVD workspace = 320+320+640 = ~1280GB
        assert 1270 < est.total_gb < 1290, (
            f"Expected ~1280GB (K+U+workspace), got {est.total_gb}"
        )

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

    def test_peak_kinship_accounts_for_genotype_copy(self):
        """Phase 1 (kinship) accounts for numpy genotype copy during accumulation.

        During kinship accumulation, the numpy input and working copy
        coexist. The estimate must use genotypes_gb * 2, not genotypes_gb.
        """
        # Choose dimensions where kinship phase is large relative to eigendecomp
        est = estimate_workflow_memory(1_000, 500_000)

        genotypes_gb = 1_000 * 500_000 * 8 / 1e9  # 4.0 GB
        kinship_gb = 1_000**2 * 8 / 1e9  # 0.008 GB
        # peak_kinship = genotypes * 2 + kinship
        expected_peak_kinship = genotypes_gb * 2 + kinship_gb

        assert est.total_gb >= expected_peak_kinship, (
            f"total_gb ({est.total_gb:.4f}) should be >= peak_kinship "
            f"({expected_peak_kinship:.4f}) which includes working copy"
        )

    def test_sufficient_flag_correct(self):
        """Sufficient flag should reflect available vs required."""
        # Tiny estimate should always be sufficient
        est = estimate_workflow_memory(100, 100)
        assert est.sufficient is True

        # 200k estimate will not be sufficient on most machines (needs ~640GB)
        est = estimate_workflow_memory(200_000, 95_000)
        # Don't check exact match - just verify it's False for this huge estimate
        # (would need 640GB+ which no typical machine has)
        assert est.sufficient is False, (
            "200k sample workflow should exceed available memory"
        )


@pytest.mark.tier0
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


@pytest.mark.tier0
class TestEigendecompMemory:
    """Tests for eigendecomposition memory usage."""


@pytest.mark.tier0
class TestEigendecompMemoryGate:
    """Integration: eigendecompose_kinship respects check_memory flag."""

    def test_eigendecomp_raises_on_insufficient_memory(self):
        """MemoryError raised when memory is scarce.

        Mocks psutil.virtual_memory to report 1 byte available.
        Should raise MemoryError before LAPACK runs.
        """

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = rng.standard_normal((50, 50))
        K = (K + K.T) / 2

        # Build a mock that satisfies both get_memory_snapshot() and
        # check_memory_available(). Must have numeric .available and .total.
        mock_vmem = MagicMock()
        mock_vmem.available = 1  # 1 byte — definitely not enough
        mock_vmem.total = 1

        with patch("jamma.core.memory.psutil.virtual_memory", return_value=mock_vmem):
            with pytest.raises(MemoryError, match="Insufficient memory"):
                eigendecompose_kinship(K, check_memory=True)

    def test_eigendecomp_skips_check_when_disabled(self):
        """eigendecompose_kinship with check_memory=False skips memory check."""
        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = rng.standard_normal((30, 30))
        K = (K + K.T) / 2

        # Should succeed even if we don't mock memory (check_memory=False)
        eigenvalues, eigenvectors = eigendecompose_kinship(K, check_memory=False)
        assert eigenvalues.shape == (30,)
        assert eigenvectors.shape == (30, 30)


@pytest.mark.tier0
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


@pytest.mark.tier0
class TestCleanupMemory:
    """Tests for memory cleanup function."""

    def test_cleanup_memory_returns_snapshot(self):
        """cleanup_memory should return MemorySnapshot after cleanup."""
        snap = cleanup_memory(verbose=False)

        assert isinstance(snap, MemorySnapshot)
        assert snap.rss_gb > 0

    def test_cleanup_memory_verbose_logs(self):
        """cleanup_memory with verbose=True runs gc and returns valid snapshot.

        Verifies the function performs garbage collection (gc.collect) and
        returns a MemorySnapshot with valid data, not just assert True.
        """
        import gc

        # Create some garbage to collect
        _garbage = [object() for _ in range(1000)]
        del _garbage

        # Record gc generation-0 count before cleanup
        gen0_before = gc.get_count()[0]

        snap = cleanup_memory(verbose=True)

        # cleanup_memory calls gc.collect() which resets gen-0 counter
        gen0_after = gc.get_count()[0]

        # After gc.collect(), generation-0 count should be lower than before
        # (gc.collect resets uncollected count for generation 0)
        assert gen0_after <= gen0_before, (
            f"gc.collect() should have been called: "
            f"gen0 before={gen0_before}, after={gen0_after}"
        )

        # Return value must be a valid MemorySnapshot
        assert isinstance(snap, MemorySnapshot)
        assert snap.rss_gb > 0, "RSS should be positive after cleanup"
        assert snap.available_gb > 0, "Available memory should be positive"

    def test_cleanup_frees_memory_after_allocation(self):
        """Cleanup completes without error after allocating and deleting arrays.

        RSS-based assertions are non-deterministic under parallel test workers
        (-n3) because other workers allocate/free memory concurrently. This test
        verifies that the allocate/delete/gc/cleanup_memory sequence runs without
        error and returns a valid snapshot -- not that RSS decreased by a
        specific amount.
        """
        import gc

        # Allocate a moderate array
        big_array = np.zeros((1000, 1000), dtype=np.float64)  # 8MB
        _ = big_array.sum()  # Touch it

        # Delete and cleanup
        del big_array
        gc.collect()

        after = cleanup_memory(verbose=False)

        # Structural assertions: cleanup returned a valid snapshot
        assert after is not None
        assert isinstance(after.rss_gb, float)
        assert after.rss_gb > 0


@pytest.mark.tier0
class TestLmmMemoryEstimation:
    """Tests for estimate_lmm_memory function (LMM-phase only)."""

    def test_lmm_estimate_at_most_workflow(self):
        """LMM-only estimate should be <= full workflow estimate.

        With DSYEVR, eigendecomp workspace is tiny so LMM phase dominates
        the workflow total — LMM-only equals full pipeline. With DSYEVD,
        eigendecomp dominates so LMM-only is strictly less.
        """
        lmm_est = estimate_lmm_memory(100_000, 10_000)
        full_est = estimate_workflow_memory(100_000, 10_000)

        assert lmm_est.total_gb <= full_est.total_gb, (
            f"LMM-only ({lmm_est.total_gb:.1f}GB) should be <= "
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


@pytest.mark.tier0
class TestMemoryEstimateVsActualAllocation:
    """Regression tests: estimates must cover actual runtime tensor shapes.

    These tests verify that memory estimators account for the dominant
    intermediate buffers (Uab_batch, Iab_batch) created during LMM
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
        """Streaming LMM estimate must include eigenvectors + per-chunk buffers."""
        from jamma.core.memory import estimate_lmm_streaming_memory

        n_samples = 10_000
        chunk_size = 5_000

        est = estimate_lmm_streaming_memory(n_samples, 95_000, chunk_size=chunk_size)

        # Minimum must include eigenvectors + per-chunk intermediate buffers.
        # Estimators always use standard (non-fused) estimate since they don't
        # know the backend or lmm_mode.
        uab_iab_gb = _uab_iab_gb(n_samples, chunk_size, n_cvt=1, use_fused=False)
        eigenvectors_gb = n_samples**2 * 8 / 1e9

        assert est.total_peak_gb >= eigenvectors_gb + uab_iab_gb, (
            f"total_peak_gb ({est.total_peak_gb:.4f}GB) should be >= "
            f"eigenvectors ({eigenvectors_gb:.4f}GB) + "
            f"per-chunk ({uab_iab_gb:.4f}GB)"
        )


@pytest.mark.tier0
class TestKinshipDtypeAccounting:
    """Verify memory model accounts for float64 genotype copy in kinship."""

    def test_workflow_genotypes_gb_is_float64(self):
        """estimate_workflow_memory must use float64 (8 bytes) for genotypes.

        compute_centered_kinship converts genotypes to float64 via
        np.array(genotypes_filtered, dtype=np.float64).
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


@pytest.mark.tier0
class TestGateCorrectnessLmmMemory:
    """Tests that LMM batch runner memory gate correctly blocks/passes."""

    def test_lmm_gate_passes_with_ample_memory(self):
        """Memory check should pass when plenty of memory is available."""

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 500 * 1e9  # 500GB

            est = estimate_lmm_memory(1_000, 1_000)
            assert est.sufficient is True

    def test_lmm_gate_blocks_with_scarce_memory(self):
        """Memory check should fail when memory is insufficient."""

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 1 * 1e9  # 1GB

            # 100k samples needs ~80GB eigenvectors alone
            est = estimate_lmm_memory(100_000, 10_000)
            assert est.sufficient is False

    def test_lmm_gate_threshold_boundary(self):
        """Memory check should account for safety margin (10% capped at 10GB).

        _check_available uses: (total_gb + min(total_gb * 0.1, 10)) < available_gb.
        """

        # Compute total_gb deterministically (mock memory so it doesn't affect total)
        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 1000 * 1e9
            est_dry = estimate_lmm_memory(100, 100)

        margin = min(est_dry.total_gb * 0.1, 10.0)
        needed_with_margin = est_dry.total_gb + margin

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


@pytest.mark.tier0
class TestGateCorrectnessRunnerStreaming:
    """Tests that streaming runner memory gate correctly blocks/passes."""

    def test_streaming_gate_passes_with_ample_memory(self):
        """Memory check should pass when plenty of memory is available."""

        from jamma.core.memory import estimate_lmm_streaming_memory

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 500 * 1e9

            est = estimate_lmm_streaming_memory(1_000, 10_000)
            assert est.sufficient is True

    def test_streaming_gate_blocks_with_scarce_memory(self):
        """Memory check should fail when memory is insufficient."""

        from jamma.core.memory import estimate_lmm_streaming_memory

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 1 * 1e9

            est = estimate_lmm_streaming_memory(100_000, 95_000)
            assert est.sufficient is False


@pytest.mark.tier0
class TestSafetyMarginCap:
    """Verify 10GB absolute cap on safety margin."""

    def test_margin_capped_at_10gb_for_large_requirements(self):
        """Safety margin caps at 10GB for large memory requirements."""

        # 500GB required: old formula = 500*1.1 = 550GB needed
        # new formula = 500 + min(50, 10) = 510GB needed
        with patch("jamma.core.memory.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 515 * 1e9  # 515GB
            # Should PASS with capped margin (510GB < 515GB)
            assert check_memory_available(500.0) is True

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 505 * 1e9  # 505GB
            # Should FAIL (510GB > 505GB)
            with pytest.raises(MemoryError):
                check_memory_available(500.0)

    def test_small_requirements_use_percentage_margin(self):
        """Small requirements use 10% margin (not capped)."""

        # 10GB required: margin = min(1, 10) = 1GB, total = 11GB
        with patch("jamma.core.memory.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 11.5 * 1e9
            assert check_memory_available(10.0) is True

        with patch("jamma.core.memory.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 10.5 * 1e9  # 10.5 < 11
            with pytest.raises(MemoryError):
                check_memory_available(10.0)


class TestFormatDuration:
    """Tests for _format_duration human-readable formatting."""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            pytest.param(0.5, "<1s", id="sub_second"),
            pytest.param(1, "1s", id="one_second"),
            pytest.param(30, "30s", id="seconds"),
            pytest.param(59, "59s", id="just_under_minute"),
            pytest.param(59.6, "59s", id="no_60s_rounding"),
            pytest.param(60, "1 min", id="exactly_60_seconds"),
            pytest.param(120, "2 min", id="minutes"),
            pytest.param(3599, "59 min", id="no_60_min_at_boundary"),
            pytest.param(3599.5, "59 min", id="no_60_min_fractional"),
            pytest.param(3600, "1h", id="exactly_60_minutes"),
            pytest.param(5400, "1h 30m", id="hours_and_minutes"),
            pytest.param(7199, "1h 59m", id="no_60_minutes_rounding"),
            pytest.param(7199.9, "1h 59m", id="no_60_minutes_rounding_fractional"),
            pytest.param(7261, "2h 1m", id="large_duration"),
        ],
    )
    def test_format_duration(self, seconds, expected):
        """_format_duration uses truncation (not rounding) at all boundaries."""
        assert _format_duration(seconds) == expected


@pytest.mark.tier0
class TestUabIabGbFused:
    """Tests for fused Uab memory estimation."""

    def test_uab_iab_gb_fused_n_cvt1(self):
        """Fused path memory is chunk * n_samples * 8 bytes (UtG_T only)."""
        result = _uab_iab_gb(1000, 500, n_cvt=1, use_fused=True)
        expected = 500 * 1000 * 8 / 1e9
        assert result == pytest.approx(expected)

    def test_uab_iab_gb_fused_n_cvt_gt1_unchanged(self):
        """Fused path with n_cvt>1 falls back to standard calculation."""
        fused = _uab_iab_gb(1000, 500, n_cvt=2, use_fused=True)
        standard = _uab_iab_gb(1000, 500, n_cvt=2, use_fused=False)
        assert fused == standard

    def test_uab_iab_gb_fused_vs_standard_reduction(self):
        """Fused n_cvt=1 is strictly less than standard n_cvt=1."""
        fused = _uab_iab_gb(1000, 500, n_cvt=1, use_fused=True)
        standard = _uab_iab_gb(1000, 500, n_cvt=1, use_fused=False)
        assert fused < standard

    def test_uab_iab_gb_default_not_fused(self):
        """Default use_fused=False preserves existing behavior."""
        result = _uab_iab_gb(1000, 500, n_cvt=1)
        n_index = (1 + 3) * (1 + 2) // 2  # 6
        expected = (500 * 1000 * n_index * 8 + 500 * 3 * n_index * 8) / 1e9
        assert result == pytest.approx(expected)
