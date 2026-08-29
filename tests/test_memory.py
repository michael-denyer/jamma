"""Tests for memory estimation module."""

import numpy as np
import pytest

from jamma.core import (
    MemoryBreakdown,
    MemorySnapshot,
    check_memory_available,
    cleanup_memory,
    get_memory_snapshot,
    log_memory_snapshot,
)
from jamma.core.eigen_plan import _dsyevd_peak_gb, dsyevr_peak_gb, square_matrix_gb
from jamma.core.estimates import _format_duration
from jamma.core.memory import (
    _uab_iab_gb,
    eigen_cost,
    estimate_lmm_memory,
    fits,
    kinship_cost,
    lmm_cost,
    require,
)
from tests.builders import BOUNDARY_SIZES
from tests.fakes.memory import use_fake_psutil

pytestmark = pytest.mark.tier0


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
            check_memory_available(1_000_000, operation="kinship")

        msg = str(exc_info.value)
        assert "1000000" in msg or "1e+06" in msg.lower()  # Required amount
        assert "GB available" in msg  # Available amount
        assert "kinship" in msg  # Operation name


class TestEigendecompMemoryGate:
    """Integration: eigendecompose_kinship respects check_memory flag."""

    def test_eigendecomp_raises_on_insufficient_memory(self, monkeypatch):
        """MemoryError raised when memory is scarce.

        Mocks psutil.virtual_memory to report 1 byte available.
        Should raise MemoryError before LAPACK runs.
        """

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = rng.standard_normal((50, 50))
        K = (K + K.T) / 2

        use_fake_psutil(monkeypatch, available=1, total=1)

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
        snap = cleanup_memory(verbose=False)

        assert isinstance(snap, MemorySnapshot)
        assert snap.rss_gb > 0

    def test_cleanup_memory_verbose_logs(self):
        """cleanup_memory with verbose=True runs gc and returns valid snapshot."""
        from unittest.mock import patch

        with patch("gc.collect", wraps=__import__("gc").collect) as mock_gc:
            snap = cleanup_memory(verbose=True)

        mock_gc.assert_called()

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


class TestLmmMemoryEstimation:
    """Tests for estimate_lmm_memory function (LMM-phase only)."""

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


class TestKinshipDtypeAccounting:
    """Verify memory model accounts for float64 genotype copy in kinship."""

    def test_lmm_genotypes_gb_is_float64(self):
        """estimate_lmm_memory must use float64 for genotypes."""
        n_samples = 10_000
        n_snps = 50_000

        est = estimate_lmm_memory(n_samples, n_snps)

        expected_gb = n_samples * n_snps * 8 / 1e9
        assert abs(est.genotypes_gb - expected_gb) < 1e-9

    def test_lmm_batch_gb_grows_with_n_cvt(self):
        """estimate_lmm_memory.lmm_batch_gb must grow with n_cvt.

        Regression: callers that forget to pass n_cvt silently get the
        default (n_cvt=1) estimate, which underestimates Uab/Iab for
        multi-covariate runs and lets preflight pass before the real
        allocation OOMs. The estimator itself correctly scales with
        n_cvt — this test pins that contract so any fix that re-breaks
        it fails loudly.
        """
        n_samples = 10_000
        batch_size = 5_000

        est_1 = estimate_lmm_memory(
            n_samples, 1_000, lmm_batch_size=batch_size, n_cvt=1
        )
        est_5 = estimate_lmm_memory(
            n_samples, 1_000, lmm_batch_size=batch_size, n_cvt=5
        )
        est_20 = estimate_lmm_memory(
            n_samples, 1_000, lmm_batch_size=batch_size, n_cvt=20
        )

        # lmm_batch_gb is the only component that depends on n_cvt;
        # all other fields (eigenvectors, genotypes, etc.) are invariant.
        assert est_5.lmm_batch_gb > est_1.lmm_batch_gb, (
            f"n_cvt=5 ({est_5.lmm_batch_gb:.4f}GB) should exceed "
            f"n_cvt=1 ({est_1.lmm_batch_gb:.4f}GB)"
        )
        assert est_20.lmm_batch_gb > est_5.lmm_batch_gb, (
            f"n_cvt=20 ({est_20.lmm_batch_gb:.4f}GB) should exceed "
            f"n_cvt=5 ({est_5.lmm_batch_gb:.4f}GB)"
        )


class TestGateCorrectnessLmmMemory:
    """Tests that LMM batch runner memory gate correctly blocks/passes."""

    def test_lmm_gate_passes_with_ample_memory(self, monkeypatch):
        """Memory check should pass when plenty of memory is available."""

        use_fake_psutil(monkeypatch, available=500 * 1e9)  # 500GB

        est = estimate_lmm_memory(1_000, 1_000)
        assert est.sufficient is True

    def test_lmm_gate_blocks_with_scarce_memory(self, monkeypatch):
        """Memory check should fail when memory is insufficient."""

        use_fake_psutil(monkeypatch, available=1 * 1e9)  # 1GB

        # 100k samples needs ~80GB eigenvectors alone
        est = estimate_lmm_memory(100_000, 10_000)
        assert est.sufficient is False

    def test_lmm_gate_threshold_boundary(self, monkeypatch):
        """Memory check should account for safety margin (10% capped at 10GB).

        _check_available uses: (total_gb + min(total_gb * 0.1, 10)) < available_gb.
        """

        # Compute total_gb deterministically (pin memory so it doesn't affect total)
        use_fake_psutil(monkeypatch, available=1000 * 1e9)
        est_dry = estimate_lmm_memory(100, 100)

        margin = min(est_dry.total_gb * 0.1, 10.0)
        needed_with_margin = est_dry.total_gb + margin

        # Set available to just above the margin (should pass)
        use_fake_psutil(monkeypatch, available=(needed_with_margin + 0.001) * 1e9)
        est = estimate_lmm_memory(100, 100)
        assert est.sufficient is True

        # Set available to just under the margin (should fail)
        use_fake_psutil(monkeypatch, available=(needed_with_margin - 0.001) * 1e9)
        est = estimate_lmm_memory(100, 100)
        assert est.sufficient is False


class TestSafetyMarginCap:
    """Verify 10GB absolute cap on safety margin."""

    def test_margin_capped_at_10gb_for_large_requirements(self, monkeypatch):
        """Safety margin caps at 10GB for large memory requirements."""

        # 500GB required: old formula = 500*1.1 = 550GB needed
        # new formula = 500 + min(50, 10) = 510GB needed
        use_fake_psutil(monkeypatch, available=515 * 1e9)  # 515GB
        # Should PASS with capped margin (510GB < 515GB)
        assert check_memory_available(500.0) is True

        use_fake_psutil(monkeypatch, available=505 * 1e9)  # 505GB
        # Should FAIL (510GB > 505GB)
        with pytest.raises(MemoryError):
            check_memory_available(500.0)

    def test_small_requirements_use_percentage_margin(self, monkeypatch):
        """Small requirements use 10% margin (not capped)."""

        # 10GB required: margin = min(1, 10) = 1GB, total = 11GB
        use_fake_psutil(monkeypatch, available=11.5 * 1e9)
        assert check_memory_available(10.0) is True

        use_fake_psutil(monkeypatch, available=10.5 * 1e9)  # 10.5 < 11
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


class TestUabIabGb:
    """Tests for Uab/Iab memory estimation."""

    def test_uab_iab_gb_formula(self):
        """_uab_iab_gb combines the Uab_batch and Iab_batch float64 buffers."""
        result = _uab_iab_gb(1000, 500, n_cvt=1)
        n_index = (1 + 3) * (1 + 2) // 2  # 6
        expected = (500 * 1000 * n_index * 8 + 500 * 3 * n_index * 8) / 1e9
        assert result == pytest.approx(expected)


@pytest.mark.tier0
class TestPhaseCostFunctions:
    """Table-driven checks over the three phase cost functions.

    Each phase function must reproduce the same peak a hand-rolled formula
    gives at every size in ``BOUNDARY_SIZES`` — the same sizes the jlinalg
    BLAS tests sweep. This is a re-homing refactor (E5): the phase functions
    replace inline arithmetic that used to live only inside
    ``estimate_streaming_memory``, and must not shift any estimate.
    """

    @pytest.mark.parametrize("n", BOUNDARY_SIZES)
    def test_kinship_cost_is_accumulator_plus_chunk_plus_scratch(self, n):
        """kinship_cost = kinship accumulator + one genotype chunk + scratch."""
        kinship_gb = square_matrix_gb(n)
        chunk_gb = n * 1000 * 8 / 1e9
        scratch_gb = 1.5
        assert kinship_cost(kinship_gb, chunk_gb, scratch_gb) == pytest.approx(
            kinship_gb + chunk_gb + scratch_gb
        )

    @pytest.mark.parametrize("n", BOUNDARY_SIZES)
    def test_eigen_cost_defaults_to_dsyevd_peak(self, n):
        """With no driver-aware figure, eigen_cost is the conservative DSYEVD peak."""
        assert eigen_cost(n) == pytest.approx(_dsyevd_peak_gb(n))

    @pytest.mark.parametrize("n", BOUNDARY_SIZES)
    def test_eigen_cost_uses_caller_supplied_peak(self, n):
        """A caller-supplied peak (e.g. from plan_eigen_driver) wins outright."""
        driver_peak = dsyevr_peak_gb(n)
        assert eigen_cost(n, driver_peak) == pytest.approx(driver_peak)

    @pytest.mark.parametrize("n", BOUNDARY_SIZES)
    def test_lmm_cost_sums_its_five_components(self, n):
        """lmm_cost = eigenvectors + chunk + rotation + grid REML + Uab/Iab."""
        eigenvectors_gb = square_matrix_gb(n)
        lmm_chunk_gb = n * 500 * 8 / 1e9
        rotation_buffer_gb = n * 500 * 8 / 1e9
        grid_reml_gb = 50 * 500 * 8 / 1e9
        uab_iab_gb = _uab_iab_gb(n, 500, n_cvt=1)
        assert lmm_cost(
            eigenvectors_gb, lmm_chunk_gb, rotation_buffer_gb, grid_reml_gb, uab_iab_gb
        ) == pytest.approx(
            eigenvectors_gb
            + lmm_chunk_gb
            + rotation_buffer_gb
            + grid_reml_gb
            + uab_iab_gb
        )

    def test_100k_dsyevd_peak_matches_docs_user_guide(self):
        """docs/USER_GUIDE.md's approximate-sample-limits table assumes this peak.

        At n=100k: K (80GB) + U (80GB) + DSYEVD workspace (~160GB) = ~320GB.
        """
        assert 315 < eigen_cost(100_000) < 325

    def test_100k_dsyevr_peak_matches_docs_user_guide(self):
        """At n=100k: K (80GB) + U (80GB) + DSYEVR workspace (~0.03GB) = ~160GB."""
        assert 155 < eigen_cost(100_000, dsyevr_peak_gb(100_000)) < 165


@pytest.mark.tier0
class TestFitsAndRequire:
    """Tests for the fits predicate and the one require() raise site."""

    def test_fits_true_when_ample(self):
        assert fits(10.0, 1000.0) is True

    def test_fits_false_when_scarce(self):
        assert fits(1000.0, 10.0) is False

    def test_fits_matches_check_memory_available_boundary(self, monkeypatch):
        """fits() and check_memory_available() apply the identical margin."""
        use_fake_psutil(monkeypatch, available=515 * 1e9)
        assert fits(500.0, 515.0) is True
        assert check_memory_available(500.0) is True

        use_fake_psutil(monkeypatch, available=505 * 1e9)
        assert fits(500.0, 505.0) is False
        with pytest.raises(MemoryError):
            check_memory_available(500.0)

    def test_require_passes_when_sufficient(self):
        require(1.0, 1000.0, "test")

    def test_require_raises_insufficient_memory(self):
        with pytest.raises(MemoryError, match="Insufficient memory"):
            require(1000.0, 1.0, "test")

    def test_require_raises_budget_exceeded(self):
        with pytest.raises(MemoryError, match="exceeds"):
            require(500.0, 1000.0, "test", budget_gb=10.0)

    def test_require_checks_budget_before_availability(self):
        """A budget below the requirement fails even when memory is ample."""
        with pytest.raises(MemoryError, match="budget"):
            require(500.0, 1e9, "test", budget_gb=10.0)
