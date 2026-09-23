"""Tests for memory estimation module."""

import numpy as np
import pytest

from jamma.core import (
    MemorySnapshot,
    get_memory_snapshot,
    log_memory_snapshot,
)
from jamma.core.eigen_plan import _dsyevd_peak_gb, dsyevr_peak_gb
from jamma.core.estimates import _format_duration
from jamma.core.memory import (
    array_gb,
    fits,
    headroom_gb,
    margin_gb,
    require,
)
from jamma.lmm.chunk_sizing import lmm_extra_bytes_per_snp
from jamma.lmm.dispatch import DispatchPath
from tests.builders import association_price_plan
from tests.fakes.memory import use_fake_psutil

pytestmark = pytest.mark.tier0


def _fallback_uab_iab_gb(n_samples: int, chunk_size: int, n_cvt: int = 1) -> float:
    """The full Uab+Iab batch the NumPy fallback materialises, in GB.

    The fallback is the largest of the four paths, so these tests price with it.
    """
    return (
        chunk_size
        * lmm_extra_bytes_per_snp(n_samples, n_cvt, DispatchPath.NUMPY_FALLBACK)
        / 1e9
    )


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


def _batch_quote_gb(
    n_samples: int, n_snps: int, chunk_size: int = 20_000, n_cvt: int = 1
) -> float:
    """The batch association quote on the NumPy fallback, the largest path."""
    plan = association_price_plan(
        "batch",
        n_samples=n_samples,
        n_snps=n_snps,
        chunk_size=chunk_size,
        n_cvt=n_cvt,
    )
    return plan.price(eigen=None).association_gb


class TestAssociationQuote:
    """The association phase as ``price()`` quotes it."""

    def test_batch_quote_prices_eigenvectors_but_no_kinship_or_workspace(self):
        """U, the whole genotype matrix, one rotation buffer, and one Uab/Iab chunk."""
        n_samples, n_snps, batch = 100_000, 10_000, 20_000
        total = _batch_quote_gb(n_samples, n_snps, batch)
        assert total == pytest.approx(
            array_gb(n_samples, n_samples)
            + array_gb(n_samples, n_snps)
            + array_gb(n_samples, batch)
            + _fallback_uab_iab_gb(n_samples, batch)
        )
        assert total < _dsyevd_peak_gb(n_samples)

    def test_batch_quote_100k_under_200gb(self):
        """At 100k samples with 100 SNPs, LMM should need under 200GB.

        This is the exact scenario from the xlarge benchmark bug:
        300.6GB available, but old check demanded 320GB (eigendecomp peak).
        """
        est = _batch_quote_gb(100_000, 100)
        assert est < 200, (
            f"LMM for 100k samples × 100 SNPs should need <200GB, got {est:.1f}GB"
        )

    @pytest.mark.parametrize("mode", ["streaming", "loco"])
    def test_chunked_quote_is_independent_of_n_snps(self, mode):
        """Streaming and LOCO hold one genotype chunk, never the whole matrix."""
        few, many = (
            association_price_plan(
                mode, n_samples=10_000, n_snps=n_snps, chunk_size=5_000
            )
            .price(eigen=None)
            .association_gb
            for n_snps in (10_000, 10_000_000)
        )
        assert few == many


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
        """The fallback figure the caller supplies covers what runtime allocates.

        Runtime allocates:
        - Uab_batch: (chunk_size, n_samples, n_index) float64
        - Iab_batch: (chunk_size, n_cvt+2, n_index) float64
        """
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2

        # Actual allocation sizes
        uab_bytes = chunk_size * n_samples * n_index * 8
        iab_bytes = chunk_size * (n_cvt + 2) * n_index * 8
        actual_uab_iab_gb = (uab_bytes + iab_bytes) / 1e9

        priced_gb = _fallback_uab_iab_gb(n_samples, chunk_size, n_cvt)

        assert abs(priced_gb - actual_uab_iab_gb) < 1e-9, (
            f"the fallback price for ({n_samples}, {chunk_size}, {n_cvt}) is "
            f"{priced_gb:.6f}GB but actual is {actual_uab_iab_gb:.6f}GB"
        )


class TestKinshipDtypeAccounting:
    """Verify memory model accounts for float64 genotype copy in kinship."""

    def test_batch_genotypes_are_priced_as_float64(self):
        """The batch quote holds the genotype matrix as float64."""
        n_samples = 10_000
        n_snps = 50_000

        growth = _batch_quote_gb(n_samples, n_snps) - _batch_quote_gb(n_samples, 0)

        expected_gb = n_samples * n_snps * 8 / 1e9
        assert growth == pytest.approx(expected_gb)

    def test_batch_quote_grows_with_n_cvt(self):
        """The NumPy-fallback batch quote must grow with n_cvt.

        Regression: callers that forget to pass n_cvt silently get the
        default (n_cvt=1) estimate, which underestimates Uab/Iab for
        multi-covariate runs and lets preflight pass before the real
        allocation OOMs.
        """
        quote = {
            n_cvt: _batch_quote_gb(10_000, 1_000, 5_000, n_cvt) for n_cvt in (1, 5, 20)
        }

        assert quote[5] > quote[1], (
            f"n_cvt=5 ({quote[5]:.4f}GB) should exceed n_cvt=1 ({quote[1]:.4f}GB)"
        )
        assert quote[20] > quote[5], (
            f"n_cvt=20 ({quote[20]:.4f}GB) should exceed n_cvt=5 ({quote[5]:.4f}GB)"
        )


class TestGateCorrectnessLmmMemory:
    """Tests that the LMM batch runner memory gate correctly blocks/passes."""

    def test_lmm_gate_passes_with_ample_memory(self):
        """The gate passes when plenty of memory is available."""
        required = _batch_quote_gb(1_000, 1_000)
        assert fits(required, 500.0) is True

    def test_lmm_gate_blocks_with_scarce_memory(self):
        """The gate fails when memory is insufficient.

        100k samples needs ~80GB of eigenvectors alone.
        """
        required = _batch_quote_gb(100_000, 10_000)
        assert fits(required, 1.0) is False

    def test_lmm_gate_threshold_boundary(self):
        """The gate accounts for the safety margin (10% capped at 10GB)."""
        required = _batch_quote_gb(100, 100)
        needed = required + margin_gb(required)

        assert fits(required, needed + 0.001) is True
        assert fits(required, needed - 0.001) is False


class TestSafetyMarginCap:
    """Verify 10GB absolute cap on safety margin."""

    def test_margin_capped_at_10gb_for_large_requirements(self):
        """Safety margin caps at 10GB for large memory requirements."""
        # 500GB required: uncapped would be 500*1.1 = 550GB, capped is 510GB.
        require(500.0, 515.0)

        with pytest.raises(MemoryError):
            require(500.0, 505.0)

    def test_small_requirements_use_percentage_margin(self):
        """Small requirements use 10% margin (not capped)."""
        # 10GB required: margin = min(1, 10) = 1GB, so 11GB is needed.
        require(10.0, 11.5)

        with pytest.raises(MemoryError):
            require(10.0, 10.5)


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
class TestEigenPeakMatchesDocs:
    """The eigendecomposition peaks the user guide's sample-limit table assumes."""

    def test_100k_dsyevd_peak_matches_docs_user_guide(self):
        """docs/USER_GUIDE.md's approximate-sample-limits table assumes this peak.

        At n=100k: K (80GB) + U (80GB) + DSYEVD workspace (~160GB) = ~320GB.
        """
        assert 315 < _dsyevd_peak_gb(100_000) < 325

    def test_100k_dsyevr_peak_matches_docs_user_guide(self):
        """At n=100k: K (80GB) + U (80GB) + DSYEVR workspace (~0.03GB) = ~160GB."""
        assert 155 < dsyevr_peak_gb(100_000) < 165


@pytest.mark.tier0
class TestFitsAndRequire:
    """Tests for the fits predicate and the one require() raise site."""

    def test_fits_true_when_ample(self):
        assert fits(10.0, 1000.0) is True

    def test_fits_false_when_scarce(self):
        assert fits(1000.0, 10.0) is False

    def test_fits_and_require_agree_at_the_margin_boundary(self):
        """The predicate and the raise site apply the identical margin."""
        assert fits(500.0, 515.0) is True
        require(500.0, 515.0)

        assert fits(500.0, 505.0) is False
        with pytest.raises(MemoryError):
            require(500.0, 505.0)

    @pytest.mark.parametrize("available", [0.001, 1.0, 64.0, 110.0, 110.5, 1000.0])
    def test_headroom_is_the_largest_requirement_that_fits(self, available):
        """headroom_gb inverts required + margin_gb(required) on both margin regimes."""
        head = headroom_gb(available)
        assert head + margin_gb(head) == pytest.approx(available)
        assert fits(head * (1 - 1e-9), available) is True
        assert fits(head * (1 + 1e-9), available) is False

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
