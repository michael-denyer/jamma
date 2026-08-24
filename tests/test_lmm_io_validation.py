"""LMM I/O and dispatch validation.

Covers format_assoc_line per test_type, IncrementalAssocWriter rejection of
unknown test_types, _build_results field mapping per lmm_mode, the
``python -m jamma --help`` smoke test, the erfc-vs-chi2.sf equivalence used
by HWE p-values, and degenerate-SNP NaN propagation through the NumPy
batch-stat functions.
"""

import math
import subprocess
import sys

import numpy as np
import pytest

from jamma.lmm.io import (
    HEADER_WALD,
    IncrementalAssocWriter,
    format_assoc_line,
)
from jamma.lmm.results import _build_results
from jamma.lmm.schema import FORMAT_COLUMNS, HEADERS, RESULT_FIELDS, LmmConfig
from jamma.lmm.stats import AssocResult

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> AssocResult:
    """Create an AssocResult with all fields populated."""
    defaults = {
        "chr": "1",
        "rs": "rs123",
        "ps": 1000,
        "n_miss": 0,
        "allele1": "A",
        "allele0": "G",
        "af": 0.25,
        "beta": 0.5,
        "se": 0.1,
        "logl_H1": -100.0,
        "l_remle": 1.5,
        "p_wald": 0.01,
        "p_score": 0.02,
        "l_mle": 1.6,
        "p_lrt": 0.03,
    }
    defaults.update(overrides)
    return AssocResult(**defaults)


# ---------------------------------------------------------------------------
# format_assoc_line tests (#6)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestFormatAssocLine:
    """Verify format_assoc_line produces correct columns for each test_type."""

    @pytest.mark.parametrize("test_type", ["wald", "score", "lrt", "all"])
    def test_column_count_matches_header(self, test_type: str) -> None:
        """Each test_type line should have same number of columns as its header."""
        result = _make_result()
        line = format_assoc_line(result, test_type)
        header = HEADERS[test_type]
        assert len(line.split("\t")) == len(header.split("\t"))

    @pytest.mark.parametrize("test_type", ["wald", "score", "lrt", "all"])
    def test_stat_columns_match_format_columns(self, test_type: str) -> None:
        """Stat columns (after 7-column prefix) should match FORMAT_COLUMNS."""
        result = _make_result()
        line = format_assoc_line(result, test_type)
        parts = line.split("\t")
        stat_parts = parts[7:]  # Skip 7-column prefix
        expected_cols = FORMAT_COLUMNS[test_type]
        assert len(stat_parts) == len(expected_cols)
        # Verify each stat column is the correct field value
        for col_name, col_val in zip(expected_cols, stat_parts, strict=True):
            expected_val = getattr(result, col_name)
            assert float(col_val) == pytest.approx(expected_val, rel=1e-5)

    def test_invalid_test_type_raises_value_error(self) -> None:
        """Invalid test_type should raise ValueError, not KeyError."""
        result = _make_result()
        with pytest.raises(ValueError, match="Unknown test_type"):
            format_assoc_line(result, "waldd")

    def test_headers_generated_from_format_columns(self) -> None:
        """Verify HEADERS dict matches the named constant in io.py."""
        assert HEADERS["wald"] == HEADER_WALD

    def test_writer_rejects_invalid_test_type(self, tmp_path) -> None:
        """IncrementalAssocWriter should reject invalid test_type at init."""
        with pytest.raises(ValueError, match="Unknown test_type"):
            IncrementalAssocWriter(tmp_path / "out.txt", test_type="bad")


# ---------------------------------------------------------------------------
# _build_results tests (#7)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestBuildResults:
    """Verify _build_results field mapping for each lmm_mode."""

    def _make_arrays(self, lmm_mode: int, n: int = 3) -> dict[str, np.ndarray]:
        """Create arrays dict matching RESULT_FIELDS for the given mode."""
        return {
            key: np.arange(n, dtype=np.float64) + 1.0 for key in RESULT_FIELDS[lmm_mode]
        }

    def _make_snp_info(self, n: int = 3) -> list[dict]:
        return [
            {"chr": "1", "rs": f"rs{i}", "pos": i * 100, "a1": "A", "a0": "G"}
            for i in range(n)
        ]

    @pytest.mark.parametrize("lmm_mode", [1, 2, 3, 4])
    def test_correct_fields_populated(self, lmm_mode: int) -> None:
        """Each mode should populate exactly the fields in RESULT_FIELDS."""
        n = 3
        arrays = self._make_arrays(lmm_mode, n)
        snp_indices = np.arange(n)
        afs = np.full(n, 0.3)
        miss = np.zeros(n, dtype=int)
        snp_info = self._make_snp_info(n)

        results = _build_results(lmm_mode, snp_indices, afs, miss, snp_info, arrays)
        assert len(results) == n

        field_map = RESULT_FIELDS[lmm_mode]
        for j, r in enumerate(results):
            for array_key, field_name in field_map.items():
                val = getattr(r, field_name)
                assert val is not None, (
                    f"Field {field_name} is None for mode {lmm_mode}"
                )
                assert val == pytest.approx(float(arrays[array_key][j]))

    def test_lrt_mode_has_nan_beta_se(self) -> None:
        """LRT mode (2) should set beta and se to NaN."""
        n = 2
        arrays = self._make_arrays(2, n)
        results = _build_results(
            2,
            np.arange(n),
            np.full(n, 0.3),
            np.zeros(n, dtype=int),
            self._make_snp_info(n),
            arrays,
        )
        for r in results:
            assert math.isnan(r.beta)
            assert math.isnan(r.se)

    def test_invalid_lmm_mode_raises_value_error(self) -> None:
        """Invalid lmm_mode should raise ValueError, not KeyError."""
        snp = [{"chr": "1", "rs": "x", "pos": 0, "a1": "A", "a0": "G"}]
        with pytest.raises(ValueError, match="Unknown lmm_mode"):
            _build_results(
                99,
                np.array([0]),
                np.array([0.3]),
                np.array([0]),
                snp,
                {},
            )

    def test_missing_array_key_raises_value_error(self) -> None:
        """Missing array key should raise ValueError with helpful message."""
        n = 1
        # Provide incomplete arrays for mode 1 (missing 'pwalds')
        arrays = {k: np.ones(n) for k in list(RESULT_FIELDS[1].keys())[:-1]}
        with pytest.raises(ValueError, match="Missing arrays"):
            _build_results(
                1,
                np.arange(n),
                np.full(n, 0.3),
                np.zeros(n, dtype=int),
                self._make_snp_info(n),
                arrays,
            )


# ---------------------------------------------------------------------------
# __main__.py smoke test (#13)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_python_m_jamma_help() -> None:
    """Verify 'python -m jamma --help' works and shows usage."""
    result = subprocess.run(
        [sys.executable, "-m", "jamma", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout or "usage:" in result.stdout.lower()


# ---------------------------------------------------------------------------
# erfc vs chi2.sf equivalence test (#14)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_erfc_matches_chi2_sf() -> None:
    """Verify math.erfc HWE computation matches scipy.stats.chi2.sf for df=1."""
    scipy_stats = pytest.importorskip("scipy.stats")

    chi_sq_values = np.array([0.0, 0.001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0])

    erfc_pvalues = np.array([math.erfc(math.sqrt(x / 2.0)) for x in chi_sq_values])
    scipy_pvalues = scipy_stats.chi2.sf(chi_sq_values, df=1)

    # For reasonable chi_sq values, should match to high precision
    np.testing.assert_allclose(erfc_pvalues, scipy_pvalues, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# PR #17 review fixes
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestDegenerateSNPNaN:
    """NumPy batch stat functions must return NaN for degenerate (constant) SNPs."""

    def test_wald_degenerate_snps_return_nan(self) -> None:
        """batch_calc_wald_stats_numpy returns NaN for zero-variance SNPs."""
        from jamma.lmm.likelihood_numpy import (
            batch_calc_wald_stats_numpy,
            batch_compute_iab_numpy,
            batch_compute_uab_numpy,
            golden_section_optimize_lambda_numpy,
        )

        rng = np.random.default_rng(42)
        n_samples, n_snps, n_cvt = 50, 5, 1

        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
        UtW = rng.standard_normal((n_samples, n_cvt))
        Uty = rng.standard_normal(n_samples)
        UtG = rng.standard_normal((n_samples, n_snps))
        # Make SNPs 0 and 3 constant (zero variance after rotation)
        UtG[:, 0] = 0.0
        UtG[:, 3] = 0.0

        Uab = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
        Iab = batch_compute_iab_numpy(n_cvt, Uab)
        lambdas, _, _ = golden_section_optimize_lambda_numpy(
            n_cvt, eigenvalues, Uab, Iab
        )
        betas, ses, pwalds = batch_calc_wald_stats_numpy(
            n_cvt, lambdas, eigenvalues, Uab, n_samples
        )

        # Degenerate SNPs should be NaN
        for idx in [0, 3]:
            assert np.isnan(betas[idx]), f"beta[{idx}] should be NaN"
            assert np.isnan(ses[idx]), f"se[{idx}] should be NaN"
            assert np.isnan(pwalds[idx]), f"p_wald[{idx}] should be NaN"

        # Valid SNPs should be finite
        for idx in [1, 2, 4]:
            assert np.isfinite(betas[idx]), f"beta[{idx}] should be finite"
            assert np.isfinite(ses[idx]), f"se[{idx}] should be finite"
            assert np.isfinite(pwalds[idx]), f"p_wald[{idx}] should be finite"

    def test_score_degenerate_snps_return_nan(self) -> None:
        """batch_calc_score_stats_numpy returns NaN for zero-variance SNPs."""
        from jamma.lmm.likelihood_numpy import (
            batch_calc_score_stats_numpy,
            batch_compute_uab_numpy,
        )

        rng = np.random.default_rng(42)
        n_samples, n_snps, n_cvt = 50, 4, 1

        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
        UtW = rng.standard_normal((n_samples, n_cvt))
        Uty = rng.standard_normal(n_samples)
        UtG = rng.standard_normal((n_samples, n_snps))
        UtG[:, 1] = 0.0  # Make SNP 1 constant

        Uab = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
        Hi_eval = 1.0 / (1.0 * eigenvalues + 1.0)

        betas, ses, p_scores = batch_calc_score_stats_numpy(
            n_cvt, Hi_eval, Uab, n_samples
        )

        assert np.isnan(betas[1])
        assert np.isnan(ses[1])
        assert np.isnan(p_scores[1])
        assert np.isfinite(betas[0])
        assert np.isfinite(p_scores[0])


@pytest.mark.tier0
class TestNegativeLRTClamp:
    """_batch_lrt_pvalues_numpy must clamp negative LRT stats to 0."""

    def test_negative_lrt_returns_pvalue_one(self) -> None:
        """When H1 logl < H0 logl, LRT stat is negative → p-value should be 1.0."""
        from jamma.lmm.likelihood_numpy import _batch_lrt_pvalues_numpy

        logl_H0 = -100.0
        # Some H1 logls worse than null (negative LRT stat)
        logls_mle = np.array([-101.0, -105.0, -100.5, -99.0, -98.0])
        p_lrts = _batch_lrt_pvalues_numpy(logls_mle, logl_H0)

        # First 3 have worse H1 → clamped to 0 → chi2_sf(0) = 1.0
        np.testing.assert_allclose(p_lrts[:3], 1.0)
        # Last 2 have better H1 → valid p-values < 1
        assert all(p_lrts[3:] < 1.0)


@pytest.mark.tier0
class TestNumpyRunnerValidation:
    """run_lmm_association_numpy input validation tests."""

    def test_eigenvalue_without_eigenvector_raises(self) -> None:
        """Providing eigenvalues without eigenvectors raises ValueError."""
        from jamma.lmm.runner_numpy import run_lmm_association_numpy

        with pytest.raises(ValueError, match="Must provide both"):
            run_lmm_association_numpy(
                genotypes=np.ones((10, 5)),
                phenotypes=np.ones(10),
                kinship=np.eye(10),
                snp_info=[{"chr": "1", "rs": "x", "pos": 0, "a1": "A", "a0": "G"}] * 5,
                eigenvalues=np.ones(10),
                eigenvectors=None,
                config=LmmConfig(check_memory=False, show_progress=False),
            )

    # No invalid-lmm_mode test here. It read as runner validation but the raise
    # came from LmmConfig(lmm_mode=5) while the argument list was evaluated, so
    # the runner was never entered. LmmConfig owns the rule and
    # test_validate_runner_inputs.py::TestLmmConfigValidation covers 0, -1, 5, 99.


@pytest.mark.tier0
class TestComputeNumpyInvalidMode:
    """compute_lmm_chunk_numpy must raise on invalid lmm_mode."""

    def test_invalid_mode_raises_value_error(self) -> None:
        """lmm_mode=99 should raise ValueError, not return all-None dict."""
        from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy

        with pytest.raises(ValueError, match="lmm_mode must be"):
            compute_lmm_chunk_numpy(
                lmm_mode=99,  # type: ignore[bad-argument-type]
                n_cvt=1,
                eigenvalues=np.ones(10),
                Uab_batch=np.ones((5, 10, 3)),
                n_samples=10,
            )


@pytest.mark.tier0
class TestBetaincValidation:
    """betainc must validate a > 0 and b > 0."""

    def test_a_zero_raises(self) -> None:
        from jamma.lmm.special import betainc

        with pytest.raises(ValueError, match="a must be > 0"):
            betainc(0.0, 0.5, 0.5)

    def test_b_negative_raises(self) -> None:
        from jamma.lmm.special import betainc

        with pytest.raises(ValueError, match="b must be > 0"):
            betainc(1.0, -1.0, 0.5)


@pytest.mark.tier0
@pytest.mark.tier0
class TestMode4WaldOverwritesScore:
    """Mode=4 must use Wald betas/ses, not Score's.

    The mode=4 composition runs Score first, then LRT, then Wald. Wald's
    REML-optimized beta/SE must overwrite Score's values. This test verifies
    the overwrite invariant by comparing mode=4 betas/ses against mode=1
    (Wald-only) — they must match exactly.
    """

    def test_mode4_betas_ses_match_wald_only(self, monkeypatch) -> None:
        """Mode=4 betas/ses must equal mode=1 (Wald) betas/ses.

        Driven through the full-Uab NumPy path with the extension cleared, which
        is the only state the runner reaches compute_lmm_chunk_numpy in.
        """
        from jamma.lmm import compute_numpy as cn
        from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy
        from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy
        from jamma.lmm.prepare_common import _compute_null_model_common

        monkeypatch.setattr(cn, "_accel", None)

        rng = np.random.default_rng(99)
        n_samples, n_snps, n_cvt = 50, 10, 1

        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
        UtW = rng.standard_normal((n_samples, n_cvt))
        Uty = rng.standard_normal(n_samples)
        UtG = rng.standard_normal((n_samples, n_snps))

        Uab = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)

        logl_H0, _, Hi_eval_null = _compute_null_model_common(
            4, eigenvalues, UtW, Uty, n_cvt, show_progress=False
        )

        # Mode 1 — Wald only
        wald_only = compute_lmm_chunk_numpy(
            lmm_mode=1,
            n_cvt=n_cvt,
            eigenvalues=eigenvalues,
            Uab_batch=Uab,
            n_samples=n_samples,
        )

        # Mode 4 — All tests composed
        all_tests = compute_lmm_chunk_numpy(
            lmm_mode=4,
            n_cvt=n_cvt,
            eigenvalues=eigenvalues,
            Uab_batch=Uab,
            n_samples=n_samples,
            Hi_eval_null=Hi_eval_null,
            logl_H0=logl_H0,
        )

        # Wald fields must match exactly (same code path)
        np.testing.assert_array_equal(all_tests["betas"], wald_only["betas"])
        np.testing.assert_array_equal(all_tests["ses"], wald_only["ses"])
        np.testing.assert_array_equal(all_tests["pwalds"], wald_only["pwalds"])
        np.testing.assert_array_equal(all_tests["lambdas"], wald_only["lambdas"])

        # Mode 4 must also have LRT and Score fields
        assert all_tests["p_lrts"] is not None
        assert all_tests["p_scores"] is not None


@pytest.mark.tier0
def test_jax_free_export_surface() -> None:
    """from jamma.lmm import * must not fail on NumPy-only exports."""
    import jamma.lmm as lmm_module
    from jamma.lmm import __all__ as lmm_all

    for name in lmm_all:
        assert hasattr(lmm_module, name), (
            f"jamma.lmm.__all__ lists {name!r} but it is not defined"
        )


@pytest.mark.tier0
def test_pipeline_hwe_numpy_raises() -> None:
    """HWE filtering with NumPy backend raises ValueError."""
    from pathlib import Path

    from jamma.pipeline import PipelineConfig, PipelineRunner

    config = PipelineConfig(
        bfile=Path("dummy"),
        lmm_mode=1,
        backend="numpy",
        hwe_threshold=0.001,
    )
    runner = PipelineRunner(config)
    with pytest.raises(ValueError, match=r"HWE filtering.*not supported.*NumPy.*batch"):
        runner.run()


@pytest.mark.tier0
def test_check_hwe_support_accepts_numpy_streaming() -> None:
    """HWE filtering with numpy-streaming does NOT raise."""
    from pathlib import Path

    from jamma.lmm.runner import ExecutionPlan
    from jamma.pipeline import PipelineConfig, PipelineRunner

    plan = ExecutionPlan("numpy", "streaming", "test")

    config = PipelineConfig(
        bfile=Path("dummy"),
        lmm_mode=1,
        backend="numpy-streaming",
        hwe_threshold=0.001,
    )
    runner = PipelineRunner(config)
    # Should not raise — numpy-streaming supports HWE
    runner._check_hwe_support(plan)
