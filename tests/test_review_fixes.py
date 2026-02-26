"""Tests for PR review fixes: dispatch table validation, config edge cases."""

import math
import subprocess
import sys

import numpy as np
import pytest

from jamma.lmm.io import (
    HEADER_ALL,
    HEADER_LRT,
    HEADER_SCORE,
    HEADER_WALD,
    IncrementalAssocWriter,
    format_assoc_line,
)
from jamma.lmm.results import _build_results
from jamma.lmm.schema import FORMAT_COLUMNS, HEADERS, RESULT_FIELDS
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
        """Verify HEADERS dict matches named constants in io.py."""
        assert HEADERS["wald"] == HEADER_WALD
        assert HEADERS["score"] == HEADER_SCORE
        assert HEADERS["lrt"] == HEADER_LRT
        assert HEADERS["all"] == HEADER_ALL

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
# ensure_jax_configured reconfig test (#12)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.requires_jax
class TestEnsureJaxConfigured:
    """Test ensure_jax_configured locking behavior."""

    def test_reconfig_with_non_default_args_raises(self) -> None:
        """Calling with non-default args after config should raise RuntimeError."""
        from jamma.core import jax_config

        # Save and reset state
        original = jax_config._jax_configured
        jax_config._jax_configured = True
        try:
            with pytest.raises(RuntimeError, match="non-default args"):
                jax_config.ensure_jax_configured(enable_x64=False)
            with pytest.raises(RuntimeError, match="non-default args"):
                jax_config.ensure_jax_configured(platform="cpu")
        finally:
            jax_config._jax_configured = original

    def test_reconfig_with_defaults_is_noop(self) -> None:
        """Calling with default args after config should silently succeed."""
        from jamma.core import jax_config

        original = jax_config._jax_configured
        jax_config._jax_configured = True
        try:
            # Should not raise
            jax_config.ensure_jax_configured()
        finally:
            jax_config._jax_configured = original

    def test_configure_jax_sets_configured_flag(self) -> None:
        """Direct configure_jax() should set _jax_configured so no false warning."""
        from jamma.core import jax_config

        original = jax_config._jax_configured
        jax_config._jax_configured = False
        try:
            jax_config.configure_jax()
            assert jax_config._jax_configured is True
        finally:
            jax_config._jax_configured = original


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
