"""Tests for load_gemma_assoc and compare_assoc_results in validation/compare.py."""

from __future__ import annotations

import math

import pytest

from jamma.validation.compare import (
    AssocComparisonResult,
    compare_assoc_results,
    load_gemma_assoc,
)
from jamma.validation.tolerances import LambdaBoundaryPolicy, ToleranceConfig
from tests.assoc_test_helpers import make_assoc as _make_assoc
from tests.fakes.assoc_files import (
    ALL_TESTS_COLS,
    ALL_TESTS_FULL_COLS,
    LRT_COLS,
    LRT_FULL_COLS,
    SCORE_COLS,
    WALD_FULL_COLS,
    WALD_SHORT_COLS,
    write_assoc,
)

pytestmark = pytest.mark.tier0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Unique per-column sentinel values — a field-offset bug swaps columns, and
# distinct values make swaps visible. Columns use 0.11, 0.22, 0.33, ...
_SENTINEL_AF = 0.11
_SENTINEL_BETA = 0.22
_SENTINEL_SE = 0.33
_SENTINEL_LOGL = -100.44  # negative to distinguish from positives
_SENTINEL_L_REMLE = 0.55
_SENTINEL_L_MLE = 0.66
_SENTINEL_P_WALD = 0.0077  # p-values distinct from other floats
_SENTINEL_P_LRT = 0.0088
_SENTINEL_P_SCORE = 0.0099


# ---------------------------------------------------------------------------
# load_gemma_assoc
# ---------------------------------------------------------------------------


class TestLoadGemmaAssoc:
    """Tests for load_gemma_assoc file parser."""

    def test_wald_full_format(self, tmp_path):
        """Parse Wald-full format with logl_H1 column."""
        path = tmp_path / "result.assoc.txt"
        write_assoc(
            path,
            WALD_FULL_COLS,
            [
                [
                    "1",
                    "rs1",
                    "1000",
                    "0",
                    "A",
                    "G",
                    _SENTINEL_AF,
                    _SENTINEL_BETA,
                    _SENTINEL_SE,
                    _SENTINEL_LOGL,
                    _SENTINEL_L_REMLE,
                    _SENTINEL_P_WALD,
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.chr == "1"
        assert r.rs == "rs1"
        assert r.ps == 1000
        assert r.n_miss == 0
        assert r.af == pytest.approx(_SENTINEL_AF)
        assert r.beta == pytest.approx(_SENTINEL_BETA)
        assert r.se == pytest.approx(_SENTINEL_SE)
        assert r.logl_H1 == pytest.approx(_SENTINEL_LOGL)
        assert r.l_remle == pytest.approx(_SENTINEL_L_REMLE)
        assert r.p_wald == pytest.approx(_SENTINEL_P_WALD)

    def test_wald_short_format(self, tmp_path):
        """Parse Wald-short format (no logl_H1)."""
        path = tmp_path / "result.assoc.txt"
        write_assoc(
            path,
            WALD_SHORT_COLS,
            [
                [
                    "1",
                    "rs1",
                    "1000",
                    "0",
                    "A",
                    "G",
                    _SENTINEL_AF,
                    _SENTINEL_BETA,
                    _SENTINEL_SE,
                    _SENTINEL_L_REMLE,
                    _SENTINEL_P_WALD,
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.af == pytest.approx(_SENTINEL_AF)
        assert r.beta == pytest.approx(_SENTINEL_BETA)
        assert r.se == pytest.approx(_SENTINEL_SE)
        assert r.logl_H1 is None
        assert r.l_remle == pytest.approx(_SENTINEL_L_REMLE)
        assert r.p_wald == pytest.approx(_SENTINEL_P_WALD)

    def test_score_format(self, tmp_path):
        """Parse Score test format."""
        path = tmp_path / "result.assoc.txt"
        write_assoc(
            path,
            SCORE_COLS,
            [
                [
                    "1",
                    "rs1",
                    "1000",
                    "0",
                    "A",
                    "G",
                    _SENTINEL_AF,
                    _SENTINEL_BETA,
                    _SENTINEL_SE,
                    _SENTINEL_P_SCORE,
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.af == pytest.approx(_SENTINEL_AF)
        assert r.beta == pytest.approx(_SENTINEL_BETA)
        assert r.se == pytest.approx(_SENTINEL_SE)
        assert r.p_score == pytest.approx(_SENTINEL_P_SCORE)
        assert r.p_wald is None
        assert r.logl_H1 is None

    def test_lrt_format(self, tmp_path):
        """Parse LRT format."""
        path = tmp_path / "result.assoc.txt"
        write_assoc(
            path,
            LRT_COLS,
            [
                [
                    "1",
                    "rs1",
                    "1000",
                    "0",
                    "A",
                    "G",
                    _SENTINEL_AF,
                    _SENTINEL_L_MLE,
                    _SENTINEL_P_LRT,
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.af == pytest.approx(_SENTINEL_AF)
        assert r.l_mle == pytest.approx(_SENTINEL_L_MLE)
        assert r.p_lrt == pytest.approx(_SENTINEL_P_LRT)
        assert math.isnan(r.beta)
        assert math.isnan(r.se)
        assert r.logl_H1 is None

    def test_lrt_full_format(self, tmp_path):
        """Parse LRT-full format with logl_H1 column."""
        path = tmp_path / "result.assoc.txt"
        write_assoc(
            path,
            LRT_FULL_COLS,
            [
                [
                    "1",
                    "rs1",
                    "1000",
                    "0",
                    "A",
                    "G",
                    _SENTINEL_AF,
                    _SENTINEL_LOGL,
                    _SENTINEL_L_MLE,
                    _SENTINEL_P_LRT,
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.af == pytest.approx(_SENTINEL_AF)
        assert r.logl_H1 == pytest.approx(_SENTINEL_LOGL)
        assert r.l_mle == pytest.approx(_SENTINEL_L_MLE)
        assert r.p_lrt == pytest.approx(_SENTINEL_P_LRT)
        assert math.isnan(r.beta)
        assert math.isnan(r.se)
        assert r.l_remle is None
        assert r.p_wald is None

    def test_all_tests_format(self, tmp_path):
        """Parse all-tests format (-lmm 4)."""
        path = tmp_path / "result.assoc.txt"
        write_assoc(
            path,
            ALL_TESTS_COLS,
            [
                [
                    "1",
                    "rs1",
                    "1000",
                    "0",
                    "A",
                    "G",
                    _SENTINEL_AF,
                    _SENTINEL_BETA,
                    _SENTINEL_SE,
                    _SENTINEL_L_REMLE,
                    _SENTINEL_L_MLE,
                    _SENTINEL_P_WALD,
                    _SENTINEL_P_LRT,
                    _SENTINEL_P_SCORE,
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.af == pytest.approx(_SENTINEL_AF)
        assert r.beta == pytest.approx(_SENTINEL_BETA)
        assert r.se == pytest.approx(_SENTINEL_SE)
        assert r.l_remle == pytest.approx(_SENTINEL_L_REMLE)
        assert r.l_mle == pytest.approx(_SENTINEL_L_MLE)
        assert r.p_wald == pytest.approx(_SENTINEL_P_WALD)
        assert r.p_lrt == pytest.approx(_SENTINEL_P_LRT)
        assert r.p_score == pytest.approx(_SENTINEL_P_SCORE)
        assert r.logl_H1 is None  # all_tests (not all_tests_full)

    def test_all_tests_full_format(self, tmp_path):
        """Parse all-tests-full format with logl_H1 column."""
        path = tmp_path / "result.assoc.txt"
        write_assoc(
            path,
            ALL_TESTS_FULL_COLS,
            [
                [
                    "1",
                    "rs1",
                    "1000",
                    "0",
                    "A",
                    "G",
                    _SENTINEL_AF,
                    _SENTINEL_BETA,
                    _SENTINEL_SE,
                    _SENTINEL_LOGL,
                    _SENTINEL_L_REMLE,
                    _SENTINEL_L_MLE,
                    _SENTINEL_P_WALD,
                    _SENTINEL_P_LRT,
                    _SENTINEL_P_SCORE,
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.af == pytest.approx(_SENTINEL_AF)
        assert r.beta == pytest.approx(_SENTINEL_BETA)
        assert r.se == pytest.approx(_SENTINEL_SE)
        assert r.logl_H1 == pytest.approx(_SENTINEL_LOGL)
        assert r.l_remle == pytest.approx(_SENTINEL_L_REMLE)
        assert r.l_mle == pytest.approx(_SENTINEL_L_MLE)
        assert r.p_wald == pytest.approx(_SENTINEL_P_WALD)
        assert r.p_lrt == pytest.approx(_SENTINEL_P_LRT)
        assert r.p_score == pytest.approx(_SENTINEL_P_SCORE)

    def test_multiple_snps(self, tmp_path):
        """Parse file with multiple SNPs."""
        path = tmp_path / "result.assoc.txt"
        write_assoc(
            path,
            WALD_FULL_COLS,
            [
                [
                    "1",
                    "rs1",
                    "1000",
                    "0",
                    "A",
                    "G",
                    "0.3",
                    "0.5",
                    "0.1",
                    "-100.0",
                    "0.5",
                    "0.01",
                ],
                [
                    "1",
                    "rs2",
                    "2000",
                    "1",
                    "T",
                    "C",
                    "0.4",
                    "0.3",
                    "0.2",
                    "-95.0",
                    "0.6",
                    "0.03",
                ],
                [
                    "2",
                    "rs3",
                    "3000",
                    "0",
                    "A",
                    "T",
                    "0.1",
                    "0.8",
                    "0.05",
                    "-110.0",
                    "0.4",
                    "1e-5",
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 3
        assert results[0].rs == "rs1"
        assert results[1].rs == "rs2"
        assert results[2].rs == "rs3"
        assert results[2].p_wald == pytest.approx(1e-5)

    def test_invalid_header_raises(self, tmp_path):
        """Invalid header format raises ValueError."""
        path = tmp_path / "bad.assoc.txt"
        with open(path, "w") as f:
            f.write("foo\tbar\tbaz\n")
            f.write("1\t2\t3\n")

        with pytest.raises(ValueError, match="Unexpected header format"):
            load_gemma_assoc(path)

    def test_file_not_found(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_gemma_assoc(tmp_path / "nonexistent.assoc.txt")


# ---------------------------------------------------------------------------
# compare_assoc_results
# ---------------------------------------------------------------------------


class TestCompareAssocResults:
    """Tests for compare_assoc_results comparison logic."""

    def test_identical_wald_results_pass(self):
        """Identical Wald results should pass comparison."""
        results = [_make_assoc(rs=f"rs{i}") for i in range(5)]

        comparison = compare_assoc_results(results, results)

        assert comparison.passed is True
        assert comparison.n_snps == 5
        assert comparison.beta.passed is True
        assert comparison.se.passed is True
        assert comparison.af.passed is True
        assert comparison.p_wald.passed is True
        assert len(comparison.mismatched_snps) == 0

    def test_small_differences_within_tolerance(self):
        """Small numerical differences within tolerance should pass."""
        actual = [_make_assoc(rs="rs1", beta=0.500001, se=0.100001, af=0.300001)]
        expected = [_make_assoc(rs="rs1", beta=0.5, se=0.1, af=0.3)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is True

    def test_large_beta_difference_fails(self):
        """Beta difference outside tolerance should fail."""
        actual = [_make_assoc(rs="rs1", beta=0.6)]
        expected = [_make_assoc(rs="rs1", beta=0.5)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.beta.passed is False

    def test_numeric_failure_reports_every_snp_index(self):
        """A column result identifies every failing association row."""
        actual = [
            _make_assoc(rs=f"rs{i}", beta=beta)
            for i, beta in enumerate((10.0, 2.0, 30.0, 4.0))
        ]
        expected = [
            _make_assoc(rs=f"rs{i}", beta=beta)
            for i, beta in enumerate((1.0, 2.0, 3.0, 4.0))
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.beta.failed_indices == (0, 2)

    def test_snp_count_mismatch(self):
        """Different number of SNPs populates the early-return skip fields."""
        actual = [_make_assoc(rs="rs1")]
        expected = [_make_assoc(rs="rs1"), _make_assoc(rs="rs2")]

        comparison = compare_assoc_results(actual, expected)

        # Overall fails; beta carries the mismatch diagnostic
        assert comparison.passed is False
        assert comparison.n_snps == 1  # reports len(actual)
        assert comparison.beta.passed is False
        assert "SNP count mismatch" in comparison.beta.message
        assert comparison.beta.max_abs_diff == float("inf")
        assert comparison.beta.failed_indices == (1,)

        # All other Wald-always-present fields must carry the skip result
        skip_substr = "Skipped due to SNP count mismatch"
        for field in (
            comparison.se,
            comparison.p_wald,
            comparison.logl_H1,
            comparison.l_remle,
            comparison.af,
        ):
            assert field.passed is True  # skipped results pass vacuously
            assert skip_substr in field.message

        # Wald-test input → score/lrt/mle should be None (not present in test type)
        assert comparison.p_score is None
        assert comparison.p_lrt is None
        assert comparison.l_mle is None

        # No IDs populated because early-return skips the ID-diff loop
        assert comparison.mismatched_snps == []

    def test_snp_count_mismatch_all_tests_populates_optional_fields(self):
        """SNP-count-mismatch in all-tests mode populates p_score/p_lrt/l_mle skips."""
        actual = [_make_assoc(rs="rs1", p_score=0.05, p_lrt=0.02, l_mle=0.8)]
        expected = [
            _make_assoc(rs="rs1", p_score=0.05, p_lrt=0.02, l_mle=0.8),
            _make_assoc(rs="rs2", p_score=0.06, p_lrt=0.03, l_mle=0.9),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        # All-tests detected → optional fields must be skip-results, not None
        assert comparison.p_score is not None
        assert comparison.p_lrt is not None
        assert comparison.l_mle is not None
        assert "Skipped due to SNP count mismatch" in comparison.p_score.message
        assert "Skipped due to SNP count mismatch" in comparison.p_lrt.message
        assert "Skipped due to SNP count mismatch" in comparison.l_mle.message

    def test_snp_id_mismatch_fails_overall(self):
        """Mismatched SNP IDs populate the list AND fail overall comparison."""
        actual = [_make_assoc(rs="rs1"), _make_assoc(rs="rs_X")]
        expected = [_make_assoc(rs="rs1"), _make_assoc(rs="rs2")]

        comparison = compare_assoc_results(actual, expected)

        assert len(comparison.mismatched_snps) == 1
        assert "rs_X!=rs2" in comparison.mismatched_snps[0]
        # The mismatched-ID list must force overall failure even if values match
        assert comparison.passed is False

    def test_score_test_with_real_difference_fails(self):
        """Score-test detection runs the comparison (not a tautological pass)."""
        actual = [
            _make_assoc(
                rs="rs1",
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_score=0.05,
            ),
        ]
        # Large p_score difference (50x): way beyond any reasonable tolerance
        expected = [
            _make_assoc(
                rs="rs1",
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_score=2.5,
            ),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.p_score is not None
        assert comparison.p_score.passed is False

    def test_score_test_detection_skips_wald(self):
        """Score-test detection skips Wald-specific columns with a skip message."""
        results = [
            _make_assoc(
                rs="rs1",
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_score=0.05,
            ),
        ]

        comparison = compare_assoc_results(results, results)

        assert comparison.passed is True
        assert comparison.p_score is not None
        assert comparison.p_score.passed is True
        assert comparison.p_wald.passed is True  # skipped → passes vacuously
        assert "skipped" in comparison.p_wald.message.lower()

    def test_lrt_with_real_difference_fails(self):
        """LRT detection runs the comparison (not a tautological pass)."""
        actual = [
            _make_assoc(
                rs="rs1",
                beta=float("nan"),
                se=float("nan"),
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_lrt=0.02,
                l_mle=0.8,
            ),
        ]
        # Large p_lrt difference (25x): way beyond p_lrt_rtol (5e-3)
        expected = [
            _make_assoc(
                rs="rs1",
                beta=float("nan"),
                se=float("nan"),
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_lrt=0.5,
                l_mle=0.8,
            ),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.p_lrt is not None
        assert comparison.p_lrt.passed is False

    def test_lrt_test_detection(self):
        """LRT results compare p_lrt and l_mle, skip Wald columns."""
        results = [
            _make_assoc(
                rs="rs1",
                beta=float("nan"),
                se=float("nan"),
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_lrt=0.02,
                l_mle=0.8,
            ),
        ]

        comparison = compare_assoc_results(results, results)

        assert comparison.passed is True
        assert comparison.p_lrt is not None
        assert comparison.p_lrt.passed is True
        assert comparison.l_mle is not None
        assert comparison.l_mle.passed is True

    def test_all_tests_with_real_difference_fails(self):
        """All-tests mode runs the comparison (not a tautological pass)."""
        actual = [
            _make_assoc(
                rs="rs1",
                p_wald=0.01,
                p_lrt=0.02,
                p_score=0.05,
                l_remle=0.5,
                l_mle=0.8,
                logl_H1=-100.0,
            ),
        ]
        # Large p_score difference
        expected = [
            _make_assoc(
                rs="rs1",
                p_wald=0.01,
                p_lrt=0.02,
                p_score=1.5,
                l_remle=0.5,
                l_mle=0.8,
                logl_H1=-100.0,
            ),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.p_score is not None
        assert comparison.p_score.passed is False

    def test_all_tests_detection(self):
        """All-tests mode (-lmm 4) compares all column types."""
        results = [
            _make_assoc(
                rs="rs1",
                p_wald=0.01,
                p_lrt=0.02,
                p_score=0.05,
                l_remle=0.5,
                l_mle=0.8,
                logl_H1=-100.0,
            ),
        ]

        comparison = compare_assoc_results(results, results)

        assert comparison.passed is True
        assert comparison.p_score is not None
        assert comparison.p_lrt is not None
        assert comparison.l_mle is not None

    def test_custom_tolerance_config(self):
        """Custom tolerance config should be respected."""
        actual = [_make_assoc(rs="rs1", beta=0.55)]
        expected = [_make_assoc(rs="rs1", beta=0.5)]

        # Default beta_rtol (1e-2) should fail (10% diff)
        default_result = compare_assoc_results(actual, expected)
        assert default_result.beta.passed is False

        # Relaxed config should pass
        relaxed = ToleranceConfig.relaxed()
        relaxed_result = compare_assoc_results(actual, expected, config=relaxed)
        assert relaxed_result.beta.passed is True

    def test_af_normalized_to_maf(self):
        """AF is normalized to MAF (<=0.5) before comparison."""
        # JAMMA reports 0.3, GEMMA reports 0.7 (same allele, different convention)
        actual = [_make_assoc(rs="rs1", af=0.3)]
        expected = [_make_assoc(rs="rs1", af=0.7)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.af.passed is True

    def test_lambda_boundary_all_at_lower_bound(self):
        """Lambda values all at REML lower boundary should be skipped."""
        actual = [_make_assoc(rs="rs1", l_remle=1e-5)]
        expected = [_make_assoc(rs="rs1", l_remle=1e-5)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is True
        assert "boundary" in comparison.l_remle.message.lower()

    @pytest.mark.parametrize(
        ("actual_lambda", "expected_lambda", "expected_classes"),
        [
            (1e-5, 1.0, "lower/interior"),
            (1.0, 1e-5, "interior/lower"),
            (1e5, 1.0, "upper/interior"),
            (1e-5, 1e5, "lower/upper"),
        ],
    )
    def test_lambda_boundary_class_mismatch_fails(
        self, actual_lambda, expected_lambda, expected_classes
    ):
        """A boundary exemption requires both optimizers to hit the same bound."""
        actual = [_make_assoc(rs="rs1", l_remle=actual_lambda)]
        expected = [_make_assoc(rs="rs1", l_remle=expected_lambda)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.l_remle.passed is False
        assert expected_classes in comparison.l_remle.message

    def test_matching_lower_boundary_values_are_exempt(self):
        """Small pinning differences pass when both values classify as lower-bound."""
        actual = [_make_assoc(rs="rs1", l_remle=1.00001e-5)]
        expected = [_make_assoc(rs="rs1", l_remle=1e-5)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is True
        assert "matching lower boundary" in comparison.l_remle.message

    def test_interior_lambda_values_keep_strict_tolerance(self):
        """Boundary handling does not relax comparison of interior optima."""
        actual = [_make_assoc(rs="rs1", l_remle=1.001)]
        expected = [_make_assoc(rs="rs1", l_remle=1.0)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.l_remle.passed is False

    def test_former_lower_threshold_values_are_interior(self):
        """Values above the optimizer bound do not inherit the old exemption."""
        actual = [_make_assoc(rs="rs1", l_remle=2e-5)]
        expected = [_make_assoc(rs="rs1", l_remle=9e-5)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.l_remle.passed is False
        assert "boundary" not in comparison.l_remle.message

    def test_lambda_failure_location_uses_original_snp_index(self):
        """Filtering an exempt pair does not renumber a later failing SNP."""
        actual = [
            _make_assoc(rs="rs1", l_remle=1e-5),
            _make_assoc(rs="rs2", l_remle=2.0),
        ]
        expected = [
            _make_assoc(rs="rs1", l_remle=1e-5),
            _make_assoc(rs="rs2", l_remle=1.0),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.l_remle.passed is False
        assert comparison.l_remle.worst_location == (1,)
        assert comparison.l_remle.failed_indices == (1,)
        assert "at (1,)" in comparison.l_remle.message

    def test_lambda_failures_map_every_filtered_index_to_original_rows(self):
        """Boundary filtering preserves all failing association row indices."""
        actual = [
            _make_assoc(rs="rs0", l_remle=1e-5),
            _make_assoc(rs="rs1", l_remle=10.0),
            _make_assoc(rs="rs2", l_remle=1e-5),
            _make_assoc(rs="rs3", l_remle=30.0),
        ]
        expected = [
            _make_assoc(rs="rs0", l_remle=1e-5),
            _make_assoc(rs="rs1", l_remle=1.0),
            _make_assoc(rs="rs2", l_remle=1e-5),
            _make_assoc(rs="rs3", l_remle=3.0),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.l_remle.failed_indices == (1, 3)

    @pytest.mark.parametrize("field", ["l_remle", "l_mle"])
    def test_lambda_class_and_numeric_failures_are_both_reported(self, field):
        actual = [
            _make_assoc(rs=f"rs{i}", p_lrt=0.2, p_score=0.3, **{field: value})
            for i, value in enumerate((1e-5, 10.0, 1e-5))
        ]
        expected = [
            _make_assoc(rs=f"rs{i}", p_lrt=0.2, p_score=0.3, **{field: value})
            for i, value in enumerate((1.0, 1.0, 1e-5 * (1 + 1e-5)))
        ]
        comparison = compare_assoc_results(actual, expected)
        assert getattr(comparison, field).failed_indices == (0, 1)

    def test_lambda_class_mismatch_reports_worst_difference(self):
        """Class mismatch diagnostics keep the largest full-column error."""
        actual = [
            _make_assoc(rs="rs1", l_remle=1e-5),
            _make_assoc(rs="rs2", l_remle=1e5),
        ]
        expected = [
            _make_assoc(rs="rs1", l_remle=1.0),
            _make_assoc(rs="rs2", l_remle=1.0),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.l_remle.passed is False
        assert comparison.l_remle.max_abs_diff == pytest.approx(99999.0)
        assert comparison.l_remle.worst_location == (1,)
        assert comparison.l_remle.failed_indices == (0, 1)

    def test_remle_matching_upper_values_use_strict_tolerance(self):
        """REML keeps its prior policy of no upper-bound magnitude exemption."""
        actual = [_make_assoc(rs="rs1", l_remle=1e5)]
        expected = [_make_assoc(rs="rs1", l_remle=99900.0)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.l_remle.passed is False

    def test_lambda_boundary_policy_uses_configured_optimizer_bounds(self):
        """Callers can match the comparator policy to non-default optimizer bounds."""
        policy = LambdaBoundaryPolicy(lower=0.1, upper=10.0, rtol=1e-3)
        config = ToleranceConfig(lambda_boundary=policy)
        actual = [_make_assoc(rs="rs1", l_remle=0.10005)]
        expected = [_make_assoc(rs="rs1", l_remle=0.1)]

        comparison = compare_assoc_results(actual, expected, config)

        assert comparison.passed is True
        assert "matching lower boundary" in comparison.l_remle.message

    def test_mle_boundary_class_mismatch_fails(self):
        """The MLE lambda exemption also requires the same boundary class."""
        actual = [
            _make_assoc(
                rs="rs1",
                beta=float("nan"),
                se=float("nan"),
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_lrt=0.02,
                l_mle=1e5,
            )
        ]
        expected = [
            _make_assoc(
                rs="rs1",
                beta=float("nan"),
                se=float("nan"),
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_lrt=0.02,
                l_mle=1.0,
            )
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert comparison.l_mle is not None
        assert comparison.l_mle.passed is False
        assert "upper/interior" in comparison.l_mle.message

    @pytest.mark.parametrize(
        ("actual_lambda", "expected_lambda", "passes"),
        [
            (float("nan"), float("nan"), True),
            (float("nan"), 1.0, False),
            (float("inf"), float("inf"), False),
            (1e-8, 1e-8, False),
        ],
    )
    def test_lambda_invalid_values_are_explicit(
        self, actual_lambda, expected_lambda, passes
    ):
        """Only paired NaNs receive the degenerate-result exemption."""
        actual = [_make_assoc(rs="rs1", l_remle=actual_lambda)]
        expected = [_make_assoc(rs="rs1", l_remle=expected_lambda)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is passes
        assert comparison.l_remle.passed is passes
        assert "invalid" in comparison.l_remle.message.lower()

    def test_lambda_boundary_partial_in_all_tests(self):
        """Partial boundary lambda values: boundary excluded, rest compared."""
        # All-tests mode to exercise the boundary branches at compare.py:729-753.
        # Three SNPs: one at lower boundary (1e-5), two non-boundary (0.5).
        actual = [
            _make_assoc(
                rs="rs1",
                l_remle=1e-5,
                l_mle=1e-5,
                p_wald=0.01,
                p_lrt=0.02,
                p_score=0.05,
            ),
            _make_assoc(
                rs="rs2", l_remle=0.5, l_mle=0.5, p_wald=0.01, p_lrt=0.02, p_score=0.05
            ),
            _make_assoc(
                rs="rs3", l_remle=0.5, l_mle=0.5, p_wald=0.01, p_lrt=0.02, p_score=0.05
            ),
        ]
        expected = [
            _make_assoc(
                rs="rs1",
                l_remle=1e-5,
                l_mle=1e-5,
                p_wald=0.01,
                p_lrt=0.02,
                p_score=0.05,
            ),
            _make_assoc(
                rs="rs2", l_remle=0.5, l_mle=0.5, p_wald=0.01, p_lrt=0.02, p_score=0.05
            ),
            _make_assoc(
                rs="rs3", l_remle=0.5, l_mle=0.5, p_wald=0.01, p_lrt=0.02, p_score=0.05
            ),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is True
        # Partial-boundary branch emits "excluding N boundary values"
        assert "excluding 1 boundary values" in comparison.l_remle.message
        assert comparison.l_mle is not None
        assert "excluding 1 boundary values" in comparison.l_mle.message

    def test_lambda_boundary_l_mle_upper_bound(self):
        """l_mle at upper boundary (>= 1e4) is excluded but l_remle is not."""
        # Upper bound applies only to l_mle (MLE), not l_remle (REML).
        # This exercises the MLE-specific upper-bound check at compare.py:762-768.
        # Use two SNPs so the excluded one can be filtered while the other is compared.
        actual = [
            _make_assoc(
                rs="rs1", l_remle=0.5, l_mle=1e5, p_wald=0.01, p_lrt=0.02, p_score=0.05
            ),
            _make_assoc(
                rs="rs2", l_remle=0.5, l_mle=0.5, p_wald=0.01, p_lrt=0.02, p_score=0.05
            ),
        ]
        expected = [
            _make_assoc(
                rs="rs1", l_remle=0.5, l_mle=1e5, p_wald=0.01, p_lrt=0.02, p_score=0.05
            ),
            _make_assoc(
                rs="rs2", l_remle=0.5, l_mle=0.5, p_wald=0.01, p_lrt=0.02, p_score=0.05
            ),
        ]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is True
        # l_mle had one boundary value excluded; l_remle had none
        assert comparison.l_mle is not None
        assert "excluding 1 boundary values" in comparison.l_mle.message
        assert "boundary" not in comparison.l_remle.message.lower()

    def test_result_dataclass_fields(self):
        """AssocComparisonResult has expected fields."""
        results = [_make_assoc(rs="rs1")]
        comparison = compare_assoc_results(results, results)

        assert isinstance(comparison, AssocComparisonResult)
        assert isinstance(comparison.n_snps, int)
        assert comparison.n_snps == 1
        assert hasattr(comparison, "beta")
        assert hasattr(comparison, "se")
        assert hasattr(comparison, "p_wald")
        assert hasattr(comparison, "logl_H1")
        assert hasattr(comparison, "l_remle")
        assert hasattr(comparison, "af")
        assert hasattr(comparison, "mismatched_snps")
