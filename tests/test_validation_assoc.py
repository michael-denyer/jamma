"""Tests for load_gemma_assoc and compare_assoc_results in validation/compare.py."""

from __future__ import annotations

import math

import pytest

from jamma.lmm.stats import AssocResult
from jamma.validation.compare import (
    AssocComparisonResult,
    compare_assoc_results,
    load_gemma_assoc,
)
from jamma.validation.tolerances import ToleranceConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assoc(
    rs: str = "rs1",
    *,
    beta: float = 0.5,
    se: float = 0.1,
    af: float = 0.3,
    p_wald: float | None = 0.01,
    logl_H1: float | None = -100.0,
    l_remle: float | None = 0.5,
    p_score: float | None = None,
    p_lrt: float | None = None,
    l_mle: float | None = None,
) -> AssocResult:
    return AssocResult(
        chr="1",
        rs=rs,
        ps=1000,
        n_miss=0,
        allele1="A",
        allele0="G",
        af=af,
        beta=beta,
        se=se,
        logl_H1=logl_H1,
        l_remle=l_remle,
        p_wald=p_wald,
        p_score=p_score,
        p_lrt=p_lrt,
        l_mle=l_mle,
    )


def _write_wald_full(path, rows):
    """Write Wald-full format (.assoc.txt) with logl_H1."""
    cols = [
        "chr",
        "rs",
        "ps",
        "n_miss",
        "allele1",
        "allele0",
        "af",
        "beta",
        "se",
        "logl_H1",
        "l_remle",
        "p_wald",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


def _write_wald_short(path, rows):
    """Write Wald-short format (no logl_H1)."""
    cols = [
        "chr",
        "rs",
        "ps",
        "n_miss",
        "allele1",
        "allele0",
        "af",
        "beta",
        "se",
        "l_remle",
        "p_wald",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


def _write_score(path, rows):
    """Write Score format."""
    cols = [
        "chr",
        "rs",
        "ps",
        "n_miss",
        "allele1",
        "allele0",
        "af",
        "beta",
        "se",
        "p_score",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


def _write_lrt(path, rows):
    """Write LRT format (no logl_H1)."""
    cols = ["chr", "rs", "ps", "n_miss", "allele1", "allele0", "af", "l_mle", "p_lrt"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


def _write_all_tests(path, rows):
    """Write all-tests format (no logl_H1)."""
    cols = [
        "chr",
        "rs",
        "ps",
        "n_miss",
        "allele1",
        "allele0",
        "af",
        "beta",
        "se",
        "l_remle",
        "l_mle",
        "p_wald",
        "p_lrt",
        "p_score",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


# ---------------------------------------------------------------------------
# load_gemma_assoc
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestLoadGemmaAssoc:
    """Tests for load_gemma_assoc file parser."""

    def test_wald_full_format(self, tmp_path):
        """Parse Wald-full format with logl_H1 column."""
        path = tmp_path / "result.assoc.txt"
        _write_wald_full(
            path,
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
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.chr == "1"
        assert r.rs == "rs1"
        assert r.ps == 1000
        assert r.n_miss == 0
        assert r.af == pytest.approx(0.3)
        assert r.beta == pytest.approx(0.5)
        assert r.se == pytest.approx(0.1)
        assert r.logl_H1 == pytest.approx(-100.0)
        assert r.l_remle == pytest.approx(0.5)
        assert r.p_wald == pytest.approx(0.01)

    def test_wald_short_format(self, tmp_path):
        """Parse Wald-short format (no logl_H1)."""
        path = tmp_path / "result.assoc.txt"
        _write_wald_short(
            path,
            [
                ["1", "rs1", "1000", "0", "A", "G", "0.3", "0.5", "0.1", "0.5", "0.01"],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        assert results[0].logl_H1 is None
        assert results[0].l_remle == pytest.approx(0.5)
        assert results[0].p_wald == pytest.approx(0.01)

    def test_score_format(self, tmp_path):
        """Parse Score test format."""
        path = tmp_path / "result.assoc.txt"
        _write_score(
            path,
            [
                ["1", "rs1", "1000", "0", "A", "G", "0.3", "0.5", "0.1", "0.05"],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        assert results[0].p_score == pytest.approx(0.05)
        assert results[0].p_wald is None
        assert results[0].logl_H1 is None

    def test_lrt_format(self, tmp_path):
        """Parse LRT format."""
        path = tmp_path / "result.assoc.txt"
        _write_lrt(
            path,
            [
                ["1", "rs1", "1000", "0", "A", "G", "0.3", "0.8", "0.02"],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        assert results[0].l_mle == pytest.approx(0.8)
        assert results[0].p_lrt == pytest.approx(0.02)
        assert math.isnan(results[0].beta)
        assert math.isnan(results[0].se)

    def test_all_tests_format(self, tmp_path):
        """Parse all-tests format (-lmm 4)."""
        path = tmp_path / "result.assoc.txt"
        _write_all_tests(
            path,
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
                    "0.5",
                    "0.8",
                    "0.01",
                    "0.02",
                    "0.05",
                ],
            ],
        )
        results = load_gemma_assoc(path)

        assert len(results) == 1
        r = results[0]
        assert r.p_wald == pytest.approx(0.01)
        assert r.p_lrt == pytest.approx(0.02)
        assert r.p_score == pytest.approx(0.05)
        assert r.l_remle == pytest.approx(0.5)
        assert r.l_mle == pytest.approx(0.8)
        assert r.logl_H1 is None  # all_tests (not all_tests_full)

    def test_multiple_snps(self, tmp_path):
        """Parse file with multiple SNPs."""
        path = tmp_path / "result.assoc.txt"
        _write_wald_full(
            path,
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


@pytest.mark.tier0
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

    def test_snp_count_mismatch(self):
        """Different number of SNPs should fail."""
        actual = [_make_assoc(rs="rs1")]
        expected = [_make_assoc(rs="rs1"), _make_assoc(rs="rs2")]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is False
        assert "SNP count mismatch" in comparison.beta.message

    def test_snp_id_mismatch_detected(self):
        """Mismatched SNP IDs should be reported."""
        actual = [_make_assoc(rs="rs1"), _make_assoc(rs="rs_X")]
        expected = [_make_assoc(rs="rs1"), _make_assoc(rs="rs2")]

        comparison = compare_assoc_results(actual, expected)

        assert len(comparison.mismatched_snps) == 1
        assert "rs_X!=rs2" in comparison.mismatched_snps[0]

    def test_score_test_detection(self):
        """Score test results skip Wald-specific columns."""
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
        assert comparison.p_wald.passed is True  # skipped → passes
        assert "skipped" in comparison.p_wald.message.lower()

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

    def test_lambda_boundary_handling(self):
        """Lambda values at optimization boundary should be excluded."""
        actual = [_make_assoc(rs="rs1", l_remle=1e-5)]
        expected = [_make_assoc(rs="rs1", l_remle=1e-5)]

        comparison = compare_assoc_results(actual, expected)

        assert comparison.passed is True
        assert "boundary" in comparison.l_remle.message.lower()

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
