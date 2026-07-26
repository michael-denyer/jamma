"""Comparison utilities for validating JAMMA output against reference GEMMA.

This module provides structured comparison functions that return detailed results
rather than raising exceptions, enabling programmatic validation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from jamma.lmm.schema import HEADERS, MODE_SPECS
from jamma.lmm.stats import AssocResult
from jamma.validation.tolerances import ToleranceConfig


@dataclass
class ComparisonResult:
    """Result of a numerical array comparison.

    Provides structured information about pass/fail status and the nature
    of any discrepancies found.

    Attributes:
        passed: Whether the comparison passed within tolerance.
        max_abs_diff: Maximum absolute difference found.
        max_rel_diff: Maximum relative difference found (inf if expected was 0).
        worst_location: Index tuple of the worst mismatch, or None if passed.
        message: Human-readable description of the result.

    Example:
        >>> result = compare_arrays(actual, expected, rtol=1e-6, atol=1e-12)
        >>> if not result.passed:
        ...     print(f"Failed at {result.worst_location}: {result.message}")
    """

    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    worst_location: tuple[int, ...] | None
    message: str


def compare_arrays(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float,
    atol: float,
    name: str = "array",
) -> ComparisonResult:
    """Compare two arrays with tolerance and return structured result.

    Uses numpy.testing.assert_allclose internally but catches the assertion
    to return a structured ComparisonResult instead of raising.

    Args:
        actual: The computed array to validate.
        expected: The reference array to compare against.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        name: Name to use in error messages for context.

    Returns:
        ComparisonResult with pass/fail status and diagnostic information.

    Example:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.0, 2.0, 3.0])
        >>> result = compare_arrays(a, b, rtol=1e-6, atol=1e-12, name="test")
        >>> result.passed
        True
    """
    if actual.shape != expected.shape:
        return ComparisonResult(
            passed=False,
            max_abs_diff=np.inf,
            max_rel_diff=np.inf,
            worst_location=None,
            message=(
                f"{name} shape mismatch: "
                f"actual {actual.shape} vs expected {expected.shape}"
            ),
        )

    try:
        assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} comparison",
        )
        # Passed - compute stats anyway for reporting
        abs_diff = np.abs(actual - expected)
        max_abs_diff = float(np.max(abs_diff))

        # Relative difference: avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / np.abs(expected)
            rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
        max_rel_diff = float(np.max(rel_diff))

        return ComparisonResult(
            passed=True,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            worst_location=None,
            message=(
                f"{name} comparison passed "
                f"(max abs diff: {max_abs_diff:.2e}, max rel diff: {max_rel_diff:.2e})"
            ),
        )

    except AssertionError:
        # Compute detailed diagnostics
        abs_diff = np.abs(actual - expected)
        max_abs_diff = float(np.max(abs_diff))

        # Find location of worst absolute difference
        # Convert numpy int64 to plain int for cleaner display/serialization
        worst_idx_raw = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        worst_idx = tuple(int(i) for i in worst_idx_raw)

        # Relative difference at worst location
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / np.abs(expected)
            rel_diff = np.where(np.isfinite(rel_diff), rel_diff, np.inf)
        max_rel_diff = float(np.max(rel_diff))

        return ComparisonResult(
            passed=False,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            worst_location=worst_idx,
            message=f"{name} comparison failed at {worst_idx}: "
            f"actual={actual[worst_idx]:.10e}, expected={expected[worst_idx]:.10e}, "
            f"abs_diff={abs_diff[worst_idx]:.2e} (rtol={rtol}, atol={atol})",
        )


def compare_kinship_matrices(
    actual: np.ndarray,
    expected: np.ndarray,
    config: ToleranceConfig | None = None,
) -> ComparisonResult:
    """Compare kinship matrices with appropriate tolerance.

    Kinship matrices should be symmetric and positive semi-definite.
    This function compares the full matrices using the kinship-specific
    tolerance from the configuration.

    Args:
        actual: Computed kinship matrix (n x n).
        expected: Reference GEMMA kinship matrix (n x n).
        config: Tolerance configuration. Uses default if None.

    Returns:
        ComparisonResult with pass/fail status and diagnostic information.

    Example:
        >>> K1 = np.eye(3) * 0.5
        >>> K2 = np.eye(3) * 0.5 + 1e-10
        >>> result = compare_kinship_matrices(K1, K2)
        >>> result.passed
        True
    """
    if config is None:
        config = ToleranceConfig()

    return compare_arrays(
        actual=actual,
        expected=expected,
        rtol=config.kinship_rtol,
        atol=config.atol,
        name="kinship matrix",
    )


def load_gemma_kinship(path: Path) -> np.ndarray:
    """Load GEMMA kinship matrix from .cXX.txt format.

    GEMMA outputs kinship matrices as space-separated values,
    one row per line. The matrix is symmetric.

    Args:
        path: Path to the kinship matrix file (.cXX.txt or .sXX.txt).

    Returns:
        2D numpy array containing the kinship matrix.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as a numeric matrix.

    Example:
        >>> K = load_gemma_kinship(Path("output/result.cXX.txt"))
        >>> K.shape
        (1940, 1940)
    """
    return np.loadtxt(path)


# GEMMA .assoc.txt column headers are identical to AssocResult field names, so
# parsing is a header->field map keyed by column name rather than a per-format
# positional unpack. Which columns are present varies by layout, not their order.


def _assoc_header_layouts() -> frozenset[tuple[str, ...]]:
    """Build the accepted .assoc.txt header layouts from the output schema.

    The four canonical layouts are ``schema.HEADERS`` (one per LMM mode). Three
    extra layouts are GEMMA-version quirks the schema does not model: they differ
    only by an optional ``logl_H1`` column that some versions emit for Wald/all
    tests and others emit for LRT.
    """
    cols = {tt: tuple(h.split("\t")) for tt, h in HEADERS.items()}
    # Number of leading metadata columns (chr..af), derived rather than hardcoded.
    n_prefix = len(cols["lrt"]) - len(MODE_SPECS[2].stat_columns)
    wald, lrt, all_tests = cols["wald"], cols["lrt"], cols["all"]
    return frozenset(
        {
            wald,  # Wald with logl_H1
            tuple(c for c in wald if c != "logl_H1"),  # Wald without logl_H1
            cols["score"],
            lrt,  # LRT without logl_H1
            (*lrt[:n_prefix], "logl_H1", *lrt[n_prefix:]),  # LRT with logl_H1
            all_tests,  # all-tests with logl_H1
            tuple(c for c in all_tests if c != "logl_H1"),  # all-tests without
        }
    )


_ASSOC_HEADER_LAYOUTS = _assoc_header_layouts()


def _opt_float(row: dict[str, str], name: str) -> float | None:
    """Cast an optional .assoc.txt cell, or None when its layout omits the column."""
    raw = row.get(name)
    return None if raw is None else float(raw)


def load_gemma_assoc(path: Path) -> list[AssocResult]:
    """Load GEMMA association results from .assoc.txt format.

    Parses the tab-separated .assoc.txt format produced by GEMMA's LMM modes:
    - Wald test (-lmm 1): Has logl_H1, l_remle, p_wald columns
    - LRT (-lmm 2): Has l_mle, p_lrt columns (no beta/se)
    - Score test (-lmm 3): Has p_score column (no logl_H1, l_remle, p_wald)
    - All tests (-lmm 4): Has l_remle, l_mle, p_wald, p_lrt, p_score columns

    The header is matched against the layouts derived from the shared output
    schema; each cell is then mapped to the AssocResult field with the matching
    column name. LRT formats omit beta/se, which are filled with NaN.

    Args:
        path: Path to the association results file (.assoc.txt).

    Returns:
        List of AssocResult dataclass instances, one per SNP.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.

    Example:
        >>> results = load_gemma_assoc(Path("output/result.assoc.txt"))
        >>> len(results)
        12226
    """
    results = []
    with open(path) as f:
        cols = tuple(f.readline().strip().split("\t"))
        if cols not in _ASSOC_HEADER_LAYOUTS:
            expected = "\n".join(
                "  " + "\t".join(layout) for layout in sorted(_ASSOC_HEADER_LAYOUTS)
            )
            raise ValueError(
                f"Unexpected header format. Expected one of:\n{expected}\n"
                f"Got: {list(cols)}"
            )

        for line in f:
            fields = line.strip().split("\t")
            row = dict(zip(cols, fields, strict=True))
            results.append(
                AssocResult(
                    chr=row["chr"],
                    rs=row["rs"],
                    ps=int(row["ps"]),
                    n_miss=int(row["n_miss"]),
                    allele1=row["allele1"],
                    allele0=row["allele0"],
                    af=float(row["af"]),
                    # GEMMA LRT formats omit beta/se; AssocResult requires them.
                    beta=float(row.get("beta", "nan")),
                    se=float(row.get("se", "nan")),
                    logl_H1=_opt_float(row, "logl_H1"),
                    l_remle=_opt_float(row, "l_remle"),
                    p_wald=_opt_float(row, "p_wald"),
                    p_score=_opt_float(row, "p_score"),
                    l_mle=_opt_float(row, "l_mle"),
                    p_lrt=_opt_float(row, "p_lrt"),
                )
            )
    return results


@dataclass
class AssocComparisonResult:
    """Result of comparing two sets of association results.

    Provides structured comparison results for each numeric column
    (beta, se, p_wald, logl_H1, l_remle, af) and overall pass/fail status.

    Attributes:
        passed: Whether all column comparisons passed.
        n_snps: Number of SNPs compared.
        beta: Comparison result for effect sizes.
        se: Comparison result for standard errors.
        p_wald: Comparison result for p-values (Wald test).
        logl_H1: Comparison result for log-likelihoods.
        l_remle: Comparison result for lambda REML values.
        af: Comparison result for allele frequencies.
        mismatched_snps: List of SNP rs IDs that don't match between files.
        p_score: Comparison result for Score test p-values (only for lmm_mode=3).
        p_lrt: Comparison result for LRT p-values (only for lmm_mode=2).
        l_mle: Comparison result for MLE lambda values (only for lmm_mode=2).
    """

    passed: bool
    n_snps: int
    beta: ComparisonResult
    se: ComparisonResult
    p_wald: ComparisonResult
    logl_H1: ComparisonResult
    l_remle: ComparisonResult
    af: ComparisonResult
    mismatched_snps: list[str]
    p_score: ComparisonResult | None = None  # Only for Score test (-lmm 3)
    p_lrt: ComparisonResult | None = None  # Only for LRT (-lmm 2)
    l_mle: ComparisonResult | None = None  # Only for LRT (-lmm 2)


_LAMBDA_LOWER_BOUND = 1e-4
_LAMBDA_UPPER_BOUND = 1e4


def _skipped_result(message: str) -> ComparisonResult:
    """A vacuously-passing result for a column not applicable to the test type."""
    return ComparisonResult(
        passed=True,
        max_abs_diff=0.0,
        max_rel_diff=0.0,
        worst_location=None,
        message=message,
    )


def _column(field: str, rows: list[AssocResult], default: float) -> np.ndarray:
    """Extract one AssocResult field across rows, substituting default for None."""
    return np.array(
        [getattr(r, field) if getattr(r, field) is not None else default for r in rows]
    )


def _compare_with_boundary(
    actual_arr: np.ndarray,
    expected_arr: np.ndarray,
    boundary_mask: np.ndarray,
    rtol: float,
    atol: float,
    name: str,
) -> ComparisonResult:
    """Compare lambda arrays, excluding values pinned at the optimizer boundary.

    Lambda converges at the optimization bounds for weak-signal SNPs; JAMMA's
    golden-section and GEMMA's Brent agree on *whether* a value is at the bound
    but not on its exact pinned magnitude, so boundary values are excluded.
    """
    if np.all(boundary_mask):
        return _skipped_result(
            f"{name} comparison skipped (all values at optimization boundary)"
        )
    if np.any(boundary_mask):
        keep = ~boundary_mask
        return compare_arrays(
            actual_arr[keep],
            expected_arr[keep],
            rtol,
            atol,
            f"{name} (excluding {int(np.sum(boundary_mask))} boundary values)",
        )
    return compare_arrays(actual_arr, expected_arr, rtol, atol, name)


def compare_assoc_results(
    actual: list[AssocResult],
    expected: list[AssocResult],
    config: ToleranceConfig | None = None,
) -> AssocComparisonResult:
    """Compare association results with column-appropriate tolerances.

    Compares lists of AssocResult objects from JAMMA and reference GEMMA output.
    Uses appropriate tolerance thresholds for each statistic type:
    - beta: beta_rtol (effect sizes from linear algebra)
    - se: se_rtol (standard errors with sqrt operations)
    - p_wald: pvalue_rtol (CDF computations may differ) - Wald test only
    - p_score: pvalue_rtol (CDF computations may differ) - Score test only
    - p_lrt: p_lrt_rtol (chi-squared magnifies logl differences) - LRT only
    - logl_H1: logl_rtol (log-likelihood values) - Wald test only
    - l_remle: lambda_rtol (variance ratio estimates) - Wald test only
    - l_mle: lambda_rtol (MLE lambda values) - LRT only
    - af: af_rtol (allele frequency of counted allele, BIM A1)

    Args:
        actual: Computed association results from JAMMA.
        expected: Reference GEMMA association results.
        config: Tolerance configuration. Uses default if None.

    Returns:
        AssocComparisonResult with per-column comparison details.

    Example:
        >>> jamma_results = load_gemma_assoc(Path("jamma_output.assoc.txt"))
        >>> gemma_results = load_gemma_assoc(Path("gemma_output.assoc.txt"))
        >>> comparison = compare_assoc_results(jamma_results, gemma_results)
        >>> comparison.passed
        True
    """
    if config is None:
        config = ToleranceConfig()

    # Detect test type from reference data using majority of first N records
    # (not just first record, to handle degenerate/NaN first SNP)
    _SAMPLE_SIZE = min(5, len(expected))
    sample = expected[:_SAMPLE_SIZE]

    is_all_tests = len(sample) > 0 and all(
        r.p_wald is not None and r.p_lrt is not None and r.p_score is not None
        for r in sample
    )
    is_score_test = (
        len(sample) > 0
        and not is_all_tests
        and all(r.p_score is not None for r in sample)
        and all(r.p_wald is None for r in sample)
    )
    is_lrt_test = (
        len(sample) > 0
        and not is_all_tests
        and all(r.p_lrt is not None for r in sample)
        and all(r.p_wald is None for r in sample)
    )

    # Check for SNP count mismatch
    if len(actual) != len(expected):
        mismatch_result = ComparisonResult(
            passed=False,
            max_abs_diff=np.inf,
            max_rel_diff=np.inf,
            worst_location=None,
            message=f"SNP count mismatch: {len(actual)} vs {len(expected)}",
        )
        skip_result = _skipped_result("Skipped due to SNP count mismatch")
        return AssocComparisonResult(
            passed=False,
            n_snps=len(actual),
            beta=mismatch_result,
            se=skip_result,
            p_wald=skip_result,
            logl_H1=skip_result,
            l_remle=skip_result,
            af=skip_result,
            mismatched_snps=[],
            p_score=skip_result if (is_score_test or is_all_tests) else None,
            p_lrt=skip_result if (is_lrt_test or is_all_tests) else None,
            l_mle=skip_result if (is_lrt_test or is_all_tests) else None,
        )

    # Check for mismatched SNP IDs
    mismatched = [
        f"{i}:{a.rs}!={e.rs}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True))
        if a.rs != e.rs
    ]

    # Always-present columns. AF is normalized to MAF (<= 0.5) first because
    # JAMMA reports MAF while GEMMA's AF can exceed 0.5 for the same allele.
    actual_beta = np.array([r.beta for r in actual])
    expected_beta = np.array([r.beta for r in expected])
    actual_se = np.array([r.se for r in actual])
    expected_se = np.array([r.se for r in expected])
    actual_af = np.array([r.af for r in actual])
    expected_af = np.array([r.af for r in expected])
    actual_maf = np.minimum(actual_af, 1.0 - actual_af)
    expected_maf = np.minimum(expected_af, 1.0 - expected_af)

    beta_result = compare_arrays(
        actual_beta, expected_beta, config.beta_rtol, config.atol, "beta"
    )
    se_result = compare_arrays(
        actual_se, expected_se, config.se_rtol, config.atol, "se"
    )
    af_result = compare_arrays(
        actual_maf, expected_maf, config.af_rtol, config.atol, "af"
    )

    # Test-type-specific columns. Each closure computes one column's comparison
    # with the right tolerance and boundary/skip rule; columns inactive for the
    # detected mode get a skip-result (always-present output slots) or stay None
    # (optional output slots).
    def _pvalue(field: str, rtol: float) -> ComparisonResult:
        return compare_arrays(
            _column(field, actual, np.nan),
            _column(field, expected, np.nan),
            rtol,
            config.atol,
            field,
        )

    def _logl(skip_message: str) -> ComparisonResult:
        expected_logl = _column("logl_H1", expected, 0.0)
        if np.allclose(expected_logl, 0.0):
            return _skipped_result(skip_message)
        return compare_arrays(
            _column("logl_H1", actual, 0.0),
            expected_logl,
            config.logl_rtol,
            config.atol,
            "logl_H1",
        )

    def _lambda(field: str, *, check_upper: bool) -> ComparisonResult:
        actual_arr = _column(field, actual, np.nan)
        expected_arr = _column(field, expected, np.nan)
        mask = (expected_arr <= _LAMBDA_LOWER_BOUND) | (
            actual_arr <= _LAMBDA_LOWER_BOUND
        )
        if check_upper:
            mask = (
                mask
                | (expected_arr >= _LAMBDA_UPPER_BOUND)
                | (actual_arr >= _LAMBDA_UPPER_BOUND)
            )
        return _compare_with_boundary(
            actual_arr, expected_arr, mask, config.lambda_rtol, config.atol, field
        )

    no_id_mismatch = len(mismatched) == 0

    p_score_result: ComparisonResult | None = None
    p_lrt_result: ComparisonResult | None = None
    l_mle_result: ComparisonResult | None = None

    # Each branch computes the columns active for its test type and the overall
    # pass from exactly those columns. The two must agree on which columns exist,
    # so they share one dispatch rather than repeating the test-type conditions.
    if is_all_tests:
        pwald_result = _pvalue("p_wald", config.pvalue_rtol)
        p_score_result = _pvalue("p_score", config.pvalue_rtol)
        p_lrt_result = _pvalue("p_lrt", config.p_lrt_rtol)
        logl_result = _logl("logl_H1 skipped (not in GEMMA -lmm 4 format)")
        lambda_result = _lambda("l_remle", check_upper=False)
        l_mle_result = _lambda("l_mle", check_upper=True)
        all_passed = (
            beta_result.passed
            and se_result.passed
            and af_result.passed
            and pwald_result.passed
            and logl_result.passed
            and lambda_result.passed
            and p_score_result.passed
            and p_lrt_result.passed
            and l_mle_result.passed
            and no_id_mismatch
        )
    elif is_score_test:
        p_score_result = _pvalue("p_score", config.pvalue_rtol)
        pwald_result = _skipped_result("p_wald skipped (Score test)")
        logl_result = _skipped_result("logl_H1 skipped (Score test)")
        lambda_result = _skipped_result("l_remle skipped (Score test)")
        all_passed = (
            beta_result.passed
            and se_result.passed
            and af_result.passed
            and p_score_result.passed
            and no_id_mismatch
        )
    elif is_lrt_test:
        p_lrt_result = _pvalue("p_lrt", config.p_lrt_rtol)
        l_mle_result = _lambda("l_mle", check_upper=True)
        pwald_result = _skipped_result("p_wald skipped (LRT)")
        logl_result = _skipped_result("logl_H1 skipped (LRT)")
        lambda_result = _skipped_result("l_remle skipped (LRT)")
        # LRT has NaN beta/se by construction, so check both sides are all-NaN
        # rather than the (meaningless) compare result.
        beta_all_nan = bool(
            np.all(np.isnan(actual_beta)) and np.all(np.isnan(expected_beta))
        )
        se_all_nan = bool(np.all(np.isnan(actual_se)) and np.all(np.isnan(expected_se)))
        all_passed = (
            af_result.passed
            and beta_all_nan
            and se_all_nan
            and p_lrt_result.passed
            and l_mle_result.passed
            and no_id_mismatch
        )
    else:  # Wald
        pwald_result = _pvalue("p_wald", config.pvalue_rtol)
        logl_result = _logl("logl_H1 skipped (reference missing logl_H1 column)")
        lambda_result = _lambda("l_remle", check_upper=False)
        all_passed = (
            beta_result.passed
            and se_result.passed
            and af_result.passed
            and pwald_result.passed
            and logl_result.passed
            and lambda_result.passed
            and no_id_mismatch
        )

    return AssocComparisonResult(
        passed=all_passed,
        n_snps=len(actual),
        beta=beta_result,
        se=se_result,
        p_wald=pwald_result,
        logl_H1=logl_result,
        l_remle=lambda_result,
        af=af_result,
        mismatched_snps=mismatched,
        p_score=p_score_result,
        p_lrt=p_lrt_result,
        l_mle=l_mle_result,
    )
