"""Comparison utilities for validating JAMMA output against reference GEMMA.

This module provides structured comparison functions that return detailed results
rather than raising exceptions, enabling programmatic validation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

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


def load_gemma_assoc(path: Path) -> list[AssocResult]:
    """Load GEMMA association results from .assoc.txt format.

    Parses the tab-separated .assoc.txt format produced by GEMMA's LMM modes:
    - Wald test (-lmm 1): Has logl_H1, l_remle, p_wald columns
    - LRT (-lmm 2): Has l_mle, p_lrt columns (no beta/se)
    - Score test (-lmm 3): Has p_score column (no logl_H1, l_remle, p_wald)
    - All tests (-lmm 4): Has l_remle, l_mle, p_wald, p_lrt, p_score columns

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
        header = f.readline().strip()
        actual_cols = header.split("\t")

        # Support multiple GEMMA output formats:
        # Format 1: Wald test with logl_H1 (full output)
        expected_cols_wald_full = [
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
        # Format 2: Wald test without logl_H1 (some GEMMA versions)
        expected_cols_wald_short = [
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
        # Format 3: Score test (-lmm 3)
        expected_cols_score = [
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
        # Format 4: LRT (-lmm 2)
        expected_cols_lrt = [
            "chr",
            "rs",
            "ps",
            "n_miss",
            "allele1",
            "allele0",
            "af",
            "l_mle",
            "p_lrt",
        ]
        # Format 4b: LRT with logl_H1 (-lmm 2, some GEMMA versions)
        expected_cols_lrt_full = [
            "chr",
            "rs",
            "ps",
            "n_miss",
            "allele1",
            "allele0",
            "af",
            "logl_H1",
            "l_mle",
            "p_lrt",
        ]
        # Format 5: All tests (-lmm 4) without logl_H1
        expected_cols_all = [
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
        # Format 5b: All tests (-lmm 4) with logl_H1 (some GEMMA versions)
        expected_cols_all_full = [
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
            "l_mle",
            "p_wald",
            "p_lrt",
            "p_score",
        ]

        if actual_cols == expected_cols_wald_full:
            format_type = "wald_full"
        elif actual_cols == expected_cols_wald_short:
            format_type = "wald_short"
        elif actual_cols == expected_cols_score:
            format_type = "score"
        elif actual_cols == expected_cols_lrt:
            format_type = "lrt"
        elif actual_cols == expected_cols_lrt_full:
            format_type = "lrt_full"
        elif actual_cols == expected_cols_all:
            format_type = "all_tests"
        elif actual_cols == expected_cols_all_full:
            format_type = "all_tests_full"
        else:
            raise ValueError(
                f"Unexpected header format. Expected one of:\n"
                f"  {expected_cols_wald_full}\n"
                f"  {expected_cols_wald_short}\n"
                f"  {expected_cols_score}\n"
                f"  {expected_cols_lrt}\n"
                f"  {expected_cols_lrt_full}\n"
                f"  {expected_cols_all}\n"
                f"  {expected_cols_all_full}\n"
                f"Got: {actual_cols}"
            )

        for line in f:
            fields = line.strip().split("\t")
            if format_type == "wald_full":
                results.append(
                    AssocResult(
                        chr=fields[0],
                        rs=fields[1],
                        ps=int(fields[2]),
                        n_miss=int(fields[3]),
                        allele1=fields[4],
                        allele0=fields[5],
                        af=float(fields[6]),
                        beta=float(fields[7]),
                        se=float(fields[8]),
                        logl_H1=float(fields[9]),
                        l_remle=float(fields[10]),
                        p_wald=float(fields[11]),
                    )
                )
            elif format_type == "wald_short":
                results.append(
                    AssocResult(
                        chr=fields[0],
                        rs=fields[1],
                        ps=int(fields[2]),
                        n_miss=int(fields[3]),
                        allele1=fields[4],
                        allele0=fields[5],
                        af=float(fields[6]),
                        beta=float(fields[7]),
                        se=float(fields[8]),
                        logl_H1=None,
                        l_remle=float(fields[9]),
                        p_wald=float(fields[10]),
                    )
                )
            elif format_type == "score":
                results.append(
                    AssocResult(
                        chr=fields[0],
                        rs=fields[1],
                        ps=int(fields[2]),
                        n_miss=int(fields[3]),
                        allele1=fields[4],
                        allele0=fields[5],
                        af=float(fields[6]),
                        beta=float(fields[7]),
                        se=float(fields[8]),
                        logl_H1=None,
                        l_remle=None,
                        p_wald=None,
                        p_score=float(fields[9]),
                    )
                )
            elif format_type == "all_tests":
                results.append(
                    AssocResult(
                        chr=fields[0],
                        rs=fields[1],
                        ps=int(fields[2]),
                        n_miss=int(fields[3]),
                        allele1=fields[4],
                        allele0=fields[5],
                        af=float(fields[6]),
                        beta=float(fields[7]),
                        se=float(fields[8]),
                        logl_H1=None,
                        l_remle=float(fields[9]),
                        l_mle=float(fields[10]),
                        p_wald=float(fields[11]),
                        p_lrt=float(fields[12]),
                        p_score=float(fields[13]),
                    )
                )
            elif format_type == "all_tests_full":
                results.append(
                    AssocResult(
                        chr=fields[0],
                        rs=fields[1],
                        ps=int(fields[2]),
                        n_miss=int(fields[3]),
                        allele1=fields[4],
                        allele0=fields[5],
                        af=float(fields[6]),
                        beta=float(fields[7]),
                        se=float(fields[8]),
                        logl_H1=float(fields[9]),
                        l_remle=float(fields[10]),
                        l_mle=float(fields[11]),
                        p_wald=float(fields[12]),
                        p_lrt=float(fields[13]),
                        p_score=float(fields[14]),
                    )
                )
            elif format_type == "lrt_full":
                results.append(
                    AssocResult(
                        chr=fields[0],
                        rs=fields[1],
                        ps=int(fields[2]),
                        n_miss=int(fields[3]),
                        allele1=fields[4],
                        allele0=fields[5],
                        af=float(fields[6]),
                        beta=float("nan"),  # LRT format has no beta
                        se=float("nan"),  # LRT format has no se
                        logl_H1=float(fields[7]),
                        l_remle=None,
                        p_wald=None,
                        l_mle=float(fields[8]),
                        p_lrt=float(fields[9]),
                    )
                )
            else:  # lrt
                results.append(
                    AssocResult(
                        chr=fields[0],
                        rs=fields[1],
                        ps=int(fields[2]),
                        n_miss=int(fields[3]),
                        allele1=fields[4],
                        allele0=fields[5],
                        af=float(fields[6]),
                        beta=float("nan"),  # LRT format has no beta
                        se=float("nan"),  # LRT format has no se
                        logl_H1=None,
                        l_remle=None,
                        p_wald=None,
                        l_mle=float(fields[7]),
                        p_lrt=float(fields[8]),
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
    - p_lrt: pvalue_rtol (CDF computations may differ) - LRT only
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

    # Detect test type from reference data
    # All-tests has ALL of p_wald, p_lrt, p_score populated
    # Must check before individual test type checks (all-tests has p_wald set)
    is_all_tests = (
        len(expected) > 0
        and expected[0].p_wald is not None
        and expected[0].p_lrt is not None
        and expected[0].p_score is not None
    )
    # Score test has p_score only; LRT has p_lrt only; Wald has p_wald only
    is_score_test = (
        len(expected) > 0
        and expected[0].p_score is not None
        and expected[0].p_wald is None
    )
    is_lrt_test = (
        len(expected) > 0
        and expected[0].p_lrt is not None
        and expected[0].p_wald is None
    )

    # Helper to create skipped comparison result
    def _skipped_result(msg: str) -> ComparisonResult:
        return ComparisonResult(
            passed=True,
            max_abs_diff=0.0,
            max_rel_diff=0.0,
            worst_location=None,
            message=msg,
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
    mismatched = []
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if a.rs != e.rs:
            mismatched.append(f"{i}:{a.rs}!={e.rs}")

    # Extract arrays for comparison (always present)
    actual_beta = np.array([r.beta for r in actual])
    expected_beta = np.array([r.beta for r in expected])
    actual_se = np.array([r.se for r in actual])
    expected_se = np.array([r.se for r in expected])
    actual_af = np.array([r.af for r in actual])
    expected_af = np.array([r.af for r in expected])

    # Compare always-present columns
    beta_result = compare_arrays(
        actual_beta, expected_beta, config.beta_rtol, config.atol, "beta"
    )
    se_result = compare_arrays(
        actual_se, expected_se, config.se_rtol, config.atol, "se"
    )
    af_result = compare_arrays(
        actual_af, expected_af, config.af_rtol, config.atol, "af"
    )

    # Handle test-type specific columns
    p_score_result = None
    p_lrt_result = None
    l_mle_result = None
    if is_all_tests:
        # All-tests mode: compare ALL columns with per-column tolerances
        actual_pwald = np.array(
            [r.p_wald if r.p_wald is not None else np.nan for r in actual]
        )
        expected_pwald = np.array(
            [r.p_wald if r.p_wald is not None else np.nan for r in expected]
        )
        pwald_result = compare_arrays(
            actual_pwald, expected_pwald, config.pvalue_rtol, config.atol, "p_wald"
        )

        # Score test p-values
        actual_pscore = np.array(
            [r.p_score if r.p_score is not None else np.nan for r in actual]
        )
        expected_pscore = np.array(
            [r.p_score if r.p_score is not None else np.nan for r in expected]
        )
        p_score_result = compare_arrays(
            actual_pscore, expected_pscore, config.pvalue_rtol, config.atol, "p_score"
        )

        # LRT p-values: RELAXED tolerance (rtol=5e-3) per Phase 7.1 decision [07.1-02]
        # Chi-squared distribution magnifies small log-likelihood differences.
        # Covariate models compound the effect, requiring slightly wider margin.
        lrt_pvalue_rtol = 5e-3
        actual_plrt = np.array(
            [r.p_lrt if r.p_lrt is not None else np.nan for r in actual]
        )
        expected_plrt = np.array(
            [r.p_lrt if r.p_lrt is not None else np.nan for r in expected]
        )
        p_lrt_result = compare_arrays(
            actual_plrt, expected_plrt, lrt_pvalue_rtol, config.atol, "p_lrt"
        )

        # logl_H1: compare if present in reference (all_tests_full format)
        expected_logl_all = np.array(
            [r.logl_H1 if r.logl_H1 is not None else 0.0 for r in expected]
        )
        if np.allclose(expected_logl_all, 0.0) or all(
            r.logl_H1 is None for r in expected
        ):
            logl_result = _skipped_result(
                "logl_H1 skipped (not in GEMMA -lmm 4 format)"
            )
        else:
            actual_logl_all = np.array(
                [r.logl_H1 if r.logl_H1 is not None else 0.0 for r in actual]
            )
            logl_result = compare_arrays(
                actual_logl_all,
                expected_logl_all,
                config.logl_rtol,
                config.atol,
                "logl_H1",
            )

        # l_remle with boundary handling
        actual_lambda = np.array(
            [r.l_remle if r.l_remle is not None else np.nan for r in actual]
        )
        expected_lambda = np.array(
            [r.l_remle if r.l_remle is not None else np.nan for r in expected]
        )
        lambda_lower_bound = 1e-4
        boundary_mask = (expected_lambda <= lambda_lower_bound) | (
            actual_lambda <= lambda_lower_bound
        )
        if np.all(boundary_mask):
            lambda_result = _skipped_result(
                "l_remle comparison skipped (all values at optimization boundary)"
            )
        elif np.any(boundary_mask):
            non_boundary_actual = actual_lambda[~boundary_mask]
            non_boundary_expected = expected_lambda[~boundary_mask]
            lambda_result = compare_arrays(
                non_boundary_actual,
                non_boundary_expected,
                config.lambda_rtol,
                config.atol,
                f"l_remle (excluding {np.sum(boundary_mask)} boundary values)",
            )
        else:
            lambda_result = compare_arrays(
                actual_lambda,
                expected_lambda,
                config.lambda_rtol,
                config.atol,
                "l_remle",
            )

        # l_mle with boundary handling
        actual_lmle = np.array(
            [r.l_mle if r.l_mle is not None else np.nan for r in actual]
        )
        expected_lmle = np.array(
            [r.l_mle if r.l_mle is not None else np.nan for r in expected]
        )
        lambda_upper_bound = 1e4
        boundary_mask_mle = (
            (expected_lmle <= lambda_lower_bound)
            | (actual_lmle <= lambda_lower_bound)
            | (expected_lmle >= lambda_upper_bound)
            | (actual_lmle >= lambda_upper_bound)
        )
        if np.all(boundary_mask_mle):
            l_mle_result = _skipped_result(
                "l_mle comparison skipped (all values at optimization boundary)"
            )
        elif np.any(boundary_mask_mle):
            non_boundary_actual_mle = actual_lmle[~boundary_mask_mle]
            non_boundary_expected_mle = expected_lmle[~boundary_mask_mle]
            l_mle_result = compare_arrays(
                non_boundary_actual_mle,
                non_boundary_expected_mle,
                config.lambda_rtol,
                config.atol,
                f"l_mle (excluding {np.sum(boundary_mask_mle)} boundary values)",
            )
        else:
            l_mle_result = compare_arrays(
                actual_lmle,
                expected_lmle,
                config.lambda_rtol,
                config.atol,
                "l_mle",
            )

    elif is_score_test:
        # Score test: compare p_score, skip Wald-specific columns
        actual_pscore = np.array([r.p_score for r in actual])
        expected_pscore = np.array([r.p_score for r in expected])
        p_score_result = compare_arrays(
            actual_pscore, expected_pscore, config.pvalue_rtol, config.atol, "p_score"
        )
        pwald_result = _skipped_result("p_wald skipped (Score test)")
        logl_result = _skipped_result("logl_H1 skipped (Score test)")
        lambda_result = _skipped_result("l_remle skipped (Score test)")
    elif is_lrt_test:
        # LRT: compare p_lrt and l_mle, skip Wald-specific columns
        actual_plrt = np.array([r.p_lrt for r in actual])
        expected_plrt = np.array([r.p_lrt for r in expected])
        p_lrt_result = compare_arrays(
            actual_plrt, expected_plrt, config.pvalue_rtol, config.atol, "p_lrt"
        )

        actual_lmle = np.array(
            [r.l_mle if r.l_mle is not None else np.nan for r in actual]
        )
        expected_lmle = np.array(
            [r.l_mle if r.l_mle is not None else np.nan for r in expected]
        )

        # Lambda MLE comparison with boundary handling (same as REML)
        lambda_lower_bound = 1e-4
        lambda_upper_bound = 1e4  # Also check upper bound for MLE
        boundary_mask = (
            (expected_lmle <= lambda_lower_bound)
            | (actual_lmle <= lambda_lower_bound)
            | (expected_lmle >= lambda_upper_bound)
            | (actual_lmle >= lambda_upper_bound)
        )

        if np.all(boundary_mask):
            l_mle_result = _skipped_result(
                "l_mle comparison skipped (all values at optimization boundary)"
            )
        elif np.any(boundary_mask):
            non_boundary_actual = actual_lmle[~boundary_mask]
            non_boundary_expected = expected_lmle[~boundary_mask]
            l_mle_result = compare_arrays(
                non_boundary_actual,
                non_boundary_expected,
                config.lambda_rtol,
                config.atol,
                f"l_mle (excluding {np.sum(boundary_mask)} boundary values)",
            )
        else:
            l_mle_result = compare_arrays(
                actual_lmle,
                expected_lmle,
                config.lambda_rtol,
                config.atol,
                "l_mle",
            )

        pwald_result = _skipped_result("p_wald skipped (LRT)")
        logl_result = _skipped_result("logl_H1 skipped (LRT)")
        lambda_result = _skipped_result("l_remle skipped (LRT)")
    else:
        # Wald test: compare Wald-specific columns
        actual_pwald = np.array(
            [r.p_wald if r.p_wald is not None else np.nan for r in actual]
        )
        expected_pwald = np.array(
            [r.p_wald if r.p_wald is not None else np.nan for r in expected]
        )
        pwald_result = compare_arrays(
            actual_pwald, expected_pwald, config.pvalue_rtol, config.atol, "p_wald"
        )

        actual_logl = np.array(
            [r.logl_H1 if r.logl_H1 is not None else 0.0 for r in actual]
        )
        expected_logl = np.array(
            [r.logl_H1 if r.logl_H1 is not None else 0.0 for r in expected]
        )

        # Skip logl_H1 comparison if reference is all zeros (short format)
        if np.allclose(expected_logl, 0.0):
            logl_result = _skipped_result(
                "logl_H1 skipped (reference missing logl_H1 column)"
            )
        else:
            logl_result = compare_arrays(
                actual_logl, expected_logl, config.logl_rtol, config.atol, "logl_H1"
            )

        actual_lambda = np.array(
            [r.l_remle if r.l_remle is not None else np.nan for r in actual]
        )
        expected_lambda = np.array(
            [r.l_remle if r.l_remle is not None else np.nan for r in expected]
        )

        # Lambda comparison with boundary handling
        lambda_lower_bound = 1e-4
        boundary_mask = (expected_lambda <= lambda_lower_bound) | (
            actual_lambda <= lambda_lower_bound
        )

        if np.all(boundary_mask):
            lambda_result = _skipped_result(
                "l_remle comparison skipped (all values at optimization boundary)"
            )
        elif np.any(boundary_mask):
            non_boundary_actual = actual_lambda[~boundary_mask]
            non_boundary_expected = expected_lambda[~boundary_mask]
            lambda_result = compare_arrays(
                non_boundary_actual,
                non_boundary_expected,
                config.lambda_rtol,
                config.atol,
                f"l_remle (excluding {np.sum(boundary_mask)} boundary values)",
            )
        else:
            lambda_result = compare_arrays(
                actual_lambda,
                expected_lambda,
                config.lambda_rtol,
                config.atol,
                "l_remle",
            )

    # Overall pass if all relevant columns pass and no mismatched SNPs
    # Note: For LRT, beta/se are NaN so we skip those checks
    if is_all_tests:
        # All-tests: ALL column comparisons must pass
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
            and len(mismatched) == 0
        )
    elif is_lrt_test:
        # LRT has no beta/se, only check AF and LRT-specific columns
        all_passed = af_result.passed and len(mismatched) == 0
        all_passed = all_passed and p_lrt_result.passed and l_mle_result.passed
    else:
        all_passed = (
            beta_result.passed
            and se_result.passed
            and af_result.passed
            and len(mismatched) == 0
        )
        # Add test-type specific checks
        if is_score_test:
            all_passed = all_passed and p_score_result.passed
        else:
            all_passed = (
                all_passed
                and pwald_result.passed
                and logl_result.passed
                and lambda_result.passed
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
