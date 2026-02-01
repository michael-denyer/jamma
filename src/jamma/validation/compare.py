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

    Parses the tab-separated .assoc.txt format produced by GEMMA's -lmm 1 mode.
    Returns a list of AssocResult dataclass instances for comparison.

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
        # Validate header matches expected format
        expected_cols = [
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
        actual_cols = header.split("\t")
        if actual_cols != expected_cols:
            raise ValueError(
                f"Unexpected header format. Expected: {expected_cols}, "
                f"Got: {actual_cols}"
            )

        for line in f:
            fields = line.strip().split("\t")
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
        p_wald: Comparison result for p-values.
        logl_H1: Comparison result for log-likelihoods.
        l_remle: Comparison result for lambda REML values.
        af: Comparison result for allele frequencies.
        mismatched_snps: List of SNP rs IDs that don't match between files.
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
    - p_wald: pvalue_rtol (CDF computations may differ)
    - logl_H1: logl_rtol (log-likelihood values)
    - l_remle: lambda_rtol (variance ratio estimates)
    - af: af_rtol (allele frequencies)

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

    # Check for SNP count mismatch
    if len(actual) != len(expected):
        return AssocComparisonResult(
            passed=False,
            n_snps=len(actual),
            beta=ComparisonResult(
                passed=False,
                max_abs_diff=np.inf,
                max_rel_diff=np.inf,
                worst_location=None,
                message=f"SNP count mismatch: {len(actual)} vs {len(expected)}",
            ),
            se=ComparisonResult(
                passed=False,
                max_abs_diff=0,
                max_rel_diff=0,
                worst_location=None,
                message="Skipped due to SNP count mismatch",
            ),
            p_wald=ComparisonResult(
                passed=False,
                max_abs_diff=0,
                max_rel_diff=0,
                worst_location=None,
                message="Skipped due to SNP count mismatch",
            ),
            logl_H1=ComparisonResult(
                passed=False,
                max_abs_diff=0,
                max_rel_diff=0,
                worst_location=None,
                message="Skipped due to SNP count mismatch",
            ),
            l_remle=ComparisonResult(
                passed=False,
                max_abs_diff=0,
                max_rel_diff=0,
                worst_location=None,
                message="Skipped due to SNP count mismatch",
            ),
            af=ComparisonResult(
                passed=False,
                max_abs_diff=0,
                max_rel_diff=0,
                worst_location=None,
                message="Skipped due to SNP count mismatch",
            ),
            mismatched_snps=[],
        )

    # Check for mismatched SNP IDs
    mismatched = []
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if a.rs != e.rs:
            mismatched.append(f"{i}:{a.rs}!={e.rs}")

    # Extract arrays for comparison
    actual_beta = np.array([r.beta for r in actual])
    expected_beta = np.array([r.beta for r in expected])
    actual_se = np.array([r.se for r in actual])
    expected_se = np.array([r.se for r in expected])
    actual_pwald = np.array([r.p_wald for r in actual])
    expected_pwald = np.array([r.p_wald for r in expected])
    actual_logl = np.array([r.logl_H1 for r in actual])
    expected_logl = np.array([r.logl_H1 for r in expected])
    actual_lambda = np.array([r.l_remle for r in actual])
    expected_lambda = np.array([r.l_remle for r in expected])
    # For AF comparison, normalize both to MAF (<=0.5) since GEMMA reports AF
    # and JAMMA reports MAF. The values are complements (af vs 1-af).
    actual_af = np.array([r.af for r in actual])
    expected_af = np.array([r.af for r in expected])
    # Normalize expected to MAF for comparison
    expected_maf = np.minimum(expected_af, 1.0 - expected_af)

    # Compare each column with appropriate tolerance
    beta_result = compare_arrays(
        actual_beta, expected_beta, config.beta_rtol, config.atol, "beta"
    )
    se_result = compare_arrays(
        actual_se, expected_se, config.se_rtol, config.atol, "se"
    )
    pwald_result = compare_arrays(
        actual_pwald, expected_pwald, config.pvalue_rtol, config.atol, "p_wald"
    )
    logl_result = compare_arrays(
        actual_logl, expected_logl, config.logl_rtol, config.atol, "logl_H1"
    )
    lambda_result = compare_arrays(
        actual_lambda, expected_lambda, config.lambda_rtol, config.atol, "l_remle"
    )
    af_result = compare_arrays(
        actual_af, expected_maf, config.af_rtol, config.atol, "af"
    )

    # Overall pass if all columns pass and no mismatched SNPs
    all_passed = (
        beta_result.passed
        and se_result.passed
        and pwald_result.passed
        and logl_result.passed
        and lambda_result.passed
        and af_result.passed
        and len(mismatched) == 0
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
    )
