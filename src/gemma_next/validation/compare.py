"""Comparison utilities for validating GEMMA-Next output against reference GEMMA.

This module provides structured comparison functions that return detailed results
rather than raising exceptions, enabling programmatic validation workflows.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from gemma_next.validation.tolerances import ToleranceConfig


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
            message=f"{name} shape mismatch: actual {actual.shape} vs expected {expected.shape}",
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
            message=f"{name} comparison passed (max abs diff: {max_abs_diff:.2e}, max rel diff: {max_rel_diff:.2e})",
        )

    except AssertionError as e:
        # Compute detailed diagnostics
        abs_diff = np.abs(actual - expected)
        max_abs_diff = float(np.max(abs_diff))

        # Find location of worst absolute difference
        # Convert numpy int64 to plain int for cleaner display/serialization
        worst_idx = tuple(int(i) for i in np.unravel_index(np.argmax(abs_diff), abs_diff.shape))

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


def load_gemma_assoc(path: Path) -> "pd.DataFrame":
    """Load GEMMA association results from .assoc.txt format.

    Note: This is a placeholder for Phase 3 implementation.

    Args:
        path: Path to the association results file (.assoc.txt).

    Returns:
        pandas DataFrame with association results.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError(
        "load_gemma_assoc will be implemented in Phase 3 (LMM testing). "
        f"Cannot load {path}"
    )


def compare_assoc_results(
    actual: "pd.DataFrame",
    expected: "pd.DataFrame",
    config: ToleranceConfig | None = None,
) -> dict[str, ComparisonResult]:
    """Compare association results with column-appropriate tolerances.

    Note: This is a placeholder for Phase 3 implementation.

    Args:
        actual: Computed association results DataFrame.
        expected: Reference GEMMA association results DataFrame.
        config: Tolerance configuration. Uses default if None.

    Returns:
        Dictionary mapping column names to ComparisonResult objects.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError(
        "compare_assoc_results will be implemented in Phase 3 (LMM testing)."
    )
