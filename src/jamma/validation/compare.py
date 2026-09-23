"""Comparison utilities for validating JAMMA output against reference GEMMA.

This module provides structured comparison functions that return detailed results
rather than raising exceptions, enabling programmatic validation workflows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np

from jamma.lmm.assoc_output import AssocResult
from jamma.lmm.schema import MODE_SPECS, LmmMode
from jamma.validation.tolerances import LambdaBoundaryPolicy, ToleranceConfig


@dataclass(frozen=True)
class ComparisonResult:
    """Result of a numerical array comparison.

    Provides structured information about pass/fail status and the nature
    of any discrepancies found.

    Attributes:
        passed: Whether the comparison passed within tolerance.
        max_abs_diff: Maximum absolute difference found.
        max_rel_diff: Maximum relative difference found (inf if expected was 0).
        worst_location: Index tuple of the worst mismatch, or None if passed.
        failed_indices: Flat indices of every value outside tolerance.
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
    failed_indices: tuple[int, ...]
    message: str


def compare_arrays(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float,
    atol: float,
    name: str = "array",
    *,
    floor: np.ndarray | None = None,
    exempt: np.ndarray | None = None,
) -> ComparisonResult:
    """Compare two arrays with tolerance and return structured result.

    Uses numpy.isclose semantics, |a - b| <= atol + rtol * |b|, with NaN equal
    to NaN, and returns a structured ComparisonResult instead of raising.

    Args:
        actual: The computed array to validate.
        expected: The reference array to compare against.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        name: Name to use in error messages for context.
        floor: Optional per-element absolute tolerance added to ``atol``, for
            columns whose scale is set by another column (beta by its standard
            error). NaN entries count as zero.
        exempt: Optional boolean mask of entries that pass without comparison
            and are left out of the reported differences.

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
        return _failed_result(
            f"{name} shape mismatch: "
            f"actual {actual.shape} vs expected {expected.shape}",
            tuple(range(max(actual.size, expected.size))),
        )

    if actual.size == 0:
        return ComparisonResult(
            passed=True,
            max_abs_diff=0.0,
            max_rel_diff=0.0,
            worst_location=None,
            failed_indices=(),
            message=f"{name} comparison passed (empty arrays)",
        )

    atol_eff: float | np.ndarray = atol
    if floor is not None:
        atol_eff = atol + np.nan_to_num(floor, nan=0.0)
    close = np.isclose(actual, expected, rtol=rtol, atol=atol_eff, equal_nan=True)
    abs_diff = np.abs(actual - expected)
    if exempt is not None:
        close |= exempt
        abs_diff[exempt] = 0.0
    if bool(np.all(close)):
        # Passed - compute stats anyway for reporting
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
            failed_indices=(),
            message=(
                f"{name} comparison passed "
                f"(max abs diff: {max_abs_diff:.2e}, max rel diff: {max_rel_diff:.2e})"
            ),
        )

    else:
        max_abs_diff = float(np.max(abs_diff))

        # Find location of worst absolute difference
        # Convert numpy int64 to plain int for cleaner display/serialization
        worst_idx_raw = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        worst_idx = tuple(int(i) for i in worst_idx_raw)

        # Relative difference at worst location
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / np.abs(expected)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, np.inf)
        if exempt is not None:
            rel_diff[exempt] = 0.0
        max_rel_diff = float(np.max(rel_diff))
        failed_indices = tuple(int(i) for i in np.flatnonzero(~close))

        return ComparisonResult(
            passed=False,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            worst_location=worst_idx,
            failed_indices=failed_indices,
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

    The four canonical layouts are the ``ModeSpec.header`` of each LMM mode. Three
    extra layouts are GEMMA-version quirks the schema does not model: they differ
    only by an optional ``logl_H1`` column that some runs omit.
    """
    cols = {s.test_type: tuple(s.header.split("\t")) for s in MODE_SPECS.values()}
    wald, lrt, all_tests = cols["wald"], cols["lrt"], cols["all"]
    return frozenset(
        {
            wald,  # Wald with logl_H1
            tuple(c for c in wald if c != "logl_H1"),  # Wald without logl_H1
            cols["score"],
            lrt,  # LRT with logl_H1
            tuple(c for c in lrt if c != "logl_H1"),  # LRT without logl_H1
            all_tests,  # all-tests with logl_H1
            tuple(c for c in all_tests if c != "logl_H1"),  # all-tests without
        }
    )


_ASSOC_HEADER_LAYOUTS = _assoc_header_layouts()


def _opt_float(row: dict[str, str], name: str) -> float | None:
    """Cast an optional .assoc.txt cell, or None when its layout omits the column."""
    raw = row.get(name)
    return None if raw is None else float(raw)


def _float_or_nan(row: dict[str, str], name: str) -> float:
    """Cast a required .assoc.txt cell, or NaN when a layout omits the column.

    Separate from ``_opt_float`` because the two absences mean different things.
    A missing optional column is None, meaning the test does not report it. A
    missing beta or se is NaN, ``AssocResult``'s default, because GEMMA's
    LRT formats do not write them.
    """
    raw = row.get(name)
    return float("nan") if raw is None else float(raw)


@dataclass(frozen=True)
class AssocDataset(Sequence[AssocResult]):
    """Association rows with a mode that survives an empty file or slice."""

    mode: LmmMode
    rows: tuple[AssocResult, ...]

    def __post_init__(self) -> None:
        _mode_from_rows(self)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[AssocResult]:
        return iter(self.rows)

    @overload
    def __getitem__(self, index: int) -> AssocResult: ...

    @overload
    def __getitem__(self, index: slice) -> AssocDataset: ...

    def __getitem__(self, index: int | slice) -> AssocResult | AssocDataset:
        if isinstance(index, slice):
            return AssocDataset(self.mode, self.rows[index])
        return self.rows[index]


def load_gemma_assoc(
    path: Path, *, mode: LmmMode | None = None, require_logl: bool = False
) -> AssocDataset:
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
        mode: The LMM mode the header must declare. Any mode when None.
        require_logl: Reject a header that omits ``logl_H1`` for a mode that
            carries it. GEMMA omits the column in some layouts; JAMMA never does.

    Returns:
        A sequence of AssocResult rows retaining the mode declared by its header.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the header is not an accepted layout, declares a mode
            other than ``mode``, omits a required ``logl_H1``, or if two rows
            share an rs ID.

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
                    beta=_float_or_nan(row, "beta"),
                    se=_float_or_nan(row, "se"),
                    logl_H1=_opt_float(row, "logl_H1"),
                    l_remle=_opt_float(row, "l_remle"),
                    p_wald=_opt_float(row, "p_wald"),
                    p_score=_opt_float(row, "p_score"),
                    l_mle=_opt_float(row, "l_mle"),
                    p_lrt=_opt_float(row, "p_lrt"),
                )
            )
    header_mode = _MODE_BY_P_VALUE_FIELDS[frozenset(cols) & _P_VALUE_FIELDS]
    if mode is not None and header_mode != mode:
        raise ValueError(f"Expected a mode {mode} header, got mode {header_mode}")
    carried = {c.field_name for c in MODE_SPECS[header_mode].stat_columns}
    if require_logl and "logl_H1" in carried - set(cols):
        raise ValueError(f"Mode {header_mode} header omits logl_H1: {list(cols)}")
    if len({row.rs for row in results}) != len(results):
        raise ValueError(f"Duplicate SNP IDs in {path}")
    return AssocDataset(header_mode, tuple(results))


@dataclass(frozen=True)
class AssocComparisonResult:
    """Result of comparing two sets of association results.

    Attributes:
        n_snps: Number of SNPs in the actual results.
        columns: One comparison per column the mode carries, plus ``af``. A
            ``logl_H1`` the reference omits is absent; a SNP count mismatch
            leaves the single column ``n_snps``.
        mismatched_snps: ``index:actual!=expected`` for each differing rs ID.

    Example:
        >>> comparison["l_remle"].failed_indices
        (3,)
    """

    n_snps: int
    columns: Mapping[str, ComparisonResult]
    mismatched_snps: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether every rs ID matches and every column passed."""
        return not self.mismatched_snps and all(
            column.passed for column in self.columns.values()
        )

    def __getitem__(self, column: str) -> ComparisonResult:
        return self.columns[column]


_P_VALUE_FIELDS = frozenset({"p_wald", "p_lrt", "p_score"})
_OPTIONAL_FIELDS = _P_VALUE_FIELDS | {"logl_H1", "l_remle", "l_mle"}

# No two modes carry the same set of p-value columns, so the set a file's rows
# carry names its mode outright.
_MODE_BY_P_VALUE_FIELDS: dict[frozenset[str], LmmMode] = {
    frozenset(
        c.field_name for c in spec.stat_columns if c.field_name in _P_VALUE_FIELDS
    ): mode
    for mode, spec in MODE_SPECS.items()
}


_BETA_MODES = frozenset(
    mode
    for mode, spec in MODE_SPECS.items()
    if any(c.field_name == "beta" for c in spec.stat_columns)
)


def _present_fields(rows: Sequence[AssocResult]) -> frozenset[str]:
    """Optional fields every row carries; rows that disagree are a schema error."""

    def present(row: AssocResult) -> frozenset[str]:
        return frozenset(f for f in _OPTIONAL_FIELDS if getattr(row, f) is not None)

    first = present(rows[0])
    for index, row in enumerate(rows):
        if present(row) != first:
            raise ValueError(f"Inconsistent association schema at row {index}")
    return first


def _mode_from_rows(rows: Sequence[AssocResult]) -> LmmMode | None:
    """Validate current rows against a declared mode, or infer an undeclared mode."""
    if isinstance(rows, AssocDataset):
        # The tuple is fixed, but its AssocResult elements remain mutable.
        row_mode = _mode_from_rows(rows.rows)
        if row_mode is not None and row_mode != rows.mode:
            raise ValueError(f"Rows carry mode {row_mode}, expected mode {rows.mode}")
        return rows.mode
    if not rows:
        return None
    p_fields = _present_fields(rows) & _P_VALUE_FIELDS
    try:
        mode = _MODE_BY_P_VALUE_FIELDS[p_fields]
    except KeyError:
        raise ValueError(
            f"Association schema has no recognized p-value columns: {sorted(p_fields)}"
        ) from None
    if mode not in _BETA_MODES and not all(
        np.isnan(row.beta) and np.isnan(row.se) for row in rows
    ):
        raise ValueError(f"Mode {mode} rows carry beta or se; they must be NaN")
    return mode


def _failed_result(
    message: str, failed_indices: tuple[int, ...] = ()
) -> ComparisonResult:
    """A failing result with no measurable difference to report."""
    return ComparisonResult(
        passed=False,
        max_abs_diff=np.inf,
        max_rel_diff=np.inf,
        worst_location=None,
        failed_indices=failed_indices,
        message=message,
    )


def _column(field: str, rows: Sequence[AssocResult]) -> np.ndarray:
    """Extract one AssocResult field across rows, with NaN for None."""
    return np.array(
        [np.nan if getattr(r, field) is None else getattr(r, field) for r in rows]
    )


def _classify_lambdas(values: np.ndarray, policy: LambdaBoundaryPolicy) -> np.ndarray:
    """Classify optimizer outputs against the bounds used for that run."""
    classes = np.full(values.shape, "interior", dtype=object)
    invalid = (
        ~np.isfinite(values)
        | (values <= 0)
        | (values < policy.lower * (1 - policy.rtol))
        | (values > policy.upper * (1 + policy.rtol))
    )
    classes[values <= policy.lower * (1 + policy.rtol)] = "lower"
    classes[values >= policy.upper * (1 - policy.rtol)] = "upper"
    classes[invalid] = "invalid"
    return classes


def _compare_lambdas(
    actual_arr: np.ndarray,
    expected_arr: np.ndarray,
    policy: LambdaBoundaryPolicy,
    rtol: float,
    atol: float,
    name: str,
    *,
    exempt_upper: bool,
) -> ComparisonResult:
    """Compare lambdas after classifying both optimizer outputs.

    Matching lower-bound hits are exempt from magnitude comparison. MLE may
    also exempt matching upper-bound hits. Paired NaNs represent the same
    degenerate result and are exempt. Every other invalid value or class
    disagreement fails before interior comparison.
    """
    actual_classes = _classify_lambdas(actual_arr, policy)
    expected_classes = _classify_lambdas(expected_arr, policy)
    paired_nan = np.isnan(actual_arr) & np.isnan(expected_arr)
    invalid = (actual_classes == "invalid") | (expected_classes == "invalid")
    class_mismatch = (actual_classes != expected_classes) | (invalid & ~paired_nan)
    matching_boundary = (actual_classes == "lower") & (expected_classes == "lower")
    if exempt_upper:
        matching_boundary |= (actual_classes == "upper") & (expected_classes == "upper")
    exempt = matching_boundary | paired_nan
    if np.any(class_mismatch):
        mismatch_indices = np.flatnonzero(class_mismatch)
        mismatched_actual = actual_arr[mismatch_indices]
        mismatched_expected = expected_arr[mismatch_indices]
        finite = np.isfinite(mismatched_actual) & np.isfinite(mismatched_expected)
        if np.all(finite):
            abs_diffs = np.abs(mismatched_actual - mismatched_expected)
            local_worst = int(np.argmax(abs_diffs))
            max_abs_diff = float(abs_diffs[local_worst])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diffs = abs_diffs / np.abs(mismatched_expected)
            max_rel_diff = float(np.max(rel_diffs))
        else:
            local_worst = int(np.flatnonzero(~finite)[0])
            max_abs_diff = np.inf
            max_rel_diff = np.inf
        index = int(mismatch_indices[local_worst])
        numeric_failure = ~exempt & ~np.isclose(
            actual_arr, expected_arr, rtol=rtol, atol=atol, equal_nan=True
        )
        return ComparisonResult(
            passed=False,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            worst_location=(index,),
            failed_indices=tuple(
                int(i) for i in np.flatnonzero(class_mismatch | numeric_failure)
            ),
            message=(
                f"{name} optimizer-bound class mismatch at ({index},): "
                f"{actual_classes[index]}/{expected_classes[index]} "
                f"(actual={actual_arr[index]!r}, expected={expected_arr[index]!r})"
            ),
        )

    if np.any(exempt):
        name = (
            f"{name} (excluding {int(np.sum(matching_boundary))} matching boundary "
            f"values and {int(np.sum(paired_nan))} paired invalid NaN values)"
        )
    return compare_arrays(actual_arr, expected_arr, rtol, atol, name, exempt=exempt)


def compare_assoc_results(
    actual: Sequence[AssocResult],
    expected: Sequence[AssocResult],
    config: ToleranceConfig | None = None,
) -> AssocComparisonResult:
    """Compare association results with column-appropriate tolerances.

    Compares sequences of AssocResult objects from JAMMA and reference GEMMA output.
    Parsed datasets retain their mode even when empty. Empty in-memory sequences
    have no declared mode; they can still report an empty result or count mismatch.
    Uses appropriate tolerance thresholds for each statistic type:
    - beta: beta_rtol (effect sizes from linear algebra)
    - se: se_rtol (standard errors with sqrt operations)
    - p_wald: pvalue_rtol (CDF computations may differ) - Wald test only
    - p_score: pvalue_rtol (CDF computations may differ) - Score test only
    - p_lrt: p_lrt_rtol (chi-squared magnifies logl differences) - LRT only
    - logl_H1: logl_rtol (log-likelihood values) - Wald test only
    - l_remle: lambda_rtol (variance ratio estimates) - Wald test only
    - l_mle: lambda_rtol (MLE lambda values) - LRT only
    - af: af_atol (absolute; counted-allele frequency, BIM A1)

    Args:
        actual: Computed association results from JAMMA.
        expected: Reference GEMMA association results.
        config: Tolerance configuration. Uses default if None.

    Returns:
        AssocComparisonResult with per-column comparison details.

    Raises:
        ValueError: If the two inputs carry different LMM modes, or if rows
            within one of them disagree about which columns are present.

    Example:
        >>> jamma_results = load_gemma_assoc(Path("jamma_output.assoc.txt"))
        >>> gemma_results = load_gemma_assoc(Path("gemma_output.assoc.txt"))
        >>> comparison = compare_assoc_results(jamma_results, gemma_results)
        >>> comparison.passed
        True
    """
    if config is None:
        config = ToleranceConfig()

    mode = _mode_from_rows(expected)
    actual_mode = _mode_from_rows(actual)
    if actual_mode is not None and mode is not None and actual_mode != mode:
        raise ValueError(
            f"Association schemas differ: actual mode {actual_mode} "
            f"but reference mode {mode}"
        )
    mode = mode if mode is not None else actual_mode

    if len(actual) != len(expected):
        count_result = _failed_result(
            f"SNP count mismatch: {len(actual)} vs {len(expected)}",
            tuple(
                range(min(len(actual), len(expected)), max(len(actual), len(expected)))
            ),
        )
        return AssocComparisonResult(len(actual), {"n_snps": count_result}, ())

    mismatched = tuple(
        f"{i}:{a.rs}!={e.rs}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True))
        if a.rs != e.rs
    )

    def _plain(field: str, rtol: float, atol: float = config.atol) -> ComparisonResult:
        return compare_arrays(
            _column(field, actual), _column(field, expected), rtol, atol, field
        )

    def _beta() -> ComparisonResult:
        return compare_arrays(
            _column("beta", actual),
            _column("beta", expected),
            config.beta_rtol,
            config.atol,
            "beta",
            floor=config.beta_se_floor * np.abs(_column("se", expected)),
        )

    def _logl() -> ComparisonResult:
        missing = tuple(i for i, row in enumerate(actual) if row.logl_H1 is None)
        if missing:
            return _failed_result(
                "logl_H1 column missing from actual association schema", missing
            )
        return _plain("logl_H1", config.logl_rtol)

    def _lambda(field: str, *, exempt_upper: bool) -> ComparisonResult:
        return _compare_lambdas(
            _column(field, actual),
            _column(field, expected),
            config.lambda_boundary,
            config.lambda_rtol,
            config.atol,
            field,
            exempt_upper=exempt_upper,
        )

    # schema.MODE_SPECS says which columns a mode carries; this says how each
    # one is compared.
    column_rules: dict[str, Callable[[], ComparisonResult]] = {
        "af": lambda: _plain("af", 0.0, config.atol + config.af_atol),
        "beta": _beta,
        "se": lambda: _plain("se", config.se_rtol),
        "p_wald": lambda: _plain("p_wald", config.pvalue_rtol),
        "p_score": lambda: _plain("p_score", config.pvalue_rtol),
        "p_lrt": lambda: _plain("p_lrt", config.p_lrt_rtol),
        "logl_H1": _logl,
        "l_remle": lambda: _lambda("l_remle", exempt_upper=False),
        "l_mle": lambda: _lambda("l_mle", exempt_upper=True),
    }
    fields = ["af"]
    if mode is not None:
        fields += [c.field_name for c in MODE_SPECS[mode].stat_columns]
    if all(row.logl_H1 is None for row in expected):
        fields = [f for f in fields if f != "logl_H1"]
    return AssocComparisonResult(
        len(actual), {f: column_rules[f]() for f in fields}, mismatched
    )
