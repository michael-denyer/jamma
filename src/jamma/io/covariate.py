"""GEMMA-format covariate file I/O.

This module provides reading of GEMMA-format covariate files, which are used
to specify confounders (e.g., age, sex, population structure PCs) in LMM analysis.

GEMMA covariate file format:
- Whitespace/tab/space delimited (no header row)
- Row order matches .fam file (positional matching, not ID-based)
- Missing values encoded as "NA" (case-sensitive)
- First column MUST be all 1s if user wants an intercept in the model
- GEMMA does NOT auto-add an intercept column
"""

from pathlib import Path

import numpy as np
from loguru import logger


def read_covariate_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read GEMMA-format covariate file.

    Parses a whitespace-delimited covariate file with no header. Each row
    corresponds to a sample in positional order (matching .fam file). Missing
    values are encoded as "NA" (case-sensitive, per GEMMA behavior).

    Args:
        path: Path to the covariate file.

    Returns:
        Tuple of (covariates, indicator_cvt):
        - covariates: (n_samples, n_cvt) float64 array with NaN for missing values
        - indicator_cvt: (n_samples,) int32 array with 0 for rows containing any NA,
          1 for rows with all valid values

    Raises:
        ValueError: If file is empty, rows have inconsistent column counts,
            or values cannot be parsed as numeric (except "NA" for missing).

    Example:
        Covariate file contents (intercept + age + sex):
        ```
        1  35.0  0
        1  42.0  1
        1  NA    1
        1  28.0  0
        ```

        >>> covariates, indicator = read_covariate_file(Path("covariates.txt"))
        >>> covariates.shape
        (4, 3)
        >>> indicator  # Row 3 has NA, marked invalid
        array([1, 1, 0, 1], dtype=int32)
    """
    rows: list[list[str]] = []

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:  # Skip empty lines (GEMMA behavior)
                continue
            parts = stripped.split()
            rows.append(parts)

    if not rows:
        raise ValueError(f"Covariate file is empty: {path}")

    n_samples = len(rows)
    n_cvt = len(rows[0])

    # Validate all rows have same column count
    for i, row in enumerate(rows):
        if len(row) != n_cvt:
            raise ValueError(
                f"Covariate file row {i + 1} has {len(row)} columns "
                f"but expected {n_cvt} (based on first row)"
            )

    covariates = np.zeros((n_samples, n_cvt), dtype=np.float64)
    indicator_cvt = np.ones(n_samples, dtype=np.int32)

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if val == "NA":  # Case-sensitive, matches GEMMA
                covariates[i, j] = np.nan
                indicator_cvt[i] = 0  # Any NA invalidates the entire row
            else:
                try:
                    covariates[i, j] = float(val)
                except ValueError as e:
                    raise ValueError(
                        f"Covariate file row {i + 1}, column {j + 1}: "
                        f"cannot parse '{val}' as numeric (use 'NA' for missing)"
                    ) from e

    return covariates, indicator_cvt


def encode_categorical_covariates(
    covariates: np.ndarray, cat_columns: list[int]
) -> np.ndarray:
    """One-hot encode specified covariate columns as dummy variables.

    JAMMA-specific feature. GEMMA's -cat flag is for SNP categories in VC mode,
    not covariate encoding. Users of GEMMA would pre-encode categoricals as
    dummy variables in the covariate file. JAMMA's -cat automates this step.

    For each categorical column, sorted unique non-NaN values are identified.
    The first sorted value is the reference level and is dropped. The remaining
    k-1 levels produce one dummy column each (0/1). Rows with NaN in the
    original categorical column get NaN in all resulting dummy columns.

    After encoding, any zero-variance dummy columns are warned about and removed
    (they would cause rank-deficient W).

    Args:
        covariates: (n_samples, n_cvt) float64 array from read_covariate_file.
        cat_columns: 1-indexed column indices to treat as categorical
            (e.g., [1, 3] means columns 1 and 3).

    Returns:
        Expanded covariate array with categorical columns replaced by dummies.

    Raises:
        ValueError: If any column index is out of range [1, n_cvt].
    """
    n_samples, n_cvt = covariates.shape

    # Deduplicate column indices (user may pass -cat 2 -cat 2)
    cat_columns = sorted(set(cat_columns))

    # Validate column indices
    for col in cat_columns:
        if col < 1 or col > n_cvt:
            raise ValueError(
                f"-cat column index {col} out of range [1, {n_cvt}] "
                f"(covariate file has {n_cvt} columns)"
            )

    zero_indexed_cat_columns = {c - 1 for c in cat_columns}

    # Build the output column list once, in ascending input order. Each
    # original column contributes one block: itself unchanged (not
    # categorical), its k-1 dummy columns, a NaN marker column, or nothing
    # (constant categorical with no missing rows).
    blocks: list[np.ndarray] = []
    is_dummy: list[bool] = []

    for col_idx in range(n_cvt):
        col_values = covariates[:, col_idx]

        if col_idx not in zero_indexed_cat_columns:
            blocks.append(col_values.reshape(n_samples, 1))
            is_dummy.append(False)
            continue

        # Get sorted unique non-NaN values
        non_nan_values = col_values[~np.isnan(col_values)]
        unique_levels = np.sort(np.unique(non_nan_values))

        if len(unique_levels) > 20:
            logger.warning(
                f"Categorical column {col_idx + 1} has {len(unique_levels)} unique "
                f"levels — this is unusually high for a categorical variable. "
                f"Verify this column is not continuous."
            )

        # Drop reference level (first sorted value)
        dummy_levels = unique_levels[1:]  # k-1 levels

        if len(dummy_levels) == 0:
            # Single level or all NaN -- no dummies to create.
            # If column has NaN rows, keep a NaN-only column to preserve the
            # missingness signal for pipeline sample filtering. Otherwise
            # remove the column entirely (it's constant, zero variance).
            nan_mask_col = np.isnan(col_values)
            if np.any(nan_mask_col):
                marker = np.zeros((n_samples, 1), dtype=np.float64)
                marker[nan_mask_col, 0] = np.nan
                blocks.append(marker)
                is_dummy.append(False)
                logger.warning(
                    f"Categorical column {col_idx + 1} has only 1 non-NaN level; "
                    f"kept as NaN marker column to preserve {int(nan_mask_col.sum())} "
                    f"missing-value row(s)"
                )
            continue

        # Create dummy columns
        n_dummies = len(dummy_levels)
        dummies = np.zeros((n_samples, n_dummies), dtype=np.float64)
        nan_mask = np.isnan(col_values)

        for i, level in enumerate(dummy_levels):
            dummies[:, i] = (col_values == level).astype(np.float64)
            # Propagate NaN for missing rows
            dummies[nan_mask, i] = np.nan

        blocks.append(dummies)
        is_dummy.extend([True] * n_dummies)

    result = np.hstack(blocks) if blocks else np.empty((n_samples, 0), dtype=np.float64)

    # Check for zero-variance among ONLY the newly created dummy columns
    dummy_col_indices = [i for i, d in enumerate(is_dummy) if d]
    if dummy_col_indices:
        col_vars = np.nanvar(result[:, dummy_col_indices], axis=0)
        zero_var_positions = [
            dummy_col_indices[i] for i, v in enumerate(col_vars) if v == 0.0
        ]
        n_zero_var = len(zero_var_positions)

        if n_zero_var > 0:
            logger.warning(
                f"Dropped {n_zero_var} zero-variance dummy column"
                f"{'s' if n_zero_var > 1 else ''} from categorical encoding"
            )
            keep_mask = np.ones(result.shape[1], dtype=bool)
            for idx in zero_var_positions:
                keep_mask[idx] = False
            result = result[:, keep_mask]

    return result
