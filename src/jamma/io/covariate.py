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
