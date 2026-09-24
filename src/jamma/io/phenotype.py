"""GEMMA ``-p`` phenotype files.

Whitespace-separated, no header, one row per sample in genotype-file order
(positional, as GEMMA matches ``-p``, ``-c`` and ``-widv``). ``NA`` and ``-9``
mark a missing value, as they do in a ``.fam`` phenotype column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MISSING_PHENOTYPE_TOKENS = ("-9", "NA")


def phenotype_values(tokens: np.ndarray) -> np.ndarray:
    """Parse phenotype strings to float64, NaN where a token marks missing.

    Raises:
        ValueError: If a token is neither missing nor a number.
    """
    values = tokens.copy()
    missing = np.isin(values, MISSING_PHENOTYPE_TOKENS)
    values[missing] = "0"
    phenotypes = values.astype(np.float64)
    phenotypes[missing] = np.nan
    return phenotypes


def read_phenotype_table(path: Path) -> np.ndarray:
    """Read a ``-p`` file as an ``(n_rows, n_columns)`` array of strings.

    Raises:
        ValueError: If the file cannot be read, is empty, or its rows have
            different column counts.
    """
    try:
        table = np.loadtxt(path, dtype=str, ndmin=2)
    except (ValueError, OSError) as e:
        raise ValueError(f"Failed to read phenotype file {path}: {e}") from e
    if table.size == 0:
        raise ValueError(f"Phenotype file is empty: {path}")
    return table


def phenotype_file_column(table: np.ndarray, column: int) -> np.ndarray:
    """Return phenotype ``column`` (1-based) of a table ``read_phenotype_table`` read.

    Raises:
        ValueError: If the table has fewer columns than ``column``.
    """
    n_cols = table.shape[1]
    if column > n_cols:
        raise ValueError(
            f"phenotype column {column} exceeds available columns "
            f"in phenotype file ({n_cols} column{'s' if n_cols != 1 else ''} "
            "available)"
        )
    return phenotype_values(table[:, column - 1])
