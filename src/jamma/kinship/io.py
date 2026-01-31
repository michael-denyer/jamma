"""Kinship matrix I/O in GEMMA format."""

from pathlib import Path

import numpy as np


def write_kinship_matrix(K: np.ndarray, path: Path) -> None:
    """Write kinship matrix in GEMMA .cXX.txt format.

    GEMMA format specifications (from legacy/src/param.cpp:1886-1911):
    - outfile.precision(10): 10 significant digits using general format
    - Tab separator between values
    - Newline after each row
    - No header row
    - No sample IDs in matrix file

    Args:
        K: Kinship matrix (n x n), should be symmetric.
        path: Output file path (typically .cXX.txt).

    Example:
        >>> write_kinship_matrix(K, Path("output/result.cXX.txt"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for i in range(K.shape[0]):
            # Use .10g format: 10 significant digits, general format
            # This matches C++ iostream precision(10) behavior
            values = [f"{K[i, j]:.10g}" for j in range(K.shape[1])]
            f.write("\t".join(values) + "\n")
