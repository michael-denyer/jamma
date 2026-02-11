"""Kinship matrix I/O in GEMMA format."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np


def read_kinship_matrix(path: Path, n_samples: int | None = None) -> np.ndarray:
    """Read kinship matrix from GEMMA .cXX.txt format.

    Args:
        path: Path to kinship matrix file (.cXX.txt format)
        n_samples: Expected number of samples (optional validation)

    Returns:
        Kinship matrix as numpy array (n x n)

    Raises:
        ValueError: If matrix is not square, not symmetric, or dimension mismatch
    """
    # Load matrix - handles tab and space separated
    K = np.loadtxt(path, dtype=np.float64)

    # Validate square
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kinship matrix must be square, got shape {K.shape}")

    # Validate dimension if n_samples provided
    if n_samples is not None and K.shape[0] != n_samples:
        raise ValueError(
            f"Kinship matrix dimension {K.shape[0]} does not match "
            f"expected n_samples={n_samples}"
        )

    # Validate symmetric
    if not np.allclose(K, K.T, rtol=1e-10):
        raise ValueError("Kinship matrix is not symmetric")

    return K


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


def write_loco_kinship_matrices(
    loco_kinships: Iterator[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str = "result",
) -> list[Path]:
    """Write per-chromosome LOCO kinship matrices to disk.

    For each (chr_name, K) pair yielded by the iterator, writes the matrix
    to ``{output_dir}/{prefix}.loco.cXX.chr{chr_name}.txt`` using GEMMA
    format via ``write_kinship_matrix()``.

    This is a convenience wrapper for the ``gk -loco`` standalone command.

    Args:
        loco_kinships: Iterator yielding (chromosome_name, kinship_matrix)
            pairs. Typically produced by ``compute_loco_kinship_streaming()``.
        output_dir: Directory for output files (created if needed).
        prefix: Filename prefix (default "result").

    Returns:
        List of Paths to the written kinship files.

    Example:
        >>> from jamma.kinship import compute_loco_kinship_streaming
        >>> loco_iter = compute_loco_kinship_streaming(Path("data/study"))
        >>> paths = write_loco_kinship_matrices(loco_iter, Path("output"))
        >>> len(paths)  # One file per chromosome
        19
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for chr_name, K in loco_kinships:
        kinship_path = output_dir / f"{prefix}.loco.cXX.chr{chr_name}.txt"
        write_kinship_matrix(K, kinship_path)
        written.append(kinship_path)

    return written
