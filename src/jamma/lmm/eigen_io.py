"""Eigendecomposition file I/O in GEMMA format.

Read and write eigenvalue (.eigenD.txt) and eigenvector (.eigenU.txt) files
in GEMMA-compatible format. Used for eigendecomposition reuse across
multi-phenotype workflows.

Format follows GEMMA param.cpp WriteVector/WriteMatrix:
- eigenD: one value per line, 10 significant digits (.10g format)
- eigenU: tab-separated rows, 10 significant digits per value
- No headers in either file
"""

from pathlib import Path

import numpy as np


def read_eigenvalues(path: Path) -> np.ndarray:
    """Read eigenvalues from a GEMMA .eigenD.txt file.

    Args:
        path: Path to eigenvalue file (one value per line).

    Returns:
        1D array of eigenvalues (n_samples,).
    """
    return np.loadtxt(path, dtype=np.float64)


def read_eigenvectors(path: Path) -> np.ndarray:
    """Read eigenvectors from a GEMMA .eigenU.txt file.

    Args:
        path: Path to eigenvector file (tab-separated matrix).

    Returns:
        2D array of eigenvectors (n_samples, n_samples).
    """
    return np.loadtxt(path, dtype=np.float64)


def read_eigen_files(
    eigenD_path: Path,
    eigenU_path: Path,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read both eigenvalue and eigenvector files with validation.

    Validates internal consistency: eigenvalue count must match
    eigenvector rows and columns (square matrix). Optionally validates
    against expected sample count.

    Args:
        eigenD_path: Path to eigenvalue file (.eigenD.txt).
        eigenU_path: Path to eigenvector file (.eigenU.txt).
        n_samples: Expected number of samples (optional validation).

    Returns:
        Tuple of (eigenvalues, eigenvectors).

    Raises:
        ValueError: If dimensions are inconsistent or do not match
            n_samples.
    """
    eigenvalues = read_eigenvalues(eigenD_path)
    eigenvectors = read_eigenvectors(eigenU_path)

    n_eval = eigenvalues.shape[0]

    if eigenvectors.ndim != 2:
        raise ValueError(
            f"Eigenvector file must contain a 2D matrix, "
            f"got {eigenvectors.ndim}D array from {eigenU_path}"
        )

    n_rows, n_cols = eigenvectors.shape

    if n_rows != n_cols:
        raise ValueError(
            f"Eigenvector matrix must be square, got shape "
            f"({n_rows}, {n_cols}) from {eigenU_path}"
        )

    if n_eval != n_rows:
        raise ValueError(
            f"Eigenvalue count ({n_eval}) does not match eigenvector "
            f"dimensions ({n_rows} x {n_cols}). Files may be mismatched: "
            f"{eigenD_path}, {eigenU_path}"
        )

    if n_samples is not None and n_eval != n_samples:
        raise ValueError(
            f"Eigenvalue count ({n_eval}) does not match expected "
            f"n_samples={n_samples}. Eigen files may be from a different dataset."
        )

    return eigenvalues, eigenvectors


def write_eigenvalues(eigenvalues: np.ndarray, path: Path) -> None:
    """Write eigenvalues in GEMMA .eigenD.txt format.

    Writes one eigenvalue per line using 10 significant digits,
    matching GEMMA's precision(10) output.

    Args:
        eigenvalues: 1D array of eigenvalues.
        path: Output file path (typically .eigenD.txt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for val in eigenvalues:
            f.write(f"{val:.10g}\n")


def write_eigenvectors(eigenvectors: np.ndarray, path: Path) -> None:
    """Write eigenvectors in GEMMA .eigenU.txt format.

    Writes tab-separated rows using 10 significant digits per value,
    matching GEMMA's precision(10) output.

    Args:
        eigenvectors: 2D array of eigenvectors (n_samples, n_samples).
        path: Output file path (typically .eigenU.txt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for row in eigenvectors:
            values = [f"{v:.10g}" for v in row]
            f.write("\t".join(values) + "\n")


def write_eigen_files(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    output_dir: Path,
    prefix: str = "result",
) -> tuple[Path, Path]:
    """Write both eigenvalue and eigenvector files.

    Convenience wrapper that writes {prefix}.eigenD.txt and
    {prefix}.eigenU.txt to the specified output directory.

    Args:
        eigenvalues: 1D array of eigenvalues.
        eigenvectors: 2D array of eigenvectors.
        output_dir: Directory for output files.
        prefix: Filename prefix (default "result").

    Returns:
        Tuple of (eigenD_path, eigenU_path).
    """
    output_dir = Path(output_dir)
    eigenD_path = output_dir / f"{prefix}.eigenD.txt"
    eigenU_path = output_dir / f"{prefix}.eigenU.txt"

    write_eigenvalues(eigenvalues, eigenD_path)
    write_eigenvectors(eigenvectors, eigenU_path)

    return eigenD_path, eigenU_path
