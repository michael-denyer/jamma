"""Eigendecomposition file I/O in GEMMA format.

Read and write eigenvalue (.eigenD.txt) and eigenvector (.eigenU.txt) files
in GEMMA-compatible format. Used for eigendecomposition reuse across
multi-phenotype workflows.

Format follows GEMMA param.cpp WriteVector/WriteMatrix:
- eigenD: one value per line, 10 significant digits (.10g format)
- eigenU: tab-separated rows, 10 significant digits per value
- No headers in either file

A .npy binary sidecar cache is written alongside text files after each
read or write. Subsequent reads use np.load() (~3s at 50k) instead of
text parsing (~4 min). The cache is invalidated when the text file is
newer. Cache writes are best-effort — failures (read-only FS, full disk)
are logged and silently ignored.
"""

from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io.matrix_reader import read_matrix_parallel
from jamma.io.matrix_writer import write_matrix_parallel

# ---------------------------------------------------------------------------
# .npy sidecar cache helpers
# ---------------------------------------------------------------------------


def _npy_cache_path(txt_path: Path) -> Path:
    """Derive .npy sidecar path from a text file path.

    .eigenU.txt → .eigenU.npy, .eigenD.txt → .eigenD.npy
    """
    return txt_path.with_suffix(".npy")


def _npy_cache_valid(txt_path: Path, npy_path: Path) -> bool:
    """Check if .npy cache exists and is at least as new as the text file."""
    if not npy_path.exists():
        return False
    try:
        txt_mtime = txt_path.stat().st_mtime
        npy_stat = npy_path.stat()
        return npy_stat.st_mtime >= txt_mtime and npy_stat.st_size > 0
    except OSError:
        return False


def _write_npy_cache(array: np.ndarray, npy_path: Path) -> None:
    """Write .npy cache, swallowing errors (read-only FS, full disk, etc.)."""
    try:
        np.save(npy_path, array)
    except OSError as e:
        logger.debug(f"Could not write .npy cache {npy_path}: {e}")


def _load_npy_cache(npy_path: Path) -> np.ndarray | None:
    """Load .npy cache, returning None on corruption or error."""
    try:
        return np.load(npy_path)
    except Exception as e:
        logger.warning(f"Corrupt .npy cache {npy_path}, will re-parse text: {e}")
        try:
            npy_path.unlink()
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_eigenvalues(path: Path) -> np.ndarray:
    """Read eigenvalues from a GEMMA .eigenD.txt file.

    Uses .npy sidecar cache when available. The eigenvalue file is small
    (one value per line), so no parallel parsing is needed.

    Args:
        path: Path to eigenvalue file (one value per line).

    Returns:
        1-D float64 array of eigenvalues, shape (n_samples,).

    Raises:
        ValueError: If file is empty or contains non-numeric data.
    """
    path = Path(path)
    npy_path = _npy_cache_path(path)

    if _npy_cache_valid(path, npy_path):
        data = _load_npy_cache(npy_path)
        if data is not None:
            logger.info(f"Reading eigenvalues from cache {npy_path}")
            data = np.atleast_1d(data)
            if data.ndim != 1:
                raise ValueError(
                    f"Cached eigenvalue file has wrong shape {data.shape}: {npy_path}"
                )
            return data

    logger.info(f"Reading eigenvalues from {path}")
    try:
        data = np.loadtxt(path, dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"Cannot parse eigenvalue file {path}: {e}") from e

    if data.size == 0:
        raise ValueError(f"Eigenvalue file is empty: {path}")

    # np.loadtxt returns 0-D scalar for single-line files
    data = np.atleast_1d(data)

    if data.ndim != 1:
        raise ValueError(
            f"Eigenvalue file must be single-column, got shape {data.shape}: {path}"
        )

    _write_npy_cache(data, npy_path)
    return data


def read_eigenvectors(path: Path) -> np.ndarray:
    """Read eigenvectors from a GEMMA .eigenU.txt file.

    Uses .npy sidecar cache when available. Falls back to parallel text
    parsing for large matrices (>= 500 rows), or np.loadtxt for small ones.

    Args:
        path: Path to eigenvector file (tab-separated matrix).

    Returns:
        2-D float64 array of eigenvectors, shape (n_samples, n_samples).

    Raises:
        ValueError: If file is empty, non-numeric, or not a square matrix.
    """
    path = Path(path)
    npy_path = _npy_cache_path(path)

    if _npy_cache_valid(path, npy_path):
        data = _load_npy_cache(npy_path)
        if data is not None:
            logger.info(f"Reading eigenvectors from cache {npy_path}")
            data = np.atleast_2d(data)
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError(
                    f"Cached eigenvector file has wrong shape {data.shape}: {npy_path}"
                )
            return data

    logger.info(f"Reading eigenvectors from {path}")
    try:
        data = read_matrix_parallel(path, delimiter=None)
    except ValueError as e:
        raise ValueError(f"Cannot parse eigenvector file {path}: {e}") from e

    if data.size == 0:
        raise ValueError(f"Eigenvector file is empty: {path}")

    data = np.atleast_2d(data)

    if data.ndim != 2:
        raise ValueError(
            f"Eigenvector file must be a 2D matrix, got {data.ndim}D array: {path}"
        )

    if data.shape[0] != data.shape[1]:
        raise ValueError(
            f"Eigenvector matrix must be square, got shape {data.shape}: {path}"
        )

    _write_npy_cache(data, npy_path)
    return data


def read_eigen_files(
    eigenD_path: Path,
    eigenU_path: Path,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read both eigenvalue and eigenvector files with cross-validation.

    Validates that eigenvalue count matches eigenvector dimensions.
    Optionally validates against expected sample count.

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
    n_rows = eigenvectors.shape[0]

    if n_eval != n_rows:
        raise ValueError(
            f"Eigenvalue count ({n_eval}) does not match eigenvector "
            f"dimensions ({n_rows} x {n_rows}). Files may be mismatched: "
            f"{eigenD_path}, {eigenU_path}"
        )

    if n_samples is not None and n_eval != n_samples:
        raise ValueError(
            f"Eigen files have {n_eval} samples but pipeline expects "
            f"{n_samples} after phenotype/covariate filtering. "
            f"Re-run with -eigen to regenerate eigen files matching "
            f"the current filtering."
        )

    n_negative = int(np.sum(eigenvalues < 0))
    if n_negative > 0:
        raise ValueError(
            f"Eigenvalue file contains {n_negative} negative value(s). "
            f"Kinship eigenvalues must be non-negative. "
            f"File: {eigenD_path}"
        )

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_eigenvalues(eigenvalues: np.ndarray, path: Path) -> None:
    """Write eigenvalues in GEMMA .eigenD.txt format.

    Writes one eigenvalue per line using 10 significant digits,
    matching GEMMA's precision(10) output. Also writes .npy sidecar
    for fast subsequent reads.

    Args:
        eigenvalues: 1D array of eigenvalues.
        path: Output file path (typically .eigenD.txt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing eigenvalues to {path}")
    np.savetxt(path, eigenvalues, fmt="%.10g")
    _write_npy_cache(eigenvalues, _npy_cache_path(path))


def write_eigenvectors(eigenvectors: np.ndarray, path: Path) -> None:
    """Write eigenvectors in GEMMA .eigenU.txt format.

    Writes tab-separated rows using 10 significant digits per value,
    matching GEMMA's precision(10) output. Also writes .npy sidecar
    for fast subsequent reads.

    Args:
        eigenvectors: 2D array of eigenvectors (n_samples, n_samples).
        path: Output file path (typically .eigenU.txt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_matrix_parallel(eigenvectors, path, fmt="%.10g", delimiter="\t")
    _write_npy_cache(eigenvectors, _npy_cache_path(path))


def write_eigen_files(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    output_dir: Path,
    prefix: str = "result",
) -> tuple[Path, Path]:
    """Write both eigenvalue and eigenvector files.

    Convenience wrapper that writes {prefix}.eigenD.txt and
    {prefix}.eigenU.txt to the specified output directory, plus
    .npy sidecars for fast reads.

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
