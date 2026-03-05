"""Eigendecomposition file I/O — binary .npy default, GEMMA text legacy option.

Binary .npy is the default write format for performance at scale.
At 50k samples, .npy reads take ~3s vs ~4 min for text parsing.

GEMMA text format (.eigenD.txt / .eigenU.txt) remains available via
legacy_text=True for interoperability with external tools.

Read behaviour:
- .npy suffix: loads directly via np.load.
- .txt suffix: checks for .npy sidecar cache (created by write_* or by first
  text parse) when available. Falls back to text parsing.

Format follows GEMMA param.cpp WriteVector/WriteMatrix:
- eigenD: one value per line, 10 significant digits (.10g format)
- eigenU: tab-separated rows, 10 significant digits per value
- No headers in either file
"""

from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io.matrix_reader import read_matrix_parallel
from jamma.io.matrix_writer import write_matrix_parallel
from jamma.utils.npy_cache import npy_cache_valid

# ---------------------------------------------------------------------------
# .npy sidecar cache helpers (used for text-format files only)
# ---------------------------------------------------------------------------


def _npy_cache_path(txt_path: Path) -> Path:
    """Derive .npy sidecar path from a text file path.

    .eigenU.txt → .eigenU.npy, .eigenD.txt → .eigenD.npy
    """
    return txt_path.with_suffix(".npy")


def _write_npy_cache(array: np.ndarray, npy_path: Path) -> None:
    """Write .npy cache, swallowing errors (read-only FS, full disk, etc.)."""
    try:
        np.save(npy_path, array)
    except OSError as e:
        logger.warning(f"Could not write .npy cache {npy_path}: {e}")


def _load_npy_cache(npy_path: Path) -> np.ndarray | None:
    """Load .npy cache, returning None on corruption or error."""
    try:
        return np.load(npy_path)
    except (OSError, ValueError) as e:
        logger.warning(f"Corrupt .npy cache {npy_path}, will re-parse text: {e}")
        try:
            npy_path.unlink()
        except OSError as unlink_err:
            logger.warning(f"Could not remove corrupt cache {npy_path}: {unlink_err}")
        return None


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_eigenvalues(path: Path) -> np.ndarray:
    """Read eigenvalues from a .npy or GEMMA .eigenD.txt file.

    When path suffix is .npy, loads directly via np.load.
    When path suffix is .txt, uses .npy sidecar cache when available,
    otherwise parses text. For text format, the file is small (one value
    per line), so np.loadtxt suffices without parallel parsing.

    Args:
        path: Path to eigenvalue file (.eigenD.npy or .eigenD.txt).

    Returns:
        1-D float64 array of eigenvalues, shape (n_samples,).

    Raises:
        ValueError: If file is empty, contains non-numeric data, or wrong shape.
    """
    path = Path(path)

    # Direct .npy load
    if path.suffix == ".npy":
        logger.info(f"Reading eigenvalues from {path}")
        data = np.load(path)
        data = np.atleast_1d(data)
        if data.ndim != 1:
            raise ValueError(
                f"Eigenvalue .npy file has wrong shape {data.shape}: {path}"
            )
        return data

    # Text path: try sidecar cache
    npy_path = _npy_cache_path(path)

    if npy_cache_valid(path, npy_path):
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
    """Read eigenvectors from a .npy or GEMMA .eigenU.txt file.

    When path suffix is .npy, loads directly via np.load.
    When path suffix is .txt, uses .npy sidecar cache when available.
    Falls back to parallel text parsing for large matrices (>= 500 rows),
    or np.loadtxt for small ones.

    Args:
        path: Path to eigenvector file (.eigenU.npy or .eigenU.txt).

    Returns:
        2-D float64 array of eigenvectors, shape (n_samples, n_samples).

    Raises:
        ValueError: If file is empty, non-numeric, or not a square matrix.
    """
    path = Path(path)

    # Direct .npy load
    if path.suffix == ".npy":
        logger.info(f"Reading eigenvectors from {path}")
        data = np.load(path)
        data = np.atleast_2d(data)
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(
                f"Eigenvector .npy file has wrong shape {data.shape}: {path}"
            )
        return data

    # Text path: try sidecar cache
    npy_path = _npy_cache_path(path)

    if npy_cache_valid(path, npy_path):
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

    Handles both .npy paths (binary, default) and .txt paths (legacy text).
    Validates that eigenvalue count matches eigenvector dimensions.
    Optionally validates against expected sample count.

    Args:
        eigenD_path: Path to eigenvalue file (.eigenD.npy or .eigenD.txt).
        eigenU_path: Path to eigenvector file (.eigenU.npy or .eigenU.txt).
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


def write_eigenvalues(
    eigenvalues: np.ndarray, path: Path, *, legacy_text: bool = False
) -> None:
    """Write eigenvalues in binary .npy format (default) or GEMMA .eigenD.txt.

    Binary format is the default for performance. Use legacy_text=True for
    GEMMA-compatible output.

    GEMMA text format: one eigenvalue per line, 10 significant digits,
    matching GEMMA's precision(10) output.

    When legacy_text=True, also writes a .npy sidecar for fast subsequent reads.

    Args:
        eigenvalues: 1D array of eigenvalues.
        path: Output file path. When legacy_text=False (default), path is used
            as base and .npy suffix is derived. When legacy_text=True, writes
            to path as-is (typically .eigenD.txt) plus .npy sidecar.
        legacy_text: If True, write GEMMA text format + sidecar. Default False
            writes only binary .npy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if legacy_text:
        logger.info(f"Writing eigenvalues to {path}")
        np.savetxt(path, eigenvalues, fmt="%.10g")
        _write_npy_cache(eigenvalues, _npy_cache_path(path))
    else:
        npy_path = path.with_suffix(".npy")
        logger.info(f"Writing eigenvalues to {npy_path}")
        np.save(npy_path, eigenvalues)


def write_eigenvectors(
    eigenvectors: np.ndarray, path: Path, *, legacy_text: bool = False
) -> None:
    """Write eigenvectors in binary .npy format (default) or GEMMA .eigenU.txt.

    Binary format is the default for performance. Use legacy_text=True for
    GEMMA-compatible output.

    GEMMA text format: tab-separated rows, 10 significant digits per value,
    matching GEMMA's precision(10) output.

    When legacy_text=True, also writes a .npy sidecar for fast subsequent reads.

    Args:
        eigenvectors: 2D array of eigenvectors (n_samples, n_samples).
        path: Output file path. When legacy_text=False (default), path is used
            as base and .npy suffix is derived. When legacy_text=True, writes
            to path as-is (typically .eigenU.txt) plus .npy sidecar.
        legacy_text: If True, write GEMMA text format + sidecar. Default False
            writes only binary .npy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if legacy_text:
        write_matrix_parallel(eigenvectors, path, fmt="%.10g", delimiter="\t")
        _write_npy_cache(eigenvectors, _npy_cache_path(path))
    else:
        npy_path = path.with_suffix(".npy")
        logger.info(f"Writing eigenvectors to {npy_path}")
        np.save(npy_path, eigenvectors)


def write_eigen_files(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    output_dir: Path,
    prefix: str = "result",
    *,
    legacy_text: bool = False,
) -> tuple[Path, Path]:
    """Write both eigenvalue and eigenvector files.

    Binary .npy is the default format (no text files written). Use
    legacy_text=True for GEMMA-compatible .txt + .npy sidecar output.

    Binary naming: {prefix}.eigenD.npy and {prefix}.eigenU.npy
    Text naming: {prefix}.eigenD.txt and {prefix}.eigenU.txt (+ sidecars)

    Args:
        eigenvalues: 1D array of eigenvalues.
        eigenvectors: 2D array of eigenvectors.
        output_dir: Directory for output files.
        prefix: Filename prefix (default "result").
        legacy_text: If True, write GEMMA text format + sidecars. Default False
            writes only binary .npy.

    Returns:
        Tuple of (eigenD_path, eigenU_path). Paths reflect actual files written
        (.npy by default, .txt with legacy_text=True).
    """
    output_dir = Path(output_dir)
    suffix = ".txt" if legacy_text else ".npy"
    eigenD_path = output_dir / f"{prefix}.eigenD{suffix}"
    eigenU_path = output_dir / f"{prefix}.eigenU{suffix}"
    write_eigenvalues(eigenvalues, eigenD_path, legacy_text=legacy_text)
    write_eigenvectors(eigenvectors, eigenU_path, legacy_text=legacy_text)
    return eigenD_path, eigenU_path
