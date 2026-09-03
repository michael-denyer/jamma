"""Eigendecomposition file I/O — binary .npy default, GEMMA text legacy option.

Binary .npy is the default write format for performance at scale.
At 50k samples, .npy reads take ~3s vs ~4 min for text parsing.

GEMMA text format (.eigenD.txt / .eigenU.txt) remains available via
legacy_text=True for interoperability with external tools.

Read behaviour:
- .npy suffix: loads eagerly via np.load (full read into RAM).
- .txt suffix: checks for .npy sidecar cache (memory-mapped, demand-paged via
  mmap_mode='r') when available. Falls back to text parsing.

Format follows GEMMA param.cpp WriteVector/WriteMatrix:
- eigenD: one value per line, 10 significant digits (.10g format)
- eigenU: tab-separated rows, 10 significant digits per value
- No headers in either file
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io.matrix_reader import read_matrix_parallel
from jamma.io.matrix_writer import write_matrix_parallel
from jamma.utils.atomic_publish import atomic_output
from jamma.utils.npy_cache import (
    read_array_artifact,
    save_npy_atomic,
    write_npy_cache,
)

# ---------------------------------------------------------------------------
# .npy sidecar cache helpers (used for text-format files only)
# ---------------------------------------------------------------------------


def _npy_cache_path(txt_path: Path) -> Path:
    """Derive .npy sidecar path from a text file path.

    .eigenU.txt → .eigenU.npy, .eigenD.txt → .eigenD.npy
    """
    return txt_path.with_suffix(".npy")


def _checked_shape(data: np.ndarray, *, what: str, ndim: int, path: Path) -> np.ndarray:
    """Promote to ``ndim`` and reject anything that is not that shape.

    Eigenvalues must be a vector; eigenvectors a square matrix. ``np.loadtxt``
    returns a 0-D scalar for a one-line file, which ``atleast_*d`` repairs.
    """
    data = np.atleast_1d(data) if ndim == 1 else np.atleast_2d(data)
    ok = data.ndim == ndim and (ndim == 1 or data.shape[0] == data.shape[1])
    if not ok:
        expected = "a vector" if ndim == 1 else "a square matrix"
        raise ValueError(
            f"{what.capitalize()} file has wrong shape {data.shape}, "
            f"expected {expected}: {path}"
        )
    return data


def _read_array(
    path: Path,
    *,
    what: str,
    ndim: int,
    parse_text: Callable[[Path], np.ndarray],
) -> np.ndarray:
    """Read one eigen array from .npy, its .npy sidecar, or GEMMA text.

    Sidecar loads are read-only memory maps; see ``read_array_artifact``.
    """
    return read_array_artifact(
        path,
        what=what,
        parse_text=parse_text,
        check=lambda data, p: _checked_shape(data, what=what, ndim=ndim, path=p),
        mmap_mode="r",
    )


def _read_eigenvalues(path: Path) -> np.ndarray:
    """1-D float64 eigenvalues from .eigenD.npy or GEMMA .eigenD.txt.

    May be a read-only ``np.memmap`` when loaded from the .npy sidecar;
    callers must not mutate it in place.

    Raises:
        ValueError: If the file is empty, non-numeric, or not a vector.
    """
    return _read_array(
        path,
        what="eigenvalue",
        ndim=1,
        parse_text=lambda p: np.loadtxt(p, dtype=np.float64),
    )


def _read_eigenvectors(path: Path) -> np.ndarray:
    """2-D float64 eigenvectors from .eigenU.npy or GEMMA .eigenU.txt.

    Text parsing goes through ``read_matrix_parallel``. May be a read-only
    ``np.memmap`` when loaded from the .npy sidecar; callers must not mutate
    it in place.

    Raises:
        ValueError: If the file is empty, non-numeric, or not a square matrix.
    """
    return _read_array(
        path,
        what="eigenvector",
        ndim=2,
        parse_text=lambda p: read_matrix_parallel(p, delimiter=None),
    )


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
    eigenvalues = _read_eigenvalues(eigenD_path)
    eigenvectors = _read_eigenvectors(eigenU_path)

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


def _write_array(
    array: np.ndarray,
    path: Path,
    *,
    what: str,
    legacy_text: bool,
    save_text: Callable[[np.ndarray, Path], None],
) -> None:
    """Write one eigen array as .npy, or as GEMMA text plus a .npy sidecar.

    With ``legacy_text`` the array goes to ``path`` as-is (typically .txt)
    through ``save_text``, then to the sidecar for fast re-reads. Otherwise
    only ``path`` with its suffix swapped to .npy is written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_text:
        logger.info(f"Writing {what} to {path}")
        save_text(array, path)
        write_npy_cache(array, _npy_cache_path(path))
    else:
        npy_path = path.with_suffix(".npy")
        logger.info(f"Writing {what} to {npy_path}")
        save_npy_atomic(array, npy_path)


def _write_eigenvalues(
    eigenvalues: np.ndarray, path: Path, *, legacy_text: bool = False
) -> None:
    """One eigenvalue per line at 10 significant digits, GEMMA's precision(10)."""
    _write_array(
        eigenvalues,
        path,
        what="eigenvalues",
        legacy_text=legacy_text,
        save_text=_save_eigenvalues_text,
    )


def _save_eigenvalues_text(eigenvalues: np.ndarray, path: Path) -> None:
    """Write legacy eigenvalues without exposing a partial destination."""
    with atomic_output(path) as temporary:
        np.savetxt(temporary, eigenvalues, fmt="%.10g")


def _write_eigenvectors(
    eigenvectors: np.ndarray, path: Path, *, legacy_text: bool = False
) -> None:
    """Tab-separated rows at 10 significant digits, GEMMA's precision(10)."""
    _write_array(
        eigenvectors,
        path,
        what="eigenvectors",
        legacy_text=legacy_text,
        save_text=lambda a, p: write_matrix_parallel(a, p, fmt="%.10g", delimiter="\t"),
    )


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
    _write_eigenvalues(eigenvalues, eigenD_path, legacy_text=legacy_text)
    _write_eigenvectors(eigenvectors, eigenU_path, legacy_text=legacy_text)
    return eigenD_path, eigenU_path
