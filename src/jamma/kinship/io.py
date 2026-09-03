"""Kinship matrix I/O in GEMMA format and binary .npy format.

Binary .npy is the default write format for performance at scale.
GEMMA text format (.cXX.txt) is available via legacy_text=True for
interoperability with external tools.

Read behaviour: .npy paths load directly. .txt paths check for a .npy sibling
(preferred if at least as new as the text file and not corrupt). Falls back to
text parsing and writes the sibling for the next read.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io.matrix_writer import write_matrix_parallel
from jamma.utils.npy_cache import read_array_artifact, save_npy_atomic

_SYMMETRY_RTOL = 1e-10


def _validate_kinship(K: np.ndarray, n_samples: int | None, source: str) -> None:
    """Validate kinship matrix shape and symmetry.

    Args:
        K: Loaded kinship matrix.
        n_samples: Expected number of samples (optional).
        source: Description of where K was loaded from (for error messages).

    Raises:
        ValueError: If matrix is not square, not symmetric, or dimension mismatch.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kinship matrix must be square, got shape {K.shape}")
    if n_samples is not None and K.shape[0] != n_samples:
        raise ValueError(
            f"Kinship matrix dimension {K.shape[0]} does not match "
            f"expected n_samples={n_samples}"
        )
    if not np.allclose(K, K.T, rtol=_SYMMETRY_RTOL):
        raise ValueError("Kinship matrix is not symmetric")


def read_kinship_matrix(path: Path, n_samples: int | None = None) -> np.ndarray:
    """Read kinship matrix from binary .npy or GEMMA .cXX.txt format.

    Auto-detects format based on path suffix and sibling files:
    - .npy suffix: loads directly via np.load.
    - .txt suffix: checks for .npy sibling at least as new; uses it if valid,
      otherwise falls back to text parsing and writes the sibling.

    The matrix is always loaded eagerly and writable: the pipeline applies
    individual weights to it in place.

    Args:
        path: Path to kinship matrix file (.cXX.npy or .cXX.txt).
        n_samples: Expected number of samples (optional validation).

    Returns:
        Kinship matrix as numpy array (n x n).

    Raises:
        ValueError: If matrix is not square, not symmetric, or dimension mismatch.
    """

    def parse_text(txt_path: Path) -> np.ndarray:
        if n_samples is not None and n_samples > 50_000:
            logger.warning(
                f"Reading {n_samples}x{n_samples} kinship matrix from text file. "
                f"This may be slow (~{n_samples**2 * 24 / 1e9:.0f}GB parse memory). "
                "Consider binary kinship format for large cohorts."
            )
        # Load matrix - handles tab and space separated
        return np.loadtxt(txt_path, dtype=np.float64)

    def check(K: np.ndarray, source: Path) -> np.ndarray:
        _validate_kinship(K, n_samples, str(source))
        return K

    return read_array_artifact(
        path, what="kinship matrix", parse_text=parse_text, check=check
    )


def write_kinship_matrix(
    K: np.ndarray, path: Path, *, legacy_text: bool = False
) -> Path:
    """Write kinship matrix in binary .npy format (default) or GEMMA .cXX.txt.

    Binary format is the default for performance. At 50k+ samples, .npy I/O
    takes seconds vs minutes for text. Use legacy_text=True for GEMMA
    interoperability.

    GEMMA text format specifications (from legacy/src/param.cpp:1886-1911):
    - outfile.precision(10): 10 significant digits using general format
    - Tab separator between values
    - Newline after each row
    - No header row
    - No sample IDs in matrix file

    Args:
        K: Kinship matrix (n x n), should be symmetric.
        path: Output file path. When legacy_text=False (default), writes to
            path.with_suffix(".npy"). When legacy_text=True, writes to path
            (typically .cXX.txt).
        legacy_text: If True, write GEMMA-compatible text format. Default False
            writes binary .npy.

    Returns:
        Path to the file actually written (.npy or .txt).

    Raises:
        ValueError: If K is not C-contiguous when writing binary (avoids
            memory doubling from copy at 100k samples).

    Example:
        >>> write_kinship_matrix(K, Path("output/result.cXX.txt"))  # writes .npy
        >>> write_kinship_matrix(K, Path("output/result.cXX.txt"), legacy_text=True)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if legacy_text:
        write_matrix_parallel(K, path, fmt="%.10g", delimiter="\t")
        return path
    else:
        if not K.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "Kinship matrix must be C-contiguous before binary save to avoid "
                "memory doubling at large sizes. Use np.ascontiguousarray(K) first."
            )
        npy_path = path.with_suffix(".npy")
        save_npy_atomic(K, npy_path)
        return npy_path


def write_loco_kinship_matrices(
    loco_kinships: Iterable[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str = "result",
    *,
    legacy_text: bool = False,
) -> list[Path]:
    """Write per-chromosome LOCO kinship matrices to disk.

    For each (chr_name, K) pair yielded by the iterator, writes the matrix
    to the output directory. By default writes binary .npy; use legacy_text=True
    for GEMMA text format.

    Binary naming: ``{prefix}.loco.cXX.chr{chr_name}.npy``
    Text naming: ``{prefix}.loco.cXX.chr{chr_name}.txt``

    This is a convenience wrapper for the ``gk -loco`` standalone command.

    Args:
        loco_kinships: Iterable of (chromosome_name, kinship_matrix) pairs,
            consumed once. Typically produced by
            ``compute_loco_kinship_streaming()``.
        output_dir: Directory for output files (created if needed).
        prefix: Filename prefix (default "result").
        legacy_text: If True, write GEMMA text format. Default False writes
            binary .npy.

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
        base_path = output_dir / f"{prefix}.loco.cXX.chr{chr_name}.txt"
        if not legacy_text:
            K = np.ascontiguousarray(K)
        kinship_path = write_kinship_matrix(K, base_path, legacy_text=legacy_text)
        written.append(kinship_path)

    return written
