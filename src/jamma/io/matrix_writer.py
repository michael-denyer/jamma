"""Parallel matrix text writer for large matrices.

Provides write_matrix_parallel() which uses multiprocessing to format matrix
rows across CPU cores. At 100k x 100k (10B floats), np.savetxt is single-threaded
and takes ~30 minutes. Parallel formatting reduces this to ~2-4 minutes.

Uses file-backed numpy.memmap for worker IPC instead of shared memory to avoid
SIGBUS crashes when Docker's /dev/shm is capped at 64 MB (cpython#114390).

Output is byte-identical to np.savetxt for all matrix sizes.
"""

import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger


def _format_rows_chunk(args: tuple) -> bytes:
    """Format a chunk of matrix rows as text bytes.

    Must be a top-level function for pickling with spawn context.

    Args:
        args: Tuple of (file_path, start, end, ncols, fmt,
              delimiter, shape, dtype_str).

    Returns:
        Encoded bytes for the row chunk.
    """
    (file_path, start, end, ncols, fmt, delimiter, shape, dtype_str) = args

    try:
        matrix = np.memmap(file_path, dtype=np.dtype(dtype_str), mode="r", shape=shape)

        row_fmt = delimiter.join([fmt] * ncols)
        lines = []
        for i in range(start, end):
            lines.append(row_fmt % tuple(matrix[i]))
        chunk_text = "\n".join(lines) + "\n"
        return chunk_text.encode("ascii")
    except Exception as e:
        raise RuntimeError(
            f"_format_rows_chunk failed on rows {start}-{end}: {e}"
        ) from e


def write_matrix_parallel(
    matrix: np.ndarray,
    path: Path,
    fmt: str = "%.10g",
    delimiter: str = "\t",
    n_workers: int | None = None,
    min_rows_for_parallel: int = 500,
) -> None:
    """Write a 2D matrix to a text file, optionally using parallel formatting.

    For matrices with fewer than min_rows_for_parallel rows, falls back to
    np.savetxt. For larger matrices, distributes row formatting across
    multiple processes for significant speedup.

    Output is byte-identical to np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter).

    Args:
        matrix: 2D numpy array to write.
        path: Output file path.
        fmt: Format string for each element (default "%.10g").
        delimiter: Column separator (default tab).
        n_workers: Number of worker processes (default: min(cpu_count, 16)).
        min_rows_for_parallel: Row threshold for parallel path (default 500).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = matrix.shape

    if n_rows < min_rows_for_parallel:
        logger.info(f"Writing {n_rows}x{n_cols} matrix to {path.resolve()}")
        np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter)
        return

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 16)
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    logger.info(
        f"Writing {n_rows}x{n_cols} matrix to {path.resolve()} ({n_workers} workers)"
    )

    # Ensure contiguous float64 for memmap compatibility
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)

    rows_per_chunk = max(100, n_rows // n_workers)
    chunks_args = []

    ctx = mp.get_context("spawn")

    fd, tmp_path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)  # Avoid fd leak -- tofile opens by path
    try:
        try:
            matrix.tofile(tmp_path)
        except OSError as e:
            raise OSError(
                f"Failed to write {matrix.nbytes / (1024**3):.1f} GB temp file "
                f"to {tmp_path} for parallel matrix IPC. "
                f"Set TMPDIR to a filesystem with sufficient space."
            ) from e

        for start in range(0, n_rows, rows_per_chunk):
            end = min(start + rows_per_chunk, n_rows)
            chunks_args.append(
                (
                    tmp_path,
                    start,
                    end,
                    n_cols,
                    fmt,
                    delimiter,
                    matrix.shape,
                    str(matrix.dtype),
                )
            )

        with ctx.Pool(processes=n_workers) as pool:
            try:
                with open(path, "wb") as f:
                    for chunk_bytes in pool.imap(_format_rows_chunk, chunks_args):
                        f.write(chunk_bytes)
            except BaseException as e:
                logger.opt(exception=e).error(f"Pool error writing {path}: {e}")
                pool.terminate()
                pool.join()
                # Best-effort cleanup of partial file
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_err:
                    logger.warning(
                        f"Failed to delete partial output {path}: {cleanup_err}"
                    )
                raise
    finally:
        try:
            os.unlink(tmp_path)
        except OSError as e:
            logger.warning(f"Failed to remove temp memmap file {tmp_path}: {e}")
