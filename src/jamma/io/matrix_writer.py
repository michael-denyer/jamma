"""Parallel matrix text writer for large matrices.

Provides write_matrix_parallel() which uses multiprocessing to format matrix
rows across CPU cores. At 100k x 100k (10B floats), np.savetxt is single-threaded
and takes ~30 minutes. Parallel formatting reduces this to ~2-4 minutes.

Output is byte-identical to np.savetxt for all matrix sizes.
"""

import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
from loguru import logger


def _format_rows_chunk(args: tuple) -> bytes:
    """Format a chunk of matrix rows as text bytes.

    Must be a top-level function for pickling with spawn context.

    Args:
        args: Tuple of (shm_name, start, end, ncols, fmt,
              delimiter, shape, dtype_str).

    Returns:
        Encoded bytes for the row chunk.
    """
    (shm_name, start, end, ncols, fmt, delimiter, shape, dtype_str) = args

    shm = None
    try:
        shm = SharedMemory(name=shm_name, create=False)
        matrix = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)

        row_fmt = delimiter.join([fmt] * ncols)
        lines = []
        for i in range(start, end):
            lines.append(row_fmt % tuple(matrix[i]))
        chunk_text = "\n".join(lines) + "\n"
        return chunk_text.encode("ascii")
    except Exception as e:
        raise type(e)(f"_format_rows_chunk failed on rows {start}-{end}: {e}") from e
    finally:
        if shm is not None:
            shm.close()


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

    # Ensure contiguous float64 for shared memory
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)

    rows_per_chunk = max(100, n_rows // n_workers)
    chunks_args = []

    ctx = mp.get_context("spawn")

    shm = None
    try:
        shm = SharedMemory(create=True, size=matrix.nbytes)
        shm_array = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
        np.copyto(shm_array, matrix)

        for start in range(0, n_rows, rows_per_chunk):
            end = min(start + rows_per_chunk, n_rows)
            chunks_args.append(
                (
                    shm.name,
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
                logger.error(f"Pool error writing {path}: {e}", exc_info=True)
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
        if shm is not None:
            shm.close()
            shm.unlink()
