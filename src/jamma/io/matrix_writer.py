"""Parallel matrix text writer for large matrices.

Provides write_matrix_parallel() which uses multiprocessing to format matrix
rows across CPU cores. At 100k x 100k (10B floats), np.savetxt is single-threaded
and takes ~30 minutes. Parallel formatting reduces this to ~2-4 minutes.

Uses file-backed numpy.memmap for worker IPC instead of shared memory to avoid
SIGBUS crashes when Docker's /dev/shm is capped at 64 MB (cpython#114390).

Workers write formatted text to per-chunk temp files instead of returning bytes
through the IPC pipe, keeping memory usage bounded regardless of worker count.

Output is byte-identical to np.savetxt for all matrix sizes.
"""

import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger


def _format_rows_to_file(args: tuple) -> str:
    """Format a chunk of matrix rows and write to a temp file.

    Must be a top-level function for pickling with spawn context.

    Args:
        args: Tuple of (memmap_path, out_path, start, end, ncols, fmt,
              delimiter, shape, dtype_str).

    Returns:
        Path to the output chunk file (same as out_path).
    """
    (memmap_path, out_path, start, end, ncols, fmt, delimiter, shape, dtype_str) = args

    try:
        matrix = np.memmap(
            memmap_path, dtype=np.dtype(dtype_str), mode="r", shape=shape
        )
        row_fmt = delimiter.join([fmt] * ncols)
        with open(out_path, "wb") as f:
            for i in range(start, end):
                line = (row_fmt % tuple(matrix[i])) + "\n"
                f.write(line.encode("ascii"))
        return out_path
    except Exception as e:
        raise RuntimeError(
            f"_format_rows_to_file failed on rows {start}-{end}: {e}"
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
        n_workers: Number of worker processes (default: cpu_count).
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
        n_workers = os.cpu_count() or 1
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    logger.info(
        f"Writing {n_rows}x{n_cols} matrix to {path.resolve()} ({n_workers} workers)"
    )

    # Ensure contiguous float64 for memmap compatibility
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)

    rows_per_chunk = max(100, n_rows // n_workers)
    ctx = mp.get_context("spawn")

    # Create temp dir for memmap + per-chunk output files
    tmp_dir = tempfile.mkdtemp(prefix="jamma_mwrite_")
    memmap_path = os.path.join(tmp_dir, "matrix.dat")
    chunk_paths: list[str] = []

    try:
        try:
            matrix.tofile(memmap_path)
        except OSError as e:
            raise OSError(
                f"Failed to write {matrix.nbytes / (1024**3):.1f} GB temp file "
                f"to {memmap_path} for parallel matrix IPC. "
                f"Set TMPDIR to a filesystem with sufficient space."
            ) from e

        # Build chunk args — each worker writes to its own temp file
        chunks_args = []
        for idx, start in enumerate(range(0, n_rows, rows_per_chunk)):
            chunk_out = os.path.join(tmp_dir, f"chunk_{idx:06d}.txt")
            chunk_paths.append(chunk_out)
            chunks_args.append(
                (
                    memmap_path,
                    chunk_out,
                    start,
                    min(start + rows_per_chunk, n_rows),
                    n_cols,
                    fmt,
                    delimiter,
                    matrix.shape,
                    str(matrix.dtype),
                )
            )

        with ctx.Pool(processes=n_workers) as pool:
            try:
                # imap preserves order; workers write to disk, return only path
                for _ in pool.imap(_format_rows_to_file, chunks_args):
                    pass
            except BaseException as e:
                logger.opt(exception=e).error(f"Pool error writing {path}: {e}")
                pool.terminate()
                pool.join()
                raise

        # Concatenate chunk files in order
        try:
            with open(path, "wb") as f_out:
                for chunk_path in chunk_paths:
                    with open(chunk_path, "rb") as f_in:
                        while True:
                            buf = f_in.read(8 * 1024 * 1024)  # 8 MB reads
                            if not buf:
                                break
                            f_out.write(buf)
        except BaseException:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                logger.warning(f"Failed to delete partial output {path}: {cleanup_err}")
            raise
    finally:
        # Clean up all temp files
        for p in [memmap_path, *chunk_paths]:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError as e:
            logger.warning(f"Failed to remove temp dir {tmp_dir}: {e}")
