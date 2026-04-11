"""Parallel matrix text writer for large matrices.

Provides write_matrix_parallel() which uses multiprocessing to format matrix
rows across CPU cores. At 100k x 100k (10B floats), np.savetxt is single-threaded
and takes ~30 minutes. Parallel formatting reduces this to ~2-4 minutes.

Uses file-backed numpy.memmap for worker IPC instead of shared memory to avoid
SIGBUS crashes when Docker's /dev/shm is capped at 64 MB (cpython#114390).

Workers write formatted text to per-chunk temp files, returning only the file
path through the IPC pipe. This keeps memory usage bounded regardless of worker
count.

Temp files are created adjacent to the output file (same filesystem) to avoid
filling /tmp on systems where it's a small tmpfs (e.g. Databricks). Chunks are
deleted eagerly during concatenation to minimize peak disk usage.

Output is byte-identical to np.savetxt for all matrix sizes.
"""

import multiprocessing as mp
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

# Cap worker count — beyond this, disk I/O is the bottleneck, not CPU
# formatting. Excess workers just add process overhead and memory pressure.
_MAX_WRITERS = 32


def _format_rows_to_file(args: tuple) -> None:
    """Format a chunk of matrix rows and write to a temp file.

    Must be a top-level function for pickling with spawn context.

    Args:
        args: Tuple of (memmap_path, out_path, start, end, ncols, fmt,
              delimiter, shape, dtype_str).
    """
    (memmap_path, out_path, start, end, ncols, fmt, delimiter, shape, dtype_str) = args

    try:
        matrix = np.memmap(
            memmap_path, dtype=np.dtype(dtype_str), mode="r", shape=shape
        )
        delimiter_bytes = delimiter.encode("ascii")
        newline = b"\n"
        with open(out_path, "wb") as f:
            for i in range(start, end):
                row = matrix[i]
                # Format each value individually and join — avoids creating a
                # Python tuple of 125k float objects (~3 MB per row) that the
                # old `row_fmt % tuple(row)` approach required.
                parts = [(fmt % row[j]).encode("ascii") for j in range(len(row))]
                f.write(delimiter_bytes.join(parts) + newline)
    except Exception as e:
        raise RuntimeError(
            f"_format_rows_to_file failed on rows {start}-{end}: {e}"
        ) from e


def _estimate_text_size(n_rows: int, n_cols: int) -> int:
    """Estimate text file size for a matrix written with %.10g format.

    Each float averages ~12 chars, plus one delimiter per column and a newline.
    """
    bytes_per_row = n_cols * 12 + n_cols  # values + delimiters/newline
    return n_rows * bytes_per_row


def _create_temp_dir(output_path: Path) -> str:
    """Create temp dir on the same filesystem as the output file.

    Tries the output's parent directory first, falls back to the system
    default (TMPDIR / /tmp) if that fails.
    """
    output_dir = output_path.parent
    try:
        return tempfile.mkdtemp(prefix=".jamma_mwrite_", dir=output_dir)
    except OSError:
        logger.warning(
            f"Cannot create temp dir in {output_dir}, falling back to system tmpdir"
        )
        return tempfile.mkdtemp(prefix="jamma_mwrite_")


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

    Temp files (memmap + chunks) are placed on the same filesystem as the
    output to avoid filling /tmp. Chunks are deleted eagerly during
    concatenation to minimize peak disk usage.

    Output is byte-identical to np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter).

    Args:
        matrix: 2D numpy array to write.
        path: Output file path.
        fmt: Format string for each element (default "%.10g").
        delimiter: Column separator (default tab).
        n_workers: Number of worker processes (default: min(cpu_count, 32)).
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
        n_workers = min(os.cpu_count() or 1, _MAX_WRITERS)
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    logger.info(
        f"Writing {n_rows}x{n_cols} matrix to {path.resolve()} ({n_workers} workers)"
    )

    # Ensure contiguous float64 for memmap compatibility
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)

    rows_per_chunk = max(100, n_rows // n_workers)

    # Pre-flight disk space check (warn only — unreliable on network FS)
    # Peak during worker phase: memmap + ALL chunks (workers run concurrently)
    memmap_bytes = matrix.nbytes
    text_bytes = _estimate_text_size(n_rows, n_cols)
    peak_bytes = memmap_bytes + text_bytes  # memmap + all chunks ≈ full output
    try:
        usage = shutil.disk_usage(path.parent)
        if usage.free < peak_bytes:
            logger.warning(
                f"Low disk space: {usage.free / (1024**3):.1f} GB free, "
                f"estimated peak ~{peak_bytes / (1024**3):.0f} GB needed "
                f"(memmap {memmap_bytes / (1024**3):.0f} GB + "
                f"chunks {text_bytes / (1024**3):.0f} GB). "
                f"Write may fail with ENOSPC."
            )
    except OSError as e:
        logger.warning(
            f"Could not check disk space for {path.parent}: {e}. Skipping space check."
        )

    ctx = mp.get_context("spawn")

    # Create temp dir on same filesystem as output (avoids filling /tmp)
    tmp_dir = _create_temp_dir(path)
    tmp_dir_p = Path(tmp_dir)
    memmap_path = str(tmp_dir_p / "matrix.dat")
    chunk_paths: list[str] = []

    try:
        try:
            matrix.tofile(memmap_path)
        except OSError as e:
            raise OSError(
                f"Failed to write {matrix.nbytes / (1024**3):.1f} GB temp file "
                f"to {memmap_path} for parallel matrix IPC. "
                f"Ensure the output directory has sufficient disk space."
            ) from e

        # Build chunk args — each worker writes to its own temp file
        chunks_args = []
        for idx, start in enumerate(range(0, n_rows, rows_per_chunk)):
            chunk_out = str(tmp_dir_p / f"chunk_{idx:06d}.txt")
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
                # imap (not imap_unordered) preserves chunk order for concatenation
                for _ in pool.imap(_format_rows_to_file, chunks_args):
                    pass
            # BaseException (not Exception) to ensure pool cleanup even on
            # KeyboardInterrupt or SystemExit — orphaned workers would leak.
            except BaseException as e:
                logger.opt(exception=e).error(f"Pool error writing {path}: {e}")
                pool.terminate()
                pool.join()
                raise

        # Free memmap before concatenation — at 125k samples this is 126 GB
        try:
            Path(memmap_path).unlink()
        except OSError as e:
            logger.warning(f"Could not delete memmap {memmap_path}: {e}")
        memmap_path = None  # prevent double-delete in finally

        # Concatenate chunk files in order, deleting each after use
        try:
            with open(path, "wb") as f_out:
                for chunk_path in chunk_paths:
                    with open(chunk_path, "rb") as f_in:
                        while True:
                            buf = f_in.read(8 * 1024 * 1024)  # 8 MB reads
                            if not buf:
                                break
                            f_out.write(buf)
                    # Eagerly delete — frees disk before writing the next chunk
                    try:
                        Path(chunk_path).unlink()
                    except OSError as e:
                        logger.debug(
                            f"Could not eagerly delete chunk {chunk_path}: {e}"
                        )
        except BaseException as e:
            logger.opt(exception=e).error(
                f"Failed during chunk concatenation to {path}: {e}"
            )
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                logger.warning(f"Failed to delete partial output {path}: {cleanup_err}")
            raise
    finally:
        # Clean up any remaining temp files (error paths)
        if memmap_path is not None:
            try:
                Path(memmap_path).unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning(f"Failed to clean up temp memmap {memmap_path}: {e}")
        for p in chunk_paths:
            try:
                Path(p).unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning(f"Failed to clean up chunk file {p}: {e}")
        try:
            tmp_dir_p.rmdir()
        except OSError as e:
            logger.debug(f"Could not remove temp dir {tmp_dir}: {e}")
