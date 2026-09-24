"""Atomic matrix text output with bounded parallel formatting.

Default %.10g/tab output uses a native C++ formatter and ordered Python
threads. Custom formats and installations without the extension use the
process writer below, with file-backed IPC and temporary chunks beside output.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io._native_matrix_writer import native_formatter, write_native_matrix
from jamma.io._parallel_text import (
    MemmapRef,
    default_worker_count,
    run_spawn_pool,
    temp_dir_beside,
)
from jamma.utils.atomic_publish import AtomicOutput, unlink_quietly

# Values formatted per `%` call inside a worker. Bounds the tuple of NumPy
# scalars a wide row creates; 4096 of them is about 130 KB of scalar objects.
_FORMAT_SLICE = 4096


@dataclass(frozen=True, slots=True)
class MatrixWriteTask:
    """Picklable message passed to one matrix-formatting worker."""

    matrix: MemmapRef
    output_path: str
    start_row: int
    stop_row: int
    fmt: str
    delimiter: str


def _format_rows_to_file(task: MatrixWriteTask) -> None:
    """Format a chunk of matrix rows and write to a temp file.

    Must be a top-level function for pickling with spawn context.
    """
    try:
        matrix = task.matrix.open("r")
        # np.memmap.__getitem__ is a Python method, so iterating a memmap slice
        # pays a Python call per value; a plain ndarray view iterates in C.
        rows = np.asarray(matrix)
        n_cols = task.matrix.shape[1]
        slices = [
            (start, min(_FORMAT_SLICE, n_cols - start))
            for start in range(0, n_cols, _FORMAT_SLICE)
        ]
        # One "%g\t%g\t..." template per distinct slice width, built once per
        # task. Inside a template the delimiter is format text, so its percent
        # signs are escaped; the join between slices below inserts it literally.
        template_delimiter = task.delimiter.replace("%", "%%")
        templates = {
            width: template_delimiter.join([task.fmt] * width)
            for width in {width for _, width in slices}
        }
        with open(task.output_path, "wb") as f:
            for i in range(task.start_row, task.stop_row):
                row = rows[i]
                # Formatting a whole slice with one `%` is 1.6x faster per value
                # than formatting elements one at a time, and the slice bounds
                # the tuple of NumPy scalars at _FORMAT_SLICE elements, so a
                # 125k-wide row never materialises 4 MB of scalar objects. The
                # scalars stay NumPy so that `%r` prints what np.savetxt prints.
                parts = [
                    templates[width] % tuple(row[start : start + width])
                    for start, width in slices
                ]
                f.write(task.delimiter.join(parts).encode("ascii") + b"\n")
    except Exception as e:
        raise RuntimeError(
            f"_format_rows_to_file failed on rows {task.start_row}-{task.stop_row}: {e}"
        ) from e


def _estimate_text_size(n_rows: int, n_cols: int) -> int:
    """Estimate text file size for a matrix written with %.10g format.

    Each float averages ~12 chars, plus one delimiter per column and a newline.
    """
    bytes_per_row = n_cols * 12 + n_cols  # values + delimiters/newline
    return n_rows * bytes_per_row


def write_matrix_parallel(
    matrix: np.ndarray,
    path: Path,
    fmt: str = "%.10g",
    delimiter: str = "\t",
    n_workers: int | None = None,
    min_rows_for_parallel: int = 500,
) -> None:
    """Write a 2D matrix to a text file, optionally using parallel formatting.

    Small matrices use np.savetxt. Larger default-format real matrices use
    GIL-free native formatting with bounded buffers, Python threads, and one
    ordered file writer. The matrix must not be mutated during the call.
    Custom formats or unavailable native support use processes and temporary
    files beside the output.

    Output is byte-identical to np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter).

    Args:
        matrix: 2D numpy array to write.
        path: Output file path.
        fmt: Format string for each element (default "%.10g").
        delimiter: Column separator (default tab).
        n_workers: Formatting workers, threads for native output and processes
            otherwise (default: physical CPU count capped at 32).
        min_rows_for_parallel: Row threshold for parallel path (default 500).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = matrix.shape

    if n_rows < min_rows_for_parallel or matrix.size == 0:
        logger.info(f"Writing {n_rows}x{n_cols} matrix to {path.resolve()}")
        # Publish atomically: np.savetxt truncates its target on open, so an
        # interrupted or failed write would otherwise destroy a pre-existing
        # valid file.
        with AtomicOutput(path) as publish_tmp:
            np.savetxt(publish_tmp, matrix, fmt=fmt, delimiter=delimiter)
        return

    if n_workers is None:
        n_workers = default_worker_count()
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    logger.info(
        f"Writing {n_rows}x{n_cols} matrix to {path.resolve()} ({n_workers} workers)"
    )

    if fmt == "%.10g" and delimiter == "\t" and matrix.dtype.kind in "fiub":
        formatter = native_formatter()
        if formatter is not None:
            write_native_matrix(matrix, path, n_workers, formatter)
            return

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

    with temp_dir_beside(path, prefix=".jamma_mwrite_") as tmp_dir:
        memmap = MemmapRef(str(tmp_dir / "matrix.dat"), matrix.shape, str(matrix.dtype))
        try:
            matrix.tofile(memmap.path)
        except OSError as e:
            raise OSError(
                f"Failed to write {matrix.nbytes / (1024**3):.1f} GB temp file "
                f"to {memmap.path} for parallel matrix IPC. "
                f"Ensure the output directory has sufficient disk space."
            ) from e

        tasks = [
            MatrixWriteTask(
                matrix=memmap,
                output_path=str(tmp_dir / f"chunk_{idx:06d}.txt"),
                start_row=start,
                stop_row=min(start + rows_per_chunk, n_rows),
                fmt=fmt,
                delimiter=delimiter,
            )
            for idx, start in enumerate(range(0, n_rows, rows_per_chunk))
        ]
        run_spawn_pool(
            _format_rows_to_file,
            tasks,
            error_context=f"writing {path}",
            n_workers=n_workers,
        )

        # Eager deletes bound peak disk: the memmap is 126 GB at 125k samples,
        # and each chunk goes as soon as it is copied. Concatenating into an
        # AtomicOutput temp means a failure never destroys an existing file.
        unlink_quietly(memmap.path)
        with AtomicOutput(path) as publish_tmp, open(publish_tmp, "wb") as f_out:
            for task in tasks:
                with open(task.output_path, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_out, 8 << 20)
                unlink_quietly(task.output_path)
