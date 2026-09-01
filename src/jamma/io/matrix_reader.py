"""Parallel matrix text reader for large matrices.

Provides read_matrix_parallel() which uses multiprocessing to parse matrix
rows across CPU cores. At 50k x 50k (2.5B floats), np.loadtxt is single-threaded
even in numpy 2.x (C-based tokenizer, but one core). Parallel parsing brings
cold reads from ~4 minutes to ~50 seconds on a 48-core machine.

Uses file-backed numpy.memmap for worker IPC instead of shared memory to avoid
SIGBUS crashes when Docker's /dev/shm is capped at 64 MB (cpython#114390).

Workers open the source text file at pre-computed byte offsets, seek to the chunk
start, and parse via np.loadtxt(f, max_rows=N) directly on the file handle. This
streams line-by-line internally instead of buffering the entire byte range in RAM.
The memmap-to-dense final copy uses block-by-block transfer (1024 rows at a time)
so that only a small window of memmap pages is faulted into physical memory at a
time, significantly reducing peak RSS compared to np.array(mm) which faults the
entire memmap at once.

Mirrors the conventions in matrix_writer.py: spawn context, file-backed memmap,
top-level picklable functions, temp dir on same filesystem as input.
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io._parallel_text import (
    default_worker_count,
    run_spawn_pool,
    temp_dir_beside,
    unlink_quietly,
)


@dataclass(frozen=True, slots=True)
class MatrixReadTask:
    """Picklable message passed to one matrix-parsing worker."""

    txt_path: str
    memmap_path: str
    shape: tuple[int, int]
    dtype: str
    start_byte: int
    end_byte: int
    start_row: int
    row_count: int
    delimiter: str | None


def _parse_chunk_to_memmap(task: MatrixReadTask) -> None:
    """Parse a byte range of a text file and write parsed rows into a memmap.

    Must be a top-level function for pickling with spawn context.

    Uses np.loadtxt with max_rows directly on the file handle instead of
    buffering the entire byte range via f.read(). This streams line-by-line
    internally, keeping per-worker memory bounded.
    """
    try:
        with open(task.txt_path, "rb") as f:
            f.seek(task.start_byte)
            chunk = np.loadtxt(
                f,
                dtype=np.dtype(task.dtype),
                delimiter=task.delimiter,
                max_rows=task.row_count,
            )
        chunk = np.atleast_2d(chunk)
        if chunk.shape[0] != task.row_count:
            raise RuntimeError(
                f"Row count mismatch in chunk at byte offset {task.start_byte}: "
                f"expected {task.row_count} data rows, parsed {chunk.shape[0]}. "
                f"File may have been modified during read."
            )

        mm = np.memmap(
            task.memmap_path,
            dtype=np.dtype(task.dtype),
            mode="r+",
            shape=task.shape,
        )
        mm[task.start_row : task.start_row + chunk.shape[0], :] = chunk
        del mm  # release memmap reference
    except MemoryError:
        raise  # Let parent process handle OOM directly
    except Exception as e:
        raise RuntimeError(
            f"_parse_chunk_to_memmap failed at bytes "
            f"{task.start_byte}-{task.end_byte}, "
            f"row offset {task.start_row}: {e}"
        ) from e


def _is_data_line(line: bytes) -> bool:
    """Return True if line is a non-blank, non-comment data line.

    Matches np.loadtxt behaviour: blank lines and lines starting with '#'
    (after stripping leading whitespace) are skipped.
    """
    stripped = line.lstrip()
    return len(stripped) > 0 and not stripped.startswith(b"#")


def _count_data_lines_between(f, start: int, end: int) -> int:
    """Count data lines between byte offsets using readline iteration.

    Uses f.readline() instead of ``for line in f`` because Python's file
    iterator uses an internal read-ahead buffer that makes f.tell()
    unreliable (returns buffer position, not line position).

    Avoids materializing the entire byte range in memory — at 200GB files
    with 4 workers, each chunk is ~50GB which would OOM before parsing.
    """
    f.seek(start)
    count = 0
    while True:
        line = f.readline()
        if not line or f.tell() > end:
            break
        if _is_data_line(line):
            count += 1
    return count


def _scan_chunk_boundaries(
    path: Path, n_workers: int, delimiter: str | None = None
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """Scan a text file to find byte offsets aligned to newline boundaries.

    Two-pass scan: first pass counts total data rows and detects column count;
    second pass seeks to approximate boundaries and counts per-chunk rows via
    bounded line-by-line iteration.

    Args:
        path: Input text file path.
        n_workers: Number of parallel chunks to create.
        delimiter: Column separator (None = whitespace).

    Returns:
        Tuple of (n_rows, n_cols, chunks) where each chunk is
        (start_byte, end_byte, start_row, n_rows_in_chunk).
    """
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Matrix file is empty: {path}")

    # First pass: find first data line (for column count) and count all data rows.
    first_line = b""
    n_rows = 0
    with open(path, "rb") as f:
        for raw_line in f:
            if _is_data_line(raw_line):
                if not first_line:
                    first_line = raw_line
                n_rows += 1

    if n_rows == 0 or not first_line:
        raise ValueError(f"Matrix file has no data rows: {path}")

    if delimiter is not None:
        n_cols = len(first_line.split(delimiter.encode()))
    else:
        n_cols = len(first_line.split())

    # Compute byte boundaries aligned to newlines
    target_chunk_size = file_size // n_workers
    chunks: list[tuple[int, int, int, int]] = []
    current_row = 0

    with open(path, "rb") as f:
        chunk_start = 0
        for _i in range(n_workers - 1):
            # Seek to approximate split point and find next newline
            target = chunk_start + target_chunk_size
            if target >= file_size:
                break
            f.seek(target)
            # Read ahead to find newline boundary
            f.readline()  # advance past next newline boundary
            chunk_end = f.tell()

            # Count data rows in this chunk via bounded line iteration
            # (no large byte buffer allocation)
            rows_in_chunk = _count_data_lines_between(f, chunk_start, chunk_end)

            if rows_in_chunk > 0:
                chunks.append((chunk_start, chunk_end, current_row, rows_in_chunk))
                current_row += rows_in_chunk
                chunk_start = chunk_end

        # Final chunk: everything remaining
        if chunk_start < file_size:
            rows_in_last = n_rows - current_row
            chunks.append((chunk_start, file_size, current_row, rows_in_last))

    return n_rows, n_cols, chunks


def _cleanup_temp_memmap(tmp_dir: str, memmap_path: str) -> None:
    """Clean up the temp memmap file and its parent directory."""
    unlink_quietly(memmap_path)
    try:
        Path(tmp_dir).rmdir()
    except FileNotFoundError:
        pass
    except OSError as e:
        # loguru may be torn down when this runs from a finalizer at shutdown.
        with contextlib.suppress(Exception):
            logger.warning(f"Could not remove temp dir {tmp_dir}: {e}")


def read_matrix_parallel(
    path: Path | str,
    delimiter: str | None = None,
    n_workers: int | None = None,
    min_rows_for_parallel: int = 500,
) -> np.ndarray:
    """Read a 2D matrix from a text file, optionally using parallel parsing.

    For matrices with fewer than min_rows_for_parallel rows, falls back to
    np.loadtxt. For larger matrices, distributes row parsing across multiple
    processes for significant speedup.

    Args:
        path: Input text file path.
        delimiter: Column separator (None = whitespace, matching np.loadtxt default).
        n_workers: Number of worker processes (default: min(cpu_count, 32)).
        min_rows_for_parallel: Row threshold for parallel path (default 500).

    Returns:
        2D float64 numpy array (C-contiguous).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Matrix file is empty: {path}")

    # Quick row count to decide parallel vs serial.
    # Read the first data line to get bytes-per-line, then extrapolate.
    # Previous 64KB sample approach failed for wide matrices (75k+ columns
    # produce ~1.5MB lines, so no newline appeared in the sample).
    # Skip comment lines (starting with '#') to avoid inflated line length.
    with open(path, "rb") as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"Matrix file is empty: {path}")
        while first_line.startswith(b"#"):
            first_line = f.readline()
            if not first_line:
                raise ValueError(f"Matrix file contains only comments: {path}")
        bytes_per_line = len(first_line)
    n_rows_approx = max(1, file_size // bytes_per_line)

    if n_rows_approx < min_rows_for_parallel:
        logger.info(f"Reading {path.name} via np.loadtxt (small matrix)")
        return np.atleast_2d(np.loadtxt(path, dtype=np.float64, delimiter=delimiter))

    if n_workers is None:
        n_workers = default_worker_count()
    n_workers = max(1, n_workers)

    logger.info(f"Reading {path.name} via parallel parse ({n_workers} workers)")

    n_rows, n_cols, chunks = _scan_chunk_boundaries(path, n_workers, delimiter)
    shape = (n_rows, n_cols)

    logger.debug(f"Matrix dimensions: {n_rows}x{n_cols}, {len(chunks)} chunks")

    tmp_dir = temp_dir_beside(path, prefix=".jamma_mread_")
    memmap_path = str(Path(tmp_dir) / "matrix.dat")

    try:
        # Create zero-filled memmap for workers to write into
        mm = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=shape)
        del mm  # release memmap reference; workers reopen in r+ mode

        dtype_str = str(np.dtype(np.float64))
        chunk_args = [
            MatrixReadTask(
                txt_path=str(path),
                memmap_path=memmap_path,
                shape=shape,
                dtype=dtype_str,
                start_byte=sb,
                end_byte=eb,
                start_row=sr,
                row_count=nr,
                delimiter=delimiter,
            )
            for sb, eb, sr, nr in chunks
        ]

        run_spawn_pool(
            _parse_chunk_to_memmap,
            chunk_args,
            error_context=f"reading {path}",
            n_workers=n_workers,
        )

        # Block-by-block copy: dense output is pre-allocated at full size,
        # but only ~1024 rows of memmap pages are faulted at a time.  For
        # matrices larger than physical memory this significantly reduces
        # peak RSS vs np.array(mm) which faults the entire memmap.
        result = np.empty(shape, dtype=np.float64)
        mm = np.memmap(memmap_path, dtype=np.float64, mode="r", shape=shape)
        block_rows = min(1024, shape[0])
        for start in range(0, shape[0], block_rows):
            end = min(start + block_rows, shape[0])
            result[start:end] = mm[start:end]
        del mm

        return result
    finally:
        _cleanup_temp_memmap(tmp_dir, memmap_path)
