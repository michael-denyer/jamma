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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io._parallel_text import (
    MemmapRef,
    default_worker_count,
    run_spawn_pool,
    temp_dir_beside,
)


@dataclass(frozen=True, slots=True)
class MatrixReadTask:
    """Picklable message passed to one matrix-parsing worker."""

    txt_path: str
    matrix: MemmapRef
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
                dtype=np.dtype(task.matrix.dtype),
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

        mm = task.matrix.open("r+")
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


def _first_data_line(path: Path) -> bytes:
    """Return the first data line, raising if the file has none."""
    with open(path, "rb") as f:
        for line in f:
            if _is_data_line(line):
                return line
    raise ValueError(f"Matrix file has no data rows: {path}")


def _scan_chunk_boundaries(
    path: Path, n_workers: int
) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Split a text file into newline-aligned byte ranges and count their rows.

    One pass: seeks to approximate boundaries and counts each chunk's data rows
    via bounded line-by-line iteration; the counts sum to the total.

    Args:
        path: Input text file path.
        n_workers: Number of parallel chunks to create.

    Returns:
        Tuple of (n_rows, chunks) where each chunk is
        (start_byte, end_byte, start_row, n_rows_in_chunk).
    """
    file_size = path.stat().st_size
    target_chunk_size = file_size // n_workers
    chunks: list[tuple[int, int, int, int]] = []
    current_row = 0

    with open(path, "rb") as f:
        chunk_start = 0
        for _i in range(n_workers - 1):
            target = chunk_start + target_chunk_size
            if target >= file_size:
                break
            f.seek(target)
            f.readline()  # advance past next newline boundary
            chunk_end = f.tell()

            rows_in_chunk = _count_data_lines_between(f, chunk_start, chunk_end)

            if rows_in_chunk > 0:
                chunks.append((chunk_start, chunk_end, current_row, rows_in_chunk))
                current_row += rows_in_chunk
                chunk_start = chunk_end

        if chunk_start < file_size:
            rows_in_last = _count_data_lines_between(f, chunk_start, file_size)
            chunks.append((chunk_start, file_size, current_row, rows_in_last))
            current_row += rows_in_last

    return current_row, chunks


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

    # Extrapolate the row count from the first data line's length to decide
    # parallel vs serial; a fixed-size sample holds no newline once rows are
    # wide (75k columns is ~1.5 MB per line).
    first_line = _first_data_line(path)
    n_rows_approx = max(1, file_size // len(first_line))

    if n_rows_approx < min_rows_for_parallel:
        logger.info(f"Reading {path.name} via np.loadtxt (small matrix)")
        return np.atleast_2d(np.loadtxt(path, dtype=np.float64, delimiter=delimiter))

    if n_workers is None:
        n_workers = default_worker_count()
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    logger.info(f"Reading {path.name} via parallel parse ({n_workers} workers)")

    n_rows, chunks = _scan_chunk_boundaries(path, n_workers)
    n_cols = len(first_line.split(None if delimiter is None else delimiter.encode()))

    logger.debug(f"Matrix dimensions: {n_rows}x{n_cols}, {len(chunks)} chunks")

    with temp_dir_beside(path, prefix=".jamma_mread_") as tmp_dir:
        memmap = MemmapRef(str(tmp_dir / "matrix.dat"), (n_rows, n_cols), "float64")
        memmap.open("w+")  # create the zero-filled file workers reopen r+

        run_spawn_pool(
            _parse_chunk_to_memmap,
            [
                MatrixReadTask(
                    txt_path=str(path),
                    matrix=memmap,
                    start_byte=sb,
                    end_byte=eb,
                    start_row=sr,
                    row_count=nr,
                    delimiter=delimiter,
                )
                for sb, eb, sr, nr in chunks
            ],
            error_context=f"reading {path}",
            n_workers=n_workers,
        )

        # Block-by-block copy: only ~1024 rows of memmap pages are faulted at
        # a time, unlike np.array(mm), which faults the entire memmap.
        result = np.empty(memmap.shape, dtype=np.float64)
        mm = memmap.open("r")
        block_rows = min(1024, n_rows)
        for start in range(0, n_rows, block_rows):
            result[start : start + block_rows] = mm[start : start + block_rows]
        del mm
        return result
