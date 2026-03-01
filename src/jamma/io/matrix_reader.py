"""Parallel matrix text reader for large matrices.

Provides read_matrix_parallel() which uses multiprocessing to parse matrix
rows across CPU cores. At 50k x 50k (2.5B floats), np.loadtxt is single-threaded
even in numpy 2.x (C-based tokenizer, but one core). Parallel parsing brings
cold reads from ~4 minutes to ~50 seconds on a 48-core machine.

Uses file-backed numpy.memmap for worker IPC instead of shared memory to avoid
SIGBUS crashes when Docker's /dev/shm is capped at 64 MB (cpython#114390).

Workers open the source text file at pre-computed byte offsets aligned to newline
boundaries, parse their chunk via np.loadtxt(BytesIO(raw)), and write the result
directly into the shared memmap. This keeps IPC overhead near zero regardless of
matrix size.

Mirrors the conventions in matrix_writer.py: spawn context, file-backed memmap,
top-level picklable functions, temp dir on same filesystem as input.
"""

import multiprocessing as mp
import os
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from loguru import logger

# Cap worker count — beyond this, disk I/O contention and memory pressure
# outweigh the benefit of additional parse parallelism.
_MAX_READERS = 32


def _parse_chunk_to_memmap(args: tuple) -> None:
    """Parse a byte range of a text file and write parsed rows into a memmap.

    Must be a top-level function for pickling with spawn context.

    Args:
        args: Tuple of (txt_path, memmap_path, shape, dtype_str,
              start_byte, end_byte, start_row, delimiter).
    """
    (
        txt_path,
        memmap_path,
        shape,
        dtype_str,
        start_byte,
        end_byte,
        start_row,
        delimiter,
    ) = args

    try:
        with open(txt_path, "rb") as f:
            f.seek(start_byte)
            raw = f.read(end_byte - start_byte)

        chunk = np.loadtxt(BytesIO(raw), dtype=np.dtype(dtype_str), delimiter=delimiter)
        chunk = np.atleast_2d(chunk)

        mm = np.memmap(memmap_path, dtype=np.dtype(dtype_str), mode="r+", shape=shape)
        n_rows = chunk.shape[0]
        mm[start_row : start_row + n_rows, :] = chunk
        del mm  # flush
    except Exception as e:
        raise RuntimeError(
            f"_parse_chunk_to_memmap failed at bytes {start_byte}-{end_byte}, "
            f"row offset {start_row}: {e}"
        ) from e


def _scan_chunk_boundaries(
    path: Path, n_workers: int
) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Scan a text file to find byte offsets aligned to newline boundaries.

    Returns:
        Tuple of (n_rows, n_cols, chunks) where each chunk is
        (start_byte, end_byte, start_row).
    """
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Matrix file is empty: {path}")

    # Read first line to detect column count
    with open(path, "rb") as f:
        first_line = f.readline()
    n_cols = len(first_line.split())

    # Count total lines (= rows) via fast byte scan.
    # Track the last byte read to handle files without a trailing newline.
    n_rows = 0
    last_byte = b""
    with open(path, "rb") as f:
        while True:
            buf = f.read(8 * 1024 * 1024)
            if not buf:
                break
            n_rows += buf.count(b"\n")
            last_byte = buf[-1:]
    if last_byte and last_byte != b"\n":
        n_rows += 1

    if n_rows == 0:
        raise ValueError(f"Matrix file has no data rows: {path}")

    # Compute byte boundaries aligned to newlines
    target_chunk_size = file_size // n_workers
    chunks: list[tuple[int, int, int]] = []
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

            # Count rows in this chunk
            f.seek(chunk_start)
            chunk_bytes = f.read(chunk_end - chunk_start)
            rows_in_chunk = chunk_bytes.count(b"\n")

            if rows_in_chunk > 0:
                chunks.append((chunk_start, chunk_end, current_row))
                current_row += rows_in_chunk
                chunk_start = chunk_end

        # Final chunk: everything remaining
        if chunk_start < file_size:
            chunks.append((chunk_start, file_size, current_row))

    return n_rows, n_cols, chunks


def _create_temp_dir(input_path: Path) -> str:
    """Create temp dir on the same filesystem as the input file.

    Tries the input's parent directory first, falls back to the system
    default (TMPDIR / /tmp) if that fails.
    """
    input_dir = input_path.parent
    try:
        return tempfile.mkdtemp(prefix=".jamma_mread_", dir=input_dir)
    except OSError:
        logger.warning(
            f"Cannot create temp dir in {input_dir}, falling back to system tmpdir"
        )
        return tempfile.mkdtemp(prefix="jamma_mread_")


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
    # Read the first line to get bytes-per-line, then extrapolate.
    # Previous 64KB sample approach failed for wide matrices (75k+ columns
    # produce ~1.5MB lines, so no newline appeared in the sample).
    with open(path, "rb") as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"Matrix file is empty: {path}")
        bytes_per_line = len(first_line)
    n_rows_approx = max(1, file_size // bytes_per_line)

    if n_rows_approx < min_rows_for_parallel:
        logger.info(f"Reading {path.name} via np.loadtxt (small matrix)")
        return np.atleast_2d(np.loadtxt(path, dtype=np.float64, delimiter=delimiter))

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, _MAX_READERS)
    n_workers = max(1, n_workers)

    logger.info(f"Reading {path.name} via parallel parse ({n_workers} workers)")

    n_rows, n_cols, chunks = _scan_chunk_boundaries(path, n_workers)
    shape = (n_rows, n_cols)

    logger.debug(f"Matrix dimensions: {n_rows}x{n_cols}, {len(chunks)} chunks")

    tmp_dir = _create_temp_dir(path)
    memmap_path = os.path.join(tmp_dir, "matrix.dat")

    try:
        # Create zero-filled memmap for workers to write into
        mm = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=shape)
        del mm  # flush creation, workers open in r+ mode

        dtype_str = str(np.dtype(np.float64))
        chunk_args = [
            (str(path), memmap_path, shape, dtype_str, sb, eb, sr, delimiter)
            for sb, eb, sr in chunks
        ]

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            try:
                for _ in pool.imap(_parse_chunk_to_memmap, chunk_args):
                    pass
            except BaseException as e:
                logger.opt(exception=e).error(f"Pool error reading {path}: {e}")
                pool.terminate()
                pool.join()
                raise

        # Copy memmap to contiguous array, then delete memmap
        mm = np.memmap(memmap_path, dtype=np.float64, mode="r", shape=shape)
        result = np.array(mm)
        del mm

        return result
    finally:
        try:
            os.unlink(memmap_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Failed to clean up temp memmap {memmap_path}: {e}")
        try:
            os.rmdir(tmp_dir)
        except OSError as e:
            logger.debug(f"Could not remove temp dir {tmp_dir}: {e}")
