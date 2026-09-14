"""Ordered text writes with bounded buffers and GIL-free native conversion."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from jamma._build_support.build_models import MATRIX_TEXT_SPEC
from jamma._native import _load_c_module
from jamma.utils.atomic_publish import AtomicOutput

_VALUES_PER_BLOCK = 65_536
_BYTES_PER_VALUE = 32
FormatInto = Callable[[np.ndarray, bytearray], int]


@cache
def native_formatter() -> FormatInto | None:
    """Load once, with the same ABI/rebuild/fallback policy as the compute modules."""
    module = _load_c_module(MATRIX_TEXT_SPEC, expected_abi=1)
    return None if module is None else module.format_into


def _format_block(
    matrix: np.ndarray, buffer: bytearray, format_into: FormatInto
) -> int:
    # Conversion is bounded by a block, including for Fortran order, negative
    # strides and non-native byte order. Contiguous float64 needs no copy.
    return format_into(np.ascontiguousarray(matrix, dtype=np.float64), buffer)


def write_native_matrix(
    matrix: np.ndarray, path: Path, n_workers: int, format_into: FormatInto
) -> None:
    """Write %.10g/tab text atomically with at most two buffers per worker.

    The caller retains the matrix and must not mutate it during the write.
    Each block has at most 65,536 values, or one complete row when wider.
    Workers borrow input slices and format into private reusable bytearrays.
    Only this thread writes the file, in row order.
    """
    rows, columns = matrix.shape
    block_rows = max(1, _VALUES_PER_BLOCK // columns)
    starts = iter(range(0, rows, block_rows))
    capacity = min(rows, block_rows) * columns * _BYTES_PER_VALUE
    workers = min(n_workers, (rows + block_rows - 1) // block_rows)

    with AtomicOutput(path) as temporary, open(temporary, "wb") as output:
        if workers <= 1:
            buffer = bytearray(capacity)
            for start in starts:
                used = _format_block(
                    matrix[start : start + block_rows], buffer, format_into
                )
                with memoryview(buffer) as view:
                    output.write(view[:used])
            return

        pending: deque[tuple[Future[int], bytearray]] = deque()
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="matrix-text")
        try:
            for _ in range(2 * workers):
                start = next(starts, None)
                if start is None:
                    break
                buffer = bytearray(capacity)
                future = pool.submit(
                    _format_block,
                    matrix[start : start + block_rows],
                    buffer,
                    format_into,
                )
                pending.append((future, buffer))
            while pending:
                future, buffer = pending.popleft()
                used = future.result()
                with memoryview(buffer) as view:
                    output.write(view[:used])
                start = next(starts, None)
                if start is not None:
                    future = pool.submit(
                        _format_block,
                        matrix[start : start + block_rows],
                        buffer,
                        format_into,
                    )
                    pending.append((future, buffer))
        finally:
            # Includes KeyboardInterrupt and write failures. No worker touches
            # the file; wait for borrowed matrix/buffer references before exit.
            pool.shutdown(wait=True, cancel_futures=True)
