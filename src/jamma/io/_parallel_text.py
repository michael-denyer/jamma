"""Shared plumbing for the parallel text matrix reader and writer.

matrix_reader.py parses a text matrix into a memmap across a spawn pool, and
matrix_writer.py formats a memmap back to text across one. The pool setup, the
worker cap, and the temp-dir-beside-the-target are the same in both; only the
worker function, the gather step, and the output shape differ. Those identical
pieces live here.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from loguru import logger

__all__ = [
    "MAX_WORKERS",
    "MemmapRef",
    "default_worker_count",
    "run_spawn_pool",
    "temp_dir_beside",
]

MAX_WORKERS = 32


def default_worker_count() -> int:
    """CPU count capped at MAX_WORKERS. Callers apply their own floor policy."""
    return min(os.cpu_count() or 1, MAX_WORKERS)


@dataclass(frozen=True, slots=True)
class MemmapRef:
    """Picklable description of a file-backed matrix shared with pool workers."""

    path: str
    shape: tuple[int, int]
    dtype: str

    def open(self, mode: Literal["r", "r+", "w+"]) -> np.memmap:
        """Map the backing file with this shape and dtype."""
        return np.memmap(
            self.path, dtype=np.dtype(self.dtype), mode=mode, shape=self.shape
        )


@contextmanager
def temp_dir_beside(target: Path, prefix: str) -> Iterator[Path]:
    """Create a temp dir on the same filesystem as ``target`` for one block.

    Tries the target's parent directory first so large temp files land beside
    the input or output rather than on a possibly RAM-backed /tmp. Falls back
    to the system default on OSError. The directory and everything in it is
    removed when the block exits, on success, error, or interrupt.

    Args:
        target: The input or output file whose parent directory is preferred.
        prefix: mkdtemp prefix (dotted so the beside-the-target dir is hidden).

    Yields:
        The created temp directory.
    """
    parent = target.parent
    try:
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    except OSError as e:
        logger.warning(
            f"Cannot create temp dir in {parent} ({e}), falling back to system "
            "tmpdir. If /tmp is RAM-backed (tmpfs), this may increase memory "
            "usage significantly for large matrices."
        )
        path = Path(tempfile.mkdtemp(prefix=prefix.lstrip(".")))
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except OSError as e:
            logger.warning(f"Could not remove temp dir {path}: {e}")


WorkItem = TypeVar("WorkItem")


def run_spawn_pool(
    worker_fn: Callable[[WorkItem], object],
    args: Iterable[WorkItem],
    *,
    error_context: str,
    n_workers: int,
) -> None:
    """Run ``worker_fn`` over ``args`` in an ordered spawn pool, draining results.

    Uses ``imap`` (not ``imap_unordered``) so both callers can rely on chunk
    order for their gather step. Workers communicate through files, so the
    drained results are discarded. On any BaseException, including
    KeyboardInterrupt and SystemExit, the pool is terminated and joined so no
    worker is orphaned, then the error is re-raised. The exception is logged
    unless it is a KeyboardInterrupt or SystemExit, where an interrupt is not
    an error worth a stack trace.

    Args:
        worker_fn: A module-level function (spawn requires it be picklable).
        args: One argument per task, consumed lazily.
        error_context: Phrase for the error log, e.g. "reading /path".
        n_workers: Pool size; caller has already resolved and validated it.
    """
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        try:
            for _ in pool.imap(worker_fn, args):
                pass
        except BaseException as e:
            pool.terminate()
            pool.join()
            if not isinstance(e, (KeyboardInterrupt, SystemExit)):
                logger.opt(exception=e).error(f"Pool error {error_context}: {e}")
            raise
