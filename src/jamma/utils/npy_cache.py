"""Shared .npy sidecar cache validation and atomic publication for binary I/O."""

import os
import uuid
from pathlib import Path

import numpy as np
from loguru import logger


def save_npy_atomic(array: np.ndarray, npy_path: Path) -> None:
    """Save ``array`` to ``npy_path``, publishing it atomically.

    np.save truncates its target on open, so an interrupted save leaves a
    partial file. npy_cache_valid only rejects a zero-byte sidecar, so a
    truncated .npy is preferred over the text source on the next read. The
    write therefore goes to a sibling temp on the same filesystem and is
    renamed onto the destination only once it is complete.

    The pid and uuid in the temp name keep concurrent writers off each
    other's file. The suffix stays .npy so np.save does not append its own.

    Raises:
        OSError: If the write or the rename fails. The destination is left
            untouched and the temp is removed.
    """
    tmp_path = npy_path.parent / (
        f".{npy_path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.npy"
    )
    try:
        np.save(tmp_path, array)
        tmp_path.replace(npy_path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_err:
            logger.debug(f"Could not remove temp file {tmp_path}: {cleanup_err}")
        raise


def npy_cache_valid(txt_path: Path, npy_path: Path) -> bool:
    """Check if .npy cache exists and is at least as new as the text file.

    If the text file doesn't exist, any non-empty .npy cache is considered
    valid (binary-only write scenario).

    Args:
        txt_path: Path to the text file (may not exist).
        npy_path: Path to the .npy cache file.

    Returns:
        True if the .npy cache is usable.
    """
    if not npy_path.exists():
        return False
    try:
        npy_stat = npy_path.stat()
        if npy_stat.st_size == 0:
            return False
        if not txt_path.exists():
            return True
        txt_mtime = txt_path.stat().st_mtime
        return npy_stat.st_mtime >= txt_mtime
    except OSError as e:
        logger.warning(
            f"Could not stat .npy cache {npy_path}, falling back to text: {e}"
        )
        return False
