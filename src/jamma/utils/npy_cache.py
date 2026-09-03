"""Shared .npy sidecar cache validation and atomic publication for binary I/O.

``read_array_artifact`` is the one reader for the "binary .npy default, GEMMA
text legacy, .npy sidecar cache" contract that kinship and eigen files share.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from jamma.utils.atomic_publish import atomic_output


def save_npy_atomic(array: np.ndarray, npy_path: Path) -> None:
    """Save ``array`` to ``npy_path``, publishing it atomically.

    np.save truncates its target on open, so an interrupted or failed save
    leaves a partial file. npy_cache_valid only rejects a zero-byte sidecar, so
    a truncated .npy is preferred over the text source on the next read. The
    write therefore goes to a sibling temp and is renamed onto the destination
    only once it is complete.

    This protects against an interrupted or failed write, not against power
    loss; see jamma.utils.atomic_publish for why nothing here fsyncs.

    Raises:
        OSError: If the write or the rename fails. The destination is left
            untouched and the temp is removed.
    """
    with atomic_output(npy_path, suffix=".npy") as tmp_path:
        np.save(tmp_path, array)


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


def write_npy_cache(array: np.ndarray, npy_path: Path) -> None:
    """Write the .npy sidecar, swallowing filesystem errors.

    The sidecar is a read accelerator, not the artifact, so a read-only
    filesystem or a full disk must not abort a caller whose real output
    already landed. That tolerance is the only thing this adds over
    ``save_npy_atomic``.
    """
    try:
        save_npy_atomic(array, npy_path)
    except OSError as e:
        logger.warning(f"Could not write .npy cache {npy_path}: {e}")


def load_npy_cache(
    npy_path: Path, *, mmap_mode: Literal["r"] | None = None
) -> np.ndarray | None:
    """Load a .npy sidecar, removing it and returning None when it is corrupt.

    With ``mmap_mode="r"`` the result is a read-only memory map whose pages
    the OS loads on demand. A truncated or unreadable sidecar is unlinked so
    the caller re-parses the text and rewrites it.
    """
    try:
        return np.load(npy_path, mmap_mode=mmap_mode)
    except (OSError, ValueError) as e:
        logger.warning(f"Corrupt .npy cache {npy_path}, will re-parse text: {e}")
        try:
            npy_path.unlink()
        except OSError as unlink_err:
            logger.warning(f"Could not remove corrupt cache {npy_path}: {unlink_err}")
        return None


def read_array_artifact(
    path: Path,
    *,
    what: str,
    parse_text: Callable[[Path], np.ndarray],
    check: Callable[[np.ndarray, Path], np.ndarray],
    mmap_mode: Literal["r"] | None = None,
) -> np.ndarray:
    """Read one array from .npy, its .npy sidecar, or GEMMA text.

    A .npy path loads directly. A text path loads its sidecar when the sidecar
    is at least as new as the text and not corrupt, else parses the text and
    writes the sidecar for next time. ``check`` runs on every branch before
    the array is returned or cached, so a sidecar never holds an array the
    caller would reject; it may return a promoted view of its input.
    ``mmap_mode`` applies to sidecar loads only.

    Raises:
        ValueError: If the text cannot be parsed, is empty, or ``check``
            rejects the array.
    """
    path = Path(path)
    if path.suffix == ".npy":
        logger.info(f"Reading {what} from {path}")
        return check(np.load(path), path)

    npy_path = path.with_suffix(".npy")
    if npy_cache_valid(path, npy_path):
        data = load_npy_cache(npy_path, mmap_mode=mmap_mode)
        if data is not None:
            logger.info(f"Reading {what} from cache {npy_path}")
            return check(data, npy_path)

    logger.info(f"Reading {what} from {path}")
    try:
        data = parse_text(path)
    except ValueError as e:
        raise ValueError(f"Cannot parse {what} file {path}: {e}") from e
    if data.size == 0:
        raise ValueError(f"{what.capitalize()} file is empty: {path}")
    data = check(data, path)
    write_npy_cache(data, npy_path)
    return data
