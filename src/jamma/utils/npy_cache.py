"""Shared .npy sidecar cache validation for binary I/O."""

from pathlib import Path

from loguru import logger


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
