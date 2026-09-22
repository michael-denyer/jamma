"""Process and system memory snapshots for diagnostics.

The one place JAMMA reads process RSS. ``log_memory_snapshot`` replaces the
old ``utils.logging.log_rss_memory``; runners log through it at phase
boundaries.
"""

from typing import NamedTuple

import psutil
from loguru import logger


class MemorySnapshot(NamedTuple):
    """Snapshot of current memory state for debugging.

    All values in GB.
    """

    rss_gb: float  # Resident Set Size (actual RAM used by process)
    vms_gb: float  # Virtual Memory Size (total address space)
    available_gb: float  # Available system memory
    total_gb: float  # Total system memory
    percent_used: float  # Percentage of total system memory in use


def get_memory_snapshot() -> MemorySnapshot:
    """Get current memory usage snapshot.

    Returns:
        MemorySnapshot with RSS, VMS, available, and total memory.

    Example:
        >>> snap = get_memory_snapshot()
        >>> print(f"Using {snap.rss_gb:.1f}GB of {snap.total_gb:.1f}GB")
    """
    mem_info = psutil.Process().memory_info()
    vm = psutil.virtual_memory()

    return MemorySnapshot(
        rss_gb=mem_info.rss / 1e9,
        vms_gb=mem_info.vms / 1e9,
        available_gb=vm.available / 1e9,
        total_gb=vm.total / 1e9,
        percent_used=((vm.total - vm.available) / vm.total) * 100,
    )


def log_memory_snapshot(label: str = "", level: str = "INFO") -> MemorySnapshot:
    """Log current memory state with optional label.

    Useful for debugging memory issues in Databricks notebooks or
    tracking memory across benchmark runs.

    Args:
        label: Optional label for this snapshot (e.g., "after_eigendecomp").
        level: Log level ("DEBUG", "INFO", "WARNING").

    Returns:
        MemorySnapshot for chaining/assertions.

    Example:
        >>> log_memory_snapshot("before_100k_run")
        INFO | Memory [before_100k_run]: using 89.5GB,
             160.2GB free of 256.0GB (35.0% used)
    """
    snap = get_memory_snapshot()
    label_str = f" [{label}]" if label else ""
    msg = (
        f"Memory{label_str}: using {snap.rss_gb:.1f}GB, "
        f"{snap.available_gb:.1f}GB free of {snap.total_gb:.1f}GB "
        f"({snap.percent_used:.1f}% used)"
    )
    logger.log(level, msg)
    return snap
