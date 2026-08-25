"""Process and system memory snapshots for diagnostics.

The one place JAMMA reads process RSS. ``log_memory_snapshot`` replaces the
old ``utils.logging.log_rss_memory``; runners log through it at phase
boundaries.
"""

import gc
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


def cleanup_memory(verbose: bool = True) -> MemorySnapshot:
    """Free memory after a computation run.

    Call this between benchmark runs or after large computations to
    prevent memory accumulation that can cause OOM/SIGSEGV errors.

    This function:
    1. Runs Python garbage collection
    2. Runs a second GC pass
    3. Logs memory before/after cleanup if verbose

    Args:
        verbose: If True (default), log memory before and after cleanup.

    Returns:
        MemorySnapshot after cleanup.

    Example:
        >>> # After a benchmark run
        >>> del kinship, eigenvectors, results
        >>> cleanup_memory()
        INFO | Memory [before_cleanup]: using 89.5GB, 160.2GB free of 256.0GB
        INFO | Memory [after_cleanup]: using 12.3GB, 237.4GB free of 256.0GB
        INFO | Freed 77.2GB (process was using 89.5GB, now 12.3GB)

    Note:
        For best results, explicitly `del` large arrays before calling
        this function. Python's reference counting means arrays won't
        be freed if references still exist.
    """
    before = log_memory_snapshot("before_cleanup") if verbose else get_memory_snapshot()

    gc.collect()
    gc.collect()

    if verbose:
        after = log_memory_snapshot("after_cleanup")
        freed_gb = before.rss_gb - after.rss_gb
        if freed_gb > 0.1:  # Only log if meaningful change
            logger.info(
                f"Freed {freed_gb:.1f}GB (process was using "
                f"{before.rss_gb:.1f}GB, now {after.rss_gb:.1f}GB)"
            )
        elif freed_gb < -0.1:
            logger.warning(
                f"Memory increased by {-freed_gb:.1f}GB during cleanup "
                f"(was {before.rss_gb:.1f}GB, now {after.rss_gb:.1f}GB)"
            )
    else:
        after = get_memory_snapshot()

    return after
