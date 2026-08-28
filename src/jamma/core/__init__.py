"""Core infrastructure modules for JAMMA.

This package contains backend detection, configuration, memory estimation,
threading, hardware context, and telemetry.
"""

from jamma.core.backend import get_backend_info
from jamma.core.memory import (
    MemoryBreakdown,
    StreamingMemoryBreakdown,
    check_memory_available,
    estimate_lmm_memory,
    estimate_streaming_memory,
)
from jamma.core.memory_snapshot import (
    MemorySnapshot,
    cleanup_memory,
    get_memory_snapshot,
    log_memory_snapshot,
)

__all__ = [
    "MemoryBreakdown",
    "MemorySnapshot",
    "StreamingMemoryBreakdown",
    "check_memory_available",
    "cleanup_memory",
    "estimate_lmm_memory",
    "estimate_streaming_memory",
    "get_backend_info",
    "get_memory_snapshot",
    "log_memory_snapshot",
]
