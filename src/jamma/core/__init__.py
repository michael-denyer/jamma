"""Core infrastructure modules for JAMMA.

This package contains backend detection, configuration, memory estimation,
threading, hardware context, and telemetry.
"""

from jamma.core.backend import get_backend_info
from jamma.core.config import OutputConfig
from jamma.core.memory import (
    MemoryBreakdown,
    MemorySnapshot,
    StreamingMemoryBreakdown,
    check_memory_available,
    check_memory_before_run,
    cleanup_memory,
    estimate_lmm_memory,
    estimate_lmm_streaming_memory,
    estimate_streaming_memory,
    estimate_workflow_memory,
    get_memory_snapshot,
    log_memory_snapshot,
)

__all__ = [
    "MemoryBreakdown",
    "MemorySnapshot",
    "OutputConfig",
    "StreamingMemoryBreakdown",
    "check_memory_available",
    "check_memory_before_run",
    "cleanup_memory",
    "estimate_lmm_memory",
    "estimate_lmm_streaming_memory",
    "estimate_streaming_memory",
    "estimate_workflow_memory",
    "get_backend_info",
    "get_memory_snapshot",
    "log_memory_snapshot",
]
