"""Core computational modules for JAMMA.

This package contains the core algorithms and data structures:
- config: Configuration dataclasses
- jax_config: JAX configuration and verification
- kinship: Kinship matrix computation
- lmm: Linear mixed model fitting
"""

from jamma.core.config import OutputConfig
from jamma.core.jax_config import (
    configure_jax,
    get_jax_info,
    verify_jax_installation,
)
from jamma.core.memory import (
    MemoryBreakdown,
    MemorySnapshot,
    StreamingMemoryBreakdown,
    check_memory_available,
    check_memory_before_run,
    cleanup_memory,
    estimate_streaming_memory,
    estimate_workflow_memory,
    get_memory_snapshot,
    log_memory_snapshot,
)

__all__ = [
    "OutputConfig",
    "configure_jax",
    "get_jax_info",
    "verify_jax_installation",
    "MemoryBreakdown",
    "MemorySnapshot",
    "StreamingMemoryBreakdown",
    "check_memory_available",
    "check_memory_before_run",
    "cleanup_memory",
    "estimate_streaming_memory",
    "estimate_workflow_memory",
    "get_memory_snapshot",
    "log_memory_snapshot",
]
