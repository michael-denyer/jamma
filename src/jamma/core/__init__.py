"""Core computational modules for JAMMA.

This package contains the core algorithms and data structures:
- config: Configuration dataclasses
- jax_config: JAX configuration and verification (optional — requires JAX)
- kinship: Kinship matrix computation
- lmm: Linear mixed model fitting
"""

from jamma.core.backend import get_backend_info, has_jax
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

# JAX configuration — probe for JAX first, then import unconditionally
# so that non-JAX import errors propagate immediately.
_HAS_JAX = has_jax()

if _HAS_JAX:
    from jamma.core.jax_config import (  # noqa: F401
        configure_jax,
        ensure_jax_configured,
        get_jax_info,
        verify_jax_installation,
    )

__all__ = [
    "get_backend_info",
    "OutputConfig",
    "MemoryBreakdown",
    "MemorySnapshot",
    "StreamingMemoryBreakdown",
    "check_memory_available",
    "check_memory_before_run",
    "cleanup_memory",
    "estimate_lmm_memory",
    "estimate_lmm_streaming_memory",
    "estimate_streaming_memory",
    "estimate_workflow_memory",
    "get_memory_snapshot",
    "log_memory_snapshot",
]
if _HAS_JAX:
    __all__ += [
        "configure_jax",
        "ensure_jax_configured",
        "get_jax_info",
        "verify_jax_installation",
    ]
