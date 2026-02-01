"""Core computational modules for GEMMA-Next.

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
    StreamingMemoryBreakdown,
    check_memory_available,
    estimate_streaming_memory,
    estimate_workflow_memory,
)

__all__ = [
    "OutputConfig",
    "configure_jax",
    "get_jax_info",
    "verify_jax_installation",
    "MemoryBreakdown",
    "StreamingMemoryBreakdown",
    "check_memory_available",
    "estimate_streaming_memory",
    "estimate_workflow_memory",
]
