"""Core computational modules for GEMMA-Next.

This package contains the core algorithms and data structures:
- config: Configuration dataclasses
- jax_config: JAX configuration and verification
- kinship: Kinship matrix computation
- lmm: Linear mixed model fitting
"""

from gemma_next.core.config import OutputConfig
from gemma_next.core.jax_config import (
    configure_jax,
    get_jax_info,
    verify_jax_installation,
)

__all__ = [
    "OutputConfig",
    "configure_jax",
    "get_jax_info",
    "verify_jax_installation",
]
