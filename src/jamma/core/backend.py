"""Compute backend detection and dispatch.

JAMMA supports two compute backends:
- JAX: GPU/TPU acceleration (preferred when available)
- Rust: CPU-only, no external dependencies (fallback)

Backend selection is automatic based on hardware but can be overridden
via the JAMMA_BACKEND environment variable.
"""

import os
from functools import cache
from typing import Literal

from loguru import logger

Backend = Literal["jax", "rust"]


@cache
def get_compute_backend() -> Backend:
    """Detect the best available compute backend.

    Priority:
    1. JAMMA_BACKEND environment variable (if set to 'jax' or 'rust')
    2. GPU available -> JAX
    3. Otherwise -> Rust

    Returns:
        Backend identifier ('jax' or 'rust').

    Examples:
        >>> import os
        >>> os.environ["JAMMA_BACKEND"] = "rust"
        >>> get_compute_backend.cache_clear()
        >>> get_compute_backend()
        'rust'
    """
    # Check for environment override
    override = os.environ.get("JAMMA_BACKEND", "").lower().strip()
    if override in ("jax", "rust"):
        logger.debug(f"Backend override via JAMMA_BACKEND={override}")
        return override

    # Check for GPU via JAX
    if _has_gpu():
        logger.debug("GPU detected, using JAX backend")
        return "jax"

    # Fall back to Rust
    logger.debug("No GPU detected, using Rust backend")
    return "rust"


def _has_gpu() -> bool:
    """Check if a GPU is available via JAX.

    Returns:
        True if JAX can access a GPU, False otherwise.
    """
    try:
        import jax

        devices = jax.devices()
        return any(d.platform in ("gpu", "cuda", "rocm") for d in devices)
    except ImportError:
        logger.debug("JAX not installed, no GPU support")
        return False
    except Exception as e:
        logger.debug(f"Error checking for GPU: {e}")
        return False


def is_rust_available() -> bool:
    """Check if the Rust backend (jamma_core) is available.

    Returns:
        True if jamma_core can be imported, False otherwise.
    """
    try:
        import jamma_core  # noqa: F401

        return True
    except ImportError:
        return False


def get_backend_info() -> dict:
    """Get information about available backends.

    Returns:
        Dictionary with backend availability and selection info.
    """
    return {
        "selected": get_compute_backend(),
        "rust_available": is_rust_available(),
        "gpu_available": _has_gpu(),
        "override": os.environ.get("JAMMA_BACKEND", None),
    }
