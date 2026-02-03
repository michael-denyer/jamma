"""Eigendecomposition backend detection and dispatch.

JAMMA supports three backends:

- jax.scipy: JAX pipeline + scipy/LAPACK eigendecomposition.
  Uses scipy.linalg.eigh for eigendecomp, JAX for other operations.
  Requires properly configured BLAS (OpenBLAS can SIGSEGV at 100k+ samples).

- jax.rust: JAX pipeline + faer/Rust eigendecomposition.
  Uses jamma_core (faer) for eigendecomp, JAX for other operations.
  Preferred for 100k+ samples due to stability.

- rust: Pure Rust LMM (NOT YET IMPLEMENTED).
  Future backend for complete Rust-based LMM computation.
  Selecting this backend will error until implemented.

Backend selection is automatic (preferring jax.rust when jamma_core is available)
but can be overridden via the JAMMA_BACKEND environment variable.
"""

import os
import warnings
from functools import cache
from typing import Literal

from loguru import logger

# Canonical backend names (excludes "rust" which is a stub)
Backend = Literal["jax.scipy", "jax.rust"]

# Backward compatibility aliases (old name -> canonical name)
# NOTE: "rust" is NOT here because bare "rust" now means pure Rust LMM (future)
BACKEND_ALIASES: dict[str, str] = {
    "jax": "jax.scipy",  # Old "jax" -> new "jax.scipy"
}


def normalize_backend_name(value: str) -> str:
    """Normalize backend name to canonical form.

    Handles case-insensitivity, whitespace, and backward compatibility aliases.
    Emits deprecation warnings for old backend names.

    Args:
        value: Backend name from user input or environment.

    Returns:
        Normalized backend name. Note that "rust" passes through unchanged
        (the stub error is handled in get_compute_backend).

    Examples:
        >>> normalize_backend_name("jax.scipy")
        'jax.scipy'
        >>> normalize_backend_name("JAX.RUST")
        'jax.rust'
    """
    normalized = value.lower().strip()

    # Check for deprecated aliases
    if normalized in BACKEND_ALIASES:
        canonical = BACKEND_ALIASES[normalized]
        warnings.warn(
            f"Backend name '{normalized}' is deprecated. Use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical

    # Return as-is (including "rust" which will error in get_compute_backend)
    return normalized


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
