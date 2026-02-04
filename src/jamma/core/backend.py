"""Eigendecomposition backend detection and dispatch.

JAMMA supports three backends:

- jax.numpy: JAX pipeline + numpy/LAPACK eigendecomposition.
  Uses numpy.linalg.eigh for eigendecomp (via MKL when available), JAX for other
  operations. Preferred for 100k+ samples due to lower memory overhead vs faer.

- jax.rust: JAX pipeline + faer/Rust eigendecomposition.
  Uses jamma_core (faer) for eigendecomp, JAX for other operations.
  Pure Rust alternative, but has copy overhead limiting max scale.

- rust: Pure Rust LMM (NOT YET IMPLEMENTED).
  Future backend for complete Rust-based LMM computation.
  Selecting this backend will error until implemented.

Backend selection is automatic (preferring jax.numpy) but can be overridden
via the JAMMA_BACKEND environment variable.
"""

import os
import warnings
from functools import cache
from typing import Literal

from loguru import logger

# Canonical backend names (excludes "rust" which is a stub)
Backend = Literal["jax.numpy", "jax.rust"]

# Backward compatibility aliases (old name -> canonical name)
# NOTE: "rust" is NOT here because bare "rust" now means pure Rust LMM (future)
BACKEND_ALIASES: dict[str, str] = {
    "jax": "jax.numpy",  # Old "jax" -> new "jax.numpy"
    "jax.scipy": "jax.numpy",  # Old "jax.scipy" -> new "jax.numpy"
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
        >>> normalize_backend_name("jax.numpy")
        'jax.numpy'
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
    1. JAMMA_BACKEND environment variable override
       - Valid values: 'jax.numpy', 'jax.rust', 'auto'
       - Old names 'jax' and 'jax.scipy' map to 'jax.numpy' with deprecation warning
       - Bare 'rust' errors (pure Rust LMM not yet implemented)
    2. Auto-selection: jax.numpy (preferred due to lower memory overhead)

    Returns:
        Backend identifier ('jax.numpy' or 'jax.rust').

    Raises:
        ValueError: If 'rust' (pure Rust LMM) is selected but not implemented.

    Examples:
        >>> import os
        >>> os.environ["JAMMA_BACKEND"] = "jax.rust"
        >>> get_compute_backend.cache_clear()
        >>> get_compute_backend()
        'jax.rust'
    """
    # Check for environment override
    override = os.environ.get("JAMMA_BACKEND", "").strip()
    if override:
        override = normalize_backend_name(override)

        # Handle pure Rust LMM stub (not yet implemented)
        if override == "rust":
            raise ValueError(
                "Backend 'rust' (pure Rust LMM) is not yet implemented. "
                "Use 'jax.numpy' or 'jax.rust' instead."
            )

        # Handle explicit backend selection
        if override in ("jax.numpy", "jax.rust"):
            logger.debug(f"Backend override via JAMMA_BACKEND={override}")
            return override

        # "auto" falls through to auto-selection below

    # Auto-selection: prefer jax.numpy (lower memory overhead than faer)
    # jax.rust requires extra copies: K -> k_vec -> faer_mat -> U
    # jax.numpy operates directly on the numpy array
    logger.debug("Using jax.numpy backend (default)")
    return "jax.numpy"


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
    """Check if jamma_core is available (enables jax.rust backend).

    The jax.rust backend uses faer (via jamma_core) for eigendecomposition
    instead of scipy. This provides better stability at 100k+ samples.

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
        Dictionary with backend availability and selection info:
        - selected: Currently selected backend ('jax.numpy' or 'jax.rust')
        - rust_available: True if jamma_core is installed (legacy field name)
        - jax_rust_available: True if jamma_core is installed (clearer name)
        - gpu_available: True if JAX can access a GPU
        - override: Value of JAMMA_BACKEND env var, or None
    """
    rust_avail = is_rust_available()
    return {
        "selected": get_compute_backend(),
        "rust_available": rust_avail,
        "jax_rust_available": rust_avail,
        "gpu_available": _has_gpu(),
        "override": os.environ.get("JAMMA_BACKEND", None),
    }
