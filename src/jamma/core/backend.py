"""Backend detection and information for JAMMA.

JAMMA supports two compute backends:
- "jax": JAX-accelerated pipeline (GPU/multi-device support, default when JAX installed)
- "numpy": Pure-NumPy pipeline (works everywhere, no JAX required)

Backend selection priority:
1. JAMMA_BACKEND environment variable (overrides all)
2. `requested` argument to detect_backend()
3. Auto-detection: JAX if importable, else NumPy
"""

from __future__ import annotations

import functools
import os
from typing import Literal

from loguru import logger

BackendRequest = Literal["auto", "jax", "numpy"]
BackendResolved = Literal["jax", "numpy"]


@functools.cache
def has_jax() -> bool:
    """Check whether JAX is importable.

    Used by __init__.py modules to gate JAX-dependent imports. Cached
    after the first call (import success/failure won't change at runtime).
    """
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False
    except (RuntimeError, OSError) as e:
        # JAX installed but broken (e.g. bad CUDA, corrupted install)
        logger.error(
            f"JAX installed but failed to import: {type(e).__name__}: {e}. "
            f"Falling back to NumPy backend. Fix the JAX installation or "
            f"use --backend numpy to suppress this warning."
        )
        return False


def detect_backend(requested: BackendRequest = "auto") -> BackendResolved:
    """Detect or validate the compute backend.

    Checks the JAMMA_BACKEND environment variable first; it overrides the
    `requested` argument.  Validates the final selection and raises ValueError
    for unknown backend names.

    Args:
        requested: Requested backend — "auto", "jax", or "numpy".  "auto"
            returns "jax" when JAX is importable, "numpy" otherwise.

    Returns:
        "jax" or "numpy".

    Raises:
        ValueError: If `requested` (or JAMMA_BACKEND) is not in
            ("auto", "jax", "numpy"), or if "jax" is explicitly requested
            but JAX is not installed.

    Example:
        >>> detect_backend("auto")
        'jax'
        >>> detect_backend("numpy")
        'numpy'
    """
    env_override = os.environ.get("JAMMA_BACKEND")
    effective = env_override if env_override is not None else requested

    valid = ("auto", "jax", "numpy")
    if effective not in valid:
        raise ValueError(
            f"Unknown backend {effective!r}. Must be one of {valid}. "
            "Set JAMMA_BACKEND or pass requested= explicitly."
        )

    if effective == "numpy":
        return "numpy"

    if effective == "jax":
        if not has_jax():
            raise ValueError(
                "Backend 'jax' was explicitly requested but JAX is not installed. "
                "Install JAX with: pip install jamma[jax]"
            )
        return "jax"

    # effective == "auto"
    return "jax" if has_jax() else "numpy"


def log_backend_selection(active: BackendResolved, requested: BackendRequest) -> None:
    """Log the selected backend at INFO level.

    Args:
        active: The backend that was selected ("jax" or "numpy").
        requested: The originally requested backend ("auto", "jax", or "numpy").

    Example:
        >>> log_backend_selection("numpy", "auto")
        # logs: "Backend: numpy"
        >>> log_backend_selection("jax", "jax")
        # logs: "Backend: jax (explicitly requested)"
    """
    env_override = os.environ.get("JAMMA_BACKEND")
    if env_override:
        logger.info(f"Backend: {active} (from JAMMA_BACKEND)")
    elif requested != "auto":
        logger.info(f"Backend: {active} (explicitly requested)")
    elif active == "numpy":
        logger.info(
            f"Backend: {active} (JAX not installed; "
            "install with: pip install jamma[jax])"
        )
    else:
        logger.info(f"Backend: {active}")


def get_backend_info() -> dict[str, str | bool]:
    """Get information about the compute backend.

    Returns:
        Dictionary with backend info:
        - selected: Backend name ("jax" or "numpy")
        - jax_available: True if JAX is importable
    """
    return {
        "selected": detect_backend(),
        "jax_available": has_jax(),
    }
