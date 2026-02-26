"""Backend detection and information for JAMMA.

JAMMA supports two compute backends:
- "jax": JAX-accelerated pipeline (GPU/multi-device support, default when JAX installed)
- "numpy": Pure-NumPy pipeline (works everywhere, no JAX required)

Backend selection priority:
1. JAMMA_BACKEND environment variable (overrides all)
2. `requested` argument to detect_backend()
3. Auto-detection: JAX if importable, else NumPy
"""

import os

from loguru import logger


def detect_backend(requested: str = "auto") -> str:
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
        try:
            import jax  # noqa: F401
        except ImportError as err:
            raise ValueError(
                "Backend 'jax' was explicitly requested but JAX is not installed. "
                "Install JAX with: pip install jamma[jax]"
            ) from err
        return "jax"

    # effective == "auto"
    try:
        import jax  # noqa: F401

        return "jax"
    except ImportError:
        return "numpy"


def log_backend_selection(active: str, requested: str) -> None:
    """Log the selected backend at INFO level.

    Args:
        active: The backend that was selected ("jax" or "numpy").
        requested: The originally requested backend ("auto", "jax", or "numpy").

    Example:
        >>> log_backend_selection("numpy", "auto")
        # logs: "Backend: numpy"
        >>> log_backend_selection("numpy", "jax")
        # logs: "Backend: numpy (forced)"
    """
    if requested != "auto" and requested != active:
        logger.info(f"Backend: {active} (forced)")
    else:
        logger.info(f"Backend: {active}")


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
        logger.warning(
            f"GPU detection failed ({type(e).__name__}: {e}). "
            f"Falling back to CPU. Check JAX/CUDA installation."
        )
        return False


def get_backend_info() -> dict:
    """Get information about the compute backend.

    Returns:
        Dictionary with backend info:
        - selected: Backend name ("jax" or "numpy")
        - gpu_available: True if JAX can access a GPU
    """
    return {
        "selected": detect_backend(),
        "gpu_available": _has_gpu(),
    }
