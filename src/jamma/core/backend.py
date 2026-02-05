"""Backend information for JAMMA.

JAMMA uses a single backend: JAX pipeline with numpy/LAPACK eigendecomposition.
Uses numpy.linalg.eigh for eigendecomp (via MKL when available), JAX for
vectorized SNP processing.
"""

from loguru import logger


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


def get_backend_info() -> dict:
    """Get information about the compute backend.

    Returns:
        Dictionary with backend info:
        - selected: Backend name ('jax.numpy')
        - gpu_available: True if JAX can access a GPU
    """
    return {
        "selected": "jax.numpy",
        "gpu_available": _has_gpu(),
    }
