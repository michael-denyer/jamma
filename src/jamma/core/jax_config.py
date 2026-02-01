"""JAX configuration utilities for JAMMA.

This module provides configuration and verification functions for JAX,
ensuring proper setup for numerical computations. JAX is used for kinship
matrix computation and linear mixed model fitting.

IMPORTANT: GEMMA requires 64-bit precision for numerical equivalence.
Default JAX uses 32-bit, so configure_jax() must be called before any
JAX computations to enable x64 mode.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
from loguru import logger


def _pin_blas_threads(n_threads: int = 1) -> None:
    """Pin BLAS thread count to avoid oversubscription with JAX.

    JAX manages its own parallelism; having BLAS libraries spawn additional
    threads causes contention and slowdowns. This sets environment variables
    for common BLAS implementations.

    Must be called BEFORE importing numpy/scipy in the process.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n_threads))  # macOS Accelerate


# Pin threads on module import (before JAX does any work)
_pin_blas_threads(1)


def configure_jax(
    enable_x64: bool = True,
    platform: str | None = None,
    persistent_cache: bool = True,
) -> None:
    """Configure JAX for JAMMA computations.

    This function should be called once at application startup before any
    JAX operations. It configures precision and optionally the compute platform.

    Args:
        enable_x64: Enable 64-bit floating point precision. Required for
            numerical equivalence with GEMMA C++ implementation. Defaults to True.
        platform: Optional platform name ("cpu", "gpu", "tpu"). If None,
            JAX auto-selects the best available platform.
        persistent_cache: Enable XLA compilation cache persistence. Speeds up
            subsequent runs by reusing compiled kernels. Defaults to True.

    Example:
        >>> configure_jax()  # Enable x64, auto-select platform
        >>> configure_jax(platform="cpu")  # Force CPU backend
    """
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
        logger.debug("JAX 64-bit precision enabled")

    if platform is not None:
        jax.config.update("jax_platform_name", platform)
        logger.debug(f"JAX platform set to: {platform}")

    if persistent_cache:
        # Enable XLA compilation cache - reuses compiled kernels across runs
        # Only cache compilations that take >1s to avoid cache bloat
        cache_dir = os.path.expanduser("~/.cache/jax")
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        logger.debug(f"JAX compilation cache enabled: {cache_dir}")

    info = get_jax_info()
    logger.info(
        f"JAX configured: version={info['version']}, "
        f"backend={info['backend']}, devices={len(info['devices'])}"
    )


def get_jax_info() -> dict[str, Any]:
    """Get information about the current JAX configuration.

    Returns a dictionary with JAX version, backend, and available devices.
    Useful for logging and debugging.

    Returns:
        Dictionary with keys:
            - version: JAX version string
            - backend: Current default backend name (cpu/gpu/tpu)
            - devices: List of available device descriptions
            - x64_enabled: Whether 64-bit precision is enabled
    """
    return {
        "version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "x64_enabled": jax.config.jax_enable_x64,
    }


def verify_jax_installation() -> bool:
    """Verify that JAX is properly installed and functional.

    Runs a simple JIT-compiled matrix multiplication to confirm that:
    - JAX imports work
    - XLA compilation works
    - Basic linear algebra operations succeed

    Returns:
        True if verification succeeds.

    Raises:
        RuntimeError: If JAX verification fails, with details about the failure.

    Example:
        >>> configure_jax()
        >>> verify_jax_installation()
        True
    """
    try:
        # Test JIT compilation with matrix multiply
        @jax.jit
        def _matmul_test(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return jnp.matmul(a, b)

        # Create small test matrices
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        # Run JIT-compiled function
        result = _matmul_test(a, b)

        # Verify result shape and approximate values
        expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
        if result.shape != (2, 2):
            raise RuntimeError(f"Unexpected result shape: {result.shape}")

        if not jnp.allclose(result, expected):
            raise RuntimeError(f"Incorrect matmul result: {result}")

        logger.debug("JAX installation verified: JIT compilation and matmul working")
        return True

    except Exception as e:
        error_msg = f"JAX verification failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
