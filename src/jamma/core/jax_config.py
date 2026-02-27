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
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import psutil
from loguru import logger

# Suppress unhelpful JAX buffer donation warnings — fires when
# @jit donate_argnums can't reuse a buffer (shape/type mismatch).
warnings.filterwarnings("ignore", message="Some donated buffers were not usable")


def configure_jax(
    enable_x64: bool = True,
    platform: str | None = None,
    persistent_cache: bool = True,
    n_cpu_devices: int | None = None,
) -> None:
    """Configure JAX for JAMMA computations.

    This function should be called once at application startup before any
    JAX operations. It configures precision and optionally the compute platform.

    The ``n_cpu_devices`` setting MUST be applied before any backend-initializing
    call (e.g. ``jax.devices()``, ``jax.default_backend()``). It is therefore set
    as the absolute first operation in this function.

    Args:
        enable_x64: Enable 64-bit floating point precision. Required for
            numerical equivalence with GEMMA C++ implementation. Defaults to True.
        platform: Optional platform name ("cpu", "gpu", "tpu"). If None,
            JAX auto-selects the best available platform.
        persistent_cache: Enable XLA compilation cache persistence. Speeds up
            subsequent runs by reusing compiled kernels. Defaults to True.
        n_cpu_devices: Number of virtual CPU devices for JAX. If None, auto-configures
            as ``max(1, physical_cores // 2)``. Set ``JAMMA_JAX_DEVICES`` env var to
            override. Single device (n=1) leaves JAX default behaviour unchanged.

    Example:
        >>> configure_jax()  # Enable x64, auto-select platform, auto-configure devices
        >>> configure_jax(platform="cpu")  # Force CPU backend
        >>> configure_jax(n_cpu_devices=4)  # Explicitly set 4 virtual CPU devices
    """
    # CRITICAL: jax_num_cpu_devices MUST be set before any backend-initializing call.
    # This must be the absolute first operation in configure_jax().
    _configure_cpu_devices(n_cpu_devices)

    if enable_x64:
        jax.config.update("jax_enable_x64", True)
        logger.debug("JAX 64-bit precision enabled")

    if platform is not None:
        jax.config.update("jax_platform_name", platform)
        logger.debug(f"JAX platform set to: {platform}")

    if persistent_cache:
        # Enable XLA compilation cache - reuses compiled kernels across runs
        # Only cache compilations that take >1s to avoid cache bloat
        cache_dir = JAX_CACHE_DIR
        try:
            os.makedirs(cache_dir, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
            logger.debug(f"JAX compilation cache enabled: {cache_dir}")
        except OSError as e:
            logger.info(
                f"Could not create JAX cache dir {cache_dir}: {e}. "
                f"JIT compilation will not be cached across runs."
            )

    global _jax_configured
    _jax_configured = True

    info = get_jax_info()
    logger.info(
        f"JAX configured: version={info['version']}, "
        f"backend={info['backend']}, devices={info['n_cpu_devices']}"
    )


def _configure_cpu_devices(n_cpu_devices: int | None) -> None:
    """Set jax_num_cpu_devices before any backend-initializing call.

    Priority:
    1. JAMMA_JAX_DEVICES env var (explicit override)
    2. n_cpu_devices argument (explicit caller override)
    3. Auto-config: max(1, physical_cores // 2)

    Only calls jax.config.update when n > 1 — single device leaves JAX
    default behaviour unchanged.

    Args:
        n_cpu_devices: Caller-requested device count, or None for auto-config.

    Raises:
        ValueError: If JAMMA_JAX_DEVICES is set to a non-integer or non-positive
            value, or if n_cpu_devices is less than 1.
    """
    env_override = os.environ.get("JAMMA_JAX_DEVICES")
    if env_override is not None:
        try:
            n = int(env_override.strip())
        except ValueError as err:
            raise ValueError(
                f"JAMMA_JAX_DEVICES={env_override!r} is not a valid integer. "
                "Set to a positive integer or unset to use auto-config."
            ) from err
        if n < 1:
            raise ValueError(
                f"JAMMA_JAX_DEVICES={n} must be >= 1. "
                "Set to a positive integer or unset to use auto-config."
            )
        if n > 1:
            jax.config.update("jax_num_cpu_devices", n)
        logger.debug(f"JAX CPU devices from JAMMA_JAX_DEVICES: {n}")
        return

    if n_cpu_devices is not None:
        if n_cpu_devices < 1:
            raise ValueError(
                f"n_cpu_devices={n_cpu_devices} must be >= 1. "
                "Use None for auto-configuration."
            )
        if n_cpu_devices > 1:
            jax.config.update("jax_num_cpu_devices", n_cpu_devices)
        logger.debug(f"JAX CPU devices from argument: {n_cpu_devices}")
        return

    # Auto-configure: half the physical cores, at least 1
    physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    n = max(1, physical_cores // 2)
    if n > 1:
        jax.config.update("jax_num_cpu_devices", n)
    logger.debug(
        f"JAX CPU devices auto-configured: {n} (physical_cores={physical_cores})"
    )


_jax_configured = False

# Cache directory used by configure_jax() — exposed as constant so tests
# can verify the path without hardcoding it.
JAX_CACHE_DIR = os.path.expanduser("~/.cache/jax")


def is_jax_configured() -> bool:
    """Return whether configure_jax() has been called.

    Used by other modules (e.g. threading) to guard against premature
    JAX backend initialization.
    """
    return _jax_configured


def ensure_jax_configured(
    enable_x64: bool = True,
    platform: str | None = None,
) -> None:
    """Configure JAX for 64-bit precision. Idempotent -- safe to call multiple times.

    Raises RuntimeError if called with non-default arguments after JAX has already
    been configured, since the configuration is locked after first call.

    Device count auto-configuration is applied on first call only.

    Args:
        enable_x64: Enable 64-bit precision (default True).
        platform: JAX platform override (default None = auto-detect).

    Raises:
        RuntimeError: If called with non-default args after JAX is already configured.
    """
    global _jax_configured
    if _jax_configured:
        if not enable_x64 or platform is not None:
            raise RuntimeError(
                "ensure_jax_configured() called with non-default args after "
                "JAX already configured. JAX configuration is locked after "
                f"first call. Requested: enable_x64={enable_x64}, "
                f"platform={platform}. Call configure_jax() before any "
                "other JAMMA operations to override defaults."
            )
        return
    configure_jax(enable_x64=enable_x64, platform=platform)


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
            - n_cpu_devices: Number of active CPU devices
    """
    cpu_devices = jax.devices("cpu")
    return {
        "version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "x64_enabled": jax.config.jax_enable_x64,
        "n_cpu_devices": len(cpu_devices),
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

        # Verify result values
        expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
        if not jnp.allclose(result, expected):
            raise RuntimeError(f"Incorrect matmul result: {result}")

        logger.debug("JAX installation verified: JIT compilation and matmul working")
        return True

    except Exception as e:
        logger.error(f"JAX verification failed: {type(e).__name__}: {e}")
        raise
