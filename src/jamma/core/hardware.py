"""Hardware context collection for benchmark reproducibility.

This module provides utilities for collecting hardware and software context
needed to make benchmark results comparable across machines, and a fail-fast
guard to prevent silent float32 fallback in benchmark entry points.
"""

from __future__ import annotations

import os
import platform

import jax
import numpy as np
import psutil
from threadpoolctl import threadpool_info

from jamma.core.threading import get_blas_thread_count


def get_hardware_context() -> dict[str, str | int | bool | list[str]]:
    """Collect hardware and software context for benchmark reproducibility.

    Gathers CPU, BLAS, JAX, NumPy, platform, and Python version information
    into a JSON-serializable dict. Benchmark results without hardware context
    are not comparable across machines.

    Returns:
        Dictionary with keys:
            - cpu_model: CPU model string.
            - cpu_count_physical: Physical (non-hyperthreaded) core count.
            - cpu_count_logical: Logical core count (includes hyperthreading).
            - blas_backend: BLAS library name ("mkl", "openblas", or "unknown").
            - blas_threads: Current BLAS thread target.
            - jax_version: JAX version string.
            - jax_backend: JAX default backend ("cpu", "gpu", "tpu").
            - jax_device_count: Number of JAX-visible devices.
            - jax_x64_enabled: Whether JAX 64-bit precision is enabled.
            - numpy_version: NumPy version string.
            - platform: Full platform string from platform.platform().
            - python_version: Python version string.

    Example:
        >>> import json
        >>> ctx = get_hardware_context()
        >>> print(json.dumps(ctx, indent=2))  # all values are JSON-serializable
    """
    return {
        "cpu_model": _get_cpu_model(),
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_count_logical": os.cpu_count() or 1,
        "blas_backend": _get_blas_backend(),
        "blas_threads": get_blas_thread_count(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_device_count": len(jax.devices()),
        "jax_x64_enabled": bool(jax.config.jax_enable_x64),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def assert_x64_precision() -> None:
    """Assert JAX 64-bit precision is enabled.

    Benchmark and profiling entry points must call this before any computation
    to prevent silent float32 fallback that would corrupt results.

    Raises:
        RuntimeError: If jax.config.jax_enable_x64 is False.

    Example:
        >>> from jamma.core.jax_config import configure_jax
        >>> configure_jax()
        >>> assert_x64_precision()  # passes silently
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "JAX 64-bit precision is not enabled. "
            "Benchmarks require x64 for GEMMA-equivalent results. "
            "Call jamma.core.jax_config.configure_jax() before running benchmarks."
        )


def _get_cpu_model() -> str:
    """Return the CPU model string.

    On Linux, reads /proc/cpuinfo for the 'model name' field.
    Falls back to platform.processor(), then platform.machine().

    Returns:
        Non-empty CPU model string.
    """
    # Try /proc/cpuinfo on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    _, _, value = line.partition(":")
                    model = value.strip()
                    if model:
                        return model
    except OSError:
        pass

    # Fall back to platform.processor() (useful on macOS, returns "arm" or "i386")
    model = platform.processor()
    if model:
        return model

    # Last resort: architecture string
    return platform.machine() or "unknown"


def _get_blas_backend() -> str:
    """Return the BLAS backend name from threadpool_info.

    Iterates over threadpool entries, returns the internal_api of the first
    entry with user_api == "blas". Returns "unknown" if none found.

    Returns:
        BLAS backend name (e.g., "mkl", "openblas") or "unknown".
    """
    for entry in threadpool_info():
        if entry.get("user_api") == "blas":
            return entry.get("internal_api", "unknown")
    return "unknown"
