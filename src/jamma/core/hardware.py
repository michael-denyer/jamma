"""Hardware context collection for benchmark reproducibility.

This module provides utilities for collecting hardware and software context
needed to make benchmark results comparable across machines.
"""

from __future__ import annotations

import os
import platform
from contextlib import suppress
from typing import TypedDict

import numpy as np
import psutil

from jamma.core.threading import get_blas_backend, get_blas_thread_count


class HardwareContext(TypedDict):
    """Hardware and software context for a benchmark run.

    Every value is JSON-serializable. Field docs are on
    :func:`get_hardware_context`.
    """

    cpu_model: str
    cpu_count_physical: int
    cpu_count_logical: int
    blas_backend: str
    blas_threads: int
    numpy_version: str
    platform: str
    python_version: str


def get_hardware_context() -> HardwareContext:
    """Collect hardware and software context for benchmark reproducibility.

    Gathers CPU, BLAS, NumPy, platform, and Python version information
    into a JSON-serializable dict. Benchmark results without hardware context
    are not comparable across machines.

    Returns:
        Dictionary with keys:
            - cpu_model: CPU model string.
            - cpu_count_physical: Physical (non-hyperthreaded) core count.
            - cpu_count_logical: Logical core count (includes hyperthreading).
            - blas_backend: BLAS library name (e.g. "mkl", "openblas",
              "accelerate") or "unknown".
            - blas_threads: Current BLAS thread target.
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
        "blas_backend": get_blas_backend(),
        "blas_threads": get_blas_thread_count(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def _get_cpu_model() -> str:
    """Return the CPU model string.

    On Linux, reads /proc/cpuinfo for the 'model name' field.
    Falls back to platform.processor(), then platform.machine().

    Returns:
        Non-empty CPU model string.
    """
    # Try /proc/cpuinfo on Linux. Absent on macOS and unreadable in some
    # containers, so an OSError just means "fall through to the next source".
    with suppress(OSError), open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                _, _, value = line.partition(":")
                model = value.strip()
                if model:
                    return model

    # Fall back to platform.processor() (useful on macOS, returns "arm" or "i386")
    model = platform.processor()
    if model:
        return model

    # Last resort: architecture string
    return platform.machine() or "unknown"
