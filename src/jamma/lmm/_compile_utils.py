"""Runtime probes for the ``_lmm_accel`` C extension.

Provides lightweight availability checks used by the LMM runners to decide
between the C-accelerated and pure-NumPy code paths.
"""

from __future__ import annotations

import importlib

__all__ = [
    "get_c_extension_capabilities",
    "is_c_extension_usable",
]


def get_c_extension_capabilities() -> tuple[bool, bool]:
    """Return `_lmm_accel` availability and OpenMP capability.

    Returns:
        Tuple ``(available, has_openmp)``. ``has_openmp`` is False when the
        extension is unavailable, stale, or missing the capability flag.
    """
    from loguru import logger

    try:
        mod = importlib.import_module("jamma.lmm._lmm_accel")
    except ImportError:
        logger.debug("C extension _lmm_accel not importable")
        return False, False
    except (OSError, AttributeError) as e:
        logger.warning(
            f"C extension _lmm_accel not usable: {type(e).__name__}: {e}. "
            "Run: python -m jamma.lmm._compile_accel"
        )
        return False, False

    if not hasattr(mod, "ABI_VERSION"):
        logger.debug(
            "C extension imported but missing ABI_VERSION — "
            "likely stale; run: python -m jamma.lmm._compile_accel"
        )
        return False, False

    return True, bool(getattr(mod, "HAS_OPENMP", False))


def is_c_extension_usable() -> bool:
    """Convenience wrapper: return only the availability flag.

    Delegates to ``get_c_extension_capabilities()`` and discards the
    OpenMP flag.

    Returns:
        True if _lmm_accel imports successfully and exposes ABI_VERSION.
    """
    available, _has_openmp = get_c_extension_capabilities()
    return available
