"""Shared utilities for C extension detection and auto-recompilation.

Provides ``is_c_extension_usable`` for lightweight availability probes and
``auto_recompile_c_extension`` for auto-rebuilding stale/missing C extensions
at runtime.
"""

from __future__ import annotations

import importlib
import sys


def is_c_extension_usable() -> bool:
    """Check whether the LMM C extension is importable and has a valid ABI.

    This is the single source of truth for C extension availability checks.
    Both ``pipeline._auto_select_backend`` and ``compute_numpy._try_import_accel``
    ultimately depend on import + ABI validation — this function consolidates
    the lightweight probe used by the auto-backend selector.

    Returns:
        True if _lmm_accel imports successfully and exposes ABI_VERSION.
    """
    from loguru import logger

    try:
        mod = importlib.import_module("jamma.lmm._lmm_accel")
    except ImportError:
        logger.debug("C extension _lmm_accel not importable")
        return False
    except (OSError, AttributeError) as e:
        logger.warning(
            f"C extension _lmm_accel not usable: {type(e).__name__}: {e}. "
            "Run: python -m jamma.lmm._compile_accel"
        )
        return False

    if not hasattr(mod, "ABI_VERSION"):
        logger.debug(
            "C extension imported but missing ABI_VERSION — "
            "likely stale; run: python -m jamma.lmm._compile_accel"
        )
        return False

    return True


def auto_recompile_c_extension(
    module_name: str,
    compiler_module: str,
    sys_module_key: str,
    label: str,
) -> bool:
    """Auto-recompile a C extension when import fails.

    Imports the compiler module, invokes ``compile_extension(verbose=False)``,
    evicts the stale module from ``sys.modules``, and returns True on success.
    Handles ``ImportError`` from the compiler module gracefully (returns False).

    Args:
        module_name: Human-readable name for log messages (e.g. "_lmm_accel").
        compiler_module: Dotted import path to the compile module
            (e.g. "jamma.lmm._compile_accel").
        sys_module_key: sys.modules key to evict after recompilation
            (e.g. "jamma.lmm._lmm_accel").
        label: Context label for log messages (e.g. "LMM" or "eigendecomp").

    Returns:
        True if recompilation succeeded; False otherwise.
    """
    from loguru import logger

    try:
        compiler = importlib.import_module(compiler_module)
    except ImportError:
        logger.debug(
            f"{compiler_module} not available — auto-recompilation of "
            f"{module_name} not possible"
        )
        return False

    logger.info(
        f"C extension {module_name} needs recompilation "
        f"(ABI mismatch or missing). Compiling now..."
    )

    try:
        success = compiler.compile_extension(verbose=False)
    except Exception as e:
        logger.warning(
            f"Auto-recompilation of {module_name} raised "
            f"{type(e).__name__}: {e}. "
            f"Falling back to pure-Python ({label}). "
            f"To diagnose, run: python -m {compiler_module}"
        )
        return False

    if not success:
        logger.warning(
            f"Auto-recompilation of {module_name} failed. "
            f"Falling back to pure-Python ({label}). "
            f"To diagnose, run: python -m {compiler_module}"
        )
        return False

    # Evict stale module from sys.modules so re-import picks up the new .so
    sys.modules.pop(sys_module_key, None)

    logger.info(f"C extension {module_name} recompiled successfully.")
    return True
