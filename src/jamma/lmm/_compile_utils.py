"""Shared utilities for C extension auto-recompilation.

Provides a single ``auto_recompile_c_extension`` helper used by both the LMM
and eigendecomp subsystems to rebuild stale or missing C extensions at runtime
without duplicating the compiler-invocation and sys.modules eviction logic.
"""

from __future__ import annotations

import importlib
import sys


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
