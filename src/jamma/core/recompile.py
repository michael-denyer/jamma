"""Runtime C extension auto-recompilation.

When a C extension (e.g. _lmm_accel, _jlinalg) fails to import because
of ABI mismatch or missing .so, ``auto_recompile_c_extension`` invokes
the corresponding compile module (``jamma.lmm._compile_accel`` or
``jamma.jlinalg._compile_jlinalg``), evicts the stale entry from
``sys.modules``, and returns True on success.

Both compile modules import their helpers from
``jamma._build_support`` — which ships inside the installed wheel —
so ABI-mismatch recompile succeeds on wheel installs just as it does
from a source checkout. This shim is deliberately thin: it owns only
the import-retry and error-surfacing contract; all compiler discovery,
flag selection, and OpenMP detection live in ``jamma._build_support``.

Called from:
    src/jamma/jlinalg/__init__.py  — when _jlinalg import fails
    src/jamma/lmm/compute_numpy.py — when _lmm_accel import fails
"""

from __future__ import annotations

import importlib
import subprocess
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

    def _on_retry(msg: str) -> None:
        # Surface OMP downgrade (or other retry notices) as a warning so
        # users whose runtime recompile silently falls back to
        # single-threaded execution can see it. The build-time path in
        # hatch_build.py already warns on OMP downgrade; this closes the
        # gap for ABI-mismatch recompiles on wheel installs.
        logger.warning(f"{module_name} recompile retry: {msg}")

    try:
        try:
            success = compiler.compile_extension(verbose=False, on_retry=_on_retry)
        except TypeError:
            # Older compile_extension without on_retry kwarg — fall back so
            # a partial-upgrade environment doesn't break. Downgrade
            # notices will not surface in this case.
            success = compiler.compile_extension(verbose=False)
    except (OSError, subprocess.SubprocessError, ImportError, RuntimeError) as e:
        # Narrow catch: genuine build-environment failures (missing compiler,
        # broken subprocess, unimportable helper, compile driver's explicit
        # RuntimeError). Programming bugs (AttributeError, KeyError, TypeError
        # other than the on_retry-kwarg one above) propagate so they surface
        # as real tracebacks instead of a silent pure-Python fallback.
        logger.warning(
            f"Auto-recompilation of {module_name} raised "
            f"{type(e).__name__}: {e}. "
            f"Falling back to pure-Python ({label}). "
            f"To diagnose, run: python -m {compiler_module}",
            exc_info=True,
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
