"""Generic C extension auto-recompilation utility."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import sysconfig


def find_c_compiler() -> tuple[str, list[str]] | None:
    """Find a usable C compiler, trying multiple candidates.

    Checks in order:
    1. ``$CC`` environment variable (if set)
    2. ``sysconfig`` configured compiler (what Python was built with)
    3. ``cc``, ``clang``, ``gcc`` as fallbacks

    Each candidate must exist on PATH and respond to ``--version``
    to be considered usable.

    Returns:
        Tuple of (compiler_command, extra_flags) if found, None otherwise.
    """
    import os

    candidates: list[str] = []

    # $CC takes priority
    cc_env = os.environ.get("CC")
    if cc_env:
        candidates.append(cc_env)

    # What Python was built with
    cc_sysconfig = sysconfig.get_config_var("CC")
    if cc_sysconfig and cc_sysconfig not in candidates:
        candidates.append(cc_sysconfig)

    # Common fallbacks
    for fallback in ("cc", "clang", "gcc"):
        if fallback not in candidates:
            candidates.append(fallback)

    for candidate in candidates:
        cmd = candidate.split()[0]
        extra = candidate.split()[1:]
        if not shutil.which(cmd):
            continue
        # Verify the compiler can actually execute
        try:
            probe = subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            if probe.returncode == 0:
                return cmd, extra
        except (OSError, subprocess.TimeoutExpired):
            continue

    return None


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
