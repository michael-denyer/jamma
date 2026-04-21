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

import contextlib
import importlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Iterator
from pathlib import Path


def _lock_path_for(sys_module_key: str) -> Path:
    """Compute the lock file path for a given extension module.

    Lives next to the .so target so concurrent processes installed against
    the same site-packages serialize on the same lock. Falls back to a
    tempdir-based path keyed on the module name if the package directory
    can't be located (defensive — should not happen in normal installs).
    """
    try:
        package_name, _, mod_name = sys_module_key.rpartition(".")
        if package_name:
            spec = importlib.util.find_spec(package_name)
            if spec is not None and spec.submodule_search_locations:
                pkg_dir = Path(next(iter(spec.submodule_search_locations)))
                ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
                return pkg_dir / f"{mod_name}{ext_suffix}.lock"
    except (ImportError, ValueError, OSError, StopIteration):
        pass
    safe_key = sys_module_key.replace(".", "_").replace("/", "_")
    return Path(tempfile.gettempdir()) / f"jamma_{safe_key}.lock"


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Cross-platform exclusive file lock.

    Uses fcntl.flock on POSIX and msvcrt.locking on Windows. Falls back to
    a no-op if neither is importable (e.g. exotic platforms) — better to
    risk a rare race than to refuse to recompile at all. The lock file is
    created if missing and left in place after release; this matches how
    pip/uv handle their own lockfiles and avoids a TOCTOU between unlink
    and re-lock by a sibling process.
    """
    from loguru import logger

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Read-only site-packages (system Python on locked-down hosts).
        # Skip locking — a pure-Python fallback is preferable to crashing.
        logger.debug(
            f"recompile lock: cannot create {lock_path.parent} "
            f"(errno={getattr(e, 'errno', '?')}: {e}); proceeding unlocked"
        )
        yield
        return

    fd = None
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        logger.debug(
            f"recompile lock: cannot open {lock_path} "
            f"(errno={getattr(e, 'errno', '?')}: {e}); proceeding unlocked"
        )
        yield
        return

    try:
        if sys.platform == "win32":
            try:
                import msvcrt
            except ImportError as e:
                # Broken Python install on Windows — msvcrt is stdlib. Distinct
                # from filesystem limitation, surface as warning.
                logger.warning(
                    f"recompile lock: msvcrt unavailable ({e}); "
                    f"concurrent recompiles on this interpreter are unserialized"
                )
                with contextlib.suppress(OSError):
                    os.close(fd)
                yield
                return

            # Blocking exclusive lock on the first byte; the entire file is
            # zero-length so this is effectively whole-file.
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            try:
                import fcntl
            except ImportError as e:
                # Exotic Python build without fcntl — distinct from filesystem
                # limitation; surface as warning.
                logger.warning(
                    f"recompile lock: fcntl unavailable ({e}); "
                    f"concurrent recompiles on this interpreter are unserialized"
                )
                with contextlib.suppress(OSError):
                    os.close(fd)
                yield
                return

            fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as e:
        # Locking unsupported on this filesystem (some network mounts) —
        # proceed unlocked. The atomic-replace in compile_and_link still
        # protects readers from observing a partial .so.
        logger.debug(
            f"recompile lock: flock/locking failed on {lock_path} "
            f"(errno={getattr(e, 'errno', '?')}: {e}); proceeding unlocked "
            f"(atomic .so replace is still in effect)"
        )
        with contextlib.suppress(OSError):
            os.close(fd)
        yield
        return

    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        with contextlib.suppress(OSError):
            os.close(fd)


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

    # Serialize concurrent recompiles (pytest-xdist workers, parallel
    # Databricks jobs, multiple notebook kernels). Without this, two
    # workers can race on the same .so output path and produce a
    # corrupted file. The atomic os.replace inside compile_and_link is
    # the secondary defense for code paths that bypass this shim
    # (e.g. ``python -m jamma.jlinalg._compile_jlinalg`` invoked twice).
    lock_path = _lock_path_for(sys_module_key)
    with _file_lock(lock_path):
        # Re-check after acquiring the lock — a sibling process may have
        # already recompiled while we were blocked. If the import now
        # succeeds, skip the redundant build.
        if sys_module_key not in sys.modules:
            try:
                importlib.import_module(sys_module_key)
                logger.info(
                    f"C extension {module_name} was recompiled by another process; "
                    f"using existing build."
                )
                return True
            except ImportError as e:
                # Still broken — fall through to recompile. Log at debug so
                # a post-mortem can distinguish "no sibling rebuild happened"
                # from "sibling rebuilt but new .so also fails to import"
                # (e.g. the sibling linked against a different numpy ABI).
                logger.debug(
                    f"post-lock re-import of {sys_module_key} failed ({e}); "
                    f"proceeding with recompile"
                )

        try:
            # Inspect signature to decide whether compile_extension accepts
            # on_retry — safer than catching TypeError, which would also
            # swallow genuine TypeErrors raised inside compile_extension
            # (bad cast, bad subprocess arg, etc.) and mask real bugs.
            import inspect

            try:
                compile_sig = inspect.signature(compiler.compile_extension)
                supports_on_retry = "on_retry" in compile_sig.parameters or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in compile_sig.parameters.values()
                )
            except (TypeError, ValueError):
                # Signature introspection failed — assume modern API and let
                # any real TypeError propagate rather than retrying silently.
                supports_on_retry = True

            if supports_on_retry:
                success = compiler.compile_extension(verbose=False, on_retry=_on_retry)
            else:
                # Legacy compile_extension without on_retry kwarg — partial
                # upgrade environment. Downgrade notices will not surface.
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
