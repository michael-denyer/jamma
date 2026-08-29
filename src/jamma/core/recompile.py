"""Runtime C extension loading and auto-recompilation.

``_load_c_module(spec, expected_abi)`` is the one seam both C-extension callers
use: it imports the extension named by a ``BuildSpec``, validates its ABI and
core symbols, and rebuilds it once on a stale or missing ``.so`` before falling
back to pure Python. ``jamma.lmm.compute_numpy`` and ``jamma.jlinalg`` used to
each own a copy of that import/ABI/recompile/retry machine; now they call this.

``auto_recompile_c_extension(spec)`` is the rebuild half: it calls
``jamma._build_support.compile_and_link.compile_extension(spec, ...)``,
serialises concurrent callers on a file lock, evicts the stale ``sys.modules``
entry, and returns True on success. ``compile_extension`` ships inside the
wheel, so ABI-mismatch recompile succeeds on wheel installs as it does from a
source checkout. It evicts only the extension module itself, never the parent
package, so nothing here re-enters its own import machinery: the #181
self-deadlock (flock is per open-file-description, so a re-entrant second
acquisition on the same thread blocks forever) cannot recur because there is
no re-import of the parent package to trigger it.

Called from:
    src/jamma/jlinalg/__init__.py  — at import, to load _jlinalg
    src/jamma/lmm/compute_numpy.py — at import, to load _lmm_accel
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from jamma._build_support.compile_and_link import BuildSpec


def _lock_path_for(sys_module_key: str) -> Path:
    """Compute the lock file path for a given extension module.

    Lives next to the .so target so concurrent processes installed against
    the same site-packages serialize on the same lock. Falls back to a
    tempdir-based path keyed on the module name if the package directory
    can't be located (defensive — should not happen in normal installs).
    """
    # Any of these means "the package directory could not be located", which
    # the tempdir path below already handles.
    with contextlib.suppress(ImportError, ValueError, OSError, StopIteration):
        package_name, _, mod_name = sys_module_key.rpartition(".")
        if package_name:
            spec = importlib.util.find_spec(package_name)
            if spec is not None and spec.submodule_search_locations:
                pkg_dir = Path(next(iter(spec.submodule_search_locations)))
                ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
                return pkg_dir / f"{mod_name}{ext_suffix}.lock"
    safe_key = sys_module_key.replace(".", "_").replace("/", "_")
    return Path(tempfile.gettempdir()) / f"jamma_{safe_key}.lock"


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive POSIX file lock, or proceed unlocked when locking is unavailable.

    Uses ``fcntl.flock``. Windows is unreachable — both compile targets refuse
    it — so there is no ``msvcrt`` path. On any ``OSError`` (read-only
    site-packages, or a filesystem without flock such as some network mounts) it
    logs at debug and proceeds unlocked; the atomic ``os.replace`` in
    ``compile_and_link`` still stops readers from observing a partial ``.so``.
    The lock file is created if missing and left in place after release, which
    matches how pip/uv handle their own lockfiles and avoids a TOCTOU between
    unlink and re-lock by a sibling process.
    """
    from loguru import logger

    fd = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as e:
        logger.debug(
            f"recompile lock unavailable on {lock_path} "
            f"(errno={getattr(e, 'errno', '?')}: {e}); proceeding unlocked "
            f"(atomic .so replace is still in effect)"
        )
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        yield
        return

    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def auto_recompile_c_extension(spec: BuildSpec) -> bool:
    """Auto-recompile a C extension when its import or ABI check failed.

    Calls ``compile_extension(spec, ..., on_retry=...)`` directly, evicts the
    stale module from ``sys.modules``, and returns True on success.

    Args:
        spec: The ``BuildSpec`` for the target. Uses ``module_name`` (log name),
            ``sys_module_key`` (the key to evict), and ``fallback_label``.

    Returns:
        True if recompilation succeeded; False otherwise.
    """
    from loguru import logger

    from jamma._build_support.compile_and_link import compile_extension

    module_name = spec.module_name
    sys_module_key = spec.sys_module_key
    label = spec.fallback_label

    logger.info(
        f"C extension {module_name} needs recompilation "
        f"(ABI mismatch or missing). Compiling now..."
    )

    def _on_retry(msg: str) -> None:
        # Surface OMP downgrade (or other retry notices) as a warning so users
        # whose runtime recompile silently falls back to single-threaded can see
        # it. The build-time path in hatch_build.py already warns on OMP
        # downgrade; this closes the gap for ABI-mismatch recompiles on wheels.
        logger.warning(f"{module_name} recompile retry: {msg}")

    # Serialize concurrent recompiles (pytest-xdist workers, parallel Databricks
    # jobs, multiple notebook kernels). Without this, two workers can race on the
    # same .so output path and produce a corrupted file. The atomic os.replace
    # inside compile_and_link is the secondary defense for code paths that bypass
    # this shim (e.g. ``python -m jamma.jlinalg._compile_jlinalg`` invoked twice).
    lock_path = _lock_path_for(sys_module_key)
    with _file_lock(lock_path):
        # Re-check after acquiring the lock — a sibling process may have already
        # recompiled while we were blocked. If the import now succeeds, skip the
        # redundant build.
        #
        # Gate on ``sys_module_key not in sys.modules`` deliberately:
        #   * callers got here via a failed import, so the key is NOT in
        #     sys.modules and we probe for a fresh sibling build.
        #   * a stale cached module (e.g. after a failed importlib.reload) would
        #     make a naked import return the stale object and skip the rebuild
        #     the caller explicitly asked for.
        if sys_module_key not in sys.modules:
            try:
                importlib.import_module(sys_module_key)
                logger.info(
                    f"C extension {module_name} was recompiled by another "
                    f"process; using existing build."
                )
                return True
            except ImportError as e:
                # Still broken — fall through to recompile. Log at debug so a
                # post-mortem can tell "no sibling rebuild happened" from
                # "sibling rebuilt but the new .so also fails to import".
                logger.debug(
                    f"post-lock re-import of {sys_module_key} failed ({e}); "
                    f"proceeding with recompile"
                )

        try:
            success = compile_extension(
                spec,
                Path(__file__).parents[1],  # the installed jamma/ package dir
                on_retry=_on_retry,
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            # Narrow catch: genuine build-environment failures (missing compiler,
            # broken subprocess, the compile driver's own RuntimeError).
            # Programming bugs (AttributeError, KeyError, TypeError) propagate
            # so they surface as real tracebacks instead of a silent
            # pure-Python fallback.
            logger.warning(
                f"Auto-recompilation of {module_name} raised "
                f"{type(e).__name__}: {e}. "
                f"Falling back to pure-Python ({label}).",
                exc_info=True,
            )
            return False

        if not success:
            logger.warning(
                f"Auto-recompilation of {module_name} failed. "
                f"Falling back to pure-Python ({label})."
            )
            return False

        # Evict stale module from sys.modules so re-import picks up the new .so
        sys.modules.pop(sys_module_key, None)

    logger.info(f"C extension {module_name} recompiled successfully.")
    return True


def _import_and_validate(spec: BuildSpec, expected_abi: int) -> ModuleType | None:
    """Import ``spec.sys_module_key`` and validate its ABI and core symbols.

    Returns the module when ABI_VERSION matches ``expected_abi`` and every
    ``spec.required_attrs`` symbol is present, else None. A missing symbol on an
    otherwise ABI-matched build means a corrupt build, so it is treated the same
    as an import failure and drives a rebuild in the caller.
    """
    from loguru import logger

    try:
        mod = importlib.import_module(spec.sys_module_key)
    except ImportError as e:
        logger.warning(
            f"{spec.module_name} not available ({e}) — usually an ABI "
            f"mismatch or a missing build artifact. Falling back to "
            f"pure-Python ({spec.fallback_label})."
        )
        return None

    abi = getattr(mod, "ABI_VERSION", None)
    if abi is None:
        logger.warning(
            f"{spec.module_name} not available: ABI_VERSION missing from the "
            f"compiled module — usually an ABI mismatch. Falling back to "
            f"pure-Python ({spec.fallback_label})."
        )
        return None
    if abi != expected_abi:
        logger.warning(
            f"{spec.module_name} ABI mismatch: compiled={abi}, "
            f"expected={expected_abi}. Stale .so needs recompilation."
        )
        return None

    missing = [name for name in spec.required_attrs if getattr(mod, name, None) is None]
    if missing:
        logger.warning(
            f"{spec.module_name} not available: required symbols missing: "
            f"{missing} — usually an ABI mismatch from a partial build. "
            f"Falling back to pure-Python ({spec.fallback_label})."
        )
        return None

    return mod


def _load_c_module(spec: BuildSpec, expected_abi: int) -> ModuleType | None:
    """Load and validate a C extension, rebuilding once on a stale/missing .so.

    The one seam both C-extension callers use in place of a hand-written
    import/ABI/recompile/retry block. Honours ``JAMMA_FORCE_NUMPY_FALLBACK``:
    when truthy it returns None WITHOUT importing or recompiling, so the
    ASAN/UBSAN sanitizer workflow's ``dlopen`` never runs (RESEARCH §"Pitfall 4":
    ASAN + dlopen inside dispatched BLAS can raise false-positive
    heap-buffer-overflow reports). Otherwise it imports and validates the
    extension, and on failure rebuilds it once and retries.

    Args:
        spec: The ``BuildSpec`` naming the extension to load and rebuild.
        expected_abi: The ABI version the caller was built against. Kept a
            parameter, not a spec field, because the constant lives next to the
            C source's ABI in the caller module.

    Returns:
        The validated extension module, or None to use the pure-Python fallback.
    """
    from jamma.core.constants import env_flag

    if env_flag("JAMMA_FORCE_NUMPY_FALLBACK"):
        return None

    mod = _import_and_validate(spec, expected_abi)
    if mod is not None:
        return mod

    if auto_recompile_c_extension(spec):
        mod = _import_and_validate(spec, expected_abi)
    return mod
