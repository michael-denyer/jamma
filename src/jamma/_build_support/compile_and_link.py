"""Composition root for JAMMA's shared native-extension build pipeline.

The public imports remain here for runtime rebuild compatibility. Build policy
lives in ``build_models`` and compiler execution in ``build_execution``.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from .build_execution import Toolchain, detect_toolchain, execute_build
from .build_models import (
    JLINALG_SPEC,
    LINK_FLAGS_BY_PLATFORM,
    LMM_ACCEL_SPEC,
    BuildSpec,
    apply_sanitizer_overrides,
    resolve_build_spec,
)

#: The composition root's own surface: the two drivers, the two specs they are
#: driven with, and the toolchain type/detector the wheel backend needs. Build
#: policy constants live in ``build_models`` and execution internals in
#: ``build_execution``; import those from their home module rather than adding
#: a re-export here. ``hatch_build.py`` reads exactly ``LMM_ACCEL_SPEC``,
#: ``JLINALG_SPEC``, ``run_build`` and ``detect_toolchain`` off this module,
#: and ``core.recompile`` monkeypatches ``compile_extension`` on it.
__all__ = (
    "JLINALG_SPEC",
    "LMM_ACCEL_SPEC",
    "BuildResult",
    "BuildSpec",
    "Toolchain",
    "compile_extension",
    "detect_toolchain",
    "run_build",
)

# ---------------------------------------------------------------------------
# run_build — the twelve preflight+compile steps, once, driven by a BuildSpec
# ---------------------------------------------------------------------------


BuildPhase = Literal["preflight", "build", "ok"]


@dataclass(frozen=True)
class BuildResult:
    """Result of ``run_build``. One type, no ``Optional`` field a caller reads.

    ``phase`` says how far the build got: ``"preflight"`` for a guard firing
    before any source was touched (no numpy, no compiler, missing
    headers/sources, Windows — the wheel path turns this into a pure-Python
    fallback, the dev path treats it as an error), ``"build"`` for a compile,
    link, or atomic-publish failure inside ``execute_build``, and ``"ok"`` for
    success. The compile/link/publish distinction is not modelled here because
    no caller branches on it: both consumers print ``error``, which already
    names the failing stage. ``error`` is ``""`` on success and a
    human-readable message otherwise — callers read it directly, with no
    ``result.error if result else "unknown"`` guard. ``output_path`` is set
    only when ``phase == "ok"``.
    """

    phase: BuildPhase
    error: str = ""
    output_path: Path | None = None
    used_openmp: bool = False
    used_openmp_link: bool = False

    @property
    def ok(self) -> bool:
        return self.phase == "ok"

    @property
    def skipped(self) -> bool:
        """True for a preflight guard — no compiler, no numpy, etc.

        Distinguishes "nothing was attempted" from "compile or link failed",
        which the wheel build reports differently (a warning + silent
        pure-Python fallback vs. an error naming the failed phase).
        """
        return self.phase == "preflight"


def run_build(
    spec: BuildSpec,
    package_dir: Path,
    toolchain: Toolchain,
    *,
    dev_mode: bool,
    diagnose: bool = False,
    on_retry: Callable[[str], None] | None = None,
    verbose_print: Callable[..., None] = print,
    error_print: Callable[..., None] | None = None,
    env: dict[str, str] | os._Environ[str] | None = None,
) -> BuildResult:
    """Run the spec-specific preflight checks and two-phase compile.

    Toolchain detection (compiler, Python/NumPy includes, OpenMP flags) has
    already happened once in ``toolchain`` — the caller detects it via
    ``detect_toolchain()`` and reuses it across every ``BuildSpec`` it builds
    in this process. What remains here is spec-specific: the sources-exist
    check, the ``EXT_SUFFIX``/output path, ``resolve_build_spec``, sanitizer
    overrides, platform link flags, and the ``execute_build`` call under a
    temp dir.

    Preflight failures print through ``error_print`` — as ``WARNING`` and a
    pure-Python-fallback note under the wheel build, as ``ERROR`` in dev mode —
    and return ``BuildResult(phase="preflight", ...)``. Platform link flags are
    taken once from ``LINK_FLAGS_BY_PLATFORM``; the wheel path no longer adds a
    second ``-undefined dynamic_lookup`` on macOS.
    """
    resolved_env = os.environ if env is None else env
    if error_print is None:
        error_print = verbose_print
    lead = "WARNING" if not dev_mode else "ERROR"
    tail = " (pure-Python fallback)." if not dev_mode else ""

    def _skip(message: str) -> BuildResult:
        error_print(f"{lead}: {message}{tail}")
        return BuildResult(phase="preflight", error=message)

    pkg_dir = package_dir.joinpath(*spec.package_parts)
    src_dir = pkg_dir.joinpath(*spec.source_parts)
    sources = [src_dir / name for name in spec.sources]
    lapack_sources = [src_dir / name for name in spec.lapack_sources]

    missing = [str(s) for s in sources if not s.exists()]
    if missing:
        return _skip(
            f"C source files missing: {missing}. If building from sdist, "
            "verify the archive is complete"
        )

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_name = f"{spec.output_stem}{ext_suffix}"
    out_path = pkg_dir / out_name

    include_dirs = [toolchain.python_inc, toolchain.numpy_inc]
    include_dirs.extend(str(pkg_dir.joinpath(*parts)) for parts in spec.include_parts)

    diag_flags = (
        toolchain.diagnose_flags() if spec.supports_diagnose and diagnose else ()
    )
    base_extras = resolve_build_spec(
        spec, dev_mode=dev_mode, env=resolved_env, diagnose_flags=diag_flags
    )

    # -lm is the universal extra link flag; apply_sanitizer_overrides is a no-op
    # unless JAMMA_SANITIZE is set, and also instruments the LAPACK sources.
    extra_cflags, extra_link_flags, extra_lapack_cflags = apply_sanitizer_overrides(
        list(base_extras), ["-lm"]
    )

    # Platform link flags taken ONCE — the macOS -undefined dynamic_lookup that
    # the wheel path used to append twice is now a single copy for every caller.
    ldflags = list(LINK_FLAGS_BY_PLATFORM.get(toolchain.system, ()))

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{spec.output_stem.lstrip('_')}_build_"))
    try:
        result = execute_build(
            sources=sources,
            lapack_sources=lapack_sources,
            include_dirs=include_dirs,
            cc_cmd=toolchain.cc_cmd,
            cc_extra=list(toolchain.cc_extra),
            omp_compile=list(toolchain.omp_compile),
            omp_link=list(toolchain.omp_link),
            ldflags=ldflags,
            output=out_path,
            tmp_dir=tmp_dir,
            extra_cflags=extra_cflags,
            extra_link_flags=extra_link_flags,
            extra_lapack_cflags=extra_lapack_cflags,
            on_retry=on_retry,
            verbose_print=verbose_print,
            error_print=error_print,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not result.success:
        return BuildResult(
            phase="build",
            error=result.error or "unknown",
            used_openmp=result.used_openmp,
            used_openmp_link=result.used_openmp_link,
        )

    return BuildResult(
        phase="ok",
        output_path=out_path,
        used_openmp=result.used_openmp,
        used_openmp_link=result.used_openmp_link,
    )


def compile_extension(
    spec: BuildSpec,
    package_dir: Path,
    *,
    verbose: bool = False,
    diagnose: bool = False,
    on_retry: Callable[[str], None] | None = None,
    out: TextIO | None = None,
) -> bool:
    """Detect the toolchain and drive ``run_build`` for one ``BuildSpec``.

    The one dev-mode compile entry point both ``_lmm_accel`` and ``_jlinalg``
    call. Evicts only ``spec.sys_module_key`` from ``sys.modules`` on success
    — never the parent package — so callers relying on a stale ``jamma.lmm``
    or ``jamma.jlinalg`` import are not affected and the module re-execution
    that caused the #181 self-deadlock cannot happen. Whether the freshly
    built ``.so`` actually loads is left to the caller: ``python -m`` shims
    prove it in a fresh subprocess, and ``auto_recompile_c_extension`` proves
    it by calling ``_import_and_validate`` on the current process immediately
    after.

    Args:
        spec: The ``BuildSpec`` to build (``LMM_ACCEL_SPEC`` / ``JLINALG_SPEC``).
        package_dir: The installed ``jamma/`` package directory.
        verbose: Print per-command compile details. When False (default), only
            errors and a one-line summary print.
        diagnose: Emit compiler vectorization reports (accel only; ignored when
            ``spec.supports_diagnose`` is False).
        on_retry: Optional callback invoked with a message when the build
            retries without OpenMP. Defaults to the same output stream as
            everything else this function prints.
        out: Stream to print to. Defaults to ``sys.stderr``.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    stream = sys.stderr if out is None else out

    def _say(*args: object) -> None:
        print(*args, file=stream, flush=True)

    def _detail(*args: object) -> None:
        if verbose:
            _say(*args)

    def _retry(msg: str) -> None:
        if on_retry is not None:
            on_retry(msg)
        else:
            _say(msg)

    toolchain = detect_toolchain(verbose_print=_detail, error_print=_say)
    if isinstance(toolchain, str):
        _say(f"ERROR: {spec.output_stem} compilation failed: {toolchain}")
        return False

    result = run_build(
        spec,
        package_dir,
        toolchain,
        dev_mode=True,
        diagnose=diagnose,
        on_retry=_retry,
        verbose_print=_detail,
        error_print=_say,
    )
    if not result.ok:
        if not result.skipped:
            _say(f"ERROR: {spec.output_stem} compilation failed: {result.error}")
        return False

    sys.modules.pop(spec.sys_module_key, None)

    omp_status = "OpenMP" if result.used_openmp else "single-threaded"
    _say(f"{spec.output_stem} compiled: {result.output_path} ({omp_status})")
    return True
