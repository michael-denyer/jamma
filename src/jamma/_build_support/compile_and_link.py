"""Composition root for JAMMA's shared native-extension build pipeline.

The public imports remain here for runtime rebuild compatibility. Build policy
lives in ``build_models`` and compiler execution in ``build_execution``.
"""

from __future__ import annotations

import os
import shutil
import sysconfig
import tempfile
from pathlib import Path

from .build_execution import Toolchain, detect_toolchain, execute_build
from .build_models import (
    JLINALG_SPEC,
    LMM_ACCEL_SPEC,
    BuildReport,
    BuildResult,
    BuildSpec,
    resolve_flags,
)

#: Import build policy and execution internals from their home modules rather
#: than adding a re-export here.
__all__ = (
    "JLINALG_SPEC",
    "LMM_ACCEL_SPEC",
    "BuildReport",
    "BuildResult",
    "BuildSpec",
    "Toolchain",
    "compile_extension",
    "detect_toolchain",
    "run_build",
)


def run_build(
    spec: BuildSpec,
    package_dir: Path,
    toolchain: Toolchain,
    *,
    dev_mode: bool,
    report: BuildReport,
) -> BuildResult:
    """Check the spec's sources exist, resolve its flags, and build it.

    Prints nothing about the outcome: a preflight failure returns
    ``BuildResult(phase="preflight", ...)`` and each caller words it for its
    own path (a pure-Python fallback for the wheel, an error in dev mode).
    """
    pkg_dir = package_dir.joinpath(*spec.package_parts)
    src_dir = pkg_dir.joinpath(*spec.source_parts)

    missing = [str(src_dir / n) for n in spec.sources if not (src_dir / n).exists()]
    if missing:
        return BuildResult(
            phase="preflight",
            error=(
                f"C source files missing: {missing}. If building from sdist, "
                "verify the archive is complete"
            ),
        )

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_path = pkg_dir / f"{spec.output_stem}{ext_suffix}"

    include_dirs = [toolchain.python_inc, toolchain.numpy_inc]
    include_dirs.extend(str(pkg_dir.joinpath(*parts)) for parts in spec.include_parts)

    flags = resolve_flags(
        spec, dev_mode=dev_mode, system=toolchain.system, env=os.environ
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{spec.output_stem.lstrip('_')}_build_"))
    try:
        return execute_build(
            spec, src_dir, include_dirs, toolchain, flags, out_path, tmp_dir, report
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def compile_extension(spec: BuildSpec, package_dir: Path, report: BuildReport) -> bool:
    """Detect the toolchain and drive ``run_build`` for one ``BuildSpec``.

    Proving the new ``.so`` loads, and evicting any stale module from
    ``sys.modules``, is left to the caller.

    Failures go to ``report.warn`` and the success summary to
    ``report.detail``.
    """
    toolchain = detect_toolchain(report)
    if isinstance(toolchain, str):
        report.warn(f"ERROR: {spec.output_stem} compilation failed: {toolchain}")
        return False

    result = run_build(spec, package_dir, toolchain, dev_mode=True, report=report)
    if not result.ok:
        report.warn(f"ERROR: {spec.output_stem} compilation failed: {result.error}")
        return False

    omp_status = "OpenMP" if result.used_openmp else "single-threaded"
    report.detail(f"{spec.output_stem} compiled: {result.output_path} ({omp_status})")
    return True
