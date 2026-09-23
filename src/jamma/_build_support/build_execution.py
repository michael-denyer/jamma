"""Host toolchain detection and atomic compile/link execution."""

from __future__ import annotations

import contextlib
import os
import platform as _platform
import subprocess
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from .build_models import SHARED_LINK_FLAGS as _SHARED_LINK_FLAGS
from .build_models import (
    BuildReport,
    BuildResult,
    BuildSpec,
    ResolvedFlags,
    resolve_cflags_for,
)


@dataclass(frozen=True)
class Toolchain:
    """The host C toolchain, detected once per process.

    Everything a build needs that depends on the host rather than on the
    ``BuildSpec``, so a process building both specs probes the compiler and
    OpenMP only once.
    """

    cc_cmd: str
    cc_extra: tuple[str, ...]
    python_inc: str
    numpy_inc: str
    system: str
    omp_compile: tuple[str, ...]
    omp_link: tuple[str, ...]


def detect_toolchain(report: BuildReport) -> Toolchain | str:
    """Detect the host C toolchain once, or return the reason it is unusable.

    The ``find_compiler`` and ``openmp_detect`` imports are lazy and relative
    so they resolve through ``sys.modules`` when ``hatch_build.py`` loads this
    module by file path under PEP 517 build isolation.

    Returns:
        A ``Toolchain``, or the reason detection failed. The reason is not
        printed: the caller words it as a dev-mode error or a wheel warning.
    """
    system = _platform.system()

    try:
        import numpy as np
    except ImportError:
        return "numpy not available"
    if int(np.__version__.split(".")[0]) < 2:
        return (
            f"numpy {np.__version__} is 1.x — C extension requires numpy >= 2.0 "
            "headers (build with numpy >= 2.0 to avoid an ABI mismatch)"
        )

    from .find_compiler import find_c_compiler  # lazy relative import

    compiler = find_c_compiler()
    if compiler is None:
        return (
            "no usable C compiler found on PATH (tried $CC, sysconfig, cc, "
            "clang, gcc). Install: apt-get install -y gcc (Linux) or "
            "xcode-select --install (macOS)"
        )
    cc_cmd, cc_extra = compiler

    python_inc = sysconfig.get_config_var("INCLUDEPY") or ""
    python_h = Path(python_inc) / "Python.h" if python_inc else None
    if not python_h or not python_h.exists():
        return (
            f"Python.h not found at {python_inc}. Install development headers: "
            "apt-get install -y python3-dev (Linux)"
        )

    if system == "Windows":
        return "Windows is not supported for C extension compilation"

    from .openmp_detect import detect_openmp_flags  # lazy relative import

    omp_compile, omp_link, cc_cmd = detect_openmp_flags(cc_cmd, system, report)

    return Toolchain(
        cc_cmd=cc_cmd,
        cc_extra=tuple(cc_extra),
        python_inc=python_inc,
        numpy_inc=np.get_include(),
        system=system,
        omp_compile=tuple(omp_compile),
        omp_link=tuple(omp_link),
    )


@dataclass(frozen=True, slots=True)
class _CompileSucceeded:
    objects: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _CompileFailed:
    stderr: str


_CompileAttempt: TypeAlias = _CompileSucceeded | _CompileFailed


def _compile_sources(
    spec: BuildSpec,
    src_dir: Path,
    include_dirs: Sequence[str],
    toolchain: Toolchain,
    flags: ResolvedFlags,
    tmp_dir: Path,
    report: BuildReport,
    *,
    omp_compile: Sequence[str],
    object_suffix: str,
) -> _CompileAttempt:
    """Compile every source or return the first failure explicitly."""
    objects: list[Path] = []
    for name in spec.sources:
        source = src_dir / name
        object_path = tmp_dir / f"{source.stem}{object_suffix}.o"
        cflags = resolve_cflags_for(
            flags, include_dirs, lapack=name in spec.lapack_sources
        )
        command = [
            toolchain.cc_cmd,
            *toolchain.cc_extra,
            *cflags,
            *omp_compile,
            "-c",
            str(source),
            "-o",
            str(object_path),
        ]
        report.detail(f"compile: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            report.warn(f"Compile failed for {name}:")
            report.warn(result.stderr)
            return _CompileFailed(result.stderr.strip())
        objects.append(object_path)
    return _CompileSucceeded(tuple(objects))


def _link_objects(
    toolchain: Toolchain,
    flags: ResolvedFlags,
    objects: Sequence[Path],
    output: Path,
    report: BuildReport,
    *,
    omp_link: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Link one shared library; the OpenMP retry differs only by ``omp_link``."""
    command = [
        toolchain.cc_cmd,
        *toolchain.cc_extra,
        *_SHARED_LINK_FLAGS,
        *[str(o) for o in objects],
        "-o",
        str(output),
        *flags.platform_link,
        *omp_link,
        *flags.link_libs,
    ]
    report.detail(f"link: {' '.join(command)}")
    return subprocess.run(command, capture_output=True, text=True)


def execute_build(
    spec: BuildSpec,
    src_dir: Path,
    include_dirs: Sequence[str],
    toolchain: Toolchain,
    flags: ResolvedFlags,
    output: Path,
    tmp_dir: Path,
    report: BuildReport,
) -> BuildResult:
    """Two-phase compile + link with OpenMP retry.

    Phase 1 (compile): each of ``spec.sources`` → .o, LAPACK sources with the
    strict-IEEE flags (``resolve_cflags_for``). First attempt uses the
    toolchain's OpenMP compile flags. On failure, retries once without them
    and also drops the OpenMP link flags, since single-threaded objects
    cannot link the OpenMP runtime.

    Phase 2 (link): all .o → the shared library. On failure with OpenMP link
    flags, retries once without them (the "libiomp5 revoked mid-build" path).

    The two-phase split prevents dual OpenMP runtime (libgomp + libiomp5 →
    OMP: Error #13). GCC's -fopenmp implicitly adds -lgomp at link time; when
    libiomp5 (Intel OpenMP, bundled with MKL numpy) is also linked, both
    runtimes initialize and abort ("OMP: Error #13: Assertion failure at
    kmp_runtime.cpp"). Splitting into compile (.o) then link (.so) lets us
    pass -fopenmp only to the compiler and link only libiomp5.

    Each retry is reported once through ``report.warn`` with the first
    attempt's stderr, so the root cause surfaces even if the retry succeeds.
    Compile failures print the compiler's stderr through ``report.warn``: a
    silent "compile failed" is a debugging dead end.

    Returns:
        ``BuildResult`` with ``phase="ok"`` or ``phase="build"``; on failure
        ``error`` names the failing stage and no partial output is kept.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    used_openmp = bool(toolchain.omp_compile)
    omp_link: Sequence[str] = toolchain.omp_link
    compile_attempt = _compile_sources(
        spec,
        src_dir,
        include_dirs,
        toolchain,
        flags,
        tmp_dir,
        report,
        omp_compile=toolchain.omp_compile,
        object_suffix="",
    )

    if isinstance(compile_attempt, _CompileFailed) and used_openmp:
        report.warn(
            "OpenMP compilation failed, retrying without OpenMP "
            f"(single-threaded). first-attempt stderr: "
            f"{compile_attempt.stderr or '<empty>'}"
        )
        compile_attempt = _compile_sources(
            spec,
            src_dir,
            include_dirs,
            toolchain,
            flags,
            tmp_dir,
            report,
            omp_compile=(),
            object_suffix="_noomp",
        )
        omp_link = ()
        used_openmp = False

    if isinstance(compile_attempt, _CompileFailed):
        return BuildResult(
            phase="build", error=f"compile failed: {compile_attempt.stderr}"
        )

    # Link to a PID-suffixed sibling, then replace() onto the output: the
    # rename is atomic on one filesystem, so concurrent recompilers never see
    # a half-written .so or clobber each other's temp file.
    link_tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    objects = compile_attempt.objects
    link = _link_objects(toolchain, flags, objects, link_tmp, report, omp_link=omp_link)

    if link.returncode != 0 and omp_link:
        first_stderr = link.stderr.strip()
        report.warn(
            "link failed, retrying without OpenMP runtime. "
            f"first-attempt stderr: {first_stderr or '<empty>'}"
        )
        omp_link = ()
        link = _link_objects(
            toolchain, flags, objects, link_tmp, report, omp_link=omp_link
        )

    if link.returncode != 0:
        with contextlib.suppress(OSError):
            link_tmp.unlink()
        return BuildResult(
            phase="build",
            error=f"link failed: {link.stderr}",
            used_openmp=used_openmp,
        )

    # A stale .so is better than a truncated one, so a failed rename is a
    # build failure.
    try:
        link_tmp.replace(output)
    except OSError as e:
        with contextlib.suppress(OSError):
            link_tmp.unlink()
        # The link succeeded; keep its real OpenMP state for telemetry.
        return BuildResult(
            phase="build",
            error=f"atomic replace of {output} failed: {e}",
            used_openmp=used_openmp,
            used_openmp_link=bool(omp_link),
        )

    return BuildResult(
        phase="ok",
        output_path=output,
        used_openmp=used_openmp,
        used_openmp_link=bool(omp_link),
    )
