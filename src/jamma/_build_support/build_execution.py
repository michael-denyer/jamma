"""Host toolchain detection and atomic compile/link execution."""

from __future__ import annotations

import contextlib
import os
import platform as _platform
import subprocess
import sysconfig
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .build_models import SHARED_LINK_FLAGS as _SHARED_LINK_FLAGS
from .build_models import resolve_cflags_for

# ---------------------------------------------------------------------------
# CompileResult — structured return from execute_build
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Toolchain:
    """The host C toolchain, detected once per process.

    Carries everything ``run_build`` needs that depends on the host rather
    than on which ``BuildSpec`` is being built: the compiler command, the
    Python/NumPy include directories, and the OpenMP compile/link flags.
    Building two specs (``_lmm_accel``, ``_jlinalg``) in the same process —
    as ``hatch_build.py`` and CI both do — detects this once and reuses it,
    rather than re-probing the compiler and re-running the OpenMP libiomp5
    dance for every spec.
    """

    cc_cmd: str
    cc_extra: tuple[str, ...]
    python_inc: str
    numpy_inc: str
    system: str
    omp_compile: tuple[str, ...]
    omp_link: tuple[str, ...]

    def diagnose_flags(self) -> tuple[str, ...]:
        """Vectorization-report flags for this compiler (clang vs gcc).

        Identifying the compiler is toolchain probing, so it belongs here
        rather than in the composition root. Called only on the
        ``diagnose=True`` path, so an ordinary build spends no ``cc
        --version`` subprocess, and the answer is the same for every
        ``BuildSpec`` built with this toolchain.
        """
        return _diagnose_flags(self.cc_cmd)


def _diagnose_flags(cc_cmd: str) -> tuple[str, ...]:
    """Vectorization-report flags for ``cc_cmd`` (clang ``-Rpass`` vs gcc)."""
    try:
        probe = subprocess.run(
            [cc_cmd, "--version"], capture_output=True, text=True, timeout=5
        )
        compiler_id = probe.stdout.lower() if probe.returncode == 0 else ""
    except (subprocess.TimeoutExpired, OSError):
        compiler_id = ""
    if "clang" in compiler_id:
        return (
            "-Rpass=loop-vectorize",
            "-Rpass-missed=loop-vectorize",
            "-Rpass-analysis=loop-vectorize",
        )
    return ("-fopt-info-vec-all",)


def detect_toolchain(
    *,
    verbose_print: Callable[..., None] = print,
    error_print: Callable[..., None] | None = None,
) -> Toolchain | str:
    """Detect the host C toolchain once, or return the reason it is unusable.

    Performs every preflight step that depends on the host rather than on a
    particular ``BuildSpec``: numpy availability and version, compiler
    discovery (``$CC``, sysconfig, ``cc``/``clang``/``gcc`` fallbacks),
    ``Python.h`` presence, the Windows reject, and OpenMP flag detection.
    A build entry point calls this once and passes the result to every spec it
    builds; ``run_build`` no longer takes ``find_c_compiler`` or
    ``detect_openmp_flags`` as injected parameters.

    The imports of ``find_compiler`` and ``openmp_detect`` are lazy and
    relative so this module keeps working when ``hatch_build.py`` loads it
    by file path under ``importlib.util.spec_from_file_location`` (the PEP
    517 build backend registers all three ``_build_support`` helper modules
    on ``sys.modules`` before calling anything, so the relative import
    resolves via the ``sys.modules`` short-circuit rather than a package
    lookup that would fail under build isolation).

    Returns:
        A ``Toolchain`` on success, or a human-readable string naming the
        reason detection failed (no compiler, no numpy, missing headers,
        Windows). The string is not printed here — the caller decides
        whether to log it as a dev-mode error or a wheel-build warning.
    """
    if error_print is None:
        error_print = verbose_print
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

    omp_compile, omp_link, cc_cmd = detect_openmp_flags(
        cc_cmd, system, verbose_print, _warn=error_print
    )

    return Toolchain(
        cc_cmd=cc_cmd,
        cc_extra=tuple(cc_extra),
        python_inc=python_inc,
        numpy_inc=np.get_include(),
        system=system,
        omp_compile=tuple(omp_compile),
        omp_link=tuple(omp_link),
    )


def link_cmd(
    cc_cmd: str,
    cc_extra: list[str],
    objs: list[Path],
    out: Path,
    ldflags: list[str],
    omp_link: list[str],
    extra: list[str],
) -> list[str]:
    """Build one shared-library link command.

    Called for the first attempt and, on link failure with ``omp_link``
    non-empty, for the OMP-free retry. The two calls differ only by
    ``omp_link``, so a hand-edit to one link command would otherwise require
    the same edit twice.
    """
    return [
        cc_cmd,
        *cc_extra,
        *_SHARED_LINK_FLAGS,
        *[str(o) for o in objs],
        "-o",
        str(out),
        *ldflags,
        *omp_link,
        *extra,
    ]


@dataclass
class CompileResult:
    """Result of a two-phase compile+link invocation.

    Attributes:
        success: True iff both compile and link phases finished with rc=0.
        used_openmp: True iff the compile phase used ``omp_compile`` flags
            (i.e. the retry-without-OMP path was not taken).
        used_openmp_link: True iff the link phase used ``omp_link`` flags.
            Can be False while ``used_openmp=True`` if the link phase retried
            without OMP runtime.
        output_path: Path to the final shared library on success, None otherwise.
        error: Human-readable error message on failure, None on success.
    """

    success: bool
    used_openmp: bool
    used_openmp_link: bool
    output_path: Path | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# execute_build — two-phase compile + link with OpenMP retry
# ---------------------------------------------------------------------------


def execute_build(
    sources: list[Path],
    lapack_sources: list[Path],
    include_dirs: list[str],
    cc_cmd: str,
    cc_extra: list[str],
    omp_compile: list[str],
    omp_link: list[str],
    ldflags: list[str],
    output: Path,
    *,
    tmp_dir: Path | None = None,
    extra_cflags: list[str] | None = None,
    extra_link_flags: list[str] | None = None,
    extra_lapack_cflags: list[str] | None = None,
    extra_source_includes: dict[str, list[str]] | None = None,
    on_retry: Callable[[str], None] | None = None,
    verbose_print: Callable[..., None] = print,
    error_print: Callable[..., None] | None = None,
) -> CompileResult:
    """Two-phase compile + link with OpenMP retry.

    Phase 1 (compile): each source → .o using ``resolve_cflags_for`` dispatch
    (LAPACK vs baseline). First attempt uses ``omp_compile``. On failure and
    ``omp_compile`` non-empty, retries once without OMP compile flags and
    also clears OMP link flags (since single-threaded objects cannot link
    the OMP runtime).

    Phase 2 (link): all .o → .so (or platform shared-lib suffix) with
    ``omp_link`` and ``ldflags``. On failure and ``omp_link`` non-empty,
    retries once without ``omp_link`` (the "libiomp5 revoked mid-build"
    path). Extra link flags from the caller (e.g. ``-lm``) are appended.

    The two-phase split prevents dual OpenMP runtime (libgomp + libiomp5 →
    OMP: Error #13). GCC's -fopenmp implicitly adds -lgomp at link time; when
    libiomp5 (Intel OpenMP, bundled with MKL numpy) is also linked, both
    runtimes initialize and abort ("OMP: Error #13: Assertion failure at
    kmp_runtime.cpp"). Splitting into compile (.o) then link (.so) lets us
    pass -fopenmp only to the compiler and link only libiomp5.

    Args:
        sources: All source files to compile (baseline + LAPACK).
        lapack_sources: Subset of ``sources`` that require LAPACK_CFLAGS.
            Membership is tested by ``str(source)`` equality.
        include_dirs: Include directories (``-I<d>``) applied to every source.
        cc_cmd: Compiler command (e.g. ``cc``, ``clang``, ``gcc``).
        cc_extra: Extra compiler invocation args (e.g. target triple flags).
        omp_compile: OpenMP compile flags (e.g. ``["-fopenmp"]``). Empty list
            disables OpenMP entirely.
        omp_link: OpenMP link flags (e.g. ``["/path/to/libiomp5.so"]``).
        ldflags: Extra link flags (``-lm``, ``-ldl``, ``-lpthread``, etc.).
        output: Final shared-library path.
        tmp_dir: Directory for intermediate .o files. If None, a temp dir
            is created via ``tempfile.mkdtemp(prefix="jamma-build-")``. Caller
            is responsible for cleanup in either case.
        extra_cflags: User-supplied CFLAGS spliced into BASE_CFLAGS (see
            ``resolve_cflags_for``). Not applied to LAPACK sources.
        extra_link_flags: Extra flags appended after ``ldflags`` and the
            OpenMP link flags.
        extra_lapack_cflags: Extra flags appended to LAPACK_CFLAGS for LAPACK
            sources (forwarded to ``resolve_cflags_for``). Used exclusively
            by the sanitizer instrumentation flow — assembled by
            ``apply_sanitizer_overrides()`` so ``eigh.c`` is also instrumented
            when ``JAMMA_SANITIZE`` is set. LAPACK sources stay
            ``-O2 -fno-fast-math`` baseline; the trailing ``-O1`` from the
            sanitizer override wins because gcc/clang honour the last ``-O``.
        extra_source_includes: Per-source extra ``-I<d>`` flags keyed by
            source filename (``src.name``).
        on_retry: Optional callback invoked with a human-readable reason
            string when a retry path is taken. Intended for warning logs.
            For retry paths with underlying compiler output, the first-attempt
            stderr is included in the message so the root cause surfaces even
            if the retry succeeds.
        verbose_print: Printer used for non-error progress messages.
        error_print: Printer used for compile/link failure diagnostics. MUST
            always be visible — callers that silence verbose_print (e.g.
            verbose=False in dev-mode compilers) must NOT silence this. If
            None, defaults to ``verbose_print`` (callers who share a single
            always-visible printer can ignore this). Per CLAUDE.md "No Quiet
            Flags Anywhere", compilation stderr on failure must reach the
            user — a silent "compile failed" is a debugging dead end.

    Returns:
        CompileResult. On failure, ``success=False`` and ``error`` describes
        which phase failed; partial state is not retained.
    """
    extra_cflags = list(extra_cflags or [])
    extra_link_flags = list(extra_link_flags or [])
    extra_lapack_cflags = list(extra_lapack_cflags or [])
    extra_source_includes = dict(extra_source_includes or {})

    # error_print defaults to verbose_print so existing callers keep their
    # behavior, but callers that silence verbose_print (dev-mode compilers
    # with verbose=False) should pass an always-visible printer here.
    if error_print is None:
        error_print = verbose_print

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="jamma-build-"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Precompute LAPACK dispatch set — str() comparison avoids Path.resolve()
    # cross-platform quirks. Pattern lifted from _compile_jlinalg.py:190.
    lapack_source_set = {str(s) for s in lapack_sources}

    # Mutable box so _compile_sources can report the first failure stderr
    # back up for inclusion in the retry notice. Without this, an OMP
    # downgrade on compile masks the real cause (e.g. missing omp.h).
    first_stderr_holder: list[str] = [""]

    def _compile_sources(
        current_omp_compile: list[str], suffix: str
    ) -> list[Path] | None:
        """Compile each source to .o. Returns list of .o paths or None on failure."""
        objs: list[Path] = []
        for src in sources:
            obj_file = tmp_dir / f"{src.stem}{suffix}.o"
            per_source_includes = extra_source_includes.get(src.name, [])
            cflags = resolve_cflags_for(
                src,
                lapack_source_set,
                include_dirs,
                extra_cflags=extra_cflags,
                extra_source_includes=per_source_includes,
                extra_lapack_cflags=extra_lapack_cflags,
            )
            cmd = [
                cc_cmd,
                *cc_extra,
                *cflags,
                *current_omp_compile,
                "-c",
                str(src),
                "-o",
                str(obj_file),
            ]
            verbose_print(f"compile: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Compile failure diagnostics go to error_print (always
                # visible) so runtime recompile with verbose=False still
                # surfaces the root cause rather than a bare
                # "compile failed" with no stderr.
                error_print(f"Compile failed for {src.name}:")
                error_print(result.stderr)
                first_stderr_holder[0] = (result.stderr or "").strip()
                return None
            objs.append(obj_file)
        return objs

    # First attempt: with OpenMP.
    used_openmp = bool(omp_compile)
    current_omp_link = omp_link.copy()
    compile_objs = _compile_sources(omp_compile, "")

    if compile_objs is None and used_openmp:
        first_stderr = first_stderr_holder[0]
        msg = (
            "OpenMP compilation failed, retrying without OpenMP "
            f"(single-threaded). first-attempt stderr: "
            f"{first_stderr or '<empty>'}"
        )
        if on_retry is not None:
            on_retry(msg)
        verbose_print(msg + "...")
        compile_objs = _compile_sources([], "_noomp")
        current_omp_link = []  # no OMP runtime to link
        used_openmp = False

    if compile_objs is None:
        return CompileResult(
            success=False,
            used_openmp=False,
            used_openmp_link=False,
            error="compile failed",
        )

    # Phase 2: link.
    # Link to a sibling temp path then os.replace() onto the final output.
    # On POSIX and Windows os.replace() is atomic when src and dst are on the
    # same filesystem — concurrent recompilers (pytest-xdist workers, parallel
    # Databricks tasks) can never observe a half-written .so. The PID suffix
    # also guarantees parallel linkers don't clobber each other's tmp file.
    link_tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    first_link_cmd = link_cmd(
        cc_cmd,
        cc_extra,
        compile_objs,
        link_tmp,
        ldflags,
        current_omp_link,
        extra_link_flags,
    )
    verbose_print(f"link: {' '.join(first_link_cmd)}")
    link_result = subprocess.run(first_link_cmd, capture_output=True, text=True)

    used_openmp_link = bool(current_omp_link)

    if link_result.returncode != 0 and current_omp_link:
        # Include the first-attempt stderr in the retry notice so the
        # root cause (missing -lpthread, wrong libiomp5 path, broken
        # RPATH, etc.) surfaces even if the retry succeeds. Without
        # this, a silent OMP downgrade masks real link bugs.
        first_stderr = (link_result.stderr or "").strip()
        msg = (
            "link failed, retrying without OpenMP runtime. "
            f"first-attempt stderr: {first_stderr or '<empty>'}"
        )
        if on_retry is not None:
            on_retry(msg)
        verbose_print(msg + "...")
        retry_link_cmd = link_cmd(
            cc_cmd,
            cc_extra,
            compile_objs,
            link_tmp,
            ldflags,
            [],
            extra_link_flags,
        )
        verbose_print(f"link: {' '.join(retry_link_cmd)}")
        link_result = subprocess.run(retry_link_cmd, capture_output=True, text=True)
        used_openmp_link = False

    if link_result.returncode != 0:
        # Tidy up any partial tmp from the failed link attempt.
        with contextlib.suppress(OSError):
            link_tmp.unlink()
        return CompileResult(
            success=False,
            used_openmp=used_openmp,
            used_openmp_link=False,
            error=f"link failed: {link_result.stderr}",
        )

    # Atomic publish — readers see either the old .so or the new one, never
    # a partially-written file. On Windows os.replace handles in-use targets
    # less gracefully than POSIX, but a stale .so is still better than a
    # truncated one, so we let the OSError surface as a link failure.
    try:
        link_tmp.replace(output)
    except OSError as e:
        with contextlib.suppress(OSError):
            link_tmp.unlink()
        # Link succeeded; the rename is what failed. Preserve the real
        # used_openmp_link value so telemetry doesn't misreport the build.
        return CompileResult(
            success=False,
            used_openmp=used_openmp,
            used_openmp_link=used_openmp_link,
            error=f"atomic replace of {output} failed: {e}",
        )

    return CompileResult(
        success=True,
        used_openmp=used_openmp,
        used_openmp_link=used_openmp_link,
        output_path=output,
    )
