"""Canonical compile flags, source lists, and compile+link driver.

Single source of truth consumed by hatch_build.py (wheel), _compile_jlinalg.py
(jlinalg dev-mode), and _compile_accel.py (lmm dev-mode). See
.planning/phases/123-compile-and-link-helper/123-CONTEXT.md for design rationale.

No flag literal belongs outside this module; a pre-commit lint (added in a
later wave) enforces that.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data constants — lifted verbatim from src/jamma/jlinalg/_compile_jlinalg.py
# (lines 69-81 for source lists, 154-181 for flag lists). Any change here
# must propagate to all three entry points in Wave 3.
# ---------------------------------------------------------------------------

# Default source names — callers supply their own source directory.
BASELINE_SOURCES: tuple[str, ...] = (
    "platform.c",
    "pymodule.c",
    "blas_dispatch.c",
    "snp_stats.c",
)

# LAPACK sources require strict IEEE 754 (-O2 -fno-fast-math) — no unrolling,
# no fast-math — to match vendor LAPACK's numerical guarantees.
LAPACK_SOURCES: tuple[str, ...] = ("eigh.c",)

# Base compile flags (shared by all non-LAPACK sources — no SIMD flags here).
# -fno-finite-math-only is TRAILING on purpose so it overrides a user-supplied
# -Ofast; see resolve_cflags_for's ordering comment.
BASE_CFLAGS: tuple[str, ...] = (
    "-O3",
    "-ftree-vectorize",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-funroll-loops",
    "-fno-finite-math-only",
    "-Wframe-larger-than=131072",
    "-fPIC",
    "-std=c11",
)

# LAPACK compile flags: strict IEEE 754, -O2 only, NO unrolling, NO fast-math.
LAPACK_CFLAGS: tuple[str, ...] = (
    "-O2",
    "-fno-fast-math",
    "-fno-finite-math-only",
    "-Wframe-larger-than=131072",
    "-fPIC",
    "-std=c11",
)

# Platform-default link flags. Caller still appends omp_link + ldflags.
LINK_FLAGS_BY_PLATFORM: dict[str, tuple[str, ...]] = {
    "Linux": ("-ldl", "-lpthread"),
    "Darwin": ("-undefined", "dynamic_lookup"),
}


# ---------------------------------------------------------------------------
# resolve_cflags_for — pure dispatch function
# ---------------------------------------------------------------------------


def resolve_cflags_for(
    source_path: Path,
    lapack_source_set: set[str],
    include_dirs: list[str],
    extra_cflags: list[str] | None = None,
    extra_source_includes: list[str] | None = None,
) -> list[str]:
    """Return compile flags for a single source.

    Dispatches BASE_CFLAGS vs LAPACK_CFLAGS based on whether ``str(source_path)``
    is in ``lapack_source_set``. Appends ``-I<d>`` for each include dir and each
    extra_source_includes entry.

    BASE_CFLAGS ordering: ``[-O3, -ftree-vectorize, -fno-math-errno,
    -fno-trapping-math, -funroll-loops, *extra_cflags, -fno-finite-math-only,
    -Wframe-larger-than=..., -fPIC, -std=c11]``

    The ``extra_cflags`` insertion BEFORE ``-fno-finite-math-only`` is
    load-bearing: user CFLAGS may contain ``-Ofast`` (which implies
    ``-ffinite-math-only``), and the trailing explicit ``-fno-finite-math-only``
    must override it so isnan() keeps working. DO NOT change this order.
    See hatch_build.py:592-611 for the equivalent before-image.

    The LAPACK path deliberately does NOT splice ``extra_cflags`` — LAPACK
    sources are strict IEEE 754, and a user-supplied ``-Ofast`` would defeat
    that split. If a future caller legitimately needs extra LAPACK flags, add
    a separate ``extra_lapack_cflags`` parameter.

    NOTE on extra_source_includes signature:
      * Here in ``resolve_cflags_for`` it is ``list[str]`` (paths for THIS one
        source).
      * In ``compile_jlinalg`` it is ``dict[str, list[str]]`` keyed by source
        filename. The dict form lets callers specify per-source includes (e.g.
        compile_test_harness passes tests_dir only to test_*.c sources).
        ``compile_jlinalg`` looks up each source in the dict and forwards the
        matching list to ``resolve_cflags_for``.
    """
    extra_cflags = list(extra_cflags or [])
    extra_source_includes = list(extra_source_includes or [])
    include_flags = [f"-I{d}" for d in include_dirs] + [
        f"-I{d}" for d in extra_source_includes
    ]

    if str(source_path) in lapack_source_set:
        # LAPACK sources: strict IEEE 754. Caller-supplied extra_cflags (e.g.
        # -Ofast from a user CFLAGS env) would defeat the IEEE 754 split, so
        # they are deliberately NOT merged here.
        return [*LAPACK_CFLAGS, *include_flags]

    # Baseline path: splice extra_cflags BEFORE -fno-finite-math-only so the
    # trailing explicit flag overrides a user -Ofast. Rebuild explicitly
    # rather than mutate BASE_CFLAGS — BASE_CFLAGS is frozen as a tuple.
    return [
        "-O3",
        "-ftree-vectorize",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-funroll-loops",
        *extra_cflags,
        "-fno-finite-math-only",
        "-Wframe-larger-than=131072",
        "-fPIC",
        "-std=c11",
        *include_flags,
    ]


# ---------------------------------------------------------------------------
# CompileResult — structured return from compile_jlinalg
# ---------------------------------------------------------------------------


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
        obj_files: List of .o files produced during the compile phase. Caller
            is responsible for cleanup (helpers create a temp dir that the
            caller may delete after link; see `tmp_dir` parameter).
    """

    success: bool
    used_openmp: bool
    used_openmp_link: bool
    output_path: Path | None = None
    error: str | None = None
    obj_files: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# compile_jlinalg — two-phase compile + link with OpenMP retry
# ---------------------------------------------------------------------------


def compile_jlinalg(
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
    extra_source_includes: dict[str, list[str]] | None = None,
    on_retry: Callable[[str], None] | None = None,
    verbose_print: Callable[..., None] = print,
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
    OMP: Error #13). -fopenmp is compile-only; libiomp5 is link-only.
    # Two-step compile+link to prevent dual OpenMP runtime.
    # GCC's -fopenmp implicitly adds -lgomp at link time. When libiomp5
    # (Intel OpenMP, bundled with MKL numpy) is also linked, both runtimes
    # initialize and abort: "OMP: Error #13: Assertion failure at
    # kmp_runtime.cpp". Splitting into compile (.o) then link (.so) lets
    # us pass -fopenmp only to the compiler and link only libiomp5.

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
        extra_link_flags: Extra flags for the link command line (after
            ``ldflags``, before ``-o``).
        extra_source_includes: Per-source extra ``-I<d>`` flags keyed by
            source filename (``src.name``). Used by compile_test_harness
            to supply tests_dir only to test_*.c sources.
        on_retry: Optional callback invoked with a human-readable reason
            string when a retry path is taken. Intended for warning logs.
        verbose_print: Printer used for non-error progress messages.

    Returns:
        CompileResult. On failure, ``success=False`` and ``error`` describes
        which phase failed; partial state is not retained.
    """
    extra_cflags = list(extra_cflags or [])
    extra_link_flags = list(extra_link_flags or [])
    extra_source_includes = dict(extra_source_includes or {})

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="jamma-build-"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Precompute LAPACK dispatch set — str() comparison avoids Path.resolve()
    # cross-platform quirks. Pattern lifted from _compile_jlinalg.py:190.
    lapack_source_set = {str(s) for s in lapack_sources}

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
                verbose_print(f"Compile failed for {src.name}:")
                verbose_print(result.stderr)
                return None
            objs.append(obj_file)
        return objs

    # First attempt: with OpenMP.
    used_openmp = bool(omp_compile)
    current_omp_link = list(omp_link)
    compile_objs = _compile_sources(omp_compile, "")

    if compile_objs is None and used_openmp:
        msg = "OpenMP compilation failed, retrying without OpenMP (single-threaded)"
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
    link_cmd = [
        cc_cmd,
        *cc_extra,
        "-shared",
        "-fPIC",
        *[str(o) for o in compile_objs],
        "-o",
        str(output),
        *ldflags,
        *current_omp_link,
        *extra_link_flags,
    ]
    verbose_print(f"link: {' '.join(link_cmd)}")
    link_result = subprocess.run(link_cmd, capture_output=True, text=True)

    used_openmp_link = bool(current_omp_link)

    if link_result.returncode != 0 and current_omp_link:
        msg = "link failed, retrying without OpenMP runtime"
        if on_retry is not None:
            on_retry(msg)
        verbose_print(msg + "...")
        retry_link_cmd = [
            cc_cmd,
            *cc_extra,
            "-shared",
            "-fPIC",
            *[str(o) for o in compile_objs],
            "-o",
            str(output),
            *ldflags,
            *extra_link_flags,
        ]
        verbose_print(f"link: {' '.join(retry_link_cmd)}")
        link_result = subprocess.run(retry_link_cmd, capture_output=True, text=True)
        used_openmp_link = False

    if link_result.returncode != 0:
        return CompileResult(
            success=False,
            used_openmp=used_openmp,
            used_openmp_link=False,
            error=f"link failed: {link_result.stderr}",
            obj_files=compile_objs,
        )

    return CompileResult(
        success=True,
        used_openmp=used_openmp,
        used_openmp_link=used_openmp_link,
        output_path=output,
        obj_files=compile_objs,
    )
