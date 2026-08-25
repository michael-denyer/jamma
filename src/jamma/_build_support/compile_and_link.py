"""Canonical compile flags, source lists, and compile+link driver.

Single source of truth consumed by hatch_build.py (PEP 517 wheel backend),
_compile_jlinalg.py (dev-mode + runtime recompile), and _compile_accel.py
(dev-mode + runtime recompile). Ships inside the installed package as
``jamma._build_support.compile_and_link`` so runtime ABI-mismatch
recompile reaches the same helpers the wheel was built with.

No flag literal belongs outside this module; the pre-commit hook
``no-compile-flag-literals-outside-build-support``
(scripts/check_compile_flag_literals.py) enforces that, and
``verify-compile-invocations-match`` confirms the three entry points route
through ``run_build`` below (which drives ``compile_jlinalg``).
"""

from __future__ import annotations

import contextlib
import os
import platform as _platform
import shutil
import subprocess
import sysconfig
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data constants — THE single source of truth. The three entry points
# (hatch_build.py, _compile_jlinalg.py, _compile_accel.py) all import from
# here; no flag literal may live elsewhere.
# ---------------------------------------------------------------------------

# Default source names — callers supply their own source directory.
BASELINE_SOURCES: tuple[str, ...] = (
    "platform.c",
    "pymodule.c",
    "blas_dispatch.c",
    "snp_stats.c",
)

# LMM accelerator sources — callers supply their own source directory.
# _lmm_accel.c owns import_array(); every other unit here must define
# NO_IMPORT_ARRAY before including _lmm_support.h, or its NumPy C-API pointer
# stays NULL and the first PyArray_* call segfaults. Both entry points that
# build the accelerator (hatch_build.py, _compile_accel.py) read this tuple, so
# a new source lands in the wheel and the dev rebuild together. macOS links
# with -undefined dynamic_lookup, so a source missing from here does NOT fail
# the link — it fails at import, or silently much later.
LMM_ACCEL_SOURCES: tuple[str, ...] = (
    "_lmm_accel.c",
    "_lmm_support.c",
    "_lmm_stats.c",
    "_lmm_tests.c",
    "_lmm_kernels_general.c",
    "_lmm_kernels_ncvt1.c",
)

# LAPACK sources require strict IEEE 754 (-O2 -fno-fast-math) — no unrolling,
# no fast-math — to match vendor LAPACK's numerical guarantees. Relaxing the
# split (e.g. moving eigh.c into BASELINE_SOURCES or loosening LAPACK_CFLAGS)
# breaks JAMMA-vs-GEMMA validation tolerances documented in CLAUDE.md and
# docs/GEMMA_EQUIVALENCE.md — p-values, effect sizes, and eigenvalues all drift.
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
# Do NOT relax to -O3/-ffast-math: vendor LAPACK (MKL / Accelerate) relies on
# exact IEEE rounding for eigenvalue/eigenvector accuracy. A relaxed build
# produces kinship/eigendecomp results that drift beyond JAMMA's validation
# tolerances vs GEMMA (see CLAUDE.md validation tolerances table,
# docs/GEMMA_EQUIVALENCE.md).
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
# BuildSpec — per-target description of one C extension build
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildSpec:
    """The toolchain-independent description of one C extension target.

    One instance per compiled extension (``_lmm_accel``, ``_jlinalg``). It
    carries everything ``run_build`` needs that does *not* depend on the host
    toolchain: where the sources live, which of them require the strict-IEEE
    LAPACK flags, the output name, the dev-mode-only extra flags, and whether
    progress prints to stdout or stderr. The toolchain-dependent parts (the
    compiler, the Python/NumPy include dirs, OpenMP, the sanitizer overrides)
    are discovered by ``run_build`` at build time, so the same spec drives the
    portable wheel build and the ``-march=native`` dev rebuild alike.

    Paths are stored as ``parts`` tuples relative to the package directory
    (``src/jamma`` under the PEP 517 build, the installed ``jamma/`` at
    runtime), so a spec is a pure value with no absolute path baked in. The
    caller supplies the package directory; ``run_build`` joins the parts.

    ``-march=native`` lives here in ``dev_extra_cflags`` rather than in
    ``_compile_accel.py`` so the compile-flag-literal lint has one sanctioned
    home and the dev-only flag can never reach ``hatch_build.py``: the wheel
    path does not read this field. Wheels stay portable by construction.
    """

    # Location, relative to the package directory.
    package_parts: tuple[str, ...]  # ("lmm",) / ("jlinalg",)
    source_parts: tuple[str, ...]  # () / ("src",) — subdir holding the .c files
    include_parts: tuple[tuple[str, ...], ...]  # extra -I dirs, e.g. (("include",),)
    # Sources, by bare filename; ``lapack_sources`` is the strict-IEEE subset.
    sources: tuple[str, ...]
    lapack_sources: tuple[str, ...]
    output_stem: str  # "_lmm_accel" / "_jlinalg" — EXT_SUFFIX appended at build
    # Dev-mode-only base cflags. The wheel path never applies these.
    dev_extra_cflags: tuple[str, ...] = ()  # ("-march=native",) / ()
    reads_sentinel_env: bool = False  # honour JAMMA_SENTINEL_UB (accel only)
    supports_diagnose: bool = False  # accept the vectorization-report flags
    # Runtime load identity — used by core.recompile._load_c_module and
    # auto_recompile_c_extension when a stale/missing .so must be reimported or
    # rebuilt. Stored rather than derived so tests can inject synthetic keys.
    module_name: str = ""  # log name, e.g. "_lmm_accel"
    compiler_module: str = ""  # dotted path to the compile module
    sys_module_key: str = ""  # sys.modules key of the built extension
    fallback_label: str = ""  # human label for the pure-Python fallback path
    # Symbols a valid, ABI-matched build always exports. Their absence means a
    # corrupt build rather than a stale one, so _load_c_module treats it as an
    # import failure and rebuilds. ABI equality is the real completeness check;
    # this is the belt-and-braces list the caller used to import by name.
    required_attrs: tuple[str, ...] = ()


# -march=native is dev-mode only and portable wheels must not carry it; it
# lives in LMM_ACCEL_SPEC.dev_extra_cflags, applied only on the dev rebuild
# path, never by hatch_build.py.
LMM_ACCEL_SPEC = BuildSpec(
    package_parts=("lmm",),
    source_parts=(),
    include_parts=(),
    sources=LMM_ACCEL_SOURCES,
    lapack_sources=(),
    output_stem="_lmm_accel",
    dev_extra_cflags=("-march=native",),
    reads_sentinel_env=True,
    supports_diagnose=True,
    module_name="_lmm_accel",
    compiler_module="jamma.lmm._compile_accel",
    sys_module_key="jamma.lmm._lmm_accel",
    fallback_label="LMM",
    required_attrs=(
        "HAS_OPENMP",
        "create_workspace_fused_c",
        "compute_lmm_chunk_fused_c",
        "create_workspace_fused_general_c",
        "compute_lmm_chunk_fused_general_c",
    ),
)

JLINALG_SPEC = BuildSpec(
    package_parts=("jlinalg",),
    source_parts=("src",),
    include_parts=(("include",),),
    sources=BASELINE_SOURCES + LAPACK_SOURCES,
    lapack_sources=LAPACK_SOURCES,
    output_stem="_jlinalg",
    dev_extra_cflags=(),
    reads_sentinel_env=False,
    supports_diagnose=False,
    module_name="_jlinalg",
    compiler_module="jamma.jlinalg._compile_jlinalg",
    sys_module_key="jamma.jlinalg._jlinalg",
    fallback_label="jlinalg",
    required_attrs=(
        "HAS_OPENMP",
        "blas_backend",
        "blas_has_dgemm",
        "blas_has_dgeqrf",
        "blas_has_dgesvd",
        "blas_has_dsyevd",
        "blas_has_dsyevr",
        "blas_has_dsyrk",
        "blas_has_lapacke_dsyevd",
        "blas_is_ilp64",
        "compute_snp_stats_chunk",
        "dgemm",
        "dsyrk",
        "eigh",
        "get_n_threads",
        "jlinalg_isa",
        "qr",
        "set_n_threads",
        "svd",
    ),
)

# Opt-in sentinel macro for the sanitizer-workflow self-test. When
# JAMMA_SENTINEL_UB is set, _lmm_accel.c's gated heap-OOB function
# jamma_sentinel_oob is compiled in so ASAN can be proven to catch a real bug.
# A -D preprocessor macro, not an -O/-f flag, so the compile-flag-literal lint
# does not cover it; the named constant keeps it greppable. Wheel builds never
# set the env var.
_SENTINEL_UB_DEFINE = "-DJAMMA_SENTINEL_UB"


def _sentinel_env_on(env: dict[str, str] | os._Environ[str]) -> bool:
    """Truthy check for JAMMA_SENTINEL_UB: "" and "0" are off, anything else on."""
    return env.get("JAMMA_SENTINEL_UB", "").strip() not in ("", "0")


def resolve_build_spec(
    spec: BuildSpec,
    *,
    dev_mode: bool,
    env: dict[str, str] | os._Environ[str] | None = None,
    diagnose_flags: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the base ``extra_cflags`` for a build, pre-sanitizer.

    Pure and toolchain-independent: it reads only the spec and the environment,
    so a test can assert on it with zero mocks. ``run_build`` calls it once the
    compiler is known (to pass ``diagnose_flags``); the sentinel meta-test calls
    it directly to prove ``-DJAMMA_SENTINEL_UB`` lands when the env var is set.

    Wheel path (``dev_mode=False``): honour ``CFLAGS`` and nothing else — never
    ``-march=native`` — so the wheel stays portable. Dev path: the spec's
    ``dev_extra_cflags`` (``-march=native`` for the accelerator), then any
    diagnose flags, then the sentinel macro when the env var is set. The order
    matches the four hand-written call sites this replaces.
    """
    resolved_env = os.environ if env is None else env
    if not dev_mode:
        return tuple(resolved_env.get("CFLAGS", "").split())
    extras = [*spec.dev_extra_cflags, *diagnose_flags]
    if spec.reads_sentinel_env and _sentinel_env_on(resolved_env):
        extras.append(_SENTINEL_UB_DEFINE)
    return tuple(extras)


# ---------------------------------------------------------------------------
# apply_sanitizer_overrides — env-var driven sanitizer flag injection seam
# ---------------------------------------------------------------------------


def apply_sanitizer_overrides(
    extra_cflags: list[str] | None,
    extra_link_flags: list[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Augment compile/link flags with sanitizer flags when JAMMA_SANITIZE is set.

    Returns ``(cflags, link_flags, lapack_cflags)``. When ``JAMMA_SANITIZE`` is
    unset or empty, returns the inputs unchanged plus an empty
    ``lapack_cflags``. Reads ``os.environ`` ONCE per call; the four compile
    entry points (``hatch_build.py``, ``_compile_jlinalg.py``,
    ``_compile_accel.py``, plus runtime recompile in ``core/recompile.py``)
    MUST NOT duplicate the env-var read — that would defeat the
    single-source-of-truth invariant for compile flags.

    JAMMA_SANITIZE format: comma-separated ``-fsanitize`` values, e.g.
    ``"address,undefined"`` or ``"address"`` alone. Trailing ``-O1`` wins
    over BASE_CFLAGS' ``-O3`` and LAPACK_CFLAGS' ``-O2`` because gcc/clang
    honour the LAST ``-O`` flag on the command line — so the sanitizer
    build is debuggable without a separate -O override path.
    """
    extra_cflags = list(extra_cflags or [])
    extra_link_flags = list(extra_link_flags or [])
    sanitizers = os.environ.get("JAMMA_SANITIZE", "").strip()
    if not sanitizers:
        return extra_cflags, extra_link_flags, []
    san_cflags = [
        f"-fsanitize={sanitizers}",
        "-fno-omit-frame-pointer",
        "-O1",
    ]
    san_link = [f"-fsanitize={sanitizers}"]
    return (
        [*extra_cflags, *san_cflags],
        [*extra_link_flags, *san_link],
        san_cflags.copy(),
    )


# ---------------------------------------------------------------------------
# resolve_cflags_for — pure dispatch function
# ---------------------------------------------------------------------------


def resolve_cflags_for(
    source_path: Path,
    lapack_source_set: set[str],
    include_dirs: list[str],
    extra_cflags: list[str] | None = None,
    extra_source_includes: list[str] | None = None,
    extra_lapack_cflags: list[str] | None = None,
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

    LAPACK path: deliberately does NOT splice ``extra_cflags`` — LAPACK sources
    are strict IEEE 754, and a user-supplied ``-Ofast`` would defeat that split.
    For the sanitizer use case, a separate ``extra_lapack_cflags``
    parameter is forwarded by ``compile_jlinalg`` from the
    ``apply_sanitizer_overrides()`` triple, so ``eigh.c`` is also instrumented
    when ``JAMMA_SANITIZE`` is set. The trailing ``-O1`` from the sanitizer
    flags wins over LAPACK_CFLAGS' ``-O2`` (last ``-O`` on the command line
    wins) without needing a separate override path.

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
    extra_lapack_cflags = list(extra_lapack_cflags or [])
    include_flags = [f"-I{d}" for d in include_dirs] + [
        f"-I{d}" for d in extra_source_includes
    ]

    if str(source_path) in lapack_source_set:
        # LAPACK sources: strict IEEE 754. Caller-supplied extra_cflags (e.g.
        # -Ofast from a user CFLAGS env) would defeat the IEEE 754 split, so
        # they are deliberately NOT merged here. extra_lapack_cflags IS spliced
        # — it is reserved for the sanitizer flow where the caller has already
        # asserted the flags are IEEE-safe (apply_sanitizer_overrides only adds
        # -fsanitize=..., -fno-omit-frame-pointer, -O1, none of which break
        # IEEE 754 rounding).
        return [*LAPACK_CFLAGS, *extra_lapack_cflags, *include_flags]

    # Baseline path: splice extra_cflags BEFORE -fno-finite-math-only so the
    # trailing explicit flag overrides a user -Ofast. Slice BASE_CFLAGS rather
    # than re-listing literals — keeps BASE_CFLAGS as the single source of truth
    # so that adding a flag there (e.g. -fno-plt) doesn't silently drop it on
    # the extra_cflags path.
    splice_idx = BASE_CFLAGS.index("-fno-finite-math-only")
    return [
        *BASE_CFLAGS[:splice_idx],
        *extra_cflags,
        *BASE_CFLAGS[splice_idx:],
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
    extra_lapack_cflags: list[str] | None = None,
    extra_source_includes: dict[str, list[str]] | None = None,
    link_shared: bool = True,
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
        output: Final output path (shared library when ``link_shared=True``,
            executable when ``link_shared=False``).
        tmp_dir: Directory for intermediate .o files. If None, a temp dir
            is created via ``tempfile.mkdtemp(prefix="jamma-build-")``. Caller
            is responsible for cleanup in either case.
        extra_cflags: User-supplied CFLAGS spliced into BASE_CFLAGS (see
            ``resolve_cflags_for``). Not applied to LAPACK sources.
        extra_link_flags: Extra flags for the link command line (after
            ``ldflags``, before ``-o``).
        extra_lapack_cflags: Extra flags appended to LAPACK_CFLAGS for LAPACK
            sources (forwarded to ``resolve_cflags_for``). Used exclusively
            by the sanitizer instrumentation flow — assembled by
            ``apply_sanitizer_overrides()`` so ``eigh.c`` is also instrumented
            when ``JAMMA_SANITIZE`` is set. LAPACK sources stay
            ``-O2 -fno-fast-math`` baseline; the trailing ``-O1`` from the
            sanitizer override wins because gcc/clang honour the last ``-O``.
        extra_source_includes: Per-source extra ``-I<d>`` flags keyed by
            source filename (``src.name``). Used by compile_test_harness
            to supply tests_dir only to test_*.c sources.
        link_shared: When True (default), link as a shared library with
            ``-shared -fPIC``. When False, link as a plain executable (no
            ``-shared`` / ``-fPIC`` at link time). Used by
            compile_test_harness to produce the Unity test binary, which
            must be a runnable executable rather than a .so.
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

    # Shared-library link takes -shared -fPIC; executable link takes neither.
    # (See ``link_shared`` parameter docstring.)
    link_mode_flags: tuple[str, ...] = ("-shared", "-fPIC") if link_shared else ()

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
    link_cmd = [
        cc_cmd,
        *cc_extra,
        *link_mode_flags,
        *[str(o) for o in compile_objs],
        "-o",
        str(link_tmp),
        *ldflags,
        *current_omp_link,
        *extra_link_flags,
    ]
    verbose_print(f"link: {' '.join(link_cmd)}")
    link_result = subprocess.run(link_cmd, capture_output=True, text=True)

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
        retry_link_cmd = [
            cc_cmd,
            *cc_extra,
            *link_mode_flags,
            *[str(o) for o in compile_objs],
            "-o",
            str(link_tmp),
            *ldflags,
            *extra_link_flags,
        ]
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
            obj_files=compile_objs,
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
            obj_files=compile_objs,
        )

    return CompileResult(
        success=True,
        used_openmp=used_openmp,
        used_openmp_link=used_openmp_link,
        output_path=output,
        obj_files=compile_objs,
    )


# ---------------------------------------------------------------------------
# run_build — the eleven preflight+compile steps, once, driven by a BuildSpec
# ---------------------------------------------------------------------------


@dataclass
class BuildOutcome:
    """Result of ``run_build``.

    ``ok`` is True only when compile and link both succeeded. ``skipped`` marks
    a preflight guard firing (no numpy, no compiler, missing headers/sources,
    Windows) as distinct from a compile/link failure — the wheel path turns
    both into a pure-Python fallback, while the dev path returns False either
    way. ``output_path`` and ``out_name`` are populated whenever the sources
    were located, so the caller can register the wheel artifact or run its
    import probe.
    """

    ok: bool
    result: CompileResult | None = None
    output_path: Path | None = None
    out_name: str = ""
    skipped: bool = False


def _diagnose_flags(cc_cmd: str) -> list[str]:
    """Vectorization-report flags for ``cc_cmd`` (clang ``-Rpass`` vs gcc)."""
    try:
        probe = subprocess.run(
            [cc_cmd, "--version"], capture_output=True, text=True, timeout=5
        )
        compiler_id = probe.stdout.lower() if probe.returncode == 0 else ""
    except (subprocess.TimeoutExpired, OSError):
        compiler_id = ""
    if "clang" in compiler_id:
        return [
            "-Rpass=loop-vectorize",
            "-Rpass-missed=loop-vectorize",
            "-Rpass-analysis=loop-vectorize",
        ]
    return ["-fopt-info-vec-all"]


def run_build(
    spec: BuildSpec,
    package_dir: Path,
    *,
    dev_mode: bool,
    find_c_compiler: Callable[[], tuple[str, list[str]] | None],
    detect_openmp_flags: Callable[..., tuple[list[str], list[str], str]],
    diagnose: bool = False,
    on_retry: Callable[[str], None] | None = None,
    verbose_print: Callable[..., None] = print,
    error_print: Callable[..., None] | None = None,
    env: dict[str, str] | os._Environ[str] | None = None,
) -> BuildOutcome:
    """Run the preflight checks and the two-phase compile for one ``BuildSpec``.

    The eleven steps every compile entry point used to hand-write: numpy>=2,
    compiler discovery, Python.h, Windows reject, ``EXT_SUFFIX``, numpy include,
    OpenMP detection, platform link flags, sanitizer overrides, and the
    ``compile_jlinalg`` call under a temp dir. The import probe and any wheel
    ``force_include`` registration stay with the caller, since the wheel backend
    does neither. ``find_c_compiler`` and ``detect_openmp_flags`` are injected so
    this module never imports its ``_build_support`` siblings — the PEP 517
    backend loads it by file path, where such an import would fail.

    Preflight failures print through ``error_print`` — as ``WARNING`` and a
    pure-Python-fallback note under the wheel build, as ``ERROR`` in dev mode —
    and return ``BuildOutcome(ok=False, skipped=True)``. Platform link flags are
    taken once from ``LINK_FLAGS_BY_PLATFORM``; the wheel path no longer adds a
    second ``-undefined dynamic_lookup`` on macOS.
    """
    resolved_env = os.environ if env is None else env
    if error_print is None:
        error_print = verbose_print
    system = _platform.system()
    lead = "WARNING" if not dev_mode else "ERROR"
    tail = " (pure-Python fallback)." if not dev_mode else ""

    def _skip(message: str) -> BuildOutcome:
        error_print(f"{lead}: {message}{tail}")
        return BuildOutcome(ok=False, skipped=True)

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

    try:
        import numpy as np
    except ImportError:
        return _skip("numpy not available")
    if int(np.__version__.split(".")[0]) < 2:
        return _skip(
            f"numpy {np.__version__} is 1.x — C extension requires numpy >= 2.0 "
            "headers (build with numpy >= 2.0 to avoid an ABI mismatch)"
        )

    cc_env = resolved_env.get("CC")
    if cc_env is not None and not cc_env.strip():
        return _skip("CC is set but empty")
    compiler = find_c_compiler()
    if compiler is None:
        return _skip(
            "no usable C compiler found on PATH (tried $CC, sysconfig, cc, "
            "clang, gcc). Install: apt-get install -y gcc (Linux) or "
            "xcode-select --install (macOS)"
        )
    cc_cmd, cc_extra = compiler

    python_inc = sysconfig.get_config_var("INCLUDEPY") or ""
    python_h = Path(python_inc) / "Python.h" if python_inc else None
    if not python_h or not python_h.exists():
        return _skip(
            f"Python.h not found at {python_inc}. Install development headers: "
            "apt-get install -y python3-dev (Linux)"
        )

    if system == "Windows":
        return _skip("Windows is not supported for C extension compilation")

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_name = f"{spec.output_stem}{ext_suffix}"
    out_path = pkg_dir / out_name

    numpy_inc = np.get_include()
    include_dirs = [python_inc, numpy_inc]
    include_dirs.extend(str(pkg_dir.joinpath(*parts)) for parts in spec.include_parts)

    omp_compile, omp_link, cc_cmd = detect_openmp_flags(
        cc_cmd, system, verbose_print, _warn=error_print
    )

    diag_flags = tuple(
        _diagnose_flags(cc_cmd) if spec.supports_diagnose and diagnose else ()
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
    ldflags = list(LINK_FLAGS_BY_PLATFORM.get(system, ()))

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{spec.output_stem.lstrip('_')}_build_"))
    try:
        result = compile_jlinalg(
            sources=sources,
            lapack_sources=lapack_sources,
            include_dirs=include_dirs,
            cc_cmd=cc_cmd,
            cc_extra=cc_extra,
            omp_compile=omp_compile,
            omp_link=omp_link,
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

    return BuildOutcome(
        ok=result.success,
        result=result,
        output_path=out_path,
        out_name=out_name,
    )
