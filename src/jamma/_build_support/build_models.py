"""Immutable build specifications, source manifests, and flag policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

# ---------------------------------------------------------------------------
# Data constants — THE single source of truth. The three entry points
# (hatch_build.py, _compile_jlinalg.py, _compile_accel.py) reach these through
# the compile_and_link facade rather than importing them directly, but this
# module owns the values: no flag literal may live elsewhere, which
# scripts/check_compile_flag_literals.py enforces against the entry points.
# ---------------------------------------------------------------------------

# Default source names — callers supply their own source directory.
BASELINE_SOURCES: tuple[str, ...] = (
    "platform.c",
    "pymodule.c",
    "blas_dispatch.c",
    "blas_operations.c",
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
    "_lmm_accel_ncvt1.c",
    "_lmm_accel_general.c",
    "_lmm_support.c",
    "_lmm_stats.c",
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

# Every target in this tree is a Python extension module, so every link is a
# shared-library link. Both entry points build one; nothing links an executable.
SHARED_LINK_FLAGS: tuple[str, ...] = ("-shared", "-fPIC")

# Platform-default link flags, placed before the OpenMP runtime on the link line.
LINK_FLAGS_BY_PLATFORM: dict[str, tuple[str, ...]] = {
    "Linux": ("-ldl", "-lpthread"),
    "Darwin": ("-undefined", "dynamic_lookup"),
}

# Libraries every extension links, placed after the OpenMP runtime.
LINK_LIBS: tuple[str, ...] = ("-lm",)


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
    compiler, the Python/NumPy include dirs, and OpenMP) are discovered by the
    caller and passed to ``run_build``, so the same spec drives the portable
    wheel build and the ``-march=native`` dev rebuild alike.

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
    # Runtime load identity — used by core.recompile._load_c_module and
    # auto_recompile_c_extension when a stale/missing .so must be reimported or
    # rebuilt. Stored rather than derived so tests can inject synthetic keys.
    sys_module_key: str
    fallback_label: str
    # Dev-mode-only base cflags. The wheel path never applies these.
    dev_extra_cflags: tuple[str, ...] = ()  # ("-march=native",) / ()
    reads_sentinel_env: bool = False  # honour JAMMA_SENTINEL_UB (accel only)
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
    sys_module_key="jamma.lmm._lmm_accel",
    fallback_label="LMM",
    required_attrs=(
        "HAS_OPENMP",
        "create_workspace_c",
        "compute_lmm_chunk_c",
        "workspace_sizes_c",
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
    sys_module_key="jamma.jlinalg._jlinalg",
    fallback_label="jlinalg",
    required_attrs=(
        "HAS_OPENMP",
        "blas_backend",
        "blas_has_dgemm",
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
        "set_n_threads",
    ),
)

# Opt-in sentinel macro for the sanitizer-workflow self-test. When
# JAMMA_SENTINEL_UB is set, _lmm_accel.c's gated heap-OOB function
# jamma_sentinel_oob is compiled in so ASAN can be proven to catch a real bug.
# A -D preprocessor macro, not an -O/-f flag, so the compile-flag-literal lint
# does not cover it; the named constant keeps it greppable. Wheel builds never
# set the env var.
_SENTINEL_UB_DEFINE = "-DJAMMA_SENTINEL_UB"


def _env_on(env: Mapping[str, str], name: str) -> bool:
    """Presence-based truthiness: "" and "0" are off, anything else on.

    Mirrors ``jamma.core.constants.env_flag``, which this module cannot import
    under PEP 517 build isolation.
    """
    return env.get(name, "").strip() not in ("", "0")


# ---------------------------------------------------------------------------
# BuildReport, BuildResult — the output channel and outcome of one build
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildReport:
    """Where build output goes.

    ``detail`` receives verbose-only progress (compiler command lines, OpenMP
    probing). ``warn`` receives what must always be visible: compile
    failures with their stderr, OpenMP retries, and unsafe OpenMP setups.
    """

    detail: Callable[[str], None]
    warn: Callable[[str], None]

    @classmethod
    def to_stream(cls, stream: TextIO, *, verbose: bool) -> BuildReport:
        """Print ``warn`` to ``stream``, and ``detail`` too when ``verbose``."""

        def say(message: str) -> None:
            print(message, file=stream, flush=True)

        return cls(detail=say if verbose else lambda _message: None, warn=say)


BuildPhase = Literal["preflight", "build", "ok"]


@dataclass(frozen=True)
class BuildResult:
    """Result of one build. The builder prints nothing about the outcome.

    ``phase`` says how far the build got: ``"preflight"`` for a guard firing
    before any source was touched (missing sources), ``"build"`` for a
    compile, link, or atomic-publish failure, and ``"ok"`` for success.
    ``error`` is ``""`` on success and names the failing stage otherwise.
    ``output_path`` is set only when ``phase == "ok"``.
    """

    phase: BuildPhase
    error: str = ""
    output_path: Path | None = None
    used_openmp: bool = False
    used_openmp_link: bool = False

    @property
    def ok(self) -> bool:
        return self.phase == "ok"


# ---------------------------------------------------------------------------
# ResolvedFlags — every spec- and environment-dependent flag, resolved once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedFlags:
    """The spec- and environment-dependent flags for one build.

    ``base_extra`` is spliced into BASE_CFLAGS before ``-fno-finite-math-only``.
    ``lapack_extra`` is appended to LAPACK_CFLAGS and is non-empty only for a
    sanitizer build. ``platform_link`` precedes the OpenMP runtime on the link
    line and ``link_libs`` follows it.
    """

    base_extra: tuple[str, ...]
    lapack_extra: tuple[str, ...]
    platform_link: tuple[str, ...]
    link_libs: tuple[str, ...]


def resolve_flags(
    spec: BuildSpec,
    *,
    dev_mode: bool,
    system: str,
    env: Mapping[str, str],
) -> ResolvedFlags:
    """Resolve every flag that depends on the spec, the platform, or ``env``.

    Pure: it reads only its arguments, so a test asserts on it with zero mocks.

    Wheel path (``dev_mode=False``): honour ``CFLAGS`` and nothing else, never
    ``-march=native``, so the wheel stays portable. Dev path: the spec's
    ``dev_extra_cflags`` (``-march=native`` for the accelerator), then the
    sentinel macro when ``JAMMA_SENTINEL_UB`` is set.

    ``JAMMA_SANITIZE`` (comma-separated ``-fsanitize`` values such as
    ``"address,undefined"``) instruments every source, LAPACK included, and
    the link. Its trailing ``-O1`` wins over BASE_CFLAGS' ``-O3`` and
    LAPACK_CFLAGS' ``-O2`` because gcc/clang honour the last ``-O`` flag.
    """
    if dev_mode:
        base_extra = list(spec.dev_extra_cflags)
        if spec.reads_sentinel_env and _env_on(env, "JAMMA_SENTINEL_UB"):
            base_extra.append(_SENTINEL_UB_DEFINE)
    else:
        base_extra = env.get("CFLAGS", "").split()
    san_cflags: tuple[str, ...] = ()
    san_link: tuple[str, ...] = ()
    if _env_on(env, "JAMMA_SANITIZE"):
        sanitizers = env["JAMMA_SANITIZE"].strip()
        san_cflags = (f"-fsanitize={sanitizers}", "-fno-omit-frame-pointer", "-O1")
        san_link = (f"-fsanitize={sanitizers}",)
    return ResolvedFlags(
        base_extra=(*base_extra, *san_cflags),
        lapack_extra=san_cflags,
        platform_link=LINK_FLAGS_BY_PLATFORM.get(system, ()),
        link_libs=(*LINK_LIBS, *san_link),
    )


def resolve_cflags_for(
    flags: ResolvedFlags, include_dirs: Sequence[str], *, lapack: bool
) -> list[str]:
    """Return the compile flags for one source.

    BASE_CFLAGS ordering: ``[-O3, -ftree-vectorize, -fno-math-errno,
    -fno-trapping-math, -funroll-loops, *base_extra, -fno-finite-math-only,
    -Wframe-larger-than=..., -fPIC, -std=c11]``

    The ``base_extra`` insertion BEFORE ``-fno-finite-math-only`` is
    load-bearing: user CFLAGS may contain ``-Ofast`` (which implies
    ``-ffinite-math-only``), and the trailing explicit ``-fno-finite-math-only``
    must override it so isnan() keeps working. DO NOT change this order.

    LAPACK sources deliberately do NOT splice ``base_extra``: they are strict
    IEEE 754, and a user-supplied ``-Ofast`` would defeat that split. They take
    only ``lapack_extra``, the sanitizer flags, none of which break IEEE 754
    rounding.
    """
    include_flags = [f"-I{d}" for d in include_dirs]
    if lapack:
        return [*LAPACK_CFLAGS, *flags.lapack_extra, *include_flags]
    # Slice BASE_CFLAGS rather than re-listing literals, so a flag added there
    # is never dropped on this path.
    splice_idx = BASE_CFLAGS.index("-fno-finite-math-only")
    return [
        *BASE_CFLAGS[:splice_idx],
        *flags.base_extra,
        *BASE_CFLAGS[splice_idx:],
        *include_flags,
    ]
