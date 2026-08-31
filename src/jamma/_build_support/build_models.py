"""Immutable build specifications, source manifests, and flag policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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
    # Dev-mode-only base cflags. The wheel path never applies these.
    dev_extra_cflags: tuple[str, ...] = ()  # ("-march=native",) / ()
    reads_sentinel_env: bool = False  # honour JAMMA_SENTINEL_UB (accel only)
    supports_diagnose: bool = False  # accept the vectorization-report flags
    # Runtime load identity — used by core.recompile._load_c_module and
    # auto_recompile_c_extension when a stale/missing .so must be reimported or
    # rebuilt. Stored rather than derived so tests can inject synthetic keys.
    module_name: str = ""  # log name, e.g. "_lmm_accel"
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
    sys_module_key="jamma.lmm._lmm_accel",
    fallback_label="LMM",
    required_attrs=(
        "HAS_OPENMP",
        "create_workspace_ncvt1_c",
        "compute_lmm_chunk_ncvt1_c",
        "create_workspace_general_c",
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
    unset, empty, or ``"0"``, returns the inputs unchanged plus an empty
    ``lapack_cflags`` — the same presence-based truthiness every other
    ``JAMMA_*`` toggle uses (``jamma.core.constants.env_flag``, mirrored here
    as ``_sentinel_env_on`` above because this module cannot import the
    runtime ``jamma`` package). Reads ``os.environ`` ONCE per call;
    ``run_build`` is the one production caller (every compile entry point,
    including runtime recompile via ``core/recompile.py``, routes through
    it). No caller may duplicate the env-var read — that would defeat the
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
    if sanitizers in ("", "0"):
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
        filename. The dict form lets callers specify per-source includes.
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
