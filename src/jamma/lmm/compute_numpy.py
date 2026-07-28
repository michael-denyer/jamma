"""NumPy mode dispatch for LMM chunk computation.

Dispatches to C extension (_lmm_accel) for Wald (batch/workspace/split),
Score (batch), LRT (batch), and fused mode-4 when available. Supports
n_cvt=1 (split/batch paths) and n_cvt>1 up to 100 (general workspace path).
Falls back to NumPy Python path when C functions are unavailable or n_cvt>100.
Also exports split-workspace and general-workspace APIs for direct use
by runners.

The caller is responsible for:
- Computing Uab in the appropriate format: Uab_batch (n_snps, n_samples,
  n_index) for chunk dispatch, or SoA-layout arrays (uab_varying_soa,
  uab_invariant_soa) for workspace-based paths.
- There is no async dispatch in the NumPy backend — results are immediately
  available after the call returns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict, TypeVar

import numpy as np

from jamma.core.constants import env_flag
from jamma.lmm.likelihood_numpy import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    batch_calc_wald_stats_from_pab_numpy,
    batch_compute_iab_numpy,
    compute_iab_invariant_scalars_ncvt1,
    golden_section_optimize_lambda_mle_numpy,
    golden_section_optimize_lambda_numpy,
    golden_section_optimize_lambda_split_ncvt1_numpy,
)

if TYPE_CHECKING:
    from jamma.lmm.dispatch import DispatchPath, KernelCaps

_EXPECTED_ABI_VERSION = 11  # Must match ABI_VERSION in _lmm_accel.c
MAX_C_N_CVT = 100  # Must match MAX_N_CVT in _lmm_accel.c


class AccelImport(NamedTuple):
    """C extension import result — zero-cost named tuple.

    Fields match the positional unpack at module level. NamedTuple is a tuple
    subclass, so existing destructuring continues to work unchanged.
    """

    accel_available: bool
    split_available: bool
    general_available: bool
    has_openmp: bool
    mode4_available: bool
    compute_batch_c: Callable[..., Any] | None
    compute_batch_split_c: Callable[..., Any] | None
    create_workspace_split_c: Callable[..., Any] | None
    compute_lmm_chunk_split_c: Callable[..., Any] | None
    create_workspace_general_c: Callable[..., Any] | None
    compute_lmm_chunk_general_c: Callable[..., Any] | None
    compute_score_batch_c: Callable[..., Any] | None
    compute_lrt_batch_c: Callable[..., Any] | None
    create_workspace_mode4_split_c: Callable[..., Any] | None
    compute_mode4_chunk_split_c: Callable[..., Any] | None
    compute_score_batch_general_c: Callable[..., Any] | None
    compute_lrt_batch_general_c: Callable[..., Any] | None
    compute_score_split_c: Callable[..., Any] | None
    compute_lrt_split_c: Callable[..., Any] | None
    compute_score_split_general_c: Callable[..., Any] | None
    compute_lrt_split_general_c: Callable[..., Any] | None
    compute_score_fused_c: Callable[..., Any] | None
    compute_lrt_fused_c: Callable[..., Any] | None
    create_workspace_fused_c: Callable[..., Any] | None
    compute_lmm_chunk_fused_c: Callable[..., Any] | None
    create_workspace_mode4_fused_c: Callable[..., Any] | None
    compute_mode4_chunk_fused_c: Callable[..., Any] | None
    create_workspace_fused_general_c: Callable[..., Any] | None
    compute_lmm_chunk_fused_general_c: Callable[..., Any] | None
    create_workspace_mode4_fused_general_c: Callable[..., Any] | None
    compute_mode4_chunk_fused_general_c: Callable[..., Any] | None
    create_workspace_score_fused_c: Callable[..., Any] | None
    compute_score_fused_ws_c: Callable[..., Any] | None
    create_workspace_lrt_fused_c: Callable[..., Any] | None
    compute_lrt_fused_ws_c: Callable[..., Any] | None


# The five availability flags vs the thirty object-valued symbol fields. Both
# the all-unavailable sentinel and the loader fill object fields with None and
# flags with False, derived from AccelImport._fields so a new field can't be
# forgotten in one place.
_FLAG_FIELDS = (
    "accel_available",
    "split_available",
    "general_available",
    "has_openmp",
    "mode4_available",
)
_OBJECT_FIELDS = tuple(f for f in AccelImport._fields if f not in _FLAG_FIELDS)

# Build the sentinel from the field split (flags False, symbols None) so a new
# field can't be forgotten. The intermediate is typed dict[str, Any] so the **
# spread type-checks: pyrefly can't map spread keys to params, so a narrowly
# typed dict (e.g. dict[str, bool] from fromkeys(..., False)) would flag False
# against the Callable symbol fields. Any keeps the runtime values exact while
# letting the unpack pass — mirroring the loader's **symbols spread below.
_unavailable_fields: dict[str, Any] = {
    **dict.fromkeys(_FLAG_FIELDS, False),
    **dict.fromkeys(_OBJECT_FIELDS, None),
}
_ACCEL_UNAVAILABLE = AccelImport(**_unavailable_fields)

# Core split symbols a valid ABI build always exports. Maps AccelImport field
# name -> C symbol name (they differ only for the two "batch" entries).
_CORE_SYMBOLS = {
    "compute_batch_c": "compute_lmm_batch_c",
    "compute_batch_split_c": "compute_lmm_batch_split_c",
    "create_workspace_split_c": "create_workspace_split_c",
    "compute_lmm_chunk_split_c": "compute_lmm_chunk_split_c",
}


class _OptionalGroup(NamedTuple):
    """One optional C-symbol group loaded after the ABI-validated core import.

    Every optional symbol shares its AccelImport field name with its C symbol
    name, so a group is just the field names plus how to report it missing. A
    group loads as a unit (the build exports them together per ABI level); if
    any member is absent the whole group stays unbound and is logged once.
    ``level`` "error" marks symbols a valid ABI v11 build must export, so their
    absence signals a corrupt build.
    """

    fields: tuple[str, ...]
    level: Literal["warning", "debug", "error"]
    message: str


# Optional capabilities in ABI order (matches the historical load sequence).
_OPTIONAL_GROUPS: tuple[_OptionalGroup, ...] = (
    _OptionalGroup(
        ("create_workspace_general_c", "compute_lmm_chunk_general_c"),
        "warning",
        "C extension missing general n_cvt symbols "
        "(create_workspace_general_c / compute_lmm_chunk_general_c). "
        "Falling back to Python path for n_cvt > 1.",
    ),
    _OptionalGroup(
        ("compute_score_batch_c",),
        "warning",
        "C extension missing compute_score_batch_c. Score will use Python path.",
    ),
    _OptionalGroup(
        ("compute_lrt_batch_c",),
        "warning",
        "C extension missing compute_lrt_batch_c. LRT will use Python path.",
    ),
    _OptionalGroup(
        ("create_workspace_mode4_split_c", "compute_mode4_chunk_split_c"),
        "warning",
        "C extension missing mode-4 fused functions. "
        "Mode 4 will use reconstruct+compose fallback.",
    ),
    _OptionalGroup(
        ("compute_score_batch_general_c",),
        "warning",
        "C extension missing compute_score_batch_general_c. "
        "Score for n_cvt>1 will use Python path.",
    ),
    _OptionalGroup(
        ("compute_lrt_batch_general_c",),
        "warning",
        "C extension missing compute_lrt_batch_general_c. "
        "LRT for n_cvt>1 will use Python path.",
    ),
    _OptionalGroup(
        ("compute_score_split_c",),
        "warning",
        "C extension missing compute_score_split_c. "
        "Score split will fall back to reconstruct_uab_from_soa.",
    ),
    _OptionalGroup(
        ("compute_lrt_split_c",),
        "warning",
        "C extension missing compute_lrt_split_c. "
        "LRT split will fall back to reconstruct_uab_from_soa.",
    ),
    _OptionalGroup(
        ("compute_score_split_general_c",),
        "debug",
        "C extension missing compute_score_split_general_c. "
        "Score split for n_cvt>1 will fall back to reconstruct_uab_from_soa.",
    ),
    _OptionalGroup(
        ("compute_lrt_split_general_c",),
        "debug",
        "C extension missing compute_lrt_split_general_c. "
        "LRT split for n_cvt>1 will fall back to reconstruct_uab_from_soa.",
    ),
    _OptionalGroup(
        ("compute_score_fused_c",),
        "warning",
        "C extension missing compute_score_fused_c. "
        "Score fused will fall back to split path.",
    ),
    _OptionalGroup(
        ("compute_lrt_fused_c",),
        "warning",
        "C extension missing compute_lrt_fused_c. "
        "LRT fused will fall back to split path.",
    ),
    _OptionalGroup(
        ("create_workspace_score_fused_c",),
        "debug",
        "C extension missing create_workspace_score_fused_c.",
    ),
    _OptionalGroup(
        ("compute_score_fused_ws_c",),
        "debug",
        "C extension missing compute_score_fused_ws_c.",
    ),
    _OptionalGroup(
        ("create_workspace_lrt_fused_c",),
        "debug",
        "C extension missing create_workspace_lrt_fused_c.",
    ),
    _OptionalGroup(
        ("compute_lrt_fused_ws_c",),
        "debug",
        "C extension missing compute_lrt_fused_ws_c.",
    ),
    _OptionalGroup(
        (
            "create_workspace_fused_c",
            "compute_lmm_chunk_fused_c",
            "create_workspace_mode4_fused_c",
            "compute_mode4_chunk_fused_c",
        ),
        "error",
        "C extension ABI validated but fused Uab symbols missing. This indicates "
        "build corruption — recompile: python -m jamma.lmm._compile_accel",
    ),
    _OptionalGroup(
        (
            "create_workspace_fused_general_c",
            "compute_lmm_chunk_fused_general_c",
            "create_workspace_mode4_fused_general_c",
            "compute_mode4_chunk_fused_general_c",
        ),
        "error",
        "C extension ABI validated but fused general symbols missing. This "
        "indicates build corruption — recompile: python -m jamma.lmm._compile_accel",
    ),
)


def _try_import_accel() -> AccelImport:
    """Attempt to import the C extension and validate ABI version.

    Honours ``JAMMA_FORCE_NUMPY_FALLBACK`` (Phase 116.1) — when truthy
    (anything other than "" or "0"), returns ``_ACCEL_UNAVAILABLE``
    without attempting the .so import. The ASAN/UBSAN sanitizer workflow
    sets this so ``dlopen`` never runs (RESEARCH §"Pitfall 4": ASAN +
    dlopen interaction can produce false-positive heap-buffer-overflow
    reports inside dispatched BLAS calls).

    Returns:
        AccelImport with availability flags and C function references
        (None when unavailable).
    """
    # Phase 116.1: same convention as jamma.jlinalg.__init__ — truthy values
    # are anything other than "", "0".
    if env_flag("JAMMA_FORCE_NUMPY_FALLBACK"):
        return _ACCEL_UNAVAILABLE

    try:
        import jamma.lmm._lmm_accel as mod
    except ImportError as e:
        from loguru import logger

        logger.debug(f"C extension import failed: {e}")
        return _ACCEL_UNAVAILABLE

    # The ABI-validated core: version, OpenMP flag, and the split symbols a
    # valid build always exports. Their absence is treated as an import failure.
    abi = getattr(mod, "ABI_VERSION", None)
    has_omp = getattr(mod, "HAS_OPENMP", None)
    core = {field: getattr(mod, csym, None) for field, csym in _CORE_SYMBOLS.items()}
    if abi is None or has_omp is None or any(v is None for v in core.values()):
        from loguru import logger

        logger.debug("C extension import failed: ABI/core symbols missing")
        return _ACCEL_UNAVAILABLE

    if abi != _EXPECTED_ABI_VERSION:
        from loguru import logger

        logger.warning(
            "C extension ABI mismatch: "
            f"compiled={abi}, expected={_EXPECTED_ABI_VERSION}. "
            "Stale .so needs recompilation."
        )
        return _ACCEL_UNAVAILABLE

    from loguru import logger

    # Load each optional group by name; bind it only if every member is present,
    # else leave the fields None and log once at the group's level.
    symbols = dict.fromkeys(_OBJECT_FIELDS, None)
    symbols.update(core)
    for group in _OPTIONAL_GROUPS:
        loaded = {field: getattr(mod, field, None) for field in group.fields}
        if all(v is not None for v in loaded.values()):
            symbols.update(loaded)
        elif group.level == "error":
            logger.error(group.message)
        elif group.level == "warning":
            logger.warning(group.message)
        else:
            logger.debug(group.message)

    return AccelImport(
        accel_available=True,
        split_available=True,
        general_available=symbols["create_workspace_general_c"] is not None
        and symbols["compute_lmm_chunk_general_c"] is not None,
        has_openmp=has_omp,
        mode4_available=symbols["create_workspace_mode4_split_c"] is not None
        and symbols["compute_mode4_chunk_split_c"] is not None,
        **symbols,
    )


def _auto_recompile() -> bool:
    """Auto-recompile the LMM C extension and reimport into sys.modules."""
    from jamma.core.recompile import auto_recompile_c_extension

    return auto_recompile_c_extension(
        module_name="_lmm_accel",
        compiler_module="jamma.lmm._compile_accel",
        sys_module_key="jamma.lmm._lmm_accel",
        label="LMM",
    )


_FORCE_NUMPY_FALLBACK = env_flag("JAMMA_FORCE_NUMPY_FALLBACK")

# Auto-recompile and retry once if the C extension is unavailable. Phase 116.1:
# when JAMMA_FORCE_NUMPY_FALLBACK is set, skip the retry — auto_recompile would
# compile the .so and import it into sys.modules, defeating the gate's purpose
# (RESEARCH §"Pitfall 4": ASAN must never see the .so loaded).
_accel = _try_import_accel()
if not _accel.accel_available and not _FORCE_NUMPY_FALLBACK and _auto_recompile():
    _accel = _try_import_accel()

# Bind module-level names from the (possibly retried) AccelImport exactly once.
# A single bind point means the initial-load and retry paths cannot drift —
# adding a C symbol touches only AccelImport and this block.
_C_ACCEL_AVAILABLE = _accel.accel_available
_C_SPLIT_AVAILABLE = _accel.split_available
_C_GENERAL_AVAILABLE = _accel.general_available
_C_HAS_OPENMP = _accel.has_openmp
_C_MODE4_AVAILABLE = _accel.mode4_available
_compute_lmm_batch_c = _accel.compute_batch_c
_compute_lmm_batch_split_c = _accel.compute_batch_split_c
_create_workspace_split_c = _accel.create_workspace_split_c
_compute_lmm_chunk_split_c = _accel.compute_lmm_chunk_split_c
_create_workspace_general_c = _accel.create_workspace_general_c
_compute_lmm_chunk_general_c = _accel.compute_lmm_chunk_general_c
_compute_score_batch_c = _accel.compute_score_batch_c
_compute_lrt_batch_c = _accel.compute_lrt_batch_c
_create_workspace_mode4_split_c = _accel.create_workspace_mode4_split_c
_compute_mode4_chunk_split_c = _accel.compute_mode4_chunk_split_c
_compute_score_batch_general_c = _accel.compute_score_batch_general_c
_compute_lrt_batch_general_c = _accel.compute_lrt_batch_general_c
_compute_score_split_c = _accel.compute_score_split_c
_compute_lrt_split_c = _accel.compute_lrt_split_c
_compute_score_split_general_c = _accel.compute_score_split_general_c
_compute_lrt_split_general_c = _accel.compute_lrt_split_general_c
_compute_score_fused_c = _accel.compute_score_fused_c
_compute_lrt_fused_c = _accel.compute_lrt_fused_c
_create_workspace_fused_c = _accel.create_workspace_fused_c
_compute_lmm_chunk_fused_c = _accel.compute_lmm_chunk_fused_c
_create_workspace_mode4_fused_c = _accel.create_workspace_mode4_fused_c
_compute_mode4_chunk_fused_c = _accel.compute_mode4_chunk_fused_c
_create_workspace_fused_general_c = _accel.create_workspace_fused_general_c
_compute_lmm_chunk_fused_general_c = _accel.compute_lmm_chunk_fused_general_c
_create_workspace_mode4_fused_general_c = _accel.create_workspace_mode4_fused_general_c
_compute_mode4_chunk_fused_general_c = _accel.compute_mode4_chunk_fused_general_c
_create_workspace_score_fused_c = _accel.create_workspace_score_fused_c
_compute_score_fused_ws_c = _accel.compute_score_fused_ws_c
_create_workspace_lrt_fused_c = _accel.create_workspace_lrt_fused_c
_compute_lrt_fused_ws_c = _accel.compute_lrt_fused_ws_c

if not _C_ACCEL_AVAILABLE and not _FORCE_NUMPY_FALLBACK:
    from loguru import logger as _logger

    _logger.warning(
        "C extension _lmm_accel not available — using pure-Python path "
        "(LMM may be slower without C extension; magnitude depends on "
        "dataset size and core count). To compile, run: "
        "python -m jamma.lmm._compile_accel"
    )
    del _logger
    _C_HAS_OPENMP = False
    _C_GENERAL_AVAILABLE = False
    _C_MODE4_AVAILABLE = False

_C_FUSED_AVAILABLE = _create_workspace_fused_c is not None
_C_MODE4_FUSED_AVAILABLE = _create_workspace_mode4_fused_c is not None
_C_FUSED_GENERAL_AVAILABLE = _create_workspace_fused_general_c is not None
_C_MODE4_FUSED_GENERAL_AVAILABLE = _create_workspace_mode4_fused_general_c is not None
_C_SCORE_FUSED_AVAILABLE = _compute_score_fused_c is not None
_C_LRT_FUSED_AVAILABLE = _compute_lrt_fused_c is not None
_C_SCORE_FUSED_WS_AVAILABLE = (
    _create_workspace_score_fused_c is not None
    and _compute_score_fused_ws_c is not None
)
_C_LRT_FUSED_WS_AVAILABLE = (
    _create_workspace_lrt_fused_c is not None and _compute_lrt_fused_ws_c is not None
)


def select_current_dispatch_path(
    n_cvt: int,
    lmm_mode: LmmMode,
    *,
    log_choices: bool = True,
) -> DispatchPath:
    """Select the dispatch path from the currently loaded C capabilities."""
    from jamma.lmm.dispatch import select_dispatch_path

    return select_dispatch_path(
        n_cvt, lmm_mode, current_kernel_caps(), log_choices=log_choices
    )


def current_kernel_caps() -> KernelCaps:
    """Snapshot which optional C kernels the loaded build exports.

    Read at call time, not import time, so tests that toggle the
    ``_C_*_AVAILABLE`` flags to drive a dispatch path take effect.
    """
    from jamma.lmm.dispatch import KernelCaps  # deferred: circular dep

    return KernelCaps(
        split=_C_SPLIT_AVAILABLE,
        general=_C_GENERAL_AVAILABLE,
        fused=_C_FUSED_AVAILABLE,
        fused_general=_C_FUSED_GENERAL_AVAILABLE,
        mode4=_C_MODE4_AVAILABLE,
        mode4_fused=_C_MODE4_FUSED_AVAILABLE,
        mode4_fused_general=_C_MODE4_FUSED_GENERAL_AVAILABLE,
        score_fused=_C_SCORE_FUSED_AVAILABLE,
        score_fused_ws=_C_SCORE_FUSED_WS_AVAILABLE,
        lrt_fused=_C_LRT_FUSED_AVAILABLE,
        lrt_fused_ws=_C_LRT_FUSED_WS_AVAILABLE,
    )


def _require(
    symbol: Callable[..., Any] | None, what: str, abi: int
) -> Callable[..., Any]:
    """Return a bound C symbol, or raise naming what is missing and how to fix it.

    Every kernel entry point below needs the same guard, and hand-writing it
    each time drifted: some sites raised while others asserted, and an assert
    vanishes under ``python -O``, turning a clear diagnostic into a
    ``NoneType is not callable`` from inside the C call.

    Args:
        symbol: The module-level C symbol, or None when the build omits it.
        what: Human name of the capability, used in the error message.
        abi: Minimum ``_lmm_accel`` ABI version that exports the symbol.
    """
    if symbol is None:
        raise RuntimeError(
            f"{what} requires the _lmm_accel C extension with ABI version "
            f"{abi}+. Recompile: python -m jamma.lmm._compile_accel"
        )
    return symbol


class WaldResult(TypedDict):
    """Result dict from REML Wald pipeline (both C and Python paths)."""

    lambdas: np.ndarray
    logls: np.ndarray
    betas: np.ndarray
    ses: np.ndarray
    pwalds: np.ndarray


LmmMode = Literal[1, 2, 3, 4]


def _compute_wald_c(
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    Iab_batch: np.ndarray,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald via C extension (n_cvt=1 only).

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, 6).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (should be >= 20 for 1e-5 tolerance).
        Iab_batch: Pre-computed identity-weighted Pab (n_snps, 3, 6).
        n_threads: OpenMP thread count for the C extension.

    Returns:
        WaldResult with keys: lambdas, logls, betas, ses, pwalds.
    """
    return _require(_compute_lmm_batch_c, "Batch Wald C compute", 11)(
        eigenvalues,
        Uab_batch,
        Iab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
    )


def _compute_wald_split_c(
    eigenvalues: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    Iab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald via split-Uab C extension (n_cvt=1 only).

    Expects SoA layout arrays (no per-call transpose). Callers must pass
    arrays already in SoA layout from batch_compute_uab_split_soa_numpy or
    batch_compute_uab_varying_soa_numpy + compute_uab_invariant_soa.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_varying_soa: SNP-varying columns (n_snps, 3, n_samples) [wx, xx, xy].
        uab_invariant_soa: SNP-invariant columns (3, n_samples) [ww, wy, yy].
        Iab_batch: Pre-computed identity-weighted Pab (n_snps, 3, 6).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        WaldResult with keys: lambdas, logls, betas, ses, pwalds.
    """
    return _require(_compute_lmm_batch_split_c, "Batch split Wald C compute", 11)(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        Iab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
    )


def create_lmm_workspace(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> object:
    """Create a persistent C workspace for the split-Uab REML pipeline.

    The workspace holds precomputed lambda_grid, hi_eval_grid, logdet_h_grid,
    grid_inv, and invariant Iab column sums (iab_s_ww etc.). It is reused
    across all chunks — eliminating per-chunk C malloc and grid precomputation.

    Returns a PyCapsule that is freed automatically when it goes out of scope.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (3, n_samples) — rows [ww, wy, yy].
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        PyCapsule wrapping lmm_workspace_t (opaque; pass to compute_wald_split_c_ws).
    """
    return _require(_create_workspace_split_c, "Split C workspace", 11)(
        eigenvalues,
        uab_invariant_soa,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
    )


def compute_wald_split_c_ws(
    workspace: object,
    uab_varying_soa: np.ndarray,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald for one chunk using a pre-built workspace.

    Uses precomputed grids from create_lmm_workspace — no per-chunk malloc,
    no eigenvalue reprocessing, no invariant Iab computation. Iab logdet
    is computed inside C from the workspace's precomputed iab_s_ww.

    Args:
        workspace: PyCapsule from create_lmm_workspace.
        uab_varying_soa: SNP-varying Uab (n_snps, 3, n_samples) — SoA layout.
        n_threads: OpenMP thread count.

    Returns:
        WaldResult with keys: lambdas, logls, betas, ses, pwalds.
    """
    return _require(_compute_lmm_chunk_split_c, "Split Wald C compute", 11)(
        workspace, uab_varying_soa, n_threads
    )


def create_lmm_workspace_mode4(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    hi_eval_null: np.ndarray,
    logl_H0: float,
) -> object:
    """Create a persistent C workspace for fused mode-4 (Wald/Score/LRT).

    Extends the Wald workspace with null-model Hi_eval and MLE fields,
    enabling the fused kernel to compute all three test statistics in a
    single OpenMP loop without Uab reconstruction.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (3, n_samples) -- rows [ww, wy, yy].
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.
        hi_eval_null: Null-model Hi_eval (n_samples,).
        logl_H0: Null model MLE log-likelihood.

    Returns:
        PyCapsule wrapping mode-4 lmm_workspace_t.
    """
    return _require(_create_workspace_mode4_split_c, "Fused mode-4 C workspace", 6)(
        eigenvalues,
        uab_invariant_soa,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
        hi_eval_null,
        logl_H0,
    )


def compute_mode4_split_c_ws(
    workspace: object,
    uab_varying_soa: np.ndarray,
    n_threads: int,
) -> dict[str, np.ndarray]:
    """Compute fused mode-4 (Wald/Score/LRT) for one chunk using a workspace.

    Single-pass fused kernel: no Uab reconstruction, no separate Score/LRT
    calls. Returns all 8 output arrays directly from the C extension.

    Args:
        workspace: PyCapsule from create_lmm_workspace_mode4.
        uab_varying_soa: SNP-varying Uab (n_snps, 3, n_samples) -- SoA layout.
        n_threads: OpenMP thread count.

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds,
        p_scores, lambdas_mle, p_lrts.
    """
    return _require(_compute_mode4_chunk_split_c, "Fused mode-4 C compute", 6)(
        workspace, uab_varying_soa, n_threads
    )


def create_lmm_workspace_fused(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    w: np.ndarray,
    Uty: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> object:
    """Create fused workspace that holds w/Uty for on-the-fly Uab computation.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (3, n_samples) -- SoA [ww, wy, yy].
        w: UtW[:,0] (n_samples,).
        Uty: Rotated phenotype (n_samples,).
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        PyCapsule wrapping lmm_workspace_t (fused).
    """
    return _require(_create_workspace_fused_c, "Fused C workspace", 8)(
        eigenvalues,
        uab_invariant_soa,
        w,
        Uty,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
    )


def compute_wald_fused_c_ws(
    workspace: object,
    utg_t: np.ndarray,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald from UtG_T directly -- no uab_varying_soa needed.

    Args:
        workspace: PyCapsule from create_lmm_workspace_fused.
        utg_t: Rotated genotypes transposed (n_snps, n_samples).
        n_threads: OpenMP thread count.

    Returns:
        WaldResult dict with lambdas, logls, betas, ses, pwalds.
    """
    return _require(_compute_lmm_chunk_fused_c, "Fused C compute", 8)(
        workspace, utg_t, n_threads
    )


def create_lmm_workspace_mode4_fused(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    w: np.ndarray,
    Uty: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    *,
    hi_eval_null: np.ndarray,
    logl_H0: float,
) -> object:
    """Create fused mode-4 workspace with w/Uty + null model for Score/LRT.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (3, n_samples) -- SoA [ww, wy, yy].
        w: UtW[:,0] (n_samples,).
        Uty: Rotated phenotype (n_samples,).
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.
        hi_eval_null: Null-model Hi_eval (n_samples,).
        logl_H0: Null MLE log-likelihood.

    Returns:
        PyCapsule wrapping lmm_workspace_t (mode=4, fused).
    """
    return _require(_create_workspace_mode4_fused_c, "Fused mode-4 C workspace", 8)(
        eigenvalues,
        uab_invariant_soa,
        w,
        Uty,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
        hi_eval_null,
        logl_H0,
    )


def compute_mode4_fused_c_ws(
    workspace: object,
    utg_t: np.ndarray,
    n_threads: int,
) -> dict:
    """Compute fused mode-4 from UtG_T directly -- no uab_varying_soa needed.

    Args:
        workspace: PyCapsule from create_lmm_workspace_mode4_fused.
        utg_t: Rotated genotypes transposed (n_snps, n_samples).
        n_threads: OpenMP thread count.

    Returns:
        Dict with lambdas, logls, betas, ses, pwalds, p_scores,
        lambdas_mle, p_lrts.
    """
    return _require(_compute_mode4_chunk_fused_c, "Fused mode-4 C compute", 8)(
        workspace, utg_t, n_threads
    )


def create_lmm_workspace_fused_general(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    *,
    n_cvt: int,
    invariant_indices: np.ndarray,
    varying_indices: np.ndarray,
    logdet_diag_rows: np.ndarray,
    logdet_diag_cols: np.ndarray,
    level_offsets: np.ndarray,
    level_counts: np.ndarray,
    entries: np.ndarray,
    idx_xx: int,
    idx_xy: int,
    idx_yy: int,
    var_a_cols: np.ndarray,
    var_b_cols: np.ndarray,
) -> object:
    """Create fused general workspace for n_cvt >= 2 Wald computation.

    Stores UtW (transposed to column-major), Uty, and var_a/b_cols for
    on-the-fly varying Uab computation from UtG_T.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (n_inv, n_samples) -- SoA layout.
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.
        n_cvt: Number of covariates.
        invariant_indices: Invariant column indices (n_inv,) int32.
        varying_indices: Varying column indices (n_var,) int32.
        logdet_diag_rows: Logdet diagonal rows (n_cvt+1,) int32.
        logdet_diag_cols: Logdet diagonal cols (n_cvt+1,) int32.
        level_offsets: Level offsets (n_rows,) int32.
        level_counts: Level counts (n_rows,) int32.
        entries: Recursion entries (n_entries*4,) int32 stride-4.
        idx_xx: Genotype-genotype index.
        idx_xy: Genotype-phenotype index.
        idx_yy: Phenotype-phenotype index.
        var_a_cols: Column a for each varying pair (n_var,) int32.
        var_b_cols: Column b for each varying pair (n_var,) int32.

    Returns:
        PyCapsule wrapping lmm_workspace_general_t (fused).
    """
    return _require(_create_workspace_fused_general_c, "Fused general C workspace", 9)(
        eigenvalues,
        uab_invariant_soa,
        UtW,
        Uty,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
        n_cvt,
        invariant_indices,
        varying_indices,
        logdet_diag_rows,
        logdet_diag_cols,
        level_offsets,
        level_counts,
        entries,
        idx_xx,
        idx_xy,
        idx_yy,
        var_a_cols,
        var_b_cols,
    )


def compute_wald_fused_general_c_ws(
    workspace: object,
    utg_t: np.ndarray,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald from UtG_T using fused general workspace.

    Args:
        workspace: PyCapsule from create_lmm_workspace_fused_general.
        utg_t: Rotated genotypes transposed (n_snps, n_samples).
        n_threads: OpenMP thread count.

    Returns:
        WaldResult dict with lambdas, logls, betas, ses, pwalds.
    """
    return _require(_compute_lmm_chunk_fused_general_c, "Fused general C compute", 9)(
        workspace, utg_t, n_threads
    )


def create_lmm_workspace_mode4_fused_general(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    *,
    n_cvt: int,
    invariant_indices: np.ndarray,
    varying_indices: np.ndarray,
    logdet_diag_rows: np.ndarray,
    logdet_diag_cols: np.ndarray,
    level_offsets: np.ndarray,
    level_counts: np.ndarray,
    entries: np.ndarray,
    idx_xx: int,
    idx_xy: int,
    idx_yy: int,
    var_a_cols: np.ndarray,
    var_b_cols: np.ndarray,
    hi_eval_null: np.ndarray,
    logl_H0: float,
) -> object:
    """Create mode-4 fused general workspace for n_cvt >= 2.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (n_inv, n_samples) -- SoA layout.
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.
        n_cvt: Number of covariates.
        invariant_indices: Invariant column indices (n_inv,) int32.
        varying_indices: Varying column indices (n_var,) int32.
        logdet_diag_rows: Logdet diagonal rows (n_cvt+1,) int32.
        logdet_diag_cols: Logdet diagonal cols (n_cvt+1,) int32.
        level_offsets: Level offsets (n_rows,) int32.
        level_counts: Level counts (n_rows,) int32.
        entries: Recursion entries (n_entries*4,) int32 stride-4.
        idx_xx: Genotype-genotype index.
        idx_xy: Genotype-phenotype index.
        idx_yy: Phenotype-phenotype index.
        var_a_cols: Column a for each varying pair (n_var,) int32.
        var_b_cols: Column b for each varying pair (n_var,) int32.
        hi_eval_null: Null-model Hi_eval (n_samples,).
        logl_H0: Null MLE log-likelihood.

    Returns:
        PyCapsule wrapping lmm_workspace_general_t (mode=4, fused).
    """
    return _require(
        _create_workspace_mode4_fused_general_c, "Fused general mode-4 C workspace", 9
    )(
        eigenvalues,
        uab_invariant_soa,
        UtW,
        Uty,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
        n_cvt,
        invariant_indices,
        varying_indices,
        logdet_diag_rows,
        logdet_diag_cols,
        level_offsets,
        level_counts,
        entries,
        idx_xx,
        idx_xy,
        idx_yy,
        var_a_cols,
        var_b_cols,
        hi_eval_null,
        logl_H0,
    )


def compute_mode4_fused_general_c_ws(
    workspace: object,
    utg_t: np.ndarray,
    n_threads: int,
) -> dict:
    """Compute mode-4 (Wald+Score+LRT) from UtG_T using fused general workspace.

    Args:
        workspace: PyCapsule from create_lmm_workspace_mode4_fused_general.
        utg_t: Rotated genotypes transposed (n_snps, n_samples).
        n_threads: OpenMP thread count.

    Returns:
        Dict with lambdas, logls, betas, ses, pwalds, p_scores,
        lambdas_mle, p_lrts.
    """
    return _require(
        _compute_mode4_chunk_fused_general_c, "Fused general mode-4 C compute", 9
    )(workspace, utg_t, n_threads)


def create_lmm_workspace_general(
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    n_cvt: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> object:
    """Create a persistent C workspace for the general n_cvt REML pipeline.

    Builds the Pab recursion table via build_pab_table_for_c(), then passes
    all flat arrays to the C extension's create_workspace_general_c().

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab (n_inv, n_samples) — SoA layout.
        n_samples: Number of samples.
        n_cvt: Number of covariates.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Number of coarse grid points.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        PyCapsule wrapping lmm_workspace_general_t.
    """
    create = _require(_create_workspace_general_c, "General n_cvt C workspace", 4)

    from jamma.lmm.likelihood import build_pab_table_for_c

    table = build_pab_table_for_c(n_cvt)

    return create(
        eigenvalues,
        uab_invariant_soa,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        n_threads,
        n_cvt,
        table["invariant_indices"],
        table["varying_indices"],
        table["logdet_diag_rows"],
        table["logdet_diag_cols"],
        table["level_offsets"],
        table["level_counts"],
        table["entries"],
        table["idx_xx"],
        table["idx_xy"],
        table["idx_yy"],
    )


def compute_wald_general_c_ws(
    workspace: object,
    uab_varying_soa: np.ndarray,
    n_threads: int,
) -> WaldResult:
    """Compute REML Wald for one chunk using a general n_cvt workspace.

    Args:
        workspace: PyCapsule from create_lmm_workspace_general.
        uab_varying_soa: SNP-varying Uab (n_snps, n_var, n_samples) — SoA.
        n_threads: OpenMP thread count.

    Returns:
        WaldResult with keys: lambdas, logls, betas, ses, pwalds.
    """
    return _require(_compute_lmm_chunk_general_c, "General n_cvt C chunk compute", 4)(
        workspace, uab_varying_soa, n_threads
    )


def _compute_score_c(
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Hi_eval_null: np.ndarray,
    n_samples: int,
    n_threads: int,
) -> dict[str, np.ndarray]:
    """Compute Score test via C extension (n_cvt=1 only).

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, 6).
        Hi_eval_null: Pre-computed null-model Hi_eval (n_samples,).
        n_samples: Number of samples.
        n_threads: OpenMP thread count.

    Returns:
        Dict with keys: betas, ses, p_scores.
    """
    return _require(_compute_score_batch_c, "Batch Score C compute", 11)(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_threads,
    )


def _compute_lrt_c(
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict[str, np.ndarray]:
    """Compute LRT via C extension (n_cvt=1 only).

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, 6).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations.
        logl_H0: Null model MLE log-likelihood (scalar).
        n_threads: OpenMP thread count.

    Returns:
        Dict with keys: lambdas_mle, p_lrts.
    """
    return _require(_compute_lrt_batch_c, "Batch LRT C compute", 11)(
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        n_threads,
    )


def _compute_wald_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    Iab_batch: np.ndarray | None = None,
    n_threads: int = 1,
) -> WaldResult:
    """Compute REML-optimized Wald test statistics.

    Dispatches to C extension: n_cvt=1 uses the batch path (compute_lmm_batch_c),
    n_cvt>1 uses the general workspace path. Falls back to Python split path
    (n_cvt=1) or generic Python path (n_cvt>1) when C extension is unavailable.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (should be >= 20 for 1e-5 tolerance;
            C extension requires >= 1). Runner-level code enforces the minimum.
        Iab_batch: Pre-computed identity-weighted Pab. If None, computed internally.
        n_threads: OpenMP thread count passed to C extension (ignored on Python path).

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds.
    """

    if _C_ACCEL_AVAILABLE and n_cvt == 1:
        if Iab_batch is None:
            Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
        return _compute_wald_c(
            eigenvalues,
            Uab_batch,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            Iab_batch,
            n_threads,
        )

    if _C_GENERAL_AVAILABLE and 1 < n_cvt <= MAX_C_N_CVT:
        # Use C extension for general n_cvt via split-Uab workspace
        from jamma.lmm.likelihood import classify_uab_columns

        inv_indices, var_indices = classify_uab_columns(n_cvt)

        # Build invariant SoA from Uab_batch (shared across all SNPs, use SNP 0)
        # Note: a[0, :, list_idx] returns (n_inv, n_samples) due to numpy advanced
        # indexing rules (integer + list separated by slice -> grouped at front).
        inv_list = list(inv_indices)
        if __debug__ and Uab_batch.shape[0] > 1:
            # Verify columns classified as invariant are actually constant across SNPs
            inv_sample = Uab_batch[:, :, inv_list]
            if not np.allclose(inv_sample, inv_sample[0:1], rtol=1e-12, atol=0):
                raise RuntimeError(
                    f"classify_uab_columns({n_cvt}) classified varying columns as "
                    "invariant — internal error, please report with your n_cvt value"
                )
        uab_invariant_soa = np.ascontiguousarray(
            Uab_batch[0, :, inv_list]  # (n_inv, n_samples)
        )
        # Build varying SoA from Uab_batch
        uab_varying_soa = np.ascontiguousarray(
            Uab_batch[:, :, list(var_indices)].transpose(
                0, 2, 1
            )  # (n_snps, n_var, n_samples)
        )

        # Create per-call workspace and compute
        ws = create_lmm_workspace_general(
            eigenvalues,
            uab_invariant_soa,
            n_samples,
            n_cvt,
            l_min,
            l_max,
            n_grid,
            n_refine,
            n_threads,
        )
        return compute_wald_general_c_ws(ws, uab_varying_soa, n_threads)

    if n_cvt == 1:
        # Python split path for n_cvt=1: separate invariant (ww, wy, yy)
        # and varying (wx, xx, xy) Uab columns to reduce per-SNP computation.
        # Column layout: 0=ww, 1=wx, 2=wy, 3=xx, 4=xy, 5=yy.
        # Invariant columns (ww, wy, yy) are identical across SNPs — use SNP 0.
        uab_varying_soa = np.stack(
            [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
        )  # (n_snps, 3, n_samples): rows [wx, xx, xy]
        uab_invariant_soa = np.stack(
            [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
        )  # (3, n_samples): rows [ww, wy, yy]

        iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
            uab_invariant_soa
        )
        lambdas, logls, Pab_final = golden_section_optimize_lambda_split_ncvt1_numpy(
            eigenvalues,
            uab_varying_soa,
            uab_invariant_soa,
            iab_s_ww,
            iab_s_wy,
            iab_s_yy,
            iab_logdet,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_iter=n_refine,
        )
    else:
        # Generic Python path for n_cvt > 1
        if Iab_batch is None:
            Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
        lambdas, logls, Pab_final = golden_section_optimize_lambda_numpy(
            n_cvt,
            eigenvalues,
            Uab_batch,
            Iab_batch,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_iter=n_refine,
        )

    betas, ses, pwalds = batch_calc_wald_stats_from_pab_numpy(
        n_cvt, Pab_final, n_samples
    )
    return {
        "lambdas": lambdas,
        "logls": logls,
        "betas": betas,
        "ses": ses,
        "pwalds": pwalds,
    }


def _compute_lrt_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int = 1,
) -> dict[str, np.ndarray]:
    """Compute MLE-optimized LRT statistics.

    Dispatches to C extension when available:
    - n_cvt=1: uses the per-SNP batch path (compute_lrt_batch_c).
    - n_cvt>1: uses the general batch path (compute_lrt_batch_general_c) with
      pab_table_dict built from build_pab_table_for_c(n_cvt).
    Falls back to Python golden section optimizer when C is unavailable.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (should be >= 20 for 1e-5 tolerance).
        logl_H0: Null model MLE log-likelihood (scalar).
        n_threads: OpenMP thread count passed to C extension (ignored on Python path).

    Returns:
        Dict with keys: lambdas_mle, p_lrts.
    """
    if _compute_lrt_batch_c is not None and n_cvt == 1:
        return _compute_lrt_c(
            eigenvalues,
            Uab_batch,
            len(eigenvalues),
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            n_threads,
        )

    if _compute_lrt_batch_general_c is not None and n_cvt > 1:
        from jamma.lmm.likelihood import build_pab_table_for_c

        pab_table_dict = build_pab_table_for_c(n_cvt)
        return _compute_lrt_batch_general_c(
            eigenvalues,
            Uab_batch,
            len(eigenvalues),
            n_cvt,
            pab_table_dict,
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            n_threads,
        )

    from loguru import logger

    if _compute_lrt_batch_c is None:
        logger.debug("LRT using Python path (C extension unavailable)")
    else:
        logger.debug(
            "LRT using Python path (n_cvt={} > 1, general C unavailable)", n_cvt
        )

    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts = _batch_lrt_pvalues_numpy(logls_mle, logl_H0)
    return {"lambdas_mle": lambdas_mle, "p_lrts": p_lrts}


def _compute_score_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Hi_eval_null: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    n_threads: int = 1,
) -> dict[str, np.ndarray]:
    """Compute Score test statistics (no optimization needed).

    Dispatches to C extension when available:
    - n_cvt=1: uses the per-SNP batch path (compute_score_batch_c).
    - n_cvt>1: uses the general batch path (compute_score_batch_general_c) with
      pab_table_dict built from build_pab_table_for_c(n_cvt).
    Falls back to Python batch Score when C is unavailable.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,). Used by C path for validation.
        Hi_eval_null: Pre-computed null-model Hi_eval (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        n_threads: OpenMP thread count passed to C extension (ignored on Python path).

    Returns:
        Dict with keys: betas, ses, p_scores.
    """
    if _compute_score_batch_c is not None and n_cvt == 1:
        return _compute_score_c(
            eigenvalues,
            Uab_batch,
            Hi_eval_null,
            n_samples,
            n_threads,
        )

    if _compute_score_batch_general_c is not None and n_cvt > 1:
        from jamma.lmm.likelihood import build_pab_table_for_c

        pab_table_dict = build_pab_table_for_c(n_cvt)
        return _compute_score_batch_general_c(
            eigenvalues,
            Uab_batch,
            Hi_eval_null,
            n_samples,
            n_cvt,
            pab_table_dict,
            n_threads,
        )

    from loguru import logger

    if _compute_score_batch_c is None:
        logger.debug("Score using Python path (C extension unavailable)")
    else:
        logger.debug(
            "Score using Python path (n_cvt={} > 1, general C unavailable)", n_cvt
        )

    if not np.all(np.isfinite(Hi_eval_null)):
        bad_idx = np.where(~np.isfinite(Hi_eval_null))[0]
        raise ValueError(
            f"Hi_eval_null has {len(bad_idx)} non-finite value(s) at indices "
            f"{bad_idx[:5].tolist()}. Null model optimization may have failed."
        )
    if np.any(Hi_eval_null <= 0):
        bad_idx = np.where(Hi_eval_null <= 0)[0]
        raise ValueError(
            f"Hi_eval_null has {len(bad_idx)} non-positive value(s) at indices "
            f"{bad_idx[:5].tolist()}. Check kinship matrix conditioning."
        )

    betas, ses, p_scores = batch_calc_score_stats_numpy(
        n_cvt, Hi_eval_null, Uab_batch, n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _compute_score_split_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Hi_eval_null: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    n_threads: int = 1,
) -> dict[str, np.ndarray]:
    """Compute Score test from SoA split data (no full Uab reconstruction).

    Dispatches to C extension: compute_score_split_c for n_cvt=1,
    compute_score_split_general_c for n_cvt>1. Falls back to
    reconstruct_uab_from_soa + _compute_score_numpy when C is unavailable.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Hi_eval_null: Pre-computed null-model Hi_eval (n_samples,).
        uab_varying_soa: SNP-varying Uab (n_snps, n_var, n_samples) SoA.
        uab_invariant_soa: SNP-invariant Uab (n_inv, n_samples) SoA.
        n_samples: Number of samples.
        n_threads: OpenMP thread count.

    Returns:
        Dict with keys: betas, ses, p_scores.
    """
    if _compute_score_split_c is not None and n_cvt == 1:
        return _compute_score_split_c(
            eigenvalues,
            uab_varying_soa,
            uab_invariant_soa,
            Hi_eval_null,
            n_samples,
            n_threads,
        )

    if _compute_score_split_general_c is not None and n_cvt > 1:
        from jamma.lmm.likelihood import build_pab_table_for_c

        return _compute_score_split_general_c(
            eigenvalues,
            uab_varying_soa,
            uab_invariant_soa,
            Hi_eval_null,
            n_samples,
            n_cvt,
            build_pab_table_for_c(n_cvt),
            n_threads,
        )

    # Fallback: reconstruct full Uab and use batch dispatch
    from jamma.lmm.likelihood_numpy import reconstruct_uab_from_soa

    Uab_batch = reconstruct_uab_from_soa(
        uab_invariant_soa, uab_varying_soa, n_cvt=n_cvt
    )
    return _compute_score_numpy(
        n_cvt, eigenvalues, Hi_eval_null, Uab_batch, n_samples, n_threads
    )


def _compute_lrt_split_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int = 1,
) -> dict[str, np.ndarray]:
    """Compute LRT from SoA split data (no full Uab reconstruction).

    Dispatches to C extension: compute_lrt_split_c for n_cvt=1,
    compute_lrt_split_general_c for n_cvt>1. Falls back to
    reconstruct_uab_from_soa + _compute_lrt_numpy when C is unavailable.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_varying_soa: SNP-varying Uab (n_snps, n_var, n_samples) SoA.
        uab_invariant_soa: SNP-invariant Uab (n_inv, n_samples) SoA.
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations.
        logl_H0: Null model MLE log-likelihood.
        n_threads: OpenMP thread count.

    Returns:
        Dict with keys: lambdas_mle, p_lrts.
    """
    if _compute_lrt_split_c is not None and n_cvt == 1:
        return _compute_lrt_split_c(
            eigenvalues,
            uab_varying_soa,
            uab_invariant_soa,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            n_threads,
        )

    if _compute_lrt_split_general_c is not None and n_cvt > 1:
        from jamma.lmm.likelihood import build_pab_table_for_c

        return _compute_lrt_split_general_c(
            eigenvalues,
            uab_varying_soa,
            uab_invariant_soa,
            n_samples,
            n_cvt,
            build_pab_table_for_c(n_cvt),
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            n_threads,
        )

    # Fallback: reconstruct full Uab and use batch dispatch
    from jamma.lmm.likelihood_numpy import reconstruct_uab_from_soa

    Uab_batch = reconstruct_uab_from_soa(
        uab_invariant_soa, uab_varying_soa, n_cvt=n_cvt
    )
    return _compute_lrt_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        n_threads,
    )


_LOGL_H0_REQUIRED = "logl_H0 is required for LRT (mode 2) and All (mode 4)"
_HI_EVAL_NULL_REQUIRED = "Hi_eval_null is required for Score (mode 3) and All (mode 4)"

_ModeInput = TypeVar("_ModeInput")


def _require_mode_input(value: _ModeInput | None, message: str) -> _ModeInput:
    """Return a mode-specific input, or raise naming which mode needed it.

    Only some LMM modes take logl_H0 / Hi_eval_null, so both arrive optional.
    Checking them at the point of use ties the guard to the branch that reads
    the value, which a mode check made up front cannot express.

    Args:
        value: The optional input.
        message: Error text naming the input and the modes that require it.

    Returns:
        The value, known to be present.

    Raises:
        ValueError: If the value is None.
    """
    if value is None:
        raise ValueError(message)
    return value


def _store_wald(result: dict[str, np.ndarray | None], wald: WaldResult) -> None:
    """Copy a WaldResult's five arrays into the mode-agnostic result dict.

    Spelled out per key rather than ``result.update(wald)`` because a TypedDict
    is not a ``Mapping[str, ndarray | None]`` — its value types are per-key, so
    the update overloads reject it.

    Args:
        result: The chunk result dict to populate.
        wald: Wald statistics for the chunk.
    """
    result.update(
        lambdas=wald["lambdas"],
        logls=wald["logls"],
        betas=wald["betas"],
        ses=wald["ses"],
        pwalds=wald["pwalds"],
    )


def compute_lmm_chunk_numpy(
    lmm_mode: LmmMode,
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    *,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    Hi_eval_null: np.ndarray | None = None,
    logl_H0: float | None = None,
    n_threads: int = 1,
) -> dict[str, np.ndarray | None]:
    """Compute LMM statistics for a chunk of SNPs (NumPy backend).

    Computes LMM statistics for a chunk of SNPs using NumPy batch functions.
    No async dispatch — results are immediately available.

    Args:
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (minimum 20 enforced).
        Hi_eval_null: Pre-computed 1/(lambda_null*eval+1) for Score test.
        logl_H0: Null model MLE log-likelihood for LRT.
        n_threads: OpenMP thread count passed to C extension (Wald, LRT, Score).

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds,
        lambdas_mle, p_lrts, p_scores. Keys not relevant to the
        mode are set to None.
    """
    n_refine = max(n_refine, 20)

    result: dict[str, np.ndarray | None] = {
        "lambdas": None,
        "logls": None,
        "betas": None,
        "ses": None,
        "pwalds": None,
        "lambdas_mle": None,
        "p_lrts": None,
        "p_scores": None,
    }

    if lmm_mode == 1:
        _store_wald(
            result,
            _compute_wald_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                n_threads=n_threads,
            ),
        )

    elif lmm_mode == 2:
        result.update(
            _compute_lrt_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
                _require_mode_input(logl_H0, _LOGL_H0_REQUIRED),
                n_threads=n_threads,
            )
        )

    elif lmm_mode == 3:
        result.update(
            _compute_score_numpy(
                n_cvt,
                eigenvalues,
                _require_mode_input(Hi_eval_null, _HI_EVAL_NULL_REQUIRED),
                Uab_batch,
                n_samples,
                n_threads=n_threads,
            )
        )

    elif lmm_mode == 4:
        # Both mode-4 inputs are checked before any compute runs, so an omitted
        # one still fails before the Score step rather than partway through.
        # logl_H0 first: with both absent that is the one reported, which is
        # the order the previous up-front guards established.
        null_logl = _require_mode_input(logl_H0, _LOGL_H0_REQUIRED)
        hi_eval_null = _require_mode_input(Hi_eval_null, _HI_EVAL_NULL_REQUIRED)
        # Compose all three tests; only take p_scores from Score —
        # Wald provides REML-optimized beta/SE below
        score_result = _compute_score_numpy(
            n_cvt,
            eigenvalues,
            hi_eval_null,
            Uab_batch,
            n_samples,
            n_threads=n_threads,
        )
        result["p_scores"] = score_result["p_scores"]
        result.update(
            _compute_lrt_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
                null_logl,
                n_threads=n_threads,
            )
        )
        # Pre-compute Iab once for Wald (lambda-independent)
        Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
        _store_wald(
            result,
            _compute_wald_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                Iab_batch=Iab_batch,
                n_threads=n_threads,
            ),
        )

    else:
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    return result
