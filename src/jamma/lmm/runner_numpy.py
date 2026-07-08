"""Pure-NumPy batch LMM association runner.

Input genotypes must fit in memory.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import NamedTuple

import numpy as np
import psutil
from loguru import logger

from jamma import jlinalg
from jamma.core.estimates import estimate_lmm_seconds
from jamma.core.memory import estimate_lmm_memory
from jamma.core.progress import create_progress_bar, progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.core.threading import (
    blas_threads,
    get_c_extension_thread_count,
    get_physical_core_count,
    is_blas_controllable,
    jlinalg_threads,
)
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _C_FUSED_AVAILABLE,
    _C_GENERAL_AVAILABLE,
    _C_HAS_OPENMP,
    _C_LRT_FUSED_AVAILABLE,
    _C_MODE4_FUSED_AVAILABLE,
    _C_SCORE_FUSED_AVAILABLE,
    LmmMode,
    _compute_lrt_split_numpy,
    _compute_score_split_numpy,
    compute_lmm_chunk_numpy,
    compute_mode4_fused_c_ws,
    compute_mode4_fused_general_c_ws,
    compute_mode4_split_c_ws,
    compute_wald_fused_c_ws,
    compute_wald_fused_general_c_ws,
    compute_wald_general_c_ws,
    compute_wald_split_c_ws,
    create_lmm_workspace,
    create_lmm_workspace_fused,
    create_lmm_workspace_fused_general,
    create_lmm_workspace_general,
    create_lmm_workspace_mode4,
    create_lmm_workspace_mode4_fused,
    create_lmm_workspace_mode4_fused_general,
    select_current_dispatch_path,
)
from jamma.lmm.dispatch import LmmDispatch
from jamma.lmm.impute import impute_missing_inplace
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
    reset_p_yy_warned,
)
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    _eigendecompose_or_reuse,
    compute_and_log_pve,
    validate_runner_inputs,
)
from jamma.lmm.results import (
    _build_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.schema import LmmConfig, LmmRunResult
from jamma.utils.logging import log_rss_memory

# Allow large chunks — no int32 buffer constraint.
_MAX_CHUNK = 200_000

# Memory budget bounds for auto-scaling
_MIN_BUDGET = 2_000_000_000  # 2 GB floor (original default)
_MAX_BUDGET = 40_000_000_000  # 40 GB ceiling

# Minimum number of chunks before pipelined execution is worthwhile.
_MIN_PIPELINE_CHUNKS = 8


_ALL_RESULT_KEYS = (
    "lambdas",
    "logls",
    "betas",
    "ses",
    "pwalds",
    "lambdas_mle",
    "p_lrts",
    "p_scores",
)


def _select_wald_fn(n_cvt: int):
    """Return the C workspace Wald compute function appropriate for n_cvt.

    Args:
        n_cvt: Number of covariates.

    Returns:
        compute_wald_split_c_ws for n_cvt=1; compute_wald_general_c_ws for n_cvt>1.
    """
    return compute_wald_split_c_ws if n_cvt == 1 else compute_wald_general_c_ws


def _create_wald_workspace_for_ncvt(
    n_cvt: int,
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> object:
    """Create the appropriate C Wald workspace for any n_cvt.

    Dispatches to create_lmm_workspace (split, n_cvt=1) or
    create_lmm_workspace_general (general, n_cvt>1). Returns None if the
    required C extension is unavailable.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab SoA array (n_inv, n_samples).
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid resolution.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        C PyCapsule workspace, or None if extension unavailable.
    """
    if n_cvt == 1:
        return create_lmm_workspace(
            eigenvalues,
            uab_invariant_soa,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            n_threads,
        )
    if _C_GENERAL_AVAILABLE:
        return create_lmm_workspace_general(
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
    logger.debug(
        "Wald workspace unavailable for n_cvt={} (general C extension missing)", n_cvt
    )
    return None


class _Workspaces(NamedTuple):
    """Persistent C workspaces created once before the chunk loop.

    Each is a PyCapsule (or None when its path is inactive), freed when the tuple
    goes out of scope. Shared by both the batch and streaming runners.
    """

    lmm_workspace: object | None
    score_fused_workspace: object | None
    lrt_fused_workspace: object | None


def _create_workspaces(
    dispatch: LmmDispatch,
    lmm_mode: LmmMode,
    n_cvt: int,
    eigenvalues_np: np.ndarray,
    uab_invariant_soa: np.ndarray | None,
    UtW: np.ndarray,
    Uty: np.ndarray,
    w: np.ndarray | None,
    Hi_eval_null: np.ndarray,
    logl_H0: float,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> _Workspaces:
    """Create the persistent C workspaces for one LMM run.

    Each holds precomputed lambda grids, hi_eval/logdet grids, grid_inv, and
    invariant Iab column sums — reused across all chunks without reallocation,
    eliminating per-chunk malloc/free. A Wald workspace is created for modes 1
    (Wald) and 4 (All); fused variants pass w/Uty so the C kernel computes wx/xx/xy
    on the fly from utg_t. Score (mode 3) and LRT (mode 2) workspaces are created
    separately. Returns None for any workspace whose dispatch path is inactive.

    Shared by the batch and streaming runners, which previously created these
    with byte-identical code.
    """
    if dispatch.use_split and lmm_mode in (1, 4):
        if dispatch.use_fused_general:
            # Fused general workspace: UtW + Uty for on-the-fly dot products.
            # Mode 4 extends with null-model fields for MLE/LRT.
            from jamma.lmm.likelihood import build_pab_table_for_c

            pab_c = build_pab_table_for_c(n_cvt)
            pab_kwargs = {
                k: pab_c[k]
                for k in [
                    "invariant_indices",
                    "varying_indices",
                    "logdet_diag_rows",
                    "logdet_diag_cols",
                    "level_offsets",
                    "level_counts",
                    "entries",
                    "idx_xx",
                    "idx_xy",
                    "idx_yy",
                    "var_a_cols",
                    "var_b_cols",
                ]
            }
            if lmm_mode == 4:
                lmm_workspace = create_lmm_workspace_mode4_fused_general(
                    eigenvalues_np,
                    uab_invariant_soa,
                    UtW,
                    Uty,
                    n_samples,
                    l_min,
                    l_max,
                    n_grid,
                    n_refine,
                    n_threads,
                    n_cvt=n_cvt,
                    **pab_kwargs,
                    hi_eval_null=Hi_eval_null,
                    logl_H0=logl_H0,
                )
            else:
                lmm_workspace = create_lmm_workspace_fused_general(
                    eigenvalues_np,
                    uab_invariant_soa,
                    UtW,
                    Uty,
                    n_samples,
                    l_min,
                    l_max,
                    n_grid,
                    n_refine,
                    n_threads,
                    n_cvt=n_cvt,
                    **pab_kwargs,
                )
        elif dispatch.use_fused and lmm_mode == 4:
            w = UtW[:, 0].copy()
            lmm_workspace = create_lmm_workspace_mode4_fused(
                eigenvalues_np,
                uab_invariant_soa,
                w,
                Uty,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                n_threads,
                hi_eval_null=Hi_eval_null,
                logl_H0=logl_H0,
            )
        elif dispatch.use_fused:
            w = UtW[:, 0].copy()
            lmm_workspace = create_lmm_workspace_fused(
                eigenvalues_np,
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
        elif dispatch.use_fused_mode4:
            lmm_workspace = create_lmm_workspace_mode4(
                eigenvalues_np,
                uab_invariant_soa,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                n_threads,
                Hi_eval_null,
                logl_H0,
            )
        else:
            lmm_workspace = _create_wald_workspace_for_ncvt(
                n_cvt,
                eigenvalues_np,
                uab_invariant_soa,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                n_threads,
            )
    else:
        lmm_workspace = None

    # Score/LRT workspaces (persistent across all chunks). Score: null-model dot
    # products and F-distribution constants. LRT: lambda grids and per-thread
    # scratch buffers. Both eliminate per-chunk malloc/free and precomputation.
    score_fused_workspace = None
    lrt_fused_workspace = None

    if dispatch.use_fused_score_ws:
        from jamma.lmm.compute_numpy import _create_workspace_score_fused_c

        score_fused_workspace = _create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues_np,
            uab_invariant_soa,
            n_samples,
            n_threads,
        )

    if dispatch.use_fused_lrt_ws:
        from jamma.lmm.compute_numpy import _create_workspace_lrt_fused_c

        lrt_fused_workspace = _create_workspace_lrt_fused_c(
            w,
            Uty,
            eigenvalues_np,
            uab_invariant_soa,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            n_threads,
        )

    return _Workspaces(lmm_workspace, score_fused_workspace, lrt_fused_workspace)


def _guarded_compute(
    fn: Callable[..., dict],
    *args: object,
    operation: str,
    write_offset: int,
    n_filtered: int,
    **kwargs: object,
) -> dict:
    """Call *fn* with error wrapping that identifies the failed operation.

    Extra positional and keyword arguments are forwarded to *fn*;
    *operation*, *write_offset*, and *n_filtered* are consumed by the wrapper.

    MemoryError, ValueError, TypeError, and OverflowError propagate unchanged.
    All other exceptions (including OSError, used here to model a C-kernel
    segfault) are wrapped in a RuntimeError whose message includes the
    *operation* label, *write_offset*, and *n_filtered* for diagnosis.
    """
    try:
        return fn(*args, **kwargs)
    except (MemoryError, ValueError, TypeError, OverflowError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"{operation} failed at SNP offset "
            f"{write_offset}/{n_filtered}. "
            f"Processed {write_offset} SNPs before failure."
        ) from exc


def _compose_mode4_from_split(
    wald_cr: dict,
    n_cvt: int,
    eigenvalues_np: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    *,
    Hi_eval_null: np.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict:
    """Compose mode-4 results from Wald + SoA-split Score + LRT.

    Merge order: Score, LRT, then Wald
    (Wald's REML betas/ses overwrite Score's values).
    """
    score_cr = _compute_score_split_numpy(
        n_cvt,
        eigenvalues_np,
        Hi_eval_null,
        uab_varying_soa,
        uab_invariant_soa,
        n_samples,
        n_threads,
    )
    lrt_cr = _compute_lrt_split_numpy(
        n_cvt,
        eigenvalues_np,
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
    cr: dict = dict.fromkeys(_ALL_RESULT_KEYS)
    for d in (score_cr, lrt_cr, wald_cr):
        cr.update({k: v for k, v in d.items() if v is not None})
    return cr


def dispatch_soa_split(
    lmm_mode: int,
    use_fused_mode4: bool,
    lmm_workspace: object | None,
    n_cvt: int,
    eigenvalues_np: np.ndarray,
    uab_var_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    *,
    Hi_eval_null: np.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict:
    """Dispatch SoA split computation by lmm_mode.

    Centralises the mode dispatch decision tree used by all three runner
    paths (pipeline, sequential, streaming). Does NOT manage BLAS thread
    pinning or error wrapping — callers handle those concerns.

    Args:
        lmm_mode: 1=Wald, 2=LRT, 3=Score, 4=All.
        use_fused_mode4: True when fused mode-4 C kernel is available.
        lmm_workspace: C workspace capsule (None when unavailable).
        n_cvt: Number of covariates.
        eigenvalues_np: Kinship eigenvalues (n_samples,).
        uab_var_soa: SNP-varying Uab SoA (n_snps, n_var, n_samples).
        uab_invariant_soa: SNP-invariant Uab SoA (n_inv, n_samples).
        n_samples: Number of samples.
        Hi_eval_null: Pre-computed null-model Hi_eval (n_samples,).
        l_min: Lambda lower bound for MLE optimisation.
        l_max: Lambda upper bound for MLE optimisation.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations.
        logl_H0: Null model MLE log-likelihood.
        n_threads: OpenMP thread count.

    Returns:
        Dict of result arrays (keys depend on lmm_mode).
    """
    if use_fused_mode4 and lmm_workspace is not None:
        return compute_mode4_split_c_ws(lmm_workspace, uab_var_soa, n_threads)

    if lmm_mode in (1, 4) and lmm_workspace is not None:
        wald_fn = _select_wald_fn(n_cvt)
        wald_cr = wald_fn(lmm_workspace, uab_var_soa, n_threads)
        if lmm_mode == 1:
            return wald_cr
        return _compose_mode4_from_split(
            wald_cr,
            n_cvt,
            eigenvalues_np,
            uab_var_soa,
            uab_invariant_soa,
            n_samples,
            Hi_eval_null=Hi_eval_null,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
            logl_H0=logl_H0,
            n_threads=n_threads,
        )

    if lmm_mode == 3:
        return _compute_score_split_numpy(
            n_cvt,
            eigenvalues_np,
            Hi_eval_null,
            uab_var_soa,
            uab_invariant_soa,
            n_samples,
            n_threads,
        )

    if lmm_mode == 2:
        return _compute_lrt_split_numpy(
            n_cvt,
            eigenvalues_np,
            uab_var_soa,
            uab_invariant_soa,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            n_threads,
        )

    raise ValueError(
        f"Unexpected lmm_mode={lmm_mode} in SoA split dispatch "
        f"(workspace={lmm_workspace is not None}). "
        f"Valid modes: 1 (Wald), 2 (LRT), 3 (Score), 4 (All, requires workspace)."
    )


class _ComputeContext(NamedTuple):
    """Loop-invariant inputs for the per-chunk C dispatch.

    Built once before the chunk loop and passed to _dispatch_compute for every
    chunk, so the 6-way kernel-selection ladder lives in exactly one place
    (previously duplicated across the batch pipeline, batch sequential, and
    streaming compute paths).
    """

    dispatch: LmmDispatch
    lmm_mode: LmmMode
    n_cvt: int
    lmm_workspace: object | None
    score_fused_workspace: object | None
    lrt_fused_workspace: object | None
    w: np.ndarray | None
    Uty: np.ndarray
    Hi_eval_null: np.ndarray
    uab_invariant_soa: np.ndarray | None
    eigenvalues_np: np.ndarray
    n_samples: int
    l_min: float
    l_max: float
    n_grid: int
    n_refine: int
    logl_H0: float
    n_filtered: int


def _dispatch_compute(
    ctx: _ComputeContext,
    chunk_input: np.ndarray,
    n_threads: int,
    write_offset: int,
) -> dict:
    """Dispatch one prepared chunk to the active C kernel and return its result.

    The 6-way kernel-selection ladder: fused-general / fused / fused-Score-WS /
    fused-LRT-WS / fused-Score / fused-LRT / SoA-split. ``chunk_input`` is utg_t
    for the fused paths and the varying-Uab SoA array for the split path; the
    active path is fixed by ``ctx.dispatch``. The caller owns BLAS-thread scoping,
    input preparation, and the non-split NumPy fallback.
    """
    d = ctx.dispatch
    if d.use_fused:
        if d.use_fused_general:
            if ctx.lmm_mode == 4:
                fused_fn = compute_mode4_fused_general_c_ws
                op_label = "Fused general mode-4 Uab dispatch"
            else:
                fused_fn = compute_wald_fused_general_c_ws
                op_label = "Fused general Uab dispatch"
        else:
            fused_fn = (
                compute_mode4_fused_c_ws
                if ctx.lmm_mode == 4
                else compute_wald_fused_c_ws
            )
            op_label = "Fused Uab dispatch"
        return _guarded_compute(
            fused_fn,
            ctx.lmm_workspace,
            chunk_input,
            n_threads,
            operation=op_label,
            write_offset=write_offset,
            n_filtered=ctx.n_filtered,
        )
    if d.use_fused_score_ws:
        from jamma.lmm.compute_numpy import _compute_score_fused_ws_c

        return _guarded_compute(
            _compute_score_fused_ws_c,
            ctx.score_fused_workspace,
            chunk_input,
            n_threads,
            operation="Fused Score WS dispatch",
            write_offset=write_offset,
            n_filtered=ctx.n_filtered,
        )
    if d.use_fused_lrt_ws:
        from jamma.lmm.compute_numpy import _compute_lrt_fused_ws_c

        return _guarded_compute(
            _compute_lrt_fused_ws_c,
            ctx.lrt_fused_workspace,
            chunk_input,
            n_threads,
            operation="Fused LRT WS dispatch",
            write_offset=write_offset,
            n_filtered=ctx.n_filtered,
        )
    if d.use_fused_score:
        from jamma.lmm.compute_numpy import _compute_score_fused_c

        return _guarded_compute(
            _compute_score_fused_c,
            chunk_input,
            ctx.w,
            ctx.Uty,
            ctx.Hi_eval_null,
            ctx.uab_invariant_soa,
            ctx.eigenvalues_np,
            ctx.n_samples,
            n_threads,
            operation="Fused Score dispatch",
            write_offset=write_offset,
            n_filtered=ctx.n_filtered,
        )
    if d.use_fused_lrt:
        from jamma.lmm.compute_numpy import _compute_lrt_fused_c

        return _guarded_compute(
            _compute_lrt_fused_c,
            chunk_input,
            ctx.w,
            ctx.Uty,
            ctx.eigenvalues_np,
            ctx.uab_invariant_soa,
            ctx.n_samples,
            ctx.l_min,
            ctx.l_max,
            ctx.n_grid,
            ctx.n_refine,
            ctx.logl_H0,
            n_threads,
            operation="Fused LRT dispatch",
            write_offset=write_offset,
            n_filtered=ctx.n_filtered,
        )
    # SoA split: chunk_input is the varying-Uab SoA array.
    return _guarded_compute(
        dispatch_soa_split,
        ctx.lmm_mode,
        d.use_fused_mode4,
        ctx.lmm_workspace,
        ctx.n_cvt,
        ctx.eigenvalues_np,
        chunk_input,
        ctx.uab_invariant_soa,
        ctx.n_samples,
        Hi_eval_null=ctx.Hi_eval_null,
        l_min=ctx.l_min,
        l_max=ctx.l_max,
        n_grid=ctx.n_grid,
        n_refine=ctx.n_refine,
        logl_H0=ctx.logl_H0,
        n_threads=n_threads,
        operation="SoA split dispatch",
        write_offset=write_offset,
        n_filtered=ctx.n_filtered,
    )


def compute_pipeline_core_split(n_samples: int, total_cores: int) -> tuple[int, int]:
    """Compute rotation/compute thread split for the pipeline path.

    DGEMM rotation scales with n_samples^2 * chunk_size while per-SNP
    compute scales with chunk_size * (n_grid + n_refine). For large
    n_samples rotation dominates; for small n_samples compute dominates.

    Args:
        n_samples: Number of samples in the dataset.
        total_cores: Physical core count available.

    Returns:
        (rotation_threads, compute_threads) tuple. Both >= 1.
    """
    if n_samples > 10_000:
        rot = max(1, total_cores // 2)
    elif n_samples > 1_000:
        rot = max(1, total_cores // 3)
    else:
        rot = max(1, total_cores // 4)
    return rot, max(1, total_cores - rot)


def compute_adaptive_core_split(
    rot_time: float,
    compute_time: float,
    total_cores: int,
    *,
    n_samples: int = 0,
) -> tuple[int, int]:
    """Compute rotation/compute thread split from measured first-chunk times.

    Allocates threads proportionally to observed rotation vs compute wall time.
    Falls back to static heuristic when profiling data is degenerate (both
    times near zero, which happens on small datasets where profiling overhead
    dominates).

    Args:
        rot_time: Wall time for first-chunk rotation (UT@G DGEMM), seconds.
        compute_time: Wall time for first-chunk compute (C extension), seconds.
        total_cores: Physical core count available.
        n_samples: Sample count for static fallback (only used when times are
            degenerate).

    Returns:
        (rotation_threads, compute_threads) tuple. Both >= 1.
    """
    total_time = rot_time + compute_time
    if total_time < 0.01:  # < 10ms: profiling not meaningful, use static
        return compute_pipeline_core_split(n_samples, total_cores)

    rot_fraction = rot_time / total_time
    rot_threads = max(1, min(total_cores - 1, round(total_cores * rot_fraction)))
    compute_threads = max(1, total_cores - rot_threads)
    return rot_threads, compute_threads


class _ThreadBudget:
    """Mutable rotation/compute core split shared with the pipeline callbacks.

    The pipeline driver re-derives the split from the profiled first chunk and
    rebinds these fields. Because the prepare/compute callbacks live in the
    runner's scope while the driver lives here, a bare-int ``nonlocal`` cannot
    carry the update across the boundary — the callbacks would keep reading the
    pre-profile values. A shared mutable object does: the callbacks read
    ``budget.rot`` / ``budget.omp`` and the driver mutates them in place.
    """

    __slots__ = ("omp", "rot")

    def __init__(self, rot: int, omp: int) -> None:
        self.rot = rot
        self.omp = omp


def _drive_pipeline(
    prepare: Callable[[], object | None],
    compute: Callable[[object], None],
    budget: _ThreadBudget,
    *,
    n_chunks: int,
    total_cores: int,
    n_samples: int,
    n_filtered: int,
    show_progress: bool,
    progress_label: str,
) -> float:
    """Drive the overlapped chunk pipeline shared by both NumPy runners.

    Profiles the first chunk, re-derives the rotation/compute core split from
    its measured stage durations, then overlaps rotation of chunk N+1 (a
    background ``prepare``) with C compute of chunk N (a foreground ``compute``)
    via a single-worker executor. Both stages release the GIL, so they run
    concurrently.

    Only the chunk source (in-memory fancy-index vs. disk stream) and the result
    sink differ between runners; those are supplied as ``prepare`` and
    ``compute`` callbacks. ``prepare`` returns an opaque prepared-chunk object,
    or None at exhaustion; the driver passes it straight to ``compute`` and
    never inspects it. Both callbacks read the live core split from ``budget``,
    which this function mutates after profiling.

    Args:
        prepare: Zero-arg callback that prepares the next chunk (slice/impute/
            rotate), returning an opaque object or None when no chunks remain.
        compute: Callback that runs C compute on a prepared chunk and writes its
            results. Owns its own compute/write timing and diagnostics.
        budget: Shared mutable core split; rebound from the profiled first chunk.
        n_chunks: Expected chunk count (progress total; adaptive-split guard).
        total_cores: Physical core count for the adaptive split.
        n_samples: Sample count (adaptive-split static fallback; ETA estimate).
        n_filtered: Filtered SNP count (ETA estimate; error diagnostics).
        show_progress: Whether to render a progress bar.
        progress_label: Progress-bar label.

    Returns:
        Total rotation wall-time (seconds) measured around the prepare calls,
        for the caller's timing breakdown. Compute/write time is accumulated by
        the ``compute`` callback itself.
    """
    rotation_s = 0.0

    # Profile the first chunk: prepare (rotation) then compute, timing each
    # stage so the adaptive split below uses empirically measured durations.
    t = time.perf_counter()
    first = prepare()
    t_first_rot = time.perf_counter() - t
    rotation_s += t_first_rot
    if first is None:
        return rotation_s

    t = time.perf_counter()
    compute(first)
    t_first_compute = time.perf_counter() - t
    del first

    # Re-derive the core split from measured times (only when chunks remain and
    # BLAS is controllable). Mutating the shared budget rebinds what the
    # prepare/compute callbacks read on every subsequent call.
    if n_chunks > 2 and is_blas_controllable():
        old_rot, old_omp = budget.rot, budget.omp
        budget.rot, budget.omp = compute_adaptive_core_split(
            t_first_rot, t_first_compute, total_cores, n_samples=n_samples
        )
        if (budget.rot, budget.omp) != (old_rot, old_omp):
            logger.debug(
                f"Adaptive core split: {old_rot}/{old_omp} -> "
                f"{budget.rot}/{budget.omp} "
                f"(rot={t_first_rot:.3f}s, compute={t_first_compute:.3f}s)"
            )

    # Seed the pipeline with the next chunk (uses the updated split).
    t = time.perf_counter()
    current = prepare()
    rotation_s += time.perf_counter() - t

    # Progress: profiled chunk + seeded chunk already accounted, so start at 2.
    bar = (
        create_progress_bar(
            n_chunks,
            progress_label,
            initial_eta_seconds=estimate_lmm_seconds(n_samples, n_filtered),
        )
        if show_progress and n_chunks > 1
        else None
    )
    if bar is not None:
        bar.update(2)

    i = 2
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while current is not None:
                # Prepare chunk N+1 in the background while computing chunk N;
                # both release the GIL, so rotation and compute overlap.
                future = executor.submit(prepare)
                compute(current)

                t = time.perf_counter()
                try:
                    current = future.result()
                except (MemoryError, ValueError, TypeError, OverflowError, OSError):
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"Pipeline chunk preparation failed at chunk {i} of "
                        f"{n_chunks} during overlapped rotation "
                        f"({n_filtered} SNPs total)."
                    ) from exc
                rotation_s += time.perf_counter() - t

                i += 1
                if bar is not None:
                    bar.update(i)
    finally:
        if bar is not None:
            with suppress(Exception):
                bar.update(n_chunks)
                bar.finish()

    return rotation_s


def compute_chunk_size_numpy(
    n_samples: int,
    n_filtered: int,
    n_cvt: int = 1,
    *,
    use_split: bool = False,
    lmm_mode: int = 1,
    fused_mode4: bool = False,
    use_fused_general: bool = False,
    mem_budget_bytes: int | None = None,
    pipeline_buffers: int = 1,
) -> int:
    """Compute chunk size based on RAM budget (no int32 constraint for NumPy).

    Scales the memory budget with available RAM to minimise DRAM passes
    through the eigenvector matrix during UT@G rotation.

    Args:
        n_samples: Number of samples.
        n_filtered: Number of filtered SNPs.
        n_cvt: Number of covariates.
        use_split: If True, use split Uab accounting instead of full Uab.
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All). Affects
            memory accounting: Wald uses 4 cols/SNP (3 varying + 1 utg_t),
            non-Wald uses 9 cols/SNP (3 varying + 6 reconstructed Uab peak).
        fused_mode4: If True, mode-4 uses fused C kernel (4-col accounting,
            same as Wald) instead of reconstruct+compose (9-col).
        use_fused_general: If True, fused general path is active (n_cvt>=2);
            only utg_t is allocated (single buffer, no uab_varying_soa).
        mem_budget_bytes: Explicit per-chunk memory budget in bytes.
            None (default) auto-scales with available RAM.
        pipeline_buffers: Number of live chunks (1 for sequential,
            2 for pipeline double-buffering). Divides the budget.

    Returns:
        Chunk size (number of SNPs per chunk).
    """
    if not isinstance(pipeline_buffers, int):
        raise TypeError(
            f"pipeline_buffers must be an int, got {type(pipeline_buffers).__name__}"
        )
    if pipeline_buffers < 1:
        raise ValueError(f"pipeline_buffers must be >= 1, got {pipeline_buffers}")

    if use_split and n_cvt == 1:
        if _C_FUSED_AVAILABLE and (
            lmm_mode == 1 or (lmm_mode == 4 and _C_MODE4_FUSED_AVAILABLE)
        ):
            # Fused path: jlinalg.dgemm(chunk, U, transa="T") produces
            # C-contiguous utg_t (n_snps, n_samples) directly — single buffer.
            # No intermediate allocation or contiguous copy.
            # Mode 4 only uses fused when _C_MODE4_FUSED_AVAILABLE; otherwise
            # it falls back to split SoA which needs 4x buffers.
            bytes_per_snp = n_samples * 8
        elif lmm_mode == 3 and _C_SCORE_FUSED_AVAILABLE:
            # Fused Score: utg_t only (1 col), no uab_varying_soa.
            bytes_per_snp = n_samples * 8
        elif lmm_mode == 2 and _C_LRT_FUSED_AVAILABLE:
            # Fused LRT: utg_t only (1 col), no uab_varying_soa.
            bytes_per_snp = n_samples * 8
        else:
            # SoA split paths (Wald, Score, LRT, mode-4):
            # 3 varying SoA columns + 1 utg_t per SNP, no Uab reconstruction.
            bytes_per_snp = n_samples * 4 * 8
    elif use_split and n_cvt > 1:
        from jamma.lmm.likelihood import classify_uab_columns

        _inv, var = classify_uab_columns(n_cvt)
        n_var = len(var)
        if use_fused_general:
            # Fused general path: jlinalg.dgemm produces utg_t directly.
            # Single buffer, no intermediate allocation or contiguous copy.
            bytes_per_snp = n_samples * 8
        else:
            # All modes: split C dispatch, no Uab reconstruction.
            # n_var varying SoA columns + 1 utg_t per SNP.
            bytes_per_snp = n_samples * (n_var + 1) * 8
    else:
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        bytes_per_snp = n_samples * n_index * 8

    if bytes_per_snp == 0:
        return n_filtered

    if mem_budget_bytes is not None:
        mem_budget = mem_budget_bytes
    else:
        available = psutil.virtual_memory().available
        # Budget: 15% of available RAM (up from 5%), 2 GB floor, 40 GB ceiling.
        # Modern machines (128-512 GB) can afford larger working sets. The floor
        # prevents degenerate chunk sizes on low-memory systems; the ceiling
        # prevents excessive allocation on high-memory systems.
        mem_budget = max(_MIN_BUDGET, min(int(available * 0.15), _MAX_BUDGET))

    mem_budget = mem_budget // pipeline_buffers

    chunk_from_memory = int(mem_budget / bytes_per_snp)
    chunk = max(100, min(chunk_from_memory, n_filtered, _MAX_CHUNK))
    return chunk


class RawLmmChunk(NamedTuple):
    """Raw genotype chunk handed to the shared NumPy LMM chunk runner.

    ``genotypes`` must be a mutable float64 array with shape
    ``(n_samples, filtered_end - filtered_start)``. The shared runner owns
    imputation, rotation, compute dispatch, and diagnostics after this point.
    """

    genotypes: np.ndarray
    filtered_start: int
    filtered_end: int


class LmmChunkRunStats(NamedTuple):
    """Timing and diagnostic counters from the shared chunk runner."""

    processed: int
    rotation_s: float
    compute_s: float
    result_write_s: float
    nan_counts: dict[str, int]
    n_at_lmin: int
    n_at_lmax: int
    chunk_size: int
    n_chunks: int
    used_pipeline: bool


def run_lmm_chunk_source_numpy(
    *,
    raw_chunk_source_factory: Callable[[int], Callable[[], RawLmmChunk | None]],
    chunk_sink: Callable[[dict[str, np.ndarray], int, int], None],
    U: np.ndarray,
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    Hi_eval_null: np.ndarray | None,
    logl_H0: float | None,
    n_samples: int,
    n_filtered: int,
    n_cvt: int,
    lmm_mode: LmmMode,
    filtered_means: np.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    requested_chunk_size: int | None = None,
    auto_scale_chunk_size: bool = True,
    show_progress: bool = True,
    progress_label: str = "LMM association",
    lambda_warning_prefix: str = "",
    log_dispatch_choices: bool = True,
) -> LmmChunkRunStats:
    """Run LMM association over caller-provided raw genotype chunks.

    The caller owns where raw genotype chunks come from and where result chunks
    go. This function owns the canonical NumPy LMM chunk machinery: dispatch
    selection, chunk sizing, optional pipeline driving, missing-value imputation,
    eigen-rotation, Uab preparation, C/Python compute dispatch, diagnostics, and
    timing. Batch and LOCO use this path so their chunk compute behavior cannot
    drift.
    """
    if n_filtered == 0:
        return LmmChunkRunStats(
            processed=0,
            rotation_s=0.0,
            compute_s=0.0,
            result_write_s=0.0,
            nan_counts={},
            n_at_lmin=0,
            n_at_lmax=0,
            chunk_size=0,
            n_chunks=0,
            used_pipeline=False,
        )

    if requested_chunk_size is not None and requested_chunk_size < 1:
        raise ValueError(
            f"requested_chunk_size must be >= 1, got {requested_chunk_size}"
        )
    if len(filtered_means) != n_filtered:
        raise ValueError(
            f"filtered_means length ({len(filtered_means)}) does not match "
            f"n_filtered ({n_filtered})"
        )
    if lmm_mode in (3, 4) and Hi_eval_null is None:
        raise RuntimeError("LMM Score/All mode requires Hi_eval_null")
    if lmm_mode in (2, 4) and logl_H0 is None:
        raise RuntimeError("LMM LRT/All mode requires logl_H0")

    hi_eval_for_compute = (
        np.empty(0, dtype=np.float64) if Hi_eval_null is None else Hi_eval_null
    )
    logl_H0_for_compute = float("nan") if logl_H0 is None else logl_H0

    dispatch = select_current_dispatch_path(
        n_cvt, lmm_mode, log_choices=log_dispatch_choices
    )
    use_split = dispatch.use_split
    use_fused = dispatch.use_fused
    use_fused_general = dispatch.use_fused_general
    use_fused_mode4 = dispatch.use_fused_mode4
    use_fused_score = dispatch.use_fused_score
    use_fused_score_ws = dispatch.use_fused_score_ws
    use_fused_lrt = dispatch.use_fused_lrt
    use_fused_lrt_ws = dispatch.use_fused_lrt_ws

    def _compute_engine_chunk_size(*, pipeline_buffers: int = 1) -> int:
        chunk = compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt,
            use_split=use_split,
            lmm_mode=lmm_mode,
            fused_mode4=use_fused_mode4,
            use_fused_general=use_fused_general,
            pipeline_buffers=pipeline_buffers,
        )
        if requested_chunk_size is not None:
            chunk = min(chunk, requested_chunk_size)
        return max(1, chunk)

    if requested_chunk_size is None or auto_scale_chunk_size:
        chunk_size = _compute_engine_chunk_size()
    else:
        chunk_size = requested_chunk_size

    n_chunks = (n_filtered + chunk_size - 1) // chunk_size
    use_pipeline = use_split and n_chunks >= _MIN_PIPELINE_CHUNKS

    if use_pipeline:
        if requested_chunk_size is None or auto_scale_chunk_size:
            chunk_size = _compute_engine_chunk_size(pipeline_buffers=2)
        else:
            chunk_size = max(1, chunk_size // 2)
        n_chunks = (n_filtered + chunk_size - 1) // chunk_size
        use_pipeline = use_split and n_chunks >= _MIN_PIPELINE_CHUNKS

    if show_progress:
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"  Processing in {n_chunks} chunks ({chunk_size:,} SNPs/chunk)"
            )

    omp_threads = get_c_extension_thread_count(_C_ACCEL_AVAILABLE, _C_HAS_OPENMP)

    if use_pipeline:
        logger.debug(f"Pipeline mode: overlapping rotation/compute ({n_chunks} chunks)")
        total_cores = get_physical_core_count()
        if omp_threads == 1:
            pipeline_rot_threads = total_cores
            pipeline_omp_threads = 1
        else:
            rot_threads, compute_threads = compute_pipeline_core_split(
                n_samples, total_cores
            )
            pipeline_omp_threads = min(compute_threads, omp_threads)
            pipeline_rot_threads = max(1, total_cores - pipeline_omp_threads)
            logger.debug(
                f"Pipeline core split: {pipeline_rot_threads} rotation, "
                f"{pipeline_omp_threads} compute (n_samples={n_samples:,})"
            )
    else:
        total_cores = get_physical_core_count()
        pipeline_omp_threads = omp_threads
        pipeline_rot_threads = total_cores

    budget = _ThreadBudget(pipeline_rot_threads, pipeline_omp_threads)
    n_refine = max(n_refine, 20)

    uab_invariant_soa = (
        compute_uab_invariant_soa(UtW, Uty, n_cvt) if use_split else None
    )
    w = (
        UtW[:, 0].copy()
        if (use_fused_score or use_fused_lrt or use_fused_score_ws or use_fused_lrt_ws)
        and not use_fused
        else None
    )

    lmm_workspace, score_fused_workspace, lrt_fused_workspace = _create_workspaces(
        dispatch,
        lmm_mode,
        n_cvt,
        eigenvalues_np,
        uab_invariant_soa,
        UtW,
        Uty,
        w,
        hi_eval_for_compute,
        logl_H0_for_compute,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        pipeline_omp_threads,
    )
    compute_ctx = _ComputeContext(
        dispatch,
        lmm_mode,
        n_cvt,
        lmm_workspace,
        score_fused_workspace,
        lrt_fused_workspace,
        w,
        Uty,
        hi_eval_for_compute,
        uab_invariant_soa,
        eigenvalues_np,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0_for_compute,
        n_filtered,
    )

    raw_chunk_source = raw_chunk_source_factory(chunk_size)

    if use_pipeline:
        utg_bufs = [
            np.empty((chunk_size, n_samples), dtype=np.float64),
            np.empty((chunk_size, n_samples), dtype=np.float64),
        ]
    else:
        utg_bufs = [np.empty((chunk_size, n_samples), dtype=np.float64)]

    no_fused = (
        not use_fused
        and not use_fused_score
        and not use_fused_lrt
        and not use_fused_score_ws
        and not use_fused_lrt_ws
    )
    if use_split and no_fused:
        from jamma.lmm.likelihood import classify_uab_columns

        _inv_cols, var_cols = classify_uab_columns(n_cvt)
        n_var = len(var_cols)
        if use_pipeline:
            uab_var_bufs = [
                np.empty((chunk_size, n_var, n_samples), dtype=np.float64),
                np.empty((chunk_size, n_var, n_samples), dtype=np.float64),
            ]
        else:
            uab_var_bufs = [np.empty((chunk_size, n_var, n_samples), dtype=np.float64)]
    else:
        uab_var_bufs = None

    chunk_counter = 0
    processed = 0
    rotation_s = 0.0
    compute_s = 0.0
    result_write_s = 0.0
    nan_counts: dict[str, int] = {}
    n_at_lmin = 0
    n_at_lmax = 0

    def _prepare_chunk() -> tuple[np.ndarray, int, int] | None:
        nonlocal chunk_counter

        raw = raw_chunk_source()
        while raw is not None and raw.filtered_end <= raw.filtered_start:
            raw = raw_chunk_source()
        if raw is None:
            return None

        actual_len = raw.filtered_end - raw.filtered_start
        if raw.genotypes.shape != (n_samples, actual_len):
            raise ValueError(
                "raw LMM chunk shape mismatch: expected "
                f"({n_samples}, {actual_len}), got {raw.genotypes.shape}"
            )

        buf_idx = chunk_counter % len(utg_bufs)
        chunk_counter += 1

        impute_missing_inplace(
            raw.genotypes, filtered_means[raw.filtered_start : raw.filtered_end]
        )

        utg_out = utg_bufs[buf_idx][:actual_len, :]
        with jlinalg_threads(budget.rot):
            utg_t = jlinalg.dgemm(raw.genotypes, U, transa="T", out=utg_out)

        if (
            use_fused
            or use_fused_score
            or use_fused_lrt
            or use_fused_score_ws
            or use_fused_lrt_ws
        ):
            return utg_t, raw.filtered_start, raw.filtered_end

        if use_split:
            out_var = (
                uab_var_bufs[buf_idx][:actual_len, :, :]
                if uab_var_bufs is not None and actual_len == chunk_size
                else None
            )
            uab_var_soa = batch_compute_uab_varying_soa_numpy(
                n_cvt, UtW, Uty, utg_t, out=out_var
            )
            return uab_var_soa, raw.filtered_start, raw.filtered_end

        uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, utg_t.T)
        return uab_batch, raw.filtered_start, raw.filtered_end

    def _compute_and_write(prepared: tuple[np.ndarray, int, int]) -> None:
        nonlocal processed, compute_s, result_write_s, n_at_lmin, n_at_lmax

        chunk_data, filtered_start, filtered_end = prepared
        actual_len = filtered_end - filtered_start

        t_compute_start = time.perf_counter()
        blas_ctx = blas_threads(1) if _C_ACCEL_AVAILABLE else nullcontext()
        with blas_ctx:
            if use_split:
                cr = _dispatch_compute(compute_ctx, chunk_data, budget.omp, processed)
            else:
                cr = _guarded_compute(
                    compute_lmm_chunk_numpy,
                    lmm_mode,
                    n_cvt,
                    eigenvalues_np,
                    chunk_data,
                    n_samples,
                    l_min=l_min,
                    l_max=l_max,
                    n_grid=n_grid,
                    n_refine=n_refine,
                    Hi_eval_null=hi_eval_for_compute,
                    logl_H0=logl_H0_for_compute,
                    n_threads=budget.omp,
                    operation="LMM chunk compute",
                    write_offset=processed,
                    n_filtered=n_filtered,
                )
        compute_s += time.perf_counter() - t_compute_start

        t_write_start = time.perf_counter()
        chunk_arrays = {key: cr[key][:actual_len] for key in _RESULT_FIELDS[lmm_mode]}

        chunk_lmin, chunk_lmax = count_lambda_boundary_hits(
            lmm_mode, chunk_arrays, l_min, l_max
        )
        n_at_lmin += chunk_lmin
        n_at_lmax += chunk_lmax

        for key, arr in chunk_arrays.items():
            if arr.dtype.kind != "f":
                continue
            n_nan = int(np.count_nonzero(np.isnan(arr)))
            if n_nan > 0:
                nan_counts[key] = nan_counts.get(key, 0) + n_nan

        chunk_sink(chunk_arrays, filtered_start, filtered_end)
        processed += actual_len
        result_write_s += time.perf_counter() - t_write_start

    if use_pipeline:
        rotation_s += _drive_pipeline(
            _prepare_chunk,
            _compute_and_write,
            budget,
            n_chunks=n_chunks,
            total_cores=total_cores,
            n_samples=n_samples,
            n_filtered=n_filtered,
            show_progress=show_progress,
            progress_label=progress_label,
        )
    else:
        if show_progress and n_chunks > 1:
            chunk_iterator = progress_iterator(
                range(n_chunks),
                total=n_chunks,
                desc=progress_label,
                initial_eta_seconds=estimate_lmm_seconds(n_samples, n_filtered),
            )
        else:
            chunk_iterator = range(n_chunks)

        for _chunk_idx in chunk_iterator:
            t_rot_start = time.perf_counter()
            prepared = _prepare_chunk()
            rotation_s += time.perf_counter() - t_rot_start
            if prepared is None:
                break
            _compute_and_write(prepared)

    if processed != n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {processed} results, "
            f"expected {n_filtered}. This is an internal error — please report "
            f"this issue with your dataset dimensions."
        )

    for key, n_nan in nan_counts.items():
        logger.warning(
            f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
            "check for degenerate (constant) genotypes and kinship matrix quality"
        )
    log_lambda_boundary_warning(
        n_at_lmin, n_at_lmax, l_min, l_max, prefix=lambda_warning_prefix
    )

    return LmmChunkRunStats(
        processed=processed,
        rotation_s=rotation_s,
        compute_s=compute_s,
        result_write_s=result_write_s,
        nan_counts=nan_counts,
        n_at_lmin=n_at_lmin,
        n_at_lmax=n_at_lmax,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        used_pipeline=use_pipeline,
    )


def run_lmm_association_numpy(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    snp_info: list,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    check_memory: bool = True,
    show_progress: bool = True,
    lmm_mode: LmmMode = 1,
    config: LmmConfig | None = None,
    output_path: Path | None = None,
) -> LmmRunResult:
    """Run LMM association tests using pure-NumPy batch processing.

    Processes SNPs in memory-bounded chunks using BLAS-backed NumPy operations.
    Input genotypes must fit in memory; for disk streaming
    use run_lmm_association_numpy_streaming.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples) or None when
            pre-computed eigenvalues/eigenvectors are provided. WARNING: may
            be overwritten in-place during eigendecomposition (buffer reused
            for eigenvectors). Treat as consumed; pass kinship.copy() if you
            need the original matrix after this call.
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0.
        covariates: Covariate matrix (n_samples, n_cvt) or None for intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (clamped to min 20
            internally for ~1e-5 tolerance).
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        config: LmmConfig instance. When provided, overrides individual
            threshold/mode kwargs above.
        output_path: Path for per-chunk disk streaming. When set, results
            are written incrementally and the returned LmmRunResult has
            empty associations and n_tested populated instead.
    Returns:
        LmmRunResult with per-SNP associations and PVE from null model.
            When output_path is set, associations is empty (results on
            disk) and n_tested contains the count of SNPs written.

    Raises:
        MemoryError: If check_memory=True and insufficient memory.
        ValueError: If only one of eigenvalues/eigenvectors is provided,
            or if no valid samples remain after filtering.
    """
    # Unpack config if provided (config takes precedence over individual kwargs)
    if config is not None:
        kw = config.as_kwargs()
        maf_threshold = kw["maf_threshold"]
        miss_threshold = kw["miss_threshold"]
        l_min, l_max = kw["l_min"], kw["l_max"]
        n_grid, n_refine = kw["n_grid"], kw["n_refine"]
        check_memory = kw["check_memory"]
        show_progress, lmm_mode = kw["show_progress"], kw["lmm_mode"]

    # Reset per-run warning flags so each run gets its own diagnostics
    reset_p_yy_warned()

    # Memory check before workflow (uses genotype shape, runner-specific)
    n_samples, n_snps = genotypes.shape
    start_time = time.perf_counter()

    if show_progress:
        logger.info("Performing LMM Association Test (NumPy batch)")
        logger.info(f"  Total individuals: {n_samples:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.debug(
            f"MAF threshold = {maf_threshold}, missing threshold = {miss_threshold}"
        )

    if check_memory:
        # Propagate n_cvt so the preflight correctly sizes Uab/Iab for
        # multi-covariate runs. Otherwise the estimator silently uses its
        # n_cvt=1 default and can let a multi-covariate run pass preflight
        # before OOMing at the real allocation. Intercept column counts as
        # a covariate, so minimum is 1 when no user covariates are passed.
        n_cvt = covariates.shape[1] if covariates is not None else 1
        est = estimate_lmm_memory(n_samples, n_snps, n_cvt=n_cvt)
        logger.info(
            f"LMM memory: estimated {est.total_gb:.1f}GB, "
            f"available {est.available_gb:.1f}GB"
        )
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for LMM workflow with {n_samples:,} samples × "
                f"{n_snps:,} SNPs.\n"
                f"Need: {est.total_gb:.1f}GB, Available: {est.available_gb:.1f}GB\n"
                f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
                f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"genotypes={est.genotypes_gb:.1f}GB"
            )

    # Validate inputs and apply sample filtering (shared logic for all runners)
    setup = validate_runner_inputs(
        phenotypes, kinship, covariates, eigenvalues, eigenvectors, lmm_mode
    )
    phenotypes = setup.phenotypes
    kinship = setup.kinship
    covariates = setup.covariates
    eigenvalues = setup.eigenvalues
    eigenvectors = setup.eigenvectors
    n_samples = setup.n_samples

    # Apply the same valid-mask to genotypes (runner-specific: genotypes in memory)
    if not np.all(setup.valid_mask):
        genotypes = genotypes[setup.valid_mask, :]

    n_samples, n_snps = genotypes.shape

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Vectorized SNP stats and filtering using shared functions
    col_means, missing_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, missing_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )
    snp_indices = np.where(snp_mask)[0]

    if len(snp_indices) == 0:
        logger.warning(
            f"All {n_snps} SNPs filtered out (MAF>{maf_threshold}, "
            f"miss<{miss_threshold}). No association tests to run. "
            f"Consider relaxing --maf or --miss thresholds."
        )
        if output_path is not None:
            from jamma.lmm.io import IncrementalAssocWriter

            with IncrementalAssocWriter(
                output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            ):
                pass  # Header-only file, matching streaming runner behavior
        return LmmRunResult(associations=[], n_tested=0)

    # Extract filtered stats as numpy arrays (use allele_freqs for output, not mafs)
    filtered_afs = allele_freqs[snp_indices]
    filtered_miss = missing_counts[snp_indices].astype(int)

    t_eigen_start = time.perf_counter()
    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship,
        eigenvalues,
        eigenvectors,
        show_progress,
        "lmm_numpy",
        check_memory=check_memory,
    )
    del kinship
    gc.collect()

    # Use all physical cores for BLAS rotation
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = U.T @ W
        Uty = U.T @ phenotypes

    logl_H0, lambda_null_mle, Hi_eval_null = _compute_null_model_common(
        lmm_mode,
        eigenvalues_np,
        UtW,
        Uty,
        n_cvt,
        show_progress,
        l_min=l_min,
        l_max=l_max,
    )

    t_eigen_end = time.perf_counter()

    pve, pve_se = compute_and_log_pve(eigenvalues_np, UtW, Uty, n_cvt, l_min, l_max)

    n_filtered = len(snp_indices)

    filtered_means = col_means[snp_indices]

    # Streaming mode: write per-chunk to disk, skip arrays_out allocation.
    streaming = output_path is not None
    if streaming:
        from jamma.lmm.io import IncrementalAssocWriter

        writer_ctx = IncrementalAssocWriter(
            output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
        )
        arrays_out = None
    else:
        writer_ctx = None
        arrays_out = {
            key: np.empty(n_filtered, dtype=np.float64)
            for key in _RESULT_FIELDS[lmm_mode]
        }

    def _make_batch_source(
        source_chunk_size: int,
    ) -> Callable[[], RawLmmChunk | None]:
        chunk_starts = iter(range(0, n_filtered, source_chunk_size))
        geno_buf = np.empty((n_samples, source_chunk_size), dtype=np.float64)

        def _next_chunk() -> RawLmmChunk | None:
            try:
                chunk_start = next(chunk_starts)
            except StopIteration:
                return None

            chunk_end = min(chunk_start + source_chunk_size, n_filtered)
            actual_len = chunk_end - chunk_start
            geno_chunk = geno_buf[:, :actual_len]
            geno_chunk[:] = genotypes[:, snp_indices[chunk_start:chunk_end]]
            return RawLmmChunk(geno_chunk, chunk_start, chunk_end)

        return _next_chunk

    writer_cm = writer_ctx if streaming else nullcontext()
    with writer_cm as writer:

        def _sink(
            chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
        ) -> None:
            if streaming:
                assert writer is not None
                writer.write_arrays_batch(
                    lmm_mode,
                    snp_indices[filtered_start:filtered_end],
                    snp_info,
                    filtered_afs[filtered_start:filtered_end],
                    filtered_miss[filtered_start:filtered_end],
                    chunk_arrays,
                )
            else:
                assert arrays_out is not None
                s = slice(filtered_start, filtered_end)
                for key in arrays_out:
                    arrays_out[key][s] = chunk_arrays[key]

        chunk_stats = run_lmm_chunk_source_numpy(
            raw_chunk_source_factory=_make_batch_source,
            chunk_sink=_sink,
            U=U,
            eigenvalues_np=eigenvalues_np,
            UtW=UtW,
            Uty=Uty,
            Hi_eval_null=Hi_eval_null,
            logl_H0=logl_H0,
            n_samples=n_samples,
            n_filtered=n_filtered,
            n_cvt=n_cvt,
            lmm_mode=lmm_mode,
            filtered_means=filtered_means,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
            show_progress=show_progress,
            progress_label="LMM association",
        )

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_numpy", "after_all_chunks")

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        t_eigen = t_eigen_end - t_eigen_start
        accounted = (
            t_eigen
            + chunk_stats.rotation_s
            + chunk_stats.compute_s
            + chunk_stats.result_write_s
        )
        logger.info("Timing breakdown:")
        logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
        logger.info(f"  UT@G rotation:       {chunk_stats.rotation_s:.2f}s")
        logger.info(f"  NumPy compute:       {chunk_stats.compute_s:.2f}s")
        logger.info(f"  Result write:        {chunk_stats.result_write_s:.2f}s")
        logger.info("  ----")
        logger.info(f"  Accounted:           {accounted:.2f}s")
        logger.info(f"  Total:               {elapsed:.2f}s")
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    if streaming:
        return LmmRunResult(
            associations=[],
            pve=pve,
            pve_se=pve_se,
            n_tested=chunk_stats.processed,
        )

    assert arrays_out is not None
    return LmmRunResult(
        associations=_build_results(
            lmm_mode, snp_indices, filtered_afs, filtered_miss, snp_info, arrays_out
        ),
        pve=pve,
        pve_se=pve_se,
    )
