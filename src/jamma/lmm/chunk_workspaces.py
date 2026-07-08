"""Persistent C-workspace lifecycle for the shared NumPy LMM chunk engine.

Creates the PyCapsule workspaces once before the chunk loop (lambda grids,
hi_eval/logdet grids, invariant Iab column sums, per-thread scratch) so the C
kernels reuse them across every chunk without per-chunk malloc/free. Split out
from ``chunk_runner_numpy`` so workspace allocation is isolated from the
per-chunk dispatch and the loop driver.
"""

from __future__ import annotations

from typing import NamedTuple, cast

import numpy as np
from loguru import logger

from jamma.lmm.compute_numpy import (
    _C_GENERAL_AVAILABLE,
    LmmMode,
    create_lmm_workspace,
    create_lmm_workspace_fused,
    create_lmm_workspace_fused_general,
    create_lmm_workspace_general,
    create_lmm_workspace_mode4,
    create_lmm_workspace_mode4_fused,
    create_lmm_workspace_mode4_fused_general,
)
from jamma.lmm.dispatch import LmmDispatch


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
    if dispatch.use_split and uab_invariant_soa is None:
        raise RuntimeError("split LMM dispatch requires invariant Uab columns")
    uab_invariant = cast(np.ndarray, uab_invariant_soa)

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
                    uab_invariant,
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
                    uab_invariant,
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
            w_fused = UtW[:, 0].copy()
            lmm_workspace = create_lmm_workspace_mode4_fused(
                eigenvalues_np,
                uab_invariant,
                w_fused,
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
            w_fused = UtW[:, 0].copy()
            lmm_workspace = create_lmm_workspace_fused(
                eigenvalues_np,
                uab_invariant,
                w_fused,
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
                uab_invariant,
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
                uab_invariant,
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

        if _create_workspace_score_fused_c is None:
            raise RuntimeError("fused Score workspace dispatch requires C support")
        score_fused_workspace = _create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues_np,
            uab_invariant,
            n_samples,
            n_threads,
        )

    if dispatch.use_fused_lrt_ws:
        from jamma.lmm.compute_numpy import _create_workspace_lrt_fused_c

        if _create_workspace_lrt_fused_c is None:
            raise RuntimeError("fused LRT workspace dispatch requires C support")
        lrt_fused_workspace = _create_workspace_lrt_fused_c(
            w,
            Uty,
            eigenvalues_np,
            uab_invariant,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            n_threads,
        )

    return _Workspaces(lmm_workspace, score_fused_workspace, lrt_fused_workspace)
