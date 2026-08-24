"""Persistent C-workspace lifecycle for the shared NumPy LMM chunk engine.

Creates the PyCapsule workspaces once before the chunk loop (lambda grids,
hi_eval/logdet grids, invariant Iab column sums, per-thread scratch) so the C
kernels reuse them across every chunk without per-chunk malloc/free. Split out
from ``chunk_runner_numpy`` so workspace allocation is isolated from the
per-chunk dispatch and the loop driver.
"""

from __future__ import annotations

from typing import Any, NamedTuple, assert_never, cast

import numpy as np

from jamma.lmm.compute_numpy import (
    LmmMode,
    create_lmm_workspace_fused,
    create_lmm_workspace_fused_general,
    create_lmm_workspace_mode4_fused,
    create_lmm_workspace_mode4_fused_general,
)
from jamma.lmm.dispatch import DispatchPath


class _Workspaces(NamedTuple):
    """Persistent C workspaces created once before the chunk loop.

    Each is a PyCapsule (or None when its path is inactive), freed when the tuple
    goes out of scope. Shared by both the batch and streaming runners.
    """

    lmm_workspace: object | None
    score_fused_workspace: object | None
    lrt_fused_workspace: object | None


def _create_workspaces(
    dispatch: DispatchPath,
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

    # One authoritative decode. The Wald-family workspace (modes 1/4) and the
    # Score/LRT workspaces (modes 3/2) are mutually exclusive by mode, so each
    # path builds at most one. Score: null-model dot products and F-distribution
    # constants. LRT: lambda grids and per-thread scratch. All persist across
    # every chunk, eliminating per-chunk malloc/free and precomputation.
    lmm_workspace = None
    score_fused_workspace = None
    lrt_fused_workspace = None

    # Only the mode-4 constructors take the null model; their Wald twins reject
    # these names. Building the pair once keeps each family to a single call.
    # Typed dict[str, Any] because pyrefly cannot map ** spread keys to params,
    # so a narrower type would union with the pab_* arrays at the call site.
    null_model_kwargs: dict[str, Any] = (
        {"hi_eval_null": Hi_eval_null, "logl_H0": logl_H0} if lmm_mode == 4 else {}
    )

    match dispatch:
        case DispatchPath.FUSED_GENERAL:
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
            create = (
                create_lmm_workspace_mode4_fused_general
                if lmm_mode == 4
                else create_lmm_workspace_fused_general
            )
            lmm_workspace = create(
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
                **null_model_kwargs,
            )
        case DispatchPath.FUSED:
            create = (
                create_lmm_workspace_mode4_fused
                if lmm_mode == 4
                else create_lmm_workspace_fused
            )
            if w is None:
                raise RuntimeError("fused Wald dispatch requires the null-model w")
            lmm_workspace = create(
                eigenvalues_np,
                uab_invariant,
                w,
                Uty,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                n_threads,
                **null_model_kwargs,
            )
        case DispatchPath.SOA_SPLIT:
            # Reached only for n_cvt>=2 modes 2 and 3, which compute per
            # chunk with no persistent workspace. Modes 1 and 4 at that
            # covariate count take the fused general kernel.
            pass
        case DispatchPath.FUSED_SCORE_WS:
            from jamma.lmm.compute_numpy import _c

            score_fused_workspace = _c().create_workspace_score_fused_c(
                w,
                Uty,
                Hi_eval_null,
                eigenvalues_np,
                uab_invariant,
                n_samples,
                n_threads,
            )
        case DispatchPath.FUSED_LRT_WS:
            from jamma.lmm.compute_numpy import _c

            lrt_fused_workspace = _c().create_workspace_lrt_fused_c(
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
        case DispatchPath.NUMPY_FALLBACK:
            # The NumPy fallback holds no persistent workspace.
            pass
        case _:
            assert_never(dispatch)

    return _Workspaces(lmm_workspace, score_fused_workspace, lrt_fused_workspace)
