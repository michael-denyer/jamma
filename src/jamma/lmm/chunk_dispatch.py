"""Per-chunk C/Python kernel dispatch for the shared NumPy LMM chunk engine.

Owns the kernel-selection ladder (fused-general / fused / fused-Score-WS /
fused-LRT-WS / fused-Score / fused-LRT / SoA-split) and the error wrapping that
labels a failed operation with its SNP offset. Built once as a ``_ComputeContext``
before the chunk loop and consulted per chunk. Split out from
``chunk_runner_numpy`` so the dispatch decision tree lives in exactly one place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, assert_never, cast

import numpy as np

from jamma.lmm.compute_numpy import (
    LmmMode,
    _compute_lrt_split_numpy,
    _compute_score_split_numpy,
    compute_mode4_fused_c_ws,
    compute_mode4_fused_general_c_ws,
    compute_mode4_split_c_ws,
    compute_wald_fused_c_ws,
    compute_wald_fused_general_c_ws,
    compute_wald_general_c_ws,
    compute_wald_split_c_ws,
)
from jamma.lmm.dispatch import DispatchPath

# The fused Wald/mode-4 kernels, keyed by (path, is mode 4). A table rather than
# nested conditionals: the two axes are independent, so adding a fused variant is
# a row here instead of another branch inside the dispatch match.
_FUSED_KERNELS: dict[tuple[DispatchPath, bool], tuple[Callable[..., Any], str]] = {
    (DispatchPath.FUSED, False): (compute_wald_fused_c_ws, "Fused Uab dispatch"),
    (DispatchPath.FUSED, True): (compute_mode4_fused_c_ws, "Fused Uab dispatch"),
    (DispatchPath.FUSED_GENERAL, False): (
        compute_wald_fused_general_c_ws,
        "Fused general Uab dispatch",
    ),
    (DispatchPath.FUSED_GENERAL, True): (
        compute_mode4_fused_general_c_ws,
        "Fused general mode-4 Uab dispatch",
    ),
}

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


def _guarded_compute(
    fn: Callable[..., Any],
    *args: object,
    operation: str,
    write_offset: int,
    n_filtered: int,
    **kwargs: object,
) -> dict[str, Any]:
    """Call *fn* with error wrapping that identifies the failed operation.

    Extra positional and keyword arguments are forwarded to *fn*;
    *operation*, *write_offset*, and *n_filtered* are consumed by the wrapper.

    MemoryError, ValueError, TypeError, and OverflowError propagate unchanged.
    All other exceptions (including OSError, used here to model a C-kernel
    segfault) are wrapped in a RuntimeError whose message includes the
    *operation* label, *write_offset*, and *n_filtered* for diagnosis.
    """
    try:
        return cast(dict[str, Any], fn(*args, **kwargs))
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
        wald_cr = cast(
            dict[str, np.ndarray], wald_fn(lmm_workspace, uab_var_soa, n_threads)
        )
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
    chunk, so the kernel-selection ladder lives in exactly one place
    (previously duplicated across the batch pipeline, batch sequential, and
    streaming compute paths).
    """

    dispatch: DispatchPath
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
) -> dict[str, Any]:
    """Dispatch one prepared chunk to the active C kernel and return its result.

    Matches on ``ctx.dispatch`` (a ``DispatchPath``) to select the C kernel.
    ``chunk_input`` is utg_t for the fused paths and the varying-Uab SoA array
    for the split paths. The caller owns BLAS-thread scoping, input preparation,
    and the non-split NumPy fallback (``NUMPY_FALLBACK`` never reaches here).
    """
    match ctx.dispatch:
        case DispatchPath.FUSED | DispatchPath.FUSED_GENERAL:
            fused_fn, op_label = _FUSED_KERNELS[ctx.dispatch, ctx.lmm_mode == 4]
            return _guarded_compute(
                fused_fn,
                ctx.lmm_workspace,
                chunk_input,
                n_threads,
                operation=op_label,
                write_offset=write_offset,
                n_filtered=ctx.n_filtered,
            )
        case DispatchPath.FUSED_SCORE_WS:
            from jamma.lmm.compute_numpy import _compute_score_fused_ws_c

            if _compute_score_fused_ws_c is None:
                raise RuntimeError("fused Score workspace dispatch requires C support")
            return _guarded_compute(
                _compute_score_fused_ws_c,
                ctx.score_fused_workspace,
                chunk_input,
                n_threads,
                operation="Fused Score WS dispatch",
                write_offset=write_offset,
                n_filtered=ctx.n_filtered,
            )
        case DispatchPath.FUSED_LRT_WS:
            from jamma.lmm.compute_numpy import _compute_lrt_fused_ws_c

            if _compute_lrt_fused_ws_c is None:
                raise RuntimeError("fused LRT workspace dispatch requires C support")
            return _guarded_compute(
                _compute_lrt_fused_ws_c,
                ctx.lrt_fused_workspace,
                chunk_input,
                n_threads,
                operation="Fused LRT WS dispatch",
                write_offset=write_offset,
                n_filtered=ctx.n_filtered,
            )
        case DispatchPath.FUSED_SCORE:
            from jamma.lmm.compute_numpy import _compute_score_fused_c

            if _compute_score_fused_c is None:
                raise RuntimeError("fused Score dispatch requires C support")
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
        case DispatchPath.FUSED_LRT:
            from jamma.lmm.compute_numpy import _compute_lrt_fused_c

            if _compute_lrt_fused_c is None:
                raise RuntimeError("fused LRT dispatch requires C support")
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
        case DispatchPath.SOA_SPLIT | DispatchPath.SOA_SPLIT_MODE4:
            # chunk_input is the varying-Uab SoA array.
            return _guarded_compute(
                dispatch_soa_split,
                ctx.lmm_mode,
                ctx.dispatch is DispatchPath.SOA_SPLIT_MODE4,
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
        case DispatchPath.NUMPY_FALLBACK:
            raise AssertionError(
                "_dispatch_compute must not be called for NUMPY_FALLBACK; "
                "the chunk runner gates on use_split before calling this."
            )
        case _:
            assert_never(ctx.dispatch)
