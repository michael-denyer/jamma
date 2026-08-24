"""The one kernel-selection decision for an LMM run, and the state it needs.

``make_kernel`` matches on ``DispatchPath`` exactly once. Each arm builds
whatever persistent state its path needs and binds the call that consumes it,
so a path's workspace and its invocation are written together and cannot drift
apart. This replaced a pair of six-arm matches in sibling modules, one building
a three-slot workspace tuple and one re-deciding which of them to pass where.

``RunInvariants`` is the per-run state both halves used to receive separately:
sixteen positional arguments to the workspace builder, then thirteen of the
same values re-listed as compute-context fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, assert_never, cast

import numpy as np

from jamma.lmm.compute_numpy import (
    LmmMode,
    _c,
    _compute_lrt_split_numpy,
    _compute_score_split_numpy,
    compute_lmm_chunk_numpy,
    compute_mode4_fused_c_ws,
    compute_mode4_fused_general_c_ws,
    compute_wald_fused_c_ws,
    compute_wald_fused_general_c_ws,
    create_lmm_workspace_fused,
    create_lmm_workspace_fused_general,
    create_lmm_workspace_mode4_fused,
    create_lmm_workspace_mode4_fused_general,
)
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.likelihood_numpy import compute_uab_invariant_soa

# What a kernel hands back. The C kernels return TypedDicts (WaldResult and
# friends) and the split paths return plain dicts, and a TypedDict is not
# assignable to dict[str, Any]. The engine only ever reads keys, so the
# read-only supertype is both accurate and wide enough for every path.
KernelResult = Mapping[str, Any]


@dataclass(frozen=True)
class RunInvariants:
    """Everything a kernel needs that does not vary from chunk to chunk.

    Built once by :meth:`build`, which owns the two values derived from the
    dispatch path rather than leaving each caller to derive them: the
    null-model ``w`` column and the invariant Uab columns.
    """

    dispatch: DispatchPath
    lmm_mode: LmmMode
    n_cvt: int
    n_samples: int
    n_filtered: int
    eigenvalues: np.ndarray
    UtW: np.ndarray
    Uty: np.ndarray
    Hi_eval_null: np.ndarray
    logl_H0: float
    l_min: float
    l_max: float
    n_grid: int
    n_refine: int
    w: np.ndarray | None
    uab_invariant_soa: np.ndarray | None

    @classmethod
    def build(
        cls,
        *,
        dispatch: DispatchPath,
        lmm_mode: LmmMode,
        n_cvt: int,
        n_samples: int,
        n_filtered: int,
        eigenvalues: np.ndarray,
        UtW: np.ndarray,
        Uty: np.ndarray,
        Hi_eval_null: np.ndarray,
        logl_H0: float,
        l_min: float,
        l_max: float,
        n_grid: int,
        n_refine: int,
    ) -> RunInvariants:
        """Derive the path-dependent members and freeze the rest."""
        return cls(
            dispatch=dispatch,
            lmm_mode=lmm_mode,
            n_cvt=n_cvt,
            n_samples=n_samples,
            n_filtered=n_filtered,
            eigenvalues=eigenvalues,
            UtW=UtW,
            Uty=Uty,
            Hi_eval_null=Hi_eval_null,
            logl_H0=logl_H0,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
            w=UtW[:, 0].copy() if dispatch.needs_null_w else None,
            uab_invariant_soa=(
                compute_uab_invariant_soa(UtW, Uty, n_cvt)
                if dispatch.use_split
                else None
            ),
        )

    def require_invariant_soa(self) -> np.ndarray:
        """The invariant Uab columns, which every split path is built with."""
        if self.uab_invariant_soa is None:
            raise RuntimeError("split LMM dispatch requires invariant Uab columns")
        return self.uab_invariant_soa

    def require_null_w(self) -> np.ndarray:
        """The null-model ``w`` column, which the fused paths are built with."""
        if self.w is None:
            raise RuntimeError("fused dispatch requires the null-model w")
        return self.w


@dataclass(frozen=True)
class Kernel:
    """One dispatch path's persistent state bound to the call that uses it.

    ``workspace`` is held only to keep the PyCapsule alive for as long as
    ``call`` can be invoked; nothing reads it. It is None for the two paths
    that compute a chunk outright.
    """

    label: str
    n_filtered: int
    call: Callable[[np.ndarray, int], KernelResult]
    workspace: object | None = None

    def compute_chunk(
        self, chunk_data: np.ndarray, n_threads: int, write_offset: int
    ) -> KernelResult:
        """Run one prepared chunk, labelling any failure with its SNP offset.

        MemoryError, ValueError, TypeError, and OverflowError propagate
        unchanged. Everything else (including OSError, which models a C-kernel
        segfault) is wrapped so the message names the operation and the offset.
        """
        return _guarded_compute(
            self.call,
            chunk_data,
            n_threads,
            operation=self.label,
            write_offset=write_offset,
            n_filtered=self.n_filtered,
        )


def _guarded_compute(
    fn: Callable[..., Any],
    *args: object,
    operation: str,
    write_offset: int,
    n_filtered: int,
    **kwargs: object,
) -> KernelResult:
    """Call *fn* with error wrapping that identifies the failed operation.

    Extra positional and keyword arguments are forwarded to *fn*;
    *operation*, *write_offset*, and *n_filtered* are consumed by the wrapper.
    """
    try:
        return cast(KernelResult, fn(*args, **kwargs))
    except (MemoryError, ValueError, TypeError, OverflowError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"{operation} failed at SNP offset "
            f"{write_offset}/{n_filtered}. "
            f"Processed {write_offset} SNPs before failure."
        ) from exc


def make_kernel(inv: RunInvariants, n_threads: int) -> Kernel:
    """Build the one kernel this run's dispatch path selects.

    *n_threads* sizes the per-thread scratch inside a persistent workspace and
    is fixed for the run. The thread count handed to each chunk is separate and
    may change after the pipeline profiles its first chunk.
    """
    match inv.dispatch:
        case DispatchPath.FUSED:
            return _fused_kernel(inv, n_threads)
        case DispatchPath.FUSED_GENERAL:
            return _fused_general_kernel(inv, n_threads)
        case DispatchPath.FUSED_SCORE_WS:
            return _score_ws_kernel(inv, n_threads)
        case DispatchPath.FUSED_LRT_WS:
            return _lrt_ws_kernel(inv, n_threads)
        case DispatchPath.SOA_SPLIT:
            return _soa_split_kernel(inv)
        case DispatchPath.NUMPY_FALLBACK:
            return _numpy_kernel(inv)
        case _:
            assert_never(inv.dispatch)


def _fused_kernel(inv: RunInvariants, n_threads: int) -> Kernel:
    """n_cvt=1 Wald or mode 4: the workspace packs w, and takes utg_t per chunk."""
    is_mode4 = inv.lmm_mode == 4
    create = (
        create_lmm_workspace_mode4_fused if is_mode4 else create_lmm_workspace_fused
    )
    workspace = create(
        inv.eigenvalues,
        inv.require_invariant_soa(),
        inv.require_null_w(),
        inv.Uty,
        inv.n_samples,
        inv.l_min,
        inv.l_max,
        inv.n_grid,
        inv.n_refine,
        n_threads,
        **_null_model_kwargs(inv),
    )
    compute = compute_mode4_fused_c_ws if is_mode4 else compute_wald_fused_c_ws
    # Mode 4 gets its own label, as it already did on the general path. The
    # table this replaced gave both n_cvt=1 kernels the same one, so a segfault
    # in the mode-4 kernel reported as a Wald failure.
    label = "Fused mode-4 Uab dispatch" if is_mode4 else "Fused Uab dispatch"
    return Kernel(
        label=label,
        n_filtered=inv.n_filtered,
        call=lambda chunk, threads: compute(workspace, chunk, threads),
        workspace=workspace,
    )


def _fused_general_kernel(inv: RunInvariants, n_threads: int) -> Kernel:
    """n_cvt>=2 Wald or mode 4: same shape, plus the Pab table the kernel walks."""
    from jamma.lmm.likelihood import build_pab_table_for_c

    is_mode4 = inv.lmm_mode == 4
    pab_c = build_pab_table_for_c(inv.n_cvt)
    pab_kwargs = {key: pab_c[key] for key in _PAB_TABLE_KEYS}
    create = (
        create_lmm_workspace_mode4_fused_general
        if is_mode4
        else create_lmm_workspace_fused_general
    )
    workspace = create(
        inv.eigenvalues,
        inv.require_invariant_soa(),
        inv.UtW,
        inv.Uty,
        inv.n_samples,
        inv.l_min,
        inv.l_max,
        inv.n_grid,
        inv.n_refine,
        n_threads,
        n_cvt=inv.n_cvt,
        **pab_kwargs,
        **_null_model_kwargs(inv),
    )
    compute = (
        compute_mode4_fused_general_c_ws
        if is_mode4
        else compute_wald_fused_general_c_ws
    )
    label = (
        "Fused general mode-4 Uab dispatch"
        if is_mode4
        else "Fused general Uab dispatch"
    )
    return Kernel(
        label=label,
        n_filtered=inv.n_filtered,
        call=lambda chunk, threads: compute(workspace, chunk, threads),
        workspace=workspace,
    )


def _score_ws_kernel(inv: RunInvariants, n_threads: int) -> Kernel:
    """n_cvt=1 Score: null-model dot products and F constants, computed once."""
    workspace = _c().create_workspace_score_fused_c(
        inv.require_null_w(),
        inv.Uty,
        inv.Hi_eval_null,
        inv.eigenvalues,
        inv.require_invariant_soa(),
        inv.n_samples,
        n_threads,
    )
    compute = _c().compute_score_fused_ws_c
    return Kernel(
        label="Fused Score WS dispatch",
        n_filtered=inv.n_filtered,
        call=lambda chunk, threads: compute(workspace, chunk, threads),
        workspace=workspace,
    )


def _lrt_ws_kernel(inv: RunInvariants, n_threads: int) -> Kernel:
    """n_cvt=1 LRT: lambda grids and per-thread scratch, allocated once."""
    workspace = _c().create_workspace_lrt_fused_c(
        inv.require_null_w(),
        inv.Uty,
        inv.eigenvalues,
        inv.require_invariant_soa(),
        inv.n_samples,
        inv.l_min,
        inv.l_max,
        inv.n_grid,
        inv.n_refine,
        inv.logl_H0,
        n_threads,
    )
    compute = _c().compute_lrt_fused_ws_c
    return Kernel(
        label="Fused LRT WS dispatch",
        n_filtered=inv.n_filtered,
        call=lambda chunk, threads: compute(workspace, chunk, threads),
        workspace=workspace,
    )


def _soa_split_kernel(inv: RunInvariants) -> Kernel:
    """n_cvt>=2 Score or LRT: no persistent workspace, varying Uab SoA per chunk.

    The mode guard fires here rather than per chunk, so a dispatch table that
    ever routed Wald or mode 4 this way fails before the loop starts instead of
    on its first chunk.
    """
    invariant = inv.require_invariant_soa()
    if inv.lmm_mode == 3:

        def call(chunk: np.ndarray, threads: int) -> KernelResult:
            return _compute_score_split_numpy(
                inv.n_cvt,
                inv.eigenvalues,
                inv.Hi_eval_null,
                chunk,
                invariant,
                inv.n_samples,
                threads,
            )
    elif inv.lmm_mode == 2:

        def call(chunk: np.ndarray, threads: int) -> KernelResult:
            return _compute_lrt_split_numpy(
                inv.n_cvt,
                inv.eigenvalues,
                chunk,
                invariant,
                inv.n_samples,
                inv.l_min,
                inv.l_max,
                inv.n_grid,
                inv.n_refine,
                inv.logl_H0,
                threads,
            )
    else:
        raise ValueError(
            f"Unexpected lmm_mode={inv.lmm_mode} in SoA split dispatch. This path "
            f"serves n_cvt>=2 modes 2 (LRT) and 3 (Score); modes 1 and 4 take the "
            f"fused general kernel."
        )

    return Kernel(label="SoA split dispatch", n_filtered=inv.n_filtered, call=call)


def _numpy_kernel(inv: RunInvariants) -> Kernel:
    """No C extension: the full-Uab pure-NumPy path, chunk by chunk."""

    def call(chunk: np.ndarray, threads: int) -> KernelResult:
        return compute_lmm_chunk_numpy(
            inv.lmm_mode,
            inv.n_cvt,
            inv.eigenvalues,
            chunk,
            inv.n_samples,
            l_min=inv.l_min,
            l_max=inv.l_max,
            n_grid=inv.n_grid,
            n_refine=inv.n_refine,
            Hi_eval_null=inv.Hi_eval_null,
            logl_H0=inv.logl_H0,
            n_threads=threads,
        )

    return Kernel(label="LMM chunk compute", n_filtered=inv.n_filtered, call=call)


def _null_model_kwargs(inv: RunInvariants) -> dict[str, Any]:
    """Mode 4 constructors take the null model; their Wald twins reject it."""
    if inv.lmm_mode != 4:
        return {}
    return {"hi_eval_null": inv.Hi_eval_null, "logl_H0": inv.logl_H0}


_PAB_TABLE_KEYS = (
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
)
