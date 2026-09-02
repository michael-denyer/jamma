"""The one kernel-selection decision for an LMM run, and the state it needs.

``make_kernel`` matches on ``DispatchPath`` exactly once. Each arm builds
whatever persistent state its path needs and binds the call that consumes it,
so a path's workspace and its invocation are written together and cannot drift
apart. This replaced a six-arm match and a seven-arm match in sibling modules,
one building a three-slot workspace tuple and one re-deciding which of them
to pass where.

``RunInvariants`` is the per-run state both halves used to receive separately:
sixteen positional arguments to the workspace builder, then thirteen of the
same values re-listed as compute-context fields. ``build`` reads them from the
prepared run and the config rather than having the caller re-list them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, assert_never

import numpy as np

from jamma.lmm import accel
from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.pab import build_pab_table_for_c
from jamma.lmm.prepare_common import PreparedLmmRun
from jamma.lmm.schema import LmmConfig, LmmMode
from jamma.lmm.uab import compute_uab_invariant_soa

# What a kernel hands back. The Wald C kernels return the WaldResult
# TypedDict; every other C kernel and the split paths return a plain
# dict[str, NDArray], and a TypedDict is not assignable to dict[str, Any].
# The engine only ever reads keys, so the read-only supertype is both
# accurate and wide enough for every path.
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
        dispatch: DispatchPath,
        prepared: PreparedLmmRun,
        config: LmmConfig,
        n_filtered: int,
    ) -> RunInvariants:
        """Derive the path-dependent members and freeze the rest."""
        UtW = prepared.UtW
        n_cvt = prepared.n_cvt
        return cls(
            dispatch=dispatch,
            lmm_mode=config.lmm_mode,
            n_cvt=n_cvt,
            n_samples=prepared.n_samples,
            n_filtered=n_filtered,
            eigenvalues=prepared.eigenvalues,
            UtW=UtW,
            Uty=prepared.Uty,
            Hi_eval_null=prepared.Hi_eval_null,
            logl_H0=prepared.logl_H0,
            l_min=config.l_min,
            l_max=config.l_max,
            n_grid=config.n_grid,
            n_refine=config.n_refine,
            w=UtW[:, 0].copy() if dispatch.needs_null_w else None,
            uab_invariant_soa=(
                compute_uab_invariant_soa(UtW, prepared.Uty, n_cvt)
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
    """One dispatch path's persistent state, bound to the call that uses it.

    The path's workspace, where it has one, is captured by ``call``, so the
    PyCapsule lives exactly as long as the kernel that can invoke it.
    """

    label: str
    n_filtered: int
    call: Callable[[np.ndarray, int], KernelResult]
    uses_c: bool  # True for every path but NUMPY_FALLBACK; see make_kernel

    def compute_chunk(
        self, chunk_data: np.ndarray, n_threads: int, write_offset: int
    ) -> KernelResult:
        """Run one prepared chunk, labelling any failure with its SNP offset.

        MemoryError, ValueError, TypeError, and OverflowError already say what
        went wrong, so they propagate untouched. Everything else, including the
        OSError that models a C-kernel segfault, is wrapped so the message names
        the kernel and how far the run had got.
        """
        try:
            return self.call(chunk_data, n_threads)
        except (MemoryError, ValueError, TypeError, OverflowError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"{self.label} failed at SNP offset "
                f"{write_offset}/{self.n_filtered}. "
                f"Processed {write_offset} SNPs before failure."
            ) from exc


def make_kernel(inv: RunInvariants, n_threads: int) -> Kernel:
    """Build the one kernel this run's dispatch path selects.

    *n_threads* sizes the per-thread scratch inside a persistent workspace and
    is fixed for the run. The thread count handed to each chunk is separate and
    may change after the pipeline profiles its first chunk.
    """
    match inv.dispatch:
        case (
            DispatchPath.FUSED | DispatchPath.FUSED_SCORE_WS | DispatchPath.FUSED_LRT_WS
        ):
            return _ncvt1_kernel(inv)
        case DispatchPath.FUSED_GENERAL:
            return _fused_general_kernel(inv, n_threads)
        case DispatchPath.NUMPY_FALLBACK:
            return _numpy_kernel(inv)
        case _:
            assert_never(inv.dispatch)


# The kernel label for each n_cvt=1 lmm_mode. One C entry point serves every
# mode, reading the mode off the workspace; the label is what a failure
# reports, so mode 4 keeps its own.
_NCVT1_LABEL: dict[int, str] = {
    1: "Fused Uab dispatch",
    2: "Fused LRT WS dispatch",
    3: "Fused Score WS dispatch",
    4: "Fused mode-4 Uab dispatch",
}


def _ncvt1_kernel(inv: RunInvariants) -> Kernel:
    """n_cvt=1, any mode: one workspace keyed by lmm_mode, one compute.

    The workspace packs w, the lambda grid and the null-model block the mode
    needs, built once; each chunk hands in utg_t. Scratch is sized per call,
    so the run-level thread count plays no part here.
    """
    workspace = accel.require().create_workspace_ncvt1_c(
        inv.eigenvalues,
        inv.require_invariant_soa(),
        inv.require_null_w(),
        inv.Uty,
        inv.n_samples,
        inv.l_min,
        inv.l_max,
        inv.n_grid,
        inv.n_refine,
        lmm_mode=inv.lmm_mode,
        **_null_model_kwargs(inv),
    )
    compute = accel.require().compute_lmm_chunk_ncvt1_c
    return Kernel(
        label=_NCVT1_LABEL[inv.lmm_mode],
        n_filtered=inv.n_filtered,
        call=lambda chunk, threads: compute(workspace, chunk, threads),
        uses_c=True,
    )


_GENERAL_LABEL: dict[int, str] = {
    1: "Fused general Uab dispatch",
    2: "Fused general LRT Uab dispatch",
    3: "Fused general Score Uab dispatch",
    4: "Fused general mode-4 Uab dispatch",
}


def _fused_general_kernel(inv: RunInvariants, n_threads: int) -> Kernel:
    """n_cvt>=2, any mode: same shape as n_cvt=1, plus the Pab table.

    The workspace sizes its per-thread scratch from *n_threads* once, so the
    run-level thread count is part of its construction here.
    """
    workspace = accel.require().create_workspace_general_c(
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
        build_pab_table_for_c(inv.n_cvt)._asdict(),
        lmm_mode=inv.lmm_mode,
        **_null_model_kwargs(inv),
    )
    compute = accel.require().compute_lmm_chunk_fused_general_c
    return Kernel(
        label=_GENERAL_LABEL[inv.lmm_mode],
        n_filtered=inv.n_filtered,
        call=lambda chunk, threads: compute(workspace, chunk, threads),
        uses_c=True,
    )


def _numpy_kernel(inv: RunInvariants) -> Kernel:
    """No C extension: the full-Uab pure-NumPy path, chunk by chunk."""

    def call(chunk: np.ndarray, threads: int) -> KernelResult:
        del threads
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
        )

    return Kernel(
        label="LMM chunk compute", n_filtered=inv.n_filtered, call=call, uses_c=False
    )


def _null_model_kwargs(inv: RunInvariants) -> dict[str, Any]:
    """The null-model inputs a C workspace creator takes for this mode.

    Score (3) needs ``hi_eval_null``, LRT (2) needs ``logl_H0``, mode 4 both,
    Wald (1) neither. Both creators reject an input their mode does not use.
    """
    kwargs: dict[str, Any] = {}
    if inv.lmm_mode in (3, 4):
        kwargs["hi_eval_null"] = inv.Hi_eval_null
    if inv.lmm_mode in (2, 4):
        kwargs["logl_H0"] = inv.logl_H0
    return kwargs
