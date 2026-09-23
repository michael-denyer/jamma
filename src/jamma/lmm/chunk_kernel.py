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
from typing import assert_never

import numpy as np

from jamma.lmm import accel
from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy, compute_wald_split_numpy
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.prepare_common import NullFit, RotatedBasis
from jamma.lmm.schema import MODE_SPECS, LmmConfig, LmmMode, LmmTest, ModeSpec
from jamma.lmm.uab import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_iab_invariant_scalars_ncvt1,
    compute_uab_invariant_soa,
)
from jamma.lmm.workspace import WorkspaceSpec

KernelResult = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class RunInvariants:
    """Everything a kernel needs that does not vary from chunk to chunk.

    Built once by :meth:`build`, which owns the value derived from the
    dispatch path rather than leaving each caller to derive it: the invariant
    Uab columns.
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
    uab_invariant_soa: np.ndarray | None

    @classmethod
    def build(
        cls,
        dispatch: DispatchPath,
        basis: RotatedBasis,
        fit: NullFit,
        config: LmmConfig,
        n_filtered: int,
    ) -> RunInvariants:
        """Derive the path-dependent members and freeze the rest."""
        UtW = basis.UtW
        n_cvt = basis.n_cvt
        return cls(
            dispatch=dispatch,
            lmm_mode=config.lmm_mode,
            n_cvt=n_cvt,
            n_samples=basis.n_samples,
            n_filtered=n_filtered,
            eigenvalues=basis.eigenvalues,
            UtW=UtW,
            Uty=fit.Uty,
            Hi_eval_null=fit.Hi_eval_null,
            logl_H0=fit.logl_H0,
            l_min=config.l_min,
            l_max=config.l_max,
            n_grid=config.n_grid,
            n_refine=config.n_refine,
            uab_invariant_soa=(
                compute_uab_invariant_soa(UtW, fit.Uty, n_cvt)
                if dispatch.invariant_rows(n_cvt) > 0
                else None
            ),
        )

    @property
    def mode(self) -> ModeSpec:
        """The specification of ``lmm_mode``."""
        return MODE_SPECS[self.lmm_mode]

    def require_invariant_soa(self) -> np.ndarray:
        """The invariant Uab columns, which every split path is built with."""
        if self.uab_invariant_soa is None:
            raise RuntimeError("split LMM dispatch requires invariant Uab columns")
        return self.uab_invariant_soa


@dataclass(frozen=True)
class Kernel:
    """One dispatch path's persistent state, bound to the call that uses it.

    The path's workspace, where it has one, is captured by ``call``, so the
    PyCapsule lives exactly as long as the kernel that can invoke it.
    """

    label: str
    n_filtered: int
    call: Callable[[np.ndarray, int], KernelResult]
    max_threads: int

    def compute_chunk(
        self, chunk_data: np.ndarray, n_threads: int, write_offset: int
    ) -> KernelResult:
        """Run one prepared chunk, labelling any failure with its SNP offset.

        MemoryError, ValueError, TypeError, and OverflowError already say what
        went wrong, so they propagate untouched. Everything else, including the
        OSError that models a C-kernel segfault, is wrapped so the message names
        the kernel and how far the run had got.
        """
        if n_threads > self.max_threads:
            raise ValueError(
                f"kernel thread count {n_threads} exceeds workspace capacity "
                f"{self.max_threads}"
            )
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


def make_kernel(inv: RunInvariants, workspace: WorkspaceSpec) -> Kernel:
    """Build the one kernel this run's dispatch path selects.

    ``workspace.max_threads`` sizes the workspace's thread capacity.
    The thread count handed to each chunk may be smaller, but cannot exceed the
    capacity priced before allocation.
    """
    if (
        workspace.dispatch is not inv.dispatch
        or workspace.lmm_mode != inv.lmm_mode
        or workspace.n_samples != inv.n_samples
        or workspace.n_cvt != inv.n_cvt
        or workspace.n_grid != inv.n_grid
        or workspace.n_refine != inv.n_refine
    ):
        raise ValueError("workspace specification does not match kernel invariants")
    match inv.dispatch:
        case DispatchPath.FUSED:
            return _fused_kernel(inv, workspace.max_threads)
        case DispatchPath.NUMPY_WALD:
            return _numpy_wald_kernel(inv, workspace.max_threads)
        case DispatchPath.NUMPY_FALLBACK:
            return _numpy_kernel(inv, workspace.max_threads)
        case _:
            assert_never(inv.dispatch)


def _fused_kernel(inv: RunInvariants, n_threads: int) -> Kernel:
    """Any n_cvt, any mode: one C workspace built once, one compute per chunk.

    The workspace packs the lambda grid, the null-model block the mode needs
    and per-thread scratch for *n_threads*; each chunk hands in utg_t.
    """
    workspace = accel.require().create_workspace_c(
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
        inv.n_cvt,
        lmm_mode=inv.lmm_mode,
        **_null_model_kwargs(inv),
    )
    compute = accel.require().compute_lmm_chunk_c
    return Kernel(
        label=f"Fused -lmm {inv.lmm_mode} dispatch",
        n_filtered=inv.n_filtered,
        call=lambda chunk, threads: compute(workspace, chunk, threads),
        max_threads=n_threads,
    )


def _numpy_wald_kernel(inv: RunInvariants, max_threads: int) -> Kernel:
    """n_cvt=1, mode 1, no C extension: the split Wald body in NumPy.

    The Iab scalars are derived once here rather than per chunk, which is what
    lets each chunk contribute three varying rows instead of the whole table.
    """
    invariant = inv.require_invariant_soa()
    scalars = compute_iab_invariant_scalars_ncvt1(invariant)

    def call(chunk: np.ndarray, threads: int) -> KernelResult:
        varying = batch_compute_uab_varying_soa_numpy(1, inv.UtW, inv.Uty, chunk)
        return compute_wald_split_numpy(
            inv.eigenvalues,
            varying,
            invariant,
            scalars,
            inv.n_samples,
            l_min=inv.l_min,
            l_max=inv.l_max,
            n_grid=inv.n_grid,
            n_refine=inv.n_refine,
        )

    return Kernel(
        label="NumPy Wald",
        n_filtered=inv.n_filtered,
        call=call,
        max_threads=max_threads,
    )


def _numpy_kernel(inv: RunInvariants, max_threads: int) -> Kernel:
    """No C extension: the full-Uab pure-NumPy path, chunk by chunk."""

    def call(chunk: np.ndarray, threads: int) -> KernelResult:
        del threads
        return compute_lmm_chunk_numpy(
            inv.lmm_mode,
            inv.n_cvt,
            inv.eigenvalues,
            batch_compute_uab_numpy(inv.n_cvt, inv.UtW, inv.Uty, chunk),
            inv.n_samples,
            l_min=inv.l_min,
            l_max=inv.l_max,
            n_grid=inv.n_grid,
            n_refine=inv.n_refine,
            Hi_eval_null=inv.Hi_eval_null,
            logl_H0=inv.logl_H0,
        )

    return Kernel(
        label="LMM chunk compute",
        n_filtered=inv.n_filtered,
        call=call,
        max_threads=max_threads,
    )


def _null_model_kwargs(inv: RunInvariants) -> dict[str, np.ndarray | float]:
    """The null-model inputs a C workspace creator takes for this mode.

    Score needs ``hi_eval_null`` and LRT needs ``logl_H0``. The creator
    rejects an input its mode does not use.
    """
    kwargs: dict[str, np.ndarray | float] = {}
    if LmmTest.SCORE in inv.mode.tests:
        kwargs["hi_eval_null"] = inv.Hi_eval_null
    if LmmTest.LRT in inv.mode.tests:
        kwargs["logl_H0"] = inv.logl_H0
    return kwargs
