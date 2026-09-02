"""Association policy selected once; its chunk plan is narrowed after SNP filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from jamma.core import memory
from jamma.core.eigen_plan import EigenDriverPlan
from jamma.core.memory import estimate_lmm_memory, estimate_streaming_memory
from jamma.core.threading import is_blas_controllable
from jamma.lmm import accel
from jamma.lmm.chunk_sizing import (
    LmmChunkPlan,
    chunk_budget_bytes,
    lmm_extra_bytes_per_snp,
)
from jamma.lmm.dispatch import DispatchPath, select_dispatch_path
from jamma.lmm.schema import LmmMode, parse_lmm_mode

ExecutionMode = Literal["batch", "streaming", "loco"]
RequestedBackend = Literal["auto", "numpy", "numpy-streaming"]

# SNPs per block in the streaming statistics pass when the caller names no
# chunk size. Pass 1 reads the .bed and accumulates per-SNP counts, so its
# footprint is one block of genotypes rather than the rotation and grid
# buffers the association pass carries; it needs no RAM-budgeted sizing of
# its own. Declared here, next to the pricing that assumes it, and imported
# by the executing side (runner_numpy_streaming) so the two cannot drift.
DEFAULT_STATS_CHUNK = 10_000


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Public summary of the selected execution mode."""

    mode: ExecutionMode
    reason: str = field(compare=False)

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("ExecutionPlan.reason must be non-empty")

    @property
    def runner_name(self) -> str:
        """Return the runner label used in logs and telemetry."""
        return f"numpy-{self.mode}"


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """Memory quote for one already-selected association plan.

    Pure: the gating call sites read ``available_gb`` once themselves and
    hand both figures to ``memory.require``.
    """

    total_peak_gb: float
    compute_chunk_size: int
    eigen: EigenDriverPlan | None


@dataclass(frozen=True, slots=True)
class ExecutableAssociationPlan:
    """Selected mode, dispatch, conservative geometry, and memory policy."""

    summary: ExecutionPlan
    dispatch: DispatchPath
    conservative_chunks: LmmChunkPlan
    n_samples: int
    n_snps_before_filter: int
    n_cvt: int
    mem_budget_gb: float | None

    def price(self, *, eigen: EigenDriverPlan | None = None) -> MemoryPlan:
        """Price the conservative geometry without rebuilding policy."""
        chunks = self.conservative_chunks
        # Streaming and LOCO both hold one genotype chunk, never the matrix.
        if self.summary.mode != "batch":
            extra_gb = (
                chunks.chunk_size
                * lmm_extra_bytes_per_snp(
                    self.n_samples,
                    self.n_cvt,
                    self.dispatch,
                    n_buffers=chunks.n_buffers,
                )
                / 1e9
            )
            ledger = estimate_streaming_memory(
                self.n_samples,
                chunk_size=DEFAULT_STATS_CHUNK,
                n_cvt=self.n_cvt,
                pipeline_buffers=chunks.n_buffers,
                compute_chunk_size=chunks.chunk_size,
                eigendecomp_peak_gb=None if eigen is None else eigen.required_gb,
                uab_iab_gb=extra_gb,
            )
            return MemoryPlan(
                total_peak_gb=ledger.peak_gb,
                compute_chunk_size=chunks.chunk_size,
                eigen=eigen,
            )

        return MemoryPlan(
            total_peak_gb=estimate_lmm_memory(
                self.n_samples,
                self.n_snps_before_filter,
                lmm_batch_size=chunks.chunk_size,
                n_cvt=self.n_cvt,
                n_buffers=chunks.n_buffers,
            ),
            compute_chunk_size=chunks.chunk_size,
            eigen=None,
        )


def plan_association(
    n_samples: int,
    n_snps: int,
    *,
    requested: RequestedBackend = "auto",
    n_cvt: int = 1,
    lmm_mode: LmmMode = 1,
    mem_budget: float | None = None,
    max_chunk_size: int | None = None,
    log_dispatch_choices: bool = False,
    loco: bool = False,
) -> ExecutableAssociationPlan:
    """Select all association policy and conservative geometry once.

    Plans unconditionally for whatever backend is requested. Whether a
    user-facing request may ask for numpy-streaming without the C extension
    is the requesting boundary's policy (the pipeline's
    ``_reject_streaming_without_accel``), not the planner's: the streaming
    runner itself works without the extension.

    ``loco=True`` selects the ``loco`` mode regardless of ``requested``: the
    LOCO orchestrator runs the NumPy body per chromosome over disk-read
    chunks, so it is priced like streaming (one chunk plus the
    eigendecomposition), never like batch. ``n_snps`` is then the run total,
    and ``max_chunk_size`` should be the LOCO disk-read chunk width.
    """
    valid_requests = ("auto", "numpy", "numpy-streaming")
    if requested not in valid_requests:
        raise ValueError(
            f"Unknown backend {requested!r}. Must be one of {valid_requests}."
        )

    c_ext_available = accel.available()
    mode = parse_lmm_mode(lmm_mode)
    dispatch = select_dispatch_path(
        n_cvt, mode, accel=c_ext_available, log_choices=log_dispatch_choices
    )
    # The machine is read here, once, so the planner and the pricing stay pure.
    available_gb = memory.available_ram_gb()
    chunks = LmmChunkPlan.plan(
        n_samples,
        n_snps,
        n_cvt,
        dispatch,
        budget_bytes=chunk_budget_bytes(
            mem_budget, available_bytes=int(available_gb * 1e9)
        ),
        blas_controllable=is_blas_controllable(),
        max_chunk_size=max_chunk_size,
    )

    if loco:
        summary = ExecutionPlan("loco", "LOCO per-chromosome NumPy runs")
    elif requested == "numpy-streaming":
        summary = ExecutionPlan("streaming", "Explicit numpy-streaming request")
    elif requested == "numpy":
        summary = ExecutionPlan("batch", "NumPy backend explicitly requested")
    else:
        batch_gb = estimate_lmm_memory(
            n_samples,
            n_snps,
            lmm_batch_size=chunks.chunk_size,
            n_cvt=n_cvt,
            n_buffers=chunks.n_buffers,
        )
        sufficient = memory.fits(batch_gb, available_gb)
        if c_ext_available and sufficient:
            summary = ExecutionPlan(
                "batch",
                f"C extension available, {batch_gb:.1f}GB fits in "
                f"{available_gb:.1f}GB available",
            )
        elif c_ext_available:
            summary = ExecutionPlan(
                "streaming",
                f"C extension available, {batch_gb:.1f}GB exceeds "
                f"{available_gb:.1f}GB available, using NumPy streaming",
            )
        else:
            if not sufficient:
                logger.warning(
                    f"No C extension available. Dataset requires "
                    f"~{batch_gb:.1f}GB but only "
                    f"{available_gb:.1f}GB available. "
                    "Compile the C extension for streaming support."
                )
            summary = ExecutionPlan("batch", "Fallback -- no C extension available")

    return ExecutableAssociationPlan(
        summary=summary,
        dispatch=dispatch,
        conservative_chunks=chunks,
        n_samples=n_samples,
        n_snps_before_filter=n_snps,
        n_cvt=n_cvt,
        mem_budget_gb=mem_budget,
    )
