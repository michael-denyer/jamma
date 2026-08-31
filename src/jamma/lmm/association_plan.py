"""Association policy selected once and tightened only after SNP filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from jamma.core.eigen_plan import EigenDriverPlan
from jamma.core.memory import estimate_lmm_memory, estimate_streaming_memory
from jamma.lmm import accel
from jamma.lmm.chunk_sizing import (
    LmmChunkPlan,
    lmm_extra_bytes_per_snp,
    plan_lmm_chunks,
    tighten_lmm_chunks,
)
from jamma.lmm.dispatch import DispatchPath, select_dispatch_path
from jamma.lmm.schema import LmmMode, parse_lmm_mode

ExecutionMode = Literal["batch", "streaming"]
RequestedBackend = Literal["auto", "numpy", "numpy-streaming"]

_DEFAULT_STATS_CHUNK = 10_000


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
    """Memory quote for one already-selected association plan."""

    mode: ExecutionMode
    total_peak_gb: float
    available_gb: float
    sufficient: bool
    disk_chunk_size: int | None
    compute_chunk_size: int
    eigen: EigenDriverPlan | None


@dataclass(frozen=True, slots=True)
class AssociationExecution:
    """Final post-filter inputs needed by the chunk engine."""

    mode: ExecutionMode
    dispatch: DispatchPath
    chunks: LmmChunkPlan


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
        if self.summary.mode == "streaming":
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
            estimate = estimate_streaming_memory(
                self.n_samples,
                chunk_size=_DEFAULT_STATS_CHUNK,
                n_cvt=self.n_cvt,
                pipeline_buffers=chunks.n_buffers,
                compute_chunk_size=chunks.chunk_size,
                eigendecomp_peak_gb=None if eigen is None else eigen.required_gb,
                uab_iab_gb=extra_gb,
            )
            return MemoryPlan(
                mode="streaming",
                total_peak_gb=estimate.total_peak_gb,
                available_gb=estimate.available_gb,
                sufficient=estimate.sufficient,
                disk_chunk_size=_DEFAULT_STATS_CHUNK,
                compute_chunk_size=chunks.chunk_size,
                eigen=eigen,
            )

        estimate = estimate_lmm_memory(
            self.n_samples,
            self.n_snps_before_filter,
            lmm_batch_size=chunks.chunk_size,
            n_cvt=self.n_cvt,
            n_buffers=chunks.n_buffers,
        )
        return MemoryPlan(
            mode="batch",
            total_peak_gb=estimate.total_gb,
            available_gb=estimate.available_gb,
            sufficient=estimate.sufficient,
            disk_chunk_size=None,
            compute_chunk_size=chunks.chunk_size,
            eigen=None,
        )

    def tighten_after_filter(self, n_filtered: int) -> AssociationExecution:
        """Narrow geometry once without re-reading RAM or changing policy."""
        return AssociationExecution(
            mode=self.summary.mode,
            dispatch=self.dispatch,
            chunks=tighten_lmm_chunks(self.conservative_chunks, n_filtered),
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
    _require_streaming_accel: bool = True,
) -> ExecutableAssociationPlan:
    """Select all association policy and conservative geometry once."""
    valid_requests = ("auto", "numpy", "numpy-streaming")
    if requested not in valid_requests:
        raise ValueError(
            f"Unknown backend {requested!r}. Must be one of {valid_requests}."
        )

    c_ext_available = accel.available()
    if (
        requested == "numpy-streaming"
        and _require_streaming_accel
        and not c_ext_available
    ):
        raise ValueError(
            "Backend 'numpy-streaming' requires the C extension but it is "
            "not available. Compile it with: uv run python -c "
            "'from jamma.jlinalg._compile_jlinalg import compile_extension; "
            "compile_extension()'"
        )

    mode = parse_lmm_mode(lmm_mode)
    dispatch = select_dispatch_path(
        n_cvt, mode, accel=c_ext_available, log_choices=log_dispatch_choices
    )
    chunks = plan_lmm_chunks(
        n_samples,
        n_snps,
        n_cvt,
        dispatch,
        max_chunk_size=max_chunk_size,
        mem_budget_bytes=None if mem_budget is None else int(mem_budget * 1e9),
    )

    if requested == "numpy-streaming":
        summary = ExecutionPlan("streaming", "Explicit numpy-streaming request")
    elif requested == "numpy":
        summary = ExecutionPlan("batch", "NumPy backend explicitly requested")
    else:
        estimate = estimate_lmm_memory(
            n_samples,
            n_snps,
            lmm_batch_size=chunks.chunk_size,
            n_cvt=n_cvt,
            n_buffers=chunks.n_buffers,
        )
        if c_ext_available and estimate.sufficient:
            summary = ExecutionPlan(
                "batch",
                f"C extension available, {estimate.total_gb:.1f}GB fits in "
                f"{estimate.available_gb:.1f}GB available",
            )
        elif c_ext_available:
            summary = ExecutionPlan(
                "streaming",
                f"C extension available, {estimate.total_gb:.1f}GB exceeds "
                f"{estimate.available_gb:.1f}GB available, using NumPy streaming",
            )
        else:
            if not estimate.sufficient:
                logger.warning(
                    f"No C extension available. Dataset requires "
                    f"~{estimate.total_gb:.1f}GB but only "
                    f"{estimate.available_gb:.1f}GB available. "
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
