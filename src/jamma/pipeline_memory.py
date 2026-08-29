"""The memory preflight gate: will this run fit, and if not, say so before it starts.

One ``MemoryPlan`` per run. The plan's compute chunk comes from the same
``plan_lmm_chunks`` the chunk engine calls at allocation time, given the same
dispatch path (see ``_compute_chunk`` for why ``n_snps`` stands in for the
engine's post-filter ``n_filtered``), so the estimate and the allocation
cannot be computed from different formulas (the #74 bug class). The
eigendecomposition phase is priced for the driver that will actually run
(``plan_eigen_driver``), not a blanket worst case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from loguru import logger

from jamma.core import memory
from jamma.core.eigen_plan import (
    EigenDriverPlan,
    forced_numpy_fallback,
    plan_eigen_driver,
)
from jamma.core.memory import estimate_lmm_memory, estimate_streaming_memory
from jamma.lmm.chunk_sizing import (
    LmmChunkPlan,
    lmm_extra_bytes_per_snp,
    plan_lmm_chunks,
)
from jamma.lmm.dispatch import select_dispatch_path
from jamma.lmm.runner_numpy_streaming import _DEFAULT_STATS_CHUNK
from jamma.lmm.schema import parse_lmm_mode

if TYPE_CHECKING:
    from jamma.lmm.runner import ExecutionPlan
    from jamma.pipeline_config import PipelineConfig

__all__ = ["MemoryPlan", "memory_preflight"]


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """What the preflight decided, produced once per run.

    Attributes:
        mode: Execution mode the plan was built for.
        total_peak_gb: Peak requirement across workflow phases.
        available_gb: What the machine reported free when planning.
        sufficient: Whether the peak plus the shared margin fits.
        disk_chunk_size: Streaming statistics-pass block width; None in batch.
        compute_chunk_size: Association-pass chunk width. The engine derives
            the same value from the same sizer at allocation time.
        eigen: Driver-aware eigendecomposition plan; None in batch mode,
            where eigendecompose_kinship gates itself at runtime.
    """

    mode: Literal["batch", "streaming"]
    total_peak_gb: float
    available_gb: float
    sufficient: bool
    disk_chunk_size: int | None
    compute_chunk_size: int | None
    eigen: EigenDriverPlan | None


def _eigen_driver_plan(n_valid: int) -> EigenDriverPlan:
    """Plan the eigendecomposition driver the way the runtime will.

    The kinship matrix does not exist yet, so assume it will be in-place
    eligible (float64, C-contiguous, writeable — the common case).
    """
    has_dsyevd = False
    has_dsyevr = False
    try:
        from jamma import jlinalg  # deferred: jlinalg is heavy

        has_dsyevd = bool(jlinalg.blas_has_dsyevd)
        has_dsyevr = bool(jlinalg.blas_has_dsyevr)
    except ImportError:
        logger.debug(
            "Could not import jlinalg; "
            "preflight will use the conservative DSYEVD estimate."
        )
    return plan_eigen_driver(
        n_valid,
        memory.available_ram_gb(),
        has_dsyevd=has_dsyevd,
        has_dsyevr=has_dsyevr,
        no_vendor=forced_numpy_fallback(),
        inplace_eligible=True,
    )


def _compute_chunk(
    n_valid: int, n_snps: int, n_cvt: int, lmm_mode: int
) -> tuple[LmmChunkPlan, float]:
    """The chunk plan the engine will size to, and the LMM-phase extra it holds.

    Calls the same ``plan_lmm_chunks`` the chunk engine allocates from, with
    ``n_snps`` standing in for the engine's ``n_filtered``: the preflight runs
    before SNP-level MAF/missingness filtering, so it only knows the
    pre-filter count. That is the conservative direction — filtering can only
    shrink the SNP count, and ``compute_chunk_size_numpy`` never returns a
    chunk larger than the SNP count it is given, so this call cannot plan a
    chunk wider than the one the engine will actually allocate.

    Returns:
        (chunk_plan, uab_iab_gb) for this run's dispatch path. The caller
        reads ``chunk_plan.n_buffers`` to price the rotation term at the same
        buffer count the plan (and the engine) actually use, rather than
        assuming pipelining unconditionally.
    """
    from jamma.lmm import accel  # deferred: loads the C extension

    dispatch = select_dispatch_path(
        n_cvt,
        parse_lmm_mode(lmm_mode),
        accel=accel.available(),
        log_choices=False,
    )
    plan = plan_lmm_chunks(n_valid, n_snps, n_cvt, dispatch)
    extra_gb = (
        plan.chunk_size
        * lmm_extra_bytes_per_snp(n_valid, n_cvt, dispatch, n_buffers=plan.n_buffers)
        / 1e9
    )
    return plan, extra_gb


def plan_memory(
    config: PipelineConfig,
    plan: ExecutionPlan,
    n_valid: int,
    n_snps: int,
    n_cvt: int,
) -> MemoryPlan:
    """Build the run's MemoryPlan for the chosen execution mode."""
    chunk_plan, uab_iab_gb = _compute_chunk(n_valid, n_snps, n_cvt, config.lmm_mode)
    compute_chunk = chunk_plan.chunk_size

    if plan.mode == "streaming":
        eigen = _eigen_driver_plan(n_valid)
        est = estimate_streaming_memory(
            n_valid,
            chunk_size=_DEFAULT_STATS_CHUNK,
            n_cvt=n_cvt,
            pipeline_buffers=chunk_plan.n_buffers,
            compute_chunk_size=compute_chunk,
            eigendecomp_peak_gb=eigen.required_gb,
            uab_iab_gb=uab_iab_gb,
        )
        return MemoryPlan(
            mode="streaming",
            total_peak_gb=est.total_peak_gb,
            available_gb=est.available_gb,
            sufficient=est.sufficient,
            disk_chunk_size=_DEFAULT_STATS_CHUNK,
            compute_chunk_size=compute_chunk,
            eigen=eigen,
        )

    est = estimate_lmm_memory(
        n_valid,
        n_snps,
        lmm_batch_size=compute_chunk,
        n_cvt=n_cvt,
        n_buffers=chunk_plan.n_buffers,
    )
    return MemoryPlan(
        mode="batch",
        total_peak_gb=est.total_gb,
        available_gb=est.available_gb,
        sufficient=est.sufficient,
        disk_chunk_size=None,
        compute_chunk_size=compute_chunk,
        eigen=None,
    )


def memory_preflight(
    config: PipelineConfig,
    plan: ExecutionPlan,
    n_valid: int,
    n_snps: int,
    n_cvt: int,
) -> MemoryPlan | None:
    """Run the memory preflight gate for the chosen execution plan.

    Builds one MemoryPlan, logs it, and rejects the run when it exceeds the
    user budget or what the machine has free.

    Args:
        config: Pipeline configuration. Reads ``check_memory``, ``mem_budget``
            and ``lmm_mode``.
        plan: Resolved ExecutionPlan (mode determines the estimator).
        n_valid: Sample count after valid-mask intersection.
        n_snps: Total SNPs from PLINK metadata (pre-MAF/missingness).
        n_cvt: Covariate count including the intercept.

    Returns:
        The MemoryPlan when check_memory is on, None when skipped.

    Raises:
        MemoryError: If estimated memory exceeds budget or available memory.
    """
    if not config.check_memory:
        logger.info(
            f"Memory preflight skipped ({plan.runner_name}): check_memory=False"
        )
        return None

    mem_plan = plan_memory(config, plan, n_valid, n_snps, n_cvt)
    driver_note = f", eigen driver {mem_plan.eigen.driver}" if mem_plan.eigen else ""
    logger.info(
        f"Memory estimate ({plan.runner_name}): "
        f"{mem_plan.total_peak_gb:.1f}GB required, "
        f"{mem_plan.available_gb:.1f}GB available"
        f" (compute chunk {mem_plan.compute_chunk_size}{driver_note})"
    )
    memory.require(
        mem_plan.total_peak_gb,
        mem_plan.available_gb,
        plan.runner_name,
        budget_gb=config.mem_budget,
    )
    return mem_plan
