"""The memory preflight gate: will this run fit, and if not, say so before it starts.

Split out of ``pipeline.py``, where the two halves sat 300 lines apart. They are
one question with two estimators, chosen by execution mode: streaming accounts
for chunked reads plus the compute buffers, batch accounts for the whole genotype
matrix in memory. Both honour ``check_memory`` and ``mem_budget`` the same way,
which is easier to confirm with them side by side.

Module functions taking the config rather than ``PipelineRunner`` methods,
because the config was the only instance state they read. Same shape as
``pipeline_kinship.compute_kinship``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from jamma.core.chunk import _compute_chunk_size
from jamma.core.memory import (
    StreamingMemoryBreakdown,
    estimate_lmm_memory,
    estimate_streaming_memory,
)

if TYPE_CHECKING:
    from jamma.lmm.runner import ExecutionPlan
    from jamma.pipeline_config import PipelineConfig

__all__ = ["check_streaming_memory", "memory_preflight"]


def _reject_if_over_budget(
    required_gb: float,
    available_gb: float,
    mem_budget: float | None,
    *,
    sufficient: bool,
    margin_note: str = "",
) -> None:
    """Raise MemoryError when an estimate exceeds the budget or what is available.

    Both estimators apply the same two rules in the same order, so they share
    this rather than each spelling out both messages.

    Args:
        required_gb: Estimated peak requirement.
        available_gb: What the system reports free.
        mem_budget: User-set ceiling in GB, or None for no ceiling.
        sufficient: The estimator's own verdict on available memory.
        margin_note: Appended to the requirement in the insufficient-memory
            message, e.g. the streaming estimator's safety margin.

    Raises:
        MemoryError: Naming which of the two rules failed, and how to override.
    """
    if mem_budget is not None and required_gb > mem_budget:
        raise MemoryError(
            f"Estimated memory ({required_gb:.1f}GB) exceeds "
            f"budget ({mem_budget}GB). "
            f"Use --no-check-memory to override."
        )
    if not sufficient:
        raise MemoryError(
            f"Insufficient memory: need {required_gb:.1f}GB{margin_note}, "
            f"have {available_gb:.1f}GB. "
            f"Use --no-check-memory to override."
        )


def check_streaming_memory(
    config: PipelineConfig, n_samples: int, n_snps: int, n_cvt: int = 1
) -> StreamingMemoryBreakdown | None:
    """Check memory requirements for the streaming path.

    Computes the actual chunk sizes via ``_compute_chunk_size``, then estimates
    streaming memory. Checks against ``mem_budget`` if set, and against
    available system memory.

    Args:
        config: Pipeline configuration. Reads ``check_memory`` and ``mem_budget``.
        n_samples: Number of valid samples (after phenotype/covariate filtering).
        n_snps: Number of SNPs in the dataset.
        n_cvt: Number of covariates (affects Uab array sizing).

    Returns:
        StreamingMemoryBreakdown if check_memory is True, None otherwise.

    Raises:
        MemoryError: If estimated memory exceeds budget or available memory.
    """
    if not config.check_memory:
        logger.info("Memory preflight skipped (streaming): check_memory=False")
        return None

    disk_chunk = _compute_chunk_size(n_snps)
    compute_chunk = _compute_chunk_size(
        n_snps, n_samples=n_samples, n_cvt=n_cvt, pipeline_buffers=2
    )
    est = estimate_streaming_memory(
        n_samples,
        chunk_size=disk_chunk,
        n_cvt=n_cvt,
        compute_chunk_size=compute_chunk,
    )

    logger.info(
        f"Memory estimate: {est.total_peak_gb:.1f}GB required, "
        f"{est.available_gb:.1f}GB available"
    )
    _reject_if_over_budget(
        est.total_peak_gb,
        est.available_gb,
        config.mem_budget,
        sufficient=est.sufficient,
        margin_note=" (with 10% margin)",
    )
    return est


def memory_preflight(
    config: PipelineConfig,
    plan: ExecutionPlan,
    n_valid: int,
    n_snps: int,
    n_cvt: int,
) -> None:
    """Run the memory preflight gate for the chosen execution plan.

    Streaming mode uses ``check_streaming_memory``, which accounts for chunked
    reads and the compute buffers. Batch mode uses the in-memory estimator.
    Both raise ``MemoryError`` with actionable messages on failure.

    Args:
        config: Pipeline configuration. Reads ``check_memory`` and ``mem_budget``.
        plan: Resolved ExecutionPlan (mode determines which estimator).
        n_valid: Sample count after valid-mask intersection.
        n_snps: Total SNPs from PLINK metadata (pre-MAF/missingness).
        n_cvt: Covariate count including the intercept.

    Raises:
        MemoryError: If estimated memory exceeds budget or available memory.
    """
    if plan.mode == "streaming":
        check_streaming_memory(config, n_valid, n_snps, n_cvt=n_cvt)
        return

    if not config.check_memory:
        logger.info(
            f"Memory preflight skipped ({plan.runner_name}): check_memory=False"
        )
        return

    est = estimate_lmm_memory(n_valid, n_snps, n_cvt=n_cvt)
    logger.info(
        f"Memory estimate ({plan.runner_name}): "
        f"{est.total_gb:.1f}GB required, "
        f"{est.available_gb:.1f}GB available"
    )
    _reject_if_over_budget(
        est.total_gb,
        est.available_gb,
        config.mem_budget,
        sufficient=est.sufficient,
    )
