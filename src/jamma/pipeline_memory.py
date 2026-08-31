"""Memory preflight for an already-selected association plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from jamma.core import memory
from jamma.core.eigen_plan import (
    EigenDriverPlan,
    forced_numpy_fallback,
    plan_eigen_driver,
)
from jamma.lmm.association_plan import ExecutableAssociationPlan

if TYPE_CHECKING:
    from jamma.pipeline_config import PipelineConfig

__all__ = ["memory_preflight"]


def _eigen_driver_plan(n_valid: int) -> EigenDriverPlan:
    """Plan the eigendecomposition driver the runtime will use."""
    has_dsyevd = False
    has_dsyevr = False
    try:
        from jamma import jlinalg

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


def memory_preflight(
    config: PipelineConfig,
    plan: ExecutableAssociationPlan,
) -> None:
    """Price and gate one plan without rebuilding association policy.

    Logs the quote and raises through ``memory.require`` when it does not
    fit; callers needing the numbers price the plan themselves.
    """
    summary = plan.summary
    if not config.check_memory:
        logger.info(
            f"Memory preflight skipped ({summary.runner_name}): check_memory=False"
        )
        return

    eigen = _eigen_driver_plan(plan.n_samples) if summary.mode == "streaming" else None
    quote = plan.price(eigen=eigen)
    driver_note = f", eigen driver {quote.eigen.driver}" if quote.eigen else ""
    logger.info(
        f"Memory estimate ({summary.runner_name}): "
        f"{quote.total_peak_gb:.1f}GB required, "
        f"{quote.available_gb:.1f}GB available"
        f" (compute chunk {quote.compute_chunk_size}{driver_note})"
    )
    memory.require(
        quote.total_peak_gb,
        quote.available_gb,
        summary.runner_name,
        budget_gb=plan.mem_budget_gb,
    )
