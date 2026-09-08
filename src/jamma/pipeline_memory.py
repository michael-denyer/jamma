"""Memory preflight for an already-selected association plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from jamma.core import memory
from jamma.core.eigen_plan import EigenDriverPlan, square_matrix_gb
from jamma.lmm.association_plan import DEFAULT_STATS_CHUNK, ExecutableAssociationPlan
from jamma.lmm.eigen import plan_eigen_driver_for_machine

if TYPE_CHECKING:
    from jamma.pipeline_config import PipelineConfig

__all__ = ["memory_preflight"]


def memory_preflight(
    config: PipelineConfig,
    plan: ExecutableAssociationPlan,
) -> EigenDriverPlan | None:
    """Price and gate one plan without rebuilding association policy.

    Logs the quote and raises through ``memory.require`` when it does not
    fit. Return the selected eigen driver for acquisition to execute.
    """
    summary = plan.summary
    if not config.check_memory:
        logger.info(
            f"Memory preflight skipped ({summary.runner_name}): check_memory=False"
        )
        return

    available_gb = memory.available_ram_gb()
    eigen = (
        plan_eigen_driver_for_machine(
            plan.n_samples,
            available_gb,
            budget_gb=plan.mem_budget_gb,
            inplace_eligible=True,
        )
        if config.eigenvalue_file is None
        else None
    )
    quote = plan.price(eigen=eigen)
    required_gb = max(quote.total_peak_gb, eigen.required_gb if eigen else 0.0)
    if eigen is not None:
        n_kinship = (
            plan.n_input_samples
            if config.save_kinship or config.kinship_file is not None
            else plan.n_samples
        )
        if config.kinship_file is None:
            kinship_gb = memory.estimate_streaming_memory(
                n_kinship,
                chunk_size=DEFAULT_STATS_CHUNK,
            ).kinship_gb
            kinship_gb += (
                max(0, plan.n_input_samples - n_kinship) * DEFAULT_STATS_CHUNK * 8 / 1e9
            )
        else:
            kinship_gb = square_matrix_gb(n_kinship)
        if n_kinship != plan.n_samples:
            kinship_gb = max(
                kinship_gb,
                square_matrix_gb(n_kinship) + square_matrix_gb(plan.n_samples),
            )
        required_gb = max(required_gb, kinship_gb)
    driver_note = f", eigen driver {eigen.driver}" if eigen else ""
    logger.info(
        f"Memory estimate ({summary.runner_name}): "
        f"{required_gb:.1f}GB required, "
        f"{available_gb:.1f}GB available"
        f" (pre-filter compute chunk {quote.compute_chunk_size}{driver_note})"
    )
    memory.require(
        required_gb,
        available_gb,
        summary.runner_name,
        budget_gb=plan.mem_budget_gb,
    )

    return eigen
