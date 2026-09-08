"""Memory preflight for an already-selected association plan."""

from __future__ import annotations

from loguru import logger

from jamma.core import memory
from jamma.core.eigen_plan import EigenDriverPlan
from jamma.lmm.eigen import plan_eigen_driver_for_machine
from jamma.pipeline_plan import AnalysisPlan, ProvidedEigen, StandardAnalysisPlan

__all__ = ["memory_preflight"]


def _eigen_driver(
    analysis: AnalysisPlan, available_gb: float
) -> EigenDriverPlan | None:
    """Plan the decomposition the run will execute; None when eigenpairs are read."""
    if isinstance(analysis, StandardAnalysisPlan) and isinstance(
        analysis.eigen_source, ProvidedEigen
    ):
        return None
    execution = analysis.execution
    return plan_eigen_driver_for_machine(
        execution.n_samples,
        available_gb,
        budget_gb=execution.mem_budget_gb,
        inplace_eligible=True,
    )


def memory_preflight(
    analysis: AnalysisPlan, *, check_memory: bool
) -> EigenDriverPlan | None:
    """Price and gate one plan; return the eigen driver acquisition executes."""
    execution = analysis.execution
    runner_name = execution.summary.runner_name
    if not check_memory:
        logger.info(f"Memory preflight skipped ({runner_name}): check_memory=False")
        return None
    available_gb = memory.available_ram_gb()
    eigen = _eigen_driver(analysis, available_gb)
    quote = execution.price(eigen=eigen)
    driver_note = f", eigen driver {eigen.driver}" if eigen else ""
    logger.info(
        f"Memory estimate ({runner_name}): {quote.total_peak_gb:.1f}GB required, "
        f"{available_gb:.1f}GB available"
        f" (pre-filter compute chunk {quote.compute_chunk_size}{driver_note})"
    )
    budget_gb = execution.mem_budget_gb
    memory.require(quote.total_peak_gb, available_gb, runner_name, budget_gb=budget_gb)
    return eigen
