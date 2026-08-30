"""LMM execution mode selection.

Picks batch vs streaming for a given problem size; callers dispatch on the
result themselves.

Components:
- ExecutionPlan: Public two-field summary of the selected mode.
- select_execution_mode(): Compatibility selector returning that summary.

PipelineRunner keeps the executable plan returned by ``plan_association`` and
dispatches via its own batch and streaming adapters. Public callers that only
need the selected mode use ``select_execution_mode``.

This module used to also export run_lmm(), a second dispatcher for
programmatic callers with pre-loaded data. It was removed in 7.0.0: it had no
callers in src/, scripts/ or any known downstream consumer, and its routing
duplicated PipelineRunner's.
"""

from __future__ import annotations

from typing import Literal

from loguru import logger

from jamma.lmm.association_plan import ExecutionPlan, plan_association
from jamma.lmm.schema import LmmMode

SMALL_SAMPLE_WARNING_THRESHOLD = 50


def warn_if_small_sample(n_samples: int) -> None:
    """Warn once when sample size is below the practical LMM threshold.

    JAMMA is designed for large-scale GWAS (thousands to hundreds of thousands
    of samples). Below ~50 samples, two concerns apply:

    1. LMM has insufficient statistical power regardless of optimizer — kinship
       estimation and variance component inference are unreliable with so few
       samples.
    2. JAMMA's batch-vectorized grid+golden-section lambda optimizer assumes
       the log-likelihood is unimodal in log-lambda space. Very small samples
       are one of the scenarios where that assumption can fail, and unlike
       GEMMA's Brent's method JAMMA has no mechanism to detect multimodality.
       Results may diverge meaningfully from GEMMA on such adversarial inputs.

    See docs/GEMMA_DIVERGENCES.md §6 for full context.

    Args:
        n_samples: Number of samples actually entering the LMM (post
            phenotype/covariate filtering, not the raw PLINK header count).
    """
    if n_samples < SMALL_SAMPLE_WARNING_THRESHOLD:
        logger.warning(
            f"Small sample size ({n_samples} < {SMALL_SAMPLE_WARNING_THRESHOLD}): "
            "LMM-based GWAS has insufficient statistical power at this scale, "
            "and JAMMA's batch golden-section lambda optimizer may diverge from "
            "GEMMA's Brent's method on multimodal likelihoods. "
            "See docs/GEMMA_DIVERGENCES.md §6."
        )


def select_execution_mode(
    n_samples: int,
    n_snps: int,
    *,
    requested: Literal["auto", "numpy", "numpy-streaming"] = "auto",
    n_cvt: int = 1,
    lmm_mode: LmmMode = 1,
    mem_budget: float | None = None,
) -> ExecutionPlan:
    """Select the optimal execution backend and mode.

    Central authority for backend and mode selection. Returns a structured
    ExecutionPlan rather than a bare string.

    Selection priority (when requested="auto"):
    1. C extension available + memory sufficient + (n_cvt=1 or C general available)
       -> numpy-batch
    2. C extension available + memory insufficient + C handles n_cvt
       -> numpy-streaming
    3. Fallback -> numpy-batch

    When requested is "numpy", batch mode is always used.
    Compound value "numpy-streaming" forces the exact backend and mode
    combination.

    Args:
        n_samples: Number of samples in the dataset.
        n_snps: Number of SNPs in the dataset.
        requested: Requested backend ("auto", "numpy", or
            "numpy-streaming").
        n_cvt: Number of covariates (including intercept). Used to select
            accurate memory estimates (Uab is larger with more covariates)
            and to guard against numpy-batch when the C general extension is
            unavailable for n_cvt > 1.
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All). Selects the
            dispatch path the real chunk plan is sized against, so the
            estimate here matches the chunk the engine will actually
            allocate instead of ``estimate_lmm_memory``'s 20,000-row
            default.
        mem_budget: User-set ceiling in GB, or None. Feeds the same chunk
            sizer the engine allocates from, so a tight budget narrows the
            chunk this estimate is priced against rather than only vetoing
            the run afterward.

    Returns:
        ExecutionPlan with backend, mode, and reason.
    """
    return plan_association(
        n_samples,
        n_snps,
        requested=requested,
        n_cvt=n_cvt,
        lmm_mode=lmm_mode,
        mem_budget=mem_budget,
    ).summary
