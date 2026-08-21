"""LMM execution mode selection.

Picks batch vs streaming for a given problem size; callers dispatch on the
result themselves.

Components:
- ExecutionPlan: Frozen dataclass describing the selected backend and mode.
- select_execution_mode(): Picks (backend, mode) based on hardware, memory,
  and user preferences.

PipelineRunner calls select_execution_mode() and then dispatches via its own
_run_batch/_run_streaming methods, which handle PLINK loading, incremental
writing, and timing at a higher abstraction level.

This module used to also export run_lmm(), a second dispatcher for
programmatic callers with pre-loaded data. It was removed in 7.0.0: it had no
callers in src/, scripts/ or any known downstream consumer, and its routing
duplicated PipelineRunner's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger

from jamma.core.memory import estimate_lmm_memory


def _c_extension_available() -> bool:
    """Report whether the loaded C extension is usable.

    Deferred import because compute_numpy imports this module transitively.
    """
    from jamma.lmm import compute_numpy  # deferred: circular dep

    return compute_numpy._C_ACCEL_AVAILABLE


@dataclass(frozen=True, slots=True, eq=False)
class ExecutionPlan:
    """Describes the selected backend and execution mode.

    Equality compares (backend, mode) only — reason is diagnostic metadata
    and must not affect dispatch or comparison logic.

    Attributes:
        backend: Compute backend ("numpy").
        mode: Execution mode ("batch" or "streaming").
        reason: Human-readable explanation of why this plan was chosen.
    """

    backend: Literal["numpy"]
    mode: Literal["batch", "streaming"]
    reason: str

    def __post_init__(self) -> None:
        if self.mode not in ("batch", "streaming"):
            raise ValueError(
                f"Invalid execution mode: {self.mode!r}. "
                f"Must be 'batch' or 'streaming'."
            )
        if not self.reason:
            raise ValueError("ExecutionPlan.reason must be non-empty")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecutionPlan):
            return NotImplemented
        return self.backend == other.backend and self.mode == other.mode

    def __hash__(self) -> int:
        return hash((self.backend, self.mode))

    @property
    def runner_name(self) -> str:
        """Return '{backend}-{mode}' for logging and banner display."""
        return f"{self.backend}-{self.mode}"


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

    Returns:
        ExecutionPlan with backend, mode, and reason.
    """
    # Handle compound backend requests from CLI (e.g., "numpy-streaming")
    if requested == "numpy-streaming":
        if not _c_extension_available():
            raise ValueError(
                "Backend 'numpy-streaming' requires the C extension but it is "
                "not available. Compile it with: uv run python -c "
                "'from jamma.jlinalg._compile_jlinalg import compile_extension; "
                "compile_extension()'"
            )
        return ExecutionPlan("numpy", "streaming", "Explicit numpy-streaming request")

    _valid_requests = ("auto", "numpy", "numpy-streaming")
    if requested not in _valid_requests:
        raise ValueError(
            f"Unknown backend {requested!r}. Must be one of {_valid_requests}."
        )

    # Explicit backend selection
    if requested == "numpy":
        return ExecutionPlan(
            "numpy",
            "batch",
            "NumPy backend explicitly requested",
        )

    # Auto selection
    c_ext_available = _c_extension_available()
    est = estimate_lmm_memory(n_samples, n_snps, n_cvt=n_cvt)

    # Check if C extension handles the covariate count
    if c_ext_available:
        from jamma.lmm.compute_numpy import (
            _C_GENERAL_AVAILABLE,  # deferred: circular dep
        )

        c_handles_n_cvt = n_cvt <= 1 or _C_GENERAL_AVAILABLE
        if c_handles_n_cvt:
            if est.sufficient:
                return ExecutionPlan(
                    "numpy",
                    "batch",
                    f"C extension available, {est.total_gb:.1f}GB fits in "
                    f"{est.available_gb:.1f}GB available",
                )
            return ExecutionPlan(
                "numpy",
                "streaming",
                f"C extension available, {est.total_gb:.1f}GB exceeds "
                f"{est.available_gb:.1f}GB available, using NumPy streaming",
            )

    # Fallback: pure NumPy batch (no C extension)
    if not est.sufficient:
        logger.warning(
            f"No C extension available. Dataset requires ~{est.total_gb:.1f}GB "
            f"but only {est.available_gb:.1f}GB available. "
            "Compile the C extension for streaming support."
        )
    return ExecutionPlan(
        "numpy",
        "batch",
        "Fallback -- no C extension available",
    )
