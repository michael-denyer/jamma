"""LMM execution mode selection and dispatch.

Selects batch vs streaming mode via select_execution_mode() and provides
run_lmm() as a single entry point that routes to the correct runner.

Components:
- ExecutionPlan: Frozen dataclass describing the selected backend and mode.
- select_execution_mode(): Picks (backend, mode) based on hardware, memory,
  and user preferences.
- run_lmm(): Dispatches to the appropriate runner function, normalizing
  return types to (LmmRunResult, int).

The pipeline (pipeline.py) uses select_execution_mode() for plan selection but
dispatches via its own _run_batch/_run_streaming
methods, which handle PLINK loading, incremental writing, and timing at a
higher abstraction level.
run_lmm() is the public API for programmatic callers with pre-loaded data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from jamma.core.memory import estimate_lmm_memory
from jamma.lmm._compile_utils import is_c_extension_usable
from jamma.lmm.schema import (
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    DEFAULT_MAF,
    DEFAULT_MISS,
    DEFAULT_N_GRID,
    DEFAULT_N_REFINE,
    LmmConfig,
    LmmRunResult,
)


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
        if not is_c_extension_usable():
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
    c_ext_available = is_c_extension_usable()
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


def run_lmm(
    *,
    execution_plan: ExecutionPlan | None = None,
    genotypes: np.ndarray | None = None,
    phenotypes: np.ndarray | None = None,
    kinship: np.ndarray | None = None,
    snp_info: list | None = None,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    bed_path: Path | None = None,
    output_path: Path | None = None,
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    chunk_size: int = 10_000,
    validate_genotypes: bool = True,
    config: LmmConfig | None = None,
    # Flat config overrides (used when config is None):
    maf_threshold: float = DEFAULT_MAF,
    miss_threshold: float = DEFAULT_MISS,
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    n_grid: int = DEFAULT_N_GRID,
    n_refine: int = DEFAULT_N_REFINE,
    check_memory: bool = True,
    show_progress: bool = True,
    lmm_mode: int = 1,
) -> tuple[LmmRunResult, int]:
    """Unified LMM entry point that dispatches to the correct runner.

    Routes to run_lmm_association_numpy or run_lmm_association_numpy_streaming
    based on the ExecutionPlan. If no plan is provided, auto-selects via
    select_execution_mode().

    Args:
        execution_plan: Pre-computed execution plan, or None for auto-selection.
        genotypes: Genotype matrix (required for batch modes).
        phenotypes: Phenotype vector (required for all modes).
        kinship: Kinship matrix.
        snp_info: List of SNP metadata dicts (required for batch modes).
        covariates: Covariate matrix or None.
        eigenvalues: Pre-computed eigenvalues or None.
        eigenvectors: Pre-computed eigenvectors or None.
        bed_path: PLINK file prefix (required for streaming mode).
        output_path: Path for incremental result writing.
        snps_indices: Pre-resolved column indices for -snps restriction.
        hwe_threshold: HWE p-value threshold (streaming only).
        chunk_size: SNPs per disk chunk (streaming only).
        validate_genotypes: Whether to validate genotypes (streaming only).
        config: LmmConfig instance, or None to use flat kwargs.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (clamped to min 20 internally).
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.

    Returns:
        Tuple of (LmmRunResult, n_tested) regardless of which runner is used.

    Raises:
        ValueError: If required data source is missing for the selected mode.
    """
    # Auto-select if no plan provided
    if execution_plan is None:
        if genotypes is None and phenotypes is None:
            raise ValueError(
                "Auto-selection requires at least genotypes or phenotypes to "
                "determine dataset dimensions. Provide data or an explicit "
                "ExecutionPlan."
            )
        if genotypes is None and bed_path is not None:
            raise ValueError(
                "Auto-selection with bed_path but no genotypes is ambiguous. "
                "Provide an explicit ExecutionPlan (e.g., "
                "ExecutionPlan('numpy', 'streaming', ...)) when using bed_path."
            )
        n_samples = (
            genotypes.shape[0]
            if genotypes is not None
            else (phenotypes.shape[0] if phenotypes is not None else 0)
        )
        n_snps = genotypes.shape[1] if genotypes is not None else 0
        n_cvt = covariates.shape[1] if covariates is not None else 1
        execution_plan = select_execution_mode(
            n_samples,
            n_snps,
            requested="auto",
            n_cvt=n_cvt,
        )

    logger.info(
        f"Execution plan: {execution_plan.runner_name} ({execution_plan.reason})"
    )

    # Warn on small sample sizes (n<50). The pipeline already emits this
    # warning using post-filter n_valid; do it here too for programmatic
    # callers who bypass the pipeline. Use phenotypes/genotypes shape as the
    # best available estimate of actual sample count.
    _n_for_warning: int | None = None
    if phenotypes is not None:
        _n_for_warning = phenotypes.shape[0]
    elif genotypes is not None:
        _n_for_warning = genotypes.shape[0]
    if _n_for_warning is not None:
        warn_if_small_sample(_n_for_warning)

    # Build common kwargs from config or flat args
    if config is not None:
        common_kwargs = config.as_kwargs()
    else:
        common_kwargs = {
            "maf_threshold": maf_threshold,
            "miss_threshold": miss_threshold,
            "l_min": l_min,
            "l_max": l_max,
            "n_grid": n_grid,
            "n_refine": n_refine,
            "check_memory": check_memory,
            "show_progress": show_progress,
            "lmm_mode": lmm_mode,
        }

    # Dispatch based on execution mode
    if execution_plan.mode == "batch":
        if genotypes is None:
            raise ValueError(
                "numpy-batch mode requires genotypes array, but genotypes=None"
            )
        from jamma.lmm.runner_numpy import run_lmm_association_numpy

        result = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=config,
            output_path=output_path,
            **common_kwargs if config is None else {},
        )
        return result, result.snp_count

    if execution_plan.mode == "streaming":
        if bed_path is None:
            raise ValueError(
                "NumPy streaming mode requires bed_path (PLINK .bed file path). "
                "Provide bed_path or use an in-memory backend."
            )
        from jamma.lmm.runner_numpy_streaming import (
            run_lmm_association_numpy_streaming,
        )

        streaming_kwargs = common_kwargs if config is None else {}
        result, n_tested = run_lmm_association_numpy_streaming(
            bed_path=bed_path,
            phenotypes=phenotypes,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            output_path=output_path,
            snps_indices=snps_indices,
            hwe_threshold=hwe_threshold,
            chunk_size=chunk_size,
            validate_genotypes=validate_genotypes,
            config=config,
            **streaming_kwargs,
        )
        return result, n_tested

    raise ValueError(f"Unsupported execution mode: {execution_plan.mode!r}")
