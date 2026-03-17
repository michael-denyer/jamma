"""Unified backend selection and LMM dispatch layer.

Consolidates backend+mode selection into select_execution_mode() and provides
run_lmm() as a single entry point that routes to the correct runner.

Components:
- ExecutionPlan: Frozen dataclass describing the selected backend and mode.
- select_execution_mode(): Picks (backend, mode) based on hardware, memory,
  and user preferences.
- run_lmm(): Dispatches to the appropriate runner function, normalizing
  return types to (LmmRunResult, int).

The pipeline (pipeline.py) uses select_execution_mode() for plan selection but
dispatches via its own _run_jax_backend/_run_numpy_backend methods, which handle
PLINK loading, incremental writing, and timing at a higher abstraction level.
run_lmm() is the public API for programmatic callers with pre-loaded data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from jamma.core.backend import BackendRequest, has_jax
from jamma.core.memory import estimate_lmm_memory
from jamma.lmm._compile_utils import is_c_extension_usable
from jamma.lmm.schema import LmmConfig, LmmRunResult

_VALID_PLANS = frozenset({("jax", "batch"), ("jax", "streaming"), ("numpy", "batch")})


@dataclass(frozen=True, slots=True, eq=False)
class ExecutionPlan:
    """Describes the selected backend and execution mode.

    Equality compares (backend, mode) only — reason is diagnostic metadata
    and must not affect dispatch or comparison logic.

    Attributes:
        backend: Compute backend ("jax" or "numpy").
        mode: Execution mode ("batch" or "streaming").
        reason: Human-readable explanation of why this plan was chosen.
    """

    backend: Literal["jax", "numpy"]
    mode: Literal["batch", "streaming"]
    reason: str

    def __post_init__(self) -> None:
        if (self.backend, self.mode) not in _VALID_PLANS:
            raise ValueError(
                f"Invalid execution plan: {self.backend}-{self.mode}. "
                f"Valid plans: {', '.join(f'{b}-{m}' for b, m in sorted(_VALID_PLANS))}"
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


def select_execution_mode(
    n_samples: int,
    n_snps: int,
    *,
    requested: BackendRequest = "auto",
    n_cvt: int = 1,
    lmm_mode: int = 1,
) -> ExecutionPlan:
    """Select the optimal execution backend and mode.

    Central authority for backend and mode selection. Returns a structured
    ExecutionPlan rather than a bare string.

    Selection priority (when requested="auto"):
    1. C extension available + memory sufficient + (n_cvt=1 or C general available)
       -> numpy-batch
    2. JAX available -> jax-batch (if fits) or jax-streaming (if not)
    3. Fallback -> numpy-batch

    When requested is "numpy" or "jax", the backend is forced but the mode
    (batch vs streaming) is still determined by memory availability (for JAX).
    NumPy always uses batch mode (no streaming NumPy runner exists).

    Args:
        n_samples: Number of samples in the dataset.
        n_snps: Number of SNPs in the dataset.
        requested: Requested backend ("auto", "jax", or "numpy").
        n_cvt: Number of covariates (including intercept). Used to select
            accurate memory estimates (Uab is larger with more covariates)
            and to guard against numpy-batch when the C general extension is
            unavailable for n_cvt > 1.
        lmm_mode: LMM test type (1=Wald, 2=LRT, 3=Score, 4=All). Accepted
            for API symmetry with ``run_lmm()``; not used in selection logic.

    Returns:
        ExecutionPlan with backend, mode, and reason.
    """
    _valid_requests = ("auto", "jax", "numpy")
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

    if requested == "jax":
        if not has_jax():
            raise ValueError(
                "Backend 'jax' was explicitly requested but JAX is not installed. "
                "Install JAX with: pip install jamma[jax]"
            )
        est = estimate_lmm_memory(n_samples, n_snps, n_cvt=n_cvt)
        if est.sufficient:
            return ExecutionPlan(
                "jax",
                "batch",
                "JAX backend explicitly requested, genotypes fit in memory",
            )
        return ExecutionPlan(
            "jax",
            "streaming",
            f"JAX backend explicitly requested, {est.total_gb:.1f}GB exceeds "
            f"{est.available_gb:.1f}GB available",
        )

    # Auto selection
    c_ext_available = is_c_extension_usable()
    est = estimate_lmm_memory(n_samples, n_snps, n_cvt=n_cvt)

    # Prefer numpy+C when genotypes fit in memory and C handles the n_cvt case.
    # When n_cvt > 1 we need the C general extension; if it's absent the Python
    # loop fallback is slower than JAX, so fall through to the JAX check below.
    if c_ext_available and est.sufficient:
        from jamma.lmm.compute_numpy import (
            _C_GENERAL_AVAILABLE,  # deferred: circular dep
        )

        c_handles_n_cvt = n_cvt <= 1 or _C_GENERAL_AVAILABLE
        if c_handles_n_cvt:
            return ExecutionPlan(
                "numpy",
                "batch",
                f"C extension available, {est.total_gb:.1f}GB fits in "
                f"{est.available_gb:.1f}GB available",
            )

    # JAX available: pick batch or streaming by memory
    if has_jax():
        if est.sufficient:
            # C extension is necessarily unavailable (or insufficient for n_cvt)
            # here — the c_ext + sufficient + c_handles_n_cvt case already returned
            # numpy-batch above.
            return ExecutionPlan("jax", "batch", "C extension unavailable")
        return ExecutionPlan(
            "jax",
            "streaming",
            f"{est.total_gb:.1f}GB exceeds {est.available_gb:.1f}GB, "
            "using JAX streaming",
        )

    # Fallback: pure NumPy batch (no C extension, no JAX)
    if not est.sufficient:
        logger.warning(
            f"No C extension or JAX available. Dataset requires ~{est.total_gb:.1f}GB "
            f"but only {est.available_gb:.1f}GB available. "
            "Install JAX for streaming support: pip install jamma[jax]"
        )
    return ExecutionPlan(
        "numpy",
        "batch",
        "Fallback — no C extension or JAX available",
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
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    use_gpu: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
    lmm_mode: int = 1,
) -> tuple[LmmRunResult, int]:
    """Unified LMM entry point that dispatches to the correct runner.

    Routes to run_lmm_association_numpy, run_lmm_association_jax, or
    run_lmm_association_streaming based on the ExecutionPlan. If no plan
    is provided, auto-selects via select_execution_mode().

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
        n_refine: Golden section iterations.
        use_gpu: Whether to use GPU acceleration.
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
                "ExecutionPlan('jax', 'streaming', ...)) when using bed_path."
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
            "use_gpu": use_gpu,
            "check_memory": check_memory,
            "show_progress": show_progress,
            "lmm_mode": lmm_mode,
        }

    # Dispatch based on plan
    if execution_plan.backend == "numpy" and execution_plan.mode == "batch":
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

    if execution_plan.backend == "jax" and execution_plan.mode == "batch":
        if genotypes is None:
            raise ValueError(
                "jax-batch mode requires genotypes array, but genotypes=None"
            )
        from jamma.lmm.runner_jax import run_lmm_association_jax

        result = run_lmm_association_jax(
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

    if execution_plan.backend == "jax" and execution_plan.mode == "streaming":
        if bed_path is None:
            raise ValueError("jax-streaming mode requires bed_path, but bed_path=None")
        from jamma.lmm.runner_jax_streaming import run_lmm_association_streaming

        result, n_tested = run_lmm_association_streaming(
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
            **common_kwargs if config is None else {},
        )
        return result, n_tested

    raise ValueError(f"Unsupported execution plan: {execution_plan.runner_name}")
