"""Pure-NumPy batch LMM association runner.

No JAX dependency. Input genotypes must fit in memory.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import psutil
from loguru import logger

from jamma.core.memory import estimate_lmm_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.core.threading import (
    blas_threads,
    get_physical_core_count,
    is_blas_controllable,
)
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _C_GENERAL_AVAILABLE,
    _C_MODE4_AVAILABLE,
    _C_SPLIT_AVAILABLE,
    LmmMode,
    _compute_lmm_chunk_numpy,
    compute_mode4_split_c_ws,
    compute_wald_general_c_ws,
    compute_wald_split_c_ws,
    create_lmm_workspace,
    create_lmm_workspace_general,
    create_lmm_workspace_mode4,
)
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
    reconstruct_uab_from_soa,
    reset_p_yy_warned,
)
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    _eigendecompose_or_reuse,
    compute_and_log_pve,
    validate_runner_inputs,
)
from jamma.lmm.results import (
    _build_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
    write_streaming_chunk,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.schema import LmmConfig, LmmRunResult
from jamma.utils.logging import log_rss_memory

# NumPy has no int32 buffer constraint — allow larger chunks than JAX runner.
_MAX_CHUNK = 200_000

# Memory budget bounds for auto-scaling
_MIN_BUDGET = 2_000_000_000  # 2 GB floor (original default)
_MAX_BUDGET = 40_000_000_000  # 40 GB ceiling

# Minimum number of chunks before pipelined execution is worthwhile.
_MIN_PIPELINE_CHUNKS = 30


_ALL_RESULT_KEYS = (
    "lambdas",
    "logls",
    "betas",
    "ses",
    "pwalds",
    "lambdas_mle",
    "p_lrts",
    "p_scores",
)


def _select_wald_fn(n_cvt: int):
    """Return the C workspace Wald compute function appropriate for n_cvt.

    Args:
        n_cvt: Number of covariates.

    Returns:
        compute_wald_split_c_ws for n_cvt=1; compute_wald_general_c_ws for n_cvt>1.
    """
    return compute_wald_split_c_ws if n_cvt == 1 else compute_wald_general_c_ws


def _create_wald_workspace_for_ncvt(
    n_cvt: int,
    eigenvalues: np.ndarray,
    uab_invariant_soa: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> object:
    """Create the appropriate C Wald workspace for any n_cvt.

    Dispatches to create_lmm_workspace (split, n_cvt=1) or
    create_lmm_workspace_general (general, n_cvt>1). Returns None if the
    required C extension is unavailable.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_invariant_soa: Invariant Uab SoA array (n_inv, n_samples).
        n_samples: Number of samples.
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid resolution.
        n_refine: Golden section iterations.
        n_threads: OpenMP thread count.

    Returns:
        C PyCapsule workspace, or None if extension unavailable.
    """
    if n_cvt == 1:
        return create_lmm_workspace(
            eigenvalues,
            uab_invariant_soa,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            n_threads,
        )
    if _C_GENERAL_AVAILABLE:
        return create_lmm_workspace_general(
            eigenvalues,
            uab_invariant_soa,
            n_samples,
            n_cvt,
            l_min,
            l_max,
            n_grid,
            n_refine,
            n_threads,
        )
    logger.debug(
        "Wald workspace unavailable for n_cvt={} (general C extension missing)", n_cvt
    )
    return None


def _guarded_compute(
    fn: Callable[..., dict],
    *args: object,
    operation: str,
    write_offset: int,
    n_filtered: int,
    **kwargs: object,
) -> dict:
    """Call *fn* with error wrapping that identifies the failed operation.

    Extra positional and keyword arguments are forwarded to *fn*;
    *operation*, *write_offset*, and *n_filtered* are consumed by the wrapper.

    MemoryError, ValueError, TypeError, and OverflowError propagate unchanged.
    All other exceptions are wrapped in a RuntimeError whose message includes
    the *operation* label, *write_offset*, and *n_filtered* for diagnosis.
    """
    try:
        return fn(*args, **kwargs)
    except (MemoryError, ValueError, TypeError, OverflowError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"{operation} failed at SNP offset "
            f"{write_offset}/{n_filtered}. "
            f"Processed {write_offset} SNPs before failure."
        ) from exc


def _compose_mode4_results(
    wald_cr: dict,
    n_cvt: int,
    eigenvalues_np: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    *,
    Hi_eval_null: np.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict:
    """Compose mode-4 (All) results from Wald + Score + LRT.

    Calls Score and LRT dispatch separately to avoid redundant Wald
    computation, then merges non-None values from each test result.

    Merge order matters: Wald is applied last, so its REML-optimized
    betas/ses overwrite Score's values for the same keys.
    """
    score_cr = _compute_lmm_chunk_numpy(
        3,  # Score only
        n_cvt,
        eigenvalues_np,
        Uab_batch,
        n_samples,
        Hi_eval_null=Hi_eval_null,
        n_threads=n_threads,
    )
    lrt_cr = _compute_lmm_chunk_numpy(
        2,  # LRT only
        n_cvt,
        eigenvalues_np,
        Uab_batch,
        n_samples,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_refine=n_refine,
        logl_H0=logl_H0,
        n_threads=n_threads,
    )
    cr: dict = {k: None for k in _ALL_RESULT_KEYS}
    for d in (score_cr, lrt_cr, wald_cr):
        for k, v in d.items():
            if v is not None:
                cr[k] = v
    return cr


def compute_pipeline_core_split(n_samples: int, total_cores: int) -> tuple[int, int]:
    """Compute rotation/compute thread split for the pipeline path.

    DGEMM rotation scales with n_samples^2 * chunk_size while per-SNP
    compute scales with chunk_size * (n_grid + n_refine). For large
    n_samples rotation dominates; for small n_samples compute dominates.

    Args:
        n_samples: Number of samples in the dataset.
        total_cores: Physical core count available.

    Returns:
        (rotation_threads, compute_threads) tuple. Both >= 1.
    """
    if n_samples > 10_000:
        rot = max(1, total_cores // 2)
    elif n_samples > 1_000:
        rot = max(1, total_cores // 3)
    else:
        rot = max(1, total_cores // 4)
    return rot, max(1, total_cores - rot)


def compute_adaptive_core_split(
    rot_time: float,
    compute_time: float,
    total_cores: int,
    *,
    n_samples: int = 0,
) -> tuple[int, int]:
    """Compute rotation/compute thread split from measured first-chunk times.

    Allocates threads proportionally to observed rotation vs compute wall time.
    Falls back to static heuristic when profiling data is degenerate (both
    times near zero, which happens on small datasets where profiling overhead
    dominates).

    Args:
        rot_time: Wall time for first-chunk rotation (UT@G DGEMM), seconds.
        compute_time: Wall time for first-chunk compute (C extension), seconds.
        total_cores: Physical core count available.
        n_samples: Sample count for static fallback (only used when times are
            degenerate).

    Returns:
        (rotation_threads, compute_threads) tuple. Both >= 1.
    """
    total_time = rot_time + compute_time
    if total_time < 0.01:  # < 10ms: profiling not meaningful, use static
        return compute_pipeline_core_split(n_samples, total_cores)

    rot_fraction = rot_time / total_time
    rot_threads = max(1, min(total_cores - 1, round(total_cores * rot_fraction)))
    compute_threads = max(1, total_cores - rot_threads)
    return rot_threads, compute_threads


def _compute_chunk_size_numpy(
    n_samples: int,
    n_filtered: int,
    n_cvt: int = 1,
    *,
    use_split: bool = False,
    lmm_mode: int = 1,
    fused_mode4: bool = False,
    mem_budget_bytes: int | None = None,
    pipeline_buffers: int = 1,
) -> int:
    """Compute chunk size based on RAM budget (no int32 constraint for NumPy).

    Scales the memory budget with available RAM to minimise DRAM passes
    through the eigenvector matrix during UT@G rotation.

    Args:
        n_samples: Number of samples.
        n_filtered: Number of filtered SNPs.
        n_cvt: Number of covariates.
        use_split: If True, use split Uab accounting instead of full Uab.
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All). Affects
            memory accounting: Wald uses 4 cols/SNP (3 varying + 1 UtG),
            non-Wald uses 9 cols/SNP (3 varying + 6 reconstructed Uab peak).
        fused_mode4: If True, mode-4 uses fused C kernel (4-col accounting,
            same as Wald) instead of reconstruct+compose (9-col).
        mem_budget_bytes: Explicit per-chunk memory budget in bytes.
            None (default) auto-scales with available RAM.
        pipeline_buffers: Number of live chunks (1 for sequential,
            2 for pipeline double-buffering). Divides the budget.

    Returns:
        Chunk size (number of SNPs per chunk).
    """
    if not isinstance(pipeline_buffers, int):
        raise TypeError(
            f"pipeline_buffers must be an int, got {type(pipeline_buffers).__name__}"
        )
    if pipeline_buffers < 1:
        raise ValueError(f"pipeline_buffers must be >= 1, got {pipeline_buffers}")

    if use_split and n_cvt == 1:
        if lmm_mode == 1 or (lmm_mode == 4 and fused_mode4):
            # Wald split or fused mode-4: 3 varying columns + 1 UtG per SNP
            bytes_per_snp = n_samples * 4 * 8
        else:
            # Non-Wald split: reconstruct_uab_from_soa allocates 6-col Uab
            # while 3-col varying SoA is still live = 9 cols peak
            bytes_per_snp = n_samples * 9 * 8
    elif use_split and n_cvt > 1:
        from jamma.lmm.likelihood import classify_uab_columns

        _inv, var = classify_uab_columns(n_cvt)
        n_var = len(var)
        if lmm_mode == 1:
            # Wald: workspace path, no Uab reconstruction
            bytes_per_snp = n_samples * (n_var + 1) * 8
        else:
            # Score/LRT/All: reconstruct_uab_from_soa allocates (n_snps,
            # n_samples, n_index) while varying SoA is still live.
            n_index = (n_cvt + 3) * (n_cvt + 2) // 2
            bytes_per_snp = n_samples * (n_var + n_index) * 8
    else:
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        bytes_per_snp = n_samples * n_index * 8

    if bytes_per_snp == 0:
        return n_filtered

    if mem_budget_bytes is not None:
        mem_budget = mem_budget_bytes
    else:
        available = psutil.virtual_memory().available
        # Budget: 15% of available RAM (up from 5%), 2 GB floor, 40 GB ceiling.
        # Modern machines (128-512 GB) can afford larger working sets. The floor
        # prevents degenerate chunk sizes on low-memory systems; the ceiling
        # prevents excessive allocation on high-memory systems.
        mem_budget = max(_MIN_BUDGET, min(int(available * 0.15), _MAX_BUDGET))

    mem_budget = mem_budget // pipeline_buffers

    chunk_from_memory = int(mem_budget / bytes_per_snp)
    chunk = max(100, min(chunk_from_memory, n_filtered, _MAX_CHUNK))
    return chunk


def run_lmm_association_numpy(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    snp_info: list,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    use_gpu: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
    lmm_mode: LmmMode = 1,
    config: LmmConfig | None = None,
    output_path: Path | None = None,
    clear_caches: bool = True,
) -> LmmRunResult:
    """Run LMM association tests using pure-NumPy batch processing.

    Processes SNPs in memory-bounded chunks using BLAS-backed NumPy operations.
    No JAX dependency. Input genotypes must fit in memory; for disk streaming
    use run_lmm_association_streaming.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples) or None when
            pre-computed eigenvalues/eigenvectors are provided. WARNING: may
            be overwritten in-place during eigendecomposition (buffer reused
            for eigenvectors). Treat as consumed; pass kinship.copy() if you
            need the original matrix after this call.
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0.
        covariates: Covariate matrix (n_samples, n_cvt) or None for intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (clamped to min 20
            internally for ~1e-5 tolerance).
        use_gpu: Accepted but silently ignored — NumPy backend is CPU-only.
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        config: LmmConfig instance. When provided, overrides individual
            threshold/mode kwargs above.
        output_path: Path for per-chunk disk streaming. When set, results
            are written incrementally and the returned LmmRunResult has
            empty associations and n_tested populated instead.
        clear_caches: Accepted for signature parity with JAX runners.
            No-op — NumPy has no compilation caches.

    Returns:
        LmmRunResult with per-SNP associations and PVE from null model.
            When output_path is set, associations is empty (results on
            disk) and n_tested contains the count of SNPs written.

    Raises:
        MemoryError: If check_memory=True and insufficient memory.
        ValueError: If only one of eigenvalues/eigenvectors is provided,
            or if no valid samples remain after filtering.
    """
    # Unpack config if provided (config takes precedence over individual kwargs)
    if config is not None:
        kw = config.as_kwargs()
        maf_threshold = kw["maf_threshold"]
        miss_threshold = kw["miss_threshold"]
        l_min, l_max = kw["l_min"], kw["l_max"]
        n_grid, n_refine = kw["n_grid"], kw["n_refine"]
        use_gpu, check_memory = kw["use_gpu"], kw["check_memory"]
        show_progress, lmm_mode = kw["show_progress"], kw["lmm_mode"]

    # Reset per-run warning flags so each run gets its own diagnostics
    reset_p_yy_warned()

    if use_gpu:
        logger.warning(
            "use_gpu=True ignored: NumPy backend is CPU-only. "
            "Install JAX for GPU support: pip install jamma[jax]"
        )

    # Memory check before workflow (uses genotype shape, runner-specific)
    n_samples, n_snps = genotypes.shape
    start_time = time.perf_counter()

    if show_progress:
        logger.info("Performing LMM Association Test (NumPy batch)")
        logger.info(f"  Total individuals: {n_samples:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.debug(
            f"MAF threshold = {maf_threshold}, missing threshold = {miss_threshold}"
        )

    if check_memory:
        est = estimate_lmm_memory(n_samples, n_snps)
        logger.info(
            f"LMM memory: estimated {est.total_gb:.1f}GB, "
            f"available {est.available_gb:.1f}GB"
        )
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for LMM workflow with {n_samples:,} samples × "
                f"{n_snps:,} SNPs.\n"
                f"Need: {est.total_gb:.1f}GB, Available: {est.available_gb:.1f}GB\n"
                f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
                f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"genotypes={est.genotypes_gb:.1f}GB"
            )

    # Validate inputs and apply sample filtering (shared logic for all runners)
    setup = validate_runner_inputs(
        phenotypes, kinship, covariates, eigenvalues, eigenvectors, lmm_mode
    )
    phenotypes = setup.phenotypes
    kinship = setup.kinship
    covariates = setup.covariates
    eigenvalues = setup.eigenvalues
    eigenvectors = setup.eigenvectors
    n_samples = setup.n_samples

    # Apply the same valid-mask to genotypes (runner-specific: genotypes in memory)
    if not np.all(setup.valid_mask):
        genotypes = genotypes[setup.valid_mask, :]

    n_samples, n_snps = genotypes.shape

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Vectorized SNP stats and filtering using shared functions
    col_means, missing_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, missing_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )
    snp_indices = np.where(snp_mask)[0]

    if len(snp_indices) == 0:
        logger.warning(
            f"All {n_snps} SNPs filtered out (MAF>{maf_threshold}, "
            f"miss<{miss_threshold}). No association tests to run. "
            f"Consider relaxing --maf or --miss thresholds."
        )
        if output_path is not None:
            from jamma.lmm.io import IncrementalAssocWriter

            with IncrementalAssocWriter(
                output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            ):
                pass  # Header-only file, matching streaming runner behavior
        return LmmRunResult(associations=[], n_tested=0)

    # Extract filtered stats as numpy arrays (use allele_freqs for output, not mafs)
    filtered_afs = allele_freqs[snp_indices]
    filtered_miss = missing_counts[snp_indices].astype(int)

    t_eigen_start = time.perf_counter()
    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship,
        eigenvalues,
        eigenvectors,
        show_progress,
        "lmm_numpy",
        check_memory=check_memory,
    )
    del kinship
    gc.collect()

    # Use all physical cores for BLAS rotation
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = U.T @ W
        Uty = U.T @ phenotypes

    logl_H0, lambda_null_mle, Hi_eval_null = _compute_null_model_common(
        lmm_mode,
        eigenvalues_np,
        UtW,
        Uty,
        n_cvt,
        show_progress,
        l_min=l_min,
        l_max=l_max,
    )

    t_eigen_end = time.perf_counter()

    pve, pve_se = compute_and_log_pve(eigenvalues_np, UtW, Uty, n_cvt, l_min, l_max)

    n_filtered = len(snp_indices)

    # Determine split/pipeline eligibility BEFORE chunk sizing so the
    # budget can use accurate per-SNP accounting (varying cols vs full Uab).
    # n_cvt=1: split available for all modes — Wald uses C workspace,
    #   LRT/Score/All reconstruct full Uab from SoA then call C batch.
    # n_cvt>1: split available for all modes — reconstruct_uab_from_soa now
    #   handles general n_cvt, and Score/LRT dispatch calls C general batch.
    use_split = (_C_SPLIT_AVAILABLE and n_cvt == 1) or (
        _C_GENERAL_AVAILABLE and n_cvt > 1
    )

    # Fused mode-4: single-pass Wald/Score/LRT from SoA data (no Uab
    # reconstruction).  Only for n_cvt=1 with mode-4 C kernel available.
    use_fused_mode4 = use_split and lmm_mode == 4 and n_cvt == 1 and _C_MODE4_AVAILABLE

    if lmm_mode == 4:
        if use_fused_mode4:
            logger.debug("Mode-4 dispatch: fused kernel (Wald/Score/LRT single pass)")
        else:
            reason = (
                "n_cvt > 1"
                if n_cvt > 1
                else "fused kernel unavailable"
                if not _C_MODE4_AVAILABLE
                else "C split extension unavailable"
            )
            logger.debug(f"Mode-4 dispatch: compose fallback ({reason})")

    chunk_size = _compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt,
        use_split=use_split,
        lmm_mode=lmm_mode,
        fused_mode4=use_fused_mode4,
    )
    n_chunks = (n_filtered + chunk_size - 1) // chunk_size

    # Pipeline overlaps rotation(N+1) with compute(N) using an adaptive core
    # split (see compute_pipeline_core_split). This helps when rotation ≈
    # compute per chunk (many small chunks). With large chunks (few passes
    # through U), rotation >> compute per chunk, so the pipeline overlap
    # hides nothing but costs reduced BLAS throughput. Only enable when
    # enough chunks exist for overlap to matter.
    use_pipeline = use_split and n_chunks >= _MIN_PIPELINE_CHUNKS

    if use_pipeline:
        # Pipeline has 2 chunks alive simultaneously — halve the budget
        chunk_size = _compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt,
            use_split=use_split,
            lmm_mode=lmm_mode,
            fused_mode4=use_fused_mode4,
            pipeline_buffers=2,
        )
        n_chunks = (n_filtered + chunk_size - 1) // chunk_size
        use_pipeline = use_split and n_chunks >= _MIN_PIPELINE_CHUNKS

    # OpenMP thread count for C extension (set once, reused per chunk).
    # When C extension is active, BLAS threads are set to 1 inside the compute
    # phase to prevent oversubscription between BLAS and OpenMP.
    # On macOS/Accelerate, blas_threads(1) is a no-op — Accelerate keeps using
    # all cores — so we halve OpenMP threads to share cores with BLAS.
    if _C_ACCEL_AVAILABLE:
        cores = get_physical_core_count()
        omp_threads = max(1, cores // 2) if not is_blas_controllable() else cores
    else:
        omp_threads = 1

    if show_progress:
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"  Processing in {n_chunks} chunks ({chunk_size:,} SNPs/chunk)"
            )
    # Streaming mode: write per-chunk to disk, skip arrays_out allocation.
    streaming = output_path is not None
    if streaming:
        from jamma.lmm.io import IncrementalAssocWriter

        writer_ctx = IncrementalAssocWriter(
            output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
        )
        arrays_out = None
    else:
        writer_ctx = None
        arrays_out = {
            key: np.empty(n_filtered, dtype=np.float64)
            for key in _RESULT_FIELDS[lmm_mode]
        }

    write_offset = 0

    # Timing accumulators for per-chunk phases
    t_rotation_total = 0.0
    t_numpy_compute_total = 0.0
    t_result_write_total = 0.0

    # Per-chunk diagnostic accumulators (used in streaming mode where
    # arrays_out is not available for post-loop inspection).
    nan_counts: dict[str, int] = {}
    n_at_lmin_accum = 0
    n_at_lmax_accum = 0

    chunk_starts = list(range(0, n_filtered, chunk_size))

    # Pre-allocate rotation output buffer for sequential path only.
    # Pipeline path allocates per-chunk buffers because the background thread
    # writes UtG for chunk N+1 while the foreground thread still reads chunk N.
    UtG_buf = (
        None if use_pipeline else np.empty((n_samples, chunk_size), dtype=np.float64)
    )

    # Pipeline thread budget: partition physical cores between concurrent
    # BLAS rotation (background) and C extension compute (foreground) to
    # prevent oversubscription. Without partitioning, both use all cores
    # (2N threads on N cores), causing context-switch overhead.
    # On Accelerate (uncontrollable BLAS), rotation threads are ignored
    # by blas_threads() and Accelerate uses all cores, so give compute
    # fewer threads to leave headroom.
    if use_pipeline:
        total_cores = get_physical_core_count()
        if is_blas_controllable():
            pipeline_rot_threads, pipeline_omp_threads = compute_pipeline_core_split(
                n_samples, total_cores
            )
            logger.debug(
                f"Pipeline core split: {pipeline_rot_threads} rotation, "
                f"{pipeline_omp_threads} compute (n_samples={n_samples:,})"
            )
        else:
            # Accelerate will use all cores for rotation regardless;
            # give OpenMP half to reduce contention
            pipeline_omp_threads = max(1, total_cores // 2)
            pipeline_rot_threads = total_cores  # ignored by blas_threads()
    else:
        pipeline_omp_threads = omp_threads
        pipeline_rot_threads = rotation_threads

    # Enforce minimum 20 golden section iterations for ~1e-5 lambda tolerance
    n_refine = max(n_refine, 20)

    # Precompute SNP-invariant Uab columns once (depends only on UtW, Uty).
    # Eliminates per-chunk reconstruction: uab_invariant_soa is shared across
    # all chunks and referenced via closure by _prepare_chunk.
    uab_invariant_soa = (
        compute_uab_invariant_soa(UtW, Uty, n_cvt) if use_split else None
    )

    # Create persistent C workspace once (before chunk loop).
    # Holds precomputed lambda_grid, hi_eval_grid, logdet_h_grid, grid_inv, and
    # invariant Iab column sums — reused across all chunks without reallocation.
    # PyCapsule is freed automatically when lmm_workspace goes out of scope.
    # Create Wald workspace for modes 1 (Wald) and 4 (All) — both need
    # REML Wald statistics. Modes 2 (LRT) and 3 (Score) don't need a
    # Wald workspace; they reconstruct Uab and call C batch functions.
    # Fused mode-4 workspace extends Wald with null-model fields.
    if use_split and lmm_mode in (1, 4):
        if use_fused_mode4:
            lmm_workspace = create_lmm_workspace_mode4(
                eigenvalues_np,
                uab_invariant_soa,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                pipeline_omp_threads,
                Hi_eval_null,
                logl_H0,
            )
        else:
            lmm_workspace = _create_wald_workspace_for_ncvt(
                n_cvt,
                eigenvalues_np,
                uab_invariant_soa,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                pipeline_omp_threads,
            )
    else:
        lmm_workspace = None

    def _prepare_chunk(chunk_start: int) -> tuple:
        """Slice, impute, rotate, build split Uab in SoA layout.

        BLAS operations release the GIL. Returns SoA-layout varying Uab —
        invariant Uab is precomputed once in outer scope (uab_invariant_soa).
        """
        chunk_end = min(chunk_start + chunk_size, n_filtered)
        chunk_indices = snp_indices[chunk_start:chunk_end]
        geno_chunk = genotypes[:, chunk_indices]

        # Mean-impute
        chunk_means = col_means[chunk_indices]
        missing = np.isnan(geno_chunk)
        if missing.any():  # RUN-06: skip O(n*chunk) np.where on clean data
            geno_chunk = np.where(missing, chunk_means[None, :], geno_chunk)
        del missing

        # Rotate — fresh buffer each call (pipeline path, not reusing UtG_buf)
        with blas_threads(pipeline_rot_threads):
            UtG = U.T @ geno_chunk

        # Build SNP-varying Uab in SoA layout (n_snps, n_var, n_samples)
        # — n_var=3 for n_cvt=1.
        # Invariant part (uab_invariant_soa) is from outer scope — precomputed once.
        uab_var_soa = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG)
        actual_len = chunk_end - chunk_start
        return uab_var_soa, actual_len

    def _compute_and_write(uab_var_soa: np.ndarray, actual_len: int) -> None:
        """Run C extension compute on a chunk and write results.

        Dispatches by lmm_mode:
        - Mode 1 (Wald): C workspace path (no Uab reconstruction)
        - Mode 4 fused: single-pass Wald/Score/LRT via mode-4 workspace
        - Mode 4 fallback: Wald via workspace + Score/LRT via reconstructed Uab
        - Modes 2, 3 (LRT, Score): Reconstruct Uab, C batch dispatch
        """
        nonlocal write_offset, t_numpy_compute_total, t_result_write_total
        nonlocal n_at_lmin_accum, n_at_lmax_accum

        t_compute_start = time.perf_counter()

        if use_fused_mode4 and lmm_workspace is not None:
            # Fused mode-4: single C call for all 8 output arrays
            cr = _guarded_compute(
                compute_mode4_split_c_ws,
                lmm_workspace,
                uab_var_soa,
                pipeline_omp_threads,
                operation="Fused mode-4 C workspace compute",
                write_offset=write_offset,
                n_filtered=n_filtered,
            )
        elif lmm_mode in (1, 4) and lmm_workspace is not None:
            # Wald via workspace
            wald_fn = _select_wald_fn(n_cvt)
            wald_cr = _guarded_compute(
                wald_fn,
                lmm_workspace,
                uab_var_soa,
                pipeline_omp_threads,
                operation="Wald C workspace compute",
                write_offset=write_offset,
                n_filtered=n_filtered,
            )

            if lmm_mode == 1:
                cr = wald_cr
            else:
                # Mode 4 fallback: Wald from workspace, Score+LRT from reconstructed Uab
                Uab_batch = reconstruct_uab_from_soa(
                    uab_invariant_soa, uab_var_soa, n_cvt=n_cvt
                )
                blas_ctx = blas_threads(1) if _C_ACCEL_AVAILABLE else nullcontext()
                with blas_ctx:
                    cr = _guarded_compute(
                        _compose_mode4_results,
                        wald_cr,
                        n_cvt,
                        eigenvalues_np,
                        Uab_batch,
                        n_samples,
                        Hi_eval_null=Hi_eval_null,
                        l_min=l_min,
                        l_max=l_max,
                        n_grid=n_grid,
                        n_refine=n_refine,
                        logl_H0=logl_H0,
                        n_threads=pipeline_omp_threads,
                        operation="Mode-4 Score/LRT composition",
                        write_offset=write_offset,
                        n_filtered=n_filtered,
                    )
        else:
            # Modes 2, 3: reconstruct Uab, C batch dispatch
            Uab_batch = reconstruct_uab_from_soa(
                uab_invariant_soa, uab_var_soa, n_cvt=n_cvt
            )
            blas_ctx = blas_threads(1) if _C_ACCEL_AVAILABLE else nullcontext()
            with blas_ctx:
                cr = _guarded_compute(
                    _compute_lmm_chunk_numpy,
                    lmm_mode,
                    n_cvt,
                    eigenvalues_np,
                    Uab_batch,
                    n_samples,
                    l_min=l_min,
                    l_max=l_max,
                    n_grid=n_grid,
                    n_refine=n_refine,
                    Hi_eval_null=Hi_eval_null,
                    logl_H0=logl_H0,
                    n_threads=pipeline_omp_threads,
                    operation="Score/LRT C batch dispatch",
                    write_offset=write_offset,
                    n_filtered=n_filtered,
                )

        t_numpy_compute_total += time.perf_counter() - t_compute_start

        t_write_start = time.perf_counter()
        if streaming:
            chunk_arrays = {
                key: cr[key][:actual_len] for key in _RESULT_FIELDS[lmm_mode]
            }
            n_at_lmin_accum, n_at_lmax_accum = write_streaming_chunk(
                writer,
                lmm_mode,
                snp_indices[write_offset : write_offset + actual_len],
                snp_info,
                filtered_afs[write_offset : write_offset + actual_len],
                filtered_miss[write_offset : write_offset + actual_len],
                chunk_arrays,
                l_min,
                l_max,
                nan_counts,
                n_at_lmin_accum,
                n_at_lmax_accum,
            )
        else:
            s = slice(write_offset, write_offset + actual_len)
            for key in arrays_out:
                arrays_out[key][s] = cr[key][:actual_len]
        write_offset += actual_len
        t_result_write_total += time.perf_counter() - t_write_start

    writer_cm = writer_ctx if streaming else nullcontext()

    with writer_cm as writer:
        if use_pipeline:
            # Pipelined: overlap rotation of chunk N+1 with C compute of chunk N.
            # Both operations release the GIL so they run concurrently.

            # --- Profile first chunk for adaptive core split ---
            # Prepare chunk 0 (rotation) and compute it inline to measure both
            # stage durations. Re-derive thread split from measured times so
            # remaining chunks use an empirically correct allocation.
            t_rot_start = time.perf_counter()
            uab_var_soa_first, actual_len_first = _prepare_chunk(chunk_starts[0])
            t_first_rot = time.perf_counter() - t_rot_start
            t_rotation_total += t_first_rot

            t_compute_start = time.perf_counter()
            _compute_and_write(uab_var_soa_first, actual_len_first)
            t_first_compute = time.perf_counter() - t_compute_start
            del uab_var_soa_first

            # Re-derive core split from measured times (only if chunks remain and
            # BLAS is controllable — uncontrollable BLAS ignores thread settings).
            if n_chunks > 2 and is_blas_controllable():
                old_rot = pipeline_rot_threads
                old_omp = pipeline_omp_threads
                pipeline_rot_threads, pipeline_omp_threads = (
                    compute_adaptive_core_split(
                        t_first_rot,
                        t_first_compute,
                        total_cores,
                        n_samples=n_samples,
                    )
                )
                if (pipeline_rot_threads, pipeline_omp_threads) != (old_rot, old_omp):
                    logger.debug(
                        f"Adaptive core split: {old_rot}/{old_omp} -> "
                        f"{pipeline_rot_threads}/{pipeline_omp_threads} "
                        f"(rot={t_first_rot:.3f}s, compute={t_first_compute:.3f}s)"
                    )

            # Process remaining chunks with the (possibly updated) adaptive split.
            # remaining_starts is always non-empty: use_pipeline requires
            # n_chunks >= _MIN_PIPELINE_CHUNKS (30), so at least 29 remain.
            remaining_starts = chunk_starts[1:]

            # Seed the pipeline by preparing the first remaining chunk.
            # _prepare_chunk reads pipeline_rot_threads from this scope, so it
            # uses the updated adaptive split from this point onward.
            t_rot_start = time.perf_counter()
            current = _prepare_chunk(remaining_starts[0])
            t_rotation_total += time.perf_counter() - t_rot_start

            # Progress tracking for the pipeline loop. One chunk fully computed
            # (profiled), one prepared (rotation only), so initialise at 2.
            pipeline_bar = None
            if show_progress and n_chunks > 1:
                import sys

                import progressbar as _pb

                widgets = [
                    "LMM association: ",
                    _pb.Counter(),
                    f"/{n_chunks} ",
                    _pb.Percentage(),
                    " ",
                    _pb.Bar(),
                    " ",
                    _pb.Timer(),
                    " ",
                    _pb.ETA(),
                ]
                pipeline_bar = _pb.ProgressBar(
                    max_value=n_chunks, widgets=widgets, fd=sys.stdout
                )
                pipeline_bar.start()
                # profiled chunk + seeded (prepared, not computed)
                pipeline_bar.update(2)

            with ThreadPoolExecutor(max_workers=1) as executor:
                for i_chunk, chunk_start in enumerate(remaining_starts[1:], start=3):
                    uab_var_soa, actual_len = current

                    # Submit next chunk preparation (runs in background thread)
                    future = executor.submit(_prepare_chunk, chunk_start)

                    # C extension compute on current chunk (releases GIL).
                    # Cores are partitioned: rotation gets pipeline_rot_threads,
                    # compute gets pipeline_omp_threads.
                    _compute_and_write(uab_var_soa, actual_len)
                    del uab_var_soa

                    # Wait for background preparation to complete
                    t_rot_start = time.perf_counter()
                    try:
                        current = future.result()
                    except (MemoryError, ValueError, TypeError, OverflowError):
                        raise
                    except Exception as exc:
                        raise RuntimeError(
                            f"Pipeline chunk preparation failed at SNP offset "
                            f"{write_offset}/{n_filtered} (chunk starting at "
                            f"index {chunk_start}). "
                            f"Processed {write_offset} SNPs before failure."
                        ) from exc
                    t_rotation_total += time.perf_counter() - t_rot_start

                    if pipeline_bar is not None:
                        pipeline_bar.update(i_chunk)

                # Process last chunk (no next chunk to overlap with)
                uab_var_soa, actual_len = current
                _compute_and_write(uab_var_soa, actual_len)
                del uab_var_soa

            if pipeline_bar is not None:
                pipeline_bar.update(n_chunks)
                pipeline_bar.finish()
        else:
            # Sequential path (single chunk or non-pipeline execution)
            if show_progress and n_chunks > 1:
                chunk_iterator = progress_iterator(
                    chunk_starts, total=n_chunks, desc="LMM association"
                )
            else:
                chunk_iterator = iter(chunk_starts)

            def _run_lmm_chunk(Uab_batch: np.ndarray) -> dict:
                """Run LMM compute on a Uab batch with BLAS thread control."""
                blas_ctx = blas_threads(1) if _C_ACCEL_AVAILABLE else nullcontext()
                with blas_ctx:
                    return _guarded_compute(
                        _compute_lmm_chunk_numpy,
                        lmm_mode,
                        n_cvt,
                        eigenvalues_np,
                        Uab_batch,
                        n_samples,
                        l_min=l_min,
                        l_max=l_max,
                        n_grid=n_grid,
                        n_refine=n_refine,
                        Hi_eval_null=Hi_eval_null,
                        logl_H0=logl_H0,
                        n_threads=omp_threads,
                        operation="LMM chunk compute",
                        write_offset=write_offset,
                        n_filtered=n_filtered,
                    )

            for chunk_start in chunk_iterator:
                chunk_end = min(chunk_start + chunk_size, n_filtered)
                chunk_indices = snp_indices[chunk_start:chunk_end]
                geno_chunk = genotypes[:, chunk_indices]

                # Mean-impute missing genotypes
                chunk_means = col_means[chunk_indices]
                missing = np.isnan(geno_chunk)
                if missing.any():  # RUN-06: skip O(n*chunk) np.where on clean data
                    geno_chunk = np.where(missing, chunk_means[None, :], geno_chunk)
                del missing, chunk_means

                # Rotate genotypes
                t_rot_start = time.perf_counter()
                actual_snps = geno_chunk.shape[1]
                with blas_threads(rotation_threads):
                    if actual_snps == chunk_size:
                        np.dot(U.T, geno_chunk, out=UtG_buf)
                        UtG = UtG_buf
                    else:
                        UtG = U.T @ geno_chunk
                t_rotation_total += time.perf_counter() - t_rot_start
                del geno_chunk

                # Compute
                t_compute_start = time.perf_counter()
                if use_split:
                    # Build SoA-layout varying Uab only — invariant precomputed.
                    uab_var_soa = batch_compute_uab_varying_soa_numpy(
                        n_cvt, UtW, Uty, UtG
                    )
                    del UtG
                    if use_fused_mode4 and lmm_workspace is not None:
                        # Fused mode-4: single C call for all 8 arrays
                        with blas_threads(1):
                            cr = _guarded_compute(
                                compute_mode4_split_c_ws,
                                lmm_workspace,
                                uab_var_soa,
                                omp_threads,
                                operation="Fused mode-4 C workspace compute",
                                write_offset=write_offset,
                                n_filtered=n_filtered,
                            )
                        del uab_var_soa
                    elif lmm_mode == 1 and lmm_workspace is not None:
                        # Wald: use C workspace (no full Uab needed)
                        with blas_threads(1):
                            cr = _guarded_compute(
                                _select_wald_fn(n_cvt),
                                lmm_workspace,
                                uab_var_soa,
                                omp_threads,
                                operation="Wald C workspace compute",
                                write_offset=write_offset,
                                n_filtered=n_filtered,
                            )
                        del uab_var_soa
                    elif lmm_mode == 4 and lmm_workspace is not None:
                        # Mode 4 fallback: Wald workspace + Score/LRT reconstructed Uab
                        Uab_batch = reconstruct_uab_from_soa(
                            uab_invariant_soa, uab_var_soa, n_cvt=n_cvt
                        )
                        with blas_threads(1):
                            wald_cr = _guarded_compute(
                                _select_wald_fn(n_cvt),
                                lmm_workspace,
                                uab_var_soa,
                                omp_threads,
                                operation="Wald C workspace compute",
                                write_offset=write_offset,
                                n_filtered=n_filtered,
                            )
                        del uab_var_soa
                        # Score + LRT via C batch dispatch (no redundant Wald)
                        with blas_threads(1):
                            cr = _guarded_compute(
                                _compose_mode4_results,
                                wald_cr,
                                n_cvt,
                                eigenvalues_np,
                                Uab_batch,
                                n_samples,
                                Hi_eval_null=Hi_eval_null,
                                l_min=l_min,
                                l_max=l_max,
                                n_grid=n_grid,
                                n_refine=n_refine,
                                logl_H0=logl_H0,
                                n_threads=omp_threads,
                                operation="Mode-4 Score/LRT composition",
                                write_offset=write_offset,
                                n_filtered=n_filtered,
                            )
                        del Uab_batch
                    else:
                        # LRT/Score (modes 2, 3): reconstruct Uab
                        Uab_batch = reconstruct_uab_from_soa(
                            uab_invariant_soa, uab_var_soa, n_cvt=n_cvt
                        )
                        del uab_var_soa
                        cr = _run_lmm_chunk(Uab_batch)
                        del Uab_batch
                else:
                    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
                    del UtG
                    cr = _run_lmm_chunk(Uab_batch)
                    del Uab_batch
                t_numpy_compute_total += time.perf_counter() - t_compute_start

                # Write results
                t_write_start = time.perf_counter()
                actual_len = chunk_end - chunk_start
                if streaming:
                    chunk_arrays = {
                        key: cr[key][:actual_len] for key in _RESULT_FIELDS[lmm_mode]
                    }
                    n_at_lmin_accum, n_at_lmax_accum = write_streaming_chunk(
                        writer,
                        lmm_mode,
                        snp_indices[write_offset : write_offset + actual_len],
                        snp_info,
                        filtered_afs[write_offset : write_offset + actual_len],
                        filtered_miss[write_offset : write_offset + actual_len],
                        chunk_arrays,
                        l_min,
                        l_max,
                        nan_counts,
                        n_at_lmin_accum,
                        n_at_lmax_accum,
                    )
                else:
                    s = slice(write_offset, write_offset + actual_len)
                    for key in arrays_out:
                        arrays_out[key][s] = cr[key][:actual_len]
                write_offset += actual_len
                t_result_write_total += time.perf_counter() - t_write_start
                del cr

    # Validate all results were written
    if write_offset != n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {write_offset} results,"
            f" expected {n_filtered}. This is an internal error — please report"
            f" this issue with your dataset dimensions."
        )

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_numpy", "after_all_chunks")

    # Diagnostics: use accumulated per-chunk counts for streaming,
    # post-loop arrays_out inspection for non-streaming.
    if streaming:
        for key, n_nan in nan_counts.items():
            logger.warning(
                f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
                "check for degenerate (constant) genotypes "
                "and kinship matrix quality"
            )
        log_lambda_boundary_warning(n_at_lmin_accum, n_at_lmax_accum, l_min, l_max)
    else:
        # NaN diagnostic: warn if any output arrays contain NaN results
        for key, arr in arrays_out.items():
            n_nan = int(np.sum(np.isnan(arr)))
            if n_nan > 0:
                logger.warning(
                    f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
                    "check for degenerate (constant) genotypes "
                    "and kinship matrix quality"
                )

        # Lambda boundary convergence diagnostics
        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode, arrays_out, l_min, l_max
        )
        log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        t_eigen = t_eigen_end - t_eigen_start
        accounted = (
            t_eigen + t_rotation_total + t_numpy_compute_total + t_result_write_total
        )
        logger.info("Timing breakdown:")
        logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
        logger.info(f"  UT@G rotation:       {t_rotation_total:.2f}s")
        logger.info(f"  NumPy compute:       {t_numpy_compute_total:.2f}s")
        logger.info(f"  Result write:        {t_result_write_total:.2f}s")
        logger.info("  ----")
        logger.info(f"  Accounted:           {accounted:.2f}s")
        logger.info(f"  Total:               {elapsed:.2f}s")
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    if streaming:
        return LmmRunResult(
            associations=[],
            pve=pve,
            pve_se=pve_se,
            n_tested=write_offset,
        )

    return LmmRunResult(
        associations=_build_results(
            lmm_mode, snp_indices, filtered_afs, filtered_miss, snp_info, arrays_out
        ),
        pve=pve,
        pve_se=pve_se,
    )
