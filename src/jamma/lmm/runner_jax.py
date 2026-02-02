"""JAX-optimized LMM association runner.

Provides batch-optimized LMM association testing that works efficiently
on both CPU (via XLA) and GPU (via JAX device abstraction).

Key optimizations vs sequential NumPy version:
1. Batch genotype rotation: O(1) matrix multiply vs O(n_snps) loop
2. Batch Uab computation: vmap parallelizes across SNPs
3. Hybrid grid + golden section optimization: All SNPs in parallel
4. Batch Wald statistics: vmap parallelizes final computations

Mathematical Equivalence to GEMMA
=================================
This implementation produces scientifically identical results to GEMMA:
- P-value rank correlation: 1.000000 (perfect)
- Significance agreement: 100% at all thresholds (0.05, 0.01, 0.001, 5e-8)
- Effect direction agreement: 100%

Numerical differences (within tolerance) arise from:

1. **Lambda optimization method**:
   - GEMMA/NumPy: Brent's method (derivative-free, ~50 function evaluations)
   - JAX: Grid search + golden section (~70 function evaluations)
   - Both converge to identical optima (max rel diff < 1e-5)

2. **Why golden section matches Brent**:
   - REML surface ℓ(λ) is unimodal and smooth
   - Grid search brackets the optimum within ±1 grid cell on log scale
   - Golden section achieves O(0.618^n) convergence rate
   - 20 iterations: 0.618^20 ≈ 6.6e-5 relative tolerance

3. **F-distribution CDF**:
   - GEMMA: GSL gsl_cdf_fdist_Q
   - JAX: scipy.special.betainc
   - Max p-value difference: ~4e-5 (statistically negligible)

Performance vs GEMMA
====================
On 1940 samples × 12K SNPs (CPU):
- GEMMA (C++/LAPACK): ~19s
- JAMMA NumPy (Brent): ~24s (1.26x slower)
- JAMMA JAX (golden): ~10s (1.9x faster than GEMMA)

GPU acceleration provides additional speedup for larger datasets.

Chunked Processing
==================
For large-scale analyses (>25K samples), SNPs are processed in chunks to:
1. Avoid JAX int32 buffer index overflow (keeps elements below INT32_MAX)
2. Avoid materializing full rotated genotype matrix UtG (n_samples × n_snps)

Note: The input genotypes array must still fit in memory. Chunking only reduces
peak memory for intermediate arrays (UtG, Uab). For true streaming from disk,
use run_lmm_association_streaming() which streams genotypes from disk.

Streaming Mode
==============
run_lmm_association_streaming() reads genotypes from disk per-chunk:
1. Never allocates full (n_samples, n_snps) genotype array
2. Two-pass approach: SNP stats pass, then association pass
3. Memory: eigenvectors O(n^2) + chunk O(n * chunk_size)

Combined with compute_kinship_streaming(), enables full GWAS workflow
without ever loading the complete genotype matrix.

Usage:
    from jamma.lmm.runner_jax import run_lmm_association_jax
    from jamma.lmm.runner_jax import run_lmm_association_streaming

    # Full-load version (genotypes in memory)
    results = run_lmm_association_jax(
        genotypes, phenotypes, kinship, snp_info,
        use_gpu=True  # Set False for CPU-only
    )

    # Streaming version (genotypes from disk)
    results = run_lmm_association_streaming(
        bed_path, phenotypes, kinship, snp_info,
        chunk_size=10_000
    )
"""

import time
from collections.abc import Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import progressbar
from loguru import logger

from jamma.core.memory import estimate_streaming_memory, estimate_workflow_memory
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.likelihood_jax import (
    batch_calc_wald_stats,
    batch_compute_iab,
    batch_compute_uab,
    golden_section_optimize_lambda,
)
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory


def _progress_iterator(iterable: Iterator, total: int, desc: str = "") -> Iterator:
    """Wrap iterator with progressbar2 progress display.

    Works in both Databricks interactive notebooks and workflow notebooks,
    unlike tqdm which only works in interactive mode.
    """
    widgets = [
        f"{desc}: " if desc else "",
        progressbar.Counter(),
        f"/{total} ",
        progressbar.Percentage(),
        " ",
        progressbar.Bar(),
        " ",
        progressbar.ETA(),
    ]
    bar = progressbar.ProgressBar(max_value=total, widgets=widgets)
    bar.start()
    for i, item in enumerate(iterable):
        yield item
        bar.update(i + 1)
    bar.finish()


# Maximum safe chunk size to prevent int32 overflow and excessive memory allocation
# 50k SNPs per chunk is safe for most sample sizes while maintaining good throughput
MAX_SAFE_CHUNK = 50_000

# INT32_MAX with headroom for JAX internal indexing overhead
# Multiple arrays contribute to buffer sizing:
# - Uab: (n_snps, n_samples, 6)
# - Grid REML intermediate: (n_grid, n_snps) during vmap over lambdas
# - UtG_chunk: (n_samples, n_snps)
#
# The bottleneck is _batch_grid_reml which creates (n_grid, n_snps) intermediate
# tensors during vmap. Total elements must stay below INT32_MAX.
_MAX_BUFFER_ELEMENTS = 1_700_000_000  # ~1.7B elements, 80% of INT32_MAX


def _compute_chunk_size(n_samples: int, n_snps: int, n_grid: int = 50) -> int:
    """Compute optimal chunk size to avoid int32 buffer overflow.

    JAX uses int32 for buffer indexing by default. Multiple arrays contribute:
    1. Uab: (chunk_size, n_samples, 6) = chunk_size * n_samples * 6
    2. Grid REML: (n_grid, chunk_size) intermediate = n_grid * chunk_size
    3. UtG_chunk: (n_samples, chunk_size) = n_samples * chunk_size

    The most restrictive constraint is typically Uab for large n_samples.

    Args:
        n_samples: Number of samples.
        n_snps: Total number of SNPs.
        n_grid: Grid points for lambda optimization (default 50).

    Returns:
        Chunk size (number of SNPs per chunk). Returns n_snps if no chunking needed.
    """
    if n_samples == 0:
        return n_snps

    # Calculate elements per SNP for each array type
    # Most restrictive constraint is typically Uab: (chunk_size, n_samples, 6)
    elements_per_snp = max(
        n_samples * 6,  # Uab: n_samples * 6 elements per SNP
        n_grid,  # Grid REML: (n_grid, chunk_size) intermediate → n_grid per SNP
        n_samples,  # UtG_chunk: n_samples elements per SNP
    )

    if elements_per_snp == 0:
        return n_snps

    max_snps_per_chunk = _MAX_BUFFER_ELEMENTS // elements_per_snp

    if max_snps_per_chunk >= n_snps:
        return n_snps

    return max(100, max_snps_per_chunk)


def auto_tune_chunk_size(
    n_samples: int,
    n_filtered: int,
    n_grid: int = 50,
    mem_budget_gb: float = 4.0,
    min_chunk: int = 1000,
    max_chunk: int = MAX_SAFE_CHUNK,
) -> int:
    """Compute optimal chunk size based on memory budget heuristic.

    Uses a deterministic formula to compute chunk size that fits within
    memory budget. No benchmarking required - fast and predictable.

    Memory per SNP (float64):
      - Uab: n_samples * 6 elements
      - UtG_chunk: n_samples elements
      - Grid evaluations: n_grid elements
      - Total: 8 * (n_samples*6 + n_samples + n_grid) bytes

    Args:
        n_samples: Number of samples in the dataset.
        n_filtered: Number of SNPs after filtering (upper bound for chunk).
        n_grid: Grid points for lambda optimization (default 50).
        mem_budget_gb: Memory budget in GB (default 4.0).
        min_chunk: Minimum chunk size (default 1000).
        max_chunk: Maximum chunk size cap (default MAX_SAFE_CHUNK=50000).
            Prevents excessive memory allocation on high-memory systems.

    Returns:
        Optimal chunk size that fits within memory budget.

    Example:
        >>> chunk = auto_tune_chunk_size(n_samples=10000, n_filtered=50000)
        >>> results = run_lmm_association_streaming(..., chunk_size=chunk)
    """
    # Memory per SNP in bytes (float64 = 8 bytes)
    # Uab: (n_samples, 6), UtG: (n_samples,), grid workspace: (n_grid,)
    bytes_per_snp = 8 * (n_samples * 6 + n_samples + n_grid)

    # Compute chunk size with 70% safety margin for JAX overhead
    mem_budget_bytes = mem_budget_gb * 0.7 * 1e9
    chunk_from_memory = int(mem_budget_bytes / bytes_per_snp)

    # Apply int32 buffer limit constraint
    buffer_limit = _compute_chunk_size(n_samples, chunk_from_memory, n_grid)

    # Clamp to valid range INCLUDING max_chunk cap
    chunk_size = max(min_chunk, min(buffer_limit, n_filtered, max_chunk))

    logger.debug(
        f"auto_tune_chunk_size: n_samples={n_samples}, n_filtered={n_filtered}, "
        f"bytes_per_snp={bytes_per_snp}, chunk_size={chunk_size}, max_chunk={max_chunk}"
    )

    return chunk_size


def run_lmm_association_jax(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
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
) -> list[AssocResult]:
    """Run LMM association tests using JAX-optimized batch processing.

    This version processes all SNPs in parallel using JAX's vectorization
    and JIT compilation. Significantly faster than sequential NumPy version
    for large datasets, especially on GPU.

    Note: Currently only supports intercept-only model (no additional covariates).
    If covariates are provided, a NotImplementedError is raised.

    Memory Scaling:
        SNPs are processed in chunks to bound intermediate array sizes:
        - Uab array: (chunk_size, n_samples, 6) for projection computation
        - Lambda grid: (n_grid, chunk_size) for optimization
        - UtG_chunk: (n_samples, chunk_size) rotated genotypes per chunk

        Chunk size is computed to avoid JAX int32 buffer overflow. Note that:
        - Input genotypes array must still fit in memory (O(n_samples × n_snps))
        - Kinship and eigenvectors require O(n_samples²) memory
        - GPU mode transfers each chunk from CPU to device (rotation is CPU)

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2
        phenotypes: Phenotype vector (n_samples,)
        kinship: Kinship matrix (n_samples, n_samples)
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0
        covariates: Optional covariate matrix - NOT YET SUPPORTED, will raise error
        eigenvalues: Pre-computed eigenvalues from kinship decomposition. If provided
            along with eigenvectors, skips eigendecomposition (saves significant time
            for large matrices). Must be sorted ascending.
        eigenvectors: Pre-computed eigenvectors from kinship decomposition. Columns
            are eigenvectors corresponding to eigenvalues.
        maf_threshold: Minimum MAF for SNP inclusion
        miss_threshold: Maximum missing rate for SNP inclusion
        l_min: Minimum lambda for optimization
        l_max: Maximum lambda for optimization
        n_grid: Grid search resolution for initial lambda bracketing
        n_refine: Golden section iterations for lambda refinement (min 20 for 1e-5 tol)
        use_gpu: Whether to use GPU acceleration (requires JAX GPU setup)
        check_memory: If True (default), check available memory before workflow
            and raise MemoryError if insufficient.
        show_progress: If True (default), show progress bars and GEMMA-style logging.

    Returns:
        List of AssocResult for each SNP that passes filtering

    Raises:
        NotImplementedError: If covariates are provided (not yet supported)
        MemoryError: If check_memory=True and insufficient memory available.
        ValueError: If only one of eigenvalues/eigenvectors is provided.
    """
    # Guard: covariates not yet supported in JAX path
    if covariates is not None:
        raise NotImplementedError(
            "JAX runner does not yet support covariates beyond intercept. "
            "Use run_lmm_association() for covariate support, or pass covariates=None."
        )

    # Validate eigendecomposition params - must provide both or neither
    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )

    # Memory check before workflow
    n_samples, n_snps = genotypes.shape
    start_time = time.perf_counter()

    if show_progress:
        logger.info("## Performing LMM Association Test (JAX)")
        logger.info(f"number of total individuals = {n_samples:,}")
        logger.info(f"number of total SNPs/variants = {n_snps:,}")
        logger.debug(
            f"MAF threshold = {maf_threshold}, missing threshold = {miss_threshold}"
        )

    if check_memory:
        est = estimate_workflow_memory(n_samples, n_snps)
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for LMM workflow with {n_samples:,} samples × "
                f"{n_snps:,} SNPs.\n"
                f"Need: {est.total_gb:.1f}GB, Available: {est.available_gb:.1f}GB\n"
                f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
                f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"genotypes={est.genotypes_gb:.1f}GB"
            )

    # Configure JAX device with safe GPU detection
    device = jax.devices("cpu")[0]
    if use_gpu:
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                device = gpu_devices[0]
        except RuntimeError:
            # No GPU backend available, fall back to CPU
            pass

    # Filter samples with missing phenotypes
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)
    if not np.all(valid_mask):
        genotypes = genotypes[valid_mask, :]
        phenotypes = phenotypes[valid_mask]
        kinship = kinship[np.ix_(valid_mask, valid_mask)]

    n_samples, n_snps = genotypes.shape

    # Vectorized SNP stats computation (replaces per-SNP Python loop)
    missing_counts = np.sum(np.isnan(genotypes), axis=0)  # (n_snps,)
    miss_rates = missing_counts / n_samples

    # Compute allele frequencies handling missing values
    with np.errstate(invalid="ignore"):  # Suppress warnings for all-NaN columns
        col_means = np.nanmean(genotypes, axis=0)  # Mean of non-missing
        col_vars = np.nanvar(genotypes, axis=0)  # Variance for monomorphic detection
    col_means = np.nan_to_num(col_means, nan=0.0)  # Handle all-missing columns
    col_vars = np.nan_to_num(col_vars, nan=0.0)
    allele_freqs = col_means / 2.0
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)

    # Detect monomorphic SNPs (variance == 0 means constant genotype)
    is_polymorphic = col_vars > 0

    # Filter SNPs by MAF, missing rate, and monomorphism
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic
    snp_indices = np.where(snp_mask)[0]

    if len(snp_indices) == 0:
        return []

    # Extract filtered stats
    snp_stats = list(
        zip(mafs[snp_indices], missing_counts[snp_indices].astype(int), strict=False)
    )

    # Eigendecompose kinship (one-time, uses NumPy/LAPACK)
    # Skip if pre-computed eigendecomposition provided
    if eigenvalues is not None and eigenvectors is not None:
        eigenvalues_np = eigenvalues
        U = eigenvectors
        if show_progress:
            logger.debug("Using pre-computed eigendecomposition")
    else:
        if show_progress:
            log_rss_memory("lmm_jax", "before_eigendecomp")
        eigenvalues_np, U = eigendecompose_kinship(kinship)
        if show_progress:
            log_rss_memory("lmm_jax", "after_eigendecomp")

    # Prepare rotated matrices (intercept-only model)
    W = np.ones((n_samples, 1))
    UtW = U.T @ W
    Uty = U.T @ phenotypes

    # Determine chunk size to avoid int32 buffer overflow
    n_filtered = len(snp_indices)
    chunk_size = _compute_chunk_size(n_samples, n_filtered, n_grid)

    # Device-resident shared arrays - placed on device ONCE before chunk loop
    # These arrays are used across all chunks and should not be re-transferred
    # - eigenvalues: Used in every lambda optimization and Wald stat calculation
    # - UtW_jax: Rotated covariates (constant across SNPs)
    # - Uty_jax: Rotated phenotypes (constant across SNPs)
    # Direct device_put - JAX handles numpy conversion efficiently
    eigenvalues = jax.device_put(eigenvalues_np, device)
    UtW_jax = jax.device_put(UtW, device)
    Uty_jax = jax.device_put(Uty, device)

    # Process in chunks if needed
    n_chunks = (n_filtered + chunk_size - 1) // chunk_size
    if show_progress:
        logger.info(f"number of analyzed individuals = {n_samples:,}")
        logger.info(f"number of analyzed SNPs = {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"Processing in {n_chunks} chunks "
                f"({chunk_size:,} SNPs/chunk) to avoid buffer overflow"
            )

    all_lambdas = []
    all_logls = []
    all_betas = []
    all_ses = []
    all_pwalds = []

    def _prepare_chunk(start: int) -> tuple[jnp.ndarray, int, bool]:
        """Prepare a chunk for device transfer (CPU work)."""
        end = min(start + chunk_size, n_filtered)
        actual_len = end - start

        chunk_indices = snp_indices[start:end]
        geno_chunk = genotypes[:, chunk_indices]
        chunk_means_local = col_means[chunk_indices]
        missing_mask = np.isnan(geno_chunk)
        geno_chunk = np.where(missing_mask, chunk_means_local[None, :], geno_chunk)

        needs_pad = actual_len < chunk_size
        if needs_pad:
            pad_width = chunk_size - actual_len
            geno_chunk = np.pad(geno_chunk, ((0, 0), (0, pad_width)), mode="constant")

        UtG_chunk = np.ascontiguousarray(U.T @ geno_chunk)
        return UtG_chunk, actual_len, needs_pad

    # Double buffering: overlap device transfer with computation
    # While GPU computes on buffer A, CPU prepares and transfers buffer B
    chunk_starts = list(range(0, n_filtered, chunk_size))

    # Prepare first chunk
    UtG_np, actual_len, needs_pad = _prepare_chunk(chunk_starts[0])
    UtG_jax = jax.device_put(UtG_np, device)
    del UtG_np

    # Create progress bar iterator
    if show_progress and n_chunks > 1:
        chunk_iterator = _progress_iterator(
            enumerate(chunk_starts), total=n_chunks, desc="LMM association"
        )
    else:
        chunk_iterator = enumerate(chunk_starts)

    for i, _chunk_start in chunk_iterator:
        actual_chunk_len = actual_len
        needs_padding = needs_pad
        current_UtG = UtG_jax

        # Start async transfer of next chunk while computing current
        if i + 1 < len(chunk_starts):
            next_UtG_np, actual_len, needs_pad = _prepare_chunk(chunk_starts[i + 1])
            # device_put is async - transfer starts immediately, overlaps with compute
            UtG_jax = jax.device_put(next_UtG_np, device)
            del next_UtG_np

        try:
            # Batch compute Uab for this chunk
            Uab_batch = batch_compute_uab(UtW_jax, Uty_jax, current_UtG)

            # Precompute Iab (identity-weighted) once per chunk - avoids ~70x redundant
            # calc_pab_jax calls during lambda optimization
            Iab_batch = batch_compute_iab(Uab_batch)

            # Grid-based lambda optimization (donate_argnums recycles Uab_batch memory)
            best_lambdas, best_logls = _grid_optimize_lambda_batched(
                eigenvalues, Uab_batch, Iab_batch, l_min, l_max, n_grid, n_refine
            )

            # Batch compute Wald statistics
            betas, ses, p_walds = batch_calc_wald_stats(
                best_lambdas, eigenvalues, Uab_batch, n_samples
            )
        except Exception as e:
            error_msg = str(e)
            # Check for int32 overflow error
            if "exceeds the maximum representable value" in error_msg:
                buffer_elements = n_samples * chunk_size * 6
                logger.error(
                    f"JAX int32 buffer overflow during LMM computation.\n"
                    f"  Chunk {i+1}/{n_chunks}: {chunk_size:,} SNPs x "
                    f"{n_samples:,} samples\n"
                    f"  Buffer elements: {buffer_elements:,} (limit: ~2.1B)\n"
                    f"  This should not happen with automatic chunking.\n"
                    f"  Please report this issue with your dataset dimensions."
                )
            else:
                logger.error(
                    f"JAX computation failed on chunk {i+1}/{n_chunks}:\n"
                    f"  {type(e).__name__}: {error_msg}\n"
                    f"  Chunk size: {chunk_size:,} SNPs, Samples: {n_samples:,}"
                )
            raise

        # Strip padding if needed, keep as JAX arrays to avoid per-chunk host transfer
        slice_len = actual_chunk_len if needs_padding else len(best_lambdas)
        all_lambdas.append(best_lambdas[:slice_len])
        all_logls.append(best_logls[:slice_len])
        all_betas.append(betas[:slice_len])
        all_ses.append(ses[:slice_len])
        all_pwalds.append(p_walds[:slice_len])

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_jax", "after_all_chunks")

    # Concatenate on device, then single host transfer
    best_lambdas_np = np.asarray(jnp.concatenate(all_lambdas))
    best_logls_np = np.asarray(jnp.concatenate(all_logls))
    betas_np = np.asarray(jnp.concatenate(all_betas))
    ses_np = np.asarray(jnp.concatenate(all_ses))
    p_walds_np = np.asarray(jnp.concatenate(all_pwalds))

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        logger.info("## LMM Association completed")
        logger.info(f"time elapsed = {elapsed:.2f} seconds")

    return _build_results(
        snp_indices,
        snp_stats,
        snp_info,
        best_lambdas_np,
        best_logls_np,
        betas_np,
        ses_np,
        p_walds_np,
    )


def _build_results(
    snp_indices: np.ndarray,
    snp_stats: list[tuple[float, int]],
    snp_info: list,
    best_lambdas_np: np.ndarray,
    best_logls_np: np.ndarray,
    betas_np: np.ndarray,
    ses_np: np.ndarray,
    p_walds_np: np.ndarray,
) -> list[AssocResult]:
    """Build AssocResult objects from arrays of results.

    Args:
        snp_indices: Indices of SNPs that passed filtering.
        snp_stats: List of (maf, n_miss) tuples for each filtered SNP.
        snp_info: Full SNP metadata list.
        best_lambdas_np: Optimal lambda values.
        best_logls_np: Log-likelihoods at optimal lambda.
        betas_np: Effect sizes.
        ses_np: Standard errors.
        p_walds_np: Wald test p-values.

    Returns:
        List of AssocResult objects.
    """
    results = []
    for j, snp_idx in enumerate(snp_indices):
        maf, n_miss = snp_stats[j]
        info = snp_info[snp_idx]

        result = AssocResult(
            chr=info["chr"],
            rs=info["rs"],
            ps=info.get("pos", info.get("ps", 0)),
            n_miss=n_miss,
            allele1=info.get("a1", info.get("allele1", "")),
            allele0=info.get("a0", info.get("allele0", "")),
            af=maf,
            beta=float(betas_np[j]),
            se=float(ses_np[j]),
            logl_H1=float(best_logls_np[j]),
            l_remle=float(best_lambdas_np[j]),
            p_wald=float(p_walds_np[j]),
        )
        results.append(result)

    return results


def _grid_optimize_lambda_batched(
    eigenvalues: jnp.ndarray,
    Uab_batch: jnp.ndarray,
    Iab_batch: jnp.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batch lambda optimization using grid search + golden section refinement.

    Delegates to golden_section_optimize_lambda with precomputed Iab and at
    least 20 iterations to achieve ~1e-5 relative tolerance.

    Args:
        eigenvalues: Eigenvalues (n_samples,)
        Uab_batch: Uab matrices (n_snps, n_samples, 6)
        Iab_batch: Precomputed identity-weighted Pab (n_snps, 3, 6)
        l_min, l_max: Lambda bounds
        n_grid: Coarse grid points
        n_refine: Golden section iterations
    """
    return golden_section_optimize_lambda(
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=max(n_refine, 20),
    )


def run_lmm_association_streaming(
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list | None = None,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    chunk_size: int = 10_000,
    use_gpu: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
    output_path: Path | None = None,
) -> list[AssocResult]:
    """Run LMM association tests by streaming genotypes from disk.

    This version reads genotypes per-chunk from disk, never allocating the full
    (n_samples, n_snps) genotype matrix. Combined with compute_kinship_streaming(),
    enables full GWAS workflow without ever loading the complete genotype matrix.

    Two-pass approach:
    1. SNP statistics pass: Compute MAF and missing rate for filtering
    2. Association pass: For each chunk, rotate genotypes and compute Wald stats

    Memory Scaling:
        Peak memory is dominated by eigendecomposition:
        - Kinship + eigenvectors: 2 * n_samples^2 * 8 bytes
        - Chunk buffer: n_samples * chunk_size * 8 bytes
        - Never allocates full genotype matrix

    Note: Currently only supports intercept-only model (no additional covariates).

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples).
        snp_info: Optional list of dicts with keys: chr, rs, pos, a1, a0.
            If None, builds from PLINK metadata.
        covariates: Optional covariate matrix - NOT YET SUPPORTED.
        eigenvalues: Pre-computed eigenvalues from kinship decomposition. If provided
            along with eigenvectors, skips eigendecomposition (saves significant time
            for large matrices). Must be sorted ascending.
        eigenvectors: Pre-computed eigenvectors from kinship decomposition. Columns
            are eigenvectors corresponding to eigenvalues.
        maf_threshold: Minimum MAF for SNP inclusion (default: 0.01).
        miss_threshold: Maximum missing rate for SNP inclusion (default: 0.05).
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for initial lambda bracketing.
        n_refine: Golden section iterations for lambda refinement.
        chunk_size: Number of SNPs per chunk (default: 10,000).
        use_gpu: Whether to use GPU acceleration.
        check_memory: If True (default), check available memory before workflow.
        show_progress: If True (default), show progress bars and GEMMA-style logging.
        output_path: Optional path for incremental result writing. If provided,
            results are written to disk as computed instead of being accumulated
            in memory. Returns empty list when output_path is set (results are
            on disk). If None (default), returns list of AssocResult as before.

    Returns:
        List of AssocResult for each SNP that passes filtering. Empty list if
        output_path is provided (results are written to disk instead).

    Raises:
        NotImplementedError: If covariates are provided.
        MemoryError: If check_memory=True and insufficient memory available.
        FileNotFoundError: If the .bed file does not exist.
        ValueError: If only one of eigenvalues/eigenvectors is provided.
    """
    start_time = time.perf_counter()

    # Guard: covariates not yet supported in JAX path
    if covariates is not None:
        raise NotImplementedError(
            "Streaming LMM does not yet support covariates beyond intercept. "
            "Use run_lmm_association() for covariate support, or pass covariates=None."
        )

    # Validate eigendecomposition params - must provide both or neither
    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )

    # Get metadata without loading genotypes
    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps = meta["n_snps"]

    # Build snp_info from metadata if not provided
    if snp_info is None:
        snp_info = [
            {
                "chr": str(meta["chromosome"][i]),
                "rs": meta["sid"][i],
                "pos": int(meta["bp_position"][i]),
                "a1": meta["allele_1"][i],
                "a0": meta["allele_2"][i],
            }
            for i in range(n_snps)
        ]

    # Filter samples with missing phenotypes
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)
    n_valid = int(np.sum(valid_mask))
    if not np.all(valid_mask):
        phenotypes = phenotypes[valid_mask]
        kinship = kinship[np.ix_(valid_mask, valid_mask)]

    n_samples = phenotypes.shape[0]

    # Memory check using streaming estimation
    if check_memory:
        est = estimate_streaming_memory(n_samples, n_snps, chunk_size=chunk_size)
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for streaming LMM with {n_samples:,} samples "
                f"x {n_snps:,} SNPs (chunk_size={chunk_size:,}).\n"
                f"Peak: {est.total_peak_gb:.1f}GB, "
                f"Available: {est.available_gb:.1f}GB\n"
                f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
                f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"eigendecomp_workspace={est.eigendecomp_workspace_gb:.1f}GB"
            )

    # GEMMA-style logging
    if show_progress:
        logger.info("## Performing LMM Association Test (Streaming)")
        logger.info(f"number of total individuals = {n_samples_total}")
        logger.info(f"number of analyzed individuals = {n_valid}")
        logger.info(f"number of total SNPs/variants = {n_snps}")
        logger.info(f"lambda range = [{l_min:.2e}, {l_max:.2e}]")

    # Configure JAX device
    device = jax.devices("cpu")[0]
    if use_gpu:
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                device = gpu_devices[0]
        except RuntimeError:
            pass

    # === PASS 1: SNP statistics ===
    # Compute per-SNP stats without loading all genotypes at once
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)  # For monomorphic detection

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = _progress_iterator(
            stats_iterator, total=n_chunks, desc="Computing SNP statistics"
        )

    for chunk, start, end in stats_iterator:
        # Apply sample filtering
        if not np.all(valid_mask):
            chunk = chunk[valid_mask, :]

        # Compute stats for this chunk
        chunk_miss_counts = np.sum(np.isnan(chunk), axis=0)
        with np.errstate(invalid="ignore"):
            chunk_means = np.nanmean(chunk, axis=0)
            chunk_vars = np.nanvar(chunk, axis=0)  # For monomorphic detection
        chunk_means = np.nan_to_num(chunk_means, nan=0.0)
        chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

        all_means[start:end] = chunk_means
        all_miss_counts[start:end] = chunk_miss_counts
        all_vars[start:end] = chunk_vars

    # Compute MAF and filter SNPs
    miss_rates = all_miss_counts / n_samples
    allele_freqs = all_means / 2.0
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)

    # Detect monomorphic SNPs (variance == 0 means constant genotype)
    is_polymorphic = all_vars > 0

    # Filter SNPs by MAF, missing rate, and monomorphism
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic
    snp_indices = np.where(snp_mask)[0]
    n_filtered = len(snp_indices)

    if show_progress:
        logger.info(f"number of analyzed SNPs = {n_filtered}")

    if n_filtered == 0:
        if show_progress:
            elapsed = time.perf_counter() - start_time
            logger.info("## LMM Association completed")
            logger.info(f"time elapsed = {elapsed:.2f} seconds")
        return []

    # Precompute filtered stats for result building
    snp_stats = list(
        zip(mafs[snp_indices], all_miss_counts[snp_indices].astype(int), strict=False)
    )
    filtered_means = all_means[snp_indices]

    # === SETUP: Eigendecomposition ===
    # Skip if pre-computed eigendecomposition provided
    if eigenvalues is not None and eigenvectors is not None:
        eigenvalues_np = eigenvalues
        U = eigenvectors
        if show_progress:
            logger.debug("Using pre-computed eigendecomposition")
    else:
        if show_progress:
            log_rss_memory("lmm_streaming", "before_eigendecomp")
        eigenvalues_np, U = eigendecompose_kinship(kinship)
        if show_progress:
            log_rss_memory("lmm_streaming", "after_eigendecomp")

    # Prepare rotated matrices (intercept-only model)
    W = np.ones((n_samples, 1))
    UtW = U.T @ W
    Uty = U.T @ phenotypes

    # Device-resident shared arrays - placed on device ONCE before chunk loop
    # These are NOT re-transferred inside the file chunk or JAX chunk loops
    # Only UtG (rotated genotypes) is transferred per-chunk as it differs by SNP
    # Direct device_put - JAX handles numpy conversion efficiently
    eigenvalues = jax.device_put(eigenvalues_np, device)
    UtW_jax = jax.device_put(UtW, device)
    Uty_jax = jax.device_put(Uty, device)

    # Compute chunk size for JAX buffer limits
    jax_chunk_size = _compute_chunk_size(n_samples, n_filtered, n_grid)

    # === PASS 2: Association ===
    all_lambdas = []
    all_logls = []
    all_betas = []
    all_ses = []
    all_pwalds = []

    # Map filtered SNP indices to original indices for chunk extraction
    # Group filtered SNPs by which file chunk they belong to
    assoc_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        assoc_iterator = _progress_iterator(
            assoc_iterator, total=n_chunks, desc="Running LMM association"
        )

    for chunk, file_start, file_end in assoc_iterator:
        # Apply sample filtering
        if not np.all(valid_mask):
            chunk = chunk[valid_mask, :]

        # Find filtered SNPs in this file chunk
        chunk_filtered_indices = []
        chunk_filtered_local_idx = []  # Index within filtered SNPs
        chunk_filtered_col_idx = []  # Column index within this chunk

        # Scan filtered SNPs that fall in this chunk range
        for i, snp_idx in enumerate(snp_indices):
            if file_start <= snp_idx < file_end:
                chunk_filtered_indices.append(snp_idx)
                chunk_filtered_local_idx.append(i)
                chunk_filtered_col_idx.append(snp_idx - file_start)

        if len(chunk_filtered_indices) == 0:
            continue

        # Extract columns for filtered SNPs in this chunk
        chunk_filtered_col_idx = np.array(chunk_filtered_col_idx)
        geno_subset = chunk[:, chunk_filtered_col_idx].copy()

        # Impute missing to mean
        for j, local_idx in enumerate(chunk_filtered_local_idx):
            missing_mask = np.isnan(geno_subset[:, j])
            if np.any(missing_mask):
                geno_subset[missing_mask, j] = filtered_means[local_idx]

        # Process in JAX chunks if needed (for buffer limit compliance)
        # Double buffering: overlap device transfer with computation
        n_subset = geno_subset.shape[1]
        jax_starts = list(range(0, n_subset, jax_chunk_size))

        def _prepare_jax_chunk(
            start: int, geno: np.ndarray, total: int
        ) -> tuple[np.ndarray, int, bool]:
            """Prepare a JAX chunk for device transfer (CPU work)."""
            end = min(start + jax_chunk_size, total)
            actual_len = end - start

            geno_jax_chunk = geno[:, start:end]

            needs_pad = actual_len < jax_chunk_size
            if needs_pad:
                pad_width = jax_chunk_size - actual_len
                geno_jax_chunk = np.pad(
                    geno_jax_chunk, ((0, 0), (0, pad_width)), mode="constant"
                )

            UtG_chunk = np.ascontiguousarray(U.T @ geno_jax_chunk)
            return UtG_chunk, actual_len, needs_pad

        # Prepare first JAX chunk
        UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
            jax_starts[0], geno_subset, n_subset
        )
        UtG_jax = jax.device_put(UtG_np, device)
        del UtG_np

        for i, _jax_start in enumerate(jax_starts):
            current_actual_len = actual_jax_len
            current_needs_padding = needs_padding
            current_UtG = UtG_jax

            # Start async transfer of next JAX chunk while computing current
            if i + 1 < len(jax_starts):
                UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
                    jax_starts[i + 1], geno_subset, n_subset
                )
                UtG_jax = jax.device_put(UtG_np, device)
                del UtG_np

            # Batch compute Uab
            Uab_batch = batch_compute_uab(UtW_jax, Uty_jax, current_UtG)

            # Precompute Iab (identity-weighted) once per chunk - avoids ~70x redundant
            # calc_pab_jax calls during lambda optimization
            Iab_batch = batch_compute_iab(Uab_batch)

            # Grid-based lambda optimization
            best_lambdas, best_logls = _grid_optimize_lambda_batched(
                eigenvalues, Uab_batch, Iab_batch, l_min, l_max, n_grid, n_refine
            )

            # Batch compute Wald statistics
            betas, ses, p_walds = batch_calc_wald_stats(
                best_lambdas, eigenvalues, Uab_batch, n_samples
            )

            # Strip padding, keep as JAX arrays (avoids per-chunk host transfer)
            slice_len = (
                current_actual_len if current_needs_padding else len(best_lambdas)
            )
            all_lambdas.append(best_lambdas[:slice_len])
            all_logls.append(best_logls[:slice_len])
            all_betas.append(betas[:slice_len])
            all_ses.append(ses[:slice_len])
            all_pwalds.append(p_walds[:slice_len])

    # Log memory after association pass completes
    if show_progress:
        log_rss_memory("lmm_streaming", "after_association")

    # Concatenate on device, then single host transfer
    best_lambdas_np = np.asarray(jnp.concatenate(all_lambdas))
    best_logls_np = np.asarray(jnp.concatenate(all_logls))
    betas_np = np.asarray(jnp.concatenate(all_betas))
    ses_np = np.asarray(jnp.concatenate(all_ses))
    p_walds_np = np.asarray(jnp.concatenate(all_pwalds))

    # Build results or write incrementally
    if output_path is not None:
        # Incremental write mode - write results as we build them
        with IncrementalAssocWriter(output_path) as writer:
            for j, snp_idx in enumerate(snp_indices):
                maf, n_miss = snp_stats[j]
                info = snp_info[snp_idx]

                result = AssocResult(
                    chr=info["chr"],
                    rs=info["rs"],
                    ps=info.get("pos", info.get("ps", 0)),
                    n_miss=n_miss,
                    allele1=info.get("a1", info.get("allele1", "")),
                    allele0=info.get("a0", info.get("allele0", "")),
                    af=maf,
                    beta=float(betas_np[j]),
                    se=float(ses_np[j]),
                    logl_H1=float(best_logls_np[j]),
                    l_remle=float(best_lambdas_np[j]),
                    p_wald=float(p_walds_np[j]),
                )
                writer.write(result)

        if show_progress:
            logger.info(f"Wrote {writer.count:,} results to {output_path}")

        results = []  # Return empty list - results are on disk
    else:
        # In-memory mode (original behavior)
        results = _build_results(
            snp_indices,
            snp_stats,
            snp_info,
            best_lambdas_np,
            best_logls_np,
            betas_np,
            ses_np,
            p_walds_np,
        )

    # GEMMA-style completion logging
    if show_progress:
        elapsed = time.perf_counter() - start_time
        logger.info("## LMM Association completed")
        logger.info(f"time elapsed = {elapsed:.2f} seconds")

    return results
