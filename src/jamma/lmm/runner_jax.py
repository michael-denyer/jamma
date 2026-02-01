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
use run_lmm_association() (NumPy path) with external chunking of genotypes.

Usage:
    from jamma.lmm.runner_jax import run_lmm_association_jax

    results = run_lmm_association_jax(
        genotypes, phenotypes, kinship, snp_info,
        use_gpu=True  # Set False for CPU-only
    )
"""

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jamma.core.memory import estimate_workflow_memory
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood_jax import (
    batch_calc_wald_stats,
    batch_compute_uab,
)
from jamma.lmm.stats import AssocResult

# INT32_MAX with 20% headroom for JAX internal indexing overhead
# Uab has shape (n_snps, n_samples, 6), so total elements = n_snps * n_samples * 6
# At 50K samples: max ~5,700 SNPs/chunk (2 chunks for 10K SNPs)
# At 100K samples: max ~2,800 SNPs/chunk (4 chunks for 10K SNPs)
_MAX_BUFFER_ELEMENTS = 1_700_000_000  # ~1.7B elements, 80% of INT32_MAX


def _compute_chunk_size(n_samples: int, n_snps: int) -> int:
    """Compute optimal chunk size to avoid int32 buffer overflow.

    JAX uses int32 for buffer indexing by default. The Uab array has shape
    (n_snps, n_samples, 6), so we need n_snps * n_samples * 6 < INT32_MAX.

    Args:
        n_samples: Number of samples.
        n_snps: Total number of SNPs.

    Returns:
        Chunk size (number of SNPs per chunk). Returns n_snps if no chunking needed.
    """
    # Elements per SNP in Uab: n_samples * 6
    elements_per_snp = n_samples * 6

    if elements_per_snp == 0:
        return n_snps

    max_snps_per_chunk = _MAX_BUFFER_ELEMENTS // elements_per_snp

    if max_snps_per_chunk >= n_snps:
        return n_snps  # No chunking needed

    # Use chunk size that divides work reasonably
    return max(1, max_snps_per_chunk)


def run_lmm_association_jax(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list,
    covariates: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    use_gpu: bool = False,
    check_memory: bool = True,
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
        maf_threshold: Minimum MAF for SNP inclusion
        miss_threshold: Maximum missing rate for SNP inclusion
        l_min: Minimum lambda for optimization
        l_max: Maximum lambda for optimization
        n_grid: Grid search resolution for initial lambda bracketing
        n_refine: Golden section iterations for lambda refinement (min 20 for 1e-5 tol)
        use_gpu: Whether to use GPU acceleration (requires JAX GPU setup)
        check_memory: If True (default), check available memory before workflow
            and raise MemoryError if insufficient.

    Returns:
        List of AssocResult for each SNP that passes filtering

    Raises:
        NotImplementedError: If covariates are provided (not yet supported)
        MemoryError: If check_memory=True and insufficient memory available.
    """
    # Guard: covariates not yet supported in JAX path
    if covariates is not None:
        raise NotImplementedError(
            "JAX runner does not yet support covariates beyond intercept. "
            "Use run_lmm_association() for covariate support, or pass covariates=None."
        )

    # Memory check before workflow
    n_samples, n_snps = genotypes.shape
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
    col_means = np.nan_to_num(col_means, nan=0.0)  # Handle all-missing columns
    allele_freqs = col_means / 2.0
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)

    # Filter SNPs by MAF and missing rate
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold)
    snp_indices = np.where(snp_mask)[0]

    if len(snp_indices) == 0:
        return []

    # Extract filtered stats
    snp_stats = list(
        zip(mafs[snp_indices], missing_counts[snp_indices].astype(int), strict=False)
    )

    # Eigendecompose kinship (one-time, uses NumPy/LAPACK)
    eigenvalues_np, U = eigendecompose_kinship(kinship)

    # Prepare rotated matrices (intercept-only model)
    W = np.ones((n_samples, 1))
    UtW = U.T @ W
    Uty = U.T @ phenotypes

    # Determine chunk size to avoid int32 buffer overflow
    n_filtered = len(snp_indices)
    chunk_size = _compute_chunk_size(n_samples, n_filtered)

    # Move shared data to JAX arrays on target device
    eigenvalues = jax.device_put(jnp.array(eigenvalues_np), device)
    UtW_jax = jax.device_put(jnp.array(UtW), device)
    Uty_jax = jax.device_put(jnp.array(Uty), device)

    # Process in chunks if needed
    if chunk_size < n_filtered:
        n_chunks = (n_filtered + chunk_size - 1) // chunk_size
        logger.info(
            f"Processing {n_filtered:,} SNPs in {n_chunks} chunks "
            f"({chunk_size:,} SNPs/chunk) to avoid buffer overflow"
        )

    all_lambdas = []
    all_logls = []
    all_betas = []
    all_ses = []
    all_pwalds = []

    for chunk_start in range(0, n_filtered, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_filtered)
        actual_chunk_len = chunk_end - chunk_start

        # Extract, impute, and rotate genotypes for this chunk only
        # This avoids materializing the full (n_samples, n_filtered) matrix
        chunk_indices = snp_indices[chunk_start:chunk_end]
        geno_chunk = genotypes[:, chunk_indices].copy()
        chunk_means = col_means[chunk_indices]
        missing_mask = np.isnan(geno_chunk)
        geno_chunk = np.where(missing_mask, chunk_means[None, :], geno_chunk)

        # Pad last chunk to fixed size to avoid JAX recompilation
        # JAX traces functions for each unique shape; padding keeps shapes constant
        needs_padding = actual_chunk_len < chunk_size
        if needs_padding:
            pad_width = chunk_size - actual_chunk_len
            geno_chunk = np.pad(geno_chunk, ((0, 0), (0, pad_width)), mode="constant")

        # Rotate genotypes for this chunk
        UtG_chunk = U.T @ geno_chunk  # (n_samples, chunk_size)
        UtG_jax = jax.device_put(jnp.array(UtG_chunk), device)

        # Free NumPy arrays to reduce memory pressure
        del geno_chunk, UtG_chunk

        # Batch compute Uab for this chunk
        Uab_batch = batch_compute_uab(UtW_jax, Uty_jax, UtG_jax)

        # Grid-based lambda optimization (donate_argnums recycles Uab_batch memory)
        best_lambdas, best_logls = _grid_optimize_lambda_batched(
            eigenvalues, Uab_batch, l_min, l_max, n_grid, n_refine
        )

        # Batch compute Wald statistics
        betas, ses, p_walds = batch_calc_wald_stats(
            best_lambdas, eigenvalues, Uab_batch, n_samples
        )

        # Strip padding from results if needed
        if needs_padding:
            best_lambdas = best_lambdas[:actual_chunk_len]
            best_logls = best_logls[:actual_chunk_len]
            betas = betas[:actual_chunk_len]
            ses = ses[:actual_chunk_len]
            p_walds = p_walds[:actual_chunk_len]

        # Collect results
        all_lambdas.append(np.array(best_lambdas))
        all_logls.append(np.array(best_logls))
        all_betas.append(np.array(betas))
        all_ses.append(np.array(ses))
        all_pwalds.append(np.array(p_walds))

    # Concatenate chunk results
    best_lambdas_np = np.concatenate(all_lambdas)
    best_logls_np = np.concatenate(all_logls)
    betas_np = np.concatenate(all_betas)
    ses_np = np.concatenate(all_ses)
    p_walds_np = np.concatenate(all_pwalds)

    # Build results
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
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batch lambda optimization using grid search + golden section refinement.

    Delegates to golden_section_optimize_lambda with at least 20 iterations
    to achieve ~1e-5 relative tolerance.
    """
    from jamma.lmm.likelihood_jax import golden_section_optimize_lambda

    return golden_section_optimize_lambda(
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=max(n_refine, 20),
    )
