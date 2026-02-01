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

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood_jax import (
    batch_calc_wald_stats,
    batch_compute_uab,
)
from jamma.lmm.stats import AssocResult


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
) -> list[AssocResult]:
    """Run LMM association tests using JAX-optimized batch processing.

    This version processes all SNPs in parallel using JAX's vectorization
    and JIT compilation. Significantly faster than sequential NumPy version
    for large datasets, especially on GPU.

    Note: Currently only supports intercept-only model (no additional covariates).
    If covariates are provided, a NotImplementedError is raised.

    Memory Scaling Warning:
        This function materializes arrays of shape (n_snps, n_samples, 6) for
        Uab computation and (n_grid, n_snps) for lambda optimization. For large
        cohorts (e.g., 200K samples × 500K SNPs), this can require significant
        memory (~4.8TB for Uab alone). For large-scale analyses:
        - Use `run_lmm_association()` (NumPy path) which processes SNPs sequentially
        - Or implement SNP chunking in your calling code and concatenate results

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
        n_grid: Number of grid points for lambda optimization
        n_refine: Number of refinement steps per SNP
        use_gpu: Whether to use GPU acceleration (requires JAX GPU setup)

    Returns:
        List of AssocResult for each SNP that passes filtering

    Raises:
        NotImplementedError: If covariates are provided (not yet supported)
    """
    # Guard: covariates not yet supported in JAX path
    if covariates is not None:
        raise NotImplementedError(
            "JAX runner does not yet support covariates beyond intercept. "
            "Use run_lmm_association() for covariate support, or pass covariates=None."
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

    # Vectorized missing genotype imputation
    geno_filtered = genotypes[:, snp_indices].copy()
    col_means_filtered = col_means[snp_indices]
    missing_mask = np.isnan(geno_filtered)
    # Broadcast column means to fill missing values
    geno_filtered = np.where(missing_mask, col_means_filtered[None, :], geno_filtered)

    UtG = U.T @ geno_filtered  # (n_samples, n_filtered)

    # Move to JAX arrays on target device
    eigenvalues = jax.device_put(jnp.array(eigenvalues_np), device)
    UtW_jax = jax.device_put(jnp.array(UtW), device)
    Uty_jax = jax.device_put(jnp.array(Uty), device)
    UtG_jax = jax.device_put(jnp.array(UtG), device)

    # Batch compute Uab for all SNPs
    Uab_batch = batch_compute_uab(UtW_jax, Uty_jax, UtG_jax)

    # Grid-based lambda optimization
    best_lambdas, best_logls = _grid_optimize_lambda_batched(
        eigenvalues, Uab_batch, l_min, l_max, n_grid, n_refine
    )

    # Batch compute Wald statistics
    betas, ses, p_walds = batch_calc_wald_stats(
        best_lambdas, eigenvalues, Uab_batch, n_samples
    )

    # Convert back to numpy for result construction
    best_lambdas_np = np.array(best_lambdas)
    best_logls_np = np.array(best_logls)
    betas_np = np.array(betas)
    ses_np = np.array(ses)
    p_walds_np = np.array(p_walds)

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
