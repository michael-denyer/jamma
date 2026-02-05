"""JAX kinship matrix computation.

This module provides the main kinship matrix computation function,
implementing GEMMA's centered relatedness matrix algorithm (-gk 1 mode).

The kinship matrix K is computed as:
    K = (1/p) * X_c @ X_c.T

where X_c is the centered genotype matrix with missing values imputed
to per-SNP mean, and p is the number of SNPs.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import progressbar
from jax import jit
from loguru import logger

from jamma.core import configure_jax
from jamma.core.memory import (
    check_memory_available,
    estimate_eigendecomp_memory,
    estimate_streaming_memory,
    log_memory_snapshot,
)
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.kinship.missing import impute_and_center


def _progress_iterator(iterable: Iterator, total: int, desc: str = "") -> Iterator:
    """Wrap iterator with progressbar2 progress display."""
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


# Ensure 64-bit precision for GEMMA equivalence
configure_jax()


@jit
def _accumulate_kinship(K: jnp.ndarray, X_centered: jnp.ndarray) -> jnp.ndarray:
    """Accumulate kinship contribution from centered SNP batch.

    Args:
        K: Current kinship matrix accumulator (n_samples, n_samples)
        X_centered: Centered genotype batch (n_samples, batch_snps)

    Returns:
        Updated kinship matrix with batch contribution added.
    """
    return K + jnp.matmul(X_centered, X_centered.T)


def _filter_snps(
    genotypes: np.ndarray,
    maf_threshold: float,
    miss_threshold: float,
) -> tuple[np.ndarray, int, int]:
    """Filter SNPs by MAF, missing rate, and monomorphism.

    Monomorphic SNPs (variance == 0) are always filtered to match GEMMA.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
        maf_threshold: Minimum MAF for inclusion (0 to disable MAF filter only).
        miss_threshold: Maximum missing rate for inclusion (1.0 to disable).

    Returns:
        Tuple of (filtered_genotypes, n_filtered, n_original).
    """
    n_samples, n_snps = genotypes.shape

    # Compute per-SNP statistics
    missing_counts = np.sum(np.isnan(genotypes), axis=0)
    miss_rates = missing_counts / n_samples

    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(genotypes, axis=0)
        col_vars = np.nanvar(genotypes, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    col_vars = np.nan_to_num(col_vars, nan=0.0)
    allele_freqs = col_means / 2.0
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)

    # Monomorphic SNPs always filtered (GEMMA behavior)
    is_polymorphic = col_vars > 0

    # Combined filter: MAF >= threshold AND miss_rate <= threshold AND polymorphic
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic
    n_filtered = int(np.sum(snp_mask))

    if n_filtered == 0:
        return genotypes[:, :0], 0, n_snps  # Empty array

    return genotypes[:, snp_mask], n_filtered, n_snps


def compute_centered_kinship(
    genotypes: np.ndarray,
    batch_size: int = 10000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
) -> np.ndarray:
    """Compute centered relatedness matrix (GEMMA -gk 1).

    Implements: K = (1/p) * X_c @ X_c.T
    where X_c is centered with missing values imputed to SNP mean.

    GEMMA's PlinkKin algorithm:
    1. Filter SNPs by MAF, missing rate, and monomorphism
    2. For each SNP batch: impute missing to mean, center
    3. Accumulate K += X_batch @ X_batch.T
    4. Scale K /= n_filtered_snps

    Note: Monomorphic SNPs (constant genotype) are always excluded to match GEMMA.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
            Values are typically 0, 1, or 2 representing minor allele counts.
        batch_size: SNPs per batch (default 10000, matches GEMMA).
            Batching prevents memory issues with large SNP counts.
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation
            and raise MemoryError if insufficient.

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        ValueError: If no SNPs pass filtering.

    Example:
        >>> import numpy as np
        >>> X = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0]], dtype=np.float64)
        >>> K = compute_centered_kinship(X, maf_threshold=0.01)
        >>> K.shape
        (3, 3)
        >>> np.allclose(K, K.T)  # Symmetric
        True
    """
    n_samples, n_snps_original = genotypes.shape

    # Filter SNPs by MAF, missing rate, and monomorphism
    genotypes_filtered, n_snps, n_original = _filter_snps(
        genotypes, maf_threshold, miss_threshold
    )

    if n_snps == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_original}"
        )

    if n_snps < n_original:
        n_removed = n_original - n_snps
        logger.info(
            f"Kinship filtering: {n_snps:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    # Memory check before allocation
    # Check against full pipeline peak (eigendecomp) since it always follows kinship
    if check_memory:
        eigendecomp_peak_gb = estimate_eigendecomp_memory(n_samples)
        check_memory_available(
            eigendecomp_peak_gb,
            operation=f"GWAS pipeline (eigendecomp peak: {eigendecomp_peak_gb:.1f}GB)",
        )

    # Log memory state before kinship allocation for debugging OOM
    log_memory_snapshot(f"before_kinship_{n_samples}samples")

    # Convert to JAX array
    X = jnp.array(genotypes_filtered, dtype=jnp.float64)

    # Initialize kinship accumulator
    K = jnp.zeros((n_samples, n_samples), dtype=jnp.float64)

    # Process SNPs in batches
    for start in range(0, n_snps, batch_size):
        end = min(start + batch_size, n_snps)
        X_batch = X[:, start:end]

        # Impute and center the batch
        X_centered = impute_and_center(X_batch)

        # Accumulate kinship contribution
        K = _accumulate_kinship(K, X_centered)

    # Scale by number of filtered SNPs
    K = K / n_snps

    # Log memory after kinship computation
    log_memory_snapshot(f"after_kinship_{n_samples}samples")

    # Return as numpy array for downstream compatibility
    return np.array(K)


def compute_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute centered relatedness matrix from disk-streamed genotypes.

    Implements: K = (1/p) * X_c @ X_c.T
    where X_c is centered with missing values imputed to SNP mean.

    This function reads genotype chunks directly from disk via bed-reader
    windowed reads, avoiding the need to load the full genotype matrix.

    Two-pass approach for filtering:
    1. First pass: compute per-SNP MAF, missing rate, variance for filtering
    2. Second pass: accumulate kinship from filtered SNPs only

    Note: Monomorphic SNPs (constant genotype) are always excluded to match GEMMA.

    Memory behavior:
        O(n^2 + n*chunk_size) vs O(n^2 + n*p) for full-load version.
        Only kinship (n^2) + one chunk (n*chunk_size) in memory at a time.
        Each chunk is freed after accumulation (Python GC).

    Use case:
        When genotypes don't fit in memory. At 200k samples and 95k SNPs,
        full genotypes are 76GB; streaming eliminates this allocation.

    Result equivalence:
        Produces identical kinship to compute_centered_kinship() within
        numerical precision (< 1e-10 relative tolerance).

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation
            and raise MemoryError if insufficient.
        show_progress: If True (default), show progress bar during iteration.

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        FileNotFoundError: If the PLINK .bed file does not exist.
        ValueError: If no SNPs pass filtering.

    Example:
        >>> from pathlib import Path
        >>> K = compute_kinship_streaming(Path("data/my_study"), maf_threshold=0.01)
        >>> K.shape
        (1940, 1940)
    """
    start_time = time.perf_counter()

    # Get dimensions without loading genotypes
    meta = get_plink_metadata(bed_path)
    n_samples = meta["n_samples"]
    n_snps = meta["n_snps"]

    logger.info("## Computing Kinship Matrix")
    logger.info(f"number of total individuals = {n_samples}")
    logger.info(f"number of total SNPs/variants = {n_snps}")
    logger.info(f"chunk size = {chunk_size}")

    # Memory check before allocation
    # Check against full pipeline peak (eigendecomp) since it always follows kinship
    if check_memory:
        est = estimate_streaming_memory(n_samples, n_snps, chunk_size)
        check_memory_available(
            est.total_peak_gb,
            operation=f"GWAS pipeline (eigendecomp peak: {est.total_peak_gb:.1f}GB)",
        )

    # === PASS 1: Compute per-SNP statistics for filtering ===
    # Always compute stats for monomorphic filtering (GEMMA behavior)
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = _progress_iterator(
            stats_iterator, total=n_chunks, desc="Computing SNP statistics"
        )

    for chunk, start, end in stats_iterator:
        chunk_miss_counts = np.sum(np.isnan(chunk), axis=0)
        with np.errstate(invalid="ignore"):
            chunk_means = np.nanmean(chunk, axis=0)
            chunk_vars = np.nanvar(chunk, axis=0)
        chunk_means = np.nan_to_num(chunk_means, nan=0.0)
        chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

        all_means[start:end] = chunk_means
        all_miss_counts[start:end] = chunk_miss_counts
        all_vars[start:end] = chunk_vars

    # Compute filters
    miss_rates = all_miss_counts / n_samples
    allele_freqs = all_means / 2.0
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)
    is_polymorphic = all_vars > 0

    # Combined filter: MAF, missing rate, and monomorphism (always applied)
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic

    n_filtered = int(np.sum(snp_mask))

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_snps}"
        )

    if n_filtered < n_snps:
        n_removed = n_snps - n_filtered
        logger.info(
            f"Kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )
    else:
        logger.info(f"number of analyzed SNPs = {n_filtered}")

    # Get indices of SNPs that passed filtering
    snp_indices = np.where(snp_mask)[0]

    # Initialize kinship accumulator
    K = jnp.zeros((n_samples, n_samples), dtype=jnp.float64)

    # === PASS 2: Accumulate kinship from filtered SNPs ===
    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )

    if show_progress:
        chunk_iter = _progress_iterator(
            chunk_iter, total=n_chunks, desc="Computing kinship"
        )

    for chunk, file_start, file_end in chunk_iter:
        # Find filtered SNPs in this chunk
        chunk_snp_mask = (snp_indices >= file_start) & (snp_indices < file_end)
        chunk_filtered_indices = snp_indices[chunk_snp_mask] - file_start

        if len(chunk_filtered_indices) == 0:
            continue

        # Extract only filtered columns
        X_chunk = jnp.array(chunk[:, chunk_filtered_indices])

        # Impute and center the chunk
        X_centered = impute_and_center(X_chunk)

        # Accumulate kinship contribution
        K = _accumulate_kinship(K, X_centered)

    # Scale by number of filtered SNPs
    K = K / n_filtered

    elapsed = time.perf_counter() - start_time
    logger.info("## Kinship matrix computed")
    logger.info(f"time elapsed = {elapsed:.2f} seconds")

    # Return as numpy array for downstream compatibility
    return np.array(K)
