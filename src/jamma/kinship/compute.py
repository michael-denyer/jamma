"""JAX kinship matrix computation.

This module provides the main kinship matrix computation function,
implementing GEMMA's centered relatedness matrix algorithm (-gk 1 mode).

The kinship matrix K is computed as:
    K = (1/p) * X_c @ X_c.T

where X_c is the centered genotype matrix with missing values imputed
to per-SNP mean, and p is the number of SNPs.

LOCO (Leave-One-Chromosome-Out) kinship is also supported via the
subtraction approach: K_loco_c = (S_full - S_c) / (p - p_c), where
S_full is the unscaled full kinship numerator and S_c is the contribution
from chromosome c. This avoids redundant computation.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import jit
from loguru import logger

from jamma.core import ensure_jax_configured
from jamma.core.memory import (
    check_memory_available,
    estimate_eigendecomp_memory,
    estimate_streaming_memory,
    log_memory_snapshot,
)
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.io.plink import (
    get_chromosome_partitions,
    get_plink_metadata,
    stream_genotype_chunks,
)
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize


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
    Delegates to shared utilities in jamma.core.snp_filter.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
        maf_threshold: Minimum MAF for inclusion (0 to disable MAF filter only).
        miss_threshold: Maximum missing rate for inclusion (1.0 to disable).

    Returns:
        Tuple of (filtered_genotypes, n_filtered, n_original).
    """
    n_samples, n_snps = genotypes.shape

    col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, _allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, miss_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )

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
    ensure_jax_configured()

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
    # Check against the larger of:
    # 1. Kinship phase: K (n²×8) + X JAX copy (n×p×8) + batch temps
    # 2. Eigendecomp phase: K (n²×8) + U (n²×8) + LAPACK workspace
    # When 2*p > n (typical GWAS), kinship phase with JAX genotype copy is the peak.
    # 10% safety margin: eigendecomp estimate includes full DSYEVD workspace.
    if check_memory:
        eigendecomp_peak_gb = estimate_eigendecomp_memory(n_samples)
        kinship_peak_gb = (
            n_samples**2 * 8 / 1e9  # K accumulator
            + n_samples * n_snps * 8 / 1e9  # X JAX copy of genotypes
        )
        required_gb = max(eigendecomp_peak_gb, kinship_peak_gb)
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=f"GWAS pipeline (peak: {required_gb:.1f}GB)",
        )

    # Log memory state before kinship allocation for debugging OOM
    log_memory_snapshot(f"before_kinship_{n_samples}samples")

    # Convert to JAX array
    X = jnp.array(genotypes_filtered, dtype=jnp.float64)

    # Initialize kinship accumulator
    K = jnp.zeros((n_samples, n_samples), dtype=jnp.float64)

    n_batches = (n_snps + batch_size - 1) // batch_size
    logger.info(
        f"Kinship: in-memory mode, {n_samples:,} samples x {n_snps:,} SNPs, "
        f"{n_batches} batches of {batch_size:,}"
    )

    # Process SNPs in batches
    batch_starts = list(range(0, n_snps, batch_size))
    if n_batches > 1:
        batch_iter = progress_iterator(
            enumerate(batch_starts), total=n_batches, desc="Kinship"
        )
    else:
        batch_iter = enumerate(batch_starts)

    for _, start in batch_iter:
        end = min(start + batch_size, n_snps)
        X_batch = X[:, start:end]

        # Impute and center the batch
        X_centered = impute_and_center(X_batch)

        # Accumulate kinship contribution
        K = _accumulate_kinship(K, X_centered)
        K.block_until_ready()  # Sync so progress bar reflects actual compute

    # Scale by number of filtered SNPs
    K = K / n_snps

    # Sync the K/n_snps division so memory snapshot is accurate
    K.block_until_ready()

    # Log memory after kinship computation
    log_memory_snapshot(f"after_kinship_{n_samples}samples")

    # Return as numpy array for downstream compatibility
    return np.array(K)


def compute_standardized_kinship(
    genotypes: np.ndarray,
    batch_size: int = 10000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
) -> np.ndarray:
    """Compute standardized relatedness matrix (GEMMA -gk 2).

    Implements K = (1/p) * Z @ Z.T where Z[i,k] = (x[i,k] - mean_k) / sd_k.
    Each SNP is centered and divided by its standard deviation. Monomorphic
    SNPs (sd=0) are included in the SNP count p but contribute zero to K.

    GEMMA's standardized kinship algorithm:
    1. Filter SNPs by MAF, missing rate, and monomorphism
    2. For each SNP batch: impute missing to mean, center, divide by sd
    3. Accumulate K += Z_batch @ Z_batch.T
    4. Scale K /= n_filtered_snps

    Note: Monomorphic SNPs are excluded by _filter_snps (which removes
    zero-variance SNPs). This matches GEMMA since monomorphic SNPs also
    fail MAF filtering in practice.

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
        >>> K = compute_standardized_kinship(X, maf_threshold=0.01)
        >>> K.shape
        (3, 3)
        >>> np.allclose(K, K.T)  # Symmetric
        True
    """
    ensure_jax_configured()

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
            f"Standardized kinship filtering: {n_snps:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    # Memory check before allocation
    if check_memory:
        eigendecomp_peak_gb = estimate_eigendecomp_memory(n_samples)
        kinship_peak_gb = (
            n_samples**2 * 8 / 1e9  # K accumulator
            + n_samples * n_snps * 8 / 1e9  # X JAX copy of genotypes
        )
        required_gb = max(eigendecomp_peak_gb, kinship_peak_gb)
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=f"GWAS pipeline (peak: {required_gb:.1f}GB)",
        )

    # Log memory state before kinship allocation
    log_memory_snapshot(f"before_standardized_kinship_{n_samples}samples")

    # Convert to JAX array
    X = jnp.array(genotypes_filtered, dtype=jnp.float64)

    # Initialize kinship accumulator
    K = jnp.zeros((n_samples, n_samples), dtype=jnp.float64)

    n_batches = (n_snps + batch_size - 1) // batch_size
    logger.info(
        f"Standardized kinship: in-memory mode, {n_samples:,} samples x "
        f"{n_snps:,} SNPs, {n_batches} batches of {batch_size:,}"
    )

    # Process SNPs in batches
    batch_starts = list(range(0, n_snps, batch_size))
    if n_batches > 1:
        batch_iter = progress_iterator(
            enumerate(batch_starts), total=n_batches, desc="Standardized Kinship"
        )
    else:
        batch_iter = enumerate(batch_starts)

    for _, start in batch_iter:
        end = min(start + batch_size, n_snps)
        X_batch = X[:, start:end]

        # Impute, center, and standardize the batch
        X_standardized = impute_center_and_standardize(X_batch)

        # Accumulate kinship contribution
        K = _accumulate_kinship(K, X_standardized)
        K.block_until_ready()  # Sync so progress bar reflects actual compute

    # Scale by number of filtered SNPs
    K = K / n_snps

    # Sync the K/n_snps division so memory snapshot is accurate
    K.block_until_ready()

    # Log memory after kinship computation
    log_memory_snapshot(f"after_standardized_kinship_{n_samples}samples")

    # Return as numpy array for downstream compatibility
    return np.array(K)


def compute_loco_kinship(
    genotypes: np.ndarray,
    chromosome_for_each_snp: np.ndarray,
    batch_size: int = 10000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
) -> Iterator[tuple[str, np.ndarray]]:
    """Compute LOCO kinship matrices via subtraction approach.

    For each chromosome c, computes K_loco_c = (S_full - S_c) / (p - p_c)
    where S_full is the unscaled full kinship numerator and S_c is the
    contribution from chromosome c's SNPs.

    Global centering is used (not per-chromosome centering) so the
    subtraction identity holds: S_full = sum(S_c) over all chromosomes.

    Yields one (chr_name, K_loco) pair at a time so the caller can process
    and discard each matrix without holding all LOCO matrices in memory.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
        chromosome_for_each_snp: String array of chromosome name per SNP,
            length must equal genotypes.shape[1].
        batch_size: SNPs per batch for kinship accumulation (default 10000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation.

    Yields:
        Tuple of (chr_name, K_loco) where chr_name is the chromosome being
        excluded and K_loco is the LOCO kinship matrix (n_samples, n_samples).

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        ValueError: If no SNPs pass filtering, or if all filtered SNPs are on
            a single chromosome (cannot compute LOCO).
    """
    ensure_jax_configured()

    n_samples, n_snps_original = genotypes.shape

    # Filter SNPs globally (MAF, missingness, monomorphism)
    # Compute mask once, use for both genotype and chromosome filtering
    col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, _allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, miss_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )

    n_filtered = int(np.sum(snp_mask))
    n_original = n_snps_original

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_original}"
        )

    genotypes_filtered = genotypes[:, snp_mask]
    chr_filtered = chromosome_for_each_snp[snp_mask]

    if n_filtered < n_original:
        n_removed = n_original - n_filtered
        logger.info(
            f"LOCO kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    # Memory check: S_full (n^2*8) + X_centered (n*p*8) + one S_c at a time (n^2*8)
    if check_memory:
        required_gb = (
            n_samples**2 * 8 / 1e9  # S_full
            + n_samples * n_filtered * 8 / 1e9  # X_centered (float64)
            + n_samples**2 * 8 / 1e9  # S_c (one at a time)
        )
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=f"LOCO kinship ({n_samples:,} samples, {n_filtered:,} SNPs)",
        )

    # Convert to JAX and center globally
    X = jnp.array(genotypes_filtered, dtype=jnp.float64)
    X_centered = impute_and_center(X)

    # Accumulate full kinship numerator S_full = X_centered @ X_centered.T (unscaled)
    S_full = jnp.zeros((n_samples, n_samples), dtype=jnp.float64)
    n_batches = (n_filtered + batch_size - 1) // batch_size

    logger.info(
        f"LOCO kinship: {n_samples:,} samples x {n_filtered:,} SNPs, "
        f"{n_batches} batches"
    )

    batch_starts = list(range(0, n_filtered, batch_size))
    if n_batches > 1:
        batch_iter = progress_iterator(
            enumerate(batch_starts), total=n_batches, desc="LOCO: full kinship"
        )
    else:
        batch_iter = enumerate(batch_starts)

    for _, start in batch_iter:
        end = min(start + batch_size, n_filtered)
        S_full = _accumulate_kinship(S_full, X_centered[:, start:end])
        S_full.block_until_ready()

    # Compute per-chromosome LOCO kinship via subtraction
    unique_chrs = sorted(set(np.array(chr_filtered)))
    logger.info(f"LOCO: computing {len(unique_chrs)} leave-one-out kinship matrices")

    for chr_name in unique_chrs:
        chr_mask = chr_filtered == chr_name
        p_chr = int(np.sum(chr_mask))
        p_loco = n_filtered - p_chr

        if p_loco == 0:
            raise ValueError(
                f"Cannot compute LOCO kinship: all {n_filtered} filtered SNPs "
                f"are on chromosome '{chr_name}'. LOCO requires SNPs on multiple "
                f"chromosomes."
            )

        # Compute chromosome contribution S_c
        X_chr = X_centered[:, chr_mask]
        S_chr = jnp.matmul(X_chr, X_chr.T)
        S_chr.block_until_ready()

        # K_loco = (S_full - S_c) / p_loco
        K_loco = np.array((S_full - S_chr) / p_loco)

        logger.debug(
            f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs retained"
        )

        yield (chr_name, K_loco)


def compute_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
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
        ksnps_indices: Pre-resolved column indices for -ksnps restriction, or None.

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
    ensure_jax_configured()

    start_time = time.perf_counter()

    # Get dimensions without loading genotypes
    meta = get_plink_metadata(bed_path)
    n_samples = meta["n_samples"]
    n_snps = meta["n_snps"]

    logger.info("Computing Kinship Matrix")
    logger.info(f"  Individuals: {n_samples:,}")
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chunk size: {chunk_size:,}")

    # Memory check before allocation
    # Check against full pipeline peak (eigendecomp) since it always follows kinship.
    # Use 50% safety margin: JAX kinship creates temporary arrays (centered batches,
    # batch products) that ~1.5x the naive estimate per empirical benchmarks.
    if check_memory:
        est = estimate_streaming_memory(n_samples, n_snps, chunk_size)
        check_memory_available(
            est.total_peak_gb,
            safety_margin=0.5,
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
        stats_iterator = progress_iterator(
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

    # Apply kinship SNP list restriction (if -ksnps provided)
    if ksnps_indices is not None:
        from jamma.core.snp_filter import apply_snp_list_mask

        apply_snp_list_mask(snp_mask, ksnps_indices, n_snps, "Kinship SNP list")

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
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")

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
        chunk_iter = progress_iterator(
            chunk_iter, total=n_chunks, desc="Computing kinship"
        )

    for chunk, file_start, file_end in chunk_iter:
        # Binary search for filtered SNPs in this chunk: O(log n) vs O(n)
        # snp_indices is sorted (from np.where), so searchsorted is valid
        left = np.searchsorted(snp_indices, file_start, side="left")
        right = np.searchsorted(snp_indices, file_end, side="left")
        chunk_filtered_indices = snp_indices[left:right] - file_start

        if len(chunk_filtered_indices) == 0:
            continue

        # Extract only filtered columns
        X_chunk = jnp.array(chunk[:, chunk_filtered_indices])

        # Impute and center the chunk
        X_centered = impute_and_center(X_chunk)

        # Accumulate kinship contribution
        K = _accumulate_kinship(K, X_centered)
        K.block_until_ready()  # Sync so progress bar reflects actual compute

    # Scale by number of filtered SNPs
    K = K / n_filtered

    elapsed = time.perf_counter() - start_time
    logger.info(f"Kinship matrix computed in {elapsed:.2f}s")

    # Return as numpy array for downstream compatibility
    return np.array(K)


def compute_loco_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Compute LOCO kinship matrices from disk-streamed genotypes.

    Two-pass streaming approach that accumulates both S_full and per-chromosome
    S_chr matrices in a single second pass, then derives LOCO kinship via
    subtraction: K_loco_c = (S_full - S_chr[c]) / (p - p_c).

    Pass 1: Compute per-SNP statistics for filtering (MAF, missingness, variance).
    Pass 2: Stream filtered SNPs, accumulate S_full and all S_chr simultaneously.
    After passes: Yield LOCO kinship matrices one at a time.

    Memory profile: S_full (n^2*8) + sum(S_chr) (n_chr * n^2*8) + chunk buffer.
    For mouse_hs1940 (1940 samples, 19 chromosomes): ~570 MB.

    TODO: For large-scale datasets (100k+ samples) where n_chr * n^2 * 8 exceeds
    available memory, a multi-pass approach (one chromosome per pass) would be
    needed. Not implemented in Phase 25.

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation.
        show_progress: If True (default), show progress bar during iteration.
        ksnps_indices: Pre-resolved column indices for -ksnps restriction, or None.

    Yields:
        Tuple of (chr_name, K_loco) where chr_name is the chromosome being
        excluded and K_loco is the LOCO kinship matrix (n_samples, n_samples).

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        FileNotFoundError: If the PLINK .bed file does not exist.
        ValueError: If no SNPs pass filtering, or if all filtered SNPs are on
            a single chromosome.
    """
    ensure_jax_configured()

    start_time = time.perf_counter()

    # Get dimensions and chromosome metadata
    meta = get_plink_metadata(bed_path)
    n_samples = meta["n_samples"]
    n_snps = meta["n_snps"]
    chromosomes = meta["chromosome"]

    # Build chromosome partition from metadata
    partitions = get_chromosome_partitions(bed_path)
    unique_chrs = sorted(partitions.keys())

    logger.info("Computing LOCO Kinship (streaming)")
    logger.info(f"  Individuals: {n_samples:,}")
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chromosomes: {len(unique_chrs)}")
    logger.info(f"  Chunk size: {chunk_size:,}")

    # === PASS 1: Compute per-SNP statistics for filtering ===
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks, desc="LOCO: SNP statistics"
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
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic

    # Apply kinship SNP list restriction (if -ksnps provided)
    if ksnps_indices is not None:
        from jamma.core.snp_filter import apply_snp_list_mask

        apply_snp_list_mask(snp_mask, ksnps_indices, n_snps, "Kinship SNP list")

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
            f"LOCO kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    # Build SNP-to-chromosome mapping for filtered SNPs
    snp_indices = np.where(snp_mask)[0]

    # Map each filtered SNP index to its chromosome
    chr_for_filtered = chromosomes[snp_indices]

    # Count filtered SNPs per chromosome
    n_chr_filtered: dict[str, int] = {
        chr_name: int(np.sum(chr_for_filtered == chr_name)) for chr_name in unique_chrs
    }

    # Memory check: S_full + all S_chr + chunk buffer
    if check_memory:
        n_chr_with_snps = sum(1 for c in n_chr_filtered.values() if c > 0)
        required_gb = (
            n_samples**2 * 8 / 1e9  # S_full
            + n_chr_with_snps * n_samples**2 * 8 / 1e9  # all S_chr
            + n_samples * chunk_size * 8 / 1e9  # chunk buffer
        )
        if required_gb > 10:
            logger.warning(
                f"LOCO streaming: combined S_chr allocation is "
                f"{n_chr_with_snps * n_samples**2 * 8 / 1e9:.1f}GB "
                f"({n_chr_with_snps} chromosomes x {n_samples:,} samples)"
            )
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=(
                f"LOCO kinship streaming ({n_samples:,} samples, "
                f"{n_filtered:,} SNPs, {n_chr_with_snps} chromosomes)"
            ),
        )

    # Initialize accumulators
    S_full = jnp.zeros((n_samples, n_samples), dtype=jnp.float64)
    S_chr: dict[str, jnp.ndarray] = {
        chr_name: jnp.zeros((n_samples, n_samples), dtype=jnp.float64)
        for chr_name in unique_chrs
        if n_chr_filtered.get(chr_name, 0) > 0
    }

    # === PASS 2: Accumulate S_full and per-chromosome S_chr ===
    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )

    if show_progress:
        chunk_iter = progress_iterator(
            chunk_iter, total=n_chunks, desc="LOCO: kinship accumulation"
        )

    for chunk, file_start, file_end in chunk_iter:
        # Binary search for filtered SNPs in this chunk
        left = np.searchsorted(snp_indices, file_start, side="left")
        right = np.searchsorted(snp_indices, file_end, side="left")
        chunk_snp_global_indices = snp_indices[left:right]
        chunk_filtered_local = chunk_snp_global_indices - file_start

        if len(chunk_filtered_local) == 0:
            continue

        # Extract filtered columns, impute and center
        X_chunk = jnp.array(chunk[:, chunk_filtered_local])
        X_centered = impute_and_center(X_chunk)

        # Accumulate full kinship
        S_full = _accumulate_kinship(S_full, X_centered)
        S_full.block_until_ready()

        # Group by chromosome and accumulate per-chromosome contributions
        chunk_chrs = chromosomes[chunk_snp_global_indices]
        for chr_name in set(chunk_chrs):
            X_chr_part = X_centered[:, chunk_chrs == chr_name]
            S_chr[chr_name] = _accumulate_kinship(S_chr[chr_name], X_chr_part)
            S_chr[chr_name].block_until_ready()

    # === Yield LOCO kinship matrices ===
    elapsed = time.perf_counter() - start_time
    logger.info(
        f"LOCO streaming accumulation complete in {elapsed:.2f}s, "
        f"computing {len(S_chr)} LOCO matrices"
    )

    for chr_name in sorted(S_chr.keys()):
        p_chr = n_chr_filtered[chr_name]
        p_loco = n_filtered - p_chr

        if p_loco == 0:
            raise ValueError(
                f"Cannot compute LOCO kinship: all {n_filtered} filtered SNPs "
                f"are on chromosome '{chr_name}'."
            )

        K_loco = np.array((S_full - S_chr[chr_name]) / p_loco)

        logger.debug(
            f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs retained"
        )

        # Free S_chr for this chromosome after yielding
        del S_chr[chr_name]
        yield (chr_name, K_loco)
