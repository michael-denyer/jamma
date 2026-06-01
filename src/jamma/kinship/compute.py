"""Kinship matrix computation.

This module provides the main kinship matrix computation function,
implementing GEMMA's centered relatedness matrix algorithm (-gk 1 mode).

The kinship matrix K is computed as:
    K = (1/p) * X_c @ X_c.T

where X_c is the centered genotype matrix with missing values imputed
to per-SNP mean, and p is the number of SNPs.

The standard (non-LOCO) kinship computation and in-memory LOCO kinship
use jlinalg.dsyrk exclusively. The streaming LOCO function
(compute_loco_kinship_streaming) uses NumPy for accumulation.

LOCO (Leave-One-Chromosome-Out) kinship is also supported via the
subtraction approach: K_loco_c = (S_full - S_c) / (p - p_c), where
S_full is the unscaled full kinship numerator and S_c is the contribution
from chromosome c. This avoids redundant computation.
"""

from __future__ import annotations

import gc
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
from loguru import logger

from jamma import jlinalg
from jamma.core.estimates import estimate_kinship_seconds
from jamma.core.memory import (
    check_memory_available,
    estimate_eigendecomp_memory,
    estimate_streaming_memory,
    log_memory_snapshot,
)
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.io.plink import (
    get_plink_metadata,
    partitions_from_metadata,
    stream_genotype_chunks,
    validate_genotype_values,
)
from jamma.jlinalg import compute_snp_stats_chunk
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize
from jamma.utils import chr_sort_key


@dataclass(frozen=True)
class SnpStatsCache:
    """Global SNP statistics from kinship streaming PASS 1.

    Stores per-SNP means, missing counts, and variances for ALL SNPs in the
    BIM file (unfiltered), computed over ALL samples (including those with
    missing phenotypes). Per-chromosome stats are extracted by indexing
    with chr_snp_indices: cache.col_means[chr_snp_indices].

    The all-samples population matters: ``n_samples`` is the denominator for
    miss_rate and the basis for col_means / col_vars. When filtering in the
    association pass, use ``cache.n_samples`` — NOT n_valid — to match the
    population the stats were computed from.

    Returned by ``compute_loco_kinship_streaming(return_snp_stats=True)`` so the
    association pass can reuse PASS-1 statistics instead of re-reading the BED
    once per chromosome.
    """

    col_means: np.ndarray  # shape (n_snps_total,), float64
    miss_counts: np.ndarray  # shape (n_snps_total,), intp
    col_vars: np.ndarray  # shape (n_snps_total,), float64
    n_samples: int  # sample count stats were computed over (ALL samples)

    def __post_init__(self) -> None:
        """Validate array shapes and freeze array contents."""
        if not (self.col_means.shape == self.miss_counts.shape == self.col_vars.shape):
            raise ValueError(
                f"Array shape mismatch: col_means={self.col_means.shape}, "
                f"miss_counts={self.miss_counts.shape}, "
                f"col_vars={self.col_vars.shape}"
            )
        if self.col_means.ndim != 1:
            raise ValueError(f"Expected 1-D arrays, got ndim={self.col_means.ndim}")
        for arr in (self.col_means, self.miss_counts, self.col_vars):
            arr.flags.writeable = False

    @property
    def n_snps(self) -> int:
        """Number of SNPs in the cache (unfiltered BIM count)."""
        return self.col_means.shape[0]


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Return arr as float64, copying only when the dtype differs."""
    return arr if arr.dtype == np.float64 else arr.astype(np.float64)


def _accumulate_kinship(K: np.ndarray, X_centered: np.ndarray) -> np.ndarray:
    """Accumulate kinship contribution from centered SNP batch.

    Uses jlinalg.dsyrk (symmetric rank-k update) with in-place accumulation.
    The non-LOCO kinship path uses this exclusively.

    Args:
        K: Current kinship matrix accumulator (n_samples, n_samples)
        X_centered: Centered genotype batch (n_samples, batch_snps)

    Returns:
        Updated kinship matrix with batch contribution added.
    """
    K += jlinalg.dsyrk(X_centered)
    return K


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


def _compute_kinship_inmemory(
    genotypes: np.ndarray,
    transform_fn: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
    maf_threshold: float,
    miss_threshold: float,
    check_memory: bool,
    label: str,
) -> np.ndarray:
    """Shared implementation for in-memory kinship computation.

    Implements K = (1/p) * sum(transform(X_batch) @ transform(X_batch).T)
    where transform is either centering (gk=1) or centering+standardizing (gk=2).

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
        transform_fn: Per-batch transformation (impute_and_center or
            impute_center_and_standardize).
        batch_size: SNPs per batch.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        check_memory: Check available memory before allocation.
        label: Label for logging (e.g. "Kinship", "Standardized kinship").

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        ValueError: If no SNPs pass filtering.
    """
    n_samples, _ = genotypes.shape

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
            f"{label} filtering: {n_snps:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    if check_memory:
        eigendecomp_peak_gb = estimate_eigendecomp_memory(n_samples)
        kinship_peak_gb = n_samples**2 * 8 / 1e9 + n_samples * n_snps * 8 / 1e9
        required_gb = max(eigendecomp_peak_gb, kinship_peak_gb)
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=f"GWAS pipeline (peak: {required_gb:.1f}GB)",
        )

    log_memory_snapshot(f"before_{label.lower().replace(' ', '_')}_{n_samples}samples")

    X = _ensure_float64(genotypes_filtered)
    K = np.zeros((n_samples, n_samples), dtype=np.float64)

    n_batches = (n_snps + batch_size - 1) // batch_size
    logger.info(
        f"{label}: in-memory mode, {n_samples:,} samples x {n_snps:,} SNPs, "
        f"{n_batches} batches of {batch_size:,}"
    )

    batch_starts = list(range(0, n_snps, batch_size))
    if n_batches > 1:
        batch_iter = progress_iterator(
            enumerate(batch_starts), total=n_batches, desc=label
        )
    else:
        batch_iter = enumerate(batch_starts)

    for _, start in batch_iter:
        end = min(start + batch_size, n_snps)
        X_transformed = transform_fn(X[:, start:end])
        K = _accumulate_kinship(K, X_transformed)

    K = K / n_snps

    log_memory_snapshot(f"after_{label.lower().replace(' ', '_')}_{n_samples}samples")

    return K


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

    Note: Monomorphic SNPs (constant genotype) are always excluded to match GEMMA.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
            Values are typically 0, 1, or 2 representing minor allele counts.
        batch_size: SNPs per batch (default 10000, matches GEMMA).
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
    return _compute_kinship_inmemory(
        genotypes,
        impute_and_center,
        batch_size,
        maf_threshold,
        miss_threshold,
        check_memory,
        "Kinship",
    )


def compute_standardized_kinship(
    genotypes: np.ndarray,
    batch_size: int = 10000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
) -> np.ndarray:
    """Compute standardized relatedness matrix (GEMMA -gk 2).

    Implements K = (1/p) * Z @ Z.T where Z[i,k] = (x[i,k] - mean_k) / sd_k.
    Each SNP is centered and divided by its standard deviation.

    Note: Monomorphic SNPs are excluded by _filter_snps (which removes
    zero-variance SNPs). This matches GEMMA since monomorphic SNPs also
    fail MAF filtering in practice.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
            Values are typically 0, 1, or 2 representing minor allele counts.
        batch_size: SNPs per batch (default 10000, matches GEMMA).
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
    return _compute_kinship_inmemory(
        genotypes,
        impute_center_and_standardize,
        batch_size,
        maf_threshold,
        miss_threshold,
        check_memory,
        "Standardized kinship",
    )


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

    X = _ensure_float64(genotypes_filtered)
    X_centered = impute_and_center(X)

    # Accumulate full kinship numerator S_full = X_centered @ X_centered.T (unscaled)
    S_full = np.zeros((n_samples, n_samples), dtype=np.float64)
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
        batch = X_centered[:, start:end]
        S_full += jlinalg.dsyrk(batch)

    # Compute per-chromosome LOCO kinship via subtraction
    unique_chrs = sorted(set(chr_filtered))
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
        S_chr = jlinalg.dsyrk(X_chr)

        # K_loco = (S_full - S_c) / p_loco
        K_loco = (S_full - S_chr) / p_loco

        logger.debug(
            f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs retained"
        )

        yield (chr_name, K_loco)


def _kinship_single_pass(
    bed_path: Path,
    n_samples: int,
    n_snps: int,
    chunk_size: int,
    show_progress: bool,
    valid_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Single-pass kinship: compute stats and accumulate in one BED read.

    Only valid when no MAF/missing filters are active (maf_threshold=0.0,
    miss_threshold>=1.0, ksnps_indices=None). Monomorphism filtering
    (variance > 0) is applied per-chunk inline, matching the two-pass result.

    Args:
        bed_path: Path prefix for PLINK files.
        n_samples: Number of samples.
        n_snps: Total number of SNPs.
        chunk_size: Number of SNPs per chunk.
        show_progress: Whether to show progress bar.
        valid_indices: Optional array of sample indices to keep. When provided,
            the kinship matrix is accumulated at (n_valid, n_valid) size directly,
            avoiding allocation of the full (n_samples, n_samples) matrix.

    Returns:
        Kinship matrix (n_out, n_out) where n_out = len(valid_indices) or n_samples.

    Raises:
        ValueError: If valid_indices is empty, out of bounds, or unsorted.
        ValueError: If no SNPs pass monomorphism filter.
    """
    if valid_indices is not None:
        _validate_valid_indices(valid_indices, n_samples)

    n_out = len(valid_indices) if valid_indices is not None else n_samples
    K = np.zeros((n_out, n_out), dtype=np.float64)
    n_filtered = 0

    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        chunk_iter = progress_iterator(
            chunk_iter,
            total=n_chunks,
            desc="Computing kinship (single-pass)",
            initial_eta_seconds=estimate_kinship_seconds(n_out, n_snps),
        )

    for chunk, _start, _end in chunk_iter:
        # Early valid-sample subsetting: compute stats on valid samples only.
        if valid_indices is not None:
            chunk = chunk[valid_indices, :]

        # Per-chunk monomorphism filter: exclude constant genotype columns.
        # Suppress RuntimeWarning for all-NaN columns (no valid samples in chunk).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_vars = np.nanvar(chunk, axis=0)
        poly_mask = col_vars > 0
        n_poly = np.count_nonzero(poly_mask)
        if n_poly == 0:
            continue

        X_chunk = chunk[:, poly_mask]
        X_centered = impute_and_center(X_chunk)
        K = _accumulate_kinship(K, X_centered)
        n_filtered += n_poly
        del chunk, X_chunk, X_centered

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed monomorphism filter. Original SNP count: {n_snps}"
        )

    K /= n_filtered
    return K


def compute_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
    valid_indices: np.ndarray | None = None,
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
        valid_indices: Optional array of sample indices to keep. When provided,
            the kinship matrix is accumulated at (n_valid, n_valid) size directly,
            avoiding allocation of the full (n_samples, n_samples) matrix.

    Returns:
        Kinship matrix (n_out, n_out) where n_out = len(valid_indices) or n_samples.
        Symmetric, scaled by n_filtered_snps.

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

    if valid_indices is not None:
        _validate_valid_indices(valid_indices, n_samples)

    from jamma.core.estimates import estimate_kinship_time

    n_out = len(valid_indices) if valid_indices is not None else n_samples

    logger.info("Computing Kinship Matrix")
    logger.info(
        f"  Individuals: {n_out:,}"
        + (f" (filtered from {n_samples:,})" if n_out != n_samples else "")
    )
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chunk size: {chunk_size:,}")
    logger.info(f"  Estimated time: {estimate_kinship_time(n_out, n_snps)}")

    # Memory check before allocation.
    # Use n_samples (not n_out): stream_genotype_chunks reads full BED rows
    # at (n_samples, chunk_size), subsetting to valid_indices happens after
    # allocation. Eigendecomp and kinship accumulator use n_out, but passing
    # n_samples is conservative and safe.
    if check_memory:
        est = estimate_streaming_memory(n_samples, chunk_size=chunk_size)
        check_memory_available(
            est.total_peak_gb,
            safety_margin=0.1,
            operation=f"GWAS pipeline (eigendecomp peak: {est.total_peak_gb:.1f}GB)",
        )

    # Single-pass optimization: when no MAF/missing filters are active and
    # no ksnps restriction, monomorphism filtering can be done inline per-chunk.
    # This eliminates the stats-only BED read (pass 1), halving I/O at scale
    # (e.g. ~76 GB saved at 200k samples x 95k SNPs).
    use_single_pass = (
        maf_threshold == 0.0 and miss_threshold >= 1.0 and ksnps_indices is None
    )

    if use_single_pass:
        logger.debug("Kinship: single-pass mode (no MAF/missing filters)")
        K = _kinship_single_pass(
            bed_path,
            n_samples,
            n_snps,
            chunk_size,
            show_progress,
            valid_indices=valid_indices,
        )
        elapsed = time.perf_counter() - start_time
        logger.info(f"Kinship matrix computed in {elapsed:.2f}s")
        return K

    # === PASS 1: Compute per-SNP statistics for filtering ===
    # Always compute stats for monomorphic filtering (GEMMA behavior)
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.intp)
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
        # Early valid-sample subsetting: compute stats on valid samples only.
        if valid_indices is not None:
            chunk = chunk[valid_indices, :]
        chunk = np.ascontiguousarray(chunk)
        compute_snp_stats_chunk(
            chunk,
            all_means[start:end],
            all_miss_counts[start:end],
            all_vars[start:end],
        )
        del chunk  # Free ~1.6GB per chunk at scale before next iteration

    # Compute filters (free source arrays immediately after deriving values)
    n_denom = len(valid_indices) if valid_indices is not None else n_samples
    miss_rates = all_miss_counts / n_denom
    del all_miss_counts
    allele_freqs = all_means / 2.0
    del all_means
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)
    is_polymorphic = all_vars > 0
    del all_vars

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
    K = np.zeros((n_out, n_out), dtype=np.float64)

    # === PASS 2: Accumulate kinship from filtered SNPs ===
    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )

    if show_progress:
        chunk_iter = progress_iterator(
            chunk_iter,
            total=n_chunks,
            desc="Computing kinship",
            initial_eta_seconds=estimate_kinship_seconds(n_out, n_snps),
        )

    for chunk, file_start, file_end in chunk_iter:
        # Binary search for filtered SNPs in this chunk: O(log n) vs O(n)
        # snp_indices is sorted (from np.where), so searchsorted is valid
        left = np.searchsorted(snp_indices, file_start, side="left")
        right = np.searchsorted(snp_indices, file_end, side="left")
        chunk_filtered_indices = snp_indices[left:right] - file_start

        if len(chunk_filtered_indices) == 0:
            continue

        # Early valid-sample subsetting for pass 2 (rows before columns
        # to avoid allocating an (n_samples, n_filtered_cols) intermediate).
        if valid_indices is not None:
            chunk = chunk[valid_indices, :]

        # Extract only filtered columns (fancy indexing produces a copy)
        X_chunk = chunk[:, chunk_filtered_indices]
        assert X_chunk.dtype == np.float64, (
            f"kinship accumulation requires float64 chunks (got {X_chunk.dtype}); "
            "check stream_genotype_chunks dtype arg"
        )

        # Impute and center the chunk
        X_centered = impute_and_center(X_chunk)

        # Accumulate kinship contribution (in-place numpy matmul)
        K = _accumulate_kinship(K, X_centered)

    # Scale by number of filtered SNPs
    K = K / n_filtered

    elapsed = time.perf_counter() - start_time
    logger.info(f"Kinship matrix computed in {elapsed:.2f}s")

    return K


def _yield_full_kinship_fallback(
    S_full_np: np.ndarray,
    chrs_without_snps: list[str],
    n_filtered: int,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield full kinship for chromosomes with 0 filtered SNPs.

    When a chromosome has no SNPs after filtering, there is nothing to leave
    out, so K_loco equals K_full.

    Divides S_full_np in-place to avoid allocating a separate K_full buffer
    (saves n^2 * 8 bytes — 320GB at 200k samples).  Callers must not use
    S_full_np after this function returns.

    Args:
        S_full_np: Full kinship numerator as numpy array (n_samples, n_samples).
            **Consumed in-place** — contents are overwritten with K_full.
        chrs_without_snps: Chromosomes with 0 filtered SNPs.
        n_filtered: Total number of filtered SNPs.

    Yields:
        (chr_name, K_full) pairs in biological chromosome order.
        Each matrix is an independent allocation (safe to mutate in-place).
    """
    if not chrs_without_snps:
        return
    if n_filtered == 0:
        raise ValueError(
            "Cannot compute fallback kinship: n_filtered is 0 "
            "(no SNPs passed filtering)"
        )
    # In-place division: S_full_np becomes K_full, no extra n^2 allocation.
    S_full_np /= n_filtered
    S_full_np.flags.writeable = False  # Guard against accidental re-mutation
    for chr_name in sorted(chrs_without_snps, key=chr_sort_key):
        logger.debug(f"LOCO chr {chr_name}: 0 SNPs after filtering, using full kinship")
        yield (chr_name, S_full_np.copy())


def _yield_loco_matrices(
    S_full_np: np.ndarray,
    S_chr: dict[str, np.ndarray],
    n_chr_filtered: dict[str, int],
    n_filtered: int,
    K_loco_buf: np.ndarray | None = None,
    *,
    copy_output: bool = True,
) -> Iterator[tuple[str, np.ndarray]]:
    """Compute and yield LOCO kinship matrices from S_full and per-chr accumulators.

    For each chromosome, computes K_loco = (S_full - S_chr[c]) / (p - p_c),
    freeing S_chr[c] after each yield.

    Args:
        S_full_np: Full kinship numerator as numpy array (n_samples, n_samples).
        S_chr: Per-chromosome kinship contributions.
        n_chr_filtered: Count of filtered SNPs per chromosome.
        n_filtered: Total number of filtered SNPs.
        K_loco_buf: Pre-allocated workspace (n_samples, n_samples) for K_loco.
            When provided, np.subtract(out=) avoids a temporary. When None,
            a new array is allocated per chromosome (legacy behavior).
        copy_output: If True (default), the buffer is copied before yielding
            so callers may freely materialise the iterator (e.g. ``dict()``
            or ``list()``). If False, the yielded array is the shared
            ``K_loco_buf`` itself — callers MUST fully consume each matrix
            before advancing the iterator, as the next iteration overwrites
            the buffer.

    Yields:
        (chr_name, K_loco) pairs in biological chromosome order.

    Raises:
        ValueError: If all filtered SNPs are on a single chromosome.
    """
    # Safe to del during iteration: sorted() materializes keys into a list.
    for chr_name in sorted(S_chr.keys(), key=chr_sort_key):
        p_chr = n_chr_filtered[chr_name]
        p_loco = n_filtered - p_chr

        if p_loco == 0:
            raise ValueError(
                f"Cannot compute LOCO kinship: all {n_filtered} filtered SNPs "
                f"are on chromosome '{chr_name}'."
            )

        if K_loco_buf is not None:
            # In-place subtraction avoids a temporary array (LOCO-03).
            # Copy before yielding only when the caller may materialise the
            # iterator. Sequential internal consumers can safely reuse the
            # buffer and avoid one extra n x n allocation per chromosome.
            np.subtract(S_full_np, np.asarray(S_chr[chr_name]), out=K_loco_buf)
            K_loco_buf /= p_loco
            K_loco = K_loco_buf.copy() if copy_output else K_loco_buf
        else:
            K_loco = (S_full_np - np.asarray(S_chr[chr_name])) / p_loco
        logger.debug(
            f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs retained"
        )
        del S_chr[chr_name]
        yield (chr_name, K_loco)


def _validate_valid_indices(valid_indices: np.ndarray, n_samples: int) -> None:
    """Validate valid_indices for emptiness, bounds, duplicates, and ordering.

    Args:
        valid_indices: Array of sample indices to keep.
        n_samples: Total number of samples (upper bound for indices).

    Raises:
        ValueError: If indices are empty, out of bounds, duplicated, or unsorted.
    """
    if len(valid_indices) == 0:
        raise ValueError("valid_indices must not be empty")
    if valid_indices.min() < 0 or valid_indices.max() >= n_samples:
        raise ValueError(
            f"valid_indices out of bounds: min={valid_indices.min()}, "
            f"max={valid_indices.max()}, n_samples={n_samples}"
        )
    n_unique = len(np.unique(valid_indices))
    if len(valid_indices) != n_unique:
        raise ValueError(
            f"valid_indices contains {len(valid_indices) - n_unique} duplicates"
        )
    if not np.all(np.diff(valid_indices) > 0):
        raise ValueError(
            "valid_indices must be strictly increasing (sorted, no duplicates)"
        )


def _stream_s_full_and_chr(
    bed_path: Path,
    n_samples: int,
    n_snps: int,
    snp_indices: np.ndarray,
    chromosomes: np.ndarray,
    chr_subset: list[str],
    chunk_size: int,
    show_progress: bool,
    desc: str,
    S_full_accum: bool = True,
    valid_indices: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Stream genotypes and accumulate S_full and/or per-chromosome S_chr.

    Args:
        bed_path: PLINK file prefix.
        n_samples: Number of samples.
        n_snps: Total SNPs in the BED file (for chunk iteration).
        snp_indices: Global indices of filtered SNPs (sorted).
        chromosomes: Chromosome label for every SNP in the BED file.
        chr_subset: Chromosomes to accumulate S_chr for in this pass.
        chunk_size: SNPs per disk chunk.
        show_progress: Show progress bar.
        desc: Progress bar description.
        S_full_accum: If True, also accumulate S_full. Set False for
            multi-pass batches after S_full is already computed.
        valid_indices: Row indices (into the full n_samples axis) to retain
            before accumulation. When provided, S_full and S_chr are accumulated
            at shape (n_valid, n_valid) rather than (n_samples, n_samples),
            where n_valid = len(valid_indices). When None, all samples are used.

    Returns:
        (S_full or None, dict of chr_name -> S_chr). Matrix dimension is
        n_valid x n_valid when valid_indices is provided, otherwise
        n_samples x n_samples.
    """
    if valid_indices is not None:
        _validate_valid_indices(valid_indices, n_samples)

    n_out = len(valid_indices) if valid_indices is not None else n_samples
    S_full = np.zeros((n_out, n_out), dtype=np.float64) if S_full_accum else None
    chr_set = set(chr_subset)
    S_chr: dict[str, np.ndarray] = {
        c: np.zeros((n_out, n_out), dtype=np.float64) for c in chr_subset
    }

    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )
    if show_progress:
        chunk_iter = progress_iterator(chunk_iter, total=n_chunks, desc=desc)

    for chunk, file_start, file_end in chunk_iter:
        left = np.searchsorted(snp_indices, file_start, side="left")
        right = np.searchsorted(snp_indices, file_end, side="left")
        chunk_snp_global_indices = snp_indices[left:right]
        chunk_filtered_local = chunk_snp_global_indices - file_start

        if len(chunk_filtered_local) == 0:
            continue

        # Check which target chromosomes are in this chunk before allocating
        chunk_chrs = chromosomes[chunk_snp_global_indices]
        target_chrs_in_chunk = set(chunk_chrs) & chr_set

        # Skip centering when S_full isn't needed and no target chromosomes present
        if S_full is None and not target_chrs_in_chunk:
            continue

        X_chunk = chunk[:, chunk_filtered_local]
        assert X_chunk.dtype == np.float64, (
            f"kinship accumulation requires float64 chunks (got {X_chunk.dtype}); "
            "check stream_genotype_chunks dtype arg"
        )

        # Subset rows to valid samples before centering.
        # Centering must use the valid-sample mean (not the full-sample mean)
        # to match GEMMA's behaviour.
        if valid_indices is not None:
            X_chunk = X_chunk[valid_indices, :]
        X_centered = impute_and_center(X_chunk)

        if S_full is not None:
            S_full += np.dot(X_centered, X_centered.T)

        for chr_name in target_chrs_in_chunk:
            X_chr_part = X_centered[:, chunk_chrs == chr_name]
            S_chr[chr_name] += np.dot(X_chr_part, X_chr_part.T)

    return S_full, S_chr


def compute_loco_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
    valid_indices: np.ndarray | None = None,
    _copy_yielded_matrices: bool = True,
    return_snp_stats: bool = False,
    _max_batch_chrs: int | None = None,
) -> (
    Iterator[tuple[str, np.ndarray]]
    | tuple[Iterator[tuple[str, np.ndarray]], SnpStatsCache]
):
    """Compute LOCO kinship matrices from disk-streamed genotypes.

    Two-pass streaming approach that accumulates both S_full and per-chromosome
    S_chr matrices, then derives LOCO kinship via subtraction:
    K_loco_c = (S_full - S_chr[c]) / (p - p_c).

    Pass 1: Compute per-SNP statistics for filtering (MAF, missingness, variance).
    Pass 2: Stream filtered SNPs, accumulate S_full and S_chr.

    When all S_chr fit in memory, a single second pass accumulates everything.
    When memory is insufficient (e.g. 100k+ samples), chromosomes are batched
    across multiple passes — S_full is computed once in the first pass, then
    each subsequent pass accumulates S_chr for a batch of chromosomes.

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation.
        show_progress: If True (default), show progress bar during iteration.
        ksnps_indices: Pre-resolved column indices for -ksnps restriction, or None.
        valid_indices: Row indices (into the full n_samples axis) to retain before
            accumulation. When provided, each yielded K_loco has shape
            (n_valid, n_valid) where n_valid = len(valid_indices), eliminating
            the post-hoc np.ix_ copy. When None, K_loco has shape
            (n_samples, n_samples) (default, backward-compatible).
        return_snp_stats: If True, also return the PASS-1 SnpStatsCache (global
            per-SNP means/miss-counts/variances over all samples) so callers
            (the LOCO LMM pass) can reuse it instead of re-reading the BED per
            chromosome. Changes the return type from an iterator to
            ``(iterator, SnpStatsCache)``.
        _max_batch_chrs: Debug override forcing a fixed chromosomes-per-pass
            batch size (bypasses memory-based sizing). Used by tests to exercise
            multi-pass without mocking psutil.

    Returns:
        An iterator of (chr_name, K_loco) pairs, where chr_name is the
        chromosome being excluded and K_loco is the LOCO kinship matrix with
        shape (n_valid, n_valid) when valid_indices is provided, else
        (n_samples, n_samples). When ``return_snp_stats`` is True, returns
        ``(iterator, SnpStatsCache)`` instead.

    Raises:
        MemoryError: If check_memory=True and insufficient memory for even
            S_full + one S_chr.
        FileNotFoundError: If the PLINK .bed file does not exist.
        ValueError: If no SNPs pass filtering, or if all filtered SNPs are on
            a single chromosome.
    """
    start_time = time.perf_counter()

    # Get dimensions and chromosome metadata
    meta = get_plink_metadata(bed_path)
    n_samples = meta["n_samples"]
    n_snps = meta["n_snps"]
    chromosomes = meta["chromosome"]

    if valid_indices is not None:
        _validate_valid_indices(valid_indices, n_samples)

    # Derive partitions from already-loaded metadata — avoids re-opening BED (LOCO-04)
    partitions = partitions_from_metadata(meta)
    unique_chrs = sorted(partitions.keys(), key=chr_sort_key)

    n_out = len(valid_indices) if valid_indices is not None else n_samples
    logger.info("Computing LOCO Kinship (streaming)")
    logger.info(
        f"  Individuals: {n_out:,}"
        + (f" (filtered from {n_samples:,})" if n_out != n_samples else "")
    )
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chromosomes: {len(unique_chrs)}")
    logger.info(f"  Chunk size: {chunk_size:,}")

    # === PASS 1: Compute per-SNP statistics for filtering ===
    # Stats are computed on ALL samples (not valid_indices subset). This is
    # intentional: SNP filter decisions (MAF, missingness) should use the full
    # population to match GEMMA's behaviour. valid_indices only affects PASS 2
    # kinship accumulation, not which SNPs are included.
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.intp)
    all_vars = np.zeros(n_snps, dtype=np.float64)

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks, desc="LOCO: SNP statistics"
        )

    n_unexpected_total = 0
    for chunk, start, end in stats_iterator:
        n_unexpected_total += validate_genotype_values(chunk)
        chunk = np.ascontiguousarray(chunk)
        compute_snp_stats_chunk(
            chunk,
            all_means[start:end],
            all_miss_counts[start:end],
            all_vars[start:end],
        )
        del chunk  # Free ~1.6GB per chunk at scale before next iteration

    if n_unexpected_total > 0:
        logger.warning(
            f"LOCO kinship genotype validation: {n_unexpected_total} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    # Cache global PASS-1 stats (computed over ALL samples) for the association
    # pass when requested. Must be built BEFORE the del statements below free
    # all_means / all_vars. n_samples is the population the stats span.
    snp_stats_cache = (
        SnpStatsCache(
            col_means=all_means.copy(),
            miss_counts=all_miss_counts.copy(),
            col_vars=all_vars.copy(),
            n_samples=n_samples,
        )
        if return_snp_stats
        else None
    )

    # Compute filters
    miss_rates = all_miss_counts / n_samples
    del all_miss_counts
    allele_freqs = all_means / 2.0
    del all_means
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)
    is_polymorphic = all_vars > 0
    del all_vars
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
    chrs_with_snps = [c for c in unique_chrs if n_chr_filtered.get(c, 0) > 0]
    chrs_without_snps = [c for c in unique_chrs if n_chr_filtered.get(c, 0) == 0]
    if chrs_without_snps:
        logger.warning(
            f"{len(chrs_without_snps)} chromosome(s) have 0 ksnps after filtering: "
            f"{chrs_without_snps}. LOCO will use full kinship for these "
            f"(nothing to leave out)."
        )
    n_chr_with_snps = len(chrs_with_snps)

    # Determine memory strategy: single-pass vs multi-pass batching
    from jamma.core.memory import _dsyevr_peak_gb

    # When valid_indices is provided, matrices are n_valid x n_valid (not n_samples).
    n_mat = len(valid_indices) if valid_indices is not None else n_samples
    matrix_gb = n_mat**2 * 8 / 1e9
    # Chunk buffer is n_samples (full disk read) regardless of valid_indices;
    # subsetting happens after load.
    chunk_buffer_gb = n_samples * chunk_size * 8 / 1e9
    # S_full + K_loco_buf + all S_chr + chunk buffer
    single_pass_gb = matrix_gb * (2 + n_chr_with_snps) + chunk_buffer_gb
    available_gb = psutil.virtual_memory().available / 1e9
    # Minimum: 3 matrices + chunk buffer + eigendecomp workspace.
    # The 3-matrix floor arises from the yield phase bottleneck:
    #   S_full + K_loco_buf + 1 remaining S_chr
    # In practice, single-pass holds all S_chr simultaneously (handled by
    # single_pass_gb above). This minimum catches the case where even
    # multi-pass with batch_size=1 won't fit.
    # Eigendecomp runs while the generator is suspended with S_chr still alive.
    # Uses DSYEVR peak (smaller driver) — eigendecompose_kinship() falls back
    # from DSYEVD to DSYEVR under memory pressure, making this self-consistent.
    eigendecomp_min_gb = _dsyevr_peak_gb(n_mat)
    min_required_gb = matrix_gb * 3 + chunk_buffer_gb + eigendecomp_min_gb

    if check_memory and min_required_gb > available_gb * 0.9:
        raise MemoryError(
            f"Insufficient memory for LOCO kinship: need at least "
            f"{min_required_gb:.1f}GB for S_full + K_loco_buf + one S_chr + "
            f"eigendecomp ({eigendecomp_min_gb:.1f}GB), "
            f"available {available_gb:.1f}GB"
        )

    # Single-pass vs multi-pass decision and the chromosomes-per-pass batch
    # size, decided up front so the test override (_max_batch_chrs) and the
    # memory-based sizing share one place.
    if _max_batch_chrs is not None:
        batch_size = _max_batch_chrs
        single_pass = n_chr_with_snps <= batch_size
    else:
        single_pass = single_pass_gb <= available_gb * 0.9
        if single_pass:
            batch_size = n_chr_with_snps  # unused in the single-pass branch
        else:
            # The consumer eigendecomposes each K_loco while the generator is
            # suspended with remaining S_chr matrices still alive. Reserve
            # eigendecomp workspace (full DSYEVR peak: K_loco + Z + O(N)) so the
            # batch doesn't exhaust memory before eigendecomp can run.
            eigendecomp_reserve_gb = _dsyevr_peak_gb(n_samples)
            usable_gb = (
                available_gb * 0.9
                - 2 * matrix_gb
                - chunk_buffer_gb
                - eigendecomp_reserve_gb
            )
            batch_size = max(1, int(usable_gb / matrix_gb))

    def _generate() -> Iterator[tuple[str, np.ndarray]]:
        if single_pass:
            # === SINGLE-PASS: accumulate S_full and all S_chr together ===
            if single_pass_gb > 10:
                logger.info(
                    f"LOCO streaming: single-pass ({single_pass_gb:.1f}GB for "
                    f"{n_chr_with_snps} chromosomes)"
                )

            S_full_np, S_chr = _stream_s_full_and_chr(
                bed_path,
                n_samples,
                n_snps,
                snp_indices,
                chromosomes,
                chrs_with_snps,
                chunk_size,
                show_progress,
                desc="LOCO: kinship accumulation",
                valid_indices=valid_indices,
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"LOCO streaming accumulation complete in {elapsed:.2f}s, "
                f"computing {len(S_chr)} LOCO matrices"
            )

            K_loco_buf = np.empty_like(S_full_np)
            yield from _yield_loco_matrices(
                S_full_np,
                S_chr,
                n_chr_filtered,
                n_filtered,
                K_loco_buf,
                copy_output=_copy_yielded_matrices,
            )
            yield from _yield_full_kinship_fallback(
                S_full_np, chrs_without_snps, n_filtered
            )
        else:
            # === MULTI-PASS: batch chromosomes across disk passes ===
            # S_full is computed once in pass 1; later passes accumulate only
            # their batch's S_chr. batch_size was decided above.
            n_batches = (n_chr_with_snps + batch_size - 1) // batch_size
            logger.warning(
                f"LOCO streaming: multi-pass mode ({n_batches} passes, "
                f"{batch_size} chromosomes/pass). Single-pass would need "
                f"{single_pass_gb:.1f}GB, available {available_gb:.1f}GB."
            )

            # First batch: compute S_full + first batch of S_chr
            first_batch = chrs_with_snps[:batch_size]
            S_full_np, S_chr = _stream_s_full_and_chr(
                bed_path,
                n_samples,
                n_snps,
                snp_indices,
                chromosomes,
                first_batch,
                chunk_size,
                show_progress,
                desc=f"LOCO: pass 1/{n_batches} (S_full + {len(first_batch)} chr)",
                valid_indices=valid_indices,
            )

            K_loco_buf = np.empty_like(S_full_np)
            yield from _yield_loco_matrices(
                S_full_np,
                S_chr,
                n_chr_filtered,
                n_filtered,
                K_loco_buf,
                copy_output=_copy_yielded_matrices,
            )
            del S_chr
            gc.collect()

            # Subsequent batches: only accumulate S_chr (S_full already computed)
            for batch_idx in range(1, n_batches):
                batch_start = batch_idx * batch_size
                batch_chrs = chrs_with_snps[batch_start : batch_start + batch_size]

                _, S_chr = _stream_s_full_and_chr(
                    bed_path,
                    n_samples,
                    n_snps,
                    snp_indices,
                    chromosomes,
                    batch_chrs,
                    chunk_size,
                    show_progress,
                    desc=(
                        f"LOCO: pass {batch_idx + 1}/{n_batches} "
                        f"({len(batch_chrs)} chr)"
                    ),
                    S_full_accum=False,
                    valid_indices=valid_indices,
                )

                yield from _yield_loco_matrices(
                    S_full_np,
                    S_chr,
                    n_chr_filtered,
                    n_filtered,
                    K_loco_buf,
                    copy_output=_copy_yielded_matrices,
                )
                del S_chr
                gc.collect()

            yield from _yield_full_kinship_fallback(
                S_full_np, chrs_without_snps, n_filtered
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"LOCO multi-pass complete in {elapsed:.2f}s, "
                f"{n_batches} passes over {n_chr_with_snps} chromosomes"
            )

    if return_snp_stats:
        return _generate(), snp_stats_cache
    return _generate()
