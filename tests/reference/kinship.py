"""In-memory kinship oracle.

Production computes kinship by streaming genotypes from disk
(``jamma.kinship.compute.compute_kinship_streaming`` and
``compute_loco_kinship_streaming``). These in-memory functions have no
production caller; they exist so tests can hold the streaming path to a
simpler, whole-matrix-in-RAM reference implementation of the same GEMMA
-gk 1 / -gk 2 formulas.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from loguru import logger

from jamma.core.eigen_plan import array_gb, square_matrix_gb
from jamma.core.memory import check_memory_available
from jamma.core.memory_snapshot import log_memory_snapshot
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.jlinalg import dsyrk
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Return arr as float64, copying only when the dtype differs."""
    return arr if arr.dtype == np.float64 else arr.astype(np.float64)


def _accumulate_kinship(K: np.ndarray, X_centered: np.ndarray) -> None:
    """Accumulate kinship contribution from centered SNP batch.

    Uses jlinalg.dsyrk (symmetric rank-k update) with in-place accumulation.

    Args:
        K: Current kinship matrix accumulator (n_samples, n_samples)
        X_centered: Centered genotype batch (n_samples, batch_snps)

    The accumulator is mutated in place.
    """
    dsyrk(X_centered, out=K, beta=1.0)


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
        # Kinship phase only: the accumulator plus the float64 genotype matrix.
        # Callers that eigendecompose are gated by eigendecompose_kinship.
        required_gb = square_matrix_gb(n_samples) + array_gb(n_samples, n_snps)
        check_memory_available(
            required_gb,
            operation=f"kinship accumulation (peak: {required_gb:.1f}GB)",
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
        _accumulate_kinship(K, X_transformed)

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
