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
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import jit
from loguru import logger
from tqdm import tqdm

from jamma.core import configure_jax
from jamma.core.memory import check_memory_available, estimate_streaming_memory
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.kinship.missing import impute_and_center

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


def compute_centered_kinship(
    genotypes: np.ndarray,
    batch_size: int = 10000,
    check_memory: bool = True,
) -> np.ndarray:
    """Compute centered relatedness matrix (GEMMA -gk 1).

    Implements: K = (1/p) * X_c @ X_c.T
    where X_c is centered with missing values imputed to SNP mean.

    GEMMA's PlinkKin algorithm:
    1. For each SNP batch: impute missing to mean, center
    2. Accumulate K += X_batch @ X_batch.T
    3. Scale K /= n_snps

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
            Values are typically 0, 1, or 2 representing minor allele counts.
        batch_size: SNPs per batch (default 10000, matches GEMMA).
            Batching prevents memory issues with large SNP counts.
        check_memory: If True (default), check available memory before allocation
            and raise MemoryError if insufficient.

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.

    Example:
        >>> import numpy as np
        >>> X = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0]], dtype=np.float64)
        >>> K = compute_centered_kinship(X)
        >>> K.shape
        (3, 3)
        >>> np.allclose(K, K.T)  # Symmetric
        True
    """
    n_samples, n_snps = genotypes.shape

    # Memory check before allocation
    if check_memory:
        # Kinship matrix: n^2 * 8 bytes (float64)
        kinship_gb = n_samples**2 * 8 / 1e9
        # Genotypes already allocated, but batch needs workspace
        batch_gb = n_samples * batch_size * 8 / 1e9
        required_gb = kinship_gb + batch_gb
        check_memory_available(required_gb, operation="kinship computation")

    # Convert to JAX array
    X = jnp.array(genotypes, dtype=jnp.float64)

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

    # Scale by number of SNPs
    K = K / n_snps

    # Return as numpy array for downstream compatibility
    return np.array(K)


def compute_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    check_memory: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute centered relatedness matrix from disk-streamed genotypes.

    Implements: K = (1/p) * X_c @ X_c.T
    where X_c is centered with missing values imputed to SNP mean.

    This function reads genotype chunks directly from disk via bed-reader
    windowed reads, avoiding the need to load the full genotype matrix.

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
        check_memory: If True (default), check available memory before allocation
            and raise MemoryError if insufficient.
        show_progress: If True (default), show tqdm progress bar during iteration.

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        FileNotFoundError: If the PLINK .bed file does not exist.

    Example:
        >>> from pathlib import Path
        >>> K = compute_kinship_streaming(Path("data/my_study"), chunk_size=5000)
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
    logger.info(f"number of analyzed SNPs/variants = {n_snps}")
    logger.info(f"chunk size = {chunk_size}")

    # Memory check before allocation
    if check_memory:
        est = estimate_streaming_memory(n_samples, n_snps, chunk_size)
        # For kinship phase: kinship + chunk
        kinship_phase_gb = est.kinship_gb + est.chunk_gb
        check_memory_available(
            kinship_phase_gb, operation="streaming kinship computation"
        )

    # Initialize kinship accumulator
    K = jnp.zeros((n_samples, n_samples), dtype=jnp.float64)

    # Stream chunks from disk and accumulate kinship
    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )

    if show_progress:
        chunk_iter = tqdm(
            chunk_iter,
            desc="Computing kinship",
            total=n_chunks,
            unit="chunk",
        )

    for chunk, _start, _end in chunk_iter:
        # Convert to JAX array
        X_chunk = jnp.array(chunk)

        # Impute and center the chunk
        X_centered = impute_and_center(X_chunk)

        # Accumulate kinship contribution
        K = _accumulate_kinship(K, X_centered)

    # Scale by number of SNPs
    K = K / n_snps

    elapsed = time.perf_counter() - start_time
    logger.info("## Kinship matrix computed")
    logger.info(f"time elapsed = {elapsed:.2f} seconds")

    # Return as numpy array for downstream compatibility
    return np.array(K)
