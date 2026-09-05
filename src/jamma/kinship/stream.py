"""Streaming kinship matrix computation (GEMMA -gk 1 and -gk 2).

This module computes the kinship matrix K directly from disk-streamed
genotypes, in centered (-gk 1) or standardized (-gk 2) form:

    K = (1/p) * X_c @ X_c.T                     (mode="centered", -gk 1)
    K = (1/p) * Z @ Z.T, Z col-standardized      (mode="standardized", -gk 2)

where X_c/Z is imputed to per-SNP mean (and, for standardized, scaled by
per-SNP standard deviation), and p is the filtered SNP count.

LOCO (Leave-One-Chromosome-Out) kinship lives in ``jamma.kinship.loco``; it
shares SNP selection and accumulation through ``jamma.kinship.accumulation``.
LOCO owns its two-pass batching because it keeps several matrices live
(S_full plus one S_chr per chromosome) instead of one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from jamma.core import memory
from jamma.core.estimates import estimate_kinship_seconds
from jamma.core.memory import estimate_streaming_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_stats
from jamma.core.snp_stats import collect_streamed_snp_stats
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.kinship.accumulation import (
    accumulate_kinship,
    select_kinship_snps,
    selected_chunks,
)
from jamma.kinship.accumulation import (
    validate_valid_indices as validate_valid_indices,
)
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize

KinshipMode = Literal["centered", "standardized"]

_TRANSFORMS: dict[KinshipMode, Callable[[np.ndarray], np.ndarray]] = {
    "centered": impute_and_center,
    "standardized": impute_center_and_standardize,
}


def _preflight_kinship_memory(n_samples: int, chunk_size: int) -> None:
    """Gate a kinship computation on the memory that phase actually needs.

    Sizes the kinship phase alone — the accumulator plus one genotype chunk.
    Callers that go on to eigendecompose are gated separately by
    ``eigendecompose_kinship``, and whole-workflow planning happens in
    ``PipelineRunner``, so charging kinship for those phases here would refuse
    ``-gk`` runs that fit comfortably.

    Args:
        n_samples: Number of samples in the kinship matrix.
        chunk_size: SNPs per genotype chunk held during accumulation.

    Raises:
        MemoryError: If the kinship phase will not fit in available memory.
    """
    kinship_gb = estimate_streaming_memory(n_samples, chunk_size=chunk_size).kinship_gb
    memory.require(
        kinship_gb,
        memory.available_ram_gb(),
        f"kinship accumulation (peak: {kinship_gb:.1f}GB)",
    )


def _kinship_single_pass(
    bed_path: Path,
    n_samples: int,
    n_snps: int,
    chunk_size: int,
    show_progress: bool,
    valid_indices: np.ndarray | None = None,
    filter_sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Single-pass kinship: compute stats and accumulate in one BED read.

    Only valid when no MAF/missing filters are active (maf_threshold=0.0,
    miss_threshold>=1.0, ksnps_indices=None) and mode is "centered". Monomorphism
    filtering (variance > 0) is applied per-chunk inline, matching the two-pass
    result.

    Args:
        bed_path: Path prefix for PLINK files.
        n_samples: Number of samples.
        n_snps: Total number of SNPs.
        chunk_size: Number of SNPs per chunk.
        show_progress: Whether to show progress bar.
        valid_indices: Optional array of sample indices to keep. When provided,
            the kinship matrix is accumulated at (n_valid, n_valid) size directly,
            avoiding allocation of the full (n_samples, n_samples) matrix.
        filter_sample_indices: Samples used for monomorphism filtering, or all
            samples when None. Does not change centering or output dimensions.

    Returns:
        Kinship matrix (n_out, n_out) where n_out = len(valid_indices) or n_samples.

    Raises:
        ValueError: If no SNPs pass monomorphism filter.

    Note:
        ``valid_indices`` is trusted here, already validated by the sole caller
        ``compute_kinship_streaming`` at its public boundary.
    """
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
        # Per-chunk monomorphism filter: exclude constant genotype columns.
        # compute_snp_stats is the canonical variance basis (the C kernel the
        # two-pass path uses); its var > 0 mask matches np.nanvar > 0 on genotype
        # data, and it owns the all-NaN-column handling internally.
        filter_chunk = (
            chunk if filter_sample_indices is None else chunk[filter_sample_indices, :]
        )
        _col_means, _miss_counts, col_vars = compute_snp_stats(filter_chunk)
        del filter_chunk
        poly_mask = col_vars > 0
        n_poly = np.count_nonzero(poly_mask)
        if n_poly == 0:
            continue

        X_centered = impute_and_center(chunk[:, poly_mask])
        if valid_indices is not None:
            X_centered = X_centered[valid_indices, :]
        accumulate_kinship(K, X_centered)
        n_filtered += n_poly
        del chunk, X_centered

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed monomorphism filter. Original SNP count: {n_snps}"
        )

    K /= n_filtered
    return K


def _stream_kinship_two_pass(
    bed_path: Path,
    *,
    n_samples: int,
    n_snps: int,
    n_out: int,
    chunk_size: int,
    maf_threshold: float,
    miss_threshold: float,
    show_progress: bool,
    ksnps_indices: np.ndarray | None,
    valid_indices: np.ndarray | None,
    filter_sample_indices: np.ndarray | None,
    transform: Callable[[np.ndarray], np.ndarray],
    desc: str,
) -> np.ndarray:
    """Two-pass streaming kinship accumulation shared by -gk 1 and -gk 2.

    PASS 1 collects per-SNP stats and applies the MAF/missing/monomorphism filter.
    PASS 2 streams the filtered columns one file chunk at a time, applies
    ``transform`` (centering for -gk 1, standardizing for -gk 2), and accumulates
    K via dsyrk, one call per file chunk. K is scaled by the filtered SNP count.

    The transform is the only difference between the two modes; the disk-read order,
    column grouping, and accumulation are identical, so the numerics contract of
    ``selected_chunks`` holds for both.

    Args:
        bed_path: PLINK file prefix.
        n_samples: Total sample count (disk chunk-buffer width).
        n_snps: Total SNP count.
        n_out: Kinship matrix dimension (len(valid_indices) or n_samples).
        chunk_size: SNPs per disk read.
        maf_threshold: Minimum MAF for inclusion.
        miss_threshold: Maximum missing rate for inclusion.
        show_progress: Show the PASS-2 progress bar.
        ksnps_indices: Optional -ksnps restriction, or None.
        valid_indices: Sample indices to retain (already validated), or None.
        filter_sample_indices: Samples used for SNP filtering (already validated),
            or None for all BED samples.
        transform: Per-chunk preprocessing (impute_and_center or
            impute_center_and_standardize).
        desc: Progress-bar description.

    Returns:
        Kinship matrix (n_out, n_out), symmetric, scaled by the filtered SNP count.

    Raises:
        ValueError: If no SNPs pass filtering.
    """
    stats = collect_streamed_snp_stats(
        bed_path,
        n_snps=n_snps,
        n_samples=n_samples,
        chunk_size=chunk_size,
        sample_indices=filter_sample_indices,
        validate_genotypes=False,
        show_progress=show_progress,
        progress_label="Computing SNP statistics",
        dtype=np.float32,
        sample_scope="all_samples"
        if filter_sample_indices is None
        else "valid_samples",
    )
    snp_selection = select_kinship_snps(
        stats, maf_threshold, miss_threshold, ksnps_indices, n_snps
    )
    n_filtered = len(snp_selection.indices)

    if n_filtered < n_snps:
        n_removed = n_snps - n_filtered
        logger.info(
            f"Kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )
    else:
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")

    snp_indices = snp_selection.indices
    del stats, snp_selection

    K = np.zeros((n_out, n_out), dtype=np.float64)

    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )
    if show_progress:
        chunk_iter = progress_iterator(
            chunk_iter,
            total=n_chunks,
            desc=desc,
            initial_eta_seconds=estimate_kinship_seconds(n_out, n_snps),
        )

    for X_transformed, _global_idx in selected_chunks(
        chunk_iter, snp_indices, valid_indices, transform=transform
    ):
        accumulate_kinship(K, X_transformed)

    return K / n_filtered


def compute_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
    valid_indices: np.ndarray | None = None,
    mode: KinshipMode = "centered",
    *,
    filter_sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute kinship matrix from disk-streamed genotypes (GEMMA -gk 1 or -gk 2).

    ``mode="centered"`` (-gk 1) implements K = (1/p) * X_c @ X_c.T where X_c is
    centered with missing values imputed to SNP mean. ``mode="standardized"``
    (-gk 2) implements K = (1/p) * Z @ Z.T where Z is additionally scaled by
    per-SNP standard deviation. Both read genotype chunks directly from disk via
    bed-reader windowed reads, avoiding the need to load the full genotype matrix,
    so this scales past the in-memory genotype limit (see module docstring).

    Two-pass approach for filtering: PASS 1 computes per-SNP MAF, missing rate,
    and variance; PASS 2 accumulates kinship from the filtered SNPs only.
    ``mode="centered"`` additionally single-passes (stats and accumulation in one
    BED read) when no MAF/missing/ksnps filter is active, since monomorphism
    filtering can then be done inline per-chunk. ``mode="standardized"`` is
    always two-pass: standardization needs the per-SNP variance the transform
    computes over each chunk's full rows.

    Monomorphic SNPs (constant genotype) are always excluded to match GEMMA.
    Imputation, centering, and scaling always use all BED samples. SNP filtering
    uses ``filter_sample_indices``, or all samples when it is None.
    ``valid_indices`` selects output rows only, preserving the principal submatrix
    of a full-population computation without allocating the full matrix.

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation
            and reject the run if insufficient (see ``core.memory.require``).
        show_progress: If True (default), show progress bar during iteration.
        ksnps_indices: Pre-resolved column indices for -ksnps restriction, or None.
        valid_indices: Optional array of sample indices to keep. When provided,
            the kinship matrix is accumulated at (n_valid, n_valid) size directly,
            avoiding allocation of the full (n_samples, n_samples) matrix.
        mode: "centered" (-gk 1, default) or "standardized" (-gk 2).
        filter_sample_indices: Samples used for MAF, missingness, and monomorphism
            filtering. Independent of output rows; the LMM pipeline supplies its
            analysed samples even when saving a full matrix, matching GEMMA.

    Returns:
        Kinship matrix (n_out, n_out) where n_out = len(valid_indices) or n_samples.
        Symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        FileNotFoundError: If the PLINK .bed file does not exist.
        ValueError: If no SNPs pass filtering, or mode is not recognized.

    Example:
        >>> from pathlib import Path
        >>> K = compute_kinship_streaming(Path("data/my_study"), maf_threshold=0.01)
        >>> K.shape
        (1940, 1940)
    """
    if mode not in _TRANSFORMS:
        raise ValueError(
            f"invalid kinship mode {mode!r}. Use 'centered' or 'standardized'."
        )

    start_time = time.perf_counter()

    # Get dimensions without loading genotypes
    meta = get_plink_metadata(bed_path)
    n_samples = meta.n_samples
    n_snps = meta.n_snps

    if valid_indices is not None:
        validate_valid_indices(valid_indices, n_samples)
    if filter_sample_indices is not None:
        validate_valid_indices(filter_sample_indices, n_samples)

    n_out = len(valid_indices) if valid_indices is not None else n_samples

    if mode == "standardized":
        logger.info("Computing Standardized Kinship Matrix (streaming)")
    else:
        logger.info("Computing Kinship Matrix")
    logger.info(
        f"  Individuals: {n_out:,}"
        + (f" (filtered from {n_samples:,})" if n_out != n_samples else "")
    )
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chunk size: {chunk_size:,}")

    if mode == "centered":
        from jamma.core.estimates import estimate_kinship_time

        logger.info(f"  Estimated time: {estimate_kinship_time(n_out, n_snps)}")

    # Memory check before allocation.
    # Use n_samples (not n_out): stream_genotype_chunks reads full BED rows
    # at (n_samples, chunk_size), subsetting to valid_indices happens after
    # allocation. The kinship accumulator uses n_out, but passing n_samples is
    # conservative and safe.
    if check_memory:
        _preflight_kinship_memory(n_samples, chunk_size)

    # Single-pass optimization: only applies to centered mode (-gk 1) when no
    # MAF/missing filters are active and no ksnps restriction. This eliminates
    # the stats-only BED read (pass 1), halving I/O at scale (e.g. ~76 GB saved
    # at 200k samples x 95k SNPs). Standardized mode (-gk 2) always needs the
    # per-SNP variance from PASS 1, so it never single-passes.
    use_single_pass = (
        mode == "centered"
        and maf_threshold == 0.0
        and miss_threshold >= 1.0
        and ksnps_indices is None
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
            filter_sample_indices=filter_sample_indices,
        )
        elapsed = time.perf_counter() - start_time
        logger.info(f"Kinship matrix computed in {elapsed:.2f}s")
        return K

    K = _stream_kinship_two_pass(
        bed_path,
        n_samples=n_samples,
        n_snps=n_snps,
        n_out=n_out,
        chunk_size=chunk_size,
        maf_threshold=maf_threshold,
        miss_threshold=miss_threshold,
        show_progress=show_progress,
        ksnps_indices=ksnps_indices,
        valid_indices=valid_indices,
        filter_sample_indices=filter_sample_indices,
        transform=_TRANSFORMS[mode],
        desc="Computing standardized kinship"
        if mode == "standardized"
        else "Computing kinship",
    )

    elapsed = time.perf_counter() - start_time
    if mode == "standardized":
        logger.info(f"Standardized kinship matrix computed in {elapsed:.2f}s")
    else:
        logger.info(f"Kinship matrix computed in {elapsed:.2f}s")

    return K
