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
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.kinship.accumulation import accumulate_kinship
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


def _stream_kinship(
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
    """Accumulate K from one read of the BED, filtering each chunk as it arrives.

    Every BED chunk holds whole SNP columns, so GEMMA's MAF, missing-rate and
    monomorphism filter is decided per chunk from ``compute_snp_stats`` over
    the filter samples. The surviving columns are transformed over all samples
    (centering for -gk 1, standardizing for -gk 2), rows are cut to
    ``valid_indices``, and one dsyrk per file chunk accumulates K. That is the
    column grouping a stats pass followed by an accumulation pass produced, so
    K is bit-identical to it; the second read of the file is what this saves.

    Args:
        bed_path: PLINK file prefix.
        n_samples: Total sample count (disk chunk-buffer width).
        n_snps: Total SNP count.
        n_out: Kinship matrix dimension (len(valid_indices) or n_samples).
        chunk_size: SNPs per disk read.
        maf_threshold: Minimum MAF for inclusion.
        miss_threshold: Maximum missing rate for inclusion.
        show_progress: Show the progress bar.
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
    n_filter_samples = (
        n_samples if filter_sample_indices is None else len(filter_sample_indices)
    )
    K = np.zeros((n_out, n_out), dtype=np.float64)
    n_filtered = 0

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

    for chunk, file_start, file_end in chunk_iter:
        filter_chunk = (
            chunk if filter_sample_indices is None else chunk[filter_sample_indices, :]
        )
        col_means, miss_counts, col_vars = compute_snp_stats(filter_chunk)
        del filter_chunk
        keep, _afs, _mafs = compute_snp_filter_mask(
            col_means,
            miss_counts,
            col_vars,
            n_filter_samples,
            maf_threshold,
            miss_threshold,
        )
        if ksnps_indices is not None:
            keep &= np.isin(np.arange(file_start, file_end), ksnps_indices)
        local = np.flatnonzero(keep)
        if len(local) == 0:
            continue

        X = transform(chunk[:, local])
        if valid_indices is not None:
            X = X[valid_indices, :]
        accumulate_kinship(K, X)
        n_filtered += len(local)
        del chunk, X

    if n_filtered == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_snps}"
        )
    if ksnps_indices is not None:
        logger.info(
            f"Kinship SNP list: restricting to {len(ksnps_indices)} requested SNPs "
            f"({n_filtered} retained after intersection)"
        )
    if n_filtered < n_snps:
        logger.info(
            f"Kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_snps - n_filtered:,} removed (MAF/missing/monomorphic)"
        )
    else:
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")

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

    The BED is read once. Each chunk holds whole SNP columns, so the MAF,
    missing-rate and monomorphism filter is decided per chunk before that
    chunk's surviving columns are transformed and accumulated; standardization
    likewise takes its per-SNP variance from the chunk's full rows.

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

    K = _stream_kinship(
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
