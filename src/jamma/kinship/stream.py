"""Streaming kinship matrix computation (GEMMA -gk 1 and -gk 2).

This module computes the kinship matrix K directly from disk-streamed
genotypes, in centered (-gk 1) or standardized (-gk 2) form:

    K = (1/p) * X_c @ X_c.T                     (mode="centered", -gk 1)
    K = (1/p) * Z @ Z.T, Z col-standardized      (mode="standardized", -gk 2)

where X_c/Z is imputed to per-SNP mean (and, for standardized, scaled by
per-SNP standard deviation), and p is the filtered SNP count.

``filtered_kinship_chunks`` owns the per-chunk SNP filter and preprocessing.
LOCO (Leave-One-Chromosome-Out) kinship in ``jamma.kinship.loco`` consumes the
same generator; it adds only the chromosome batching its several live matrices
(S_full plus one S_chr per chromosome) require.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from loguru import logger

from jamma.core import memory
from jamma.core.estimates import estimate_kinship_seconds
from jamma.core.memory import estimate_kinship_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import (
    compute_snp_filter_mask,
    compute_snp_stats,
    validate_snp_indices,
)
from jamma.core.snp_stats import SnpStats
from jamma.io.plink import (
    get_plink_metadata,
    stream_genotype_chunks,
    validate_genotype_values,
)
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


def _preflight_kinship_memory(
    *,
    n_input_samples: int,
    n_output_samples: int,
    n_snps: int,
    chunk_size: int,
    mem_budget: float | None,
) -> None:
    """Gate a kinship computation on the memory that phase actually needs.

    Sizes the kinship phase alone, including decoded and preprocessing blocks.
    Callers that go on to eigendecompose are gated separately by
    ``eigendecompose_kinship``, and whole-workflow planning happens in
    ``PipelineRunner``, so charging kinship for those phases here would refuse
    ``-gk`` runs that fit comfortably.

    Args:
        n_input_samples: Number of samples read from the BED file.
        n_output_samples: Number of samples in the kinship matrix.
        n_snps: Number of SNPs in the BED file.
        chunk_size: SNPs per genotype chunk held during accumulation.
        mem_budget: User-set ceiling in GB, or None for no ceiling.

    Raises:
        MemoryError: If the kinship phase will not fit in available memory,
            or exceeds ``mem_budget``.
    """
    kinship_gb = estimate_kinship_memory(
        n_input_samples=n_input_samples,
        n_output_samples=n_output_samples,
        n_snps=n_snps,
        chunk_size=chunk_size,
    )
    memory.require(
        kinship_gb,
        memory.available_ram_gb(),
        f"kinship accumulation (peak: {kinship_gb:.1f}GB)",
        budget_gb=mem_budget,
    )


class KinshipSnpFilter(NamedTuple):
    """GEMMA's kinship SNP filter, decided per BED chunk.

    Attributes:
        maf_threshold: Minimum MAF for inclusion.
        miss_threshold: Maximum missing rate for inclusion.
        restriction: Boolean mask over every BED SNP from -ksnps, or None.
        filter_rows: Samples the statistics are measured over, or None for
            every BED sample.
    """

    maf_threshold: float
    miss_threshold: float
    restriction: np.ndarray | None
    filter_rows: np.ndarray | None


def ksnps_restriction(
    ksnps_indices: np.ndarray | None, n_snps: int
) -> np.ndarray | None:
    """Boolean mask over ``n_snps`` for a validated -ksnps list, or None."""
    if ksnps_indices is None:
        return None
    restriction = np.zeros(n_snps, dtype=bool)
    restriction[ksnps_indices] = True
    return restriction


@dataclass(slots=True)
class SnpStatsSink:
    """Per-SNP filter statistics recorded as ``filtered_kinship_chunks`` reads.

    ``stats`` is None until the generator is exhausted, then holds the
    statistics of every BED SNP over the filter rows.
    """

    col_means: np.ndarray
    miss_counts: np.ndarray
    col_vars: np.ndarray
    n_unexpected: int = 0
    stats: SnpStats | None = None

    @classmethod
    def for_snps(cls, n_snps: int) -> SnpStatsSink:
        return cls(
            np.zeros(n_snps, dtype=np.float64),
            np.zeros(n_snps, dtype=np.intp),
            np.zeros(n_snps, dtype=np.float64),
        )


def filtered_kinship_chunks(
    bed_path: Path,
    *,
    n_snps: int,
    chunk_size: int,
    snp_filter: KinshipSnpFilter,
    transform: Callable[[np.ndarray], np.ndarray],
    output_rows: np.ndarray | None,
    show_progress: bool,
    desc: str,
    initial_eta_seconds: float | None = None,
    stats_sink: SnpStatsSink | None = None,
    wanted: Callable[[np.ndarray], bool] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Read the BED once, yielding each chunk's kinship columns ready to accumulate.

    Every BED chunk holds whole SNP columns, so GEMMA's MAF, missing-rate and
    monomorphism filter is decided per chunk from ``compute_snp_stats`` over
    the filter rows, then intersected with the -ksnps restriction. The
    surviving columns are transformed over all samples (centering for -gk 1,
    standardizing for -gk 2) and only then cut to ``output_rows``, so means
    and imputation do not depend on which rows a caller keeps. One yield per
    file chunk, so one ``accumulate_kinship`` per yield keeps the dsyrk
    column grouping of the BED chunks.

    Args:
        bed_path: PLINK file prefix.
        n_snps: Total SNP count.
        chunk_size: SNPs per disk read.
        snp_filter: The kinship SNP filter.
        transform: Per-chunk preprocessing over all samples.
        output_rows: Sample indices to keep after preprocessing, or None.
        show_progress: Show the progress bar.
        desc: Progress-bar description.
        initial_eta_seconds: Progress-bar ETA before the first chunk, or None.
        stats_sink: Records every SNP's filter statistics, or None.
        wanted: Predicate on a chunk's surviving global SNP indices; False
            skips the chunk before preprocessing. None keeps every chunk.

    Yields:
        ``(X, global_idx)``: float64 ``(n_out, n_kept)`` preprocessed columns
        and their ascending global BED indices.

    Raises:
        ValueError: On exhaustion, if no SNP passed filtering.
    """
    filter_rows = snp_filter.filter_rows
    n_filter_samples = 0
    n_kept = 0

    chunk_iter = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
    )
    if show_progress:
        chunk_iter = progress_iterator(
            chunk_iter,
            total=(n_snps + chunk_size - 1) // chunk_size,
            desc=desc,
            initial_eta_seconds=initial_eta_seconds,
        )

    for chunk, file_start, file_end in chunk_iter:
        filter_chunk = chunk if filter_rows is None else chunk[filter_rows, :]
        n_filter_samples = filter_chunk.shape[0]
        col_means, miss_counts, col_vars = compute_snp_stats(filter_chunk)
        if stats_sink is not None:
            stats_sink.col_means[file_start:file_end] = col_means
            stats_sink.miss_counts[file_start:file_end] = miss_counts
            stats_sink.col_vars[file_start:file_end] = col_vars
            stats_sink.n_unexpected += validate_genotype_values(filter_chunk)
        del filter_chunk
        keep, _afs, _mafs = compute_snp_filter_mask(
            col_means,
            miss_counts,
            col_vars,
            n_filter_samples,
            snp_filter.maf_threshold,
            snp_filter.miss_threshold,
        )
        if snp_filter.restriction is not None:
            keep &= snp_filter.restriction[file_start:file_end]
        local = np.flatnonzero(keep)
        if len(local) == 0:
            continue
        n_kept += len(local)
        global_idx = local + file_start
        if wanted is not None and not wanted(global_idx):
            continue

        X = transform(chunk if len(local) == chunk.shape[1] else chunk[:, local])
        del chunk
        if output_rows is not None:
            X = X[output_rows, :]
        yield X, global_idx
        del X

    if n_kept == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={snp_filter.maf_threshold}, "
            f"miss<={snp_filter.miss_threshold}, polymorphic). "
            f"Original SNP count: {n_snps}"
        )
    if stats_sink is not None:
        stats_sink.stats = SnpStats(
            col_means=stats_sink.col_means,
            miss_counts=stats_sink.miss_counts,
            col_vars=stats_sink.col_vars,
            n_samples=n_filter_samples,
            n_unexpected=stats_sink.n_unexpected,
        )


def _stream_kinship(
    bed_path: Path,
    *,
    n_snps: int,
    n_out: int,
    chunk_size: int,
    snp_filter: KinshipSnpFilter,
    show_progress: bool,
    valid_indices: np.ndarray | None,
    transform: Callable[[np.ndarray], np.ndarray],
    desc: str,
) -> tuple[np.ndarray, int]:
    """Accumulate K from one read of the BED; return it with the filtered SNP count.

    Args:
        bed_path: PLINK file prefix.
        n_snps: Total SNP count.
        n_out: Kinship matrix dimension (len(valid_indices) or n_samples).
        chunk_size: SNPs per disk read.
        snp_filter: The kinship SNP filter.
        show_progress: Show the progress bar.
        valid_indices: Sample indices to retain (already validated), or None.
        transform: Per-chunk preprocessing (impute_and_center or
            impute_center_and_standardize).
        desc: Progress-bar description.

    Returns:
        Kinship matrix (n_out, n_out) scaled by the filtered SNP count, and
        that count.

    Raises:
        ValueError: If no SNPs pass filtering.
    """
    K = np.zeros((n_out, n_out), dtype=np.float64)
    n_filtered = 0
    for X, global_idx in filtered_kinship_chunks(
        bed_path,
        n_snps=n_snps,
        chunk_size=chunk_size,
        snp_filter=snp_filter,
        transform=transform,
        output_rows=valid_indices,
        show_progress=show_progress,
        desc=desc,
        initial_eta_seconds=estimate_kinship_seconds(n_out, n_snps),
    ):
        accumulate_kinship(K, X)
        n_filtered += len(global_idx)
        del X

    K /= n_filtered
    return K, n_filtered


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
    mem_budget: float | None = None,
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
        mem_budget: User-set ceiling in GB, or None for no ceiling.

    Returns:
        Kinship matrix (n_out, n_out) where n_out = len(valid_indices) or n_samples.
        Symmetric, scaled by n_filtered_snps.

    Raises:
        MemoryError: If check_memory=True and the kinship phase does not fit
            available memory, or exceeds ``mem_budget``.
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
    validate_snp_indices(ksnps_indices, n_snps, "-ksnps")

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

    if check_memory:
        _preflight_kinship_memory(
            n_input_samples=n_samples,
            n_output_samples=n_out,
            n_snps=n_snps,
            chunk_size=chunk_size,
            mem_budget=mem_budget,
        )

    K, n_filtered = _stream_kinship(
        bed_path,
        n_snps=n_snps,
        n_out=n_out,
        chunk_size=chunk_size,
        snp_filter=KinshipSnpFilter(
            maf_threshold,
            miss_threshold,
            ksnps_restriction(ksnps_indices, n_snps),
            filter_sample_indices,
        ),
        show_progress=show_progress,
        valid_indices=valid_indices,
        transform=_TRANSFORMS[mode],
        desc="Computing standardized kinship"
        if mode == "standardized"
        else "Computing kinship",
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

    elapsed = time.perf_counter() - start_time
    if mode == "standardized":
        logger.info(f"Standardized kinship matrix computed in {elapsed:.2f}s")
    else:
        logger.info(f"Kinship matrix computed in {elapsed:.2f}s")

    return K
