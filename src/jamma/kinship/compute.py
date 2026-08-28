"""Kinship matrix computation.

This module provides the main kinship matrix computation function,
implementing GEMMA's centered relatedness matrix algorithm (-gk 1 mode).

The kinship matrix K is computed as:
    K = (1/p) * X_c @ X_c.T

where X_c is the centered genotype matrix with missing values imputed
to per-SNP mean, and p is the number of SNPs.

The standard kinship computation uses jlinalg.dsyrk exclusively. The
streaming LOCO function (compute_loco_kinship_streaming) uses NumPy for
accumulation.

LOCO (Leave-One-Chromosome-Out) kinship is computed via the subtraction
approach: K_loco_c = (S_full - S_c) / (p - p_c), where S_full is the
unscaled full kinship numerator and S_c is the contribution from
chromosome c. This avoids redundant computation.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, overload

import numpy as np
from loguru import logger

from jamma import jlinalg
from jamma.core import memory
from jamma.core.eigen_plan import array_gb, square_matrix_gb
from jamma.core.estimates import estimate_kinship_seconds
from jamma.core.memory import check_memory_available, estimate_streaming_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_stats
from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpSelection,
    SnpStats,
    SnpStatsCache,
    collect_streamed_snp_stats,
    filter_snp_stats,
)
from jamma.io.plink import (
    get_plink_metadata,
    partitions_from_metadata,
    stream_genotype_chunks,
)
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize
from jamma.utils import chr_sort_key


@dataclass(slots=True)
class LocoKinshipStream:
    """Consume-once stream of ``(chr_name, K_loco)`` LOCO matrices plus PASS-1 stats.

    Wraps the generator ``compute_loco_kinship_streaming`` builds internally.
    Iterating it drives disk reads and dsyrk accumulation lazily, chromosome by
    chromosome; each yielded ``K_loco`` aliases one shared ``(n, n)`` buffer that is
    overwritten on the next advance (LOCO-03, no per-chromosome allocation). Consume
    it exactly once, in order — the contract a bare generator always had. Do not call
    ``list()``/``dict()`` on it directly; use ``materialize()``, which copies each
    matrix, or you get N references to the same final buffer.

    Attributes:
        snp_stats: PASS-1 all-sample SnpStatsCache for the LOCO association pass to
            reuse, or None when ``valid_indices`` filtered the basis. SnpStatsCache is
            all-sample by contract (``__post_init__`` forbids caching a valid-sample
            basis), so a filtered run exports no cache. Available before iteration,
            since PASS 1 runs eagerly at construction time.
    """

    _matrices: Iterator[tuple[str, np.ndarray]]
    snp_stats: SnpStatsCache | None = None

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        return self._matrices

    def materialize(self) -> dict[str, np.ndarray]:
        """Drain the stream into a chr->matrix dict, copying each matrix.

        Test and diagnostic convenience. Production callers (the write path and the
        eigendecomposition path) never call this; both consume the stream once, in
        order, without collecting it. The per-chromosome buffer aliasing that governs
        live iteration does not apply to the copies, so the dict is safe to hold.
        """
        return {chr_name: K.copy() for chr_name, K in self}


def _accumulate_kinship(K: np.ndarray, X_centered: np.ndarray) -> None:
    """Accumulate kinship contribution from centered SNP batch.

    Uses jlinalg.dsyrk (symmetric rank-k update) with in-place accumulation.
    The non-LOCO kinship path uses this exclusively.

    Args:
        K: Current kinship matrix accumulator (n_samples, n_samples)
        X_centered: Centered genotype batch (n_samples, batch_snps)

    The accumulator is mutated in place.
    """
    jlinalg.dsyrk(X_centered, out=K, beta=1.0)


class CenteredChunk(NamedTuple):
    """One file chunk after column selection, row subset, and centering.

    X: float64 (n_out, n_sel), imputed and centered per column.
    global_idx: global BED SNP indices of X's columns, in column order, sorted
        ascending. ``X.shape[1] == len(global_idx)``. LOCO maps these to chromosomes.
    """

    X: np.ndarray
    global_idx: np.ndarray


def _selected_chunks(
    chunk_iter: Iterator[tuple[np.ndarray, int, int]],
    snp_indices: np.ndarray,
    valid_indices: np.ndarray | None,
    *,
    keep: Callable[[np.ndarray], bool] | None = None,
    transform: Callable[[np.ndarray], np.ndarray] = impute_and_center,
) -> Iterator[CenteredChunk]:
    """Select ``snp_indices`` columns, subset rows, transform, one yield per file chunk.

    Unifies the streaming (PASS 2) and LOCO accumulation loops, which share one
    mechanism. Pick the filtered columns of each BED chunk via searchsorted against
    the sorted global ``snp_indices``, subset rows to ``valid_indices``, and apply
    ``transform``. The single-pass monomorphism loop selects columns by a per-chunk
    ``nanvar`` mask instead and is deliberately not routed through here.

    Args:
        chunk_iter: Yields ``(chunk, file_start, file_end)`` from the genotype stream.
            ``chunk`` is float64 ``(n_samples, chunk_cols)`` over full BED rows.
        snp_indices: Global indices of SNPs that passed filtering, sorted ascending.
        valid_indices: Sample indices to keep, or None for all samples.
        keep: Optional predicate on a chunk's global indices, evaluated before any
            transform. Returning False skips the chunk with no work done, preserving
            LOCO's "skip chunks that contribute nothing" optimisation. None keeps every
            chunk with at least one selected column.
        transform: Per-chunk preprocessing applied to the selected, row-subset columns.
            Defaults to ``impute_and_center`` (GEMMA -gk 1). Pass
            ``impute_center_and_standardize`` for -gk 2; it self-computes each column's
            variance over the chunk's rows, which equals the full-sample variance since
            every retained row is present in the chunk.

    Yields:
        One CenteredChunk per surviving file chunk. Chunks with no selected columns (or
        that fail ``keep``) yield nothing.

    Numerics contract:
        Exactly one yield per file chunk, never re-batching selected columns across
        chunks and never splitting one chunk's selection. So one ``_accumulate_kinship``
        per yield reproduces the pre-refactor dsyrk column grouping, which splitting
        would not (bit-level). ``searchsorted`` runs on full BED chunk boundaries.
        Rows are subset before columns; the transform is per-column over the retained
        rows, so this is value-identical to selecting columns first.
    """
    assert snp_indices.ndim == 1, "snp_indices must be 1-D"
    assert len(snp_indices) < 2 or np.all(np.diff(snp_indices) > 0), (
        "snp_indices must be sorted ascending for searchsorted selection"
    )

    for chunk, file_start, file_end in chunk_iter:
        left = np.searchsorted(snp_indices, file_start, side="left")
        right = np.searchsorted(snp_indices, file_end, side="left")
        global_idx = snp_indices[left:right]
        if len(global_idx) == 0:
            continue
        if keep is not None and not keep(global_idx):
            continue

        rows = chunk if valid_indices is None else chunk[valid_indices, :]
        X_chunk = rows[:, global_idx - file_start]
        assert X_chunk.dtype == np.float64, (
            f"kinship accumulation requires float64 chunks (got {X_chunk.dtype}); "
            "check stream_genotype_chunks dtype arg"
        )
        yield CenteredChunk(transform(X_chunk), global_idx)


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
    est = estimate_streaming_memory(n_samples, chunk_size=chunk_size)
    check_memory_available(
        est.peak_kinship_gb,
        operation=f"kinship accumulation (peak: {est.peak_kinship_gb:.1f}GB)",
    )


def _select_kinship_snps(
    stats: SnpStats,
    maf_threshold: float,
    miss_threshold: float,
    ksnps_indices: np.ndarray | None,
    n_snps: int,
) -> SnpSelection:
    """Apply the kinship MAF/missing/monomorphism filter, raising if none pass.

    The streaming and LOCO kinship passes share this filter step exactly. Both
    build the same SnpFilterSpec (no HWE, "Kinship SNP list" restriction label)
    and raise the same message when every SNP is removed. Callers log their own
    retained/removed line afterwards, since the wording differs between passes.

    Args:
        stats: Per-SNP statistics from collect_streamed_snp_stats.
        maf_threshold: Minimum MAF for inclusion.
        miss_threshold: Maximum missing rate for inclusion.
        ksnps_indices: Optional -ksnps restriction, or None.
        n_snps: Total SNP count, for the error message.

    Returns:
        The SnpSelection of SNPs that passed filtering.

    Raises:
        ValueError: If no SNPs pass filtering.
    """
    selection = filter_snp_stats(
        stats,
        SnpFilterSpec(
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            restrict_indices=ksnps_indices,
            restrict_label="Kinship SNP list",
        ),
    )
    if len(selection.indices) == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_snps}"
        )
    return selection


def compute_standardized_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
    valid_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute standardized kinship (GEMMA -gk 2) from disk-streamed genotypes.

    Implements K = (1/p) * Z @ Z.T where Z[i,k] = (x[i,k] - mean_k) / sd_k, reading
    genotype chunks from disk instead of loading the full matrix. This is the
    streaming counterpart of ``compute_standardized_kinship`` and lets -gk 2 scale
    past the in-memory genotype limit, exactly as ``compute_kinship_streaming`` does
    for -gk 1.

    Always two-pass (PASS 1 stats/filter, PASS 2 standardize + accumulate); the
    single-pass -gk 1 optimisation does not apply, because standardization needs the
    per-SNP variance the transform computes over each chunk's full rows.

    Note: Monomorphic SNPs (zero variance) are excluded by the PASS-1 filter, matching
    GEMMA and the in-memory path.

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        maf_threshold: Minimum MAF for SNP inclusion (default 0.0 = no filter).
        miss_threshold: Maximum missing rate (default 1.0 = no filter).
        check_memory: If True (default), check available memory before allocation.
        show_progress: If True (default), show progress bar during iteration.
        ksnps_indices: Pre-resolved column indices for -ksnps restriction, or None.
        valid_indices: Optional array of sample indices to keep. When provided, the
            kinship matrix is accumulated at (n_valid, n_valid) size directly.

    Returns:
        Standardized kinship matrix (n_out, n_out) where n_out = len(valid_indices)
        or n_samples. Symmetric, scaled by the filtered SNP count.

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        FileNotFoundError: If the PLINK .bed file does not exist.
        ValueError: If no SNPs pass filtering.
    """
    start_time = time.perf_counter()

    meta = get_plink_metadata(bed_path)
    n_samples = meta.n_samples
    n_snps = meta.n_snps

    if valid_indices is not None:
        validate_valid_indices(valid_indices, n_samples)

    n_out = len(valid_indices) if valid_indices is not None else n_samples

    logger.info("Computing Standardized Kinship Matrix (streaming)")
    logger.info(
        f"  Individuals: {n_out:,}"
        + (f" (filtered from {n_samples:,})" if n_out != n_samples else "")
    )
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chunk size: {chunk_size:,}")

    if check_memory:
        _preflight_kinship_memory(n_samples, chunk_size)

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
        transform=impute_center_and_standardize,
        desc="Computing standardized kinship",
    )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Standardized kinship matrix computed in {elapsed:.2f}s")

    return K


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
        # Early valid-sample subsetting: compute stats on valid samples only.
        if valid_indices is not None:
            chunk = chunk[valid_indices, :]

        # Per-chunk monomorphism filter: exclude constant genotype columns.
        # compute_snp_stats is the canonical variance basis (the C kernel the
        # two-pass path uses); its var > 0 mask matches np.nanvar > 0 on genotype
        # data, and it owns the all-NaN-column handling internally.
        _col_means, _miss_counts, col_vars = compute_snp_stats(chunk)
        poly_mask = col_vars > 0
        n_poly = np.count_nonzero(poly_mask)
        if n_poly == 0:
            continue

        X_chunk = chunk[:, poly_mask]
        X_centered = impute_and_center(X_chunk)
        _accumulate_kinship(K, X_centered)
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
    n_samples = meta.n_samples
    n_snps = meta.n_snps

    if valid_indices is not None:
        validate_valid_indices(valid_indices, n_samples)

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
    # allocation. The kinship accumulator uses n_out, but passing n_samples is
    # conservative and safe.
    if check_memory:
        _preflight_kinship_memory(n_samples, chunk_size)

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
        transform=impute_and_center,
        desc="Computing kinship",
    )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Kinship matrix computed in {elapsed:.2f}s")

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
    ``_selected_chunks`` holds for both.

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
        sample_indices=valid_indices,
        validate_genotypes=False,
        show_progress=show_progress,
        progress_label="Computing SNP statistics",
        dtype=np.float32,
        sample_scope="valid_samples" if valid_indices is not None else "all_samples",
    )
    snp_selection = _select_kinship_snps(
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

    for X_transformed, _global_idx in _selected_chunks(
        chunk_iter, snp_indices, valid_indices, transform=transform
    ):
        _accumulate_kinship(K, X_transformed)

    return K / n_filtered


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
    K_loco_buf: np.ndarray,
) -> Iterator[tuple[str, np.ndarray]]:
    """Compute and yield LOCO kinship matrices from S_full and per-chr accumulators.

    For each chromosome, computes K_loco = (S_full - S_chr[c]) / (p - p_c),
    freeing S_chr[c] after each yield.

    Each yielded matrix IS the shared ``K_loco_buf``, overwritten on the next
    iteration (LOCO-03: no per-chromosome allocation). This is the consume-once
    contract ``LocoKinshipStream`` documents; consumers that need every matrix at
    once go through ``LocoKinshipStream.materialize()``, which copies.

    Args:
        S_full_np: Full kinship numerator as numpy array (n_samples, n_samples).
        S_chr: Per-chromosome kinship contributions.
        n_chr_filtered: Count of filtered SNPs per chromosome.
        n_filtered: Total number of filtered SNPs.
        K_loco_buf: Pre-allocated workspace (n_samples, n_samples) reused for every
            K_loco via ``np.subtract(out=)``, avoiding a per-chromosome temporary.

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

        # In-place subtraction avoids a temporary array (LOCO-03). K_loco is the
        # shared buffer; the consume-once contract lets sequential consumers reuse
        # it and avoid one extra n x n allocation per chromosome.
        np.subtract(S_full_np, np.asarray(S_chr[chr_name]), out=K_loco_buf)
        K_loco_buf /= p_loco
        logger.debug(
            f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs retained"
        )
        del S_chr[chr_name]
        yield (chr_name, K_loco_buf)


def validate_valid_indices(valid_indices: np.ndarray, n_samples: int) -> None:
    """Validate valid_indices for emptiness, bounds, duplicates, and ordering.

    The single source of truth for the sample-subset invariant. Called once per
    public entry path (``compute_kinship_streaming``,
    ``compute_loco_kinship_streaming``, and ``PipelineRunner._load_or_compute_kinship``
    before its ``np.ix_`` file subset). Internal helpers below a validating boundary
    trust the value and do not re-check.

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


@overload
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
    *,
    S_full_accum: Literal[True] = ...,
    valid_indices: np.ndarray | None = ...,
) -> tuple[np.ndarray, dict[str, np.ndarray]]: ...


@overload
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
    *,
    S_full_accum: Literal[False],
    valid_indices: np.ndarray | None = ...,
) -> tuple[None, dict[str, np.ndarray]]: ...


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
    *,
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

    Note:
        ``valid_indices`` is trusted here, already validated by
        ``compute_loco_kinship_streaming`` at its public boundary.
    """
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

    def keep(global_idx: np.ndarray) -> bool:
        # Skip centering when S_full isn't needed and no target chromosome is present.
        return S_full is not None or not chr_set.isdisjoint(chromosomes[global_idx])

    for X_centered, global_idx in _selected_chunks(
        chunk_iter, snp_indices, valid_indices, keep=keep
    ):
        if S_full is not None:
            _accumulate_kinship(S_full, X_centered)

        chunk_chrs = chromosomes[global_idx]
        for chr_name in set(chunk_chrs) & chr_set:
            X_chr_part = X_centered[:, chunk_chrs == chr_name]
            _accumulate_kinship(S_chr[chr_name], X_chr_part)

    return S_full, S_chr


class _LocoPassPlan(NamedTuple):
    """Memory-sizing decision for streaming LOCO kinship.

    single_pass: accumulate S_full and every per-chromosome S_chr together.
    batch_size: chromosomes processed per disk pass when multi-pass.
    single_pass_gb / min_required_gb: peak estimates surfaced for logging and
        the memory-preflight guard.
    """

    single_pass: bool
    batch_size: int
    single_pass_gb: float
    min_required_gb: float
    eigendecomp_min_gb: float


def _decide_loco_passes(
    n_mat: int,
    n_samples: int,
    n_chr_with_snps: int,
    chunk_size: int,
    available_gb: float,
    *,
    max_batch_chrs: int | None,
) -> _LocoPassPlan:
    """Decide single-pass vs multi-pass and the chromosomes-per-pass batch size.

    Pure sizing math (no I/O), so it can be unit-tested at scale. The live
    matrices are ``n_mat x n_mat`` — ``n_mat`` is ``len(valid_indices)`` when
    sample filtering is active, else ``n_samples`` — while the disk chunk buffer
    is always ``n_samples`` wide (subsetting happens after the full read). The
    eigendecomposition runs on the ``n_mat``-sized K_loco, so its workspace
    reserve is sized by ``n_mat`` too (NOT ``n_samples``); using ``n_samples``
    over-reserves on filtered datasets and can collapse ``batch_size`` to 1,
    forcing many redundant BED passes.

    Args:
        n_mat: Live matrix dimension (valid-sample count, or n_samples).
        n_samples: Total sample count (disk chunk-buffer width).
        n_chr_with_snps: Number of chromosomes that retain SNPs after filtering.
        chunk_size: SNPs per disk read.
        available_gb: Available RAM in GB (caller reads psutil and passes it in).
        max_batch_chrs: Test override forcing the chromosomes-per-pass count;
            None for memory-based sizing.

    Returns:
        A _LocoPassPlan with the decision and the peak estimates.
    """
    from jamma.core.eigen_plan import dsyevr_peak_gb

    matrix_gb = square_matrix_gb(n_mat)
    # Chunk buffer is n_samples (full disk read) regardless of valid_indices;
    # subsetting happens after load.
    chunk_buffer_gb = array_gb(n_samples, chunk_size)
    # S_full + K_loco_buf + all S_chr + chunk buffer
    single_pass_gb = matrix_gb * (2 + n_chr_with_snps) + chunk_buffer_gb
    # Minimum: 3 matrices (S_full + K_loco_buf + 1 remaining S_chr) + chunk
    # buffer + eigendecomp workspace (DSYEVR peak on the n_mat-sized K_loco).
    eigendecomp_min_gb = dsyevr_peak_gb(n_mat)
    min_required_gb = matrix_gb * 3 + chunk_buffer_gb + eigendecomp_min_gb

    if max_batch_chrs is not None:
        batch_size = max_batch_chrs
        single_pass = n_chr_with_snps <= batch_size
    else:
        single_pass = single_pass_gb <= available_gb * 0.9
        if single_pass:
            batch_size = n_chr_with_snps  # unused in the single-pass branch
        else:
            # The consumer eigendecomposes each K_loco while the generator is
            # suspended with remaining S_chr matrices still alive. Reserve
            # eigendecomp workspace (DSYEVR peak on the n_mat-sized K_loco) so
            # the batch doesn't exhaust memory before eigendecomp can run.
            eigendecomp_reserve_gb = dsyevr_peak_gb(n_mat)
            usable_gb = (
                available_gb * 0.9
                - 2 * matrix_gb
                - chunk_buffer_gb
                - eigendecomp_reserve_gb
            )
            batch_size = max(1, int(usable_gb / matrix_gb))

    return _LocoPassPlan(
        single_pass=single_pass,
        batch_size=batch_size,
        single_pass_gb=single_pass_gb,
        min_required_gb=min_required_gb,
        eigendecomp_min_gb=eigendecomp_min_gb,
    )


def compute_loco_kinship_streaming(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
    valid_indices: np.ndarray | None = None,
    *,
    _max_batch_chrs: int | None = None,
) -> LocoKinshipStream:
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
        _max_batch_chrs: Debug override forcing a fixed chromosomes-per-pass
            batch size (bypasses memory-based sizing). Used by tests to exercise
            multi-pass without mocking psutil.

    Returns:
        A consume-once LocoKinshipStream. Iterate it for (chr_name, K_loco) pairs,
        where chr_name is the chromosome being excluded and K_loco has shape
        (n_valid, n_valid) when valid_indices is provided, else
        (n_samples, n_samples). Read ``.snp_stats`` for the PASS-1 all-sample
        SnpStatsCache the LOCO association pass reuses; it is None when
        valid_indices filtered the basis (the all-sample cache is then neither
        valid nor consumable). Each yielded matrix aliases a shared buffer
        overwritten on the next advance, so consume each before advancing, or call
        ``.materialize()`` to collect independent copies.

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
    n_samples = meta.n_samples
    n_snps = meta.n_snps
    chromosomes = meta.chromosome

    if valid_indices is not None:
        validate_valid_indices(valid_indices, n_samples)

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
    # Compute stats over the analysed (valid) samples, matching GEMMA's basis and
    # the non-LOCO streaming path. The kinship SNP filter then selects SNPs on the
    # same basis PASS 2 accumulates on, so a SNP that is (near-)monomorphic among
    # the analysed individuals is not admitted just because it varies in samples
    # that were dropped.
    stats = collect_streamed_snp_stats(
        bed_path,
        n_snps=n_snps,
        n_samples=n_samples,
        chunk_size=chunk_size,
        sample_indices=valid_indices,
        validate_genotypes=True,
        show_progress=show_progress,
        progress_label="LOCO: SNP statistics",
        dtype=np.float32,
        sample_scope="valid_samples" if valid_indices is not None else "all_samples",
    )

    if stats.n_unexpected > 0:
        logger.warning(
            f"LOCO kinship genotype validation: {stats.n_unexpected} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    # Export the PASS-1 stats on the returned stream, but only when they are the
    # all-sample basis (valid_indices is None). SnpStatsCache is all-sample by
    # contract, and _chr_snp_stats_for_loco reuses it only when every sample is
    # analysed; on filtered runs it re-derives valid-sample stats, so no cache is
    # exported (None). The cache is a wrapper over the O(n_snps) arrays PASS 1
    # already holds, so building it always (when all-sample) costs a dataclass, not
    # a copy — the write-path caller simply never reads stream.snp_stats.
    snp_stats_cache = (
        SnpStatsCache(
            col_means=stats.col_means,
            miss_counts=stats.miss_counts,
            col_vars=stats.col_vars,
            n_samples=stats.n_samples,
            n_unexpected=stats.n_unexpected,
            hwe_counts=stats.hwe_counts,
            global_indices=stats.global_indices,
            sample_scope=stats.sample_scope,
        )
        if valid_indices is None
        else None
    )

    snp_selection = _select_kinship_snps(
        stats, maf_threshold, miss_threshold, ksnps_indices, n_snps
    )
    n_filtered = len(snp_selection.indices)

    if n_filtered < n_snps:
        n_removed = n_snps - n_filtered
        logger.info(
            f"LOCO kinship filtering: {n_filtered:,} SNPs retained, "
            f"{n_removed:,} removed (MAF/missing/monomorphic)"
        )

    # Build SNP-to-chromosome mapping for filtered SNPs. The PASS-1 stats are no
    # longer needed: snp_stats_cache (if any) was already built from stats above.
    snp_indices = snp_selection.indices
    del snp_selection, stats

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

    # Determine memory strategy: single-pass vs multi-pass batching.
    # When valid_indices is provided, matrices are n_valid x n_valid (not
    # n_samples); the chromosomes-per-pass sizing lives in _decide_loco_passes.
    n_mat = len(valid_indices) if valid_indices is not None else n_samples
    available_gb = memory.available_ram_gb()
    plan = _decide_loco_passes(
        n_mat,
        n_samples,
        n_chr_with_snps,
        chunk_size,
        available_gb,
        max_batch_chrs=_max_batch_chrs,
    )

    if check_memory and plan.min_required_gb > available_gb * 0.9:
        raise MemoryError(
            f"Insufficient memory for LOCO kinship: need at least "
            f"{plan.min_required_gb:.1f}GB for S_full + K_loco_buf + one S_chr + "
            f"eigendecomp ({plan.eigendecomp_min_gb:.1f}GB), "
            f"available {available_gb:.1f}GB"
        )

    single_pass = plan.single_pass
    batch_size = plan.batch_size
    single_pass_gb = plan.single_pass_gb

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

    # snp_stats is None on filtered-sample runs (valid_indices given), where the
    # all-sample cache would be neither valid nor consumed. The write-path caller
    # ignores it; the LOCO association pass reads it and re-derives valid-sample
    # stats itself when it is None.
    return LocoKinshipStream(_matrices=_generate(), snp_stats=snp_stats_cache)
