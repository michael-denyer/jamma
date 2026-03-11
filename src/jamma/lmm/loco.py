"""LOCO LMM orchestrator.

Runs leave-one-chromosome-out LMM association by looping over chromosomes:
for each chromosome c, eigendecompose K_loco_c, run LMM on chromosome c's
SNPs using that eigendecomposition, discard K_loco_c.

Memory profile (sequential processing):
    At any point holds S_full (n^2*8) from the LOCO kinship generator,
    plus one K_loco (n^2*8) during eigendecomp, plus LMM working set.
    Each K_loco is discarded after eigendecomp.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
from bed_reader import open_bed
from loguru import logger

from jamma.core.snp_filter import compute_snp_filter_mask
from jamma.core.threading import (
    blas_threads,
    get_loco_worker_count,
    get_physical_core_count,
)
from jamma.io.plink import (
    get_plink_metadata,
    partitions_from_metadata,
    stream_genotype_chunks,
    validate_genotype_values,
)
from jamma.kinship import write_kinship_matrix
from jamma.kinship.missing import impute_and_center
from jamma.lmm.compute_numpy import _compute_lmm_chunk_numpy
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy
from jamma.lmm.loco_eigen_update import secular_eigendecompose_from_full
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    compute_and_log_pve,
)
from jamma.lmm.results import (
    _yield_chunk_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.runner_numpy import _compute_chunk_size_numpy
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.schema import LazySnpMeta, LocoResult
from jamma.lmm.stats import AssocResult
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

    This eliminates 22+ per-chromosome BED re-reads in _collect_chr_snp_stats.
    """

    col_means: np.ndarray  # shape (n_snps_total,), float64
    miss_counts: np.ndarray  # shape (n_snps_total,), int32
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


class LocoStreamingMode(Enum):
    """Controls what _compute_loco_kinship_streaming_numpy yields.

    Encodes the four mutually-exclusive streaming modes as a sum type.
    Replaces three boolean flags (yield_s_chr, yield_x_c, yield_x_c_sequential)
    which encoded only 4 valid states out of 8 possible flag combinations.

    Members:
        DEFAULT: yields (loco_iter, snp_stats_cache) — (chr_name, K_loco) pairs.
        S_CHR: yields (n_filtered, s_chr_iter, snp_stats_cache) — raw Gram matrices.
        X_C: yields (n_filtered, x_c_iter, snp_stats_cache) — genotype columns.
        X_C_SEQUENTIAL: returns SequentialLocoResult — two-pass O(max_X_c) path.
    """

    DEFAULT = "default"
    S_CHR = "s_chr"
    X_C = "x_c"
    X_C_SEQUENTIAL = "x_c_sequential"


class SequentialLocoResult(NamedTuple):
    """Return value of _compute_loco_kinship_streaming_numpy(mode=X_C_SEQUENTIAL).

    s_full is the unnormalised kinship accumulator from Pass 1.
    Divide by n_filtered to obtain K_full (done by caller).
    NamedTuple subclasses tuple, so positional unpacking
    ``S, n, gen, cache = result`` works directly.
    """

    s_full: np.ndarray
    n_filtered: int
    generator: Iterator[tuple[str, np.ndarray, int]]
    snp_stats_cache: SnpStatsCache


def _collect_chr_snp_stats(
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    valid_indices: np.ndarray,
    col_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect per-SNP statistics for one chromosome via chunked BED reads.

    Shared by both JAX and NumPy LOCO chromosome runners (pass-1 logic).

    Args:
        bed_path: PLINK file prefix (without extension).
        chr_snp_indices: Global column indices for this chromosome's SNPs.
        valid_indices: Row indices of valid (non-missing) samples.
        col_chunk_size: Number of SNP columns per disk read chunk.

    Returns:
        Tuple of (col_means, miss_counts, col_vars, n_unexpected) where
        arrays are of length len(chr_snp_indices).
    """
    n_chr_snps = len(chr_snp_indices)
    col_means = np.zeros(n_chr_snps, dtype=np.float64)
    miss_counts = np.zeros(n_chr_snps, dtype=np.int32)
    col_vars = np.zeros(n_chr_snps, dtype=np.float64)
    n_unexpected_total = 0

    bed_file = Path(f"{bed_path}.bed")
    with open_bed(bed_file) as bed:
        for chunk_start in range(0, n_chr_snps, col_chunk_size):
            chunk_end = min(chunk_start + col_chunk_size, n_chr_snps)
            chunk_col_indices = chr_snp_indices[chunk_start:chunk_end]

            geno_chunk = bed.read(
                index=np.s_[valid_indices, chunk_col_indices],
                dtype=np.float64,
            )

            n_unexpected_total += validate_genotype_values(geno_chunk)

            chunk_miss = np.sum(np.isnan(geno_chunk), axis=0)
            with np.errstate(invalid="ignore"):
                chunk_means = np.nanmean(geno_chunk, axis=0)
                chunk_vars = np.nanvar(geno_chunk, axis=0)
            chunk_means = np.nan_to_num(chunk_means, nan=0.0)
            chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

            col_means[chunk_start:chunk_end] = chunk_means
            miss_counts[chunk_start:chunk_end] = chunk_miss
            col_vars[chunk_start:chunk_end] = chunk_vars

            del geno_chunk

    return col_means, miss_counts, col_vars, n_unexpected_total


def _filter_chr_snps(
    col_means: np.ndarray,
    miss_counts: np.ndarray,
    col_vars: np.ndarray,
    n_samples: int,
    maf_threshold: float,
    miss_threshold: float,
    chr_snp_indices: np.ndarray,
    snps_global_mask: np.ndarray | None,
    n_unexpected_total: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Apply SNP filtering and log warnings. Returns None if no SNPs pass.

    Shared by both JAX and NumPy LOCO chromosome runners.

    Args:
        col_means: Per-SNP means from pass-1.
        miss_counts: Per-SNP missing counts from pass-1.
        col_vars: Per-SNP variances from pass-1.
        n_samples: Number of valid samples.
        maf_threshold: Minimum MAF.
        miss_threshold: Maximum missing rate.
        chr_snp_indices: Global column indices for this chromosome.
        snps_global_mask: Boolean mask for -snps restriction, or None.
        n_unexpected_total: Count of unexpected genotype values from pass-1.
        show_progress: Whether to log progress.

    Returns:
        Tuple of (local_filtered_indices, global_filtered_indices,
        filtered_afs, filtered_miss, filtered_means) or None if empty.
    """
    if n_unexpected_total > 0:
        logger.warning(
            f"LOCO chr genotype validation: {n_unexpected_total} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, miss_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )

    if snps_global_mask is not None:
        snp_mask &= snps_global_mask[chr_snp_indices]

    local_filtered_indices = np.where(snp_mask)[0]
    n_filtered = len(local_filtered_indices)

    if show_progress:
        logger.debug(
            f"  Chromosome SNPs: {len(chr_snp_indices)}, after filter: {n_filtered}"
        )

    if n_filtered == 0:
        logger.warning(
            f"  Chromosome ({len(chr_snp_indices)} SNPs) has no SNPs after "
            f"filtering, skipping"
        )
        return None

    global_filtered_indices = chr_snp_indices[local_filtered_indices]
    filtered_afs = allele_freqs[local_filtered_indices]
    filtered_miss = miss_counts[local_filtered_indices].astype(int)
    filtered_means = col_means[local_filtered_indices]

    return (
        local_filtered_indices,
        global_filtered_indices,
        filtered_afs,
        filtered_miss,
        filtered_means,
    )


def _compute_loco_kinship_streaming_numpy(
    bed_path: Path,
    chunk_size: int = 10_000,
    maf_threshold: float = 0.0,
    miss_threshold: float = 1.0,
    check_memory: bool = True,
    show_progress: bool = True,
    ksnps_indices: np.ndarray | None = None,
    valid_indices: np.ndarray | None = None,
    _max_batch_chrs: int | None = None,
    mode: LocoStreamingMode = LocoStreamingMode.DEFAULT,
) -> (
    tuple[Iterator[tuple[str, np.ndarray]], SnpStatsCache]
    | tuple[int, Iterator[tuple[str, np.ndarray, int]], SnpStatsCache]
    | SequentialLocoResult
):
    """Compute LOCO kinship matrices using pure NumPy (no JAX dependency).

    Mirrors compute_loco_kinship_streaming from jamma.kinship but uses
    np.matmul instead of jnp.matmul. Supports multi-pass chromosome batching
    when memory is insufficient for all S_chr simultaneously (mirrors the JAX
    path in jamma.kinship.compute_loco_kinship_streaming).

    When valid_indices is provided, kinship matrices are computed at valid-sample
    size (n_valid x n_valid) rather than full n_samples x n_samples, avoiding
    full-matrix materialisation when there are missing-phenotype samples (LOCO-07).

    Args:
        bed_path: Path prefix for PLINK files (without extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate.
        check_memory: If True, check available memory before allocation.
        show_progress: If True, show progress bars.
        ksnps_indices: Pre-resolved column indices for -ksnps restriction.
        valid_indices: Row indices of valid samples. When provided, genotypes
            are subsetted to these rows before accumulation so K_loco is
            n_valid x n_valid. None means use all rows (full n_samples matrix).
        _max_batch_chrs: Debug override for batch_size_chrs. When set, forces
            multi-pass mode with at most this many chromosomes per pass. Used
            by tests to verify multi-pass equivalence without mocking psutil.
        mode: Controls what this function yields. One of:
            LocoStreamingMode.DEFAULT (default) — yields (loco_iter, snp_stats_cache)
                where loco_iter yields (chr_name, K_loco) 2-tuples.
            LocoStreamingMode.S_CHR — yields (n_filtered, s_chr_iter, snp_stats_cache)
                where s_chr_iter yields (chr_name, S_chr, p_chr) 3-tuples with
                raw (un-normalised) chromosome Gram matrices.
            LocoStreamingMode.X_C — yields (n_filtered, x_c_iter, snp_stats_cache)
                where x_c_iter yields (chr_name, X_c, p_chr) 3-tuples with centered
                chromosome genotype matrices.
            LocoStreamingMode.X_C_SEQUENTIAL — returns SequentialLocoResult with
                .s_full (unnormalised; divide by .n_filtered to get K_full),
                .n_filtered, .generator (lazy per-chromosome X_c), .snp_stats_cache.

    Returns:
        When mode=DEFAULT: Tuple of (loco_iter, snp_stats_cache) where loco_iter
        yields (chr_name, K_loco) pairs and snp_stats_cache holds global SNP
        statistics from PASS 1. K_loco matrices are n_valid x n_valid when
        valid_indices is provided.

        When mode=S_CHR: Tuple of (n_filtered, s_chr_iter, snp_stats_cache)
        where n_filtered is p_full and s_chr_iter yields (chr_name, S_chr, p_chr)
        3-tuples with the raw (un-normalised) chromosome Gram matrices.

        When mode=X_C: Tuple of (n_filtered, x_c_iter, snp_stats_cache)
        where n_filtered is p_full and x_c_iter yields (chr_name, X_c, p_chr)
        3-tuples with the centered chromosome genotype matrices.

        When mode=X_C_SEQUENTIAL: SequentialLocoResult where .s_full is the
        unnormalised full kinship accumulator (divide by .n_filtered to get K_full),
        and .generator lazily yields (chr_name, X_c, p_chr) one chromosome at a time.
    """
    import psutil

    start_time = time.perf_counter()

    meta = get_plink_metadata(bed_path)
    n_samples = meta["n_samples"]
    n_snps = meta["n_snps"]
    chromosomes = meta["chromosome"]

    # Derive partitions from already-loaded metadata — avoids re-opening BED (LOCO-04)
    partitions = partitions_from_metadata(meta)
    unique_chrs = sorted(partitions.keys(), key=chr_sort_key)

    logger.info("Computing LOCO Kinship (streaming, NumPy)")
    logger.info(f"  Individuals: {n_samples:,}")
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chromosomes: {len(unique_chrs)}")
    logger.info(f"  Chunk size: {chunk_size:,}")

    # Lazy import: progress_iterator pulls in tqdm (optional dependency)
    if show_progress:
        from jamma.core.progress import progress_iterator
    n_chunks = (n_snps + chunk_size - 1) // chunk_size

    # === PASS 1: SNP statistics for filtering ===
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)
    n_unexpected_total = 0

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks, desc="LOCO: SNP statistics (NumPy)"
        )

    for chunk, start, end in stats_iterator:
        n_unexpected_total += validate_genotype_values(chunk)
        chunk_miss_counts = np.sum(np.isnan(chunk), axis=0)
        with np.errstate(invalid="ignore"):
            chunk_means = np.nanmean(chunk, axis=0)
            chunk_vars = np.nanvar(chunk, axis=0)
        chunk_means = np.nan_to_num(chunk_means, nan=0.0)
        chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

        all_means[start:end] = chunk_means
        all_miss_counts[start:end] = chunk_miss_counts
        all_vars[start:end] = chunk_vars
        del chunk

    if n_unexpected_total > 0:
        logger.warning(
            f"LOCO kinship genotype validation: {n_unexpected_total} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    # Cache global stats for the association pass (LOCO-01).
    # Must be built BEFORE the del statements below destroy all_means / all_vars.
    # n_samples is the population these stats were computed over (ALL rows).
    snp_stats_cache = SnpStatsCache(
        col_means=all_means.copy(),
        miss_counts=all_miss_counts.copy(),
        col_vars=all_vars.copy(),
        n_samples=n_samples,
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

    snp_indices = np.where(snp_mask)[0]
    chr_for_filtered = chromosomes[snp_indices]

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

    # Memory strategy: single-pass vs multi-pass chromosome batching (LOCO-02).
    # n_samples_kinship is the size of kinship matrices — either n_valid (when
    # valid_indices is provided for early subsetting) or full n_samples (LOCO-07).
    n_samples_kinship = len(valid_indices) if valid_indices is not None else n_samples
    from jamma.core.memory import _dsyevr_peak_gb

    matrix_gb = n_samples_kinship**2 * 8 / 1e9
    # Chunk buffer is n_samples (full BED rows) — subsetting happens after read.
    chunk_buffer_gb = n_samples * chunk_size * 8 / 1e9
    n_chr_with_snps = len(chrs_with_snps)
    # S_full + K_loco_buf + all S_chr + chunk buffer
    single_pass_gb = matrix_gb * (2 + n_chr_with_snps) + chunk_buffer_gb
    available_gb = psutil.virtual_memory().available / 1e9
    # Minimum: 3 matrices (S_full + K_loco_buf + 1 S_chr) + chunk buffer +
    # eigendecomp workspace. This catches the case where even multi-pass with
    # batch_size=1 won't fit. Eigendecomp runs while generator is suspended
    # with S_chr still alive.
    # Uses DSYEVR peak (smaller driver) — eigendecompose_kinship() falls back
    # from DSYEVD to DSYEVR under memory pressure, making this self-consistent.
    eigendecomp_min_gb = _dsyevr_peak_gb(n_samples_kinship)
    min_required_gb = matrix_gb * 3 + chunk_buffer_gb + eigendecomp_min_gb

    _standard_path = mode == LocoStreamingMode.DEFAULT
    if check_memory and _standard_path and min_required_gb > available_gb * 0.9:
        raise MemoryError(
            f"Insufficient memory for NumPy LOCO kinship: need at least "
            f"{min_required_gb:.1f}GB for S_full + K_loco_buf + one S_chr + "
            f"eigendecomp ({eigendecomp_min_gb:.1f}GB), "
            f"available {available_gb:.1f}GB"
        )

    # Determine batch size: _max_batch_chrs overrides memory-based sizing (tests).
    # INVARIANT: accumulate_s_full must be True ONLY for the first pass (batch_idx==0).
    # If True for subsequent passes, S_full is accumulated multiple times, corrupting
    # all K_loco matrices — each K_loco would be subtracted from an inflated S_full.
    if mode == LocoStreamingMode.X_C_SEQUENTIAL:
        # Sequential X_c path: two BED passes reduce peak memory from O(sum_X_c)
        # to O(max_X_c). Pass 1 accumulates S_full synchronously; Pass 2 is a
        # lazy generator that re-reads the BED file once per chromosome.
        #
        # Peak memory phases (non-overlapping):
        #   Phase 1 (eigendecomp): S_full/K_full (in-place /=) + eigendecomp
        #     workspace + chunk buffer.
        #   Phase 2 (per-chr secular): U_full + d_full + one X_c + secular
        #     workspace (4 * r_eff * n for stored intermediates in delta path,
        #     plus row-batch buffers). r_eff estimated from max chromosome SNPs.
        # Phase 1 dominates because eigendecomp workspace >> one X_c.
        max_chr_snps = max(n_chr_filtered.values()) if n_chr_filtered else 0
        max_chr_gb = max_chr_snps * n_samples_kinship * 8 / 1e9
        # Phase 1 peak: K_full (in-place from S_full) + eigendecomp workspace
        phase1_gb = matrix_gb + eigendecomp_min_gb + chunk_buffer_gb
        # Phase 2 peak: U_full + d_full + one X_c + secular workspace
        # Secular workspace: 4 arrays of (r_eff, n) for stored intermediates
        # + 3 arrays of (batch, n) for row buffers. r_eff <= max_chr_snps.
        r_eff_estimate = max_chr_snps  # upper bound; LD compression reduces this
        secular_stored_gb = 4 * r_eff_estimate * n_samples_kinship * 8 / 1e9
        secular_buffers_gb = 3 * 1000 * n_samples_kinship * 8 / 1e9  # batch=1000
        phase2_gb = matrix_gb + max_chr_gb + secular_stored_gb + secular_buffers_gb
        seq_peak_gb = max(phase1_gb, phase2_gb)
        if check_memory and seq_peak_gb > available_gb * 0.9:
            raise MemoryError(
                f"Insufficient memory for sequential X_c LOCO path: need "
                f"{seq_peak_gb:.1f}GB — phase 1: K_full ({matrix_gb:.2f}GB) "
                f"+ eigendecomp ({eigendecomp_min_gb:.1f}GB) + chunk buffer "
                f"({chunk_buffer_gb:.2f}GB); phase 2: U_full "
                f"({matrix_gb:.2f}GB) + max X_c ({max_chr_gb:.2f}GB) + "
                f"secular workspace ({secular_stored_gb:.2f}GB stored + "
                f"{secular_buffers_gb:.2f}GB buffers). "
                f"Available {available_gb:.1f}GB. "
                f"Consider using standard LOCO (use_secular_update=False)."
            )
        # Sequential path: always use single-pass for S_full accumulation (Pass 1).
        single_pass = True
        batch_size_chrs = n_chr_with_snps
    elif mode == LocoStreamingMode.X_C:
        # X_c path: yields raw genotype matrices (n x p_chr), much smaller than
        # n x n S_chr matrices. Total storage: n * total_snps * 8 bytes (all chrs).
        # Caller accumulates K_full = sum(X_c @ X_c.T) / n_filtered.
        #
        # Memory: all X_c simultaneously + K_full accumulation + chunk buffer
        # X_c total: n_samples_kinship * n_filtered * 8 bytes (upper bound)
        total_snps_kinship = n_filtered
        x_c_total_gb = n_samples_kinship * total_snps_kinship * 8 / 1e9
        # Caller's K_full eigendecomp workspace (separate from this function).
        # Estimate: all X_c + chunk buffer + K_full accumulator
        x_c_peak_gb = x_c_total_gb + chunk_buffer_gb + matrix_gb
        if check_memory and x_c_peak_gb > available_gb * 0.9:
            raise MemoryError(
                f"Insufficient memory for secular LOCO X_c path: need "
                f"{x_c_peak_gb:.1f}GB for all X_c matrices "
                f"(n={n_samples_kinship} x {total_snps_kinship} SNPs = "
                f"{x_c_total_gb:.2f}GB) + K_full accumulation "
                f"({matrix_gb:.2f}GB) + chunk buffer ({chunk_buffer_gb:.2f}GB), "
                f"available {available_gb:.1f}GB. "
                f"Consider using standard LOCO (use_secular_update=False)."
            )
        # X_c path is always single-pass (all chromosomes in one disk read).
        single_pass = True
        batch_size_chrs = n_chr_with_snps
    elif mode == LocoStreamingMode.S_CHR:
        # Secular path: retains ALL S_chr + K_full accumulator during streaming,
        # then K_full eigendecomp, then per-chromosome rotated-basis updates.
        # No multi-pass fallback — all S_chr must be in memory simultaneously.
        #
        # Peak phases (non-overlapping — K_full is freed before per-chr updates):
        #   Phase 1 (streaming): all S_chr + K_full accumulator + chunk buffer
        #   Phase 2 (K_full eigen): all S_chr + eigendecomp workspace
        #   Phase 3 (per-chr update): (C-k) S_chr + d_full + U_full + M + matmul
        #     temp + U_loco + eigh workspace ≈ (C-k+3) matrices + eigendecomp
        # Phase 2 dominates when C is large (eigendecomp workspace >> 3 matrices).
        per_chr_update_gb = matrix_gb * 3 + eigendecomp_min_gb
        secular_peak_gb = (
            matrix_gb * n_chr_with_snps  # all S_chr (no S_full in secular path)
            + chunk_buffer_gb
            + max(
                matrix_gb + eigendecomp_min_gb,  # phase 2: K_full + eigen workspace
                per_chr_update_gb,  # phase 3: M + temp + U_loco + eigen workspace
            )
        )
        if check_memory and secular_peak_gb > available_gb * 0.9:
            raise MemoryError(
                f"Insufficient memory for secular LOCO update: need "
                f"{secular_peak_gb:.1f}GB for {n_chr_with_snps} S_chr matrices "
                f"({n_chr_with_snps} x {matrix_gb:.2f}GB) "
                f"+ eigendecomp ({eigendecomp_min_gb:.1f}GB) "
                f"+ per-chr update temporaries ({per_chr_update_gb:.1f}GB), "
                f"available {available_gb:.1f}GB. "
                f"Consider using standard LOCO (use_secular_update=False)."
            )
        # Secular path is always single-pass (no multi-pass fallback).
        single_pass = True
        batch_size_chrs = n_chr_with_snps
    elif _max_batch_chrs is not None:
        batch_size_chrs = _max_batch_chrs
        single_pass = n_chr_with_snps <= batch_size_chrs
    else:
        single_pass = single_pass_gb <= available_gb * 0.9
        if single_pass:
            batch_size_chrs = n_chr_with_snps  # unused in single-pass branch
        else:
            # The consumer eigendecomposes each K_loco while the generator is
            # suspended with remaining S_chr matrices still alive. Reserve
            # eigendecomp workspace so the batch doesn't exhaust memory before
            # eigendecomp can run.
            eigendecomp_reserve_gb = _dsyevr_peak_gb(n_samples_kinship)
            # S_full + K_loco_buf (2 matrices) + chunk buffer + eigendecomp workspace
            usable_gb = (
                available_gb * 0.9
                - matrix_gb * 2
                - chunk_buffer_gb
                - eigendecomp_reserve_gb
            )
            batch_size_chrs = max(1, int(usable_gb / matrix_gb))

    if not single_pass:
        n_batches = (n_chr_with_snps + batch_size_chrs - 1) // batch_size_chrs
        logger.warning(
            f"LOCO streaming (NumPy): multi-pass mode ({n_batches} passes, "
            f"{batch_size_chrs} chromosomes/pass). Single-pass would need "
            f"{single_pass_gb:.1f}GB, available {available_gb:.1f}GB."
        )

    # Helper: stream one BED pass, accumulating S_full and selected S_chr matrices.
    # When accumulate_s_full=False, S_full is untouched (subsequent passes).
    _s_full_accumulated = False

    def _stream_pass(
        batch_chrs: list[str],
        S_full_buf: np.ndarray,
        accumulate_s_full: bool,
        pass_desc: str,
    ) -> dict[str, np.ndarray] | dict[str, list[np.ndarray]]:
        """Stream genotype chunks for one pass, returning per-chromosome data.

        Args:
            batch_chrs: Chromosome names to accumulate data for this pass.
            S_full_buf: Pre-allocated S_full buffer; updated in-place when
                accumulate_s_full=True, unchanged otherwise.
            accumulate_s_full: Whether to add to S_full_buf this pass.
                Must be True ONLY for the first pass. Subsequent passes with
                accumulate_s_full=True would double-count SNPs in S_full,
                corrupting all K_loco matrices.
            pass_desc: Progress bar description string.

        Returns:
            When mode != X_C and mode != X_C_SEQUENTIAL: Dict of
            {chr_name: S_chr_matrix} for batch_chrs.
            When mode == X_C or mode == X_C_SEQUENTIAL: Dict of
            {chr_name: list[X_chr_part]} chunks for batch_chrs.
            Caller must hstack to form the full X_c matrix.
        """
        nonlocal _s_full_accumulated
        if accumulate_s_full:
            if _s_full_accumulated:
                raise RuntimeError(
                    "S_full accumulation requested more than once. "
                    "This would corrupt K_loco matrices by double-counting SNPs."
                )
            _s_full_accumulated = True

        _x_c_mode = mode in {LocoStreamingMode.X_C, LocoStreamingMode.X_C_SEQUENTIAL}
        batch_chr_set = set(batch_chrs)
        if _x_c_mode:
            # X_c mode: collect raw genotype column chunks per chromosome
            batch_X_c_chunks: dict[str, list[np.ndarray]] = {c: [] for c in batch_chrs}
        else:
            batch_S_chr: dict[str, np.ndarray] = {
                c: np.zeros((n_samples_kinship, n_samples_kinship), dtype=np.float64)
                for c in batch_chrs
            }

        accum_iter = stream_genotype_chunks(
            bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
        )
        if show_progress:
            accum_iter = progress_iterator(accum_iter, total=n_chunks, desc=pass_desc)

        for chunk, file_start, file_end in accum_iter:
            left = np.searchsorted(snp_indices, file_start, side="left")
            right = np.searchsorted(snp_indices, file_end, side="left")
            chunk_snp_global_indices = snp_indices[left:right]
            chunk_filtered_local = chunk_snp_global_indices - file_start

            if len(chunk_filtered_local) == 0:
                continue

            X_chunk = chunk[:, chunk_filtered_local].astype(np.float64)
            # Early valid-sample subsetting (LOCO-07): compute kinship at n_valid size.
            if valid_indices is not None:
                X_chunk = X_chunk[valid_indices, :]
            X_centered = impute_and_center(X_chunk)

            if accumulate_s_full:
                S_full_buf += X_centered @ X_centered.T

            chunk_chrs = chromosomes[chunk_snp_global_indices]
            target_chrs_in_chunk = set(chunk_chrs) & batch_chr_set
            for chr_name in target_chrs_in_chunk:
                X_chr_part = X_centered[:, chunk_chrs == chr_name]
                if _x_c_mode:
                    batch_X_c_chunks[chr_name].append(X_chr_part.copy())
                else:
                    batch_S_chr[chr_name] += X_chr_part @ X_chr_part.T

            del X_chunk, X_centered, chunk

        if _x_c_mode:
            return batch_X_c_chunks  # type: ignore[return-value]
        return batch_S_chr

    def _yield_batch(
        S_full_buf: np.ndarray,
        batch_data: dict[str, np.ndarray] | dict[str, list[np.ndarray]],
        K_loco_buf: np.ndarray,
    ) -> Iterator[tuple[str, np.ndarray] | tuple[str, np.ndarray, int]]:
        """Yield per-chromosome matrices, one per chromosome.

        When mode=DEFAULT: yields (chr_name, K_loco) 2-tuples. K_loco is
        computed in-place via buffer reuse, then copied before yielding so
        callers may freely materialise the iterator.

        When mode=S_CHR: yields (chr_name, S_chr, p_chr) 3-tuples with
        the raw chromosome Gram matrix (un-normalised). The caller uses
        S_chr with loco_eigendecompose_from_full.

        When mode=X_C: yields (chr_name, X_c, p_chr) 3-tuples with the
        centered chromosome genotype matrix (n, p_chr). The caller uses X_c
        with secular_eigendecompose_from_full.
        """
        for chr_name in sorted(batch_data.keys(), key=chr_sort_key):
            p_chr = n_chr_filtered[chr_name]
            p_loco = n_filtered - p_chr
            if p_loco == 0:
                raise ValueError(
                    f"Cannot compute LOCO kinship: all {n_filtered} filtered SNPs "
                    f"are on chromosome '{chr_name}'."
                )
            if mode == LocoStreamingMode.X_C:
                # Concatenate column chunks into full X_c matrix
                chunks = batch_data.pop(chr_name)  # type: ignore[arg-type]
                if chunks:
                    x_c_mat = np.hstack(chunks)
                else:
                    x_c_mat = np.zeros((n_samples_kinship, 0), dtype=np.float64)
                logger.debug(
                    f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs "
                    f"retained (X_c mode)"
                )
                yield (chr_name, x_c_mat, p_chr)
            elif mode == LocoStreamingMode.S_CHR:
                s_chr_mat = batch_data.pop(chr_name)  # type: ignore[arg-type]
                logger.debug(
                    f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs "
                    f"retained (S_chr mode)"
                )
                yield (chr_name, s_chr_mat, p_chr)
            else:
                assert K_loco_buf is not None, "K_loco_buf required when mode=DEFAULT"
                np.subtract(S_full_buf, batch_data[chr_name], out=K_loco_buf)
                K_loco_buf /= p_loco
                logger.debug(
                    f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs "
                    f"retained"
                )
                del batch_data[chr_name]
                yield (chr_name, K_loco_buf.copy())

    def _yield_matrices() -> Iterator[
        tuple[str, np.ndarray] | tuple[str, np.ndarray, int]
    ]:
        nonlocal S_full  # needed: S_full /= n_filtered is augmented assignment

        if single_pass:
            # === SINGLE-PASS: one disk read for S_full + S_chr or X_c ===
            # For S_CHR/X_C modes: S_full is NOT accumulated (placeholder).
            # Caller constructs K_full from the yielded matrices.
            batch_data = _stream_pass(
                chrs_with_snps,
                S_full,
                accumulate_s_full=mode
                not in {LocoStreamingMode.S_CHR, LocoStreamingMode.X_C},
                pass_desc="LOCO: kinship accumulation (NumPy)",
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"LOCO streaming accumulation (NumPy) complete in {elapsed:.2f}s, "
                f"computing {len(batch_data)} LOCO matrices"
            )

            yield from _yield_batch(S_full, batch_data, K_loco_buf)

        else:
            # === MULTI-PASS: batch chromosomes across disk passes (LOCO-02) ===
            # Pass 0: accumulate S_full + first batch of S_chr (accumulate_s_full=True).
            # Pass k>0: accumulate only batch S_chr (accumulate_s_full=False).
            # CRITICAL: accumulate_s_full=True ONLY for pass 0. Setting it True for
            # subsequent passes would double-count SNPs in S_full, corrupting K_loco.
            n_batches = (n_chr_with_snps + batch_size_chrs - 1) // batch_size_chrs
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size_chrs
                batch_chrs = chrs_with_snps[batch_start : batch_start + batch_size_chrs]
                accumulate_s_full = batch_idx == 0

                if accumulate_s_full:
                    desc = f"LOCO: pass 1/{n_batches} (S_full + {len(batch_chrs)} chr)"
                else:
                    desc = (
                        f"LOCO: pass {batch_idx + 1}/{n_batches} "
                        f"({len(batch_chrs)} chr)"
                    )

                batch_data = _stream_pass(
                    batch_chrs,
                    S_full,
                    accumulate_s_full=accumulate_s_full,
                    pass_desc=desc,
                )

                yield from _yield_batch(S_full, batch_data, K_loco_buf)
                del batch_data
                gc.collect()

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"LOCO streaming multi-pass (NumPy) complete in {elapsed:.2f}s, "
                f"{n_batches} passes over {n_chr_with_snps} chromosomes"
            )

        # Yield full kinship for chromosomes with 0 filtered SNPs.
        # When mode=S_CHR: yield (chr_name, zero S_chr, p_chr=0) so
        # loco_eigendecompose_from_full can handle the degenerate case.
        # When mode=X_C: yield (chr_name, zero X_c, p_chr=0) so
        # secular_eigendecompose_from_full can handle the degenerate case.
        if chrs_without_snps:
            if mode == LocoStreamingMode.X_C:
                zero_x_c = np.zeros((n_samples_kinship, 0), dtype=np.float64)
                for chr_name in sorted(chrs_without_snps, key=chr_sort_key):
                    logger.debug(
                        f"LOCO chr {chr_name}: 0 SNPs after filtering, "
                        f"X_c=empty (secular solver uses full kinship)"
                    )
                    yield (chr_name, zero_x_c, 0)
            elif mode == LocoStreamingMode.S_CHR:
                zero_s_chr = np.zeros(
                    (n_samples_kinship, n_samples_kinship), dtype=np.float64
                )
                for chr_name in sorted(chrs_without_snps, key=chr_sort_key):
                    logger.debug(
                        f"LOCO chr {chr_name}: 0 SNPs after filtering, "
                        f"S_chr=0 (secular update uses full kinship)"
                    )
                    yield (chr_name, zero_s_chr.copy(), 0)
            else:
                S_full /= n_filtered
                for chr_name in sorted(chrs_without_snps, key=chr_sort_key):
                    logger.debug(
                        f"LOCO chr {chr_name}: 0 SNPs after filtering, "
                        f"using full kinship"
                    )
                    yield (chr_name, S_full.copy())

    if mode == LocoStreamingMode.X_C_SEQUENTIAL:
        # Sequential two-pass path: Pass 1 (synchronous) accumulates S_full.
        # Pass 2 (lazy generator) re-reads the BED once per chromosome.
        S_full = np.zeros((n_samples_kinship, n_samples_kinship), dtype=np.float64)
        # Pass 1: accumulate S_full only (batch_chrs=[] skips per-chromosome data).
        _stream_pass(
            [],
            S_full,
            accumulate_s_full=True,
            pass_desc="LOCO: S_full accumulation (sequential, pass 1)",
        )
        elapsed_p1 = time.perf_counter() - start_time
        logger.info(f"LOCO sequential pass 1 (S_full) complete in {elapsed_p1:.2f}s")

        # Capture variables for the generator closure (snp_indices is already in scope).
        _seq_chrs_with_snps = list(chrs_with_snps)
        _seq_chrs_without_snps = list(chrs_without_snps)
        _seq_n_chr_filtered = dict(n_chr_filtered)
        _seq_n_samples_kinship = n_samples_kinship
        _seq_chromosomes = chromosomes
        _seq_snp_indices = snp_indices

        def _yield_x_c_sequential_gen() -> Iterator[tuple[str, np.ndarray, int]]:
            """Lazy generator: re-reads BED once per chromosome (Pass 2).

            Yields (chr_name, X_c, p_chr) one chromosome at a time.
            Only one X_c is live at a time — O(max_X_c) peak memory.
            """
            zero_x_c = np.zeros((_seq_n_samples_kinship, 0), dtype=np.float64)

            # Yield chromosomes with SNPs (one BED scan per chromosome)
            for target_chr in sorted(_seq_chrs_with_snps, key=chr_sort_key):
                p_chr = _seq_n_chr_filtered[target_chr]
                # Collect columns belonging to this chromosome
                target_chr_mask = _seq_chromosomes[_seq_snp_indices] == target_chr
                chr_local_indices = np.where(target_chr_mask)[0]
                # chr_local_indices are positions within snp_indices array.
                # We need the actual global BED column indices for this chromosome.
                target_chr_global_indices = _seq_snp_indices[chr_local_indices]

                if len(target_chr_global_indices) == 0:
                    logger.debug(
                        f"LOCO sequential chr {target_chr}: 0 SNPs, yielding empty X_c"
                    )
                    yield (target_chr, zero_x_c, 0)
                    continue

                # One full BED scan to collect X_c columns for this chromosome.
                # Each chunk: extract columns matching this chromosome.
                x_c_chunks: list[np.ndarray] = []
                chr_accum_iter = stream_genotype_chunks(
                    bed_path,
                    chunk_size=chunk_size,
                    dtype=np.float64,
                    show_progress=False,
                )
                if show_progress:
                    chr_accum_iter = progress_iterator(
                        chr_accum_iter,
                        total=n_chunks,
                        desc=f"LOCO: X_c chr {target_chr} (sequential pass 2)",
                    )
                for chunk, file_start, file_end in chr_accum_iter:
                    # Find filtered SNPs in this chunk
                    left = np.searchsorted(_seq_snp_indices, file_start, side="left")
                    right = np.searchsorted(_seq_snp_indices, file_end, side="left")
                    chunk_snp_global = _seq_snp_indices[left:right]
                    if len(chunk_snp_global) == 0:
                        del chunk
                        continue
                    # Filter to columns belonging to target_chr
                    chunk_chrs = _seq_chromosomes[chunk_snp_global]
                    chr_mask_in_chunk = chunk_chrs == target_chr
                    if not np.any(chr_mask_in_chunk):
                        del chunk
                        continue
                    # Global indices of this chromosome's SNPs in this chunk
                    chr_global_in_chunk = chunk_snp_global[chr_mask_in_chunk]
                    local_in_chunk = chr_global_in_chunk - file_start
                    X_chunk = chunk[:, local_in_chunk].astype(np.float64)
                    if valid_indices is not None:
                        X_chunk = X_chunk[valid_indices, :]
                    X_centered = impute_and_center(X_chunk)
                    x_c_chunks.append(X_centered)
                    del X_chunk, X_centered, chunk

                if x_c_chunks:
                    X_c = np.hstack(x_c_chunks)
                else:
                    X_c = zero_x_c
                del x_c_chunks

                logger.debug(
                    f"LOCO sequential chr {target_chr}: {p_chr} SNPs, "
                    f"X_c shape={X_c.shape}"
                )
                yield (target_chr, X_c, p_chr)
                del X_c
                gc.collect()

            # Yield chromosomes with 0 filtered SNPs
            for target_chr in sorted(_seq_chrs_without_snps, key=chr_sort_key):
                logger.debug(
                    f"LOCO sequential chr {target_chr}: 0 SNPs after filtering, "
                    f"yielding empty X_c (secular solver uses full kinship)"
                )
                yield (target_chr, zero_x_c, 0)

        return SequentialLocoResult(
            s_full=S_full,
            n_filtered=n_filtered,
            generator=_yield_x_c_sequential_gen(),
            snp_stats_cache=snp_stats_cache,
        )

    if mode in {LocoStreamingMode.S_CHR, LocoStreamingMode.X_C}:
        # Secular paths: caller accumulates K_full from yielded matrices.
        # Skip both S_full and K_loco_buf allocations (saves 2 n^2 matrices).
        S_full = np.empty(0)  # placeholder — never written in S_CHR/X_C mode
        K_loco_buf = None  # type: ignore[assignment]
        return n_filtered, _yield_matrices(), snp_stats_cache

    S_full = np.zeros((n_samples_kinship, n_samples_kinship), dtype=np.float64)

    K_loco_buf = np.empty_like(S_full)
    return _yield_matrices(), snp_stats_cache


def _find_loco_eigen_cache(
    eigen_dir: Path,
    prefix: str,
    chr_names: list[str],
    *,
    legacy_text: bool = False,
) -> dict[str, tuple[Path, Path]] | None:
    """Check for a complete set of per-chromosome cached eigen files.

    Looks for files named ``{prefix}.loco.chr{chr_name}.eigenD.{ext}`` and
    ``{prefix}.loco.chr{chr_name}.eigenU.{ext}`` for every chromosome.

    Dimension validation is deferred to the per-chromosome load in
    ``run_lmm_loco``, where ``read_eigen_files(n_samples=...)`` raises
    ``ValueError`` on mismatch. This avoids loading all eigen data
    eagerly just to check dimensions.

    Args:
        eigen_dir: Directory containing cached eigen files.
        prefix: Filename prefix (e.g. "result").
        chr_names: List of chromosome names to check.
        legacy_text: If True, look for .txt files instead of .npy.

    Returns:
        Dict mapping chr_name -> (eigenD_path, eigenU_path) if ALL chromosomes
        have both files. None if ANY chromosome is missing either file.
    """
    if not eigen_dir.is_dir():
        logger.warning(
            f"eigen_dir is not a directory: {eigen_dir}. Will compute from scratch."
        )
        return None

    suffix = ".txt" if legacy_text else ".npy"
    cache: dict[str, tuple[Path, Path]] = {}

    for ch in chr_names:
        d_path = eigen_dir / f"{prefix}.loco.chr{ch}.eigenD{suffix}"
        u_path = eigen_dir / f"{prefix}.loco.chr{ch}.eigenU{suffix}"

        if not d_path.exists() or not u_path.exists():
            missing = d_path if not d_path.exists() else u_path
            logger.info(
                f"LOCO eigen cache incomplete: missing {missing}. "
                f"Will compute from scratch."
            )
            return None

        cache[ch] = (d_path, u_path)

    return cache


def run_lmm_loco(
    bed_path: Path,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    lmm_mode: int = 1,
    output_path: Path | None = None,
    check_memory: bool = True,
    show_progress: bool = True,
    save_kinship: bool = False,
    kinship_output_dir: Path | None = None,
    kinship_output_prefix: str = "result",
    snps_indices: np.ndarray | None = None,
    ksnps_indices: np.ndarray | None = None,
    col_chunk_size: int = 5_000,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    backend: str = "jax",
    write_eigen: bool = False,
    eigen_dir: Path | None = None,
    eigen_prefix: str = "result",
    use_secular_update: bool = False,
) -> LocoResult:
    """Run LOCO LMM association: per-chromosome eigendecomp and association.

    For each chromosome:
    1. Compute K_loco (kinship excluding that chromosome) via streaming
    2. Optionally save K_loco to disk
    3. Subset K_loco to valid samples, delete original
    4. Eigendecompose K_loco_valid, optionally write eigen cache
    5. Run LMM association on that chromosome's SNPs
    6. Write results to shared output file

    When ``eigen_dir`` points to a directory with a complete set of
    per-chromosome eigen files (written by a previous run with
    ``write_eigen=True``), kinship computation and eigendecomposition
    are skipped entirely — eigen pairs are loaded from disk.

    Args:
        bed_path: PLINK file prefix (without .bed/.bim/.fam extension).
        phenotypes: Phenotype vector (n_samples_total,) with NaN for missing.
        covariates: Covariate matrix (n_samples_total, n_cvt) or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        lmm_mode: LMM test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        output_path: Path for incremental result writing, or None for in-memory.
        check_memory: If True, check available memory before computation.
        show_progress: If True, show progress bars and log messages.
        save_kinship: If True, save each K_loco to disk before discarding.
        kinship_output_dir: Directory for kinship output files.
        kinship_output_prefix: Prefix for kinship output filenames.
        snps_indices: Pre-resolved column indices for -snps restriction, or None.
        ksnps_indices: Pre-resolved column indices for -ksnps restriction, or
            None. When provided, only these SNPs are used for LOCO kinship
            computation. Passed through to compute_loco_kinship_streaming().
        col_chunk_size: Number of SNP columns per disk read chunk. Controls
            peak memory: n_valid * col_chunk_size * 8 bytes per chunk.
        l_min: Minimum lambda for optimization (default 1e-5).
        l_max: Maximum lambda for optimization (default 1e5).
        backend: Compute backend — "jax" (default) or "numpy".
        write_eigen: If True, write per-chromosome eigen files after
            eigendecomp. Raises ValueError if eigen_dir is None.
        eigen_dir: Directory for reading/writing per-chromosome eigen cache.
            When set, checks for cached files before computing. Combined
            with write_eigen, writes new files here.
        eigen_prefix: Prefix for eigen filenames (default "result").
        use_secular_update: If True, compute the full kinship eigendecomposition
            once via eigendecompose_kinship(K_full) then derive per-chromosome
            eigendecompositions via secular_eigendecompose_from_full instead of
            eigendecomposing each K_loco independently. Uses sequential
            streaming (one chromosome at a time) with peak memory O(n^2)
            for K_full/U_full plus O(n * max_p_chr) for the largest single
            chromosome. Only supported with the numpy backend.
            Raises ValueError if True with backend="jax", with
            save_kinship=True, or when cached eigen files exist in eigen_dir.

    Returns:
        LocoResult with associations in biological chromosome order
        (1-22, X, Y, XY, MT). Associations list is empty if output_path
        is set (results written to disk).

    Raises:
        ValueError: If only one chromosome present, if lmm_mode invalid,
            if backend is not 'jax' or 'numpy', if use_secular_update=True
            with backend='jax', with save_kinship=True, or when cached eigen
            files exist in eigen_dir.
    """
    start_time = time.perf_counter()

    if backend not in ("jax", "numpy"):
        raise ValueError(f"backend must be 'jax' or 'numpy', got {backend!r}")

    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    if write_eigen and eigen_dir is None:
        raise ValueError(
            "write_eigen=True requires eigen_dir to be set. "
            "Pass eigen_dir or use --eigen-dir on the CLI."
        )

    if use_secular_update and backend != "numpy":
        raise ValueError(
            "use_secular_update=True is only supported with backend='numpy'. "
            f"Got backend={backend!r}."
        )

    if use_secular_update and save_kinship:
        raise ValueError(
            "save_kinship=True is not supported with use_secular_update=True. "
            "The secular update path does not materialise K_loco matrices."
        )

    # Read LOCO worker count and log configuration (LOCO-08)
    loco_workers = get_loco_worker_count()
    if loco_workers > 1:
        logger.warning(
            f"JAMMA_LOCO_WORKERS={loco_workers} but parallel LOCO is not yet "
            "implemented. Running sequentially."
        )
    else:
        logger.debug("LOCO worker count: 1 (sequential)")

    # Get metadata
    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps_total = meta["n_snps"]

    # Chromosome partitions (unfiltered) — derived from already-loaded metadata
    # to avoid a redundant BIM re-read (LOCO-04)
    partitions = partitions_from_metadata(meta)
    unique_chrs = sorted(partitions.keys(), key=chr_sort_key)

    if len(unique_chrs) < 2:
        raise ValueError(
            "LOCO requires SNPs on multiple chromosomes. "
            f"Found only {len(unique_chrs)} chromosome(s): {unique_chrs}"
        )

    # Log the actual backend being used (not a re-derived plan, which could diverge)
    logger.info(f"LOCO backend: {backend}")

    if show_progress:
        logger.info("Performing LOCO LMM Association Test")
        logger.info(f"  Total individuals: {n_samples_total:,}")
        logger.info(f"  Total SNPs: {n_snps_total:,}")
        logger.info(f"  Chromosomes: {len(unique_chrs)}")

    # Sample filtering: missing phenotypes, covariate NaNs
    from jamma.lmm.prepare_common import compute_valid_mask

    valid_mask = compute_valid_mask(phenotypes, covariates)
    n_valid = int(np.sum(valid_mask))

    if n_valid == 0:
        raise ValueError("No samples with valid phenotypes")

    # Computed once: avoids re-evaluating np.all(valid_mask) inside the chromosome loop.
    all_samples_valid = n_valid == n_samples_total

    phenotypes_valid = phenotypes[valid_mask]
    covariates_valid = covariates[valid_mask, :] if covariates is not None else None

    if show_progress:
        n_filtered_samples = n_samples_total - n_valid
        logger.info(
            f"  Analyzed individuals: {n_valid:,} ({n_filtered_samples} filtered)"
        )

    # Build SNP metadata for result construction (lazy -- no upfront dict allocation)
    snp_info = LazySnpMeta(meta)

    test_type = _TEST_TYPE_MAP[lmm_mode]

    if output_path is None and n_snps_total > 100_000:
        logger.warning(
            f"LOCO in-memory mode with {n_snps_total:,} total SNPs. Results will "
            f"accumulate in memory. Provide output_path to stream results to disk."
        )

    all_results: list[AssocResult] = []

    try:
        with contextlib.ExitStack() as stack:
            writer = None
            if output_path is not None:
                writer = stack.enter_context(
                    IncrementalAssocWriter(output_path, test_type=test_type)
                )

            # Precompute global SNP membership mask for -snps restriction.
            # Avoids per-chromosome np.isin on every iteration.
            if snps_indices is not None:
                snps_global_mask: np.ndarray | None = np.zeros(n_snps_total, dtype=bool)
                snps_global_mask[snps_indices] = True
            else:
                snps_global_mask = None

            # Check for cached eigen files before computing kinship.
            # When write_eigen is True the user explicitly asked to
            # (re)generate files, so skip the cache and recompute.
            eigen_cache: dict[str, tuple[Path, Path]] | None = None
            if eigen_dir is not None and not write_eigen:
                eigen_cache = _find_loco_eigen_cache(
                    eigen_dir, eigen_prefix, unique_chrs
                )
                if eigen_cache is not None:
                    logger.info(
                        f"Found complete LOCO eigen cache in {eigen_dir} "
                        f"({len(eigen_cache)} chromosomes). "
                        f"Skipping kinship computation and eigendecomp."
                    )
                    if use_secular_update:
                        raise ValueError(
                            "use_secular_update=True conflicts with cached "
                            f"eigen files in {eigen_dir}. Either remove the "
                            "cached files, or pass eigen_dir=None to skip "
                            "the cache and use the secular update path."
                        )
                    if save_kinship:
                        logger.warning(
                            "save_kinship ignored when using cached eigen "
                            "files (kinship is not computed)"
                        )
                    if backend == "numpy":
                        logger.warning(
                            "Using cached eigen with NumPy backend: SNP "
                            "filtering will use valid-sample-only statistics "
                            "(not all-sample stats from kinship pass). This "
                            "may produce slightly different SNP filter sets "
                            "compared to the original compute run."
                        )

            # Initialise to None; reassigned inside the compute block when
            # eigen_cache is None and we actually stream kinship.
            snp_stats_cache = None
            kinship_valid_indices = None
            loco_iter = None
            # Secular update state: set when use_secular_update=True
            secular_d_full: np.ndarray | None = None
            secular_U_full: np.ndarray | None = None
            secular_p_full: int = 0
            # Sequential generator for the secular path (one X_c at a time)
            secular_x_c_seq_iter: Iterator[tuple[str, np.ndarray, int]] | None = None

            if eigen_cache is None:
                # Stream LOCO kinship matrices one at a time.
                # NumPy backend uses pure-NumPy kinship (no JAX dependency);
                # JAX backend uses JAX matmul for GPU acceleration.
                if backend == "numpy":
                    # When save_kinship=False and some samples are
                    # invalid, pass valid_indices so kinship is accumulated
                    # at n_valid x n_valid size, avoiding full n_samples^2
                    # materialisation for post-hoc subsetting (LOCO-07).
                    kinship_valid_indices = (
                        None
                        if all_samples_valid or save_kinship
                        else np.where(valid_mask)[0]
                    )

                    if use_secular_update:
                        # Sequential secular update path: two-pass streaming reduces
                        # peak memory from O(sum_X_c) to O(max_X_c).
                        # Pass 1 (inside the function) accumulates S_full synchronously.
                        # Pass 2 is a lazy generator yielding one X_c at a time.
                        _seq_result = _compute_loco_kinship_streaming_numpy(
                            bed_path,
                            maf_threshold=maf_threshold,
                            miss_threshold=miss_threshold,
                            check_memory=check_memory,
                            show_progress=show_progress,
                            ksnps_indices=ksnps_indices,
                            valid_indices=kinship_valid_indices,
                            mode=LocoStreamingMode.X_C_SEQUENTIAL,
                        )
                        S_full_secular = _seq_result.s_full
                        secular_p_full = _seq_result.n_filtered
                        secular_x_c_seq_iter = _seq_result.generator
                        snp_stats_cache = _seq_result.snp_stats_cache
                        # Normalise S_full in-place to get K_full, avoiding a
                        # second n×n allocation that the memory estimator must
                        # otherwise budget for.
                        S_full_secular /= secular_p_full
                        K_full_secular = S_full_secular
                        del S_full_secular  # just drops the name, no dealloc

                        if K_full_secular.shape[0] == 0:
                            raise ValueError(
                                "Secular update: S_full is empty (no SNPs). "
                                "Check SNP filtering and kinship SNP list."
                            )

                        t_kfull = time.perf_counter()
                        logger.info(
                            "Secular update: eigendecomposing K_full "
                            f"(n={K_full_secular.shape[0]}, "
                            f"p_full={secular_p_full})..."
                        )
                        secular_d_full, secular_U_full = eigendecompose_kinship(
                            K_full_secular, check_memory=check_memory
                        )
                        del K_full_secular
                        gc.collect()
                        logger.info(
                            f"Secular update: K_full eigendecomp done in "
                            f"{time.perf_counter() - t_kfull:.3f}s"
                        )
                    else:
                        loco_iter, snp_stats_cache = (
                            _compute_loco_kinship_streaming_numpy(
                                bed_path,
                                maf_threshold=maf_threshold,
                                miss_threshold=miss_threshold,
                                check_memory=check_memory,
                                show_progress=show_progress,
                                ksnps_indices=ksnps_indices,
                                valid_indices=kinship_valid_indices,
                            )
                        )
                else:
                    from jamma.kinship import (  # noqa: PLC0415
                        compute_loco_kinship_streaming,
                    )

                    loco_iter = compute_loco_kinship_streaming(
                        bed_path,
                        maf_threshold=maf_threshold,
                        miss_threshold=miss_threshold,
                        check_memory=check_memory,
                        show_progress=show_progress,
                        ksnps_indices=ksnps_indices,
                    )

                # Create eigen output directory before the loop (once, not per-chr).
                # (eigen_dir is guaranteed non-None when write_eigen is True
                # by the early guard at the top of this function.)
                if write_eigen:
                    try:
                        eigen_dir.mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        raise OSError(
                            f"Cannot create eigen cache directory {eigen_dir}: {e}"
                        ) from e

            first_chr_pve: float | None = None
            first_chr_pve_se: float | None = None

            # Iterate: either from cached eigen files, kinship stream,
            # or secular update (sequential K_full eigendecomp + lazy X_c generator).
            if eigen_cache is not None:
                chr_iterator = ((chr_name, None) for chr_name in unique_chrs)
            elif use_secular_update:
                # Sequential secular update: consume x_c_seq_iter one chromosome
                # at a time. For each (chr_name, X_c, p_chr), compute the
                # secular eigendecomp inline, free X_c, then yield (chr_name, eigen).
                if secular_x_c_seq_iter is None:
                    raise RuntimeError(
                        "Internal error: secular_x_c_seq_iter not initialized. "
                        "This indicates a bug in the secular update path setup."
                    )
                if secular_d_full is None or secular_U_full is None:
                    raise RuntimeError(
                        "Internal error: secular_d_full/U_full not initialized. "
                        "This indicates a bug in the secular update path setup."
                    )

                def _secular_chr_iter() -> Iterator[
                    tuple[str, tuple[np.ndarray, np.ndarray]]
                ]:
                    """Consume sequential X_c generator, yield per-chr eigendecomp."""
                    for chr_name_s, X_c_s, p_chr_s in secular_x_c_seq_iter:  # type: ignore[union-attr]
                        t_sec = time.perf_counter()
                        if show_progress:
                            chr_snp_indices_s = partitions[chr_name_s]
                            logger.info(
                                f"LOCO: chromosome {chr_name_s} "
                                f"({len(chr_snp_indices_s)} SNPs), "
                                f"secular equation update..."
                            )
                        try:
                            eigenvalues_s, U_s = secular_eigendecompose_from_full(
                                secular_d_full,  # type: ignore[arg-type]
                                secular_U_full,  # type: ignore[arg-type]
                                X_c_s,
                                secular_p_full,
                                p_chr_s,
                            )
                        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
                            raise type(e)(
                                f"Secular eigendecomposition failed for chromosome "
                                f"{chr_name_s} (p_chr={p_chr_s}): {e}"
                            ) from e
                        del X_c_s
                        gc.collect()
                        logger.debug(
                            f"Secular equation update for chr {chr_name_s}: "
                            f"{time.perf_counter() - t_sec:.3f}s"
                        )
                        yield (chr_name_s, (eigenvalues_s, U_s))

                chr_iterator = _secular_chr_iter()  # type: ignore[assignment]
            else:
                assert loco_iter is not None, (
                    "loco_iter must be set when eigen_cache is None"
                )
                chr_iterator = loco_iter  # type: ignore[assignment]

            for chr_idx, (chr_name, K_loco) in enumerate(chr_iterator):
                chr_snp_indices = partitions[chr_name]

                if eigen_cache is not None:
                    # Load cached eigen directly — no kinship or eigendecomp.
                    d_path, u_path = eigen_cache[chr_name]
                    if show_progress:
                        logger.info(
                            f"LOCO: chromosome {chr_name} "
                            f"({chr_idx + 1}/{len(unique_chrs)}), "
                            f"{len(chr_snp_indices)} SNPs, "
                            f"loading cached eigen..."
                        )
                    try:
                        eigenvalues_np, U = read_eigen_files(
                            d_path, u_path, n_samples=n_valid
                        )
                    except (ValueError, FileNotFoundError) as e:
                        raise type(e)(
                            f"LOCO eigen cache for chromosome {chr_name}: {e}"
                        ) from e
                elif use_secular_update:
                    # Sequential secular update: eigenvalues pre-computed by
                    # _secular_chr_iter() and passed as K_loco = (eigenvalues, U).
                    eigenvalues_np, U = K_loco  # type: ignore[misc]
                else:
                    # Standard path: kinship -> eigendecomp
                    if show_progress:
                        logger.info(
                            f"LOCO: chromosome {chr_name} "
                            f"({chr_idx + 1}/{len(unique_chrs)}), "
                            f"{len(chr_snp_indices)} SNPs, "
                            f"eigendecomposing..."
                        )

                    if save_kinship and kinship_output_dir is not None:
                        kinship_path = (
                            kinship_output_dir
                            / f"{kinship_output_prefix}.loco.cXX.chr{chr_name}.npy"
                        )
                        try:
                            actual_path = write_kinship_matrix(K_loco, kinship_path)
                        except OSError as e:
                            raise OSError(
                                f"Failed to save LOCO kinship for chromosome "
                                f"{chr_name} to {kinship_path}: {e}"
                            ) from e
                        if show_progress:
                            logger.info(f"  Saved LOCO kinship to {actual_path}")

                    # K_loco is already n_valid x n_valid from numpy
                    # streaming — skip post-hoc subsetting (LOCO-07).
                    if backend == "numpy" and kinship_valid_indices is not None:
                        if K_loco.shape != (n_valid, n_valid):
                            raise RuntimeError(
                                f"Expected K_loco shape ({n_valid}, {n_valid}) "
                                f"from early subsetting, got {K_loco.shape}"
                            )
                        K_loco_valid = K_loco
                        del K_loco
                    elif all_samples_valid:
                        K_loco_valid = K_loco
                        del K_loco
                    else:
                        K_loco_valid = K_loco[np.ix_(valid_mask, valid_mask)]
                        del K_loco
                        gc.collect()

                    eigenvalues_np, U = eigendecompose_kinship(
                        K_loco_valid, check_memory=check_memory
                    )
                    del K_loco_valid
                    gc.collect()

                # Write eigen files if requested (skip for cache-loaded eigen).
                if write_eigen and eigen_cache is None:
                    try:
                        write_eigen_files(
                            eigenvalues_np,
                            U,
                            eigen_dir,
                            prefix=f"{eigen_prefix}.loco.chr{chr_name}",
                        )
                    except OSError as e:
                        raise OSError(
                            f"Failed to write LOCO eigen for chromosome "
                            f"{chr_name} to {eigen_dir}: {e}"
                        ) from e
                    logger.info(f"  Wrote LOCO eigen for chr {chr_name}")

                logger.debug(
                    f"  chr {chr_name}: {backend} backend, {len(chr_snp_indices)} SNPs"
                )

                if backend == "numpy":
                    chr_results, chr_pve, chr_pve_se = _run_lmm_for_chromosome_numpy(
                        bed_path=bed_path,
                        chr_snp_indices=chr_snp_indices,
                        eigenvalues=eigenvalues_np,
                        eigenvectors=U,
                        phenotypes=phenotypes_valid,
                        covariates=covariates_valid,
                        snp_info=snp_info,
                        maf_threshold=maf_threshold,
                        miss_threshold=miss_threshold,
                        lmm_mode=lmm_mode,
                        valid_mask=valid_mask,
                        show_progress=show_progress,
                        l_min=l_min,
                        l_max=l_max,
                        snps_global_mask=snps_global_mask,
                        col_chunk_size=col_chunk_size,
                        writer=writer,
                        chr_name=chr_name,
                        snp_stats_cache=snp_stats_cache,
                        compute_pve=(first_chr_pve is None),
                    )
                elif backend == "jax":
                    # JAX path: always uses streaming internally
                    # (batch JAX for LOCO is out of scope)
                    chr_results, chr_pve, chr_pve_se = _run_lmm_for_chromosome(
                        bed_path=bed_path,
                        chr_snp_indices=chr_snp_indices,
                        eigenvalues=eigenvalues_np,
                        eigenvectors=U,
                        phenotypes=phenotypes_valid,
                        covariates=covariates_valid,
                        snp_info=snp_info,
                        maf_threshold=maf_threshold,
                        miss_threshold=miss_threshold,
                        lmm_mode=lmm_mode,
                        valid_mask=valid_mask,
                        show_progress=show_progress,
                        l_min=l_min,
                        l_max=l_max,
                        snps_global_mask=snps_global_mask,
                        col_chunk_size=col_chunk_size,
                        writer=writer,
                        compute_pve=(first_chr_pve is None),
                    )
                else:
                    raise ValueError(
                        f"Unknown LOCO backend: {backend!r}. Must be 'numpy' or 'jax'."
                    )

                if writer is None:
                    all_results.extend(chr_results)

                if first_chr_pve is None and chr_pve is not None:
                    if chr_idx > 0:
                        logger.info(
                            f"PVE computed from chromosome {chr_name} "
                            f"(earlier chromosomes had all SNPs filtered)"
                        )
                    first_chr_pve = chr_pve
                    first_chr_pve_se = chr_pve_se

                del eigenvalues_np, U
                gc.collect()

            if first_chr_pve is None:
                logger.warning(
                    "PVE could not be computed: all chromosomes had all SNPs "
                    "filtered. Check MAF/missingness thresholds."
                )

            if writer is not None and show_progress:
                logger.info(f"Wrote {writer.count:,} results to {output_path}")

            if show_progress:
                elapsed = time.perf_counter() - start_time
                pve_str = (
                    f", pve={first_chr_pve:.6f}" if first_chr_pve is not None else ""
                )
                se_str = (
                    f", se(pve)={first_chr_pve_se:.6g}"
                    if first_chr_pve_se is not None
                    else ""
                )
                logger.info(
                    f"LOCO LMM Association completed in {elapsed:.2f}s{pve_str}{se_str}"
                )

            n_tested = writer.count if writer is not None else len(all_results)
            return LocoResult(
                associations=[] if output_path is not None else all_results,
                n_tested=n_tested,
                pve=first_chr_pve,
                pve_se=first_chr_pve_se,
            )
    finally:
        if backend == "jax":
            import jax

            try:
                jax.clear_caches()
            except Exception:
                logger.warning(
                    "Failed to clear JAX caches during cleanup", exc_info=True
                )


def _run_lmm_for_chromosome(
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    snp_info: list,
    maf_threshold: float,
    miss_threshold: float,
    lmm_mode: int,
    valid_mask: np.ndarray,
    show_progress: bool = True,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    snps_global_mask: np.ndarray | None = None,
    col_chunk_size: int = 5_000,
    writer: IncrementalAssocWriter | None = None,
    compute_pve: bool = False,
) -> tuple[list[AssocResult], float | None, float | None]:
    """Run JAX LMM association on a single chromosome's SNPs.

    Reads the chromosome's SNPs from the BED file in column chunks
    (two-pass: statistics, then association), never allocating the full
    chromosome genotype matrix. Peak per-chunk allocation is
    (n_valid, col_chunk_size) instead of (n_valid, n_chr_snps).

    Args:
        bed_path: PLINK file prefix.
        chr_snp_indices: Column indices for this chromosome's SNPs.
        eigenvalues: Eigenvalues from LOCO kinship eigendecomp.
        eigenvectors: Eigenvectors from LOCO kinship eigendecomp.
        phenotypes: Phenotype vector (n_valid_samples,), already filtered.
        covariates: Covariate matrix (n_valid_samples, n_cvt) or None.
        snp_info: Full SNP metadata list (indexed by global SNP index).
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        valid_mask: Boolean mask for valid samples (for genotype subsetting).
        show_progress: Whether to log progress.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations.
        snps_global_mask: Boolean mask over all SNPs (True = included by -snps), or
            None. Pre-indexed: `snps_global_mask[chr_snp_indices]` gives the
            per-chromosome mask. Avoids per-chromosome np.isin computation.
        col_chunk_size: Number of SNP columns per disk read chunk.
        writer: Optional incremental writer for streaming results to disk.
            When provided, results are written directly and an empty list
            is returned. When None, results are accumulated and returned.
        compute_pve: If True, compute PVE from null model REML lambda.
            Set for each chromosome until PVE is successfully computed
            (typically the first chromosome with passing SNPs).

    Returns:
        Tuple of (results, pve, pve_se) where results is a list of AssocResult
        (empty if writer used), pve is the PVE estimate (None unless
        compute_pve=True), and pve_se is the standard error of PVE (None
        unless compute_pve=True and likelihood surface is not flat).
    """
    import jax  # noqa: PLC0415

    from jamma.lmm.chunk import (  # noqa: PLC0415
        _compute_chunk_size,
        compute_subchunk_starts,
    )
    from jamma.lmm.compute import (  # noqa: PLC0415
        _compute_lmm_chunk,
        block_chunk_result,
        log_jax_error,
    )
    from jamma.lmm.likelihood_jax import batch_compute_uab  # noqa: PLC0415
    from jamma.lmm.prepare import (  # noqa: PLC0415
        DevicePlacement,
        _compute_null_model,
        _select_jax_device,
        prepare_utg_chunk,
    )
    from jamma.lmm.results import _chunk_result_to_numpy  # noqa: PLC0415
    from jamma.lmm.schema import ACCUM_KEYS as _ACCUM_KEYS  # noqa: PLC0415

    n_samples = phenotypes.shape[0]
    valid_indices = np.where(valid_mask)[0]

    # === PASS 1: Chunked SNP statistics + filtering ===
    col_means, miss_counts, col_vars, n_unexpected = _collect_chr_snp_stats(
        bed_path, chr_snp_indices, valid_indices, col_chunk_size
    )

    filter_result = _filter_chr_snps(
        col_means,
        miss_counts,
        col_vars,
        n_samples,
        maf_threshold,
        miss_threshold,
        chr_snp_indices,
        snps_global_mask,
        n_unexpected,
        show_progress,
    )
    if filter_result is None:
        return [], None, None

    (
        _local_filtered_indices,
        global_filtered_indices,
        filtered_afs,
        filtered_miss,
        filtered_means_all,
    ) = filter_result
    n_filtered = len(global_filtered_indices)

    # === PASS 2: Chunked association ===
    # Eigendecomp setup
    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Rotation is pure BLAS — use all physical cores, not the JAX-reduced
    # count from get_blas_thread_count(). JAX isn't running during rotation.
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = eigenvectors.T @ W
        Uty = eigenvectors.T @ phenotypes

    # LOCO intentionally uses single-device mode — each chromosome's
    # association pass has fewer SNPs, so multi-device sharding overhead
    # outweighs the parallelism benefit.
    device = _select_jax_device(use_gpu=False)
    placement = DevicePlacement(snp=device, rep=device, n_devices=1)

    eigenvalues_jax = None
    UtW_jax = None
    Uty_jax = None
    try:
        logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
            lmm_mode,
            eigenvalues,
            UtW,
            Uty,
            n_cvt,
            placement.rep,
            show_progress=False,
            l_min=l_min,
            l_max=l_max,
        )

        chr_pve = None
        chr_pve_se = None
        if compute_pve:
            chr_pve, chr_pve_se = compute_and_log_pve(
                eigenvalues, UtW, Uty, n_cvt, l_min, l_max
            )

        eigenvalues_jax = jax.device_put(eigenvalues, placement.rep)
        UtW_jax = jax.device_put(UtW, placement.rep)
        Uty_jax = jax.device_put(Uty, placement.rep)

        jax_chunk_size = _compute_chunk_size(
            n_filtered, n_devices=placement.n_devices, n_samples=n_samples
        )

        def _prepare_jax_chunk(
            start: int, end: int, geno: np.ndarray
        ) -> tuple[np.ndarray, int]:
            """Slice a genotype subset and prepare UtG for device transfer."""
            geno_slice = geno[:, start:end]
            return prepare_utg_chunk(
                geno_slice, eigenvectors, placement, rotation_threads
            )

        total_at_lmin = 0
        total_at_lmax = 0
        results: list[AssocResult] = []

        bed_file = Path(f"{bed_path}.bed")
        with open_bed(bed_file) as bed:
            for disk_start in range(0, n_filtered, col_chunk_size):
                disk_end = min(disk_start + col_chunk_size, n_filtered)
                disk_col_indices = global_filtered_indices[disk_start:disk_end]

                geno_disk_chunk = bed.read(
                    index=np.s_[valid_indices, disk_col_indices],
                    dtype=np.float64,
                )

                chunk_filtered_means = filtered_means_all[disk_start:disk_end]
                filtered_means_broadcast = chunk_filtered_means.reshape(1, -1)
                missing_mask = np.isnan(geno_disk_chunk)
                geno_disk_chunk = np.where(
                    missing_mask, filtered_means_broadcast, geno_disk_chunk
                )

                n_disk_subset = geno_disk_chunk.shape[1]
                jax_starts = compute_subchunk_starts(
                    n_disk_subset, jax_chunk_size, placement.n_devices
                )
                jax_ends = [
                    jax_starts[i + 1] if i + 1 < len(jax_starts) else n_disk_subset
                    for i in range(len(jax_starts))
                ]

                UtG_np, actual_jax_len = _prepare_jax_chunk(
                    jax_starts[0], jax_ends[0], geno_disk_chunk
                )
                UtG_jax = jax.device_put(UtG_np, placement.snp)
                del UtG_np

                for i, _jax_start in enumerate(jax_starts):
                    current_actual_len = actual_jax_len
                    current_UtG = UtG_jax

                    if i + 1 < len(jax_starts):
                        UtG_np, actual_jax_len = _prepare_jax_chunk(
                            jax_starts[i + 1], jax_ends[i + 1], geno_disk_chunk
                        )
                        UtG_jax = jax.device_put(UtG_np, placement.snp)
                        del UtG_np

                    try:
                        Uab_batch = batch_compute_uab(
                            n_cvt, UtW_jax, Uty_jax, current_UtG
                        )

                        chunk_result = _compute_lmm_chunk(
                            lmm_mode,
                            n_cvt,
                            eigenvalues_jax,
                            Uab_batch,
                            n_samples,
                            l_min=l_min,
                            l_max=l_max,
                            n_grid=n_grid,
                            n_refine=n_refine,
                            Hi_eval_null=Hi_eval_null_jax,
                            logl_H0=logl_H0,
                        )
                        block_chunk_result(chunk_result, lmm_mode)
                    except Exception as e:
                        log_jax_error(
                            e,
                            chunk_label=f"LOCO {i + 1}",
                            chunk_snps=current_actual_len,
                            n_samples=n_samples,
                            n_cvt=n_cvt,
                        )
                        raise

                    subchunk_start = disk_start + jax_starts[i]
                    subchunk_end = subchunk_start + current_actual_len
                    arrays = _chunk_result_to_numpy(
                        chunk_result,
                        _ACCUM_KEYS[lmm_mode],
                        current_actual_len,
                    )
                    n_lmin, n_lmax = count_lambda_boundary_hits(
                        lmm_mode, arrays, l_min, l_max
                    )
                    total_at_lmin += n_lmin
                    total_at_lmax += n_lmax

                    if writer is not None:
                        writer.write_arrays_batch(
                            lmm_mode,
                            global_filtered_indices[subchunk_start:subchunk_end],
                            snp_info,
                            filtered_afs[subchunk_start:subchunk_end],
                            filtered_miss[subchunk_start:subchunk_end],
                            arrays,
                        )
                    else:
                        subchunk_results = list(
                            _yield_chunk_results(
                                lmm_mode,
                                np.arange(subchunk_start, subchunk_end),
                                global_filtered_indices,
                                filtered_afs,
                                filtered_miss,
                                snp_info,
                                arrays,
                            )
                        )
                        results.extend(subchunk_results)

                    del arrays, chunk_result, Uab_batch, current_UtG
                    if i + 1 >= len(jax_starts):
                        UtG_jax = None

                del geno_disk_chunk

        log_lambda_boundary_warning(
            total_at_lmin, total_at_lmax, l_min, l_max, prefix="LOCO "
        )

        return results, chr_pve, chr_pve_se
    finally:
        del eigenvalues_jax, UtW_jax, Uty_jax


def _run_lmm_for_chromosome_numpy(
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    snp_info: list,
    maf_threshold: float,
    miss_threshold: float,
    lmm_mode: int,
    valid_mask: np.ndarray,
    show_progress: bool = True,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    snps_global_mask: np.ndarray | None = None,
    col_chunk_size: int = 5_000,
    writer: IncrementalAssocWriter | None = None,
    chr_name: str = "",
    snp_stats_cache: SnpStatsCache | None = None,
    compute_pve: bool = False,
) -> tuple[list[AssocResult], float | None, float | None]:
    """Run NumPy LMM association on a single chromosome's SNPs.

    Pure-NumPy implementation — no JAX dependency. Mirrors the structure of
    _run_lmm_for_chromosome but uses NumPy functions throughout.

    Reads the chromosome's SNPs from the BED file in column chunks
    (two-pass: statistics, then association), never allocating the full
    chromosome genotype matrix.

    Args:
        bed_path: PLINK file prefix.
        chr_snp_indices: Column indices for this chromosome's SNPs.
        eigenvalues: Eigenvalues from LOCO kinship eigendecomp.
        eigenvectors: Eigenvectors from LOCO kinship eigendecomp.
        phenotypes: Phenotype vector (n_valid_samples,), already filtered.
        covariates: Covariate matrix (n_valid_samples, n_cvt) or None.
        snp_info: Full SNP metadata list (indexed by global SNP index).
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        valid_mask: Boolean mask for valid samples (for genotype subsetting).
        show_progress: Whether to log progress.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations.
        snps_global_mask: Boolean mask over all SNPs (True = included by -snps), or
            None. Pre-indexed: `snps_global_mask[chr_snp_indices]` gives the
            per-chromosome mask. Avoids per-chromosome np.isin computation.
        col_chunk_size: Number of SNP columns per disk read chunk.
        writer: Optional incremental writer for streaming results to disk.
            When provided, results are written directly and an empty list
            is returned. When None, results are accumulated and returned.
        compute_pve: If True, compute PVE from null model REML lambda.
            Set for each chromosome until PVE is successfully computed
            (typically the first chromosome with passing SNPs).
        snp_stats_cache: Global SNP statistics from kinship PASS 1 (LOCO-01).
            When provided, per-chromosome stats are extracted by slicing
            cache.col_means[chr_snp_indices] — eliminates a BED re-read.
            Filtering uses cache.n_samples (all-sample count) as denominator.
            When None, falls back to _collect_chr_snp_stats (legacy behavior).

    Returns:
        Tuple of (results, pve, pve_se) where results is a list of AssocResult
        (empty if writer used), pve is the PVE estimate (None unless
        compute_pve=True), and pve_se is the standard error of PVE (None
        unless compute_pve=True and likelihood surface is not flat).
    """
    n_samples = phenotypes.shape[0]
    valid_indices = np.where(valid_mask)[0]

    # === PASS 1: Chunked SNP statistics + filtering ===
    if snp_stats_cache is not None:
        # Use cached global stats, sliced to this chromosome (LOCO-01).
        # Stats were computed over ALL samples during kinship PASS 1.
        # Used for filtering only (MAF, missing rate, monomorphism) — the
        # actual genotype data is read fresh in PASS 2 using valid_indices.
        col_means = snp_stats_cache.col_means[chr_snp_indices]
        miss_counts = snp_stats_cache.miss_counts[chr_snp_indices]
        col_vars = snp_stats_cache.col_vars[chr_snp_indices]
        # Use the cache's sample count as denominator — stats were computed
        # over this population. Using n_valid would inflate miss_rates.
        filter_n_samples = snp_stats_cache.n_samples
        # Suppress per-chr n_unexpected warning: already logged in PASS 1.
        n_unexpected = 0
    else:
        col_means, miss_counts, col_vars, n_unexpected = _collect_chr_snp_stats(
            bed_path, chr_snp_indices, valid_indices, col_chunk_size
        )
        # Fallback stats are computed over valid_indices rows only,
        # so n_samples (= n_valid = phenotypes.shape[0]) is the correct denominator.
        filter_n_samples = n_samples

    filter_result = _filter_chr_snps(
        col_means,
        miss_counts,
        col_vars,
        filter_n_samples,
        maf_threshold,
        miss_threshold,
        chr_snp_indices,
        snps_global_mask,
        n_unexpected,
        show_progress,
    )
    if filter_result is None:
        return [], None, None

    (
        _local_filtered_indices,
        global_filtered_indices,
        filtered_afs,
        filtered_miss,
        filtered_means_all,
    ) = filter_result
    n_filtered = len(global_filtered_indices)

    # === PASS 2: Chunked NumPy association ===
    # Build covariate matrix
    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Rotation uses all physical cores (pure BLAS, no JAX)
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = eigenvectors.T @ W
        Uty = eigenvectors.T @ phenotypes

    # Compute null model (NumPy version, returns plain numpy arrays)
    logl_H0, _lambda_null_mle, Hi_eval_null = _compute_null_model_common(
        lmm_mode,
        eigenvalues,
        UtW,
        Uty,
        n_cvt,
        show_progress=False,
        l_min=l_min,
        l_max=l_max,
    )

    chr_pve = None
    chr_pve_se = None
    if compute_pve:
        chr_pve, chr_pve_se = compute_and_log_pve(
            eigenvalues, UtW, Uty, n_cvt, l_min, l_max
        )

    # Compute chunk size based on RAM budget
    chunk_size = _compute_chunk_size_numpy(n_samples, n_filtered, n_cvt)

    # Pre-allocate result arrays
    write_offset = 0
    arrays_out: dict[str, np.ndarray] = {
        key: np.empty(n_filtered, dtype=np.float64) for key in _RESULT_FIELDS[lmm_mode]
    }
    results: list[AssocResult] = []

    bed_file = Path(f"{bed_path}.bed")
    with open_bed(bed_file) as bed:
        for disk_start in range(0, n_filtered, col_chunk_size):
            disk_end = min(disk_start + col_chunk_size, n_filtered)
            disk_col_indices = global_filtered_indices[disk_start:disk_end]

            geno_disk_chunk = bed.read(
                index=np.s_[valid_indices, disk_col_indices],
                dtype=np.float64,
            )

            # Impute missing values with column means
            chunk_filtered_means = filtered_means_all[disk_start:disk_end]
            missing_mask = np.isnan(geno_disk_chunk)
            geno_disk_chunk = np.where(
                missing_mask, chunk_filtered_means.reshape(1, -1), geno_disk_chunk
            )

            # Process disk chunk in numpy sub-chunks
            n_disk_subset = geno_disk_chunk.shape[1]

            for sub_start in range(0, n_disk_subset, chunk_size):
                sub_end = min(sub_start + chunk_size, n_disk_subset)
                geno_sub = geno_disk_chunk[:, sub_start:sub_end]

                # Rotate genotypes
                with blas_threads(rotation_threads):
                    UtG = eigenvectors.T @ geno_sub

                # Compute Uab batch
                Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)

                # Mode dispatch
                try:
                    cr = _compute_lmm_chunk_numpy(
                        lmm_mode,
                        n_cvt,
                        eigenvalues,
                        Uab_batch,
                        n_samples,
                        l_min=l_min,
                        l_max=l_max,
                        n_grid=n_grid,
                        n_refine=n_refine,
                        Hi_eval_null=Hi_eval_null,
                        logl_H0=logl_H0,
                    )
                except Exception as e:
                    logger.error(
                        f"NumPy LMM computation failed on chr {chr_name}, "
                        f"sub-chunk [{sub_start}:{sub_end}] "
                        f"({sub_end - sub_start} SNPs), "
                        f"n_samples={n_samples}, n_cvt={n_cvt}: {e}"
                    )
                    raise

                # Write sub-chunk results to pre-allocated arrays
                actual_len = sub_end - sub_start
                s = slice(write_offset, write_offset + actual_len)
                for key in arrays_out:
                    arrays_out[key][s] = cr[key][:actual_len]
                write_offset += actual_len

            del geno_disk_chunk

    if write_offset != n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {write_offset} results, "
            f"expected {n_filtered}. This is an internal error."
        )

    # Count lambda boundary hits and log warnings
    n_lmin, n_lmax = count_lambda_boundary_hits(lmm_mode, arrays_out, l_min, l_max)
    log_lambda_boundary_warning(n_lmin, n_lmax, l_min, l_max, prefix="LOCO ")

    # Flush results
    if writer is not None:
        writer.write_arrays_batch(
            lmm_mode,
            global_filtered_indices,
            snp_info,
            filtered_afs,
            filtered_miss,
            arrays_out,
        )
    else:
        results = list(
            _yield_chunk_results(
                lmm_mode,
                np.arange(n_filtered),
                global_filtered_indices,
                filtered_afs,
                filtered_miss,
                snp_info,
                arrays_out,
            )
        )

    return results, chr_pve, chr_pve_se
