"""LOCO (Leave-One-Chromosome-Out) kinship computation.

LOCO kinship is computed via the subtraction approach:

    K_loco_c = (S_full - S_c) / (p - p_c)

where S_full is the unscaled full kinship numerator, S_c is the contribution
from chromosome c, and p / p_c are the filtered SNP counts overall and on
chromosome c. This avoids redundant per-chromosome computation.

Streaming reads genotypes in one or more disk passes and accumulates S_full
and S_chr via ``jamma.kinship.stream``'s shared chunk-selection primitives.
When every per-chromosome accumulator fits in memory alongside S_full, one
pass over the BED file suffices. Otherwise chromosomes are processed in
batches across several passes, with S_full computed once (in the first
batch) and reused by every later batch.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma.core import memory
from jamma.core.eigen_plan import array_gb, square_matrix_gb
from jamma.core.progress import progress_iterator
from jamma.core.snp_stats import SnpStatsCache, collect_streamed_snp_stats
from jamma.io.plink import (
    get_plink_metadata,
    partitions_from_metadata,
    stream_genotype_chunks,
)
from jamma.kinship.accumulation import (
    accumulate_kinship,
    select_kinship_snps,
    selected_chunks,
    validate_valid_indices,
)
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
            reuse when its sample population matches, or None when SNP filtering
            uses a sample subset. Output row selection does not change these
            statistics. Available before iteration, since PASS 1 runs eagerly.
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
    S_full_accum: bool,
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
            multi-pass batches after S_full is already computed (``S_full_accum=(i
            == 0)`` in the caller's batch loop — only the first batch computes it).
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

    for X_centered, global_idx in selected_chunks(
        chunk_iter, snp_indices, valid_indices, keep=keep
    ):
        if S_full is not None:
            accumulate_kinship(S_full, X_centered)

        chunk_chrs = chromosomes[global_idx]
        for chr_name in set(chunk_chrs) & chr_set:
            X_chr_part = X_centered[:, chunk_chrs == chr_name]
            accumulate_kinship(S_chr[chr_name], X_chr_part)

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

    A single-pass run still goes through the batch loop with ``batch_size =
    n_chr_with_snps``, one batch covering every chromosome.

    Both decisions use ``memory.fits``: single-pass when the whole-run peak
    fits, and otherwise the largest batch whose peak (the two live matrices,
    the chunk buffer, the eigendecomp reserve and ``batch_size`` S_chr
    matrices) still fits. The margin is therefore always taken of the
    requirement, via ``memory.headroom_gb``, so the multi-pass budget is the
    same margin ``fits`` would apply to the batch it produces.

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
        single_pass = memory.fits(single_pass_gb, available_gb)
        if single_pass:
            batch_size = n_chr_with_snps  # one batch covers every chromosome
        else:
            # The consumer eigendecomposes each K_loco while the generator is
            # suspended with remaining S_chr matrices still alive. Reserve
            # eigendecomp workspace (DSYEVR peak on the n_mat-sized K_loco) so
            # the batch doesn't exhaust memory before eigendecomp can run.
            fixed_gb = 2 * matrix_gb + chunk_buffer_gb + dsyevr_peak_gb(n_mat)
            usable_gb = memory.headroom_gb(available_gb) - fixed_gb
            batch_size = max(1, int(usable_gb / matrix_gb))
            if batch_size > 1 and not memory.fits(
                fixed_gb + batch_size * matrix_gb, available_gb
            ):
                batch_size -= 1  # floor landed on the tie fits rejects

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
    mem_budget: float | None = None,
    *,
    filter_sample_indices: np.ndarray | None = None,
    _max_batch_chrs: int | None = None,
) -> LocoKinshipStream:
    """Compute LOCO kinship matrices from disk-streamed genotypes.

    See the module docstring for the subtraction algorithm. Pass 1 computes
    per-SNP statistics for filtering (MAF, missingness, variance). Pass 2+
    streams filtered SNPs in one or more chromosome batches, accumulating
    S_full (first batch only, threaded into every later batch) and each
    batch's S_chr; ``_decide_loco_passes`` picks ``batch_size ==
    n_chr_with_snps`` (single disk pass) when every chromosome's accumulator
    fits in memory alongside S_full, else a smaller batch across more passes.

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
        mem_budget: User-set ceiling in GB, or None for no ceiling. Vetoes the
            run the same way ``_reject_if_over_budget`` vetoes the batch and
            streaming paths; it does not resize the chromosome batch.
        filter_sample_indices: Samples used for SNP filtering, or None for all
            BED samples. Independent of output rows and full-population centering.
        _max_batch_chrs: Debug override forcing a fixed chromosomes-per-pass
            batch size (bypasses memory-based sizing). Used by tests to exercise
            multi-pass without mocking psutil.

    Returns:
        A consume-once LocoKinshipStream. Iterate it for (chr_name, K_loco) pairs,
        where chr_name is the chromosome being excluded and K_loco has shape
        (n_valid, n_valid) when valid_indices is provided, else
        (n_samples, n_samples). Read ``.snp_stats`` for the PASS-1 all-sample
        SnpStatsCache, or None when filtering uses a sample subset. Centering
        always uses all BED samples;
        valid_indices selects matrix rows only. Each yielded matrix aliases a shared
        buffer overwritten on the next advance, so consume it before advancing, or call
        ``.materialize()`` to collect independent copies.

    Raises:
        MemoryError: If check_memory=True and insufficient memory for even
            S_full + one S_chr, or if mem_budget is set and the estimate
            exceeds it.
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
    if filter_sample_indices is not None:
        validate_valid_indices(filter_sample_indices, n_samples)

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

    # SNP filtering and output rows are independent. The LMM caller filters on
    # analysed samples even when saving a full matrix; centering uses all rows.
    stats = collect_streamed_snp_stats(
        bed_path,
        n_snps=n_snps,
        n_samples=n_samples,
        chunk_size=chunk_size,
        sample_indices=filter_sample_indices,
        validate_genotypes=True,
        show_progress=show_progress,
        progress_label="LOCO: SNP statistics",
        dtype=np.float32,
        sample_scope="all_samples"
        if filter_sample_indices is None
        else "valid_samples",
    )

    if stats.n_unexpected > 0:
        logger.warning(
            f"LOCO kinship genotype validation: {stats.n_unexpected} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

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
        if filter_sample_indices is None
        else None
    )

    snp_selection = select_kinship_snps(
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

    if mem_budget is not None:
        logger.info(f"  Memory budget: {mem_budget:.1f}GB")

    if check_memory:
        memory.require(
            plan.min_required_gb,
            available_gb,
            f"LOCO kinship (S_full + K_loco_buf + one S_chr + eigendecomp "
            f"{plan.eigendecomp_min_gb:.1f}GB)",
            budget_gb=mem_budget,
        )

    batch_size = plan.batch_size
    # At least one batch always runs, even when n_chr_with_snps == 0 (every
    # chromosome lost all its SNPs to filtering): that lone batch still computes
    # S_full, which _yield_full_kinship_fallback below needs for every chromosome.
    n_batches = max(1, -(-n_chr_with_snps // batch_size)) if batch_size else 1

    def _generate() -> Iterator[tuple[str, np.ndarray]]:
        if plan.single_pass and plan.single_pass_gb > 10:
            logger.info(
                f"LOCO streaming: single-pass ({plan.single_pass_gb:.1f}GB for "
                f"{n_chr_with_snps} chromosomes)"
            )
        elif not plan.single_pass:
            logger.warning(
                f"LOCO streaming: multi-pass mode ({n_batches} passes, "
                f"{batch_size} chromosomes/pass). Single-pass would need "
                f"{plan.single_pass_gb:.1f}GB, available {available_gb:.1f}GB."
            )

        # One batch loop covers both the single-pass and multi-pass cases:
        # single-pass is the n_batches == 1 special case of the same loop.
        # S_full is accumulated only in the first batch (S_full_accum=(i == 0))
        # and threaded into every later batch, which accumulates only its S_chr.
        S_full_np: np.ndarray | None = None
        K_loco_buf: np.ndarray | None = None
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_chrs = chrs_with_snps[batch_start : batch_start + batch_size]
            desc = (
                "LOCO: kinship accumulation"
                if n_batches == 1
                else f"LOCO: pass {i + 1}/{n_batches} "
                f"({'S_full + ' if i == 0 else ''}{len(batch_chrs)} chr)"
            )

            batch_S_full, S_chr = _stream_s_full_and_chr(
                bed_path,
                n_samples,
                n_snps,
                snp_indices,
                chromosomes,
                batch_chrs,
                chunk_size,
                show_progress,
                desc=desc,
                S_full_accum=(i == 0),
                valid_indices=valid_indices,
            )
            if i == 0:
                S_full_np = batch_S_full
                K_loco_buf = np.empty_like(S_full_np)
                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"LOCO streaming accumulation complete in {elapsed:.2f}s, "
                    f"computing {len(S_chr)} LOCO matrices"
                    if n_batches == 1
                    else f"LOCO: pass 1/{n_batches} accumulation complete in "
                    f"{elapsed:.2f}s"
                )

            assert S_full_np is not None
            assert K_loco_buf is not None
            yield from _yield_loco_matrices(
                S_full_np, S_chr, n_chr_filtered, n_filtered, K_loco_buf
            )
            del S_chr
            gc.collect()

        assert S_full_np is not None  # batch 0 always ran and set it
        yield from _yield_full_kinship_fallback(
            S_full_np, chrs_without_snps, n_filtered
        )

        if n_batches > 1:
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
