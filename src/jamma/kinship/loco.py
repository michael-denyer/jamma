"""LOCO (Leave-One-Chromosome-Out) kinship computation.

LOCO kinship is computed via the subtraction approach:

    K_loco_c = (S_full - S_c) / (p - p_c)

where S_full is the unscaled full kinship numerator, S_c is the contribution
from chromosome c, and p / p_c are the filtered SNP counts overall and on
chromosome c. This avoids redundant per-chromosome computation.

Streaming reads genotypes in one or more disk passes through
``jamma.kinship.stream.filtered_kinship_chunks``, the generator the regular
kinship path uses, and accumulates S_full and S_chr from its chunks. The first
pass also records the per-SNP statistics and filtered counts. When every
per-chromosome accumulator fits in memory alongside S_full, that one pass
suffices. Otherwise chromosomes are processed in batches across several
passes, with S_full computed once (in the first batch) and reused by every
later batch.
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
from jamma.core.memory import array_gb
from jamma.core.snp_filter import validate_snp_indices
from jamma.core.snp_stats import SnpStats
from jamma.io.plink import (
    PlinkMetadata,
    get_plink_metadata,
    partitions_from_metadata,
)
from jamma.kinship.accumulation import accumulate_kinship, validate_valid_indices
from jamma.kinship.missing import impute_and_center
from jamma.kinship.stream import (
    KinshipSnpFilter,
    SnpStatsSink,
    filtered_kinship_chunks,
    ksnps_restriction,
)
from jamma.utils import chr_sort_key


@dataclass(slots=True)
class LocoKinshipStream:
    """Consume-once stream of ``(chr_name, K_loco)`` LOCO matrices plus pass-1 stats.

    Wraps the generator ``compute_loco_kinship_streaming`` builds internally.
    Iterating it drives disk reads and dsyrk accumulation lazily, chromosome by
    chromosome; each yielded ``K_loco`` aliases one shared ``(n, n)`` buffer that is
    overwritten on the next advance (LOCO-03, no per-chromosome allocation). Consume
    it exactly once, in order — the contract a bare generator always had. Do not call
    ``list()``/``dict()`` on it directly; use ``materialize()``, which copies each
    matrix, or you get N references to the same final buffer.

    ``snp_stats`` holds the first pass's statistics over every SNP, on the
    filtering rows (``filter_sample_indices``, or every BED row). The LOCO
    association pass reuses them. Output row selection does not change them.
    The first pass ends before the first matrix is yielded, so they are
    readable from then on.
    """

    _matrices: Iterator[tuple[str, np.ndarray]]
    _stats: SnpStatsSink

    @property
    def snp_stats(self) -> SnpStats:
        """First-pass statistics; raises RuntimeError before the first yield."""
        if self._stats.stats is None:
            raise RuntimeError(
                "LOCO SNP statistics are complete only once the first kinship "
                "pass has yielded its first matrix"
            )
        return self._stats.stats

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


def _yield_loco_matrices(
    S_full_np: np.ndarray,
    S_chr: dict[str, np.ndarray],
    batch_chrs: list[str],
    n_chr_filtered: dict[str, int],
    n_filtered: int,
    K_loco_buf: np.ndarray,
) -> Iterator[tuple[str, np.ndarray]]:
    """Compute and yield LOCO kinship matrices from S_full and per-chr accumulators.

    Each yielded matrix IS the shared ``K_loco_buf``, overwritten on the next
    iteration (LOCO-03: no per-chromosome allocation). This is the consume-once
    contract ``LocoKinshipStream`` documents; consumers that need every matrix at
    once go through ``LocoKinshipStream.materialize()``, which copies.

    Yields:
        (chr_name, K_loco) pairs in ``batch_chrs`` order.

    Raises:
        ValueError: If all filtered SNPs are on a single chromosome.
    """
    for chr_name in batch_chrs:
        p_chr = n_chr_filtered[chr_name]
        p_loco = n_filtered - p_chr

        if p_loco == 0:
            raise ValueError(
                f"Cannot compute LOCO kinship: all {n_filtered} filtered SNPs "
                f"are on chromosome '{chr_name}'."
            )

        if p_chr == 0:
            np.divide(S_full_np, p_loco, out=K_loco_buf)
        else:
            np.subtract(S_full_np, S_chr.pop(chr_name), out=K_loco_buf)
            K_loco_buf /= p_loco
        logger.debug(
            f"LOCO chr {chr_name}: {p_chr} SNPs excluded, {p_loco} SNPs retained"
        )
        yield (chr_name, K_loco_buf)


def _accumulate_pass(
    chunks: Iterator[tuple[np.ndarray, np.ndarray]],
    chromosomes: np.ndarray,
    chr_subset: set[str],
    S_full: np.ndarray | None,
    n_out: int,
    n_chr_filtered: dict[str, int] | None,
) -> dict[str, np.ndarray]:
    """Accumulate one pass of chunks into S_full and the S_chr of ``chr_subset``.

    An S_chr is allocated on the first chunk that carries its chromosome, so a
    chromosome whose SNPs are all filtered out never costs an accumulator.
    When ``n_chr_filtered`` is given, the pass also counts kept SNPs per
    chromosome into it.
    """
    S_chr: dict[str, np.ndarray] = {}
    for X, global_idx in chunks:
        if S_full is not None:
            accumulate_kinship(S_full, X)
        chunk_chrs = chromosomes[global_idx]
        present, counts = np.unique(chunk_chrs, return_counts=True)
        if n_chr_filtered is not None:
            for chr_name, count in zip(present.tolist(), counts.tolist(), strict=True):
                n_chr_filtered[chr_name] += count
        for chr_name in chr_subset.intersection(present.tolist()):
            if chr_name not in S_chr:
                S_chr[chr_name] = np.zeros((n_out, n_out), dtype=np.float64)
            accumulate_kinship(S_chr[chr_name], X[:, chunk_chrs == chr_name])
        del X
    return S_chr


class LocoRetainedSet(NamedTuple):
    """What the LOCO kinship stream keeps live while its consumer works.

    Attributes:
        matrix_gb: One ``n_mat x n_mat`` accumulator: S_full, K_loco_buf, or
            one S_chr.
        chunk_buffer_gb: One disk read of ``chunk_size`` SNPs over every
            input sample. Subsetting happens after the read, so the buffer is
            ``n_samples`` wide even when the matrices are ``n_mat`` wide.
    """

    matrix_gb: float
    chunk_buffer_gb: float

    @property
    def while_consuming_gb(self) -> float:
        """S_full, K_loco_buf, one S_chr, and the disk buffer."""
        return 3 * self.matrix_gb + self.chunk_buffer_gb


def loco_retained_set(n_mat: int, n_samples: int, chunk_size: int) -> LocoRetainedSet:
    """Size the retained set for ``n_mat``-order matrices over ``n_samples`` inputs."""
    return LocoRetainedSet(array_gb(n_mat, n_mat), array_gb(n_samples, chunk_size))


class _LocoPassPlan(NamedTuple):
    """Chromosomes per disk pass for streaming LOCO kinship.

    Attributes:
        batch_size: Chromosomes accumulated per disk pass.
        single_pass: One pass covers every chromosome that has SNPs.
        required_gb: Peak for this batch size: the retained set, the
            consumer, and ``batch_size - 1`` further S_chr accumulators.
    """

    batch_size: int
    single_pass: bool
    required_gb: float


def plan_loco_passes(
    retained: LocoRetainedSet,
    consumer_gb: float,
    n_chr_with_snps: int,
    available_gb: float,
    *,
    budget_gb: float | None,
    max_batch_chrs: int | None,
) -> _LocoPassPlan:
    """Pick the chromosomes-per-pass batch size that fits both ceilings.

    Pure sizing math (no I/O), so it can be unit-tested at scale. A pass
    holds S_full, K_loco_buf, the disk buffer, and one S_chr per chromosome
    in the batch, with ``consumer_gb`` reserved for the eigen or association
    work that runs while the stream is live. The batch is the largest count
    whose peak fits ``memory.headroom_gb(available_gb)``, capped by
    ``budget_gb``, ``max_batch_chrs``, and ``n_chr_with_snps``, and never
    below one chromosome; the caller gates that floor. A single-pass run is
    ``batch_size == n_chr_with_snps``, one batch covering every chromosome.

    The margin is taken of the requirement via ``memory.headroom_gb``, the
    same margin ``fits`` applies to the batch it produces, so the two agree
    except at an exact tie, which the final ``fits`` check settles.

    Args:
        retained: The stream's live matrices and disk buffer.
        consumer_gb: Peak the downstream eigen and association work holds
            while the stream is live.
        n_chr_with_snps: Chromosomes that retain SNPs after filtering.
        available_gb: Free RAM in GB (the caller reads psutil once).
        budget_gb: User ceiling in GB without the physical-RAM margin, or None.
        max_batch_chrs: Cap on chromosomes per pass, or None.

    Returns:
        The batch size, whether it is a single pass, and its peak.
    """
    fixed_gb = 2 * retained.matrix_gb + retained.chunk_buffer_gb + consumer_gb
    capacity_gb = memory.headroom_gb(available_gb)
    if budget_gb is not None:
        capacity_gb = min(capacity_gb, budget_gb)
    batch_size = min(
        n_chr_with_snps, max(1, int((capacity_gb - fixed_gb) / retained.matrix_gb))
    )
    if max_batch_chrs is not None:
        batch_size = min(batch_size, max_batch_chrs)
    if batch_size > 1 and not memory.fits(
        fixed_gb + batch_size * retained.matrix_gb, available_gb
    ):
        batch_size -= 1
    return _LocoPassPlan(
        batch_size=batch_size,
        single_pass=n_chr_with_snps <= batch_size,
        required_gb=fixed_gb + batch_size * retained.matrix_gb,
    )


def _batch_chromosomes(
    chrs: list[str], n_chr_filtered: dict[str, int], batch_size: int
) -> list[list[str]]:
    batches: list[list[str]] = [[]]
    accumulators = 0
    for chr_name in chrs:
        if n_chr_filtered[chr_name] > 0:
            if accumulators == batch_size:
                batches.append([])
                accumulators = 0
            accumulators += 1
        batches[-1].append(chr_name)
    return batches


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
    consumer_gb: float,
    meta: PlinkMetadata | None = None,
) -> LocoKinshipStream:
    """Compute LOCO kinship matrices from disk-streamed genotypes.

    See the module docstring for the subtraction algorithm. Every pass reads
    ``filtered_kinship_chunks``, which decides the SNP filter per chunk. The
    first pass accumulates S_full and the first batch's S_chr, and records
    the per-SNP statistics and filtered counts; later passes accumulate only
    their batch's S_chr. The filter is not known before the first pass, so
    ``plan_loco_passes`` sizes batches against every chromosome the BIM (and
    -ksnps) leaves SNPs on, and picks one pass when all of them fit
    alongside S_full. An S_chr is allocated only for a chromosome that keeps
    SNPs. The ValueErrors below surface on the first advance of the stream.

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
        mem_budget: User-set ceiling in GB, or None for no ceiling. A second
            capacity beside physical RAM: the batch planner sizes the
            chromosome batch against the smaller of the two, and the gate
            before pass 1 vetoes the run when the retained set plus
            ``consumer_gb`` exceeds it.
        filter_sample_indices: Samples used for SNP filtering, or None for all
            BED samples. Independent of output rows and full-population centering.
        _max_batch_chrs: Cap on chromosomes per pass, applied on top of the
            memory-based batch size. Tests use it to exercise multi-pass
            without mocking psutil.
        consumer_gb: Peak the downstream eigen and association work holds
            while this stream is live. The gate and the batch planner both
            reserve it beside the retained set.
        meta: PLINK metadata already read from ``bed_path``, or None to read
            it here.

    Returns:
        A consume-once LocoKinshipStream. Iterate it for (chr_name, K_loco) pairs,
        where chr_name is the chromosome being excluded and K_loco has shape
        (n_valid, n_valid) when valid_indices is provided, else
        (n_samples, n_samples). Read ``.snp_stats`` for the first pass's
        statistics over the filtering rows, once the first matrix is yielded.
        Centering always uses all BED samples; valid_indices selects matrix
        rows only. Each yielded matrix aliases a shared
        buffer overwritten on the next advance, so consume it before advancing, or call
        ``.materialize()`` to collect independent copies.

    Raises:
        MemoryError: If check_memory=True and the retained set (S_full,
            K_loco_buf, one S_chr, the disk buffer) plus ``consumer_gb`` does
            not fit available RAM, or exceeds ``mem_budget``.
        FileNotFoundError: If the PLINK .bed file does not exist.
        ValueError: If no SNPs pass filtering, or if all filtered SNPs are on
            a single chromosome.
    """
    start_time = time.perf_counter()

    # Get dimensions and chromosome metadata
    if meta is None:
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

    n_mat = len(valid_indices) if valid_indices is not None else n_samples
    logger.info("Computing LOCO Kinship (streaming)")
    logger.info(
        f"  Individuals: {n_mat:,}"
        + (f" (filtered from {n_samples:,})" if n_mat != n_samples else "")
    )
    logger.info(f"  SNPs: {n_snps:,}")
    logger.info(f"  Chromosomes: {len(unique_chrs)}")
    logger.info(f"  Chunk size: {chunk_size:,}")

    retained = loco_retained_set(n_mat, n_samples, chunk_size)
    if check_memory:
        memory.require(
            retained.while_consuming_gb + consumer_gb,
            memory.available_ram_gb(),
            "LOCO working set (3 accumulators + disk buffer "
            f"{retained.while_consuming_gb:.1f}GB, consumer {consumer_gb:.1f}GB)",
            budget_gb=mem_budget,
        )

    validate_snp_indices(ksnps_indices, n_snps, "-ksnps")
    # SNP filtering and output rows are independent. The LMM caller filters on
    # analysed samples even when saving a full matrix; centering uses all rows.
    snp_filter = KinshipSnpFilter(
        maf_threshold,
        miss_threshold,
        ksnps_restriction(ksnps_indices, n_snps),
        filter_sample_indices,
    )
    candidates = (
        chromosomes
        if snp_filter.restriction is None
        else chromosomes[snp_filter.restriction]
    )
    n_chr_candidates: dict[str, int] = dict.fromkeys(unique_chrs, 0)
    for chr_name, count in zip(*np.unique(candidates, return_counts=True), strict=True):
        n_chr_candidates[str(chr_name)] = int(count)
    # The filter is decided while the first pass reads, so the planner sizes
    # against every chromosome the BIM (and -ksnps) could leave SNPs on.
    n_chr_planned = sum(1 for count in n_chr_candidates.values() if count > 0)

    available_gb = memory.available_ram_gb()
    plan = plan_loco_passes(
        retained,
        consumer_gb,
        n_chr_planned,
        available_gb,
        budget_gb=mem_budget,
        max_batch_chrs=_max_batch_chrs,
    )
    if check_memory:
        memory.require(
            plan.required_gb,
            available_gb,
            "LOCO kinship",
            budget_gb=mem_budget,
        )

    if mem_budget is not None:
        logger.info(f"  Memory budget: {mem_budget:.1f}GB")

    batch_size = plan.batch_size
    first_batch = _batch_chromosomes(unique_chrs, n_chr_candidates, batch_size)[0]
    stats_sink = SnpStatsSink.for_snps(n_snps)

    def _pass(
        chr_subset: list[str],
        desc: str,
        *,
        S_full: np.ndarray | None,
        n_chr_filtered: dict[str, int] | None,
    ) -> dict[str, np.ndarray]:
        chr_set = set(chr_subset)
        chunks = filtered_kinship_chunks(
            bed_path,
            n_snps=n_snps,
            chunk_size=chunk_size,
            snp_filter=snp_filter,
            transform=impute_and_center,
            output_rows=valid_indices,
            show_progress=show_progress,
            desc=desc,
            stats_sink=stats_sink if S_full is not None else None,
            wanted=None
            if S_full is not None
            else lambda global_idx: not chr_set.isdisjoint(chromosomes[global_idx]),
        )
        return _accumulate_pass(
            chunks, chromosomes, chr_set, S_full, n_mat, n_chr_filtered
        )

    def _generate() -> Iterator[tuple[str, np.ndarray]]:
        if plan.single_pass and plan.required_gb > 10:
            logger.info(
                f"LOCO streaming: single-pass ({plan.required_gb:.1f}GB for "
                f"up to {n_chr_planned} chromosomes)"
            )
        elif not plan.single_pass:
            single_pass_gb = (
                retained.while_consuming_gb
                + consumer_gb
                + (n_chr_planned - 1) * retained.matrix_gb
            )
            logger.warning(
                f"LOCO streaming: multi-pass mode ({batch_size} chromosomes/pass). "
                f"Single-pass would need {single_pass_gb:.1f}GB, "
                f"available {available_gb:.1f}GB."
            )

        S_full = np.zeros((n_mat, n_mat), dtype=np.float64)
        n_chr_filtered = dict.fromkeys(unique_chrs, 0)
        S_chr = _pass(
            first_batch,
            "LOCO: kinship accumulation"
            if plan.single_pass
            else f"LOCO: pass 1 (S_full + {len(first_batch)} chr)",
            S_full=S_full,
            n_chr_filtered=n_chr_filtered,
        )
        stats = stats_sink.stats
        assert stats is not None
        if stats.n_unexpected > 0:
            logger.warning(
                f"LOCO kinship genotype validation: {stats.n_unexpected} values "
                "outside expected range {0, 1, 2, NaN}"
            )
        n_filtered = sum(n_chr_filtered.values())
        if ksnps_indices is not None:
            logger.info(
                f"Kinship SNP list: restricting to {len(ksnps_indices)} requested "
                f"SNPs ({n_filtered} retained after intersection)"
            )
        if n_filtered < n_snps:
            logger.info(
                f"LOCO kinship filtering: {n_filtered:,} SNPs retained, "
                f"{n_snps - n_filtered:,} removed (MAF/missing/monomorphic)"
            )
        chrs_without_snps = [c for c in unique_chrs if n_chr_filtered[c] == 0]
        if chrs_without_snps:
            logger.warning(
                f"{len(chrs_without_snps)} chromosome(s) have 0 ksnps after "
                f"filtering: {chrs_without_snps}. LOCO will use full kinship for "
                "these (nothing to leave out)."
            )

        rest = unique_chrs[len(first_batch) :]
        batches = [first_batch]
        if rest:
            batches += _batch_chromosomes(rest, n_chr_filtered, batch_size)
        n_batches = len(batches)
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"LOCO streaming accumulation complete in {elapsed:.2f}s, "
            f"computing {len(first_batch)} LOCO matrices"
            if n_batches == 1
            else f"LOCO: pass 1/{n_batches} accumulation complete in {elapsed:.2f}s"
        )

        K_loco_buf = np.empty_like(S_full)
        for i, batch_chrs in enumerate(batches):
            if i > 0:
                accumulated = [c for c in batch_chrs if n_chr_filtered[c] > 0]
                S_chr = _pass(
                    accumulated,
                    f"LOCO: pass {i + 1}/{n_batches} ({len(accumulated)} chr)",
                    S_full=None,
                    n_chr_filtered=None,
                )
            yield from _yield_loco_matrices(
                S_full, S_chr, batch_chrs, n_chr_filtered, n_filtered, K_loco_buf
            )
            del S_chr
            gc.collect()

        if n_batches > 1:
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"LOCO multi-pass complete in {elapsed:.2f}s, "
                f"{n_batches} passes over "
                f"{len(unique_chrs) - len(chrs_without_snps)} chromosomes"
            )

    return LocoKinshipStream(_matrices=_generate(), _stats=stats_sink)
