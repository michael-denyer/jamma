"""LOCO LMM orchestrator.

Runs leave-one-chromosome-out LMM association by looping over chromosomes:
for each chromosome c, eigendecompose K_loco_c, run LMM on chromosome c's
SNPs using that eigendecomposition, discard K_loco_c.

Memory profile (sequential processing):
    At any point holds S_full (n^2*8) from the LOCO kinship generator,
    plus one K_loco (n^2*8) during eigendecomp, plus LMM working set.
    Each K_loco is discarded after eigendecomp.

``LocoConfig`` lives in ``loco_config`` and is re-exported here, so ``from
jamma.lmm.loco import LocoConfig`` keeps working. Where the eigenpairs come
from (cache or compute), and every file the cache involves, is
``loco_eigen.eigen_pairs_for``'s business.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
from bed_reader import open_bed
from loguru import logger

from jamma.core.snp_filter import validate_snp_indices
from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpSelection,
    SnpStats,
    collect_snp_stats_from_chunks,
)
from jamma.core.threading import get_loco_worker_count
from jamma.io.plink import (
    get_plink_metadata,
    partitions_from_metadata,
)
from jamma.kinship import SnpStatsCache
from jamma.lmm.association_plan import plan_association
from jamma.lmm.chunk_runner_numpy import RawLmmChunk
from jamma.lmm.genotype_source import (
    PreparedGenotypes,
    SampleBasis,
    bind_prepared_genotypes,
)
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.loco_config import DEFAULT_LOCO_CONFIG, LocoConfig
from jamma.lmm.loco_eigen import eigen_pairs_for
from jamma.lmm.runner_numpy import _run_numpy_lmm
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    TEST_TYPE_MAP,
    LmmConfig,
    LmmRunResult,
    LocoResult,
    SnpMeta,
)
from jamma.lmm.stats import AssocResult
from jamma.utils import chr_sort_key

__all__ = [
    "DEFAULT_LOCO_CONFIG",
    "LocoConfig",
    "run_lmm_loco",
]


def _collect_chr_snp_stats(
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    valid_indices: np.ndarray,
    col_chunk_size: int,
) -> SnpStats:
    """Collect per-SNP statistics for one chromosome via chunked BED reads.

    Shared by LOCO chromosome runners (pass-1 logic).

    Args:
        bed_path: PLINK file prefix (without extension).
        chr_snp_indices: Global column indices for this chromosome's SNPs.
        valid_indices: Row indices of valid (non-missing) samples.
        col_chunk_size: Number of SNP columns per disk read chunk.

    Returns:
        SnpStats with arrays of length len(chr_snp_indices). Stats are computed
        over valid_indices rows, so the denominator is len(valid_indices).
    """
    n_chr_snps = len(chr_snp_indices)

    bed_file = Path(f"{bed_path}.bed")

    def _chunks():
        with open_bed(bed_file) as bed:
            for chunk_start in range(0, n_chr_snps, col_chunk_size):
                chunk_end = min(chunk_start + col_chunk_size, n_chr_snps)
                chunk_col_indices = chr_snp_indices[chunk_start:chunk_end]
                geno_chunk = bed.read(
                    index=np.s_[valid_indices, chunk_col_indices],
                    dtype=np.float64,
                )
                yield geno_chunk, chunk_start, chunk_end

    return collect_snp_stats_from_chunks(
        _chunks(),
        n_snps=n_chr_snps,
        n_samples=len(valid_indices),
        global_indices=chr_snp_indices,
        validate_genotypes=True,
        sample_scope="valid_samples",
    )


def _chr_snp_stats_for_loco(
    snp_stats_cache: SnpStatsCache | None,
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    valid_indices: np.ndarray,
    *,
    all_samples_valid: bool,
    col_chunk_size: int,
) -> SnpStats:
    """Return per-chromosome SNP stats on the basis GEMMA uses.

    GEMMA computes each SNP's genotype mean/MAF and imputes missing genotypes over
    the *analysed* individuals only (``src/lmm.cpp`` ``AnalyzePlink``:
    ``x_mean /= (ni_test - n_miss)``, then missing genotype ``-> x_mean``). The
    all-sample statistics cached during kinship PASS 1 therefore match GEMMA only
    when every sample is analysed; when some phenotypes/covariates are missing the
    all-sample mean differs from the analysed-sample mean and would bias both the
    filter/AF and the missing-genotype imputation in PASS 2.

    So reuse the cache only when ``all_samples_valid`` (a free, exact match that
    avoids a per-chromosome BED re-read); otherwise recompute over
    ``valid_indices``, which is exactly what the non-cache / eigen-cache path does.
    """
    if snp_stats_cache is not None and all_samples_valid:
        if snp_stats_cache.sample_scope != "all_samples":
            raise ValueError(
                "LOCO SNP stats cache must use all-sample statistics; "
                f"got sample_scope={snp_stats_cache.sample_scope!r}"
            )
        return snp_stats_cache.take(chr_snp_indices)
    return _collect_chr_snp_stats(
        bed_path, chr_snp_indices, valid_indices, col_chunk_size
    )


def run_lmm_loco(
    bed_path: Path,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None = None,
    config: LmmConfig = DEFAULT_LMM_CONFIG,
    loco: LocoConfig = DEFAULT_LOCO_CONFIG,
    output_path: Path | None = None,
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
        config: Numerical settings shared with every other runner — MAF and
            missingness thresholds, lambda bounds and grid, test type,
            memory check and progress. See :class:`LmmConfig`.
        loco: LOCO-only settings — kinship and eigen output, SNP restriction,
            chunk width, text vs binary artifacts. See :class:`LocoConfig`.
        output_path: Path for incremental result writing, or None for in-memory.

    Returns:
        LocoResult with associations in biological chromosome order
        (1-22, X, Y, XY, MT). Associations list is empty if output_path
        is set (results written to disk).

    Raises:
        ValueError: If fewer than two chromosomes are present or if no samples
            have valid phenotypes. Invalid lmm_mode and write_eigen without
            eigen_dir are rejected earlier, when LmmConfig and LocoConfig are
            constructed.
    """
    show_progress = config.show_progress
    start_time = time.perf_counter()

    # Read LOCO worker count and log configuration
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
    n_samples_total = meta.n_samples
    n_snps_total = meta.n_snps

    validate_snp_indices(loco.snps_indices, n_snps_total)

    # Chromosome partitions (unfiltered) — derived from already-loaded metadata
    # to avoid a redundant BIM re-read
    partitions = partitions_from_metadata(meta)
    unique_chrs = sorted(partitions.keys(), key=chr_sort_key)

    if len(unique_chrs) < 2:
        raise ValueError(
            "LOCO requires SNPs on multiple chromosomes. "
            f"Found only {len(unique_chrs)} chromosome(s): {unique_chrs}"
        )

    logger.info("LOCO backend: numpy")

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

    phenotypes_valid = phenotypes[valid_mask]
    covariates_valid = covariates[valid_mask, :] if covariates is not None else None

    if show_progress:
        n_filtered_samples = n_samples_total - n_valid
        logger.info(
            f"  Analyzed individuals: {n_valid:,} ({n_filtered_samples} filtered)"
        )

    # Build SNP metadata columns for result construction
    snp_info = SnpMeta.from_plink_meta(meta)

    test_type = TEST_TYPE_MAP[config.lmm_mode]

    if output_path is None and n_snps_total > 100_000:
        logger.warning(
            f"LOCO in-memory mode with {n_snps_total:,} total SNPs. Results will "
            f"accumulate in memory. Provide output_path to stream results to disk."
        )

    all_results: list[AssocResult] = []

    with contextlib.ExitStack() as stack:
        writer = None
        if output_path is not None:
            writer = stack.enter_context(
                IncrementalAssocWriter(output_path, test_type=test_type)
            )

        source = eigen_pairs_for(
            bed_path,
            unique_chrs,
            loco=loco,
            maf_threshold=config.maf_threshold,
            miss_threshold=config.miss_threshold,
            valid_mask=valid_mask,
            partitions=partitions,
            check_memory=config.check_memory,
            show_progress=show_progress,
            mem_budget=config.mem_budget,
        )

        first_chr_pve: float | None = None
        first_chr_pve_se: float | None = None

        for chr_idx, (chr_name, eigenvalues_np, U) in enumerate(source.pairs):
            chr_snp_indices = partitions[chr_name]
            logger.debug(
                f"  chr {chr_name}: numpy backend, {len(chr_snp_indices)} SNPs"
            )

            chr_result = _run_lmm_for_chromosome_numpy(
                bed_path=bed_path,
                chr_snp_indices=chr_snp_indices,
                eigenvalues=eigenvalues_np,
                eigenvectors=U,
                phenotypes=phenotypes_valid,
                covariates=covariates_valid,
                snp_meta=snp_info,
                valid_mask=valid_mask,
                config=config,
                snps_indices=loco.snps_indices,
                col_chunk_size=loco.col_chunk_size,
                writer=writer,
                chr_name=chr_name,
                snp_stats_cache=source.snp_stats,
                compute_pve=(first_chr_pve is None),
            )
            chr_pve, chr_pve_se = chr_result.pve, chr_result.pve_se

            if writer is None:
                all_results.extend(chr_result.associations)

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
            pve_str = f", pve={first_chr_pve:.6f}" if first_chr_pve is not None else ""
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


class _LocoChrSource:
    """One chromosome's .bed columns as a GenotypeSource.

    LOCO captures original BED row positions for its pre-filtered phenotype
    view. ``prepare`` maps the runner's run-local sample basis through those
    positions. Statistics stay float64 BED reads on the analyzed-sample basis
    GEMMA uses, or reuse the kinship PASS-1 cache when the physical basis is
    exactly every original sample.
    """

    def __init__(
        self,
        bed_path: Path,
        chr_snp_indices: np.ndarray,
        valid_indices: np.ndarray,
        *,
        snp_meta: SnpMeta,
        col_chunk_size: int,
        snp_stats_cache: SnpStatsCache | None,
    ) -> None:
        if len(chr_snp_indices) > 0 and (
            chr_snp_indices[0] < 0 or chr_snp_indices[-1] >= len(snp_meta)
        ):
            raise ValueError("chromosome SNP identities fall outside paired SnpMeta")
        self._bed_path = bed_path
        self._chr_snp_indices = chr_snp_indices
        self._valid_indices = valid_indices
        self._snp_meta = snp_meta
        self._col_chunk_size = col_chunk_size
        self._snp_stats_cache = snp_stats_cache

    @property
    def n_snps(self) -> int:
        return len(self._chr_snp_indices)

    def prepare(
        self, samples: SampleBasis, filters: SnpFilterSpec
    ) -> PreparedGenotypes:
        if samples.source_row_count != len(self._valid_indices):
            raise ValueError(
                "sample basis row count must match LOCO run-local rows: "
                f"got {samples.source_row_count} and {len(self._valid_indices)}"
            )
        if filters.hwe_threshold > 0:
            # PipelineRunner rejects -hwe with -loco before this runs
            # (pipeline.py); a direct caller reaching here would silently
            # get unfiltered results.
            raise ValueError("HWE filtering is not supported in LOCO")
        physical_rows = self._valid_indices[samples.positions]
        cache = self._snp_stats_cache
        all_physical_samples = bool(
            cache is not None
            and len(physical_rows) == cache.n_samples
            and np.array_equal(physical_rows, np.arange(cache.n_samples))
        )
        stats = _chr_snp_stats_for_loco(
            self._snp_stats_cache,
            self._bed_path,
            self._chr_snp_indices,
            physical_rows,
            all_samples_valid=all_physical_samples,
            col_chunk_size=self._col_chunk_size,
        )

        def _iter_chunks(
            selection: SnpSelection, chunk_size: int
        ) -> Iterator[RawLmmChunk]:
            selected_columns = selection.indices
            n_filtered = len(selected_columns)
            # Keep one BED handle for the stream instead of re-reading BIM
            # metadata for every chunk.
            with open_bed(Path(f"{self._bed_path}.bed")) as bed:
                for chunk_start in range(0, n_filtered, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_filtered)
                    geno_chunk = bed.read(
                        index=np.s_[
                            physical_rows,
                            selected_columns[chunk_start:chunk_end],
                        ],
                        dtype=np.float64,
                    )
                    yield RawLmmChunk(
                        np.ascontiguousarray(geno_chunk), chunk_start, chunk_end
                    )

        return bind_prepared_genotypes(
            snp_meta=self._snp_meta,
            stats=stats,
            filters=filters,
            chunk_source=_iter_chunks,
        )


def _run_lmm_for_chromosome_numpy(
    *,
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    snp_meta: SnpMeta,
    valid_mask: np.ndarray,
    config: LmmConfig,
    col_chunk_size: int,
    chr_name: str,
    snps_indices: np.ndarray | None = None,
    writer: IncrementalAssocWriter | None = None,
    snp_stats_cache: SnpStatsCache | None = None,
    compute_pve: bool = False,
) -> LmmRunResult:
    """Run the shared NumPy LMM body on a single chromosome's SNPs.

    Builds a per-chromosome genotype source over this chromosome's columns
    and hands it to ``_run_numpy_lmm`` with the LOCO eigenpairs. The run is
    silenced (``show_progress=False``): the chromosome loop owns progress
    output, and a per-chromosome banner would repeat 20 times.

    Args:
        bed_path: PLINK file prefix.
        chr_snp_indices: Global column indices for this chromosome's SNPs.
        eigenvalues: Eigenvalues from LOCO kinship eigendecomp.
        eigenvectors: Eigenvectors from LOCO kinship eigendecomp.
        phenotypes: Phenotype vector (n_valid_samples,), already filtered.
        covariates: Covariate matrix (n_valid_samples, n_cvt) or None.
        snp_meta: Full SNP metadata columns (indexed by global SNP index).
        valid_mask: Boolean mask for valid samples (for disk row reads).
        config: Whatever run_lmm_loco was given, unchanged.
        col_chunk_size: Cap on SNP columns per disk read chunk.
        chr_name: Chromosome label, used in progress output.
        snps_indices: Global -snps restriction indices, or None.
        writer: Shared incremental writer, or None to collect in memory.
        snp_stats_cache: Global SNP statistics from kinship PASS 1.
        compute_pve: Whether to estimate PVE from this chromosome's null
            model (True only until one chromosome succeeds).

    Returns:
        LmmRunResult for this chromosome. ``associations`` is empty when
        ``writer`` is used; ``pve``/``pve_se`` are None unless computed.
    """
    source = _LocoChrSource(
        bed_path,
        chr_snp_indices,
        np.where(valid_mask)[0],
        snp_meta=snp_meta,
        col_chunk_size=col_chunk_size,
        snp_stats_cache=snp_stats_cache,
    )
    n_cvt = covariates.shape[1] if covariates is not None else 1
    execution = plan_association(
        phenotypes.shape[0],
        len(chr_snp_indices),
        requested="numpy",
        n_cvt=n_cvt,
        lmm_mode=config.lmm_mode,
        mem_budget=config.mem_budget,
        max_chunk_size=col_chunk_size,
        log_dispatch_choices=True,
    )
    return _run_numpy_lmm(
        source,
        phenotypes=phenotypes,
        kinship=None,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        config=replace(config, show_progress=False),
        output_path=None,
        writer=writer,
        snps_indices=snps_indices,
        execution=execution,
        compute_pve=compute_pve,
        banner="NumPy LOCO",
        label="lmm_loco",
        progress_label=f"LOCO chr {chr_name} association",
        lambda_warning_prefix="LOCO ",
    )
