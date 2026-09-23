"""LOCO LMM orchestrator.

Runs leave-one-chromosome-out LMM association by looping over chromosomes:
for each chromosome c, eigendecompose K_loco_c, run LMM on chromosome c's
SNPs using that eigendecomposition, discard K_loco_c.

Memory profile (sequential processing):
    At any point holds S_full (n^2*8) from the LOCO kinship generator,
    plus one K_loco (n^2*8) during eigendecomp, plus LMM working set.
    Each K_loco is discarded after eigendecomp.

With ``JAMMA_LOCO_WORKERS`` above one, ``plan_loco_workers`` lets that many
chromosomes eigendecompose at once, each on its own copy of K_loco, while
the association pass stays sequential and in chromosome order.

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

from jamma.core import memory
from jamma.core.snp_filter import validate_snp_indices
from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpSelection,
    SnpStats,
    collect_snp_stats_from_chunks,
)
from jamma.core.threading import get_loco_worker_count, get_physical_core_count
from jamma.io.plink import get_plink_metadata, partitions_from_metadata
from jamma.kinship import SnpStatsCache
from jamma.lmm.assoc_output import AssocResult, IncrementalAssocWriter
from jamma.lmm.association_plan import KinshipShape, plan_association
from jamma.lmm.chunk_runner_numpy import RawLmmChunk
from jamma.lmm.genotype_source import (
    PreparedGenotypes,
    SampleBasis,
    bind_prepared_genotypes,
)
from jamma.lmm.loco_config import DEFAULT_LOCO_CONFIG, LocoConfig, LocoRun
from jamma.lmm.loco_eigen import (
    eigen_pairs_for,
    loco_retained_set_for,
    plan_loco_eigen_driver,
)
from jamma.lmm.loco_workers import plan_loco_workers
from jamma.lmm.prepare_common import AnalysedPhenotype, EigenPairs
from jamma.lmm.runner_numpy import (
    LOCO_LABELS,
    LmmRunSpec,
    run_single,
)
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    MODE_SPECS,
    LmmConfig,
    LmmRunResult,
    SnpMeta,
)
from jamma.utils import chr_sort_key

__all__ = [
    "DEFAULT_LOCO_CONFIG",
    "LocoConfig",
    "LocoRun",
    "run_lmm_loco",
    "run_loco",
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
) -> LmmRunResult:
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
        LmmRunResult with associations in biological chromosome order
        (1-22, X, Y, XY, MT). Associations list is empty if output_path
        is set (results written to disk).

    Raises:
        ValueError: If fewer than two chromosomes are present, or if no
            samples have valid phenotypes. Invalid lmm_mode and write_eigen
            without eigen_dir are rejected earlier, when LmmConfig and
            LocoConfig are constructed.
    """
    meta = get_plink_metadata(bed_path)
    samples = AnalysedPhenotype.from_inputs(phenotypes, covariates)
    execution = plan_association(
        samples.n_samples,
        meta.n_snps,
        config=config,
        backend="loco",
        n_cvt=samples.n_cvt,
        n_input_samples=meta.n_samples,
        max_chunk_size=loco.col_chunk_size,
    )
    execution = replace(
        execution,
        kinship=KinshipShape.resolve(
            samples.n_samples,
            meta.n_samples,
            loaded=False,
            saved=loco.kinship_output_dir is not None,
        ),
    )
    run = LocoRun(
        bed_path,
        meta,
        samples,
        config,
        loco,
        execution,
        plan_loco_eigen_driver(execution, memory.available_ram_gb()),
    )
    return run_loco(run, output_path)


def run_loco(run: LocoRun, output_path: Path | None) -> LmmRunResult:
    """Run LOCO association for a run the caller has already resolved.

    Args:
        run: The resolved run. Every chromosome shares its association plan;
            the body narrows the chunk plan to that chromosome's filtered SNP
            count.
        output_path: Path for incremental result writing, or None for in-memory.

    Returns:
        See ``run_lmm_loco``.

    Raises:
        ValueError: If fewer than two chromosomes are present.
    """
    config, loco, meta = run.config, run.loco, run.meta
    show_progress = config.show_progress
    start_time = time.perf_counter()

    n_samples_total = meta.n_samples
    n_snps_total = meta.n_snps

    validate_snp_indices(loco.snps_indices, n_snps_total)

    partitions = partitions_from_metadata(meta)
    chromosomes = {
        chr_name: partitions[chr_name]
        for chr_name in sorted(partitions, key=chr_sort_key)
    }
    unique_chrs = list(chromosomes)

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
        n_filtered_samples = n_samples_total - run.samples.n_samples
        logger.info(
            f"  Analyzed individuals: {run.samples.n_samples:,} "
            f"({n_filtered_samples} filtered)"
        )

    snp_info = SnpMeta.from_plink_meta(meta)
    execution = run.execution
    workers = plan_loco_workers(
        get_loco_worker_count(),
        n_chr=len(unique_chrs),
        cores=get_physical_core_count(),
        retained=loco_retained_set_for(execution),
        eigen_plan=run.eigen_plan,
        available_gb=memory.available_ram_gb(),
        budget_gb=config.mem_budget,
        association_gb=execution.price(eigen=None).association_gb,
    )
    # The chromosome loop owns progress output; a per-chromosome banner would
    # repeat once per chromosome.
    spec = LmmRunSpec(
        config=replace(config, show_progress=False),
        execution=execution,
        snps_indices=loco.snps_indices,
        labels=LOCO_LABELS,
    )

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
                IncrementalAssocWriter(output_path, MODE_SPECS[config.lmm_mode])
            )

        source = eigen_pairs_for(run, chromosomes, workers)

        first_chr_pve: float | None = None
        first_chr_pve_se: float | None = None

        stack.callback(source.pairs.close)
        # No enumerate() here. CPython's enumerate holds its previous result
        # tuple, and through it the previous U, until the generator yields the
        # next item, so chromosome c's eigenvectors would stay live through
        # c+1's eigendecomposition.
        for chr_name, eigenvalues_np, U in source.pairs:
            chr_snp_indices = chromosomes[chr_name]
            logger.debug(
                f"  chr {chr_name}: numpy backend, {len(chr_snp_indices)} SNPs"
            )

            chr_result = run_single(
                _LocoChrSource(
                    run.bed_path,
                    chr_snp_indices,
                    n_samples_total,
                    snp_meta=snp_info,
                    col_chunk_size=loco.col_chunk_size,
                    snp_stats_cache=source.snp_stats,
                ),
                replace(
                    spec,
                    compute_pve=first_chr_pve is None,
                    labels=replace(
                        spec.labels,
                        progress_label=f"LOCO chr {chr_name} association",
                    ),
                ),
                run.samples,
                EigenPairs(eigenvalues_np, U),
                all_results if writer is None else writer,
            )
            chr_pve, chr_pve_se = chr_result.pve, chr_result.pve_se

            if first_chr_pve is None and chr_pve is not None:
                if chr_name != unique_chrs[0]:
                    logger.info(
                        f"PVE computed from chromosome {chr_name} "
                        f"(earlier chromosomes had no SNPs to test)"
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
        return LmmRunResult(
            associations=[] if output_path is not None else all_results,
            n_tested=n_tested,
            pve=first_chr_pve,
            pve_se=first_chr_pve_se,
        )


class _LocoChrSource:
    """One chromosome's .bed columns as a GenotypeSource.

    The sample basis indexes BED rows directly. Statistics stay float64 BED
    reads on the analysed-sample basis GEMMA uses, or reuse the kinship
    PASS-1 cache when that basis is every BED row.
    """

    def __init__(
        self,
        bed_path: Path,
        chr_snp_indices: np.ndarray,
        n_samples: int,
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
        self._n_samples = n_samples
        self._snp_meta = snp_meta
        self._col_chunk_size = col_chunk_size
        self._snp_stats_cache = snp_stats_cache

    @property
    def n_snps(self) -> int:
        return len(self._chr_snp_indices)

    def prepare(
        self, samples: SampleBasis, filters: SnpFilterSpec
    ) -> PreparedGenotypes:
        if samples.source_row_count != self._n_samples:
            raise ValueError(
                "sample basis row count must match the BED rows: "
                f"got {samples.source_row_count} and {self._n_samples}"
            )
        if filters.hwe_threshold > 0:
            # PipelineRunner rejects -hwe with -loco before this runs
            # (pipeline.py); a direct caller reaching here would silently
            # get unfiltered results.
            raise ValueError("HWE filtering is not supported in LOCO")
        physical_rows = samples.positions
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
            sample_basis=samples,
            chunk_source=_iter_chunks,
        )
