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
from dataclasses import replace
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core import memory
from jamma.core.snp_filter import validate_snp_indices
from jamma.core.snp_stats import SnpFilterSpec, SnpStats
from jamma.core.threading import get_loco_worker_count, get_physical_core_count
from jamma.io.plink import get_plink_metadata, partitions_from_metadata
from jamma.lmm.assoc_output import AssocResult, IncrementalAssocWriter
from jamma.lmm.association_plan import KinshipShape, plan_association
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
from jamma.lmm.runner_numpy_streaming import bed_chunk_source
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
                    source.snp_stats.take(chr_snp_indices),
                    n_samples_total,
                    snp_meta=snp_info,
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

    The sample basis indexes BED rows directly. ``stats`` covers exactly
    this chromosome's SNPs over the analysed rows, the basis GEMMA uses; its
    global indices name the BED columns.
    """

    def __init__(
        self,
        bed_path: Path,
        stats: SnpStats,
        n_samples: int,
        *,
        snp_meta: SnpMeta,
    ) -> None:
        self._bed_path = bed_path
        self._stats = stats
        self._n_samples = n_samples
        self._snp_meta = snp_meta

    @property
    def n_snps(self) -> int:
        return self._stats.n_snps

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
        return bind_prepared_genotypes(
            snp_meta=self._snp_meta,
            stats=self._stats,
            filters=filters,
            sample_basis=samples,
            chunk_source=bed_chunk_source(self._bed_path, samples),
        )
