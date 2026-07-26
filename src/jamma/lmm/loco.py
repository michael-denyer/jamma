"""LOCO LMM orchestrator.

Runs leave-one-chromosome-out LMM association by looping over chromosomes:
for each chromosome c, eigendecompose K_loco_c, run LMM on chromosome c's
SNPs using that eigendecomposition, discard K_loco_c.

Memory profile (sequential processing):
    At any point holds S_full (n^2*8) from the LOCO kinship generator,
    plus one K_loco (n^2*8) during eigendecomp, plus LMM working set.
    Each K_loco is discarded after eigendecomp.

``LocoConfig`` lives in ``loco_config`` and the eigenpair sources in
``loco_eigen``; both are re-exported here, so ``from jamma.lmm.loco import
LocoConfig`` keeps working.
"""

from __future__ import annotations

import contextlib
import gc
import time
from pathlib import Path
from typing import cast

import numpy as np
from bed_reader import open_bed
from loguru import logger

from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpStats,
    collect_snp_stats_from_chunks,
    filter_snp_stats,
)
from jamma.core.threading import (
    blas_threads,
    get_loco_worker_count,
    get_physical_core_count,
)
from jamma.io.plink import (
    get_plink_metadata,
    partitions_from_metadata,
)
from jamma.kinship import (
    SnpStatsCache,
    compute_loco_kinship_streaming,
)
from jamma.lmm.chunk_runner_numpy import RawLmmChunk, run_lmm_chunk_source_numpy
from jamma.lmm.compute_numpy import LmmMode
from jamma.lmm.eigen_cache import (
    EigenCacheComponents,
    compute_eigen_cache_key,
    eigen_cache_is_valid,
    invalidate_eigen_cache_manifest,
    write_eigen_cache_manifest,
)
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.loco_config import DEFAULT_LOCO_CONFIG, LocoConfig
from jamma.lmm.loco_eigen import (
    _cached_eigen_pairs,
    _computed_eigen_pairs,
    _find_loco_eigen_cache,
)
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    compute_and_log_pve,
)
from jamma.lmm.results import make_result_list_sink, make_writer_sink
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    TEST_TYPE_MAP,
    LazySnpMeta,
    LmmConfig,
    LocoResult,
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


def _warn_if_loco_stats_invalid(stats: SnpStats) -> None:
    if stats.n_unexpected > 0:
        logger.warning(
            f"LOCO chr genotype validation: {stats.n_unexpected} values outside "
            f"expected range {{0, 1, 2, NaN}}"
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
    # Unpacked to locals the way run_lmm_association_numpy does, so the body
    # below reads the same as it did when these were 23 flat parameters.
    maf_threshold = config.maf_threshold
    miss_threshold = config.miss_threshold
    lmm_mode = config.lmm_mode
    check_memory = config.check_memory
    show_progress = config.show_progress

    # Artifact naming (prefixes, .txt vs .npy) is not unpacked: LocoConfig owns
    # it via kinship_path()/eigen_paths()/eigen_stem(), so there is one
    # definition of each filename rather than one per helper that builds it.
    save_kinship = loco.save_kinship
    snps_indices = loco.snps_indices
    ksnps_indices = loco.ksnps_indices
    col_chunk_size = loco.col_chunk_size
    write_eigen = loco.write_eigen
    eigen_dir = loco.eigen_dir
    eigen_prefix = loco.eigen_prefix

    start_time = time.perf_counter()

    # No lmm_mode or write_eigen/eigen_dir guard here: LmmConfig and LocoConfig
    # reject both at construction, as the other runners have relied on since
    # 6.0.0. Re-checking would be dead code with a second, divergable message.

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
    n_samples_total = meta["n_samples"]
    n_snps_total = meta["n_snps"]

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

    # Reused below for the kinship subsetting decisions (kinship_valid_indices
    # sizing and the K_loco valid-sample slice). The per-chromosome SNP-stats path
    # recomputes its own valid_mask.all() in _loco_chr_common, so this is not a
    # shared value across the chromosome loop.
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

    test_type = TEST_TYPE_MAP[lmm_mode]

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

        # Precompute global SNP membership mask for -snps restriction.
        # Avoids per-chromosome np.isin on every iteration.
        if snps_indices is not None:
            snps_global_mask: np.ndarray | None = np.zeros(n_snps_total, dtype=bool)
            snps_global_mask[snps_indices] = True
        else:
            snps_global_mask = None

        # Content + parameter key over every determinant of the eigendecomposition
        # (genotype files, filter thresholds, -ksnps set, analysed-sample mask).
        # Computed once and reused on both the read (validate) and write (persist)
        # paths below. Those paths are mutually exclusive at runtime but key off
        # the same inputs.
        eigen_cache_key: str | None = None
        eigen_cache_components: EigenCacheComponents | None = None
        if eigen_dir is not None:
            eigen_cache_key, eigen_cache_components = compute_eigen_cache_key(
                bed_path,
                maf_threshold=maf_threshold,
                miss_threshold=miss_threshold,
                valid_mask=valid_mask,
                ksnps_indices=ksnps_indices,
            )

        # Check for cached eigen files before computing kinship.
        # When write_eigen is True the user explicitly asked to
        # (re)generate files, so skip the cache and recompute.
        eigen_cache: dict[str, tuple[Path, Path]] | None = None
        if eigen_dir is not None and not write_eigen:
            eigen_cache = _find_loco_eigen_cache(loco, unique_chrs)
            if eigen_cache is not None:
                # eigen_cache_key is set whenever eigen_dir is not None.
                assert eigen_cache_key is not None
                ok, reason = eigen_cache_is_valid(
                    eigen_dir, eigen_prefix, eigen_cache_key
                )
                if not ok:
                    logger.warning(
                        f"LOCO eigen cache in {eigen_dir} is stale or unverifiable "
                        f"({reason}). Kinship and eigendecomposition will be "
                        f"recomputed."
                    )
                    eigen_cache = None
            if eigen_cache is not None:
                logger.info(
                    f"Found complete LOCO eigen cache in {eigen_dir} "
                    f"({len(eigen_cache)} chromosomes). "
                    f"Skipping kinship computation and eigendecomp."
                )
                if save_kinship:
                    logger.warning(
                        "save_kinship ignored when using cached eigen "
                        "files (kinship is not computed)"
                    )
                logger.warning(
                    "Using cached eigen: SNP filtering will use "
                    "valid-sample-only statistics (not all-sample stats "
                    "from kinship pass). This may produce slightly "
                    "different SNP filter sets compared to the original "
                    "compute run."
                )

        # Where eigenpairs come from is settled once, here, rather than being
        # re-tested at every step of the chromosome loop.
        snp_stats_cache = None
        if eigen_cache is not None:
            eigen_pairs = _cached_eigen_pairs(
                eigen_cache,
                unique_chrs,
                n_valid=n_valid,
                partitions=partitions,
                show_progress=show_progress,
            )
        else:
            # When save_kinship=False and some samples are invalid, pass
            # valid_indices so kinship is accumulated at n_valid x n_valid size,
            # avoiding full n_samples^2 materialisation for post-hoc subsetting.
            kinship_valid_indices = (
                None if all_samples_valid or save_kinship else np.where(valid_mask)[0]
            )
            # Stream LOCO kinship matrices one at a time (pure NumPy), reusing
            # the shared kinship streamer and its PASS-1 SNP statistics.
            loco_stream, snp_stats_cache = compute_loco_kinship_streaming(
                bed_path,
                maf_threshold=maf_threshold,
                miss_threshold=miss_threshold,
                check_memory=check_memory,
                show_progress=show_progress,
                ksnps_indices=ksnps_indices,
                valid_indices=kinship_valid_indices,
                _copy_yielded_matrices=False,
                return_snp_stats=True,
            )

            # Create eigen output directory before the loop (once, not per-chr).
            if write_eigen:
                # write_eigen guarantees eigen_dir (entry guard at top of function).
                assert eigen_dir is not None
                try:
                    eigen_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise OSError(
                        f"Cannot create eigen cache directory {eigen_dir}: {e}"
                    ) from e
                # Invalidate any stale manifest before rewriting eigen files. The fresh
                # manifest is written only after the loop completes, so an interrupted
                # rewrite leaves no manifest and the next read recomputes rather than
                # trusting a half-rewritten cache.
                invalidate_eigen_cache_manifest(eigen_dir, eigen_prefix)

            eigen_pairs = _computed_eigen_pairs(
                loco_stream,
                unique_chrs,
                valid_mask=valid_mask,
                n_valid=n_valid,
                pre_subset=kinship_valid_indices is not None,
                all_samples_valid=all_samples_valid,
                partitions=partitions,
                check_memory=check_memory,
                show_progress=show_progress,
                loco=loco,
            )

        first_chr_pve: float | None = None
        first_chr_pve_se: float | None = None

        for chr_idx, (chr_name, eigenvalues_np, U) in enumerate(eigen_pairs):
            chr_snp_indices = partitions[chr_name]
            logger.debug(
                f"  chr {chr_name}: numpy backend, {len(chr_snp_indices)} SNPs"
            )

            chr_results, chr_pve, chr_pve_se = _run_lmm_for_chromosome_numpy(
                bed_path=bed_path,
                chr_snp_indices=chr_snp_indices,
                eigenvalues=eigenvalues_np,
                eigenvectors=U,
                phenotypes=phenotypes_valid,
                covariates=covariates_valid,
                snp_info=snp_info,
                valid_mask=valid_mask,
                config=config,
                snps_global_mask=snps_global_mask,
                col_chunk_size=col_chunk_size,
                writer=writer,
                chr_name=chr_name,
                snp_stats_cache=snp_stats_cache,
                compute_pve=(first_chr_pve is None),
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

        if write_eigen and eigen_cache is None:
            # write_eigen guarantees eigen_dir (entry guard), and the key block
            # above ran because eigen_dir is not None, so all three are set.
            assert eigen_dir is not None
            assert eigen_cache_key is not None
            assert eigen_cache_components is not None
            write_eigen_cache_manifest(
                eigen_dir,
                eigen_prefix,
                eigen_cache_key,
                components=eigen_cache_components,
            )
            logger.info(f"Wrote LOCO eigen cache manifest to {eigen_dir}")

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


def _run_lmm_for_chromosome_numpy(
    *,
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    snp_info: LazySnpMeta | list,
    valid_mask: np.ndarray,
    config: LmmConfig,
    col_chunk_size: int,
    chr_name: str,
    snps_global_mask: np.ndarray | None = None,
    writer: IncrementalAssocWriter | None = None,
    snp_stats_cache: SnpStatsCache | None = None,
    compute_pve: bool = False,
) -> tuple[list[AssocResult], float | None, float | None]:
    """Run NumPy LMM association on a single chromosome's SNPs.

    Two passes over the chromosome's columns: statistics for filtering, then
    association through the shared chunk engine. The full chromosome genotype
    matrix is never materialised.

    Args:
        bed_path: PLINK file prefix.
        chr_snp_indices: Column indices for this chromosome's SNPs.
        eigenvalues: Eigenvalues from LOCO kinship eigendecomp.
        eigenvectors: Eigenvectors from LOCO kinship eigendecomp.
        phenotypes: Phenotype vector (n_valid_samples,), already filtered.
        covariates: Covariate matrix (n_valid_samples, n_cvt) or None.
        snp_info: Full SNP metadata view (indexed by global SNP index).
        valid_mask: Boolean mask for valid samples (for genotype subsetting).
        config: Filter thresholds, test type, lambda bounds and grid. Whatever
            run_lmm_loco was given, unchanged.
        col_chunk_size: Number of SNP columns per disk read chunk.
        chr_name: Chromosome label, used in progress output.
        snps_global_mask: Boolean mask over all SNPs (True = included by -snps),
            or None. Pre-indexed, so no per-chromosome np.isin is needed.
        writer: Optional incremental writer. When provided, results stream to
            disk and an empty list is returned.
        snp_stats_cache: Global SNP statistics from kinship PASS 1. Sliced per
            chromosome when every sample is analysed, avoiding a BED re-read.
        compute_pve: If True, compute PVE from the null model REML lambda.

    Returns:
        Tuple of (results, pve, pve_se). ``results`` is empty when ``writer``
        is used; ``pve``/``pve_se`` are None unless ``compute_pve`` is set and
        the likelihood surface is not flat.
    """
    maf_threshold = config.maf_threshold
    miss_threshold = config.miss_threshold
    lmm_mode = config.lmm_mode
    show_progress = config.show_progress
    l_min = config.l_min
    l_max = config.l_max
    n_grid = config.n_grid
    n_refine = config.n_refine

    n_samples = phenotypes.shape[0]
    valid_indices = np.where(valid_mask)[0]

    # === PASS 1: Chunked SNP statistics + filtering ===
    # Reuse the all-sample kinship-PASS-1 cache only when every sample is analysed;
    # otherwise recompute over the analysed samples to match GEMMA (which computes
    # SNP mean/MAF and imputes missing genotypes over analysed individuals only).
    chr_stats = _chr_snp_stats_for_loco(
        snp_stats_cache,
        bed_path,
        chr_snp_indices,
        valid_indices,
        all_samples_valid=bool(valid_mask.all()),
        col_chunk_size=col_chunk_size,
    )
    _warn_if_loco_stats_invalid(chr_stats)

    snp_selection = filter_snp_stats(
        chr_stats,
        SnpFilterSpec(
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            restrict_global_mask=snps_global_mask,
        ),
    )
    n_filtered = len(snp_selection.indices)

    if show_progress:
        logger.debug(
            f"  Chromosome SNPs: {len(chr_snp_indices)}, after filter: {n_filtered}"
        )

    if n_filtered == 0:
        logger.warning(
            f"  Chromosome ({len(chr_snp_indices)} SNPs) has no SNPs after "
            f"filtering, skipping"
        )
        return [], None, None

    global_filtered_indices = snp_selection.indices

    # === Covariate matrix + eigenrotation ===
    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Rotation is pure BLAS — use all physical cores.
    with blas_threads(get_physical_core_count()):
        UtW = eigenvectors.T @ W
        Uty = eigenvectors.T @ phenotypes

    chr_pve: float | None = None
    chr_pve_se: float | None = None
    if compute_pve:
        chr_pve, chr_pve_se = compute_and_log_pve(
            eigenvalues, UtW, Uty, n_cvt, l_min, l_max
        )

    # === PASS 2: Chunked NumPy association ===
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

    results: list[AssocResult] = []
    with open_bed(Path(f"{bed_path}.bed")) as bed:

        def _make_loco_source(source_chunk_size: int):
            chunk_offsets = iter(range(0, n_filtered, source_chunk_size))

            def _next_chunk() -> RawLmmChunk | None:
                try:
                    chunk_start = next(chunk_offsets)
                except StopIteration:
                    return None

                chunk_end = min(chunk_start + source_chunk_size, n_filtered)
                disk_col_indices = global_filtered_indices[chunk_start:chunk_end]
                geno_chunk = bed.read(
                    index=np.s_[valid_indices, disk_col_indices],
                    dtype=np.float64,
                )
                return RawLmmChunk(
                    np.ascontiguousarray(geno_chunk), chunk_start, chunk_end
                )

            return _next_chunk

        sink_args = (
            lmm_mode,
            snp_info,
            global_filtered_indices,
            snp_selection.filtered_afs,
            snp_selection.filtered_miss,
        )
        _sink = (
            make_writer_sink(writer, *sink_args)
            if writer is not None
            else make_result_list_sink(results, *sink_args)
        )

        run_lmm_chunk_source_numpy(
            raw_chunk_source_factory=_make_loco_source,
            chunk_sink=_sink,
            U=eigenvectors,
            eigenvalues_np=eigenvalues,
            UtW=UtW,
            Uty=Uty,
            Hi_eval_null=Hi_eval_null,
            logl_H0=logl_H0,
            n_samples=n_samples,
            n_filtered=n_filtered,
            n_cvt=n_cvt,
            lmm_mode=cast(LmmMode, lmm_mode),
            filtered_means=snp_selection.filtered_means,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
            requested_chunk_size=col_chunk_size,
            auto_scale_chunk_size=True,
            show_progress=False,
            progress_label=f"LOCO chr {chr_name} association",
            lambda_warning_prefix="LOCO ",
        )

    return results, chr_pve, chr_pve_se
