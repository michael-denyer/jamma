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
from dataclasses import dataclass
from pathlib import Path

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
    validate_genotype_values,
)
from jamma.jlinalg import compute_snp_stats_chunk
from jamma.kinship import (
    SnpStatsCache,
    compute_loco_kinship_streaming,
    write_kinship_matrix,
)
from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_cache import (
    compute_eigen_cache_key,
    eigen_cache_is_valid,
    write_eigen_cache_manifest,
)
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    compute_and_log_pve,
)
from jamma.lmm.results import (
    _build_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.runner_numpy import compute_chunk_size_numpy
from jamma.lmm.schema import RESULT_FIELDS, TEST_TYPE_MAP, LazySnpMeta, LocoResult
from jamma.lmm.stats import AssocResult
from jamma.utils import chr_sort_key


def _collect_chr_snp_stats(
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    valid_indices: np.ndarray,
    col_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect per-SNP statistics for one chromosome via chunked BED reads.

    Shared by LOCO chromosome runners (pass-1 logic).

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
    miss_counts = np.zeros(n_chr_snps, dtype=np.intp)
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

            geno_chunk = np.ascontiguousarray(geno_chunk)
            compute_snp_stats_chunk(
                geno_chunk,
                col_means[chunk_start:chunk_end],
                miss_counts[chunk_start:chunk_end],
                col_vars[chunk_start:chunk_end],
            )

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

    Shared by LOCO chromosome runners.

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
    write_eigen: bool = False,
    eigen_dir: Path | None = None,
    eigen_prefix: str = "result",
    legacy_text: bool = False,
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
        write_eigen: If True, write per-chromosome eigen files after
            eigendecomp. Raises ValueError if eigen_dir is None.
        eigen_dir: Directory for reading/writing per-chromosome eigen cache.
            When set, checks for cached files before computing. Combined
            with write_eigen, writes new files here.
        eigen_prefix: Prefix for eigen filenames (default "result").
        legacy_text: If True, write (and look up) per-chromosome kinship and
            eigen artifacts as GEMMA-compatible text files (.cXX.txt,
            .eigenD.txt, .eigenU.txt) instead of binary .npy.
    Returns:
        LocoResult with associations in biological chromosome order
        (1-22, X, Y, XY, MT). Associations list is empty if output_path
        is set (results written to disk).

    Raises:
        ValueError: If fewer than two chromosomes are present, if lmm_mode is
            not in {1, 2, 3, 4}, if no samples have valid phenotypes, or if
            write_eigen=True but eigen_dir is None.
    """
    start_time = time.perf_counter()

    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    if write_eigen and eigen_dir is None:
        raise ValueError(
            "write_eigen=True requires eigen_dir to be set. "
            "Pass eigen_dir or use --eigen-dir on the CLI."
        )

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
        eigen_cache_components: dict | None = None
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
            eigen_cache = _find_loco_eigen_cache(
                eigen_dir, eigen_prefix, unique_chrs, legacy_text=legacy_text
            )
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

        # Initialise to None; reassigned inside the compute block when
        # eigen_cache is None and we actually stream kinship.
        snp_stats_cache = None
        loco_iter = None
        # When save_kinship=False and some samples are invalid, pass
        # valid_indices so kinship is accumulated at n_valid x n_valid size,
        # avoiding full n_samples^2 materialisation for post-hoc subsetting.
        kinship_valid_indices = (
            None if all_samples_valid or save_kinship else np.where(valid_mask)[0]
        )

        if eigen_cache is None:
            # Stream LOCO kinship matrices one at a time (pure NumPy), reusing
            # the shared kinship streamer and its PASS-1 SNP statistics.
            loco_iter, snp_stats_cache = compute_loco_kinship_streaming(
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

        # Iterate: either from cached eigen files or kinship stream.
        if eigen_cache is not None:
            chr_iterator = ((chr_name, None) for chr_name in unique_chrs)
        else:
            if loco_iter is None:
                raise RuntimeError(
                    "LOCO kinship iterator was not initialized. "
                    "Expected streaming kinship computation when eigen_cache "
                    "is None, but loco_iter is still None. This is an internal "
                    "error — please report it."
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
                    kinship_suffix = ".txt" if legacy_text else ".npy"
                    kinship_stem = f"{kinship_output_prefix}.loco.cXX.chr{chr_name}"
                    kinship_path = (
                        kinship_output_dir / f"{kinship_stem}{kinship_suffix}"
                    )
                    try:
                        actual_path = write_kinship_matrix(
                            K_loco, kinship_path, legacy_text=legacy_text
                        )
                    except OSError as e:
                        raise OSError(
                            f"Failed to save LOCO kinship for chromosome "
                            f"{chr_name} to {kinship_path}: {e}"
                        ) from e
                    if show_progress:
                        logger.info(f"  Saved LOCO kinship to {actual_path}")

                # K_loco is already n_valid x n_valid from early subsetting
                # (the numpy backend passes valid_indices to the
                # kinship streamer) — skip post-hoc np.ix_ copy.
                if kinship_valid_indices is not None:
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
                        legacy_text=legacy_text,
                    )
                except OSError as e:
                    raise OSError(
                        f"Failed to write LOCO eigen for chromosome "
                        f"{chr_name} to {eigen_dir}: {e}"
                    ) from e
                logger.info(f"  Wrote LOCO eigen for chr {chr_name}")

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


@dataclass
class _LocoChrContext:
    """Shared context from PASS 1 + covariate/rotation setup for a single chromosome.

    Holds the results of the common prefix shared by the NumPy chromosome
    runner: SNP statistics collection, filtering, covariate matrix
    construction, eigenrotation, and optional PVE computation.
    """

    global_filtered_indices: np.ndarray
    filtered_afs: np.ndarray
    filtered_miss: np.ndarray
    filtered_means_all: np.ndarray
    n_filtered: int
    valid_indices: np.ndarray
    W: np.ndarray
    n_cvt: int
    UtW: np.ndarray
    Uty: np.ndarray
    rotation_threads: int
    chr_pve: float | None
    chr_pve_se: float | None


def _loco_chr_common(
    bed_path: Path,
    chr_snp_indices: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    maf_threshold: float,
    miss_threshold: float,
    valid_mask: np.ndarray,
    show_progress: bool,
    l_min: float,
    l_max: float,
    snps_global_mask: np.ndarray | None,
    col_chunk_size: int,
    compute_pve: bool,
    snp_stats_cache: SnpStatsCache | None = None,
) -> _LocoChrContext | None:
    """Run PASS 1 (SNP stats + filtering) and covariate/rotation setup.

    Extracts the shared prefix for the NumPy chromosome runner.
    Returns None if all SNPs are filtered out (caller should return early).

    Args:
        bed_path: PLINK file prefix.
        chr_snp_indices: Column indices for this chromosome's SNPs.
        eigenvalues: Eigenvalues from LOCO kinship eigendecomp.
        eigenvectors: Eigenvectors from LOCO kinship eigendecomp.
        phenotypes: Phenotype vector (n_valid_samples,).
        covariates: Covariate matrix or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        valid_mask: Boolean mask for valid samples.
        show_progress: Whether to log progress.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        snps_global_mask: Boolean mask over all SNPs or None.
        col_chunk_size: Number of SNP columns per disk read chunk.
        compute_pve: If True, compute PVE from null model REML lambda.
        snp_stats_cache: Optional cached global SNP statistics (NumPy path).
            When provided, per-chromosome stats are sliced from the cache
            instead of re-reading the BED file.

    Returns:
        _LocoChrContext with shared setup results, or None if no SNPs pass filters.
    """
    n_samples = phenotypes.shape[0]
    valid_indices = np.where(valid_mask)[0]

    # === PASS 1: Chunked SNP statistics + filtering ===
    if snp_stats_cache is not None:
        # Use cached global stats, sliced to this chromosome.
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
        return None

    (
        _local_filtered_indices,
        global_filtered_indices,
        filtered_afs,
        filtered_miss,
        filtered_means_all,
    ) = filter_result
    n_filtered = len(global_filtered_indices)

    # === Covariate matrix + eigenrotation ===
    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Rotation is pure BLAS — use all physical cores.
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = eigenvectors.T @ W
        Uty = eigenvectors.T @ phenotypes

    # === PVE computation (optional) ===
    chr_pve = None
    chr_pve_se = None
    if compute_pve:
        chr_pve, chr_pve_se = compute_and_log_pve(
            eigenvalues, UtW, Uty, n_cvt, l_min, l_max
        )

    return _LocoChrContext(
        global_filtered_indices=global_filtered_indices,
        filtered_afs=filtered_afs,
        filtered_miss=filtered_miss,
        filtered_means_all=filtered_means_all,
        n_filtered=n_filtered,
        valid_indices=valid_indices,
        W=W,
        n_cvt=n_cvt,
        UtW=UtW,
        Uty=Uty,
        rotation_threads=rotation_threads,
        chr_pve=chr_pve,
        chr_pve_se=chr_pve_se,
    )


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

    Pure-NumPy implementation. Reads the chromosome's SNPs from the BED
    file in column chunks (two-pass: statistics, then association).

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
        snp_stats_cache: Global SNP statistics from kinship PASS 1.
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
    # === Shared PASS 1 + covariate/rotation setup ===
    ctx = _loco_chr_common(
        bed_path=bed_path,
        chr_snp_indices=chr_snp_indices,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        phenotypes=phenotypes,
        covariates=covariates,
        maf_threshold=maf_threshold,
        miss_threshold=miss_threshold,
        valid_mask=valid_mask,
        show_progress=show_progress,
        l_min=l_min,
        l_max=l_max,
        snps_global_mask=snps_global_mask,
        col_chunk_size=col_chunk_size,
        compute_pve=compute_pve,
        snp_stats_cache=snp_stats_cache,
    )
    if ctx is None:
        return [], None, None

    n_samples = phenotypes.shape[0]

    # === PASS 2: Chunked NumPy association ===
    # Compute null model (NumPy version, returns plain numpy arrays)
    logl_H0, _lambda_null_mle, Hi_eval_null = _compute_null_model_common(
        lmm_mode,
        eigenvalues,
        ctx.UtW,
        ctx.Uty,
        ctx.n_cvt,
        show_progress=False,
        l_min=l_min,
        l_max=l_max,
    )

    # Compute chunk size based on RAM budget
    chunk_size = compute_chunk_size_numpy(n_samples, ctx.n_filtered, ctx.n_cvt)

    # Pre-allocate result arrays
    write_offset = 0
    arrays_out: dict[str, np.ndarray] = {
        key: np.empty(ctx.n_filtered, dtype=np.float64)
        for key in RESULT_FIELDS[lmm_mode]
    }
    results: list[AssocResult] = []

    bed_file = Path(f"{bed_path}.bed")
    with open_bed(bed_file) as bed:
        for disk_start in range(0, ctx.n_filtered, col_chunk_size):
            disk_end = min(disk_start + col_chunk_size, ctx.n_filtered)
            disk_col_indices = ctx.global_filtered_indices[disk_start:disk_end]

            geno_disk_chunk = bed.read(
                index=np.s_[ctx.valid_indices, disk_col_indices],
                dtype=np.float64,
            )

            # Impute missing values with column means
            chunk_filtered_means = ctx.filtered_means_all[disk_start:disk_end]
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
                with blas_threads(ctx.rotation_threads):
                    UtG = eigenvectors.T @ geno_sub

                # Compute Uab batch
                Uab_batch = batch_compute_uab_numpy(ctx.n_cvt, ctx.UtW, ctx.Uty, UtG)

                # Mode dispatch
                try:
                    cr = compute_lmm_chunk_numpy(
                        lmm_mode,
                        ctx.n_cvt,
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
                        f"n_samples={n_samples}, n_cvt={ctx.n_cvt}: {e}"
                    )
                    raise

                # Write sub-chunk results to pre-allocated arrays
                actual_len = sub_end - sub_start
                s = slice(write_offset, write_offset + actual_len)
                for key in arrays_out:
                    arrays_out[key][s] = cr[key][:actual_len]
                write_offset += actual_len

            del geno_disk_chunk

    if write_offset != ctx.n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {write_offset} results, "
            f"expected {ctx.n_filtered}. This is an internal error."
        )

    # Count lambda boundary hits and log warnings
    n_lmin, n_lmax = count_lambda_boundary_hits(lmm_mode, arrays_out, l_min, l_max)
    log_lambda_boundary_warning(n_lmin, n_lmax, l_min, l_max, prefix="LOCO ")

    # Flush results
    if writer is not None:
        writer.write_arrays_batch(
            lmm_mode,
            ctx.global_filtered_indices,
            snp_info,
            ctx.filtered_afs,
            ctx.filtered_miss,
            arrays_out,
        )
    else:
        results = _build_results(
            lmm_mode,
            ctx.global_filtered_indices,
            ctx.filtered_afs,
            ctx.filtered_miss,
            snp_info,
            arrays_out,
        )

    return results, ctx.chr_pve, ctx.chr_pve_se
