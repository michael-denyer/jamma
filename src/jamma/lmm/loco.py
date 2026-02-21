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
from pathlib import Path

import jax
import numpy as np
from bed_reader import open_bed
from loguru import logger

from jamma.core.snp_filter import compute_snp_filter_mask
from jamma.core.threading import blas_threads
from jamma.io.plink import (
    get_chromosome_partitions,
    get_plink_metadata,
    validate_genotype_values,
)
from jamma.kinship import compute_loco_kinship_streaming, write_kinship_matrix
from jamma.lmm.chunk import _compute_chunk_size
from jamma.lmm.compute import _compute_lmm_chunk, block_chunk_result, strip_and_append
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.likelihood_jax import batch_compute_uab
from jamma.lmm.prepare import (
    _build_covariate_matrix,
    _compute_null_model,
    _select_jax_device,
)
from jamma.lmm.results import (
    _concat_jax_accumulators,
    _yield_chunk_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.runner_streaming import _init_accumulators, _LazySnpMeta
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.stats import AssocResult

# Chromosome ordering: numeric first (1-22), then special (X, Y, XY, MT)
_CHR_SPECIAL_ORDER = {"X": 23, "Y": 24, "XY": 25, "MT": 26, "M": 26}


def _chr_sort_key(chrom: str) -> tuple[int, str]:
    """Sort key for biological chromosome order (1..22, X, Y, XY, MT)."""
    upper = chrom.upper()
    if upper in _CHR_SPECIAL_ORDER:
        return (_CHR_SPECIAL_ORDER[upper], "")
    try:
        return (int(chrom), "")
    except ValueError:
        # Unknown chromosome names sort after all known ones
        logger.debug(
            f"Non-numeric chromosome '{chrom}' — sorting after known chromosomes"
        )
        return (100, chrom)


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
) -> tuple[list[AssocResult], int]:
    """Run LOCO LMM association: per-chromosome eigendecomp and association.

    For each chromosome:
    1. Compute K_loco (kinship excluding that chromosome) via streaming
    2. Eigendecompose K_loco
    3. Optionally save K_loco to disk
    4. Delete K_loco (free n^2*8 bytes)
    5. Run LMM association on that chromosome's SNPs
    6. Write results to shared output file

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

    Returns:
        Tuple of (results, n_tested) where results is a list of AssocResult
        in biological chromosome order (1-22, X, Y, XY, MT) with original
        within-chromosome SNP order preserved. Chromosomes that lack
        leave-one-out kinship (e.g., excluded by -ksnps) use full-kinship
        fallback and are yielded after all LOCO chromosomes. Empty list
        if output_path is set. n_tested is the total number of SNPs tested
        across all chromosomes.

    Raises:
        ValueError: If only one chromosome present, or if lmm_mode invalid.
    """
    start_time = time.perf_counter()

    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    # Get metadata
    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps_total = meta["n_snps"]

    # Chromosome partitions (unfiltered)
    partitions = get_chromosome_partitions(bed_path)
    unique_chrs = sorted(partitions.keys(), key=_chr_sort_key)

    if len(unique_chrs) < 2:
        raise ValueError(
            "LOCO requires SNPs on multiple chromosomes. "
            f"Found only {len(unique_chrs)} chromosome(s): {unique_chrs}"
        )

    if show_progress:
        logger.info("Performing LOCO LMM Association Test")
        logger.info(f"  Total individuals: {n_samples_total:,}")
        logger.info(f"  Total SNPs: {n_snps_total:,}")
        logger.info(f"  Chromosomes: {len(unique_chrs)}")

    # Sample filtering: missing phenotypes, covariate NaNs
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)
    if covariates is not None:
        valid_covariate = np.all(~np.isnan(covariates), axis=1)
        valid_mask = valid_mask & valid_covariate
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

    # Build SNP metadata for result construction (lazy -- no upfront dict allocation)
    snp_info = _LazySnpMeta(meta)

    test_type = _TEST_TYPE_MAP[lmm_mode]

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

        # Stream LOCO kinship matrices one at a time
        loco_iter = compute_loco_kinship_streaming(
            bed_path,
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            check_memory=check_memory,
            show_progress=show_progress,
            ksnps_indices=ksnps_indices,
        )

        for chr_idx, (chr_name, K_loco) in enumerate(loco_iter):
            chr_snp_indices = partitions[chr_name]

            if show_progress:
                logger.info(
                    f"LOCO: chromosome {chr_name} ({chr_idx + 1}/{len(unique_chrs)}), "
                    f"{len(chr_snp_indices)} SNPs, eigendecomposing..."
                )

            # Save full K_loco BEFORE subsetting (frees it before eigendecomp)
            if save_kinship and kinship_output_dir is not None:
                kinship_path = (
                    kinship_output_dir
                    / f"{kinship_output_prefix}.loco.cXX.chr{chr_name}.txt"
                )
                write_kinship_matrix(K_loco, kinship_path)
                if show_progress:
                    logger.info(f"  Saved LOCO kinship to {kinship_path}")

            # Subset to valid samples, then free the full matrix
            K_loco_valid = K_loco[np.ix_(valid_mask, valid_mask)]
            del K_loco
            gc.collect()

            # Eigendecompose the valid subset
            eigenvalues_np, U = eigendecompose_kinship(
                K_loco_valid, check_memory=check_memory
            )
            del K_loco_valid
            gc.collect()

            # Run LMM for this chromosome
            chr_results = _run_lmm_for_chromosome(
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
                snps_indices_array=snps_indices,
                col_chunk_size=col_chunk_size,
                writer=writer,
            )

            # When writer is provided, results are already on disk;
            # chr_results is empty. Otherwise accumulate in memory.
            if writer is None:
                all_results.extend(chr_results)

            # Free eigendecomp
            del eigenvalues_np, U
            gc.collect()

        # Clear JIT caches once after all chromosomes (kernels reused across
        # chromosomes since n_samples, n_cvt are identical)
        jax.clear_caches()

        if writer is not None and show_progress:
            logger.info(f"Wrote {writer.count:,} results to {output_path}")

    if show_progress:
        elapsed = time.perf_counter() - start_time
        logger.info(f"LOCO LMM Association completed in {elapsed:.2f}s")

    n_tested = writer.count if writer is not None else len(all_results)
    return ([] if output_path is not None else all_results), n_tested


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
    snps_indices_array: np.ndarray | None = None,
    col_chunk_size: int = 5_000,
    writer: IncrementalAssocWriter | None = None,
) -> list[AssocResult]:
    """Run LMM association on a single chromosome's SNPs.

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
        snps_indices_array: Numpy array of global SNP indices for -snps restriction,
            or None. Passed through for vectorized np.isin filtering.
        col_chunk_size: Number of SNP columns per disk read chunk.
        writer: Optional incremental writer for streaming results to disk.
            When provided, results are written directly and an empty list
            is returned. When None, results are accumulated and returned.

    Returns:
        List of AssocResult for this chromosome's SNPs (empty if writer used).
    """
    n_samples = phenotypes.shape[0]
    valid_indices = np.where(valid_mask)[0]

    # === PASS 1: Chunked SNP statistics ===
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

            # Compute per-SNP statistics for this chunk
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

    if n_unexpected_total > 0:
        logger.warning(
            f"LOCO chr genotype validation: {n_unexpected_total} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    # Apply SNP filter
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, miss_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )

    # Apply SNP list restriction (if -snps provided)
    if snps_indices_array is not None:
        local_snp_list_mask = np.isin(chr_snp_indices, snps_indices_array)
        snp_mask &= local_snp_list_mask

    local_filtered_indices = np.where(snp_mask)[0]
    n_filtered = len(local_filtered_indices)

    if show_progress:
        logger.debug(
            f"  Chromosome SNPs: {len(chr_snp_indices)}, after filter: {n_filtered}"
        )

    if n_filtered == 0:
        if show_progress:
            logger.warning("  Chromosome has no SNPs after filtering, skipping")
        return []

    # Build SNP stat arrays for result construction
    # Map local filtered index -> global SNP index
    global_filtered_indices = chr_snp_indices[local_filtered_indices]
    filtered_afs = allele_freqs[local_filtered_indices]
    filtered_miss = miss_counts[local_filtered_indices].astype(int)
    filtered_means_all = col_means[local_filtered_indices]

    # === PASS 2: Chunked association ===
    # filtered_chr_col_indices: global BED column indices for filtered SNPs
    filtered_chr_col_indices = chr_snp_indices[local_filtered_indices]

    # Eigendecomp setup
    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    with blas_threads():
        UtW = eigenvectors.T @ W
        Uty = eigenvectors.T @ phenotypes

    device = _select_jax_device(use_gpu=False)

    logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
        lmm_mode,
        eigenvalues,
        UtW,
        Uty,
        n_cvt,
        device,
        show_progress=False,
        l_min=l_min,
        l_max=l_max,
    )

    # Device-resident shared arrays
    eigenvalues_jax = jax.device_put(eigenvalues, device)
    UtW_jax = jax.device_put(UtW, device)
    Uty_jax = jax.device_put(Uty, device)

    # LOCO intentionally uses single-device mode — each chromosome's
    # association pass has fewer SNPs, so multi-device sharding overhead
    # outweighs the parallelism benefit.
    jax_chunk_size = _compute_chunk_size(
        n_samples, n_filtered, n_grid, n_cvt, n_devices=1
    )

    def _prepare_jax_chunk(
        start: int, geno: np.ndarray, total: int
    ) -> tuple[np.ndarray, int, bool]:
        """Prepare a JAX chunk for device transfer."""
        end = min(start + jax_chunk_size, total)
        actual_len = end - start
        geno_jax_chunk = geno[:, start:end]

        needs_pad = actual_len < jax_chunk_size
        if needs_pad:
            pad_width = jax_chunk_size - actual_len
            geno_jax_chunk = np.pad(
                geno_jax_chunk, ((0, 0), (0, pad_width)), mode="constant"
            )

        with blas_threads():
            UtG_chunk = np.ascontiguousarray(eigenvectors.T @ geno_jax_chunk)
        return UtG_chunk, actual_len, needs_pad

    # Track lambda boundary hits across all disk chunks
    total_at_lmin = 0
    total_at_lmax = 0
    results: list[AssocResult] = []

    try:
        with open_bed(bed_file) as bed:
            for disk_start in range(0, n_filtered, col_chunk_size):
                disk_end = min(disk_start + col_chunk_size, n_filtered)
                disk_col_indices = filtered_chr_col_indices[disk_start:disk_end]

                geno_disk_chunk = bed.read(
                    index=np.s_[valid_indices, disk_col_indices],
                    dtype=np.float64,
                )

                # Impute missing values with column means
                chunk_filtered_means = filtered_means_all[disk_start:disk_end]
                filtered_means_broadcast = chunk_filtered_means.reshape(1, -1)
                missing_mask = np.isnan(geno_disk_chunk)
                geno_disk_chunk = np.where(
                    missing_mask, filtered_means_broadcast, geno_disk_chunk
                )

                # Process this disk chunk through the existing JAX chunk pipeline
                n_disk_subset = geno_disk_chunk.shape[1]
                jax_starts = list(range(0, n_disk_subset, jax_chunk_size))

                # Fresh accumulator per disk chunk (flush after each)
                accum: dict[str, list] = _init_accumulators(lmm_mode)

                # Prepare first JAX chunk
                UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
                    jax_starts[0], geno_disk_chunk, n_disk_subset
                )
                UtG_jax = jax.device_put(UtG_np, device)
                del UtG_np

                for i, _jax_start in enumerate(jax_starts):
                    current_actual_len = actual_jax_len
                    current_UtG = UtG_jax

                    # Async transfer of next chunk
                    if i + 1 < len(jax_starts):
                        UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
                            jax_starts[i + 1], geno_disk_chunk, n_disk_subset
                        )
                        UtG_jax = jax.device_put(UtG_np, device)
                        del UtG_np

                    # Batch compute Uab
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
                        error_msg = str(e)
                        if "exceeds the maximum representable" in error_msg:
                            logger.error(
                                "JAX int32 buffer overflow in LOCO.\n"
                                f"  JAX chunk {i + 1}: "
                                f"{current_actual_len:,} SNPs x "
                                f"{n_samples:,} samples"
                            )
                        else:
                            logger.error(
                                "JAX computation failed in LOCO:\n"
                                f"  {type(e).__name__}: {error_msg}\n"
                                f"  Chunk: {current_actual_len:,} SNPs, "
                                f"Samples: {n_samples:,}"
                            )
                        raise

                    strip_and_append(chunk_result, accum, current_actual_len)

                del geno_disk_chunk

                # Flush this disk chunk's results immediately
                if any(accum.values()):
                    arrays = _concat_jax_accumulators(lmm_mode, accum)

                    n_lmin, n_lmax = count_lambda_boundary_hits(
                        lmm_mode, arrays, l_min, l_max
                    )
                    total_at_lmin += n_lmin
                    total_at_lmax += n_lmax

                    n_disk_snps = disk_end - disk_start
                    if writer is not None:
                        writer.write_arrays_batch(
                            lmm_mode,
                            global_filtered_indices[disk_start:disk_end],
                            snp_info,
                            filtered_afs[disk_start:disk_end],
                            filtered_miss[disk_start:disk_end],
                            arrays,
                        )
                    else:
                        disk_chunk_results = list(
                            _yield_chunk_results(
                                lmm_mode,
                                np.arange(n_disk_snps),
                                global_filtered_indices[disk_start:disk_end],
                                filtered_afs[disk_start:disk_end],
                                filtered_miss[disk_start:disk_end],
                                snp_info,
                                arrays,
                            )
                        )
                        results.extend(disk_chunk_results)

                    del arrays, accum

        # Log boundary warnings once per chromosome
        log_lambda_boundary_warning(
            total_at_lmin, total_at_lmax, l_min, l_max, prefix="LOCO "
        )

        return results
    finally:
        del eigenvalues_jax, UtW_jax, Uty_jax
