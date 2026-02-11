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

import gc
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed
from loguru import logger

from jamma.core.snp_filter import compute_snp_filter_mask
from jamma.core.threading import blas_threads
from jamma.io.plink import get_chromosome_partitions, get_plink_metadata
from jamma.kinship import compute_loco_kinship_streaming, write_kinship_matrix
from jamma.lmm.chunk import _compute_chunk_size
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.likelihood_jax import (
    batch_calc_score_stats,
    batch_calc_wald_stats,
    batch_compute_iab,
    batch_compute_uab,
    calc_lrt_pvalue_jax,
    golden_section_optimize_lambda_mle,
)
from jamma.lmm.prepare import (
    _build_covariate_matrix,
    _compute_null_model,
    _grid_optimize_lambda_batched,
    _select_jax_device,
)
from jamma.lmm.results import _concat_jax_accumulators
from jamma.lmm.runner_streaming import (
    _append_chunk_results,
    _init_accumulators,
)
from jamma.lmm.stats import AssocResult


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
) -> list[AssocResult]:
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

    Returns:
        List of AssocResult in original SNP order (empty if output_path set).

    Raises:
        ValueError: If only one chromosome present, or if lmm_mode invalid.
    """
    start_time = time.perf_counter()

    # Get metadata
    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps_total = meta["n_snps"]

    # Chromosome partitions (unfiltered)
    partitions = get_chromosome_partitions(bed_path)
    unique_chrs = sorted(partitions.keys())

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

    # Build SNP metadata for result construction
    snp_info = [
        {
            "chr": str(meta["chromosome"][i]),
            "rs": meta["sid"][i],
            "pos": int(meta["bp_position"][i]),
            "a1": meta["allele_1"][i],
            "a0": meta["allele_2"][i],
        }
        for i in range(n_snps_total)
    ]

    # Determine test type for incremental writer
    test_type_map = {1: "wald", 2: "lrt", 3: "score", 4: "all"}
    test_type = test_type_map.get(lmm_mode, "wald")

    all_results: list[AssocResult] = []

    # Use IncrementalAssocWriter as context manager when writing to disk
    writer_ctx = (
        IncrementalAssocWriter(output_path, test_type=test_type)
        if output_path is not None
        else None
    )

    try:
        writer = writer_ctx.__enter__() if writer_ctx is not None else None

        # Stream LOCO kinship matrices one at a time
        loco_iter = compute_loco_kinship_streaming(
            bed_path,
            check_memory=check_memory,
            show_progress=show_progress,
        )

        for chr_idx, (chr_name, K_loco) in enumerate(loco_iter):
            chr_snp_indices = partitions[chr_name]

            if show_progress:
                logger.info(
                    f"LOCO: chromosome {chr_name} ({chr_idx + 1}/{len(unique_chrs)}), "
                    f"{len(chr_snp_indices)} SNPs, eigendecomposing..."
                )

            # Eigendecompose K_loco (subset to valid samples)
            K_loco_valid = K_loco[np.ix_(valid_mask, valid_mask)]
            eigenvalues_np, U = eigendecompose_kinship(K_loco_valid)

            # Optionally save kinship
            if save_kinship and kinship_output_dir is not None:
                kinship_path = (
                    kinship_output_dir
                    / f"{kinship_output_prefix}.loco.cXX.chr{chr_name}.txt"
                )
                write_kinship_matrix(K_loco, kinship_path)
                if show_progress:
                    logger.info(f"  Saved LOCO kinship to {kinship_path}")

            # Free K_loco
            del K_loco, K_loco_valid
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
            )

            # Write results
            if writer is not None:
                writer.write_batch(chr_results)
            else:
                all_results.extend(chr_results)

            # Free eigendecomp
            del eigenvalues_np, U
            gc.collect()
            jax.clear_caches()

    finally:
        if writer_ctx is not None:
            writer_ctx.__exit__(None, None, None)
            if show_progress:
                logger.info(f"Wrote {writer.count:,} results to {output_path}")

    if show_progress:
        elapsed = time.perf_counter() - start_time
        logger.info(f"LOCO LMM Association completed in {elapsed:.2f}s")

    return [] if output_path is not None else all_results


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
) -> list[AssocResult]:
    """Run LMM association on a single chromosome's SNPs.

    Reads only the chromosome's SNPs from the BED file in a single read
    (chromosome subsets are small enough to fit in memory), computes SNP
    statistics, applies filters, then runs the JAX LMM pipeline using
    pre-computed eigendecomposition.

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

    Returns:
        List of AssocResult for this chromosome's SNPs.
    """
    n_samples = phenotypes.shape[0]

    # Read this chromosome's SNPs from BED file in a single read
    bed_file = Path(f"{bed_path}.bed")
    with open_bed(bed_file) as bed:
        geno_chr = bed.read(index=np.s_[:, chr_snp_indices], dtype=np.float64)

    # Apply sample filtering
    if not np.all(valid_mask):
        geno_chr = geno_chr[valid_mask, :]

    # Compute SNP statistics for filtering
    miss_counts = np.sum(np.isnan(geno_chr), axis=0)
    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(geno_chr, axis=0)
        col_vars = np.nanvar(geno_chr, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    col_vars = np.nan_to_num(col_vars, nan=0.0)

    # Apply SNP filter
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, miss_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )
    local_filtered_indices = np.where(snp_mask)[0]
    n_filtered = len(local_filtered_indices)

    if show_progress:
        logger.debug(
            f"  Chromosome SNPs: {len(chr_snp_indices)}, after filter: {n_filtered}"
        )

    if n_filtered == 0:
        return []

    # Build snp_stats for result construction
    # Map local filtered index -> global SNP index
    global_filtered_indices = chr_snp_indices[local_filtered_indices]
    snp_stats = list(
        zip(
            allele_freqs[local_filtered_indices],
            miss_counts[local_filtered_indices].astype(int),
            strict=True,
        )
    )
    filtered_means = col_means[local_filtered_indices]

    # Extract filtered genotype columns and impute missing to mean
    geno_filtered = geno_chr[:, local_filtered_indices].copy()
    filtered_means_broadcast = filtered_means.reshape(1, -1)
    missing_mask = np.isnan(geno_filtered)
    geno_filtered = np.where(missing_mask, filtered_means_broadcast, geno_filtered)

    del geno_chr
    gc.collect()

    # Eigendecomp setup
    UT = np.ascontiguousarray(eigenvectors.T)
    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    with blas_threads():
        UtW = UT @ W
        Uty = UT @ phenotypes

    device = _select_jax_device(use_gpu=False)

    logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
        lmm_mode, eigenvalues, UtW, Uty, n_cvt, device, show_progress=False
    )

    # Device-resident shared arrays
    eigenvalues_jax = jax.device_put(eigenvalues, device)
    UtW_jax = jax.device_put(UtW, device)
    Uty_jax = jax.device_put(Uty, device)

    jax_chunk_size = _compute_chunk_size(n_samples, n_filtered, n_grid, n_cvt)

    # Process in JAX chunks
    n_subset = geno_filtered.shape[1]
    jax_starts = list(range(0, n_subset, jax_chunk_size))

    accum: dict[str, list] = _init_accumulators(lmm_mode)

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
            UtG_chunk = np.ascontiguousarray(UT @ geno_jax_chunk)
        return UtG_chunk, actual_len, needs_pad

    # Prepare first chunk
    UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
        jax_starts[0], geno_filtered, n_subset
    )
    UtG_jax = jax.device_put(UtG_np, device)
    del UtG_np

    for i, _jax_start in enumerate(jax_starts):
        current_actual_len = actual_jax_len
        current_needs_padding = needs_padding
        current_UtG = UtG_jax

        # Async transfer of next chunk
        if i + 1 < len(jax_starts):
            UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
                jax_starts[i + 1], geno_filtered, n_subset
            )
            UtG_jax = jax.device_put(UtG_np, device)
            del UtG_np

        # Batch compute Uab
        Uab_batch = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, current_UtG)

        if lmm_mode == 1:
            Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
            best_lambdas, best_logls = _grid_optimize_lambda_batched(
                n_cvt,
                eigenvalues_jax,
                Uab_batch,
                Iab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
            )
            betas, ses, p_walds = batch_calc_wald_stats(
                n_cvt, best_lambdas, eigenvalues_jax, Uab_batch, n_samples
            )
            p_walds.block_until_ready()

        elif lmm_mode == 3:
            betas, ses, p_scores = batch_calc_score_stats(
                n_cvt, Hi_eval_null_jax, Uab_batch, n_samples
            )
            p_scores.block_until_ready()

        elif lmm_mode == 2:
            best_lambdas_mle, best_logls_mle = golden_section_optimize_lambda_mle(
                n_cvt,
                eigenvalues_jax,
                Uab_batch,
                l_min=l_min,
                l_max=l_max,
                n_grid=n_grid,
                n_iter=max(n_refine, 20),
            )
            p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
                best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
            )
            p_lrts.block_until_ready()

        elif lmm_mode == 4:
            _, _, p_scores = batch_calc_score_stats(
                n_cvt, Hi_eval_null_jax, Uab_batch, n_samples
            )
            best_lambdas_mle, best_logls_mle = golden_section_optimize_lambda_mle(
                n_cvt,
                eigenvalues_jax,
                Uab_batch,
                l_min=l_min,
                l_max=l_max,
                n_grid=n_grid,
                n_iter=max(n_refine, 20),
            )
            p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
                best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
            )
            Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
            best_lambdas, best_logls = _grid_optimize_lambda_batched(
                n_cvt,
                eigenvalues_jax,
                Uab_batch,
                Iab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
            )
            betas, ses, p_walds = batch_calc_wald_stats(
                n_cvt, best_lambdas, eigenvalues_jax, Uab_batch, n_samples
            )
            p_scores.block_until_ready()
            p_lrts.block_until_ready()
            p_walds.block_until_ready()

        # Strip padding and append to accumulators
        _append_chunk_results(
            lmm_mode, accum, current_actual_len, current_needs_padding, locals()
        )

    # Build results from accumulators
    results: list[AssocResult] = []
    if any(accum.values()):
        arrays = _concat_jax_accumulators(lmm_mode, accum)

        # Build results using global SNP indices
        for j in range(n_filtered):
            global_idx = int(global_filtered_indices[j])
            af, n_miss = snp_stats[j]
            info = snp_info[global_idx]
            meta_dict = {
                "chr": info["chr"],
                "rs": info["rs"],
                "ps": info["pos"],
                "n_miss": n_miss,
                "allele1": info["a1"],
                "allele0": info["a0"],
                "af": af,
            }

            if lmm_mode == 1:
                results.append(
                    AssocResult(
                        **meta_dict,
                        beta=float(arrays["betas"][j]),
                        se=float(arrays["ses"][j]),
                        logl_H1=float(arrays["logls"][j]),
                        l_remle=float(arrays["lambdas"][j]),
                        p_wald=float(arrays["pwalds"][j]),
                    )
                )
            elif lmm_mode == 3:
                results.append(
                    AssocResult(
                        **meta_dict,
                        beta=float(arrays["betas"][j]),
                        se=float(arrays["ses"][j]),
                        p_score=float(arrays["p_scores"][j]),
                    )
                )
            elif lmm_mode == 2:
                results.append(
                    AssocResult(
                        **meta_dict,
                        beta=float("nan"),
                        se=float("nan"),
                        l_mle=float(arrays["lambdas_mle"][j]),
                        p_lrt=float(arrays["p_lrts"][j]),
                    )
                )
            elif lmm_mode == 4:
                results.append(
                    AssocResult(
                        **meta_dict,
                        beta=float(arrays["betas"][j]),
                        se=float(arrays["ses"][j]),
                        logl_H1=float(arrays["logls"][j]),
                        l_remle=float(arrays["lambdas"][j]),
                        l_mle=float(arrays["lambdas_mle"][j]),
                        p_wald=float(arrays["pwalds"][j]),
                        p_lrt=float(arrays["p_lrts"][j]),
                        p_score=float(arrays["p_scores"][j]),
                    )
                )

    del eigenvalues_jax, UtW_jax, Uty_jax
    jax.clear_caches()

    return results
