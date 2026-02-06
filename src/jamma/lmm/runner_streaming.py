"""Streaming LMM association runner.

Two-pass disk streaming: (1) SNP statistics, (2) association per chunk.
Never allocates the full genotype matrix.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jamma.core.memory import estimate_streaming_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.lmm.chunk import _compute_chunk_size
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
    _eigendecompose_or_reuse,
    _grid_optimize_lambda_batched,
    _select_jax_device,
)
from jamma.lmm.results import _concat_jax_accumulators, _yield_chunk_results
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory

_ACCUM_KEYS = {
    1: ("lambdas", "logls", "betas", "ses", "pwalds"),
    2: ("lambdas_mle", "logls_mle", "p_lrts"),
    3: ("betas", "ses", "p_scores"),
    4: (
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "logls_mle",
        "p_lrts",
        "p_scores",
    ),
}

_FIRST_ARRAY_KEY = {1: "lambdas", 2: "lambdas_mle", 3: "betas", 4: "lambdas"}


def _init_accumulators(lmm_mode: int) -> dict[str, list]:
    """Create empty accumulator dict for the given mode."""
    return {k: [] for k in _ACCUM_KEYS[lmm_mode]}


def _append_chunk_results(
    lmm_mode: int,
    accum: dict[str, list],
    actual_len: int,
    needs_padding: bool,
    local_vars: dict,
) -> None:
    """Strip padding from JAX arrays and append to accumulators."""
    _var_map = {
        "lambdas": "best_lambdas",
        "logls": "best_logls",
        "betas": "betas",
        "ses": "ses",
        "pwalds": "p_walds",
        "lambdas_mle": "best_lambdas_mle",
        "logls_mle": "best_logls_mle",
        "p_lrts": "p_lrts",
        "p_scores": "p_scores",
    }
    first_key = _FIRST_ARRAY_KEY[lmm_mode]
    first_var = _var_map[first_key]
    arr = local_vars[first_var]
    slice_len = actual_len if needs_padding else len(arr)
    for key in _ACCUM_KEYS[lmm_mode]:
        accum[key].append(local_vars[_var_map[key]][:slice_len])


def run_lmm_association_streaming(
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list | None = None,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    chunk_size: int = 10_000,
    use_gpu: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
    output_path: Path | None = None,
    lmm_mode: int = 1,
) -> list[AssocResult]:
    """Run LMM association tests by streaming genotypes from disk.

    Reads genotypes per-chunk, never allocating the full genotype matrix.
    Two-pass: (1) SNP statistics for filtering, (2) association per chunk.

    Args:
        bed_path: PLINK file prefix (without .bed/.bim/.fam extension).
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples).
        snp_info: List of SNP metadata dicts, or None to build from PLINK.
        covariates: Covariate matrix (n_samples, n_cvt) or None for intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations for lambda refinement.
        chunk_size: Number of SNPs per disk chunk (default: 10,000).
        use_gpu: Whether to use GPU acceleration.
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        output_path: Path for incremental result writing, or None for in-memory.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.

    Returns:
        List of AssocResult (empty if output_path is set -- results on disk).

    Raises:
        MemoryError: If check_memory=True and insufficient memory.
        ValueError: If only one of eigenvalues/eigenvectors is provided.
    """
    start_time = time.perf_counter()

    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )

    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps = meta["n_snps"]

    if snp_info is None:
        snp_info = [
            {
                "chr": str(meta["chromosome"][i]),
                "rs": meta["sid"][i],
                "pos": int(meta["bp_position"][i]),
                "a1": meta["allele_1"][i],
                "a0": meta["allele_2"][i],
            }
            for i in range(n_snps)
        ]

    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)
    if covariates is not None:
        valid_covariate = np.all(~np.isnan(covariates), axis=1)
        valid_mask = valid_mask & valid_covariate
    n_valid = int(np.sum(valid_mask))
    if not np.all(valid_mask):
        phenotypes = phenotypes[valid_mask]
        kinship = kinship[np.ix_(valid_mask, valid_mask)]
        if covariates is not None:
            covariates = covariates[valid_mask, :]

    n_samples = phenotypes.shape[0]

    if check_memory:
        est = estimate_streaming_memory(n_samples, n_snps, chunk_size=chunk_size)
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for streaming LMM with {n_samples:,} samples "
                f"x {n_snps:,} SNPs (chunk_size={chunk_size:,}).\n"
                f"Peak: {est.total_peak_gb:.1f}GB, "
                f"Available: {est.available_gb:.1f}GB\n"
                f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
                f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"eigendecomp_workspace={est.eigendecomp_workspace_gb:.1f}GB"
            )

    if show_progress:
        logger.info("## Performing LMM Association Test (Streaming)")
        logger.info(f"number of total individuals = {n_samples_total}")
        logger.info(f"number of analyzed individuals = {n_valid}")
        logger.info(f"number of total SNPs/variants = {n_snps}")
        logger.info(f"lambda range = [{l_min:.2e}, {l_max:.2e}]")

    device = _select_jax_device(use_gpu)

    # === PASS 1: SNP statistics (without loading all genotypes) ===
    t_io_start = time.perf_counter()
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)  # For monomorphic detection

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks, desc="Computing SNP statistics"
        )

    for chunk, start, end in stats_iterator:
        # Apply sample filtering
        if not np.all(valid_mask):
            chunk = chunk[valid_mask, :]

        # Compute stats for this chunk
        chunk_miss_counts = np.sum(np.isnan(chunk), axis=0)
        with np.errstate(invalid="ignore"):
            chunk_means = np.nanmean(chunk, axis=0)
            chunk_vars = np.nanvar(chunk, axis=0)  # For monomorphic detection
        chunk_means = np.nan_to_num(chunk_means, nan=0.0)
        chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

        all_means[start:end] = chunk_means
        all_miss_counts[start:end] = chunk_miss_counts
        all_vars[start:end] = chunk_vars

    t_io_end = time.perf_counter()

    # === SNP statistics: filtering + stats construction ===
    t_snp_start = time.perf_counter()
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        all_means, all_miss_counts, all_vars, n_samples, maf_threshold, miss_threshold
    )
    snp_indices = np.where(snp_mask)[0]
    n_filtered = len(snp_indices)

    if show_progress:
        logger.info(f"number of analyzed SNPs = {n_filtered}")

    if n_filtered == 0:
        if show_progress:
            elapsed = time.perf_counter() - start_time
            logger.info("## LMM Association completed")
            logger.info(f"time elapsed = {elapsed:.2f} seconds")
        return []

    snp_stats = list(
        zip(
            allele_freqs[snp_indices],
            all_miss_counts[snp_indices].astype(int),
            strict=False,
        )
    )
    filtered_means = all_means[snp_indices]
    t_snp_end = time.perf_counter()

    # === Eigendecomp + setup ===
    t_eigen_start = time.perf_counter()
    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship, eigenvalues, eigenvectors, show_progress, "lmm_streaming"
    )
    UT = np.ascontiguousarray(U.T)  # Cache contiguous transpose for BLAS matmuls

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Prepare rotated matrices
    UtW = UT @ W
    Uty = UT @ phenotypes

    logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
        lmm_mode, eigenvalues_np, UtW, Uty, n_cvt, device, show_progress
    )

    # Device-resident shared arrays - placed on device ONCE before chunk loop
    eigenvalues = jax.device_put(eigenvalues_np, device)
    UtW_jax = jax.device_put(UtW, device)
    Uty_jax = jax.device_put(Uty, device)

    jax_chunk_size = _compute_chunk_size(n_samples, n_filtered, n_grid, n_cvt)
    t_eigen_end = time.perf_counter()

    # Timing accumulators for per-chunk phases
    t_rotation_total = 0.0
    t_jax_compute_total = 0.0
    t_result_write_total = 0.0

    # === PASS 2: Association ===
    writer = None
    if output_path is not None:
        test_type_map = {1: "wald", 2: "lrt", 3: "score", 4: "all"}
        test_type = test_type_map.get(lmm_mode, "wald")
        writer = IncrementalAssocWriter(output_path, test_type=test_type)
        writer.__enter__()

    # Track results for in-memory mode (when output_path is None)
    all_results: list[AssocResult] = []

    try:
        assoc_iterator = stream_genotype_chunks(
            bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
        )
        if show_progress:
            n_chunks = (n_snps + chunk_size - 1) // chunk_size
            assoc_iterator = progress_iterator(
                assoc_iterator, total=n_chunks, desc="Running LMM association"
            )

        for chunk, file_start, file_end in assoc_iterator:
            # Apply sample filtering
            if not np.all(valid_mask):
                chunk = chunk[valid_mask, :]

            chunk_filtered_indices = []
            chunk_filtered_local_idx = []
            chunk_filtered_col_idx = []
            for i, snp_idx in enumerate(snp_indices):
                if file_start <= snp_idx < file_end:
                    chunk_filtered_indices.append(snp_idx)
                    chunk_filtered_local_idx.append(i)
                    chunk_filtered_col_idx.append(snp_idx - file_start)

            if len(chunk_filtered_indices) == 0:
                continue

            chunk_filtered_col_idx_arr = np.array(chunk_filtered_col_idx)
            geno_subset = chunk[:, chunk_filtered_col_idx_arr].copy()

            for j, local_idx in enumerate(chunk_filtered_local_idx):
                missing_mask = np.isnan(geno_subset[:, j])
                if np.any(missing_mask):
                    geno_subset[missing_mask, j] = filtered_means[local_idx]

            n_subset = geno_subset.shape[1]
            jax_starts = list(range(0, n_subset, jax_chunk_size))

            def _prepare_jax_chunk(
                start: int, geno: np.ndarray, total: int
            ) -> tuple[np.ndarray, int, bool]:
                """Prepare a JAX chunk for device transfer (CPU work)."""
                end = min(start + jax_chunk_size, total)
                actual_len = end - start

                geno_jax_chunk = geno[:, start:end]

                needs_pad = actual_len < jax_chunk_size
                if needs_pad:
                    pad_width = jax_chunk_size - actual_len
                    geno_jax_chunk = np.pad(
                        geno_jax_chunk, ((0, 0), (0, pad_width)), mode="constant"
                    )

                UtG_chunk = np.ascontiguousarray(UT @ geno_jax_chunk)
                return UtG_chunk, actual_len, needs_pad

            # Dict-based accumulators for this file chunk
            accum: dict[str, list] = _init_accumulators(lmm_mode)

            # Prepare first JAX chunk
            t_rot_start = time.perf_counter()
            UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
                jax_starts[0], geno_subset, n_subset
            )
            t_rot_end = time.perf_counter()
            t_rotation_total += t_rot_end - t_rot_start
            UtG_jax = jax.device_put(UtG_np, device)
            del UtG_np

            for i, _jax_start in enumerate(jax_starts):
                current_actual_len = actual_jax_len
                current_needs_padding = needs_padding
                current_UtG = UtG_jax

                # Start async transfer of next JAX chunk while computing current
                if i + 1 < len(jax_starts):
                    t_rot_start = time.perf_counter()
                    UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
                        jax_starts[i + 1], geno_subset, n_subset
                    )
                    t_rot_end = time.perf_counter()
                    t_rotation_total += t_rot_end - t_rot_start
                    UtG_jax = jax.device_put(UtG_np, device)
                    del UtG_np

                # --- JAX compute timing ---
                t_jax_start = time.perf_counter()

                # Batch compute Uab (shared across all modes)
                Uab_batch = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, current_UtG)

                if lmm_mode == 1:  # Wald (existing, unchanged)
                    Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
                    best_lambdas, best_logls = _grid_optimize_lambda_batched(
                        n_cvt,
                        eigenvalues,
                        Uab_batch,
                        Iab_batch,
                        l_min,
                        l_max,
                        n_grid,
                        n_refine,
                    )
                    betas, ses, p_walds = batch_calc_wald_stats(
                        n_cvt, best_lambdas, eigenvalues, Uab_batch, n_samples
                    )
                    p_walds.block_until_ready()

                elif lmm_mode == 3:  # Score
                    betas, ses, p_scores = batch_calc_score_stats(
                        n_cvt, Hi_eval_null_jax, Uab_batch, n_samples
                    )
                    p_scores.block_until_ready()

                elif lmm_mode == 2:  # LRT
                    best_lambdas_mle, best_logls_mle = (
                        golden_section_optimize_lambda_mle(
                            n_cvt,
                            eigenvalues,
                            Uab_batch,
                            l_min=l_min,
                            l_max=l_max,
                            n_grid=n_grid,
                            n_iter=max(n_refine, 20),
                        )
                    )
                    p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
                        best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
                    )
                    p_lrts.block_until_ready()

                elif lmm_mode == 4:  # All tests
                    # Score test (cheapest, no optimization)
                    _, _, p_scores = batch_calc_score_stats(
                        n_cvt, Hi_eval_null_jax, Uab_batch, n_samples
                    )

                    # MLE optimization for LRT
                    best_lambdas_mle, best_logls_mle = (
                        golden_section_optimize_lambda_mle(
                            n_cvt,
                            eigenvalues,
                            Uab_batch,
                            l_min=l_min,
                            l_max=l_max,
                            n_grid=n_grid,
                            n_iter=max(n_refine, 20),
                        )
                    )
                    p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
                        best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
                    )

                    # REML optimization for Wald
                    Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
                    best_lambdas, best_logls = _grid_optimize_lambda_batched(
                        n_cvt,
                        eigenvalues,
                        Uab_batch,
                        Iab_batch,
                        l_min,
                        l_max,
                        n_grid,
                        n_refine,
                    )
                    betas, ses, p_walds = batch_calc_wald_stats(
                        n_cvt, best_lambdas, eigenvalues, Uab_batch, n_samples
                    )
                    p_walds.block_until_ready()

                t_jax_end = time.perf_counter()
                t_jax_compute_total += t_jax_end - t_jax_start

                # Strip padding and append to dict-based accumulators
                _append_chunk_results(
                    lmm_mode,
                    accum,
                    current_actual_len,
                    current_needs_padding,
                    locals(),
                )

            # Concatenate, build results, write/accumulate
            if any(accum.values()):
                t_write_start = time.perf_counter()
                arrays = _concat_jax_accumulators(lmm_mode, accum)
                for result in _yield_chunk_results(
                    lmm_mode,
                    chunk_filtered_local_idx,
                    snp_indices,
                    snp_stats,
                    snp_info,
                    arrays,
                ):
                    if writer is not None:
                        writer.write(result)
                    else:
                        all_results.append(result)
                del arrays, accum
                t_write_end = time.perf_counter()
                t_result_write_total += t_write_end - t_write_start

        if show_progress:
            log_rss_memory("lmm_streaming", "after_association")

            elapsed = time.perf_counter() - start_time
            logger.info("## Timing breakdown:")
            logger.info(f"##   I/O read (pass 1):   {t_io_end - t_io_start:.2f}s")
            logger.info(f"##   SNP statistics:      {t_snp_end - t_snp_start:.2f}s")
            logger.info(f"##   Eigendecomp+setup:   {t_eigen_end - t_eigen_start:.2f}s")
            logger.info(f"##   UT@G rotation:       {t_rotation_total:.2f}s")
            logger.info(f"##   JAX compute:         {t_jax_compute_total:.2f}s")
            logger.info(f"##   Result write:        {t_result_write_total:.2f}s")
            accounted = (
                (t_io_end - t_io_start)
                + (t_snp_end - t_snp_start)
                + (t_eigen_end - t_eigen_start)
                + t_rotation_total
                + t_jax_compute_total
                + t_result_write_total
            )
            logger.info(f"##   Accounted:           {accounted:.2f}s")
            logger.info(f"##   Total:               {elapsed:.2f}s")

        del eigenvalues, UtW_jax, Uty_jax

    finally:
        if writer is not None:
            writer.__exit__(None, None, None)
            if show_progress:
                logger.info(f"Wrote {writer.count:,} results to {output_path}")

    jax.clear_caches()

    if show_progress:
        elapsed = time.perf_counter() - start_time
        logger.info("## LMM Association completed")
        logger.info(f"time elapsed = {elapsed:.2f} seconds")

    return [] if output_path is not None else all_results
