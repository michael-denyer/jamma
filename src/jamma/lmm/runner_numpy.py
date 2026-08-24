"""Pure-NumPy batch LMM association runner.

Input genotypes must fit in memory.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core.memory import estimate_lmm_memory
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.lmm.chunk_runner_numpy import (
    RawLmmChunk,
    run_lmm_chunk_source_numpy,
)
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    prepare_lmm_run,
    validate_runner_inputs,
)
from jamma.lmm.results import _build_results, make_writer_sink
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    LmmConfig,
    LmmRunResult,
    RunnerTiming,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.utils.logging import log_rss_memory


def run_lmm_association_numpy(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    snp_info: list,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    config: LmmConfig = DEFAULT_LMM_CONFIG,
    output_path: Path | None = None,
) -> LmmRunResult:
    """Run LMM association tests using pure-NumPy batch processing.

    Processes SNPs in memory-bounded chunks using BLAS-backed NumPy operations.
    Input genotypes must fit in memory; for disk streaming
    use run_lmm_association_numpy_streaming.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples) or None when
            pre-computed eigenvalues/eigenvectors are provided. WARNING: may
            be overwritten in-place during eigendecomposition (buffer reused
            for eigenvectors). Treat as consumed; pass kinship.copy() if you
            need the original matrix after this call.
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0.
        covariates: Covariate matrix (n_samples, n_cvt) or None for intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (clamped to min 20
            internally for ~1e-5 tolerance).
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        config: LmmConfig instance. When provided, overrides individual
            threshold/mode kwargs above.
        output_path: Path for per-chunk disk streaming. When set, results
            are written incrementally and the returned LmmRunResult has
            empty associations and n_tested populated instead.
    Returns:
        LmmRunResult with per-SNP associations and PVE from null model.
            When output_path is set, associations is empty (results on
            disk) and n_tested contains the count of SNPs written.

    Raises:
        MemoryError: If check_memory=True and insufficient memory.
        ValueError: If only one of eigenvalues/eigenvectors is provided,
            or if no valid samples remain after filtering.
    """
    # One source for every knob. The runner reads locals rather than config.x
    # at forty-odd sites; the dual surface this replaced let a caller pass both
    # a config and a contradicting keyword.
    maf_threshold = config.maf_threshold
    miss_threshold = config.miss_threshold
    l_min, l_max = config.l_min, config.l_max
    n_grid, n_refine = config.n_grid, config.n_refine
    check_memory = config.check_memory
    show_progress = config.show_progress
    # schema.LmmMode is a loose int alias (kept loose to avoid a circular
    # import); the chunk engine wants the Literal.
    lmm_mode = config.lmm_mode

    # Memory check before workflow (uses genotype shape, runner-specific)
    n_samples, n_snps = genotypes.shape
    start_time = time.perf_counter()

    if show_progress:
        logger.info("Performing LMM Association Test (NumPy batch)")
        logger.info(f"  Total individuals: {n_samples:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.debug(
            f"MAF threshold = {maf_threshold}, missing threshold = {miss_threshold}"
        )

    if check_memory:
        # Propagate n_cvt so the preflight correctly sizes Uab/Iab for
        # multi-covariate runs. Otherwise the estimator silently uses its
        # n_cvt=1 default and can let a multi-covariate run pass preflight
        # before OOMing at the real allocation. Intercept column counts as
        # a covariate, so minimum is 1 when no user covariates are passed.
        n_cvt = covariates.shape[1] if covariates is not None else 1
        est = estimate_lmm_memory(n_samples, n_snps, n_cvt=n_cvt)
        logger.info(
            f"LMM memory: estimated {est.total_gb:.1f}GB, "
            f"available {est.available_gb:.1f}GB"
        )
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for LMM workflow with {n_samples:,} samples × "
                f"{n_snps:,} SNPs.\n"
                f"Need: {est.total_gb:.1f}GB, Available: {est.available_gb:.1f}GB\n"
                f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
                f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"genotypes={est.genotypes_gb:.1f}GB"
            )

    # Validate inputs and apply sample filtering (shared logic for all runners)
    setup = validate_runner_inputs(
        phenotypes, kinship, covariates, eigenvalues, eigenvectors, lmm_mode
    )
    phenotypes = setup.phenotypes
    kinship = setup.kinship
    covariates = setup.covariates
    eigenvalues = setup.eigenvalues
    eigenvectors = setup.eigenvectors
    n_samples = setup.n_samples

    # Apply the same valid-mask to genotypes (runner-specific: genotypes in memory)
    if not np.all(setup.valid_mask):
        genotypes = genotypes[setup.valid_mask, :]

    n_samples, n_snps = genotypes.shape

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Vectorized SNP stats and filtering using shared functions
    col_means, missing_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, missing_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )
    snp_indices = np.where(snp_mask)[0]

    if len(snp_indices) == 0:
        logger.warning(
            f"All {n_snps} SNPs filtered out (MAF>{maf_threshold}, "
            f"miss<{miss_threshold}). No association tests to run. "
            f"Consider relaxing --maf or --miss thresholds."
        )
        if output_path is not None:
            from jamma.lmm.io import IncrementalAssocWriter

            with IncrementalAssocWriter(
                output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            ):
                pass  # Header-only file, matching streaming runner behavior
        return LmmRunResult(associations=[], n_tested=0)

    # Extract filtered stats as numpy arrays (use allele_freqs for output, not mafs)
    filtered_afs = allele_freqs[snp_indices]
    filtered_miss = missing_counts[snp_indices].astype(int)

    t_eigen_start = time.perf_counter()
    prepared = prepare_lmm_run(
        kinship=kinship,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        phenotypes=phenotypes,
        W=W,
        n_cvt=n_cvt,
        lmm_mode=lmm_mode,
        l_min=l_min,
        l_max=l_max,
        show_progress=show_progress,
        check_memory=check_memory,
        label="lmm_numpy",
    )
    del kinship
    gc.collect()
    t_eigen_end = time.perf_counter()

    n_filtered = len(snp_indices)

    filtered_means = col_means[snp_indices]

    # Streaming mode: write per-chunk to disk, skip arrays_out allocation.
    streaming = output_path is not None
    if streaming:
        from jamma.lmm.io import IncrementalAssocWriter

        writer_ctx = IncrementalAssocWriter(
            output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
        )
        arrays_out = None
    else:
        writer_ctx = nullcontext()
        arrays_out = {
            key: np.empty(n_filtered, dtype=np.float64)
            for key in _RESULT_FIELDS[lmm_mode]
        }

    def _make_batch_source(
        source_chunk_size: int,
    ) -> Callable[[], RawLmmChunk | None]:
        chunk_starts = iter(range(0, n_filtered, source_chunk_size))
        geno_buf = np.empty((n_samples, source_chunk_size), dtype=np.float64)

        def _next_chunk() -> RawLmmChunk | None:
            try:
                chunk_start = next(chunk_starts)
            except StopIteration:
                return None

            chunk_end = min(chunk_start + source_chunk_size, n_filtered)
            actual_len = chunk_end - chunk_start
            geno_chunk = geno_buf[:, :actual_len]
            geno_chunk[:] = genotypes[:, snp_indices[chunk_start:chunk_end]]
            return RawLmmChunk(geno_chunk, chunk_start, chunk_end)

        return _next_chunk

    def _fill_arrays_sink(
        chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
    ) -> None:
        assert arrays_out is not None
        s = slice(filtered_start, filtered_end)
        for key in arrays_out:
            arrays_out[key][s] = chunk_arrays[key]

    with writer_ctx as writer:
        if streaming:
            assert writer is not None
            _sink = make_writer_sink(
                writer, lmm_mode, snp_info, snp_indices, filtered_afs, filtered_miss
            )
        else:
            _sink = _fill_arrays_sink

        chunk_stats = run_lmm_chunk_source_numpy(
            raw_chunk_source_factory=_make_batch_source,
            chunk_sink=_sink,
            U=prepared.U,
            eigenvalues_np=prepared.eigenvalues,
            UtW=prepared.UtW,
            Uty=prepared.Uty,
            Hi_eval_null=prepared.Hi_eval_null,
            logl_H0=prepared.logl_H0,
            n_samples=n_samples,
            n_filtered=n_filtered,
            n_cvt=n_cvt,
            lmm_mode=lmm_mode,
            filtered_means=filtered_means,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
            show_progress=show_progress,
            progress_label="LMM association",
        )

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_numpy", "after_all_chunks")

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        t_eigen = t_eigen_end - t_eigen_start
        accounted = (
            t_eigen
            + chunk_stats.rotation_s
            + chunk_stats.compute_s
            + chunk_stats.result_write_s
        )
        logger.info("Timing breakdown:")
        logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
        logger.info(f"  UT@G rotation:       {chunk_stats.rotation_s:.2f}s")
        logger.info(f"  NumPy compute:       {chunk_stats.compute_s:.2f}s")
        logger.info(f"  Result write:        {chunk_stats.result_write_s:.2f}s")
        logger.info("  ----")
        logger.info(f"  Accounted:           {accounted:.2f}s")
        logger.info(f"  Total:               {elapsed:.2f}s")
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    timing = RunnerTiming(
        rotation_s=chunk_stats.rotation_s,
        numpy_compute_s=chunk_stats.compute_s,
        result_write_s=chunk_stats.result_write_s,
    )
    if streaming:
        return LmmRunResult(
            associations=[],
            n_tested=chunk_stats.processed,
            pve=prepared.pve,
            pve_se=prepared.pve_se,
            timing=timing,
        )

    assert arrays_out is not None
    associations = _build_results(
        lmm_mode, snp_indices, filtered_afs, filtered_miss, snp_info, arrays_out
    )
    return LmmRunResult(
        associations=associations,
        n_tested=len(associations),
        pve=prepared.pve,
        pve_se=prepared.pve_se,
        timing=timing,
    )
