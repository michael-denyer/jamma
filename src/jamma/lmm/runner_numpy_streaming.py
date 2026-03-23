"""Disk-streaming NumPy LMM association runner.

Two-pass disk streaming using C extension compute path:
  Pass 1: SNP statistics for filtering (float32, lightweight).
  Pass 2: Association per chunk via C workspace (float64, compute-heavy).

Never allocates the full genotype matrix. Uses jlinalg.dgemm for rotation
and the _lmm_accel C workspace for golden-section-optimized REML/MLE.
No JAX dependency.
"""

from __future__ import annotations

import contextlib
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core.progress import create_progress_bar, progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask
from jamma.core.threading import (
    blas_threads,
    get_physical_core_count,
    is_blas_controllable,
)
from jamma.io.plink import (
    get_plink_metadata,
    stream_genotype_chunks,
    validate_genotype_values,
)
from jamma.jlinalg import compute_snp_stats_chunk
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _C_FUSED_AVAILABLE,
    _C_FUSED_GENERAL_AVAILABLE,
    _C_GENERAL_AVAILABLE,
    _C_LRT_FUSED_AVAILABLE,
    _C_LRT_FUSED_WS_AVAILABLE,
    _C_MODE4_AVAILABLE,
    _C_MODE4_FUSED_AVAILABLE,
    _C_SCORE_FUSED_AVAILABLE,
    _C_SCORE_FUSED_WS_AVAILABLE,
    _C_SPLIT_AVAILABLE,
    _compute_lmm_chunk_numpy,
    compute_mode4_fused_c_ws,
    compute_wald_fused_c_ws,
    compute_wald_fused_general_c_ws,
    create_lmm_workspace_fused,
    create_lmm_workspace_fused_general,
    create_lmm_workspace_mode4,
    create_lmm_workspace_mode4_fused,
)
from jamma.lmm.impute import impute_missing_inplace
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    _eigendecompose_or_reuse,
    compute_and_log_pve,
    validate_runner_inputs,
)
from jamma.lmm.results import (
    _build_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.runner_numpy import (
    _MIN_PIPELINE_CHUNKS,
    _compute_chunk_size_numpy,
    _create_wald_workspace_for_ncvt,
    _guarded_compute,
    compute_adaptive_core_split,
    compute_pipeline_core_split,
    dispatch_soa_split,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.schema import LazySnpMeta as _LazySnpMeta
from jamma.lmm.schema import LmmConfig, LmmRunResult, RunnerTiming
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory

# Module-level timing from the last run, for programmatic access by pipeline/benchmarks.
# Not thread-safe: concurrent calls will corrupt this dict.
# Cleared at function entry; repopulated at function exit on success.
# Use get_last_run_timing() for a safe snapshot copy.
last_run_timing: RunnerTiming = {}


def get_last_run_timing() -> RunnerTiming:
    """Return a snapshot copy of the last run's timing data.

    Safe to call from any thread -- returns an independent dict.
    """
    return dict(last_run_timing)


def run_lmm_association_numpy_streaming(
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None = None,
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
    check_memory: bool = True,
    show_progress: bool = True,
    output_path: Path | None = None,
    lmm_mode: int = 1,
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    validate_genotypes: bool = True,
    config: LmmConfig | None = None,
) -> tuple[LmmRunResult, int]:
    """Run LMM association tests by streaming genotypes from disk (NumPy/C path).

    Two-pass disk streaming: pass 1 computes SNP statistics for filtering,
    pass 2 runs C extension compute per chunk. Uses jlinalg.dgemm for
    eigenrotation and _lmm_accel C workspace for golden-section-optimized REML/MLE.

    Args:
        bed_path: PLINK file prefix (without .bed/.bim/.fam extension).
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues and eigenvectors are provided.
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
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        output_path: Path for incremental result writing, or None for in-memory.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        snps_indices: Pre-resolved column indices for -snps restriction, or None.
        hwe_threshold: HWE p-value threshold; SNPs with p < threshold are
            removed. 0.0 disables HWE filtering (default).
        validate_genotypes: Check for unexpected genotype values during pass-1.
        config: LmmConfig instance. When provided, overrides individual
            threshold/mode kwargs above.

    Returns:
        Tuple of (LmmRunResult, n_tested) where LmmRunResult contains
        associations (empty if output_path is set -- results on disk) and
        PVE from null model. n_tested is the number of SNPs that passed
        filtering and were tested.
    """
    # Unpack config if provided (config takes precedence over individual kwargs).
    if config is not None:
        kw = config.as_kwargs()
        maf_threshold = kw["maf_threshold"]
        miss_threshold = kw["miss_threshold"]
        l_min, l_max = kw["l_min"], kw["l_max"]
        n_grid, n_refine = kw["n_grid"], kw["n_refine"]
        check_memory = kw["check_memory"]
        show_progress, lmm_mode = kw["show_progress"], kw["lmm_mode"]

    start_time = time.perf_counter()

    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps = meta["n_snps"]

    if snp_info is None:
        snp_info = _LazySnpMeta(meta)

    # Validate inputs and apply sample filtering
    setup = validate_runner_inputs(
        phenotypes, kinship, covariates, eigenvalues, eigenvectors, lmm_mode
    )
    phenotypes = setup.phenotypes
    kinship = setup.kinship
    covariates = setup.covariates
    eigenvalues = setup.eigenvalues
    eigenvectors = setup.eigenvectors
    n_valid = setup.n_samples
    valid_mask = setup.valid_mask

    n_samples = phenotypes.shape[0]

    if show_progress:
        logger.info("Performing LMM Association Test (NumPy streaming)")
        logger.info(f"  Total individuals: {n_samples_total:,}")
        logger.info(f"  Analyzed individuals: {n_valid:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.info(f"  Lambda range: [{l_min:.2e}, {l_max:.2e}]")

    needs_sample_filter = not np.all(valid_mask)

    # === PASS 1: SNP statistics (single-pass C kernel) ===
    t_io_start = time.perf_counter()
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.intp)
    all_vars = np.zeros(n_snps, dtype=np.float64)

    # HWE genotype count accumulators (only when threshold > 0)
    if hwe_threshold > 0:
        all_n_aa = np.zeros(n_snps, dtype=np.int64)
        all_n_ab = np.zeros(n_snps, dtype=np.int64)
        all_n_bb = np.zeros(n_snps, dtype=np.int64)

    # Genotype validation accumulator
    n_unexpected_total = 0

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks_p1 = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks_p1, desc="Computing SNP statistics"
        )

    for chunk, start, end in stats_iterator:
        # Apply sample filtering
        if needs_sample_filter:
            chunk = chunk[valid_mask, :]

        # Single-pass SNP stats: mean, miss_count, variance, optional HWE counts
        chunk = np.ascontiguousarray(chunk)
        compute_snp_stats_chunk(
            chunk,
            all_means[start:end],
            all_miss_counts[start:end],
            all_vars[start:end],
            all_n_aa[start:end] if hwe_threshold > 0 else None,
            all_n_ab[start:end] if hwe_threshold > 0 else None,
            all_n_bb[start:end] if hwe_threshold > 0 else None,
        )

        if validate_genotypes:
            n_unexpected_total += validate_genotype_values(chunk)

    if validate_genotypes and n_unexpected_total > 0:
        logger.warning(
            f"Genotype validation: {n_unexpected_total} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    t_io_end = time.perf_counter()

    # === SNP statistics: filtering + stats construction ===
    t_snp_start = time.perf_counter()
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        all_means, all_miss_counts, all_vars, n_samples, maf_threshold, miss_threshold
    )
    del all_vars

    # Apply SNP list restriction (if -snps provided)
    if snps_indices is not None:
        from jamma.core.snp_filter import apply_snp_list_mask

        apply_snp_list_mask(snp_mask, snps_indices, n_snps, "SNP list filter")

    # Apply HWE filter (if -hwe threshold > 0)
    if hwe_threshold > 0:
        from jamma.core.snp_filter import compute_hwe_pvalues

        hwe_pvalues = compute_hwe_pvalues(all_n_aa, all_n_ab, all_n_bb)
        hwe_pass = hwe_pvalues >= hwe_threshold
        n_hwe_fail = int(np.sum(~hwe_pass & snp_mask))
        snp_mask &= hwe_pass
        logger.info(f"HWE filter: {n_hwe_fail} SNPs removed (p < {hwe_threshold})")

    snp_indices = np.where(snp_mask)[0]
    n_filtered = len(snp_indices)

    if show_progress:
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")

    if n_filtered == 0:
        if output_path is not None:
            with IncrementalAssocWriter(
                output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            ):
                pass  # Context manager writes header, no data rows
        if show_progress:
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"LMM Association completed in {elapsed:.2f}s (no SNPs passed filter)"
            )
        return LmmRunResult(associations=[]), 0

    filtered_afs = allele_freqs[snp_indices]
    filtered_miss = all_miss_counts[snp_indices].astype(int)
    del all_miss_counts, allele_freqs
    filtered_means = all_means[snp_indices]
    del all_means
    if hwe_threshold > 0:
        del all_n_aa, all_n_ab, all_n_bb

    t_snp_end = time.perf_counter()

    # === Eigendecomp + rotation + null model ===
    t_eigen_start = time.perf_counter()

    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship,
        eigenvalues,
        eigenvectors,
        show_progress,
        "lmm_numpy_streaming",
        check_memory=check_memory,
    )
    if kinship is not None:
        del kinship
    gc.collect()

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Use all physical cores for BLAS rotation
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = U.T @ W
        Uty = U.T @ phenotypes

    # Null model for Score/LRT/All
    logl_H0, lambda_null_mle, Hi_eval_null = _compute_null_model_common(
        lmm_mode,
        eigenvalues_np,
        UtW,
        Uty,
        n_cvt,
        show_progress,
        l_min=l_min,
        l_max=l_max,
    )

    t_eigen_end = time.perf_counter()

    pve, pve_se = compute_and_log_pve(eigenvalues_np, UtW, Uty, n_cvt, l_min, l_max)

    # === C workspace creation ===
    # Enforce minimum 20 golden section iterations for ~1e-5 lambda tolerance
    n_refine = max(n_refine, 20)

    # Determine split/pipeline eligibility
    use_split = (_C_SPLIT_AVAILABLE and n_cvt == 1) or (
        _C_GENERAL_AVAILABLE and n_cvt > 1
    )
    use_fused_mode4 = use_split and lmm_mode == 4 and n_cvt == 1 and _C_MODE4_AVAILABLE

    # Fused Uab path: skip uab_varying_soa, pass utg_t directly to C workspace.
    # Requires use_split (workspace + invariant SoA infrastructure).
    use_fused = use_split and (
        # n_cvt=1 fast path (existing)
        (
            n_cvt == 1
            and _C_FUSED_AVAILABLE
            and (lmm_mode == 1 or (lmm_mode == 4 and _C_MODE4_FUSED_AVAILABLE))
        )
        or
        # General n_cvt path: Wald-only (mode 1).
        # Mode-4 fused general LRT has a known bug producing NaN lambda_mle;
        # mode-4 n_cvt>=2 falls back to compose (Wald workspace + batch Score/LRT).
        (n_cvt >= 2 and _C_FUSED_GENERAL_AVAILABLE and lmm_mode == 1)
    )
    use_fused_general = use_fused and n_cvt >= 2

    # Fused Score/LRT: skip uab_varying_soa for modes 2/3 (n_cvt=1 only).
    # Prefer workspace-based path (persistent across chunks, eliminates per-chunk
    # malloc/free and redundant precomputation); fall back to stateless if unavailable.
    use_fused_score_ws = (
        use_split and n_cvt == 1 and lmm_mode == 3 and _C_SCORE_FUSED_WS_AVAILABLE
    )
    use_fused_lrt_ws = (
        use_split and n_cvt == 1 and lmm_mode == 2 and _C_LRT_FUSED_WS_AVAILABLE
    )
    # Stateless fallback: only when WS not available
    use_fused_score = (
        use_split
        and n_cvt == 1
        and lmm_mode == 3
        and _C_SCORE_FUSED_AVAILABLE
        and not use_fused_score_ws
    )
    use_fused_lrt = (
        use_split
        and n_cvt == 1
        and lmm_mode == 2
        and _C_LRT_FUSED_AVAILABLE
        and not use_fused_lrt_ws
    )

    if use_fused_score_ws:
        logger.debug(
            "Fused Score workspace path active: workspace created once, "
            "utg_t passed per-chunk (eliminates per-chunk malloc, streaming)"
        )
    elif use_fused_score:
        logger.debug(
            "Fused Score path active: utg_t passed directly to C "
            "(eliminates uab_varying_soa buffer for mode 3, streaming)"
        )
    if use_fused_lrt_ws:
        logger.debug(
            "Fused LRT workspace path active: workspace created once, "
            "utg_t passed per-chunk "
            "(eliminates per-chunk malloc/grid precompute, streaming)"
        )
    elif use_fused_lrt:
        logger.debug(
            "Fused LRT path active: utg_t passed directly to C "
            "(eliminates uab_varying_soa buffer for mode 2, streaming)"
        )

    # Auto-scale chunk_size from RAM when user provided default (10_000).
    # After pass-1 we know n_filtered and can compute optimal chunk size.
    auto_scaled = chunk_size == 10_000
    if auto_scaled:
        chunk_size = _compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt,
            use_split=use_split,
            lmm_mode=lmm_mode,
            fused_mode4=use_fused_mode4,
            use_fused_general=use_fused_general,
        )

    n_chunks = (n_filtered + chunk_size - 1) // chunk_size

    # Pipeline overlaps rotation(N+1) with compute(N) using an adaptive core
    # split. Only enable when enough chunks exist for overlap to matter.
    use_pipeline = use_split and n_chunks >= _MIN_PIPELINE_CHUNKS

    if use_pipeline:
        # Pipeline keeps 2 chunks alive simultaneously -- halve the memory
        # budget to avoid OOM.
        if auto_scaled:
            chunk_size = _compute_chunk_size_numpy(
                n_samples,
                n_filtered,
                n_cvt,
                use_split=use_split,
                lmm_mode=lmm_mode,
                fused_mode4=use_fused_mode4,
                use_fused_general=use_fused_general,
                pipeline_buffers=2,
            )
        else:
            # User-specified chunk_size: halve it to account for double-
            # buffering rather than recomputing from RAM (which would
            # discard the user's sizing intent).
            chunk_size = max(1, chunk_size // 2)
        n_chunks = (n_filtered + chunk_size - 1) // chunk_size
        use_pipeline = use_split and n_chunks >= _MIN_PIPELINE_CHUNKS

    if use_pipeline:
        logger.debug(f"Pipeline mode: overlapping rotation/compute ({n_chunks} chunks)")

    # OpenMP thread count
    if _C_ACCEL_AVAILABLE:
        cores = get_physical_core_count()
        omp_threads = max(1, cores // 2) if not is_blas_controllable() else cores
    else:
        omp_threads = 1

    # Pipeline thread budget: partition physical cores between concurrent
    # BLAS rotation (background) and C extension compute (foreground).
    if use_pipeline:
        total_cores = get_physical_core_count()
        if is_blas_controllable():
            pipeline_rot_threads, pipeline_omp_threads = compute_pipeline_core_split(
                n_samples, total_cores
            )
            logger.debug(
                f"Pipeline core split: {pipeline_rot_threads} rotation, "
                f"{pipeline_omp_threads} compute (n_samples={n_samples:,})"
            )
        else:
            pipeline_omp_threads = max(1, total_cores // 2)
            pipeline_rot_threads = total_cores
    else:
        pipeline_omp_threads = omp_threads
        pipeline_rot_threads = rotation_threads

    # Precompute SNP-invariant Uab columns once
    uab_invariant_soa = (
        compute_uab_invariant_soa(UtW, Uty, n_cvt) if use_split else None
    )

    # Extract w for fused Score/LRT paths (stateless and workspace).
    w = (
        UtW[:, 0].copy()
        if (
            use_fused_score
            or use_fused_lrt
            or use_fused_score_ws
            or use_fused_lrt_ws
        )
        and not use_fused
        else None
    )

    # Create persistent C workspace (fused or split)
    if use_split and lmm_mode in (1, 4):
        if use_fused_general:
            # Fused general Wald workspace: UtW + Uty for on-the-fly dot products
            from jamma.lmm.likelihood import build_pab_table_for_c

            pab_c = build_pab_table_for_c(n_cvt)
            lmm_workspace = create_lmm_workspace_fused_general(
                eigenvalues_np,
                uab_invariant_soa,
                UtW,
                Uty,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                pipeline_omp_threads,
                n_cvt=n_cvt,
                **{
                    k: pab_c[k]
                    for k in [
                        "invariant_indices",
                        "varying_indices",
                        "logdet_diag_rows",
                        "logdet_diag_cols",
                        "level_offsets",
                        "level_counts",
                        "entries",
                        "idx_xx",
                        "idx_xy",
                        "idx_yy",
                        "var_a_cols",
                        "var_b_cols",
                    ]
                },
            )
        elif use_fused and lmm_mode == 4:
            w = UtW[:, 0].copy()
            lmm_workspace = create_lmm_workspace_mode4_fused(
                eigenvalues_np,
                uab_invariant_soa,
                w,
                Uty,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                pipeline_omp_threads,
                hi_eval_null=Hi_eval_null,
                logl_H0=logl_H0,
            )
        elif use_fused:
            w = UtW[:, 0].copy()
            lmm_workspace = create_lmm_workspace_fused(
                eigenvalues_np,
                uab_invariant_soa,
                w,
                Uty,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                pipeline_omp_threads,
            )
        elif use_fused_mode4:
            lmm_workspace = create_lmm_workspace_mode4(
                eigenvalues_np,
                uab_invariant_soa,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                pipeline_omp_threads,
                Hi_eval_null,
                logl_H0,
            )
        else:
            lmm_workspace = _create_wald_workspace_for_ncvt(
                n_cvt,
                eigenvalues_np,
                uab_invariant_soa,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                pipeline_omp_threads,
            )
    else:
        lmm_workspace = None

    # Create Score/LRT workspaces (persistent across all chunks).
    score_fused_workspace = None
    lrt_fused_workspace = None

    if use_fused_score_ws:
        from jamma.lmm.compute_numpy import _create_workspace_score_fused_c

        score_fused_workspace = _create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues_np,
            uab_invariant_soa,
            n_samples,
            pipeline_omp_threads,
        )

    if use_fused_lrt_ws:
        from jamma.lmm.compute_numpy import _create_workspace_lrt_fused_c

        lrt_fused_workspace = _create_workspace_lrt_fused_c(
            w,
            Uty,
            eigenvalues_np,
            uab_invariant_soa,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            logl_H0,
            pipeline_omp_threads,
        )

    # === Timing accumulators ===
    last_run_timing.clear()
    t_rotation_total = 0.0
    t_numpy_compute_total = 0.0
    t_result_write_total = 0.0

    # Diagnostic accumulators
    nan_counts: dict[str, int] = {}
    n_at_lmin = 0
    n_at_lmax = 0

    all_results: list[AssocResult] = []

    # === PASS 2: Compute per chunk (float64) ===
    from jamma import jlinalg

    with contextlib.ExitStack() as stack:
        writer = None
        if output_path is not None:
            writer = stack.enter_context(
                IncrementalAssocWriter(output_path, test_type=_TEST_TYPE_MAP[lmm_mode])
            )

        chunk_iter = iter(
            stream_genotype_chunks(
                bed_path,
                chunk_size=chunk_size,
                dtype=np.float64,
                show_progress=False,
                snp_indices=snp_indices,
            )
        )

        write_offset = 0  # Cumulative SNP offset for error diagnostics

        # --- Preallocate per-chunk buffers to eliminate malloc/free ---
        # Streaming chunks come from disk (can't preallocate geno), but
        # utg_t output is always (actual, n_samples) and can be written
        # into a preallocated buffer.
        if use_pipeline:
            _utg_bufs = [
                np.empty((chunk_size, n_samples), dtype=np.float64),
                np.empty((chunk_size, n_samples), dtype=np.float64),
            ]
        else:
            _utg_bufs = [np.empty((chunk_size, n_samples), dtype=np.float64)]
        _stream_chunk_counter = 0

        # Varying SoA buffer reuse for non-fused split path.
        _no_fused_stream = (
            not use_fused
            and not use_fused_score
            and not use_fused_lrt
            and not use_fused_score_ws
            and not use_fused_lrt_ws
        )
        if use_split and _no_fused_stream:
            from jamma.lmm.likelihood import classify_uab_columns

            _n_var = 3 if n_cvt == 1 else len(classify_uab_columns(n_cvt)[1])
            _uab_var_buf_stream = np.empty(
                (chunk_size, _n_var, n_samples), dtype=np.float64
            )
        else:
            _uab_var_buf_stream = None

        def _prepare_chunk() -> tuple | None:
            """Read next chunk from disk, filter, impute, rotate.

            Returns (data, filt_start, filt_end, actual_len) or None at
            exhaustion. The dgemm rotation releases the GIL, enabling
            true parallelism with _compute_and_write's C extension
            compute in the pipeline path. Pre-rotation NumPy work
            (filtering, imputation) is lightweight and GIL-serialized.

            Uses preallocated utg_buf to avoid per-chunk dgemm allocation.
            Pipeline path double-buffers via _stream_chunk_counter % 2.
            """
            nonlocal _stream_chunk_counter

            try:
                chunk, filt_start, filt_end = next(chunk_iter)
            except StopIteration:
                return None

            if filt_end <= filt_start:
                return (None, filt_start, filt_end, 0)

            # Apply sample filtering
            if needs_sample_filter:
                chunk = chunk[valid_mask, :]

            # Mean-impute NaN
            impute_missing_inplace(chunk, filtered_means[filt_start:filt_end])

            actual_len = filt_end - filt_start
            buf_idx = _stream_chunk_counter % len(_utg_bufs)
            _stream_chunk_counter += 1

            # Rotate — control both external BLAS (via threadpoolctl) and
            # jlinalg-own dgemm (via set_n_threads) to avoid oversubscription
            # when pipeline overlaps rotation with compute.
            old_jl_threads = jlinalg.set_n_threads(pipeline_rot_threads)
            try:
                with blas_threads(pipeline_rot_threads):
                    if (
                        use_fused
                        or use_fused_score
                        or use_fused_lrt
                        or use_fused_score_ws
                        or use_fused_lrt_ws
                        or use_split
                    ):
                        utg_out = _utg_bufs[buf_idx][:actual_len, :]
                        utg_t = jlinalg.dgemm(
                            chunk, U, transa="T", out=utg_out
                        )
                    else:
                        # Non-split UtG path: (n_samples, actual_len) shape
                        # doesn't match utg_buf layout — allocate fresh.
                        UtG = jlinalg.dgemm(U, chunk, transa="T")
                        utg_t = None
            finally:
                jlinalg.set_n_threads(old_jl_threads)
            del chunk

            if (
                use_fused
                or use_fused_score
                or use_fused_lrt
                or use_fused_score_ws
                or use_fused_lrt_ws
            ):
                return (utg_t, filt_start, filt_end, actual_len)

            if use_split:
                # Reuse preallocated buffer when chunk is full-sized.
                out_var = (
                    _uab_var_buf_stream[:actual_len, :, :]
                    if _uab_var_buf_stream is not None
                    and actual_len == chunk_size
                    else None
                )
                uab_var_soa = batch_compute_uab_varying_soa_numpy(
                    n_cvt, UtW, Uty, utg_t, out=out_var
                )
                del utg_t
                return (uab_var_soa, filt_start, filt_end, actual_len)

            # Non-split full Uab
            Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
            return (Uab_batch, filt_start, filt_end, actual_len)

        def _compute_and_write(prepared: tuple) -> None:
            """Run C extension compute on prepared chunk data and write results.

            Dispatches by path (fused / split SoA / full Uab) and accumulates
            diagnostics and timing.
            """
            nonlocal write_offset, t_numpy_compute_total, t_result_write_total
            nonlocal n_at_lmin, n_at_lmax

            chunk_data, filt_start, filt_end, actual_len = prepared
            if chunk_data is None:
                return

            t_compute_start = time.perf_counter()

            if use_fused:
                if use_fused_general:
                    if lmm_mode != 1:
                        raise ValueError(
                            "Mode-4 fused general disabled (NaN lambda_mle bug)"
                        )
                    fused_fn = compute_wald_fused_general_c_ws
                    op_label = "Fused general Uab dispatch (streaming)"
                else:
                    fused_fn = (
                        compute_mode4_fused_c_ws
                        if lmm_mode == 4
                        else compute_wald_fused_c_ws
                    )
                    op_label = "Fused Uab dispatch (streaming)"
                with blas_threads(1):
                    cr = _guarded_compute(
                        fused_fn,
                        lmm_workspace,
                        chunk_data,
                        pipeline_omp_threads,
                        operation=op_label,
                        write_offset=write_offset,
                        n_filtered=n_filtered,
                    )
            elif use_fused_score_ws:
                from jamma.lmm.compute_numpy import _compute_score_fused_ws_c

                with blas_threads(1):
                    cr = _guarded_compute(
                        _compute_score_fused_ws_c,
                        score_fused_workspace,
                        chunk_data,  # utg_t
                        pipeline_omp_threads,
                        operation="Fused Score WS dispatch (streaming)",
                        write_offset=write_offset,
                        n_filtered=n_filtered,
                    )
            elif use_fused_lrt_ws:
                from jamma.lmm.compute_numpy import _compute_lrt_fused_ws_c

                with blas_threads(1):
                    cr = _guarded_compute(
                        _compute_lrt_fused_ws_c,
                        lrt_fused_workspace,
                        chunk_data,  # utg_t
                        pipeline_omp_threads,
                        operation="Fused LRT WS dispatch (streaming)",
                        write_offset=write_offset,
                        n_filtered=n_filtered,
                    )
            elif use_fused_score:
                from jamma.lmm.compute_numpy import _compute_score_fused_c

                with blas_threads(1):
                    cr = _guarded_compute(
                        _compute_score_fused_c,
                        chunk_data,  # utg_t
                        w,
                        Uty,
                        Hi_eval_null,
                        uab_invariant_soa,
                        eigenvalues_np,
                        n_samples,
                        pipeline_omp_threads,
                        operation="Fused Score dispatch (streaming)",
                        write_offset=write_offset,
                        n_filtered=n_filtered,
                    )
            elif use_fused_lrt:
                from jamma.lmm.compute_numpy import _compute_lrt_fused_c

                with blas_threads(1):
                    cr = _guarded_compute(
                        _compute_lrt_fused_c,
                        chunk_data,  # utg_t
                        w,
                        Uty,
                        eigenvalues_np,
                        uab_invariant_soa,
                        n_samples,
                        l_min,
                        l_max,
                        n_grid,
                        n_refine,
                        logl_H0,
                        pipeline_omp_threads,
                        operation="Fused LRT dispatch (streaming)",
                        write_offset=write_offset,
                        n_filtered=n_filtered,
                    )
            elif use_split:
                with blas_threads(1):
                    cr = dispatch_soa_split(
                        lmm_mode,
                        use_fused_mode4,
                        lmm_workspace,
                        n_cvt,
                        eigenvalues_np,
                        chunk_data,
                        uab_invariant_soa,
                        n_samples,
                        Hi_eval_null=Hi_eval_null,
                        l_min=l_min,
                        l_max=l_max,
                        n_grid=n_grid,
                        n_refine=n_refine,
                        logl_H0=logl_H0,
                        n_threads=pipeline_omp_threads,
                    )
            else:
                cr = _compute_lmm_chunk_numpy(
                    lmm_mode,
                    n_cvt,
                    eigenvalues_np,
                    chunk_data,
                    n_samples,
                    l_min=l_min,
                    l_max=l_max,
                    n_grid=n_grid,
                    n_refine=n_refine,
                    Hi_eval_null=Hi_eval_null,
                    logl_H0=logl_H0,
                    n_threads=pipeline_omp_threads,
                )

            t_numpy_compute_total += time.perf_counter() - t_compute_start

            # Build result arrays and accumulate diagnostics
            t_write_start = time.perf_counter()
            chunk_arrays = {
                key: cr[key][:actual_len] for key in _RESULT_FIELDS[lmm_mode]
            }

            chunk_lmin, chunk_lmax = count_lambda_boundary_hits(
                lmm_mode, chunk_arrays, l_min, l_max
            )
            n_at_lmin += chunk_lmin
            n_at_lmax += chunk_lmax

            for key, arr in chunk_arrays.items():
                if arr.dtype.kind != "f":
                    continue
                n_nan = int(np.count_nonzero(np.isnan(arr)))
                if n_nan > 0:
                    nan_counts[key] = nan_counts.get(key, 0) + n_nan

            if writer is not None:
                writer.write_arrays_batch(
                    lmm_mode,
                    snp_indices[filt_start:filt_end],
                    snp_info,
                    filtered_afs[filt_start:filt_end],
                    filtered_miss[filt_start:filt_end],
                    chunk_arrays,
                )
            else:
                chunk_results = _build_results(
                    lmm_mode,
                    snp_indices[filt_start:filt_end],
                    filtered_afs[filt_start:filt_end],
                    filtered_miss[filt_start:filt_end],
                    snp_info,
                    chunk_arrays,
                )
                all_results.extend(chunk_results)

            write_offset += actual_len
            t_result_write_total += time.perf_counter() - t_write_start

        if use_pipeline:
            # --- Profile first chunk for adaptive core split ---
            t_rot_start = time.perf_counter()
            first = _prepare_chunk()
            t_first_rot = time.perf_counter() - t_rot_start
            t_rotation_total += t_first_rot

            if first is not None:
                t_compute_start = time.perf_counter()
                _compute_and_write(first)
                t_first_compute = time.perf_counter() - t_compute_start
                del first

                # Adaptive core split from measured times. Closures
                # _prepare_chunk and _compute_and_write look up these
                # variables in the enclosing scope, so reassignment
                # here takes effect on subsequent calls.
                if n_chunks > 2 and is_blas_controllable():
                    old_rot, old_omp = pipeline_rot_threads, pipeline_omp_threads
                    pipeline_rot_threads, pipeline_omp_threads = (
                        compute_adaptive_core_split(
                            t_first_rot,
                            t_first_compute,
                            total_cores,
                            n_samples=n_samples,
                        )
                    )
                    if (pipeline_rot_threads, pipeline_omp_threads) != (
                        old_rot,
                        old_omp,
                    ):
                        logger.debug(
                            f"Adaptive core split: {old_rot}/{old_omp} -> "
                            f"{pipeline_rot_threads}/{pipeline_omp_threads} "
                            f"(rot={t_first_rot:.3f}s, "
                            f"compute={t_first_compute:.3f}s)"
                        )

                # Seed pipeline with next chunk
                t_rot_start = time.perf_counter()
                current = _prepare_chunk()
                t_rotation_total += time.perf_counter() - t_rot_start

                # Progress bar (manual update, not iterator-based)
                pipeline_bar = (
                    create_progress_bar(n_chunks, "LMM association (streaming)")
                    if show_progress and n_chunks > 1
                    else None
                )
                if pipeline_bar is not None:
                    pipeline_bar.update(2)

                # Pipeline loop: overlap rotation(N+1) with compute(N)
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        for i_chunk in range(2, n_chunks):
                            if current is None:
                                break
                            chunk_data_tuple = current

                            future = executor.submit(_prepare_chunk)

                            _compute_and_write(chunk_data_tuple)

                            t_rot_start = time.perf_counter()
                            try:
                                current = future.result()
                            except (
                                MemoryError,
                                ValueError,
                                TypeError,
                                OverflowError,
                                OSError,
                            ):
                                raise
                            except Exception as exc:
                                raise RuntimeError(
                                    f"Pipeline chunk preparation failed at offset "
                                    f"{write_offset}/{n_filtered}: {exc}"
                                ) from exc
                            t_rotation_total += time.perf_counter() - t_rot_start

                            if pipeline_bar is not None:
                                pipeline_bar.update(i_chunk + 1)

                        # Last chunk
                        if current is not None:
                            _compute_and_write(current)
                finally:
                    if pipeline_bar is not None:
                        try:
                            pipeline_bar.update(n_chunks)
                            pipeline_bar.finish()
                        except Exception:
                            pass  # Don't mask the real exception
        else:
            # Sequential fallback when too few chunks for pipeline overlap
            seq_bar = (
                create_progress_bar(n_chunks, "Running LMM (NumPy streaming)")
                if show_progress and n_chunks > 1
                else None
            )

            try:
                i_seq = 0
                while True:
                    t_rot_start = time.perf_counter()
                    prepared = _prepare_chunk()
                    t_rotation_total += time.perf_counter() - t_rot_start
                    if prepared is None:
                        break
                    _compute_and_write(prepared)
                    i_seq += 1
                    if seq_bar is not None:
                        seq_bar.update(i_seq)
            finally:
                if seq_bar is not None:
                    try:
                        seq_bar.finish()
                    except Exception:
                        pass  # Don't mask the real exception

        if write_offset < n_filtered:
            logger.warning(
                f"Processed {write_offset}/{n_filtered} SNPs -- "
                f"stream exhausted early (expected {n_filtered})"
            )

        # === Post-loop diagnostics ===
        if show_progress:
            log_rss_memory("lmm_numpy_streaming", "after_association")

        for key, n_nan in nan_counts.items():
            logger.warning(
                f"{n_nan}/{n_filtered} SNPs have NaN {key} -- "
                "check for degenerate (constant) genotypes "
                "and kinship matrix quality"
            )

        log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

        if show_progress:
            elapsed = time.perf_counter() - start_time
            t_io = t_io_end - t_io_start
            t_snp = t_snp_end - t_snp_start
            t_eigen = t_eigen_end - t_eigen_start
            accounted = (
                t_io
                + t_snp
                + t_eigen
                + t_rotation_total
                + t_numpy_compute_total
                + t_result_write_total
            )
            logger.info("Timing breakdown:")
            logger.info(f"  I/O read (pass 1):   {t_io:.2f}s")
            logger.info(f"  SNP statistics:      {t_snp:.2f}s")
            logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
            logger.info(f"  UT@G rotation:       {t_rotation_total:.2f}s")
            logger.info(f"  NumPy compute:       {t_numpy_compute_total:.2f}s")
            logger.info(f"  Result write:        {t_result_write_total:.2f}s")
            logger.info("  ----")
            logger.info(f"  Accounted:           {accounted:.2f}s")
            logger.info(f"  Total:               {elapsed:.2f}s")

        if writer is not None and show_progress:
            logger.info(f"Wrote {writer.count:,} results to {output_path}")

        if show_progress:
            elapsed = time.perf_counter() - start_time
            logger.info(f"LMM Association completed in {elapsed:.2f}s")

        last_run_timing.clear()
        last_run_timing.update(
            {
                "rotation_s": t_rotation_total,
                "numpy_compute_s": t_numpy_compute_total,
                "result_write_s": t_result_write_total,
            }
        )

        n_tested = writer.count if writer is not None else len(all_results)
        return (
            LmmRunResult(
                associations=[] if output_path is not None else all_results,
                pve=pve,
                pve_se=pve_se,
                n_tested=n_tested if output_path is not None else None,
            ),
            n_tested,
        )
