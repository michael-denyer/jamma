"""Pure-NumPy batch LMM association runner.

No JAX dependency. Input genotypes must fit in memory.
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import psutil
from loguru import logger

from jamma.core.constants import PHENOTYPE_MISSING
from jamma.core.memory import estimate_lmm_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.core.threading import (
    blas_threads,
    get_physical_core_count,
    is_blas_controllable,
)
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _C_HAS_OPENMP,
    _C_SPLIT_AVAILABLE,
    LmmMode,
    _compute_lmm_chunk_numpy,
    compute_wald_split_c_ws,
    create_lmm_workspace,
)
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
    reconstruct_uab_from_soa,
    reset_p_yy_warned,
)
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    _eigendecompose_or_reuse,
)
from jamma.lmm.results import (
    _build_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory

# NumPy has no int32 buffer constraint — allow larger chunks than JAX runner.
_MAX_CHUNK = 200_000

# Memory budget bounds for auto-scaling
_MIN_BUDGET = 2_000_000_000  # 2 GB floor (original default)
_MAX_BUDGET = 40_000_000_000  # 40 GB ceiling

# Minimum number of chunks before pipelined execution is worthwhile.
_MIN_PIPELINE_CHUNKS = 30


def compute_pipeline_core_split(n_samples: int, total_cores: int) -> tuple[int, int]:
    """Compute rotation/compute thread split for the pipeline path.

    DGEMM rotation scales with n_samples^2 * chunk_size while per-SNP
    compute scales with chunk_size * (n_grid + n_refine). For large
    n_samples rotation dominates; for small n_samples compute dominates.

    Args:
        n_samples: Number of samples in the dataset.
        total_cores: Physical core count available.

    Returns:
        (rotation_threads, compute_threads) tuple. Both >= 1.
    """
    if n_samples > 10_000:
        rot = max(1, total_cores // 2)
    elif n_samples > 1_000:
        rot = max(1, total_cores // 3)
    else:
        rot = max(1, total_cores // 4)
    return rot, max(1, total_cores - rot)


def _compute_chunk_size_numpy(
    n_samples: int,
    n_filtered: int,
    n_cvt: int = 1,
    *,
    use_split: bool = False,
    lmm_mode: int = 1,
    mem_budget_bytes: int | None = None,
    pipeline_buffers: int = 1,
) -> int:
    """Compute chunk size based on RAM budget (no int32 constraint for NumPy).

    Scales the memory budget with available RAM to minimise DRAM passes
    through the eigenvector matrix during UT@G rotation.

    Args:
        n_samples: Number of samples.
        n_filtered: Number of filtered SNPs.
        n_cvt: Number of covariates.
        use_split: If True, use split Uab accounting instead of full Uab.
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All). Affects
            memory accounting: Wald uses 4 cols/SNP (3 varying + 1 UtG),
            non-Wald uses 9 cols/SNP (3 varying + 6 reconstructed Uab peak).
        mem_budget_bytes: Explicit per-chunk memory budget in bytes.
            None (default) auto-scales with available RAM.
        pipeline_buffers: Number of live chunks (1 for sequential,
            2 for pipeline double-buffering). Divides the budget.

    Returns:
        Chunk size (number of SNPs per chunk).
    """
    if use_split and n_cvt == 1:
        if lmm_mode == 1:
            # Wald split path: 3 varying columns + 1 UtG column per SNP
            bytes_per_snp = n_samples * 4 * 8
        else:
            # Non-Wald split: reconstruct_uab_from_soa allocates 6-col Uab
            # while 3-col varying SoA is still live = 9 cols peak
            bytes_per_snp = n_samples * 9 * 8
    else:
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        bytes_per_snp = n_samples * n_index * 8

    if bytes_per_snp == 0:
        return n_filtered

    if mem_budget_bytes is not None:
        mem_budget = mem_budget_bytes
    else:
        available = psutil.virtual_memory().available
        # Budget: 15% of available RAM (up from 5%), 2 GB floor, 40 GB ceiling.
        # Modern machines (128-512 GB) can afford larger working sets. The floor
        # prevents degenerate chunk sizes on low-memory systems; the ceiling
        # prevents excessive allocation on high-memory systems.
        mem_budget = max(_MIN_BUDGET, min(int(available * 0.15), _MAX_BUDGET))

    mem_budget = mem_budget // max(1, pipeline_buffers)

    chunk_from_memory = int(mem_budget / bytes_per_snp)
    chunk = max(100, min(chunk_from_memory, n_filtered, _MAX_CHUNK))
    return chunk


def run_lmm_association_numpy(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    snp_info: list,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    use_gpu: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
    lmm_mode: LmmMode = 1,
) -> list[AssocResult]:
    """Run LMM association tests using pure-NumPy batch processing.

    Processes SNPs in memory-bounded chunks using BLAS-backed NumPy operations.
    No JAX dependency. Input genotypes must fit in memory; for disk streaming
    use run_lmm_association_streaming.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples). WARNING: may be
            overwritten in-place during eigendecomposition (buffer reused for
            eigenvectors). Treat as consumed; pass kinship.copy() if you need
            the original matrix after this call.
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
        use_gpu: Accepted but silently ignored — NumPy backend is CPU-only.
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.

    Returns:
        List of AssocResult for each SNP that passes filtering.

    Raises:
        MemoryError: If check_memory=True and insufficient memory.
        ValueError: If only one of eigenvalues/eigenvectors is provided,
            or if no valid samples remain after filtering.
    """
    # Reset per-run warning flags so each run gets its own diagnostics
    reset_p_yy_warned()

    # Validate eigendecomposition params - must provide both or neither
    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )

    if kinship is None and eigenvalues is None:
        raise ValueError(
            "Either kinship or pre-computed eigendecomposition (eigenvalues + "
            "eigenvectors) must be provided"
        )

    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    if use_gpu:
        logger.warning(
            "use_gpu=True ignored: NumPy backend is CPU-only. "
            "Install JAX for GPU support: pip install jamma[jax]"
        )

    # Memory check before workflow
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
        est = estimate_lmm_memory(n_samples, n_snps)
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

    valid_mask = ~np.isnan(phenotypes) & (phenotypes != PHENOTYPE_MISSING)
    if covariates is not None:
        valid_covariate = np.all(~np.isnan(covariates), axis=1)
        valid_mask = valid_mask & valid_covariate
    if not np.all(valid_mask):
        genotypes = genotypes[valid_mask, :]
        phenotypes = phenotypes[valid_mask]
        if kinship is not None:
            kinship = kinship[np.ix_(valid_mask, valid_mask)]
        if covariates is not None:
            covariates = covariates[valid_mask, :]

    n_samples, n_snps = genotypes.shape
    if n_samples == 0:
        raise ValueError(
            "No valid samples: all phenotypes are missing or -9"
            + (", or all have missing covariates" if covariates is not None else "")
        )

    # Validate precomputed eigenpair dimensions against (possibly filtered) n_samples
    if eigenvalues is not None and eigenvectors is not None:
        hint = (
            "Recompute eigenpairs on the filtered kinship, or pass kinship= "
            "and let JAMMA compute the eigendecomposition."
        )
        if eigenvalues.shape[0] != n_samples:
            raise ValueError(
                f"eigenvalues length ({eigenvalues.shape[0]}) does not match "
                f"n_samples ({n_samples}) after removing missing "
                f"phenotypes/covariates. {hint}"
            )
        if eigenvectors.shape != (n_samples, n_samples):
            raise ValueError(
                f"eigenvectors shape {eigenvectors.shape} does not match "
                f"({n_samples}, {n_samples}) after removing missing "
                f"phenotypes/covariates. {hint}"
            )

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
        return []

    # Extract filtered stats as numpy arrays (use allele_freqs for output, not mafs)
    filtered_afs = allele_freqs[snp_indices]
    filtered_miss = missing_counts[snp_indices].astype(int)

    t_eigen_start = time.perf_counter()
    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship,
        eigenvalues,
        eigenvectors,
        show_progress,
        "lmm_numpy",
        check_memory=check_memory,
    )
    del kinship
    gc.collect()

    # Use all physical cores for BLAS rotation
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = U.T @ W
        Uty = U.T @ phenotypes

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

    n_filtered = len(snp_indices)

    # Determine split/pipeline eligibility BEFORE chunk sizing so the
    # budget can use accurate per-SNP accounting (3 Uab columns vs 6).
    use_split = _C_SPLIT_AVAILABLE and n_cvt == 1

    chunk_size = _compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt,
        use_split=use_split,
        lmm_mode=lmm_mode,
    )
    n_chunks = (n_filtered + chunk_size - 1) // chunk_size

    # Pipeline overlaps rotation(N+1) with compute(N) using an adaptive core
    # split (see compute_pipeline_core_split). This helps when rotation ≈
    # compute per chunk (many small chunks). With large chunks (few passes
    # through U), rotation >> compute per chunk, so the pipeline overlap
    # hides nothing but costs reduced BLAS throughput. Only enable when
    # enough chunks exist for overlap to matter.
    use_pipeline = use_split and n_chunks >= _MIN_PIPELINE_CHUNKS and lmm_mode == 1

    if use_pipeline:
        # Pipeline has 2 chunks alive simultaneously — halve the budget
        chunk_size = _compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt,
            use_split=use_split,
            lmm_mode=lmm_mode,
            pipeline_buffers=2,
        )
        n_chunks = (n_filtered + chunk_size - 1) // chunk_size
        use_pipeline = use_split and n_chunks >= _MIN_PIPELINE_CHUNKS and lmm_mode == 1

    # OpenMP thread count for C extension (set once, reused per chunk).
    # When C extension is active, BLAS threads are set to 1 inside the compute
    # phase to prevent oversubscription between BLAS and OpenMP.
    # On macOS/Accelerate, blas_threads(1) is a no-op — Accelerate keeps using
    # all cores — so we halve OpenMP threads to share cores with BLAS.
    if _C_ACCEL_AVAILABLE:
        cores = get_physical_core_count()
        omp_threads = max(1, cores // 2) if not is_blas_controllable() else cores
    else:
        omp_threads = 1

    if show_progress:
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"  Processing in {n_chunks} chunks ({chunk_size:,} SNPs/chunk)"
            )
        if _C_ACCEL_AVAILABLE:
            if _C_HAS_OPENMP:
                logger.info(f"  C extension active ({omp_threads} OpenMP threads)")
            else:
                logger.info("  C extension active (no OpenMP)")

    # Pre-allocate result arrays driven by _RESULT_FIELDS mapping
    write_offset = 0
    arrays_out: dict[str, np.ndarray] = {
        key: np.empty(n_filtered, dtype=np.float64) for key in _RESULT_FIELDS[lmm_mode]
    }

    # Timing accumulators for per-chunk phases
    t_rotation_total = 0.0
    t_numpy_compute_total = 0.0
    t_result_write_total = 0.0

    chunk_starts = list(range(0, n_filtered, chunk_size))

    # Pre-allocate rotation output buffer for sequential path only.
    # Pipeline path allocates per-chunk buffers because the background thread
    # writes UtG for chunk N+1 while the foreground thread still reads chunk N.
    UtG_buf = (
        None if use_pipeline else np.empty((n_samples, chunk_size), dtype=np.float64)
    )

    # Pipeline thread budget: partition physical cores between concurrent
    # BLAS rotation (background) and C extension compute (foreground) to
    # prevent oversubscription. Without partitioning, both use all cores
    # (2N threads on N cores), causing context-switch overhead.
    # On Accelerate (uncontrollable BLAS), rotation threads are ignored
    # by blas_threads() and Accelerate uses all cores, so give compute
    # fewer threads to leave headroom.
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
            # Accelerate will use all cores for rotation regardless;
            # give OpenMP half to reduce contention
            pipeline_omp_threads = max(1, total_cores // 2)
            pipeline_rot_threads = total_cores  # ignored by blas_threads()
    else:
        pipeline_omp_threads = omp_threads
        pipeline_rot_threads = rotation_threads

    # Enforce minimum 20 golden section iterations for ~1e-5 lambda tolerance
    n_refine = max(n_refine, 20)

    # Precompute SNP-invariant Uab columns once (depends only on UtW, Uty).
    # Eliminates per-chunk reconstruction: uab_invariant_soa is shared across
    # all chunks and referenced via closure by _prepare_chunk.
    uab_invariant_soa = compute_uab_invariant_soa(UtW, Uty) if use_split else None

    # Create persistent C workspace once (before chunk loop).
    # Holds precomputed lambda_grid, hi_eval_grid, logdet_h_grid, grid_inv, and
    # invariant Iab column sums — reused across all chunks without reallocation.
    # PyCapsule is freed automatically when lmm_workspace goes out of scope.
    # Only create for Wald mode (lmm_mode==1): the C workspace uses
    # compute_wald_split_c_ws which is Wald-specific. LRT/Score/All
    # use _compute_lmm_chunk_numpy instead.
    lmm_workspace = (
        create_lmm_workspace(
            eigenvalues_np,
            uab_invariant_soa,
            n_samples,
            l_min,
            l_max,
            n_grid,
            n_refine,
            pipeline_omp_threads,
        )
        if use_split and lmm_mode == 1
        else None
    )

    def _prepare_chunk(chunk_start: int) -> tuple:
        """Slice, impute, rotate, build split Uab in SoA layout.

        BLAS operations release the GIL. Returns SoA-layout varying Uab —
        invariant Uab is precomputed once in outer scope (uab_invariant_soa).
        """
        chunk_end = min(chunk_start + chunk_size, n_filtered)
        chunk_indices = snp_indices[chunk_start:chunk_end]
        geno_chunk = genotypes[:, chunk_indices]

        # Mean-impute
        chunk_means = col_means[chunk_indices]
        missing = np.isnan(geno_chunk)
        if missing.any():  # RUN-06: skip O(n*chunk) np.where on clean data
            geno_chunk = np.where(missing, chunk_means[None, :], geno_chunk)
        del missing

        # Rotate — always allocate fresh buffer (no shared UtG_buf in pipeline)
        with blas_threads(pipeline_rot_threads):
            UtG = U.T @ geno_chunk

        # Build SNP-varying Uab in SoA layout (n_snps, 3, n_samples).
        # Invariant part (uab_invariant_soa) is from outer scope — precomputed once.
        uab_var_soa = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG)
        actual_len = chunk_end - chunk_start
        return uab_var_soa, actual_len

    if show_progress and n_chunks > 1:
        chunk_iterator = progress_iterator(
            chunk_starts, total=n_chunks, desc="LMM association"
        )
    else:
        chunk_iterator = iter(chunk_starts)

    def _compute_and_write(uab_var_soa: np.ndarray, actual_len: int) -> None:
        """Run Wald C extension compute on a chunk and write results.

        Only valid for lmm_mode == 1 (Wald). Called exclusively from the
        pipeline path which gates on lmm_mode == 1.
        """
        if lmm_mode != 1:
            raise RuntimeError(
                "_compute_and_write requires lmm_mode=1 (Wald), "
                f"got lmm_mode={lmm_mode}"
            )
        nonlocal write_offset, t_numpy_compute_total, t_result_write_total

        t_compute_start = time.perf_counter()
        try:
            cr = compute_wald_split_c_ws(
                lmm_workspace,
                uab_var_soa,
                pipeline_omp_threads,
            )
        except MemoryError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"C extension compute failed at SNP offset "
                f"{write_offset}/{n_filtered}. "
                f"Processed {write_offset} SNPs before failure."
            ) from exc
        t_numpy_compute_total += time.perf_counter() - t_compute_start

        t_write_start = time.perf_counter()
        s = slice(write_offset, write_offset + actual_len)
        for key in arrays_out:
            arrays_out[key][s] = cr[key][:actual_len]
        write_offset += actual_len
        t_result_write_total += time.perf_counter() - t_write_start

    if use_pipeline:
        # Pipelined: overlap rotation of chunk N+1 with C compute of chunk N.
        # Both operations release the GIL so they run concurrently.
        t_rot_start = time.perf_counter()
        current = _prepare_chunk(chunk_starts[0])
        t_rotation_total += time.perf_counter() - t_rot_start
        next(chunk_iterator)  # consume first element

        with ThreadPoolExecutor(max_workers=1) as executor:
            for _i, chunk_start in enumerate(chunk_iterator, start=1):
                uab_var_soa, actual_len = current

                # Submit next chunk preparation (runs in background thread)
                future = executor.submit(_prepare_chunk, chunk_start)

                # C extension compute on current chunk (releases GIL).
                # Cores are partitioned: rotation gets pipeline_rot_threads,
                # compute gets pipeline_omp_threads.
                _compute_and_write(uab_var_soa, actual_len)
                del uab_var_soa

                # Wait for background preparation to complete
                t_rot_start = time.perf_counter()
                try:
                    current = future.result()
                except MemoryError:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"Pipeline chunk preparation failed at SNP offset "
                        f"{write_offset}/{n_filtered} (chunk starting at "
                        f"index {chunk_start}). "
                        f"Processed {write_offset} SNPs before failure."
                    ) from exc
                t_rotation_total += time.perf_counter() - t_rot_start

            # Process last chunk (no next chunk to overlap with)
            uab_var_soa, actual_len = current
            _compute_and_write(uab_var_soa, actual_len)
            del uab_var_soa
    else:
        # Sequential path (non-split modes or single chunk)

        def _run_lmm_chunk(Uab_batch: np.ndarray) -> dict:
            """Run LMM compute on a Uab batch with BLAS thread control."""
            blas_ctx = blas_threads(1) if _C_ACCEL_AVAILABLE else nullcontext()
            with blas_ctx:
                try:
                    return _compute_lmm_chunk_numpy(
                        lmm_mode,
                        n_cvt,
                        eigenvalues_np,
                        Uab_batch,
                        n_samples,
                        l_min=l_min,
                        l_max=l_max,
                        n_grid=n_grid,
                        n_refine=n_refine,
                        Hi_eval_null=Hi_eval_null,
                        logl_H0=logl_H0,
                        n_threads=omp_threads,
                    )
                except MemoryError:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"LMM compute failed at SNP offset "
                        f"{write_offset}/{n_filtered}. "
                        f"Processed {write_offset} SNPs "
                        f"before failure."
                    ) from exc

        for chunk_start in chunk_iterator:
            chunk_end = min(chunk_start + chunk_size, n_filtered)
            chunk_indices = snp_indices[chunk_start:chunk_end]
            geno_chunk = genotypes[:, chunk_indices]

            # Mean-impute missing genotypes
            chunk_means = col_means[chunk_indices]
            missing = np.isnan(geno_chunk)
            if missing.any():  # RUN-06: skip O(n*chunk) np.where on clean data
                geno_chunk = np.where(missing, chunk_means[None, :], geno_chunk)
            del missing, chunk_means

            # Rotate genotypes
            t_rot_start = time.perf_counter()
            actual_snps = geno_chunk.shape[1]
            with blas_threads(rotation_threads):
                if actual_snps == chunk_size:
                    np.dot(U.T, geno_chunk, out=UtG_buf)
                    UtG = UtG_buf
                else:
                    UtG = U.T @ geno_chunk
            t_rotation_total += time.perf_counter() - t_rot_start
            del geno_chunk

            # Compute
            t_compute_start = time.perf_counter()
            if use_split:
                # Build SoA-layout varying Uab only — invariant was precomputed once.
                uab_var_soa = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG)
                del UtG
                if lmm_mode == 1 and lmm_workspace is not None:
                    # Wald: use C workspace (no full Uab needed)
                    with blas_threads(1):
                        cr = compute_wald_split_c_ws(
                            lmm_workspace,
                            uab_var_soa,
                            omp_threads,
                        )
                    del uab_var_soa
                else:
                    # LRT/Score/All: reconstruct full Uab from split components
                    Uab_batch = reconstruct_uab_from_soa(uab_invariant_soa, uab_var_soa)
                    del uab_var_soa
                    cr = _run_lmm_chunk(Uab_batch)
                    del Uab_batch
            else:
                Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
                del UtG
                cr = _run_lmm_chunk(Uab_batch)
                del Uab_batch
            t_numpy_compute_total += time.perf_counter() - t_compute_start

            # Write results
            t_write_start = time.perf_counter()
            actual_len = chunk_end - chunk_start
            s = slice(write_offset, write_offset + actual_len)
            for key in arrays_out:
                arrays_out[key][s] = cr[key][:actual_len]
            write_offset += actual_len
            t_result_write_total += time.perf_counter() - t_write_start
            del cr

    # Validate all results were written
    if write_offset != n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {write_offset} results,"
            f" expected {n_filtered}. This is an internal error — please report"
            f" this issue with your dataset dimensions."
        )

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_numpy", "after_all_chunks")

    # NaN diagnostic: warn if any output arrays contain NaN results
    for key, arr in arrays_out.items():
        n_nan = int(np.sum(np.isnan(arr)))
        if n_nan > 0:
            logger.warning(
                f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
                "check kinship matrix quality and available memory"
            )

    # Lambda boundary convergence diagnostics
    n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
        lmm_mode, arrays_out, l_min, l_max
    )
    log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        t_eigen = t_eigen_end - t_eigen_start
        accounted = (
            t_eigen + t_rotation_total + t_numpy_compute_total + t_result_write_total
        )
        logger.info("Timing breakdown:")
        logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
        logger.info(f"  UT@G rotation:       {t_rotation_total:.2f}s")
        logger.info(f"  NumPy compute:       {t_numpy_compute_total:.2f}s")
        logger.info(f"  Result write:        {t_result_write_total:.2f}s")
        logger.info("  ----")
        logger.info(f"  Accounted:           {accounted:.2f}s")
        logger.info(f"  Total:               {elapsed:.2f}s")
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    return _build_results(
        lmm_mode, snp_indices, filtered_afs, filtered_miss, snp_info, arrays_out
    )
