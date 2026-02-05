"""Memory estimation and checking for large-scale GWAS operations.

Provides pre-allocation memory checks to prevent OOM errors at 200k sample scale.
Also provides cleanup utilities for freeing memory between benchmark runs.
"""

import gc
from typing import NamedTuple

import psutil
from loguru import logger


def estimate_eigendecomp_memory(n_samples: int) -> float:
    """Estimate peak memory (GB) for eigendecomposition of kinship matrix.

    Peak memory during eigendecomposition includes:
    - K (input kinship matrix): n^2 * 8 bytes
    - U (output eigenvectors): n^2 * 8 bytes
    - workspace (DSYEVR LAPACK routine): ~26*n*8 + 10*n*4 bytes

    For 200k samples: 320GB + 320GB + 0.04GB = ~640GB
    For 100k samples: 80GB + 80GB + 0.02GB = ~160GB

    Args:
        n_samples: Number of samples (individuals).

    Returns:
        Estimated peak memory in GB.

    Example:
        >>> estimate_eigendecomp_memory(200_000)
        640.04
    """
    kinship_gb = n_samples**2 * 8 / 1e9  # K input matrix
    eigenvectors_gb = n_samples**2 * 8 / 1e9  # U output eigenvectors
    workspace_gb = (26 * n_samples * 8 + 10 * n_samples * 4) / 1e9  # DSYEVR
    return kinship_gb + eigenvectors_gb + workspace_gb


class MemoryBreakdown(NamedTuple):
    """Detailed memory breakdown for GWAS workflow.

    All values in GB. Peak memory is the maximum of eigendecomp phase
    and LMM phase since they don't overlap.
    """

    kinship_gb: float  # n^2 * 8 bytes (float64)
    genotypes_gb: float  # n * p * 4 bytes (float32)
    eigenvectors_gb: float  # n^2 * 8 bytes (float64)
    eigendecomp_workspace_gb: float  # ~26*n*8 + 10*n*4 bytes for DSYEVR
    lmm_rotated_gb: float  # n * 8 * 3 bytes (Uy, UW, rotated vectors)
    lmm_batch_gb: float  # n * batch_size * 8 bytes
    total_gb: float  # Peak memory (max of phases)
    available_gb: float  # Current available system memory
    sufficient: bool  # Whether available >= total * 1.1


def estimate_workflow_memory(
    n_samples: int,
    n_snps: int,
    lmm_batch_size: int = 20_000,
) -> MemoryBreakdown:
    """Estimate memory requirements for full GWAS workflow.

    Calculates memory for kinship computation, eigendecomposition, and LMM
    association testing. Returns the peak memory requirement.

    Note: This is a conservative estimate based on a fixed batch size. The JAX
    runner computes its own chunk size based on int32 buffer limits, which may
    differ. This check ensures sufficient memory for the dominant costs (kinship
    and eigenvector matrices at O(n_samples²)) but may underestimate memory for
    very large SNP counts on small-memory systems.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        lmm_batch_size: Batch size for LMM SNP processing (used for estimate,
            actual JAX chunk size is computed from int32 limits).

    Returns:
        MemoryBreakdown with detailed component estimates and total.

    Example:
        >>> est = estimate_workflow_memory(200_000, 95_000)
        >>> print(f"Need {est.total_gb:.0f}GB, have {est.available_gb:.0f}GB")
    """
    # Component sizes
    kinship_gb = n_samples**2 * 8 / 1e9  # float64
    genotypes_gb = n_samples * n_snps * 4 / 1e9  # float32
    eigenvectors_gb = n_samples**2 * 8 / 1e9  # float64

    # Eigendecomp workspace: DSYEVR uses O(n) workspace
    # Formula: LWORK=26*n doubles + LIWORK=10*n integers
    eigendecomp_workspace_gb = (26 * n_samples * 8 + 10 * n_samples * 4) / 1e9

    # LMM working memory
    lmm_rotated_gb = n_samples * 8 * 3 / 1e9  # Uy, UW, Ux per SNP
    lmm_batch_gb = n_samples * lmm_batch_size * 8 / 1e9

    # Peak memory calculation
    # Workflow: genotypes -> kinship -> eigendecomp -> LMM
    # Kinship can be freed after eigendecomp

    # Phase 1 (kinship): genotypes + kinship (building kinship from genotypes)
    peak_kinship = genotypes_gb + kinship_gb

    # Phase 2 (eigendecomp): kinship + eigenvectors + workspace
    # (genotypes can be freed during eigendecomp if not needed for LMM)
    peak_eigendecomp = kinship_gb + eigenvectors_gb + eigendecomp_workspace_gb

    # Phase 3 (LMM): eigenvectors + genotypes + working
    # (kinship freed, eigenvalues are small ~n*8 bytes)
    eigenvalues_gb = n_samples * 8 / 1e9
    peak_lmm = (
        eigenvectors_gb + genotypes_gb + eigenvalues_gb + lmm_rotated_gb + lmm_batch_gb
    )

    total_gb = max(peak_kinship, peak_eigendecomp, peak_lmm)

    # Check available memory
    available_gb = psutil.virtual_memory().available / 1e9
    sufficient = total_gb * 1.1 < available_gb  # 10% safety margin

    return MemoryBreakdown(
        kinship_gb=kinship_gb,
        genotypes_gb=genotypes_gb,
        eigenvectors_gb=eigenvectors_gb,
        eigendecomp_workspace_gb=eigendecomp_workspace_gb,
        lmm_rotated_gb=lmm_rotated_gb,
        lmm_batch_gb=lmm_batch_gb,
        total_gb=total_gb,
        available_gb=available_gb,
        sufficient=sufficient,
    )


class StreamingMemoryBreakdown(NamedTuple):
    """Detailed memory breakdown for streaming GWAS workflow.

    All values in GB. Peak memory is the maximum across workflow phases:
    1. Kinship accumulation: kinship + chunk
    2. Eigendecomposition: kinship + eigenvectors + workspace (typically peak)
    3. LMM: eigenvectors + chunk + rotation buffer + grid REML

    The key difference from full-load estimation is that genotypes are
    O(n * chunk_size), not O(n * n_snps).
    """

    kinship_gb: float  # n^2 * 8 bytes (float64)
    eigenvectors_gb: float  # n^2 * 8 bytes (float64)
    eigendecomp_workspace_gb: float  # ~26*n*8 + 10*n*4 bytes for DSYEVR
    chunk_gb: float  # n * chunk_size * 8 bytes (float64 for precision)
    rotation_buffer_gb: float  # n * chunk_size * 8 bytes for UtG
    grid_reml_gb: float  # n_grid * chunk_size * 8 bytes for Grid REML intermediate
    total_peak_gb: float  # Max of phases (eigendecomp is typically peak)
    available_gb: float  # Current available system memory
    sufficient: bool  # Whether available >= total * 1.1


def estimate_streaming_memory(
    n_samples: int,
    n_snps: int,  # Only for logging; not used in peak calculation
    chunk_size: int = 10_000,
    n_grid: int = 50,
) -> StreamingMemoryBreakdown:
    """Estimate memory requirements for streaming GWAS workflow.

    Calculates memory for streaming kinship computation, eigendecomposition,
    and LMM association testing. Returns the peak memory requirement.

    Key difference from full-load estimation:
    - Genotypes: O(n * chunk_size) not O(n * n_snps)
    - Peak is typically eigendecomposition (kinship + eigenvectors simultaneously)

    For 200k samples, 95k SNPs, 10k chunk, n_grid=50:
    - Kinship accumulation: 320GB + 16GB = 336GB
    - Eigendecomp: 320GB + 320GB + 0.04GB = 640GB (PEAK)
    - LMM: 320GB + 16GB + 16GB + 4MB = 356GB (still not peak since eigendecomp is 640GB)

    Note: This reveals the true constraint - eigendecomposition cannot be
    streamed and requires both kinship (input) and eigenvectors (output)
    matrices simultaneously.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (for logging only, not used in peak calculation).
        chunk_size: SNPs per chunk (default 10,000).
        n_grid: Grid points for lambda optimization (default 50).

    Returns:
        StreamingMemoryBreakdown with detailed component estimates.

    Example:
        >>> est = estimate_streaming_memory(200_000, 95_000)
        >>> print(f"Peak: {est.total_peak_gb:.0f}GB (eigendecomp)")
    """
    # Component sizes
    kinship_gb = n_samples**2 * 8 / 1e9  # float64
    eigenvectors_gb = n_samples**2 * 8 / 1e9  # float64

    # Eigendecomp workspace: DSYEVR uses O(n) workspace
    # Formula: LWORK=26*n doubles + LIWORK=10*n integers
    eigendecomp_workspace_gb = (26 * n_samples * 8 + 10 * n_samples * 4) / 1e9

    # Chunk memory (float64 for precision in kinship accumulation)
    chunk_gb = n_samples * chunk_size * 8 / 1e9
    rotation_buffer_gb = n_samples * chunk_size * 8 / 1e9  # UtG buffer

    # Grid REML intermediate: vmap over lambda values creates
    # (n_grid, chunk_size) arrays during log-likelihood evaluation
    grid_reml_gb = n_grid * chunk_size * 8 / 1e9

    # Peak memory calculation by workflow phase
    # Phase 1 (kinship accumulation): kinship + chunk
    peak_kinship = kinship_gb + chunk_gb

    # Phase 2 (eigendecomp): kinship (input) + eigenvectors (output) + workspace
    peak_eigendecomp = kinship_gb + eigenvectors_gb + eigendecomp_workspace_gb

    # Phase 3 (LMM): eigenvectors + chunk + rotation buffer + grid REML
    # (kinship can be freed after eigendecomp)
    peak_lmm = eigenvectors_gb + chunk_gb + rotation_buffer_gb + grid_reml_gb

    total_peak_gb = max(peak_kinship, peak_eigendecomp, peak_lmm)

    # Check available memory
    available_gb = psutil.virtual_memory().available / 1e9
    sufficient = total_peak_gb * 1.1 < available_gb  # 10% safety margin

    return StreamingMemoryBreakdown(
        kinship_gb=kinship_gb,
        eigenvectors_gb=eigenvectors_gb,
        eigendecomp_workspace_gb=eigendecomp_workspace_gb,
        chunk_gb=chunk_gb,
        rotation_buffer_gb=rotation_buffer_gb,
        grid_reml_gb=grid_reml_gb,
        total_peak_gb=total_peak_gb,
        available_gb=available_gb,
        sufficient=sufficient,
    )


class LmmMemoryBreakdown(NamedTuple):
    """Detailed memory breakdown for LMM association testing.

    All values in GB. Peak memory is max of eigendecomp phase and LMM phase.
    """

    genotypes_gb: float  # n_samples * n_snps * 4 bytes (float32 input)
    kinship_gb: float  # n_samples^2 * 8 bytes (if computing kinship)
    eigenvectors_gb: float  # n_samples^2 * 8 bytes
    eigendecomp_workspace_gb: float  # DSYEVR workspace
    chunk_gb: float  # n_samples * chunk_size * 8 bytes
    rotation_buffer_gb: float  # UtG rotation buffer
    total_peak_gb: float  # Max across phases
    available_gb: float  # Current system memory
    sufficient: bool  # Whether available >= total * 1.1


def estimate_lmm_memory(
    n_samples: int,
    n_snps: int,
    has_kinship: bool = False,
    chunk_size: int = 10_000,
) -> LmmMemoryBreakdown:
    """Estimate memory requirements for LMM association testing.

    Call this before running analysis to verify sufficient memory.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        has_kinship: If True, assume kinship is pre-computed (skip kinship computation).
            Note: kinship_gb still reflects memory needed to hold the loaded matrix.
        chunk_size: SNPs per chunk for streaming mode.

    Returns:
        LmmMemoryBreakdown with detailed component estimates.

    Example:
        >>> est = estimate_lmm_memory(200_000, 95_000)
        >>> print(f"Need {est.total_peak_gb:.0f}GB, have {est.available_gb:.0f}GB")
        >>> if not est.sufficient:
        ...     raise MemoryError("Insufficient memory")
    """
    # Component sizes
    genotypes_gb = n_samples * n_snps * 4 / 1e9  # float32 loading
    kinship_gb = n_samples**2 * 8 / 1e9  # float64, always needed to hold matrix
    eigenvectors_gb = n_samples**2 * 8 / 1e9  # float64

    # Eigendecomp workspace: DSYEVR uses O(n) workspace
    # Formula: LWORK=26*n doubles + LIWORK=10*n integers
    eigendecomp_workspace_gb = (26 * n_samples * 8 + 10 * n_samples * 4) / 1e9

    # Chunk memory for streaming LMM
    chunk_gb = n_samples * chunk_size * 8 / 1e9  # float64
    rotation_buffer_gb = n_samples * chunk_size * 8 / 1e9  # UtG buffer

    # Peak memory calculation by workflow phase
    if has_kinship:
        # Has pre-computed kinship: load kinship from file instead of computing
        # Phase 1 (load kinship): kinship + genotypes (loading for LMM)
        peak_phase1 = kinship_gb + genotypes_gb

        # Phase 2 (eigendecomp): kinship (input) + eigenvectors (output) + workspace
        peak_eigendecomp = kinship_gb + eigenvectors_gb + eigendecomp_workspace_gb

        # Phase 3 (LMM): eigenvectors + chunk + rotation buffer
        peak_lmm = eigenvectors_gb + chunk_gb + rotation_buffer_gb

        total_peak_gb = max(peak_phase1, peak_eigendecomp, peak_lmm)
    else:
        # Computing kinship from genotypes
        # Phase 1 (kinship): genotypes + kinship (building kinship from genotypes)
        peak_kinship = genotypes_gb + kinship_gb

        # Phase 2 (eigendecomp): kinship (input) + eigenvectors (output) + workspace
        peak_eigendecomp = kinship_gb + eigenvectors_gb + eigendecomp_workspace_gb

        # Phase 3 (LMM): eigenvectors + chunk + rotation buffer
        # (kinship freed, genotypes can be reloaded in chunks)
        peak_lmm = eigenvectors_gb + chunk_gb + rotation_buffer_gb

        total_peak_gb = max(peak_kinship, peak_eigendecomp, peak_lmm)

    # Check available memory
    available_gb = psutil.virtual_memory().available / 1e9
    sufficient = total_peak_gb * 1.1 < available_gb  # 10% safety margin

    return LmmMemoryBreakdown(
        genotypes_gb=genotypes_gb,
        kinship_gb=kinship_gb,
        eigenvectors_gb=eigenvectors_gb,
        eigendecomp_workspace_gb=eigendecomp_workspace_gb,
        chunk_gb=chunk_gb,
        rotation_buffer_gb=rotation_buffer_gb,
        total_peak_gb=total_peak_gb,
        available_gb=available_gb,
        sufficient=sufficient,
    )


def check_memory_available(
    required_gb: float,
    safety_margin: float = 0.1,
    operation: str = "operation",
) -> bool:
    """Check if sufficient memory is available, raise if not.

    Args:
        required_gb: Memory required in GB.
        safety_margin: Additional margin (0.1 = 10%).
        operation: Description for error message.

    Returns:
        True if sufficient memory available.

    Raises:
        MemoryError: If insufficient memory with detailed message.
    """
    available_gb = psutil.virtual_memory().available / 1e9
    required_with_margin = required_gb * (1 + safety_margin)

    if required_with_margin > available_gb:
        raise MemoryError(
            f"Insufficient memory for {operation}. "
            f"Need {required_gb:.1f}GB (+{safety_margin*100:.0f}% margin = "
            f"{required_with_margin:.1f}GB), but only {available_gb:.1f}GB available. "
            f"Consider using a machine with more RAM or reducing dataset size."
        )

    return True


class MemorySnapshot(NamedTuple):
    """Snapshot of current memory state for debugging.

    All values in GB.
    """

    rss_gb: float  # Resident Set Size (actual RAM used by process)
    vms_gb: float  # Virtual Memory Size (total address space)
    available_gb: float  # Available system memory
    total_gb: float  # Total system memory
    percent_used: float  # Percentage of total system memory in use


def get_memory_snapshot() -> MemorySnapshot:
    """Get current memory usage snapshot.

    Returns:
        MemorySnapshot with RSS, VMS, available, and total memory.

    Example:
        >>> snap = get_memory_snapshot()
        >>> print(f"Using {snap.rss_gb:.1f}GB of {snap.total_gb:.1f}GB")
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()

    return MemorySnapshot(
        rss_gb=mem_info.rss / 1e9,
        vms_gb=mem_info.vms / 1e9,
        available_gb=vm.available / 1e9,
        total_gb=vm.total / 1e9,
        percent_used=((vm.total - vm.available) / vm.total) * 100,
    )


def log_memory_snapshot(label: str = "", level: str = "INFO") -> MemorySnapshot:
    """Log current memory state with optional label.

    Useful for debugging memory issues in Databricks notebooks or
    tracking memory across benchmark runs.

    Args:
        label: Optional label for this snapshot (e.g., "after_eigendecomp").
        level: Log level ("DEBUG", "INFO", "WARNING").

    Returns:
        MemorySnapshot for chaining/assertions.

    Example:
        >>> log_memory_snapshot("before_100k_run")
        INFO | Memory [before_100k_run]: RSS=89.5GB, Available=160.2GB (35.0% used)
    """
    snap = get_memory_snapshot()
    label_str = f" [{label}]" if label else ""
    msg = (
        f"Memory{label_str}: RSS={snap.rss_gb:.1f}GB, "
        f"Available={snap.available_gb:.1f}GB/{snap.total_gb:.1f}GB "
        f"({snap.percent_used:.1f}% used)"
    )

    if level == "DEBUG":
        logger.debug(msg)
    elif level == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)

    return snap


def cleanup_memory(clear_jax: bool = True, verbose: bool = True) -> MemorySnapshot:
    """Free memory after a computation run.

    Call this between benchmark runs or after large computations to
    prevent memory accumulation that can cause OOM/SIGSEGV errors.

    This function:
    1. Runs Python garbage collection (multiple passes)
    2. Clears JAX compilation caches (optional, enabled by default)
    3. Logs memory before/after cleanup if verbose

    Args:
        clear_jax: If True (default), clear JAX caches. Set False if JAX
            not imported or if you want to preserve JIT-compiled functions.
        verbose: If True (default), log memory before and after cleanup.

    Returns:
        MemorySnapshot after cleanup.

    Example:
        >>> # After a benchmark run
        >>> del kinship, eigenvectors, results
        >>> cleanup_memory()
        INFO | Memory [before_cleanup]: RSS=89.5GB, Available=160.2GB/256.0GB
        INFO | Memory [after_cleanup]: RSS=12.3GB, Available=237.4GB/256.0GB
        INFO | Freed 77.2GB (RSS reduced from 89.5GB to 12.3GB)

    Note:
        For best results, explicitly `del` large arrays before calling
        this function. Python's reference counting means arrays won't
        be freed if references still exist.
    """
    if verbose:
        before = log_memory_snapshot("before_cleanup")
    else:
        before = get_memory_snapshot()

    # Multiple GC passes to handle reference cycles
    gc.collect()
    gc.collect()
    gc.collect()

    # Clear JAX caches if requested
    if clear_jax:
        try:
            import jax

            jax.clear_caches()
            # Note: jax.clear_backends() would also clear device memory
            # but reinitializes backends on next use - more aggressive
        except ImportError:
            pass  # JAX not installed, skip

    # Final GC pass after clearing caches
    gc.collect()

    if verbose:
        after = log_memory_snapshot("after_cleanup")
        freed_gb = before.rss_gb - after.rss_gb
        if freed_gb > 0.1:  # Only log if meaningful change
            logger.info(
                f"Freed {freed_gb:.1f}GB (RSS reduced from "
                f"{before.rss_gb:.1f}GB to {after.rss_gb:.1f}GB)"
            )
        elif freed_gb < -0.1:
            logger.warning(
                f"Memory increased by {-freed_gb:.1f}GB during cleanup "
                f"(RSS: {before.rss_gb:.1f}GB → {after.rss_gb:.1f}GB)"
            )
    else:
        after = get_memory_snapshot()

    return after


def check_memory_before_run(
    n_samples: int,
    n_snps: int,
    operation: str = "GWAS",
    has_kinship: bool = False,
) -> bool:
    """Pre-flight memory check with helpful diagnostics.

    Call this before starting a large computation to verify sufficient
    memory is available. Provides actionable suggestions if memory is
    insufficient.

    Args:
        n_samples: Number of samples in the dataset.
        n_snps: Number of SNPs in the dataset.
        operation: Description for error messages.
        has_kinship: If True, assume kinship is pre-computed.

    Returns:
        True if sufficient memory available.

    Raises:
        MemoryError: If insufficient memory with suggestions.

    Example:
        >>> check_memory_before_run(100_000, 100_000)
        INFO | Memory check for GWAS (100,000 samples × 100,000 SNPs):
        INFO |   Estimated peak: 160.0GB (eigendecomp phase)
        INFO |   Available: 237.4GB
        INFO |   Status: OK (47.6GB headroom)
    """
    est = estimate_lmm_memory(n_samples, n_snps, has_kinship=has_kinship)
    snap = get_memory_snapshot()

    logger.info(
        f"Memory check for {operation} ({n_samples:,} samples × {n_snps:,} SNPs):"
    )
    logger.info(f"  Estimated peak: {est.total_peak_gb:.1f}GB (eigendecomp phase)")
    logger.info(f"  Current RSS: {snap.rss_gb:.1f}GB")
    logger.info(f"  Available: {snap.available_gb:.1f}GB")

    # Check if estimated peak exceeds available
    headroom = snap.available_gb - est.total_peak_gb

    if headroom < est.total_peak_gb * 0.1:  # Less than 10% headroom
        logger.warning(
            f"  Status: RISKY ({headroom:.1f}GB headroom, recommend cleanup first)"
        )
        logger.warning("  Suggestion: Run cleanup_memory() before this computation")

        if snap.rss_gb > 10:  # Significant existing memory usage
            raise MemoryError(
                f"Insufficient memory for {operation}.\n"
                f"  Current RSS: {snap.rss_gb:.1f}GB (from previous runs?)\n"
                f"  Estimated peak: {est.total_peak_gb:.1f}GB\n"
                f"  Available: {snap.available_gb:.1f}GB\n\n"
                f"Suggestions:\n"
                f"  1. Run cleanup_memory() to free memory from previous runs\n"
                f"  2. Delete large variables: del kinship, eigenvectors, results\n"
                f"  3. Restart the Python kernel for a clean state"
            )
        else:
            raise MemoryError(
                f"Insufficient memory for {operation}.\n"
                f"  Estimated peak: {est.total_peak_gb:.1f}GB\n"
                f"  Available: {snap.available_gb:.1f}GB\n\n"
                f"Suggestions:\n"
                f"  1. Use a larger machine (need ~{est.total_peak_gb * 1.2:.0f}GB+)\n"
                f"  2. Reduce dataset size"
            )
    else:
        logger.info(f"  Status: OK ({headroom:.1f}GB headroom)")

    return True
