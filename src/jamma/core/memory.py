"""Memory estimation and checking for large-scale GWAS operations.

Provides pre-allocation memory checks to prevent OOM errors at 200k sample scale.
Also provides cleanup utilities for freeing memory between benchmark runs.
"""

import gc
from typing import NamedTuple

import psutil
from loguru import logger


def _dsyevd_workspace_gb(n: int) -> float:
    """DSYEVD workspace in GB: (1+6N+2N^2) float64s + (3+5N) int64s (upper bound)."""
    lwork_bytes = (1 + 6 * n + 2 * n * n) * 8  # float64
    # int64 on ILP64, int32 on LP64; use 8 to avoid underestimating
    liwork_bytes = (3 + 5 * n) * 8
    return (lwork_bytes + liwork_bytes) / 1e9


def _dsyevr_workspace_gb(n: int) -> float:
    """DSYEVR workspace in GB: max(1, 26*N) float64s + max(1, 10*N) int64s.

    DSYEVR (MRRR algorithm) uses O(N) workspace vs DSYEVD's O(N^2).
    At 125k samples: ~0.036 GB vs ~250 GB (excludes isuppz, 2*N ints, negligible).
    """
    lwork_bytes = max(1, 26 * n) * 8  # float64
    liwork_bytes = max(1, 10 * n) * 8  # int64 (ILP64 upper bound)
    return (lwork_bytes + liwork_bytes) / 1e9


def _square_matrix_gb(n: int) -> float:
    """Memory (GB) for an n×n float64 matrix."""
    return n * n * 8 / 1e9


def _memory_margin_gb(peak_gb: float) -> float:
    """Safety margin: 10% of peak, capped at 10GB absolute."""
    return min(peak_gb * 0.1, 10.0)


def _check_available(total_gb: float) -> tuple[float, bool]:
    """Return (available_gb, sufficient) with 10% margin capped at 10GB."""
    available_gb = psutil.virtual_memory().available / 1e9
    margin_gb = _memory_margin_gb(total_gb)
    return available_gb, (total_gb + margin_gb) < available_gb


def _eigendecomp_workspace_gb(n: int) -> float:
    """Return eigendecomp workspace in GB (DSYEVD, the default driver)."""
    return _dsyevd_workspace_gb(n)


def _eigendecomp_eigvec_gb(kinship_gb: float) -> float:
    """Return eigenvector memory (GB) for eigendecomp (non-inplace path).

    The in-place path avoids this allocation — see _dsyevd_inplace_peak_gb.
    """
    return kinship_gb


def _dsyevd_inplace_peak_gb(n: int) -> float:
    """Peak memory (GB) for in-place DSYEVD eigendecomposition.

    When inplace=True, K is reused as the eigenvector output buffer.
    Peak is: K (input/output) + DSYEVD workspace. No separate U allocation.
    Saves one full N x N matrix compared to the default path.
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    return _square_matrix_gb(n) + _dsyevd_workspace_gb(n)


def _dsyevd_peak_gb(n: int) -> float:
    """Peak memory (GB) for DSYEVD eigendecomposition (non-inplace).

    Peak is: K (scratch) + U (eigenvectors) + DSYEVD workspace.
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    kinship_gb = _square_matrix_gb(n)
    return kinship_gb + _eigendecomp_eigvec_gb(kinship_gb) + _dsyevd_workspace_gb(n)


def _dsyevr_peak_gb(n: int) -> float:
    """Peak memory (GB) for DSYEVR eigendecomposition.

    On the Python path, jlinalg_dsyevr_ext writes vendor output directly into
    the caller-owned eigenvector buffer and transposes in place, so peak is:
    K (overwritten as scratch) + U (caller output) + O(N).
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    kinship_gb = _square_matrix_gb(n)
    return kinship_gb + _eigendecomp_eigvec_gb(kinship_gb) + _dsyevr_workspace_gb(n)


def estimate_eigendecomp_memory(n_samples: int) -> float:
    """Estimate peak memory (GB) for eigendecomposition of kinship matrix.

    Returns the DSYEVD estimate (the default/faster driver). If DSYEVR is
    used under memory pressure, actual consumption will be lower — this
    is intentionally conservative for pre-flight budget planning.

    jlinalg.eigh DSYEVD path allocates:
    - K (caller scratch): n^2 * 8 bytes
    - U (caller eigenvectors/work buffer): n^2 * 8 bytes
    - workspace (DSYEVD O(n^2))

    For 200k samples: 320GB + 320GB + ~640GB = ~1280GB

    Args:
        n_samples: Number of samples (individuals).

    Returns:
        Estimated peak memory in GB (DSYEVD, conservative).
    """
    return _dsyevd_peak_gb(n_samples)


class MemoryBreakdown(NamedTuple):
    """Detailed memory breakdown for GWAS workflow (full-materialization path).

    All values in GB. Peak memory is the maximum of eigendecomp phase
    and LMM phase since they don't overlap.

    Note: Streaming is the sole execution path in production. Prefer
    StreamingMemoryBreakdown for runtime estimates. This class is retained
    for backward compatibility and direct callers of estimate_workflow_memory.
    """

    kinship_gb: float  # n^2 * 8 bytes (float64)
    genotypes_gb: float  # n * p * 8 bytes (float64)
    eigenvectors_gb: float  # n^2 * 8 bytes (float64)
    eigendecomp_workspace_gb: float  # DSYEVD O(N^2) workspace (conservative)
    lmm_rotated_gb: float  # n * 8 * 3 bytes (Uy, UW, rotated vectors)
    lmm_batch_gb: float  # n * batch_size * 8 bytes
    total_gb: float  # Peak memory (max of phases)
    available_gb: float  # Current available system memory
    sufficient: bool  # Whether available exceeds total plus margin (10% capped at 10GB)


def _uab_iab_gb(
    n_samples: int,
    chunk_size: int,
    n_cvt: int = 1,
    *,
    use_fused: bool = False,
) -> float:
    """Estimate per-chunk LMM intermediate memory (GB).

    Standard path: Uab_batch (chunk_size, n_samples, n_index) +
    Iab_batch (chunk_size, n_cvt+2, n_index).

    Fused path (n_cvt=1 only): UtG_T contiguous copy (chunk_size, n_samples).
    No Uab/Iab batch arrays -- the C workspace computes them on-the-fly.

    Args:
        n_samples: Number of samples.
        chunk_size: SNPs per chunk.
        n_cvt: Number of covariates (default 1).
        use_fused: If True and n_cvt==1, use fused Uab estimate
            (UtG_T only, eliminates Uab_batch and Iab_batch).

    Returns:
        Combined memory in GB.
    """
    if use_fused and n_cvt == 1:
        # Fused path: only UtG_T = (chunk_size, n_samples) float64
        return chunk_size * n_samples * 8 / 1e9
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    uab_bytes = chunk_size * n_samples * n_index * 8
    iab_bytes = chunk_size * (n_cvt + 2) * n_index * 8
    return (uab_bytes + iab_bytes) / 1e9


def estimate_workflow_memory(
    n_samples: int,
    n_snps: int,
    lmm_batch_size: int = 20_000,
    n_cvt: int = 1,
) -> MemoryBreakdown:
    """Estimate memory requirements for full GWAS workflow (full-materialization).

    Calculates memory for kinship computation, eigendecomposition, and LMM
    association testing assuming genotypes are fully loaded into memory.
    Returns the peak memory requirement.

    Note: This estimate assumes full genotype materialization. In the streaming
    architecture (the sole production path), genotypes are loaded as chunks and
    never fully materialized — see estimate_streaming_memory() for accurate
    runtime estimates. check_memory_before_run() uses estimate_streaming_memory
    with the actual chunk size from _compute_chunk_size().

    This function remains useful for direct callers who pass an explicit
    lmm_batch_size and want worst-case full-load estimates.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        lmm_batch_size: Batch size for LMM SNP processing. The actual
            runtime chunk size is computed by auto_tune_chunk_size() in
            chunk.py; this parameter is for explicit caller control.
        n_cvt: Number of covariates (default 1).

    Returns:
        MemoryBreakdown with detailed component estimates and total.

    Example:
        >>> est = estimate_workflow_memory(200_000, 95_000)
        >>> print(f"Need {est.total_gb:.0f}GB, have {est.available_gb:.0f}GB")
    """
    # Component sizes
    kinship_gb = _square_matrix_gb(n_samples)
    # Kinship converts genotypes to float64 via np.array(..., dtype=np.float64)
    # Full materialization — streaming path uses chunk_size instead
    # (see estimate_streaming_memory)
    genotypes_gb = n_samples * n_snps * 8 / 1e9  # float64
    eigenvectors_gb = _square_matrix_gb(n_samples)

    # Eigendecomp workspace: DSYEVD O(n^2) (default driver).
    # If DSYEVR is triggered by memory pressure, actual peak will be lower.
    eigendecomp_workspace_gb = _eigendecomp_workspace_gb(n_samples)

    # LMM working memory
    lmm_rotated_gb = n_samples * 8 * 3 / 1e9  # Uy, UW, Ux per SNP
    # UtG chunk + Uab_batch + Iab_batch (dominant intermediates)
    lmm_batch_gb = (
        n_samples * lmm_batch_size * 8 / 1e9  # UtG chunk
        + _uab_iab_gb(n_samples, lmm_batch_size, n_cvt)  # Uab + Iab
    )

    # Peak memory calculation
    # Workflow: genotypes -> kinship -> eigendecomp -> LMM
    # Kinship can be freed after eigendecomp

    # Phase 1 (kinship): input + working copy coexist during
    # kinship accumulation
    peak_kinship = genotypes_gb * 2 + kinship_gb

    # Phase 2 (eigendecomp): conservative non-inplace estimate (K + U +
    # workspace).  Inplace DSYEVD saves one N×N matrix but requires vendor
    # detection at runtime — check_memory_before_run() uses the tighter
    # _dsyevd_inplace_peak_gb() when available.
    peak_eigendecomp = _dsyevd_peak_gb(n_samples)

    # Phase 3 (LMM): eigenvectors + genotypes + working
    # (kinship freed, eigenvalues are small ~n*8 bytes)
    eigenvalues_gb = n_samples * 8 / 1e9
    peak_lmm = (
        eigenvectors_gb + genotypes_gb + eigenvalues_gb + lmm_rotated_gb + lmm_batch_gb
    )

    total_gb = max(peak_kinship, peak_eigendecomp, peak_lmm)
    available_gb, sufficient = _check_available(total_gb)

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


def estimate_lmm_memory(
    n_samples: int,
    n_snps: int,
    lmm_batch_size: int = 20_000,
    n_cvt: int = 1,
) -> MemoryBreakdown:
    """Estimate memory for the LMM phase only (full-materialization path).

    Use this when eigendecomposition is already complete and kinship has been
    freed. Unlike estimate_workflow_memory() which returns the peak across all
    phases, this returns only the LMM phase requirement.

    Includes Uab_batch (n_chunk, n_samples, n_index) and Iab_batch
    (n_chunk, n_cvt+2, n_index) which are the dominant intermediates.

    Note: The default lmm_batch_size=20_000 is a generic estimate. At runtime,
    auto_tune_chunk_size() in chunk.py computes the actual chunk size based on
    memory budget and int32 buffer constraints. Callers should use that value
    for accurate estimates.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        lmm_batch_size: Batch size for LMM SNP processing. Use the value
            from auto_tune_chunk_size() for accurate runtime estimates.
        n_cvt: Number of covariates (default 1).

    Returns:
        MemoryBreakdown with total_gb reflecting only LMM phase needs.

    Example:
        >>> est = estimate_lmm_memory(100_000, 100)
        >>> print(f"LMM needs {est.total_gb:.0f}GB")
    """
    eigenvectors_gb = _square_matrix_gb(n_samples)
    # Full materialization — streaming path uses chunk_size instead
    # (see estimate_lmm_streaming_memory)
    genotypes_gb = n_samples * n_snps * 8 / 1e9  # float64
    eigenvalues_gb = n_samples * 8 / 1e9
    lmm_rotated_gb = n_samples * 8 * 3 / 1e9
    # UtG chunk + Uab_batch + Iab_batch (dominant intermediates)
    lmm_batch_gb = (
        n_samples * lmm_batch_size * 8 / 1e9  # UtG chunk
        + _uab_iab_gb(n_samples, lmm_batch_size, n_cvt)  # Uab + Iab
    )

    total_gb = (
        eigenvectors_gb + genotypes_gb + eigenvalues_gb + lmm_rotated_gb + lmm_batch_gb
    )
    available_gb, sufficient = _check_available(total_gb)

    return MemoryBreakdown(
        kinship_gb=0.0,
        genotypes_gb=genotypes_gb,
        eigenvectors_gb=eigenvectors_gb,
        eigendecomp_workspace_gb=0.0,
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
    2. Eigendecomposition: K + U (separate) + workspace (typically peak)
    3. LMM: eigenvectors + chunk + rotation buffer + grid REML

    The key difference from full-load estimation is that genotypes are
    O(n * chunk_size), not O(n * n_snps).
    """

    kinship_gb: float  # n^2 * 8 bytes (float64)
    eigenvectors_gb: float  # n^2 * 8 bytes (float64)
    eigendecomp_workspace_gb: float  # DSYEVD O(N^2) workspace (conservative)
    chunk_gb: float  # n * chunk_size * 8 bytes (float64 for precision)
    rotation_buffer_gb: float  # n * chunk_size * 8 * pipeline_buffers bytes for UtG
    grid_reml_gb: float  # n_grid * chunk_size * 8 bytes for Grid REML intermediate
    total_peak_gb: float  # Max of phases (eigendecomp typically peak)
    available_gb: float  # Current available system memory
    sufficient: bool  # Whether available exceeds total plus margin (10% capped at 10GB)


def _streaming_component_sizes(
    n_samples: int,
    chunk_size: int,
    n_grid: int,
    pipeline_buffers: int = 1,
    compute_chunk_size: int | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Compute component memory sizes (GB) for streaming estimation.

    Args:
        n_samples: Number of samples.
        chunk_size: SNPs per disk chunk (raw genotype buffer).
        n_grid: Grid points for lambda optimization.
        pipeline_buffers: Number of simultaneous live UtG rotation buffers (default 1).
            Pass 2 when rotation-compute pipelining holds current + next buffers.
        compute_chunk_size: SNPs per compute sub-chunk (rotation/Uab/grid buffers).
            Defaults to chunk_size for backward compatibility. After per-subchunk
            flush, the actual live compute buffers are sized by compute_chunk_size, not
            the disk chunk_size.

    Returns:
        Tuple of (kinship_gb, eigenvectors_gb, eigendecomp_workspace_gb,
        chunk_gb, rotation_buffer_gb, grid_reml_gb).
    """
    if not isinstance(pipeline_buffers, int):
        raise TypeError(
            f"pipeline_buffers must be an int, got {type(pipeline_buffers).__name__}"
        )
    if pipeline_buffers < 1:
        raise ValueError(f"pipeline_buffers must be >= 1, got {pipeline_buffers}")
    if compute_chunk_size is None:
        compute_chunk_size = chunk_size
    kinship_gb = _square_matrix_gb(n_samples)
    eigenvectors_gb = _square_matrix_gb(n_samples)
    eigendecomp_workspace_gb = _eigendecomp_workspace_gb(n_samples)
    chunk_gb = n_samples * chunk_size * 8 / 1e9
    rotation_buffer_gb = n_samples * compute_chunk_size * 8 / 1e9 * pipeline_buffers
    grid_reml_gb = n_grid * compute_chunk_size * 8 / 1e9
    return (
        kinship_gb,
        eigenvectors_gb,
        eigendecomp_workspace_gb,
        chunk_gb,
        rotation_buffer_gb,
        grid_reml_gb,
    )


def estimate_streaming_memory(
    n_samples: int,
    chunk_size: int = 10_000,
    n_grid: int = 50,
    n_cvt: int = 1,
    pipeline_buffers: int = 1,
    compute_chunk_size: int | None = None,
) -> StreamingMemoryBreakdown:
    """Estimate memory requirements for streaming GWAS workflow.

    Calculates memory for streaming kinship computation, eigendecomposition,
    and LMM association testing. Returns the peak memory requirement.

    Key difference from full-load estimation:
    - Genotypes: O(n * chunk_size) not O(n * n_snps)
    - Peak is typically eigendecomposition (kinship + eigenvectors simultaneously)

    For 200k samples, 10k chunk, n_grid=50:
    - Kinship accumulation: 320GB + 16GB = 336GB
    - Eigendecomp: 320GB + 320GB + ~640GB = ~1280GB (PEAK)
    - LMM: 320GB + 16GB + 16GB + Uab/Iab

    Note: Eigendecomposition cannot be streamed. jlinalg.eigh allocates
    separate eigenvectors (K is used as scratch), so peak includes both
    K and U plus the O(n²) DSYEVD workspace.

    Args:
        n_samples: Number of samples (individuals).
        chunk_size: SNPs per disk chunk (default 10,000).
        n_grid: Grid points for lambda optimization (default 50).
        n_cvt: Number of covariates (default 1).
        pipeline_buffers: Number of simultaneous live UtG rotation buffers (default 1).
            Pass 2 when rotation-compute pipelining holds current + next buffers
            simultaneously — rotation_buffer_gb is multiplied accordingly.
        compute_chunk_size: SNPs per compute sub-chunk for rotation/Uab/grid buffers.
            Defaults to chunk_size. Pass the value from _compute_chunk_size()
            for accurate LMM phase estimates after per-subchunk flush.

    Returns:
        StreamingMemoryBreakdown with detailed component estimates.

    Example:
        >>> est = estimate_streaming_memory(200_000)
        >>> print(f"Peak: {est.total_peak_gb:.0f}GB (eigendecomp)")
    """
    if compute_chunk_size is None:
        compute_chunk_size = chunk_size
    (
        kinship_gb,
        eigenvectors_gb,
        eigendecomp_workspace_gb,
        chunk_gb,
        rotation_buffer_gb,
        grid_reml_gb,
    ) = _streaming_component_sizes(
        n_samples, chunk_size, n_grid, pipeline_buffers, compute_chunk_size
    )

    # Don't use fused estimate: callers (pipeline, kinship, check_memory_before_run)
    # don't know the backend or lmm_mode.  Fused only applies to NumPy modes 1/4.
    uab_iab_gb = _uab_iab_gb(n_samples, compute_chunk_size, n_cvt, use_fused=False)

    # Peak memory calculation by workflow phase
    peak_kinship = kinship_gb + chunk_gb
    # Eigendecomp: conservative non-inplace estimate (K + U + workspace).
    # Inplace DSYEVD saves one N×N matrix but requires vendor detection at
    # runtime — check_memory_before_run() uses the tighter
    # _dsyevd_inplace_peak_gb() when available.
    peak_eigendecomp = _dsyevd_peak_gb(n_samples)
    peak_lmm = (
        eigenvectors_gb + chunk_gb + rotation_buffer_gb + grid_reml_gb + uab_iab_gb
    )

    total_peak_gb = max(peak_kinship, peak_eigendecomp, peak_lmm)
    available_gb, sufficient = _check_available(total_peak_gb)

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


def estimate_lmm_streaming_memory(
    n_samples: int,
    n_snps: int,
    chunk_size: int = 10_000,
    n_grid: int = 50,
    n_cvt: int = 1,
    pipeline_buffers: int = 1,
    compute_chunk_size: int | None = None,
) -> StreamingMemoryBreakdown:
    """Estimate memory for the streaming LMM phase only (not the full pipeline).

    Use this when eigendecomposition is already complete and kinship has been
    freed. Unlike estimate_streaming_memory() which returns the peak across all
    phases, this returns only the LMM phase requirement.

    Includes Uab_batch and Iab_batch intermediates which are the dominant
    compute buffers during LMM computation.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (for logging only, not used in peak calculation).
        chunk_size: SNPs per disk chunk (default 10,000).
        n_grid: Grid points for lambda optimization (default 50).
        n_cvt: Number of covariates (default 1).
        pipeline_buffers: Number of simultaneous live UtG rotation buffers (default 1).
            Pass 2 when rotation-compute pipelining holds current + next buffers
            simultaneously — rotation_buffer_gb is multiplied accordingly.
        compute_chunk_size: SNPs per compute sub-chunk for rotation/Uab/grid buffers.
            Defaults to chunk_size. Pass the value from _compute_chunk_size()
            for accurate LMM phase estimates after per-subchunk flush.

    Returns:
        StreamingMemoryBreakdown with total_peak_gb reflecting only LMM phase needs.

    Example:
        >>> est = estimate_lmm_streaming_memory(100_000, 95_000)
        >>> print(f"LMM needs {est.total_peak_gb:.0f}GB")
    """
    if compute_chunk_size is None:
        compute_chunk_size = chunk_size
    (
        _kinship_gb,
        eigenvectors_gb,
        _eigendecomp_workspace_gb,
        chunk_gb,
        rotation_buffer_gb,
        grid_reml_gb,
    ) = _streaming_component_sizes(
        n_samples, chunk_size, n_grid, pipeline_buffers, compute_chunk_size
    )

    # Don't use fused estimate: callers don't know lmm_mode.
    uab_iab_gb = _uab_iab_gb(n_samples, compute_chunk_size, n_cvt, use_fused=False)
    total_peak_gb = (
        eigenvectors_gb + chunk_gb + rotation_buffer_gb + grid_reml_gb + uab_iab_gb
    )
    available_gb, sufficient = _check_available(total_peak_gb)

    return StreamingMemoryBreakdown(
        kinship_gb=0.0,
        eigenvectors_gb=eigenvectors_gb,
        eigendecomp_workspace_gb=0.0,
        chunk_gb=chunk_gb,
        rotation_buffer_gb=rotation_buffer_gb,
        grid_reml_gb=grid_reml_gb,
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

    The safety margin is percentage-based but capped at 10GB absolute.
    At large scale (500GB+), a 10% margin (50GB) is excessive — the OS
    and process overhead don't scale with eigendecomp workspace size.

    Args:
        required_gb: Memory required in GB.
        safety_margin: Additional margin as fraction (0.1 = 10%), capped
            at 10GB absolute to avoid blocking runs with adequate headroom.
        operation: Description for error message.

    Returns:
        True if sufficient memory available.

    Raises:
        MemoryError: If insufficient memory with detailed message.
    """
    available_gb = psutil.virtual_memory().available / 1e9
    margin_gb = min(required_gb * safety_margin, 10.0)
    required_with_margin = required_gb + margin_gb

    if required_with_margin > available_gb:
        raise MemoryError(
            f"Insufficient memory for {operation}. "
            f"Need {required_gb:.1f}GB (+{margin_gb:.1f}GB margin = "
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
        INFO | Memory [before_100k_run]: using 89.5GB,
             160.2GB free of 256.0GB (35.0% used)
    """
    snap = get_memory_snapshot()
    label_str = f" [{label}]" if label else ""
    msg = (
        f"Memory{label_str}: using {snap.rss_gb:.1f}GB, "
        f"{snap.available_gb:.1f}GB free of {snap.total_gb:.1f}GB "
        f"({snap.percent_used:.1f}% used)"
    )
    logger.log(level, msg)
    return snap


def cleanup_memory(verbose: bool = True) -> MemorySnapshot:
    """Free memory after a computation run.

    Call this between benchmark runs or after large computations to
    prevent memory accumulation that can cause OOM/SIGSEGV errors.

    This function:
    1. Runs Python garbage collection
    2. Runs a second GC pass
    3. Logs memory before/after cleanup if verbose

    Args:
        verbose: If True (default), log memory before and after cleanup.

    Returns:
        MemorySnapshot after cleanup.

    Example:
        >>> # After a benchmark run
        >>> del kinship, eigenvectors, results
        >>> cleanup_memory()
        INFO | Memory [before_cleanup]: using 89.5GB, 160.2GB free of 256.0GB
        INFO | Memory [after_cleanup]: using 12.3GB, 237.4GB free of 256.0GB
        INFO | Freed 77.2GB (process was using 89.5GB, now 12.3GB)

    Note:
        For best results, explicitly `del` large arrays before calling
        this function. Python's reference counting means arrays won't
        be freed if references still exist.
    """
    before = log_memory_snapshot("before_cleanup") if verbose else get_memory_snapshot()

    gc.collect()
    gc.collect()

    if verbose:
        after = log_memory_snapshot("after_cleanup")
        freed_gb = before.rss_gb - after.rss_gb
        if freed_gb > 0.1:  # Only log if meaningful change
            logger.info(
                f"Freed {freed_gb:.1f}GB (process was using "
                f"{before.rss_gb:.1f}GB, now {after.rss_gb:.1f}GB)"
            )
        elif freed_gb < -0.1:
            logger.warning(
                f"Memory increased by {-freed_gb:.1f}GB during cleanup "
                f"(was {before.rss_gb:.1f}GB, now {after.rss_gb:.1f}GB)"
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
        has_kinship: If True, assume kinship is pre-computed (unused,
            kept for backward compatibility).

    Returns:
        True if sufficient memory available.

    Raises:
        MemoryError: If insufficient memory with suggestions.

    Example:
        >>> check_memory_before_run(100_000, 100_000)
        INFO | Memory check for GWAS (100,000 samples x 100,000 SNPs):
        INFO |   Estimated peak: 640.0GB (eigendecomp phase)
        INFO |   Available: 237.4GB
        INFO |   Status: OK (47.6GB headroom)
    """
    from jamma.core.chunk import _compute_chunk_size

    compute_chunk = _compute_chunk_size(n_snps, n_samples=n_samples, pipeline_buffers=2)
    est = estimate_streaming_memory(
        n_samples, pipeline_buffers=2, compute_chunk_size=compute_chunk
    )
    snap = get_memory_snapshot()

    # EIGEN-01: Mirror eigendecompose_kinship()'s driver selection to report
    # the correct peak estimate and prevent spurious OOM aborts.
    reported_peak = est.total_peak_gb
    driver_note = "eigendecomp phase"
    _has_dsyevr = False
    _has_dsyevd = False
    try:
        from jamma import jlinalg

        _has_dsyevr = bool(getattr(jlinalg, "blas_has_dsyevr", False))
        _has_dsyevd = bool(getattr(jlinalg, "blas_has_dsyevd", False))
    except ImportError:
        logger.debug(
            "Could not import jlinalg; "
            "pre-flight check will use conservative DSYEVD estimate."
        )

    dsyevd_peak = _dsyevd_peak_gb(n_samples)
    margin = _memory_margin_gb(dsyevd_peak)
    dsyevd_fits = dsyevd_peak + margin <= snap.available_gb

    # Shared non-eigendecomp phase peaks (kinship build, LMM association).
    # Don't use fused estimate here: we don't know lmm_mode, and fused only
    # applies to modes 1/4.  Using the standard (larger) estimate is safe —
    # it may overestimate for modes 1/4 but never underestimates for 2/3.
    peak_kinship = est.kinship_gb + est.chunk_gb
    peak_lmm = (
        est.eigenvectors_gb
        + est.chunk_gb
        + est.rotation_buffer_gb
        + est.grid_reml_gb
        + _uab_iab_gb(n_samples, compute_chunk, use_fused=False)
    )

    if not dsyevd_fits and _has_dsyevr:
        dsyevr_peak = _dsyevr_peak_gb(n_samples)
        reported_peak = max(peak_kinship, dsyevr_peak, peak_lmm)
        driver_note = "eigendecomp phase, DSYEVR selected"
    elif dsyevd_fits and _has_dsyevd:
        inplace_peak = _dsyevd_inplace_peak_gb(n_samples)
        reported_peak = max(peak_kinship, inplace_peak, peak_lmm)
        driver_note = "eigendecomp phase, DSYEVD in-place"
    elif not dsyevd_fits:
        reported_peak = dsyevd_peak
        driver_note = "eigendecomp phase, DSYEVD (DSYEVR unavailable, memory tight)"
        logger.warning(
            f"DSYEVD peak ({dsyevd_peak:.1f}GB) may exceed available memory "
            f"({snap.available_gb:.1f}GB) and DSYEVR is not available. "
            f"Run may OOM."
        )

    # BLAS context: warn if active backend differs from calibration reference
    from jamma.core.estimates import get_blas_estimate_context

    blas_backend, blas_ilp64, blas_calibrated = get_blas_estimate_context()

    logger.info(
        f"Memory check for {operation} ({n_samples:,} samples × {n_snps:,} SNPs):"
    )
    logger.info(f"  BLAS backend: {blas_backend} (ILP64={blas_ilp64})")
    logger.info(f"  Estimated peak: {reported_peak:.1f}GB ({driver_note})")
    logger.info(f"  Process using: {snap.rss_gb:.1f}GB")
    logger.info(f"  Available: {snap.available_gb:.1f}GB")

    if n_samples > 40_000 and not blas_ilp64:
        logger.warning(
            f"  No ILP64 BLAS detected (active: {blas_backend}). "
            f"Eigendecomposition of {n_samples:,} samples will use NumPy "
            f"fallback, which may be significantly slower. "
            f"Install ILP64 numpy for best performance (see docs/USER_GUIDE.md)."
        )
    if not blas_calibrated:
        logger.warning(
            f"  Time estimates are calibrated to MKL ILP64. "
            f"Active BLAS ({blas_backend}) may yield significantly different runtimes."
        )

    # Check if estimated peak exceeds available
    headroom = snap.available_gb - reported_peak

    # Intentionally uncapped 10%: e.g. at 500GB peak, this warns below 50GB
    # headroom while _check_available only rejects below 10GB, giving the
    # user a ~40GB window to run cleanup_memory() before the gate rejects.
    if headroom < reported_peak * 0.1:
        logger.warning(
            f"  Status: RISKY ({headroom:.1f}GB headroom, recommend cleanup first)"
        )
        logger.warning("  Suggestion: Run cleanup_memory() before this computation")

        if snap.rss_gb > 10:  # Significant existing memory usage
            raise MemoryError(
                f"Insufficient memory for {operation}.\n"
                f"  Process using: {snap.rss_gb:.1f}GB (from previous runs?)\n"
                f"  Estimated peak: {reported_peak:.1f}GB\n"
                f"  Available: {snap.available_gb:.1f}GB\n\n"
                f"Suggestions:\n"
                f"  1. Run cleanup_memory() to free memory from previous runs\n"
                f"  2. Delete large variables: del kinship, eigenvectors, results\n"
                f"  3. Restart the Python kernel for a clean state"
            )
        else:
            raise MemoryError(
                f"Insufficient memory for {operation}.\n"
                f"  Estimated peak: {reported_peak:.1f}GB\n"
                f"  Available: {snap.available_gb:.1f}GB\n\n"
                f"Suggestions:\n"
                f"  1. Use a larger machine (need ~{reported_peak * 1.2:.0f}GB+)\n"
                f"  2. Reduce dataset size"
            )
    else:
        logger.info(f"  Status: OK ({headroom:.1f}GB headroom)")

    return True
