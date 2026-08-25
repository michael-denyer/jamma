"""Memory estimation and checking for large-scale GWAS operations.

Provides pre-allocation memory checks to prevent OOM errors at 200k sample scale.
Also provides cleanup utilities for freeing memory between benchmark runs.
"""

import gc
from typing import Literal, NamedTuple

import psutil
from loguru import logger

from jamma.core.constants import env_flag


def forced_numpy_fallback() -> bool:
    """Return True if JLINALG_NO_VENDOR_LAPACK forces the numpy eigendecomp path.

    Presence-based, matching ``docs/CONFIGURATION.md`` and the sibling
    ``JAMMA_FORCE_NUMPY_FALLBACK``: *any* value other than unset/``""``/``"0"``
    forces numpy — including ``"false"``, ``"no"``, and ``"off"``. Set the var to
    ``0`` (or leave it unset) to keep vendor LAPACK; do not expect ``"false"`` to
    mean off. The resolved decision is logged at runtime in
    ``eigendecompose_kinship``.

    Shared by the runtime path (``eigendecompose_kinship``) and the pre-flight
    estimators so both agree on whether vendor LAPACK is bypassed — otherwise a
    forced-numpy run could pass pre-flight on a smaller vendor estimate and
    then OOM.
    """
    return env_flag("JLINALG_NO_VENDOR_LAPACK")


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


def available_ram_gb() -> float:
    """Available system RAM in GB — the one place JAMMA asks psutil for it.

    Every RAM-budget reader (the estimators, the chunk sizers, the kinship
    pass planner) routes through this accessor, so a test pins the machine
    with one ``monkeypatch.setattr(memory, "available_ram_gb", ...)``
    instead of patching psutil in each importing module.
    """
    return psutil.virtual_memory().available / 1e9


def _memory_margin_gb(peak_gb: float) -> float:
    """Safety margin: 10% of peak, capped at 10GB absolute.

    The single spelling of the margin — the estimators' sufficiency verdict
    and check_memory_available both apply exactly this.
    """
    return min(peak_gb * 0.1, 10.0)


def _check_available(total_gb: float) -> tuple[float, bool]:
    """Return (available_gb, sufficient) with 10% margin capped at 10GB."""
    available = available_ram_gb()
    margin_gb = _memory_margin_gb(total_gb)
    return available, (total_gb + margin_gb) < available


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


class EigenDriverPlan(NamedTuple):
    """Chosen eigendecomposition driver and its peak-memory estimate.

    Single source of truth for the DSYEVD-inplace -> DSYEVD -> DSYEVR -> numpy
    driver decision. The runtime path (``eigendecompose_kinship``) builds its
    plan here, so a pre-flight caller using the same function cannot drift from
    it. The chosen driver can still differ per caller when they pass different
    ``inplace_eligible`` inputs.

    Attributes:
        driver: Chosen driver name (one of the four ``Literal`` values).
        use_inplace: Pass ``inplace=True`` to ``jlinalg.eigh`` (K reused as the
            eigenvector output buffer).
        use_dsyevr: DSYEVR was selected — either as the memory-pressure fallback
            from DSYEVD, or because it is the only available vendor driver.
        no_vendor: No vendor LAPACK will run (``np.linalg.eigh`` fallback).
        required_gb: Peak memory (GB) for the chosen driver. For the ``numpy``
            fallback this is a conservative DSYEVD-sized proxy, not numpy's exact
            peak.
        pre_fallback_gb: ``required_gb`` before any DSYEVR fallback (used to log
            which driver we fell back from).
        dsyevr_peak_gb: DSYEVR peak (GB).
        inplace_peak_gb: In-place DSYEVD peak (GB).
    """

    driver: Literal["DSYEVD-inplace", "DSYEVD", "DSYEVR", "numpy"]
    use_inplace: bool
    use_dsyevr: bool
    no_vendor: bool
    required_gb: float
    pre_fallback_gb: float
    dsyevr_peak_gb: float
    inplace_peak_gb: float


def plan_eigen_driver(
    n_samples: int,
    available_gb: float,
    *,
    has_dsyevd: bool,
    has_dsyevr: bool,
    no_vendor: bool,
    inplace_eligible: bool,
) -> EigenDriverPlan:
    """Select the eigendecomposition driver from memory and capability flags.

    Prefers in-place DSYEVD (smallest footprint), falls back to non-inplace
    DSYEVD, then to DSYEVR (O(N) workspace) when the DSYEVD peak plus safety
    margin would not fit. When only vendor DSYEVR is available, plans DSYEVR
    directly. With no vendor DSYEVD/DSYEVR (or a caller-forced ``no_vendor``),
    reports the numpy fallback and its conservative DSYEVD-sized footprint.

    Pure function — takes flags, returns a plan, performs no I/O. The runtime
    caller passes the real ``inplace_eligible`` (K is float64, C-contiguous,
    writeable); the pre-flight estimator passes ``inplace_eligible=True`` because
    the kinship matrix is not built yet and will normally be in-place eligible.

    Args:
        n_samples: Kinship matrix dimension.
        available_gb: Available memory (GB).
        has_dsyevd: Vendor DSYEVD available.
        has_dsyevr: Vendor DSYEVR available.
        no_vendor: Force the numpy fallback (e.g. JLINALG_NO_VENDOR_LAPACK set).
        inplace_eligible: K can be overwritten in place (float64, C-contiguous,
            writeable).

    Returns:
        EigenDriverPlan with the chosen driver, flags, and peak estimates.
    """
    dsyevd_peak = _dsyevd_peak_gb(n_samples)
    dsyevr_peak = _dsyevr_peak_gb(n_samples)
    inplace_peak = _dsyevd_inplace_peak_gb(n_samples)

    # No vendor DSYEVD *and* no vendor DSYEVR -> numpy fallback.
    if not no_vendor and not has_dsyevd and not has_dsyevr:
        no_vendor = True

    if no_vendor:
        return EigenDriverPlan(
            driver="numpy",
            use_inplace=False,
            use_dsyevr=False,
            no_vendor=True,
            required_gb=dsyevd_peak,
            pre_fallback_gb=dsyevd_peak,
            dsyevr_peak_gb=dsyevr_peak,
            inplace_peak_gb=inplace_peak,
        )

    # Only vendor DSYEVR is available (has_dsyevd is False, but the no-vendor
    # check above means has_dsyevr is True): jlinalg.eigh dispatches to DSYEVR
    # directly — there is no in-place path and no DSYEVD peak to reserve.
    if not has_dsyevd:
        return EigenDriverPlan(
            driver="DSYEVR",
            use_inplace=False,
            use_dsyevr=True,
            no_vendor=False,
            required_gb=dsyevr_peak,
            pre_fallback_gb=dsyevr_peak,
            dsyevr_peak_gb=dsyevr_peak,
            inplace_peak_gb=inplace_peak,
        )

    use_inplace = inplace_eligible
    required_gb = inplace_peak if use_inplace else dsyevd_peak
    pre_fallback_gb = required_gb
    use_dsyevr = False

    if required_gb + _memory_margin_gb(required_gb) > available_gb and has_dsyevr:
        pre_fallback_gb = required_gb
        required_gb = dsyevr_peak
        use_inplace = False
        use_dsyevr = True

    driver = "DSYEVR" if use_dsyevr else ("DSYEVD-inplace" if use_inplace else "DSYEVD")
    return EigenDriverPlan(
        driver=driver,
        use_inplace=use_inplace,
        use_dsyevr=use_dsyevr,
        no_vendor=False,
        required_gb=required_gb,
        pre_fallback_gb=pre_fallback_gb,
        dsyevr_peak_gb=dsyevr_peak,
        inplace_peak_gb=inplace_peak,
    )


class MemoryBreakdown(NamedTuple):
    """Detailed memory breakdown for GWAS workflow (full-materialization path).

    All values in GB. Peak memory is the maximum of eigendecomp phase
    and LMM phase since they don't overlap.

    Note: Streaming is the sole execution path in production. Prefer
    StreamingMemoryBreakdown for runtime estimates.
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


def estimate_lmm_memory(
    n_samples: int,
    n_snps: int,
    lmm_batch_size: int = 20_000,
    n_cvt: int = 1,
) -> MemoryBreakdown:
    """Estimate memory for the LMM phase only (full-materialization path).

    Use this when eigendecomposition is already complete and kinship has been
    freed: it returns only the LMM phase requirement, not the peak across
    all workflow phases.

    Includes Uab_batch (n_chunk, n_samples, n_index) and Iab_batch
    (n_chunk, n_cvt+2, n_index) which are the dominant intermediates.

    Note: The default lmm_batch_size=20_000 is a generic estimate; pass the
    runtime chunk size for accurate estimates.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        lmm_batch_size: Batch size for LMM SNP processing. Pass the runtime
            chunk size for accurate estimates.
        n_cvt: Number of covariates (default 1).

    Returns:
        MemoryBreakdown with total_gb reflecting only LMM phase needs.

    Example:
        >>> est = estimate_lmm_memory(100_000, 100)
        >>> print(f"LMM needs {est.total_gb:.0f}GB")
    """
    eigenvectors_gb = _square_matrix_gb(n_samples)
    # Full materialization — streaming path uses chunk_size instead
    # (see estimate_streaming_memory)
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
    dsyrk_scratch_gb: float  # Scratch the active dsyrk backend holds (0 when native)
    peak_kinship_gb: float  # Kinship phase (accumulator + chunk + dsyrk scratch)
    total_peak_gb: float  # Max of phases (eigendecomp typically peak)
    available_gb: float  # Current available system memory
    sufficient: bool  # Whether available exceeds total plus margin (10% capped at 10GB)


def _dsyrk_scratch_gb(n_samples: int) -> float:
    """Scratch the active dsyrk backend holds during kinship accumulation.

    Zero on the native path, which accumulates in place. The NumPy fallback
    blocks its accumulation and holds one block-by-n product, so budgeting only
    the accumulator would approve a run the fallback then OOMs. jlinalg owns the
    block size, so it reports the figure rather than this module re-deriving it.

    Zero too when jlinalg will not import: kinship accumulation goes through
    ``jlinalg.dsyrk``, so there is no dsyrk phase left to budget for. The
    pre-flight must still produce an estimate rather than raise.
    """
    try:
        from jamma.jlinalg import dsyrk_scratch_bytes  # deferred: jlinalg is heavy
    except ImportError:
        logger.debug("Could not import jlinalg; assuming no dsyrk scratch.")
        return 0.0

    return dsyrk_scratch_bytes(n_samples) / 1e9


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
    eigendecomp_peak_gb: float | None = None,
    uab_iab_gb: float | None = None,
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
        compute_chunk_size: SNPs per compute sub-chunk for rotation/Uab/grid
            buffers. Defaults to chunk_size. Pass the runtime compute chunk for
            accurate LMM phase estimates after per-subchunk flush.
        eigendecomp_peak_gb: Peak for the eigendecomposition phase, from
            plan_eigen_driver, when the caller knows which driver will run.
            None (default) uses the conservative non-inplace DSYEVD figure.
        uab_iab_gb: Per-chunk LMM intermediate memory, when the caller knows
            the dispatch path (lmm_extra_bytes_per_snp). None (default) uses
            the conservative full Uab/Iab batch figure.

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

    if uab_iab_gb is None:
        # Conservative full Uab/Iab batch figure, for callers that do not
        # know the dispatch path (kinship's gate).
        uab_iab_gb = _uab_iab_gb(n_samples, compute_chunk_size, n_cvt, use_fused=False)

    # Peak memory calculation by workflow phase
    dsyrk_scratch_gb = _dsyrk_scratch_gb(n_samples)
    peak_kinship = kinship_gb + chunk_gb + dsyrk_scratch_gb
    # Eigendecomp: the caller's driver-aware figure when given, else the
    # conservative non-inplace DSYEVD estimate (K + U + workspace).
    peak_eigendecomp = (
        eigendecomp_peak_gb
        if eigendecomp_peak_gb is not None
        else _dsyevd_peak_gb(n_samples)
    )
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
        dsyrk_scratch_gb=dsyrk_scratch_gb,
        peak_kinship_gb=peak_kinship,
        total_peak_gb=total_peak_gb,
        available_gb=available_gb,
        sufficient=sufficient,
    )


def check_memory_available(
    required_gb: float,
    operation: str = "operation",
) -> bool:
    """Check if sufficient memory is available, raise if not.

    Applies the shared 10% margin capped at 10GB absolute
    (``_memory_margin_gb``). At large scale (500GB+), an uncapped 10%
    margin (50GB) is excessive — the OS and process overhead don't scale
    with eigendecomp workspace size.

    Args:
        required_gb: Memory required in GB.
        operation: Description for error message.

    Returns:
        True if sufficient memory available.

    Raises:
        MemoryError: If insufficient memory with detailed message.
    """
    available_gb = available_ram_gb()
    margin_gb = _memory_margin_gb(required_gb)
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
    mem_info = psutil.Process().memory_info()
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
