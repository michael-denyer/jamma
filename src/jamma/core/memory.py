"""Memory estimation and checking for large-scale GWAS operations.

Provides pre-allocation memory checks to prevent OOM errors at 200k sample scale.
Also provides cleanup utilities for freeing memory between benchmark runs.
"""

from typing import NamedTuple

import psutil
from loguru import logger

from jamma.core.constants import n_index
from jamma.core.eigen_plan import (
    _dsyevd_peak_gb,
    _eigendecomp_workspace_gb,
    _memory_margin_gb,
    array_gb,
    square_matrix_gb,
)


def available_ram_gb() -> float:
    """Available system RAM in GB, as this module's estimators read it.

    The estimators, the chunk sizers, and the kinship pass planner in this
    module route through this accessor, so a test pins their view of the
    machine with one ``monkeypatch.setattr(memory, "available_ram_gb", ...)``.
    It is not the only psutil read in JAMMA: ``memory_snapshot.py`` queries
    ``psutil.virtual_memory()`` directly for its own logging snapshot, and the
    eigendecomposition driver in ``lmm/eigen.py`` takes its budget from that
    snapshot, not from this accessor. Pinning this function alone does not
    pin what the eigendecomp driver sees.
    """
    return psutil.virtual_memory().available / 1e9


def fits(required_gb: float, available_gb: float) -> bool:
    """Whether *required_gb* plus the shared safety margin fits in *available_gb*.

    The one predicate both the estimators' sufficiency verdict and
    ``require`` apply — see ``_memory_margin_gb`` for the margin (10% of
    ``required_gb``, capped at 10GB absolute).
    """
    return (required_gb + _memory_margin_gb(required_gb)) < available_gb


def _check_available(total_gb: float) -> tuple[float, bool]:
    """Return (available_gb, sufficient) with 10% margin capped at 10GB."""
    available = available_ram_gb()
    return available, fits(total_gb, available)


def require(
    required_gb: float,
    available_gb: float,
    operation: str = "operation",
    *,
    budget_gb: float | None = None,
) -> None:
    """Raise ``MemoryError`` when *required_gb* does not fit the run's ceiling.

    The sole place JAMMA constructs and raises a ``MemoryError``. Every other
    memory gate (the pipeline preflight's budget/availability check, the LMM
    batch runner, and LOCO's pass planner) calls this instead of raising its
    own, so every insufficient-memory error looks the same and the message
    lives in one place.

    Two independent ceilings, checked in order:

    1. ``budget_gb``, a user-set ``--mem-budget`` ceiling, when given.
    2. ``available_gb``, what the machine reports free, via ``fits``.

    Args:
        required_gb: Estimated peak requirement.
        available_gb: What the system reports free.
        operation: Description of what needed the memory, for the message.
        budget_gb: User-set ceiling in GB, or None for no ceiling.

    Raises:
        MemoryError: Naming which ceiling failed, and how to override.
    """
    over_budget = budget_gb is not None and required_gb > budget_gb
    insufficient = not fits(required_gb, available_gb)
    if not over_budget and not insufficient:
        return

    if over_budget:
        message = (
            f"Estimated memory ({required_gb:.1f}GB) for {operation} exceeds "
            f"budget ({budget_gb}GB). Use --no-check-memory to override."
        )
    else:
        margin_gb = _memory_margin_gb(required_gb)
        message = (
            f"Insufficient memory for {operation}. "
            f"Need {required_gb:.1f}GB (+{margin_gb:.1f}GB margin = "
            f"{required_gb + margin_gb:.1f}GB), but only {available_gb:.1f}GB "
            f"available. Use --no-check-memory to override, or use a machine "
            f"with more RAM."
        )
    raise MemoryError(message)


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


def _uab_iab_gb(n_samples: int, chunk_size: int, n_cvt: int = 1) -> float:
    """Estimate per-chunk LMM intermediate memory (GB).

    Uab_batch (chunk_size, n_samples, n_index) + Iab_batch
    (chunk_size, n_cvt+2, n_index).

    Args:
        n_samples: Number of samples.
        chunk_size: SNPs per chunk.
        n_cvt: Number of covariates (default 1).

    Returns:
        Combined memory in GB.
    """
    idx = n_index(n_cvt)
    uab_bytes = chunk_size * n_samples * idx * 8
    iab_bytes = chunk_size * (n_cvt + 2) * idx * 8
    return (uab_bytes + iab_bytes) / 1e9


def estimate_lmm_memory(
    n_samples: int,
    n_snps: int,
    lmm_batch_size: int = 20_000,
    n_cvt: int = 1,
    n_buffers: int = 1,
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
        n_buffers: Live chunk buffers the engine allocates (1 sequential, 2
            pipelined). Scales both the UtG rotation chunk and the Uab/Iab
            extra, matching ``_ChunkEngine``'s per-buffer allocation. Default
            1 keeps every other caller's estimate unchanged.

    Returns:
        MemoryBreakdown with total_gb reflecting only LMM phase needs.

    Example:
        >>> est = estimate_lmm_memory(100_000, 100)
        >>> print(f"LMM needs {est.total_gb:.0f}GB")
    """
    eigenvectors_gb = square_matrix_gb(n_samples)
    # Full materialization — streaming path uses chunk_size instead
    # (see estimate_streaming_memory)
    genotypes_gb = array_gb(n_samples, n_snps)  # float64
    eigenvalues_gb = array_gb(n_samples)
    lmm_rotated_gb = 3 * array_gb(n_samples)
    # UtG chunk + Uab_batch + Iab_batch (dominant intermediates), each live
    # once per buffer.
    lmm_batch_gb = n_buffers * (
        array_gb(n_samples, lmm_batch_size)  # UtG chunk
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
    lmm_chunk_gb: float  # n * compute_chunk_size * 8 bytes, association-pass raw block
    rotation_buffer_gb: float  # n * chunk_size * 8 * pipeline_buffers bytes for UtG
    grid_reml_gb: float  # n_grid * chunk_size * 8 bytes for Grid REML intermediate
    dsyrk_scratch_gb: float  # Scratch the active dsyrk backend holds (0 when native)
    peak_kinship_gb: float  # Kinship phase (accumulator + chunk + dsyrk scratch)
    peak_lmm_gb: float  # LMM phase (eigenvectors + chunk + rotation + grid + Uab/Iab)
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
) -> tuple[float, float, float, float, float, float, float]:
    """Compute component memory sizes (GB) for streaming estimation.

    Args:
        n_samples: Number of samples.
        chunk_size: SNPs per disk chunk (statistics-pass raw genotype buffer).
        n_grid: Grid points for lambda optimization.
        pipeline_buffers: Number of simultaneous live UtG rotation buffers (default 1).
            Pass 2 when rotation-compute pipelining holds current + next buffers.
        compute_chunk_size: SNPs per compute sub-chunk (rotation/Uab/grid buffers,
            and the association-pass raw genotype buffer). Defaults to chunk_size
            for backward compatibility. After per-subchunk flush, the actual live
            compute buffers are sized by compute_chunk_size, not the disk
            chunk_size.

    Returns:
        Tuple of (kinship_gb, eigenvectors_gb, eigendecomp_workspace_gb,
        chunk_gb, lmm_chunk_gb, rotation_buffer_gb, grid_reml_gb).
    """
    if not isinstance(pipeline_buffers, int):
        raise TypeError(
            f"pipeline_buffers must be an int, got {type(pipeline_buffers).__name__}"
        )
    if pipeline_buffers < 1:
        raise ValueError(f"pipeline_buffers must be >= 1, got {pipeline_buffers}")
    if compute_chunk_size is None:
        compute_chunk_size = chunk_size
    kinship_gb = square_matrix_gb(n_samples)
    eigenvectors_gb = square_matrix_gb(n_samples)
    eigendecomp_workspace_gb = _eigendecomp_workspace_gb(n_samples)
    # Statistics-pass raw genotype buffer (kinship phase reads at this width).
    chunk_gb = array_gb(n_samples, chunk_size)
    # Association-pass raw genotype buffer (LMM phase reads at this width).
    # The chunk engine's raw-chunk source hands ``prepare()`` one buffer at a
    # time even under pipelining — the overlap is between a rotated buffer
    # and the next prepare() call, not between two raw reads — so this term
    # does not scale with pipeline_buffers.
    lmm_chunk_gb = array_gb(n_samples, compute_chunk_size)
    rotation_buffer_gb = array_gb(n_samples, compute_chunk_size) * pipeline_buffers
    grid_reml_gb = array_gb(n_grid, compute_chunk_size)
    return (
        kinship_gb,
        eigenvectors_gb,
        eigendecomp_workspace_gb,
        chunk_gb,
        lmm_chunk_gb,
        rotation_buffer_gb,
        grid_reml_gb,
    )


def kinship_cost(kinship_gb: float, chunk_gb: float, dsyrk_scratch_gb: float) -> float:
    """Peak memory (GB) for the streaming kinship-accumulation phase.

    Kinship accumulator + one genotype chunk + whatever scratch the active
    dsyrk backend holds (0 on the native path).
    """
    return kinship_gb + chunk_gb + dsyrk_scratch_gb


def eigen_cost(n_samples: int, eigendecomp_peak_gb: float | None = None) -> float:
    """Peak memory (GB) for the eigendecomposition phase.

    Uses the caller's driver-aware figure (from ``plan_eigen_driver``) when
    given; otherwise the conservative non-inplace DSYEVD estimate (K + U +
    workspace), for callers that do not yet know which driver will run.
    """
    if eigendecomp_peak_gb is not None:
        return eigendecomp_peak_gb
    return _dsyevd_peak_gb(n_samples)


def lmm_cost(
    eigenvectors_gb: float,
    lmm_chunk_gb: float,
    rotation_buffer_gb: float,
    grid_reml_gb: float,
    uab_iab_gb: float,
) -> float:
    """Peak memory (GB) for the LMM association phase.

    Eigenvectors (already in memory from the eigendecomp phase) + the raw
    genotype chunk + the rotation buffer + the grid-REML intermediate + the
    Uab/Iab per-chunk intermediate.
    """
    return (
        eigenvectors_gb + lmm_chunk_gb + rotation_buffer_gb + grid_reml_gb + uab_iab_gb
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
        lmm_chunk_gb,
        rotation_buffer_gb,
        grid_reml_gb,
    ) = _streaming_component_sizes(
        n_samples, chunk_size, n_grid, pipeline_buffers, compute_chunk_size
    )

    if uab_iab_gb is None:
        # Conservative full Uab/Iab batch figure, for callers that do not
        # know the dispatch path (kinship's gate).
        uab_iab_gb = _uab_iab_gb(n_samples, compute_chunk_size, n_cvt)

    # Peak memory by workflow phase, one cost function per phase.
    dsyrk_scratch_gb = _dsyrk_scratch_gb(n_samples)
    peak_kinship = kinship_cost(kinship_gb, chunk_gb, dsyrk_scratch_gb)
    peak_eigendecomp = eigen_cost(n_samples, eigendecomp_peak_gb)
    # lmm_chunk_gb, not chunk_gb: the association pass streams its raw
    # genotype buffer at compute_chunk_size, not the disk stats-pass width.
    peak_lmm = lmm_cost(
        eigenvectors_gb, lmm_chunk_gb, rotation_buffer_gb, grid_reml_gb, uab_iab_gb
    )

    total_peak_gb = max(peak_kinship, peak_eigendecomp, peak_lmm)
    available_gb, sufficient = _check_available(total_peak_gb)

    return StreamingMemoryBreakdown(
        kinship_gb=kinship_gb,
        eigenvectors_gb=eigenvectors_gb,
        eigendecomp_workspace_gb=eigendecomp_workspace_gb,
        chunk_gb=chunk_gb,
        lmm_chunk_gb=lmm_chunk_gb,
        rotation_buffer_gb=rotation_buffer_gb,
        grid_reml_gb=grid_reml_gb,
        dsyrk_scratch_gb=dsyrk_scratch_gb,
        peak_kinship_gb=peak_kinship,
        peak_lmm_gb=peak_lmm,
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

    Thin wrapper over ``require`` (the sole ``MemoryError`` construction site),
    reading ``available_ram_gb()`` itself so existing callers keep their
    two-argument call.

    Args:
        required_gb: Memory required in GB.
        operation: Description for error message.

    Returns:
        True if sufficient memory available.

    Raises:
        MemoryError: If insufficient memory with detailed message.
    """
    require(required_gb, available_ram_gb(), operation)
    return True
