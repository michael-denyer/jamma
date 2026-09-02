"""Memory ledger and gate for large-scale GWAS operations.

The estimators price a run's phases from its shape alone; they read no
machine state. The gate (``fits``, ``require``) compares a price against an
``available_gb`` the caller read once through ``available_ram_gb``.
"""

from dataclasses import dataclass

import psutil
from loguru import logger

from jamma.core.constants import n_index
from jamma.core.eigen_plan import (
    _dsyevd_peak_gb,
    array_gb,
    square_matrix_gb,
)


def available_ram_gb() -> float:
    """Available system RAM in GB, the one read every memory decision uses.

    Every gate, the chunk planner's budget, the eigendecomposition driver
    plan, and LOCO's pass planner take their ``available_gb`` from here, so a
    test pins the whole run's view of the machine with one
    ``monkeypatch.setattr(memory, "available_ram_gb", ...)``.
    ``memory_snapshot.py`` reads psutil separately for its logging snapshot;
    nothing decides on that figure.
    """
    return psutil.virtual_memory().available / 1e9


def margin_gb(peak_gb: float) -> float:
    """Safety margin: 10% of *peak_gb*, capped at 10GB absolute.

    The single spelling of the margin. At large scale (500GB+) an uncapped
    10% (50GB) is excessive: OS and process overhead do not scale with the
    eigendecomposition workspace.
    """
    return min(peak_gb * 0.1, 10.0)


def fits(required_gb: float, available_gb: float) -> bool:
    """Whether *required_gb* plus the margin fits in *available_gb*.

    The one inequality every memory gate spells: strict, with the margin
    taken of the requirement, never of the machine.
    """
    return (required_gb + margin_gb(required_gb)) < available_gb


def headroom_gb(available_gb: float) -> float:
    """The largest requirement whose margin still fits in *available_gb*.

    The inverse of ``required + margin_gb(required)``, so a caller sizing a
    batch against a fixed budget can subtract its fixed costs from this
    figure and divide, and land where ``fits`` agrees. Taking
    ``margin_gb(available_gb)`` off the machine instead reserves 10% of the
    machine, which is more than 10% of the requirement below the 10GB cap.
    """
    if available_gb > 110.0:
        return available_gb - 10.0
    return available_gb / 1.1


def require(
    required_gb: float,
    available_gb: float,
    operation: str = "operation",
    *,
    budget_gb: float | None = None,
) -> None:
    """Raise ``MemoryError`` when *required_gb* does not fit the run's ceiling.

    The sole place JAMMA constructs and raises a ``MemoryError``. Every
    memory gate (the pipeline preflight, the LMM batch runner, the kinship
    accumulator, the eigendecomposition driver, and LOCO's pass planner)
    calls this instead of raising its own, so every insufficient-memory
    error looks the same and the message lives in one place.

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
        margin = margin_gb(required_gb)
        message = (
            f"Insufficient memory for {operation}. "
            f"Need {required_gb:.1f}GB (+{margin:.1f}GB margin = "
            f"{required_gb + margin:.1f}GB), but only {available_gb:.1f}GB "
            f"available. Use --no-check-memory to override, or use a machine "
            f"with more RAM."
        )
    raise MemoryError(message)


@dataclass(frozen=True, slots=True)
class MemoryLedger:
    """Peak memory (GB) of each streaming workflow phase.

    The phases do not overlap, so the run's peak is the largest of them.
    Eigendecomposition usually dominates: K and U live together with the
    DSYEVD O(N^2) workspace.
    """

    kinship_gb: float
    eigen_gb: float
    lmm_gb: float

    @property
    def peak_gb(self) -> float:
        """The run's peak requirement, the largest phase."""
        return max(self.kinship_gb, self.eigen_gb, self.lmm_gb)


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
) -> float:
    """Peak memory (GB) of the batch LMM phase, the full-materialization path.

    Use this when eigendecomposition is already complete and kinship has been
    freed: it prices only the LMM phase, not the peak across all workflow
    phases. Genotypes are held in full, O(n * n_snps); the streaming path
    holds one chunk instead (see ``estimate_streaming_memory``).

    Includes Uab_batch (n_chunk, n_samples, n_index) and Iab_batch
    (n_chunk, n_cvt+2, n_index), the dominant intermediates.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        lmm_batch_size: Batch size for LMM SNP processing. Pass the runtime
            chunk size for accurate estimates; the default is a generic guess.
        n_cvt: Number of covariates (default 1).
        n_buffers: Live chunk buffers the engine allocates (1 sequential, 2
            pipelined). Scales both the UtG rotation chunk and the Uab/Iab
            extra, matching ``_ChunkEngine``'s per-buffer allocation.

    Example:
        >>> print(f"LMM needs {estimate_lmm_memory(100_000, 100):.0f}GB")
    """
    eigenvectors_gb = square_matrix_gb(n_samples)
    genotypes_gb = array_gb(n_samples, n_snps)
    eigenvalues_gb = array_gb(n_samples)
    lmm_rotated_gb = 3 * array_gb(n_samples)
    lmm_batch_gb = n_buffers * (
        array_gb(n_samples, lmm_batch_size)
        + _uab_iab_gb(n_samples, lmm_batch_size, n_cvt)
    )
    return (
        eigenvectors_gb + genotypes_gb + eigenvalues_gb + lmm_rotated_gb + lmm_batch_gb
    )


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
) -> MemoryLedger:
    """Price each phase of the streaming GWAS workflow.

    Genotypes are O(n * chunk_size), not O(n * n_snps), so the peak is
    usually eigendecomposition. For 200k samples, 10k chunk, n_grid=50:

    - Kinship accumulation: 320GB + 16GB = 336GB
    - Eigendecomp: 320GB + 320GB + ~640GB = ~1280GB (peak)
    - LMM: 320GB + 16GB + 16GB + Uab/Iab

    Eigendecomposition cannot be streamed. jlinalg.eigh allocates separate
    eigenvectors (K is used as scratch), so that phase holds both K and U
    plus the O(n²) DSYEVD workspace.

    Args:
        n_samples: Number of samples (individuals).
        chunk_size: SNPs per disk chunk, the statistics-pass raw buffer.
        n_grid: Grid points for lambda optimization (default 50).
        n_cvt: Number of covariates (default 1).
        pipeline_buffers: Simultaneous live UtG rotation buffers. Pass 2 when
            rotation-compute pipelining holds current + next buffers.
        compute_chunk_size: SNPs per compute sub-chunk (rotation/Uab/grid
            buffers, and the association-pass raw genotype buffer). Defaults
            to chunk_size. Pass the runtime compute chunk for accurate LMM
            phase estimates.
        eigendecomp_peak_gb: Peak for the eigendecomposition phase, from
            plan_eigen_driver, when the caller knows which driver will run.
            None (default) uses the conservative non-inplace DSYEVD figure.
        uab_iab_gb: Per-chunk LMM intermediate memory, when the caller knows
            the dispatch path (lmm_extra_bytes_per_snp). None (default) uses
            the conservative full Uab/Iab batch figure.

    Example:
        >>> ledger = estimate_streaming_memory(200_000)
        >>> print(f"Peak: {ledger.peak_gb:.0f}GB (eigendecomp)")
    """
    if not isinstance(pipeline_buffers, int):
        raise TypeError(
            f"pipeline_buffers must be an int, got {type(pipeline_buffers).__name__}"
        )
    if pipeline_buffers < 1:
        raise ValueError(f"pipeline_buffers must be >= 1, got {pipeline_buffers}")
    if compute_chunk_size is None:
        compute_chunk_size = chunk_size
    if uab_iab_gb is None:
        uab_iab_gb = _uab_iab_gb(n_samples, compute_chunk_size, n_cvt)

    # The chunk engine's raw-chunk source hands prepare() one buffer at a
    # time even under pipelining (the overlap is between a rotated buffer and
    # the next prepare() call), so the raw LMM chunk does not scale with
    # pipeline_buffers; the rotation buffer does.
    return MemoryLedger(
        kinship_gb=kinship_cost(
            square_matrix_gb(n_samples),
            array_gb(n_samples, chunk_size),
            _dsyrk_scratch_gb(n_samples),
        ),
        eigen_gb=eigen_cost(n_samples, eigendecomp_peak_gb),
        lmm_gb=lmm_cost(
            square_matrix_gb(n_samples),
            array_gb(n_samples, compute_chunk_size),
            array_gb(n_samples, compute_chunk_size) * pipeline_buffers,
            array_gb(n_grid, compute_chunk_size),
            uab_iab_gb,
        ),
    )
