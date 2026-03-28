"""Chunk size computation for LMM association.

Computes optimal chunk sizes to fit within memory budgets during
batch SNP processing. The MAX_SAFE_CHUNK cap limits memory spikes.

``_compute_chunk_size`` and ``MAX_SAFE_CHUNK`` are canonical in
:mod:`jamma.core.chunk` and re-exported here for backwards compatibility.
"""

from loguru import logger

from jamma.core.chunk import MAX_SAFE_CHUNK, _compute_chunk_size

__all__ = ["MAX_SAFE_CHUNK", "_compute_chunk_size", "auto_tune_chunk_size"]


def auto_tune_chunk_size(
    n_samples: int,
    n_filtered: int,
    n_grid: int = 50,
    mem_budget_gb: float = 4.0,
    min_chunk: int = 1000,
    max_chunk: int = MAX_SAFE_CHUNK,
    n_cvt: int = 1,
) -> int:
    """Compute optimal chunk size based on memory budget heuristic.

    Uses a deterministic formula to compute chunk size that fits within
    memory budget. No benchmarking required - fast and predictable.

    Memory per SNP (float64):
      - Uab: n_samples * n_index elements (n_index depends on n_cvt)
      - UtG_chunk: n_samples elements
      - Grid evaluations: n_grid elements
      - Total: 8 * (n_samples*n_index + n_samples + n_grid) bytes

    Args:
        n_samples: Number of samples in the dataset.
        n_filtered: Number of SNPs after filtering (upper bound for chunk).
        n_grid: Grid points for lambda optimization (default 50).
        mem_budget_gb: Memory budget in GB (default 4.0).
        min_chunk: Minimum chunk size (default 1000).
        max_chunk: Maximum chunk size cap (default MAX_SAFE_CHUNK=50000).
            Prevents excessive memory allocation on high-memory systems.
        n_cvt: Number of covariates (default 1). Affects Uab array size.

    Returns:
        Optimal chunk size that fits within memory budget. Capped by n_filtered
        and max_chunk.

    Example:
        >>> chunk = auto_tune_chunk_size(n_samples=10000, n_filtered=50000)
        >>> results = run_lmm_association_streaming(..., chunk_size=chunk)
    """
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2

    # Memory per SNP in bytes (float64 = 8 bytes)
    # Uab: (n_samples, n_index), UtG: (n_samples,), grid workspace: (n_grid,)
    bytes_per_snp = 8 * (n_samples * n_index + n_samples + n_grid)

    # Compute chunk size with 70% safety margin for overhead
    mem_budget_bytes = mem_budget_gb * 0.7 * 1e9
    chunk_from_memory = (
        int(mem_budget_bytes / bytes_per_snp) if bytes_per_snp > 0 else n_filtered
    )

    # Clamp to bounds
    chunk_size = max(min_chunk, min(chunk_from_memory, n_filtered, max_chunk))

    # Re-apply n_filtered/max_chunk ceiling -- min_chunk can exceed them
    chunk_size = min(chunk_size, n_filtered, max_chunk)

    logger.debug(
        f"auto_tune_chunk_size: n_samples={n_samples}, n_filtered={n_filtered}, "
        f"bytes_per_snp={bytes_per_snp}, chunk_size={chunk_size}, "
        f"max_chunk={max_chunk}"
    )

    return chunk_size
