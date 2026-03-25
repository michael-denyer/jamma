"""Chunk size computation for LMM association.

Computes optimal chunk sizes to fit within memory budgets during
batch SNP processing. The MAX_SAFE_CHUNK cap limits memory spikes.
"""

from loguru import logger

# Maximum safe chunk size to prevent excessive memory spikes.
# 50k SNPs per chunk is safe for most sample sizes while maintaining good throughput.
MAX_SAFE_CHUNK = 50_000


def _compute_chunk_size(
    n_snps: int,
    n_samples: int = 0,
    n_cvt: int = 1,
    pipeline_buffers: int = 1,
) -> int:
    """Compute chunk size using system memory budget.

    When n_samples > 0, uses psutil to compute chunk size that fills
    ~70% of available memory. Falls back to MAX_SAFE_CHUNK cap when
    n_samples is 0 (legacy callers) or memory query fails.

    MAX_SAFE_CHUNK is always applied as an upper bound.

    Args:
        n_snps: Total number of SNPs (upper bound for chunk size).
        n_samples: Number of samples (0 = legacy mode, use MAX_SAFE_CHUNK cap).
        n_cvt: Number of covariates (default 1). Affects Uab memory estimate.
        pipeline_buffers: Number of simultaneous live rotation buffers (default 1).
            Pass 2 when rotation-compute pipelining holds current + next UtG arrays
            simultaneously. Divides the memory budget accordingly.

    Returns:
        Chunk size (number of SNPs per chunk).
    """
    if not isinstance(pipeline_buffers, int):
        raise TypeError(
            f"pipeline_buffers must be an int, got {type(pipeline_buffers).__name__}"
        )
    if pipeline_buffers < 1:
        raise ValueError(f"pipeline_buffers must be >= 1, got {pipeline_buffers}")

    chunk = min(n_snps, MAX_SAFE_CHUNK)  # default cap

    if n_samples > 0:
        # Use psutil available RAM at 70%
        device_budget = None
        try:
            import psutil

            device_budget = int(psutil.virtual_memory().available * 0.70)
        except Exception:
            logger.warning(
                "Could not query system memory via psutil; "
                "falling back to MAX_SAFE_CHUNK cap",
                exc_info=True,
            )

        if device_budget is not None:
            device_budget = device_budget // pipeline_buffers
            # Uab: n_samples * n_index, UtG: n_samples per SNP
            n_index = (n_cvt + 3) * (n_cvt + 2) // 2
            bytes_per_snp = n_samples * (n_index + 1) * 8
            if bytes_per_snp > 0:
                chunk_from_memory = int(device_budget / bytes_per_snp)
                chunk = max(1000, min(chunk_from_memory, n_snps, MAX_SAFE_CHUNK))

    return max(1, chunk)


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
