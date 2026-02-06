"""Chunk size computation for JAX LMM association.

Computes optimal chunk sizes to avoid JAX int32 buffer overflow
and fit within memory budgets during batch SNP processing.
"""

from loguru import logger

# Maximum safe chunk size to prevent int32 overflow and excessive memory allocation
# 50k SNPs per chunk is safe for most sample sizes while maintaining good throughput
MAX_SAFE_CHUNK = 50_000

# INT32_MAX with headroom for JAX internal indexing overhead
# Multiple arrays contribute to buffer sizing:
# - Uab: (n_snps, n_samples, n_index) where n_index = (n_cvt+3)*(n_cvt+2)//2
# - Grid REML intermediate: (n_grid, n_snps) during vmap over lambdas
# - UtG_chunk: (n_samples, n_snps)
#
# The bottleneck is the grid REML vmap which creates (n_grid, n_snps) intermediate
# tensors. Total elements must stay below INT32_MAX.
_MAX_BUFFER_ELEMENTS = 1_700_000_000  # ~1.7B elements, 80% of INT32_MAX


def _compute_chunk_size(
    n_samples: int, n_snps: int, n_grid: int = 50, n_cvt: int = 1
) -> int:
    """Compute optimal chunk size to avoid int32 buffer overflow.

    JAX uses int32 for buffer indexing by default. Multiple arrays contribute:
    1. Uab: (chunk_size, n_samples, n_index) = chunk_size * n_samples * n_index
    2. Grid REML: (n_grid, chunk_size) intermediate = n_grid * chunk_size
    3. UtG_chunk: (n_samples, chunk_size) = n_samples * chunk_size

    n_index depends on n_cvt: n_index = (n_cvt+3)*(n_cvt+2)//2.
    For n_cvt=1: n_index=6, for n_cvt=2: n_index=10.

    The most restrictive constraint is typically Uab for large n_samples.

    Args:
        n_samples: Number of samples.
        n_snps: Total number of SNPs.
        n_grid: Grid points for lambda optimization (default 50).
        n_cvt: Number of covariates (default 1).

    Returns:
        Chunk size (number of SNPs per chunk). Returns n_snps if no chunking needed.
    """
    if n_samples == 0:
        return n_snps

    n_index = (n_cvt + 3) * (n_cvt + 2) // 2

    # Calculate elements per SNP for each array type
    # Most restrictive constraint is typically Uab: (chunk_size, n_samples, n_index)
    elements_per_snp = max(
        n_samples * n_index,  # Uab: n_samples * n_index elements per SNP
        n_grid,  # Grid REML: (n_grid, chunk_size) intermediate -> n_grid per SNP
        n_samples,  # UtG_chunk: n_samples elements per SNP
    )

    if elements_per_snp == 0:
        return n_snps

    max_snps_per_chunk = _MAX_BUFFER_ELEMENTS // elements_per_snp

    if max_snps_per_chunk >= n_snps:
        return n_snps

    return max(100, max_snps_per_chunk)


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
        Optimal chunk size that fits within memory budget.

    Example:
        >>> chunk = auto_tune_chunk_size(n_samples=10000, n_filtered=50000)
        >>> results = run_lmm_association_streaming(..., chunk_size=chunk)
    """
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2

    # Memory per SNP in bytes (float64 = 8 bytes)
    # Uab: (n_samples, n_index), UtG: (n_samples,), grid workspace: (n_grid,)
    bytes_per_snp = 8 * (n_samples * n_index + n_samples + n_grid)

    # Compute chunk size with 70% safety margin for JAX overhead
    mem_budget_bytes = mem_budget_gb * 0.7 * 1e9
    chunk_from_memory = int(mem_budget_bytes / bytes_per_snp)

    # Apply int32 buffer limit constraint
    buffer_limit = _compute_chunk_size(n_samples, chunk_from_memory, n_grid, n_cvt)

    # Clamp to valid range INCLUDING max_chunk cap
    chunk_size = max(min_chunk, min(buffer_limit, n_filtered, max_chunk))

    logger.debug(
        f"auto_tune_chunk_size: n_samples={n_samples}, n_filtered={n_filtered}, "
        f"bytes_per_snp={bytes_per_snp}, chunk_size={chunk_size}, max_chunk={max_chunk}"
    )

    return chunk_size
