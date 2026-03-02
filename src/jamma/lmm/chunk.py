"""Chunk size computation for JAX LMM association.

Computes optimal chunk sizes to fit within memory budgets during
batch SNP processing. The MAX_SAFE_CHUNK cap also limits JIT
compilation overhead per chunk.
"""

from loguru import logger

# Maximum safe chunk size to prevent excessive JIT overhead and memory spikes.
# 50k SNPs per chunk is safe for most sample sizes while maintaining good throughput.
MAX_SAFE_CHUNK = 50_000


def _compute_chunk_size(
    n_samples: int,
    n_snps: int,
    n_grid: int = 50,
    n_cvt: int = 1,
    n_devices: int = 1,
) -> int:
    """Compute chunk size capped at MAX_SAFE_CHUNK with device alignment.

    This function applies the MAX_SAFE_CHUNK cap and device alignment.
    It does NOT apply a memory budget — callers that need memory-aware
    sizing should use auto_tune_chunk_size() instead.

    When n_devices > 1, the chunk size is rounded down to a multiple of
    n_devices to prevent XLA from padding partial shards.

    Args:
        n_samples: Unused. Retained for caller compatibility.
        n_snps: Total number of SNPs (upper bound for chunk size).
        n_grid: Unused. Retained for caller compatibility.
        n_cvt: Unused. Retained for caller compatibility.
        n_devices: Number of JAX virtual CPU devices (default 1). When > 1,
            the result is rounded down to a multiple of n_devices.

    Returns:
        Chunk size (number of SNPs per chunk). Returns n_snps if it fits
        within MAX_SAFE_CHUNK. When n_devices > 1, the chunk is
        rounded down to a multiple of n_devices.
    """
    chunk = min(n_snps, MAX_SAFE_CHUNK)

    # Align to device count multiples to prevent XLA padding partial shards
    if n_devices > 1 and chunk < n_snps:
        aligned = (chunk // n_devices) * n_devices
        # aligned==0 means chunk < n_devices: skip alignment
        chunk = aligned if aligned > 0 else chunk

    return max(1, chunk)


def auto_tune_chunk_size(
    n_samples: int,
    n_filtered: int,
    n_grid: int = 50,
    mem_budget_gb: float = 4.0,
    min_chunk: int = 1000,
    max_chunk: int = MAX_SAFE_CHUNK,
    n_cvt: int = 1,
    n_devices: int = 1,
) -> int:
    """Compute optimal chunk size based on memory budget heuristic.

    Uses a deterministic formula to compute chunk size that fits within
    memory budget. No benchmarking required - fast and predictable.

    Memory per SNP (float64):
      - Uab: n_samples * n_index elements (n_index depends on n_cvt)
      - UtG_chunk: n_samples elements
      - Grid evaluations: n_grid elements
      - Total: 8 * (n_samples*n_index + n_samples + n_grid) bytes

    When n_devices > 1, the final chunk size is rounded down to a multiple
    of n_devices to prevent XLA from padding partial shards.

    Args:
        n_samples: Number of samples in the dataset.
        n_filtered: Number of SNPs after filtering (upper bound for chunk).
        n_grid: Grid points for lambda optimization (default 50).
        mem_budget_gb: Memory budget in GB (default 4.0).
        min_chunk: Minimum chunk size (default 1000).
        max_chunk: Maximum chunk size cap (default MAX_SAFE_CHUNK=50000).
            Prevents excessive memory allocation on high-memory systems.
        n_cvt: Number of covariates (default 1). Affects Uab array size.
        n_devices: Number of JAX virtual CPU devices (default 1). When > 1,
            the result is rounded down to a multiple of n_devices.

    Returns:
        Optimal chunk size that fits within memory budget. Capped by n_filtered
        and max_chunk. When n_devices > 1, rounded down to a multiple of n_devices.

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
    chunk_from_memory = (
        int(mem_budget_bytes / bytes_per_snp) if bytes_per_snp > 0 else n_filtered
    )

    # Clamp to bounds
    chunk_size = max(min_chunk, min(chunk_from_memory, n_filtered, max_chunk))

    # Align to n_devices
    if n_devices > 1 and chunk_size % n_devices != 0:
        chunk_size = max(n_devices, (chunk_size // n_devices) * n_devices)

    # Re-apply ceiling after alignment — alignment rounds down but
    # max(n_devices, ...) could exceed n_filtered/max_chunk in edge cases
    chunk_size = min(chunk_size, n_filtered, max_chunk)

    logger.debug(
        f"auto_tune_chunk_size: n_samples={n_samples}, n_filtered={n_filtered}, "
        f"bytes_per_snp={bytes_per_snp}, chunk_size={chunk_size}, "
        f"max_chunk={max_chunk}, n_devices={n_devices}"
    )

    return chunk_size
