"""Chunk size computation for LMM association.

Computes optimal chunk sizes to fit within memory budgets during
batch SNP processing. The MAX_SAFE_CHUNK cap also limits JIT
compilation overhead per chunk.
"""

from loguru import logger

# Maximum safe chunk size to prevent excessive JIT overhead and memory spikes.
# 50k SNPs per chunk is safe for most sample sizes while maintaining good throughput.
MAX_SAFE_CHUNK = 50_000


def _get_device_budget_bytes(
    utilization: float = 0.70,
) -> int | None:
    """Query JAX device memory and return a byte budget.

    Returns the per-device memory budget (total device memory * utilization),
    or None if device memory stats are unavailable (CPU-only environments).

    Args:
        utilization: Fraction of device memory to use (default 0.70).

    Returns:
        Memory budget in bytes, or None if unavailable.
    """
    try:
        import jax
    except ImportError:
        return None

    try:
        devices = jax.devices()
        if not devices:
            logger.debug("No JAX devices found; skipping device memory query")
            return None
        stats = devices[0].memory_stats()
        if stats is not None and "bytes_limit" in stats:
            return int(stats["bytes_limit"] * utilization)
        logger.debug("JAX device has no bytes_limit in memory_stats()")
        return None
    except Exception:
        logger.debug("Could not query JAX device memory", exc_info=True)
        return None


def _compute_chunk_size(
    n_snps: int,
    n_devices: int = 1,
    n_samples: int = 0,
    n_cvt: int = 1,
    pipeline_buffers: int = 1,
) -> int:
    """Compute chunk size using device memory budget with alignment.

    When n_samples > 0, uses device memory introspection (GPU) or
    psutil (CPU) to compute chunk size that fills ~70% of available
    memory. Falls back to MAX_SAFE_CHUNK cap when n_samples is 0
    (legacy callers) or memory query fails.

    MAX_SAFE_CHUNK is always applied as an upper bound to limit JIT
    compilation overhead, even when memory-based sizing would allow more.

    Args:
        n_snps: Total number of SNPs (upper bound for chunk size).
        n_devices: Number of JAX virtual CPU devices (default 1).
        n_samples: Number of samples (0 = legacy mode, use MAX_SAFE_CHUNK cap).
        n_cvt: Number of covariates (default 1). Affects Uab memory estimate.
        pipeline_buffers: Number of simultaneous live rotation buffers (default 1).
            Pass 2 when rotation-compute pipelining holds current + next UtG arrays
            simultaneously. Divides the memory budget accordingly (same
            budget-division approach as _compute_chunk_size_numpy).

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
        device_budget = _get_device_budget_bytes()
        if device_budget is None:
            # CPU fallback: use psutil available RAM at 70%
            try:
                import psutil

                device_budget = int(psutil.virtual_memory().available * 0.70)
            except Exception:
                logger.warning(
                    "Could not query system memory via psutil; "
                    "falling back to MAX_SAFE_CHUNK cap",
                    exc_info=True,
                )
                device_budget = None

        if device_budget is not None:
            device_budget = device_budget // pipeline_buffers
            # Uab: n_samples * n_index, UtG: n_samples per SNP
            n_index = (n_cvt + 3) * (n_cvt + 2) // 2
            bytes_per_snp = n_samples * (n_index + 1) * 8
            if bytes_per_snp > 0:
                chunk_from_memory = int(device_budget / bytes_per_snp)
                chunk = max(1000, min(chunk_from_memory, n_snps, MAX_SAFE_CHUNK))

    # Align to device count multiples to prevent XLA padding partial shards
    if n_devices > 1 and chunk < n_snps:
        aligned = (chunk // n_devices) * n_devices
        # aligned==0 means chunk < n_devices: skip alignment
        chunk = aligned if aligned > 0 else chunk

    return max(1, chunk)


def compute_subchunk_starts(
    n_subset: int,
    chunk_size: int,
    n_devices: int,
) -> list[int]:
    """Compute sub-chunk start indices that avoid tiny tails.

    When multi-device sharding is active, a tail sub-chunk with fewer
    SNPs than n_devices causes an IndivisibleError because JAX cannot
    shard the tail across the mesh.  This function merges any too-small
    tail into the preceding sub-chunk.

    Args:
        n_subset: Total SNPs in the outer (file) chunk.
        chunk_size: JAX sub-chunk size (already device-aligned).
        n_devices: Number of JAX virtual CPU devices.

    Returns:
        List of start indices for sub-chunks within the outer chunk.
    """
    starts = list(range(0, n_subset, chunk_size))
    if n_devices > 1 and len(starts) > 1:
        tail = n_subset - starts[-1]
        if tail < n_devices:
            # Merge tail into previous sub-chunk
            starts.pop()
    return starts


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
