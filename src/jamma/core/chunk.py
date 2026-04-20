"""Chunk size computation for memory-bounded processing.

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
        except (ImportError, OSError, AttributeError):
            # ImportError: psutil not installed.
            # OSError: /proc unreadable inside restrictive containers.
            # AttributeError: older psutil lacks virtual_memory().available
            # (seen on ancient Databricks runtimes).
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
