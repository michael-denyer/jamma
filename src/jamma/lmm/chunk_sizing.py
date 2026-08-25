"""Chunk-size computation for the shared NumPy LMM chunk engine.

Sizes each genotype chunk against a RAM budget so the UT@G rotation makes as
few DRAM passes over the eigenvector matrix as possible. Split out from
``chunk_runner_numpy`` so the sizing policy lives in one small, testable place.
"""

from __future__ import annotations

from jamma.core import memory
from jamma.lmm.dispatch import DispatchPath

# Allow large chunks — no int32 buffer constraint.
_MAX_CHUNK = 200_000

# Memory budget bounds for auto-scaling
_MIN_BUDGET = 2_000_000_000  # 2 GB floor (original default)
_MAX_BUDGET = 40_000_000_000  # 40 GB ceiling


def _bytes_per_snp(n_samples: int, n_cvt: int, dispatch: DispatchPath) -> int:
    """Live float64 bytes one SNP occupies on *dispatch*'s buffers.

    Three accountings, one per shape of chunk input. The fused family hands
    ``utg_t`` straight to its kernel, so the rotation output is the only
    allocation. The SoA-split path adds the varying Uab columns beside it. The
    NumPy fallback materialises the whole Uab table.
    """
    if dispatch.feeds_raw_utg:
        # jlinalg.dgemm(chunk, U, transa="T") writes C-contiguous utg_t
        # directly: one column per SNP, no intermediate and no varying SoA.
        return n_samples * 8

    if dispatch is DispatchPath.SOA_SPLIT:
        from jamma.lmm.likelihood import classify_uab_columns

        n_var = len(classify_uab_columns(n_cvt)[1])
        return n_samples * (n_var + 1) * 8

    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    return n_samples * n_index * 8


def lmm_extra_bytes_per_snp(n_samples: int, n_cvt: int, dispatch: DispatchPath) -> int:
    """Per-SNP bytes live in the LMM phase beyond the UtG rotation buffers.

    The preflight prices the association phase as rotation buffers plus this
    figure, so its estimate follows the same dispatch knowledge the sizer
    uses. Fused paths hold no per-SNP batch arrays (the C workspace forms
    Uab on the fly); SOA_SPLIT holds the varying Uab columns; the NumPy
    fallback materialises the full Uab and Iab batches.
    """
    if dispatch.feeds_raw_utg:
        return 0
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    if dispatch is DispatchPath.SOA_SPLIT:
        from jamma.lmm.likelihood import classify_uab_columns

        n_var = len(classify_uab_columns(n_cvt)[1])
        return n_samples * n_var * 8
    # NUMPY_FALLBACK: Uab batch plus the small Iab batch
    return (n_samples + n_cvt + 2) * n_index * 8


def compute_chunk_size_numpy(
    n_samples: int,
    n_filtered: int,
    n_cvt: int = 1,
    *,
    dispatch: DispatchPath,
    mem_budget_bytes: int | None = None,
    pipeline_buffers: int = 1,
) -> int:
    """Compute chunk size based on RAM budget (no int32 constraint for NumPy).

    Scales the memory budget with available RAM to minimise DRAM passes
    through the eigenvector matrix during UT@G rotation.

    Args:
        n_samples: Number of samples.
        n_filtered: Number of filtered SNPs.
        n_cvt: Number of covariates.
        dispatch: The run's active kernel path, which decides how many float64
            columns per SNP are live at once.
        mem_budget_bytes: Explicit per-chunk memory budget in bytes.
            None (default) auto-scales with available RAM.
        pipeline_buffers: Number of live chunks (1 for sequential,
            2 for pipeline double-buffering). Divides the budget.

    Returns:
        Chunk size (number of SNPs per chunk).
    """
    if not isinstance(pipeline_buffers, int):
        raise TypeError(
            f"pipeline_buffers must be an int, got {type(pipeline_buffers).__name__}"
        )
    if pipeline_buffers < 1:
        raise ValueError(f"pipeline_buffers must be >= 1, got {pipeline_buffers}")

    bytes_per_snp = _bytes_per_snp(n_samples, n_cvt, dispatch)
    if bytes_per_snp == 0:
        return n_filtered

    if mem_budget_bytes is not None:
        mem_budget = mem_budget_bytes
    else:
        available = int(memory.available_ram_gb() * 1e9)
        # Budget: 15% of available RAM (up from 5%), 2 GB floor, 40 GB ceiling.
        # Modern machines (128-512 GB) can afford larger working sets. The floor
        # prevents degenerate chunk sizes on low-memory systems; the ceiling
        # prevents excessive allocation on high-memory systems.
        mem_budget = max(_MIN_BUDGET, min(int(available * 0.15), _MAX_BUDGET))

    mem_budget = mem_budget // pipeline_buffers

    chunk_from_memory = int(mem_budget / bytes_per_snp)
    return max(100, min(chunk_from_memory, n_filtered, _MAX_CHUNK))
