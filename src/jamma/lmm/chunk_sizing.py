"""Chunk-size computation for the shared NumPy LMM chunk engine.

Sizes each genotype chunk against a RAM budget so the UT@G rotation makes as
few DRAM passes over the eigenvector matrix as possible. Split out from
``chunk_runner_numpy`` so the sizing policy (and its C-availability inputs)
lives in one small, testable place.
"""

from __future__ import annotations

import psutil

from jamma.lmm import compute_numpy

# Allow large chunks — no int32 buffer constraint.
_MAX_CHUNK = 200_000

# Memory budget bounds for auto-scaling
_MIN_BUDGET = 2_000_000_000  # 2 GB floor (original default)
_MAX_BUDGET = 40_000_000_000  # 40 GB ceiling


def compute_chunk_size_numpy(
    n_samples: int,
    n_filtered: int,
    n_cvt: int = 1,
    *,
    use_split: bool = False,
    use_fused_general: bool = False,
    mem_budget_bytes: int | None = None,
    pipeline_buffers: int = 1,
) -> int:
    """Compute chunk size based on RAM budget (no int32 constraint for NumPy).

    Scales the memory budget with available RAM to minimise DRAM passes
    through the eigenvector matrix during UT@G rotation.

    Memory accounting turns on whether the path feeds ``utg_t`` straight to a
    C kernel, which allocates 1 col/SNP, or builds SoA-split columns, which
    allocate 4 (3 varying + 1 utg_t). The non-split full-Uab fallback allocates
    (n_cvt+3)(n_cvt+2)/2 cols/SNP.

    Args:
        n_samples: Number of samples.
        n_filtered: Number of filtered SNPs.
        n_cvt: Number of covariates.
        use_split: If True, use split Uab accounting instead of full Uab.
        use_fused_general: If True, fused general path is active (n_cvt>=2);
            only utg_t is allocated (single buffer, no uab_varying_soa).
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

    if use_split and n_cvt == 1:
        if compute_numpy._accel is not None:
            # Every n_cvt=1 C path feeds utg_t straight to the kernel:
            # jlinalg.dgemm(chunk, U, transa="T") produces C-contiguous utg_t
            # (n_snps, n_samples) directly, with no intermediate allocation and
            # no uab_varying_soa. This was four branches, one per mode, each
            # gated on a different capability flag; those flags are one bit, so
            # every mode landed on the same size.
            bytes_per_snp = n_samples * 8
        else:
            # SoA split paths (Wald, Score, LRT, mode-4):
            # 3 varying SoA columns + 1 utg_t per SNP, no Uab reconstruction.
            bytes_per_snp = n_samples * 4 * 8
    elif use_split and n_cvt > 1:
        from jamma.lmm.likelihood import classify_uab_columns

        _inv, var = classify_uab_columns(n_cvt)
        n_var = len(var)
        if use_fused_general:
            # Fused general path: jlinalg.dgemm produces utg_t directly.
            # Single buffer, no intermediate allocation or contiguous copy.
            bytes_per_snp = n_samples * 8
        else:
            # All modes: split C dispatch, no Uab reconstruction.
            # n_var varying SoA columns + 1 utg_t per SNP.
            bytes_per_snp = n_samples * (n_var + 1) * 8
    else:
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        bytes_per_snp = n_samples * n_index * 8

    if bytes_per_snp == 0:
        return n_filtered

    if mem_budget_bytes is not None:
        mem_budget = mem_budget_bytes
    else:
        available = psutil.virtual_memory().available
        # Budget: 15% of available RAM (up from 5%), 2 GB floor, 40 GB ceiling.
        # Modern machines (128-512 GB) can afford larger working sets. The floor
        # prevents degenerate chunk sizes on low-memory systems; the ceiling
        # prevents excessive allocation on high-memory systems.
        mem_budget = max(_MIN_BUDGET, min(int(available * 0.15), _MAX_BUDGET))

    mem_budget = mem_budget // pipeline_buffers

    chunk_from_memory = int(mem_budget / bytes_per_snp)
    chunk = max(100, min(chunk_from_memory, n_filtered, _MAX_CHUNK))
    return chunk
