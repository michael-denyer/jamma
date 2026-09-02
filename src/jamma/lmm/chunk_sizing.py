"""Chunk-size computation for the shared NumPy LMM chunk engine.

Sizes each genotype chunk against a RAM budget so the UT@G rotation makes as
few DRAM passes over the eigenvector matrix as possible. Split out from
``chunk_runner_numpy`` so the sizing policy lives in one small, testable place.

``plan_lmm_chunks`` is the one place that decides chunk size, chunk count, and
whether a run pipelines: the chunk engine allocates from its result, and the
memory preflight prices from the same result, so the two cannot compute
different numbers for the same run.
"""

from __future__ import annotations

from dataclasses import dataclass

from jamma.core import memory
from jamma.core.constants import n_index
from jamma.core.threading import is_blas_controllable
from jamma.lmm.dispatch import DispatchPath

# Allow large chunks — no int32 buffer constraint.
_MAX_CHUNK = 200_000

# Memory budget bounds for auto-scaling
_MIN_BUDGET = 2_000_000_000  # 2 GB floor (original default)
_MAX_BUDGET = 40_000_000_000  # 40 GB ceiling

# Minimum number of chunks before pipelined execution is worthwhile.
_MIN_PIPELINE_CHUNKS = 8

# Chunk count a split-capable run is cut to when the memory budget alone would
# leave it below _MIN_PIPELINE_CHUNKS, so rotation of chunk N+1 overlaps compute
# of chunk N even on inputs that fit in one chunk. 16 measured best on
# mouse_hs1940 (12,226 SNPs): Wald -20%, all-tests -10%, 4-covariate Wald
# unchanged.
_PIPELINE_TARGET_CHUNKS = 16

# Largest sample count the cut applies to. Every extra chunk re-streams the
# whole eigenvector matrix through the rotation GEMM, and the kernel time the
# overlap can hide shrinks relative to that GEMM as samples grow, so the cut
# turns from a win into a loss with N. Measured with
# scripts/bench_large_n_stages.py --stages association (5,000 SNPs, interleaved
# ABBA blocks, cut vs no cut): 1,410 samples -20%, 5,000 -6.4%, 10,000 -0.2%,
# 30,000 +5.6%.
_PIPELINE_CUT_MAX_SAMPLES = 10_000

# Chunk floor, so tiny inputs never pay per-chunk overhead for a handful of SNPs.
_MIN_CHUNK = 100


def _bytes_per_snp(n_samples: int, n_cvt: int, dispatch: DispatchPath) -> int:
    """Live float64 bytes one SNP occupies on *dispatch*'s buffers.

    Two accountings. The fused family hands ``utg_t`` straight to its kernel,
    so the rotation output is the only allocation. The NumPy fallback
    materialises the whole Uab table.
    """
    if dispatch.feeds_raw_utg:
        # jlinalg.dgemm(chunk, U, transa="T") writes C-contiguous utg_t
        # directly: one column per SNP, no intermediate.
        return n_samples * 8

    return n_samples * n_index(n_cvt) * 8


def lmm_extra_bytes_per_snp(
    n_samples: int, n_cvt: int, dispatch: DispatchPath, *, n_buffers: int = 1
) -> int:
    """Per-SNP bytes live in the LMM phase beyond the UtG rotation buffers.

    The preflight prices the association phase as rotation buffers plus this
    figure, so its estimate follows the same dispatch knowledge the sizer
    uses. Fused paths hold no per-SNP batch arrays (the C workspace forms
    Uab on the fly); the NumPy fallback materialises the full Uab and Iab
    batches.

    Args:
        n_samples: Number of samples.
        n_cvt: Number of covariates.
        dispatch: The run's active kernel path.
        n_buffers: Live buffer count from the same ``LmmChunkPlan`` the
            engine allocates from (1 sequential, 2 pipelined). Unused by
            every current dispatch path's pricing, kept so a future
            per-buffer-scaled path does not have to change this signature.
    """
    if dispatch.feeds_raw_utg:
        return 0
    # NUMPY_FALLBACK never pipelines (dispatch.use_split is False), so
    # n_buffers is always 1 here; the full Uab+Iab batch is priced once.
    return (n_samples + n_cvt + 2) * n_index(n_cvt) * 8


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
    return max(_MIN_CHUNK, min(chunk_from_memory, n_filtered, _MAX_CHUNK))


@dataclass(frozen=True, slots=True)
class LmmChunkPlan:
    """One run's chunk size, chunk count, and pipelining decision.

    Attributes:
        chunk_size: SNPs per chunk.
        n_chunks: Number of chunks ``n_filtered`` splits into at that size.
        n_buffers: Live chunk buffers the engine allocates (1 sequential, 2
            pipelined).
        use_pipeline: Whether the run overlaps rotation and compute.
    """

    chunk_size: int
    n_chunks: int
    n_buffers: int
    use_pipeline: bool


def plan_lmm_chunks(
    n_samples: int,
    n_filtered: int,
    n_cvt: int,
    dispatch: DispatchPath,
    *,
    max_chunk_size: int | None = None,
    mem_budget_bytes: int | None = None,
) -> LmmChunkPlan:
    """Decide chunk size, chunk count, and pipelining for one LMM run.

    The single sizing decision the chunk engine allocates from and the
    memory preflight prices from: sizes with one live buffer, counts the
    resulting chunks, and re-sizes against two live buffers only when the
    dispatch path supports pipelining (``dispatch.use_split``) and the
    single-buffer chunk count clears ``_MIN_PIPELINE_CHUNKS``. A split-capable
    run of at most ``_PIPELINE_CUT_MAX_SAMPLES`` samples that the budget alone
    leaves below that threshold is cut to ``_PIPELINE_TARGET_CHUNKS`` chunks,
    down to the ``_MIN_CHUNK`` floor, so a small input that fits in one chunk
    still overlaps rotation and compute. A
    caller-given ``max_chunk_size`` caps the final chunk size before the chunk
    count is recomputed, so a capped run still reports its true chunk count.

    Args:
        n_samples: Number of samples.
        n_filtered: Number of filtered SNPs. The preflight calls this with
            the pre-filter SNP count (statistics/MAF/missingness filtering
            has not run yet), which is conservative: it can only ever be
            greater than or equal to the real filtered count, so the chunk
            size it plans is never larger than what the engine will use.
        n_cvt: Number of covariates.
        dispatch: The run's active kernel path.
        max_chunk_size: Optional cap applied before chunk-count recomputation
            (e.g. LOCO's disk-read chunk width).
        mem_budget_bytes: Explicit per-chunk memory budget in bytes, from
            the user's ``--mem-budget``. None auto-scales against available
            RAM as before.

    Returns:
        The plan's chunk size, chunk count, live buffer count, and whether
        the run pipelines.
    """
    if max_chunk_size is not None and max_chunk_size < 1:
        raise ValueError(f"max_chunk_size must be >= 1, got {max_chunk_size}")

    def _sized(*, pipeline_buffers: int, overlap_cap: int | None = None) -> int:
        chunk = compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt,
            dispatch=dispatch,
            mem_budget_bytes=mem_budget_bytes,
            pipeline_buffers=pipeline_buffers,
        )
        if overlap_cap is not None:
            chunk = min(chunk, overlap_cap)
        if max_chunk_size is not None:
            chunk = min(chunk, max_chunk_size)
        return max(1, chunk)

    def _count(chunk_size: int) -> int:
        return (n_filtered + chunk_size - 1) // chunk_size

    # Sized twice: the first size decides whether pipelining is worth it, and
    # a pipelined run then re-sizes against a budget split across two live
    # buffers.
    chunk_size = _sized(pipeline_buffers=1)
    n_chunks = _count(chunk_size)

    # The budget alone leaves a split-capable run too few chunks to overlap
    # rotation with compute, so cut it to _PIPELINE_TARGET_CHUNKS instead. A
    # run the budget already splits past the threshold keeps its plan, and so
    # does one with more samples than the cut is measured to help. The cut
    # only pays where the BLAS cannot be throttled (Accelerate on macOS), so
    # rotation keeps every core while the kernel overlaps it. With a
    # controllable BLAS the pipelined plan splits the cores and re-limits the
    # thread pool per chunk, and the interleaved A/B on an 8-core Linux MKL
    # node measured the cut at +8.3% on mouse_hs1940's shape, against -20% on
    # an 18-core Apple M5 Pro.
    overlap_cap: int | None = None
    if (
        dispatch.use_split
        and n_chunks < _MIN_PIPELINE_CHUNKS
        and n_samples <= _PIPELINE_CUT_MAX_SAMPLES
        and not is_blas_controllable()
    ):
        overlap_cap = max(_MIN_CHUNK, -(-n_filtered // _PIPELINE_TARGET_CHUNKS))
        chunk_size = _sized(pipeline_buffers=1, overlap_cap=overlap_cap)
        n_chunks = _count(chunk_size)
    use_pipeline = dispatch.use_split and n_chunks >= _MIN_PIPELINE_CHUNKS

    if use_pipeline:
        chunk_size = _sized(pipeline_buffers=2, overlap_cap=overlap_cap)
        n_chunks = _count(chunk_size)
        use_pipeline = dispatch.use_split and n_chunks >= _MIN_PIPELINE_CHUNKS

    return LmmChunkPlan(
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        n_buffers=2 if use_pipeline else 1,
        use_pipeline=use_pipeline,
    )


def tighten_lmm_chunks(
    conservative: LmmChunkPlan,
    n_filtered: int,
) -> LmmChunkPlan:
    """Narrow a conservative plan after filtering without selecting policy."""
    if n_filtered < 0:
        raise ValueError(f"n_filtered must be >= 0, got {n_filtered}")

    chunk_size = min(conservative.chunk_size, max(1, n_filtered))
    n_chunks = (n_filtered + chunk_size - 1) // chunk_size
    use_pipeline = conservative.use_pipeline and n_chunks >= _MIN_PIPELINE_CHUNKS
    return LmmChunkPlan(
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        n_buffers=2 if use_pipeline else 1,
        use_pipeline=use_pipeline,
    )
