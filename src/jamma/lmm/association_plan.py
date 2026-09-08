"""Association policy selected once; its chunk plan is narrowed after SNP filtering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from jamma.core import memory
from jamma.core.eigen_plan import EigenDriverPlan, array_gb, square_matrix_gb
from jamma.core.memory import estimate_lmm_memory, estimate_streaming_memory
from jamma.core.threading import get_c_extension_thread_count, is_blas_controllable
from jamma.lmm import accel
from jamma.lmm.chunk_sizing import (
    LmmChunkPlan,
    chunk_budget_bytes,
    lmm_extra_bytes_per_snp,
)
from jamma.lmm.dispatch import DispatchPath, select_dispatch_path
from jamma.lmm.schema import LmmMode, parse_lmm_mode
from jamma.lmm.workspace import WorkspaceSpec

ExecutionMode = Literal["batch", "streaming", "loco"]
RequestedBackend = Literal["auto", "numpy", "numpy-streaming"]

# SNPs per block in the streaming statistics pass when the caller names no
# chunk size. Pass 1 reads the .bed and accumulates per-SNP counts, so its
# footprint is one block of genotypes rather than the rotation and grid
# buffers the association pass carries; it needs no RAM-budgeted sizing of
# its own. Declared here, next to the pricing that assumes it, and imported
# by the executing side (runner_numpy_streaming) so the two cannot drift.
DEFAULT_STATS_CHUNK = 10_000


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Public summary of the selected execution mode."""

    mode: ExecutionMode
    reason: str = field(compare=False)

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("ExecutionPlan.reason must be non-empty")

    @property
    def runner_name(self) -> str:
        """Return the runner label used in logs and telemetry."""
        return f"numpy-{self.mode}"


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """Memory quote for one already-selected association plan.

    One figure per phase of the run and their maximum. Pure: the gating call
    sites read ``available_gb`` once themselves and hand both figures to
    ``memory.require``.
    """

    total_peak_gb: float
    compute_chunk_size: int
    kinship_gb: float
    eigen_gb: float
    statistics_gb: float
    association_gb: float


@dataclass(frozen=True, slots=True)
class KinshipShape:
    """The kinship matrix a run materialises before its eigendecomposition.

    ``n_samples`` is the matrix order: the input sample count when the matrix
    is read from disk or saved (both happen at full size), else the analysed
    count. ``loaded`` means one square read replaces the accumulation pass
    over the genotypes.
    """

    n_samples: int
    loaded: bool

    @classmethod
    def resolve(
        cls, n_samples: int, n_input_samples: int, *, loaded: bool, saved: bool
    ) -> KinshipShape:
        """The input order when loaded or saved, else the analysed order."""
        return cls(n_input_samples if loaded or saved else n_samples, loaded)


@dataclass(frozen=True, slots=True)
class ExecutableAssociationPlan:
    """Selected mode, dispatch, conservative geometry, and memory policy.

    ``kinship`` is None when the run materialises no kinship matrix: the
    eigenpairs are provided, or a standalone runner already holds it.
    """

    summary: ExecutionPlan
    dispatch: DispatchPath
    conservative_chunks: LmmChunkPlan
    n_samples: int
    n_input_samples: int
    n_snps_before_filter: int
    n_cvt: int
    mem_budget_gb: float | None
    workspace: WorkspaceSpec
    phenotype_group_size: int = 1
    kinship: KinshipShape | None = None

    def __post_init__(self) -> None:
        if self.phenotype_group_size < 1:
            raise ValueError("phenotype_group_size must be >= 1")

    @property
    def resolved_kinship(self) -> KinshipShape:
        """The kinship shape of a plan that materialises a kinship matrix."""
        if self.kinship is None:
            raise ValueError("this association plan materialises no kinship matrix")
        return self.kinship

    def _group_workspace_bytes(self) -> int:
        """Fixed bytes for all phenotype kernels live in one bounded group."""
        group_size = self.phenotype_group_size
        if not self.dispatch.is_native:
            kernel_bytes = self.workspace.fixed_bytes
        else:
            per_kernel = self.workspace.persistent_bytes + (
                self.workspace.max_threads * self.workspace.per_thread_bytes
            )
            shared_transient = (
                self.workspace.max_threads * self.workspace.transient_per_thread_bytes
            )
            kernel_bytes = group_size * per_kernel + shared_transient
        if group_size == 1:
            return kernel_bytes

        # Each extra phenotype retains Uty and Hi_eval plus the invariant Uab
        # rows used to construct its kernel. Covariates, eigenpairs, rotation
        # buffers, and fallback compute scratch are shared or used sequentially.
        # The filtered phenotype input remains live while Uty and the null
        # model are prepared. The null solve also returns Hi_eval. Count all
        # three analysed-sample vectors for every additional live phenotype.
        rows = 3 + self.dispatch.invariant_rows(self.n_cvt)
        if self.dispatch.needs_null_w:
            rows += 1
        prepared_bytes = (group_size - 1) * rows * self.n_samples * 8
        return kernel_bytes + prepared_bytes

    def price(self, *, eigen: EigenDriverPlan | None) -> MemoryPlan:
        """Price every phase of the run this plan describes; the peak is their max.

        ``eigen=None`` means no decomposition runs: the eigenpairs are
        provided, or the caller is sizing the association phase alone.
        """
        chunks = self.conservative_chunks
        kinship_gb = (
            0.0 if self.kinship is None else self._kinship_phase_gb(self.kinship)
        )
        eigen_gb = 0.0 if eigen is None else eigen.required_gb
        statistics_gb = (
            self._statistics_phase_gb() if self.summary.mode == "streaming" else 0.0
        )
        association_gb = self._association_phase_gb()
        return MemoryPlan(
            total_peak_gb=max(kinship_gb, eigen_gb, statistics_gb, association_gb),
            compute_chunk_size=chunks.chunk_size,
            kinship_gb=kinship_gb,
            eigen_gb=eigen_gb,
            statistics_gb=statistics_gb,
            association_gb=association_gb,
        )

    def _kinship_phase_gb(self, kinship: KinshipShape) -> float:
        """Peak while the kinship matrix is read or accumulated."""
        if kinship.loaded:
            phase_gb = square_matrix_gb(kinship.n_samples)
        else:
            # The accumulator plus one genotype block over every input sample;
            # the streamer subsets each block after reading it.
            phase_gb = estimate_streaming_memory(
                kinship.n_samples, chunk_size=DEFAULT_STATS_CHUNK
            ).kinship_gb + array_gb(
                self.n_input_samples - kinship.n_samples, DEFAULT_STATS_CHUNK
            )
        if kinship.n_samples != self.n_samples:
            # The full matrix and its analysed-sample copy are live together.
            phase_gb = max(
                phase_gb,
                square_matrix_gb(kinship.n_samples) + square_matrix_gb(self.n_samples),
            )
        return phase_gb

    def _statistics_phase_gb(self) -> float:
        """Peak of streaming pass 1, which reads SNP statistics with U live.

        The blocks are ``DEFAULT_STATS_CHUNK`` wide over every input sample.
        """
        return square_matrix_gb(self.n_samples) + array_gb(
            self.n_input_samples, DEFAULT_STATS_CHUNK
        )

    def _association_phase_gb(self) -> float:
        """Peak of the association pass at this chunk width and phenotype group."""
        chunks = self.conservative_chunks
        workspace_gb = (
            self._group_workspace_bytes()
            + chunks.chunk_size * self.workspace.bytes_per_snp
        ) / 1e9
        input_subset_rows = self.n_input_samples - self.n_samples
        if self.summary.mode == "batch":
            batch_arrays_gb = estimate_lmm_memory(
                self.n_samples,
                self.n_snps_before_filter,
                lmm_batch_size=chunks.chunk_size,
                n_buffers=chunks.n_buffers,
                n_grid=0,
                uab_iab_gb=(
                    chunks.chunk_size
                    * lmm_extra_bytes_per_snp(self.n_samples, self.n_cvt, self.dispatch)
                    / 1e9
                ),
            )
            return (
                batch_arrays_gb
                + array_gb(input_subset_rows, self.n_snps_before_filter)
                + workspace_gb
            )
        # Streaming and LOCO both hold one genotype chunk, never the matrix.
        ledger = estimate_streaming_memory(
            self.n_samples,
            chunk_size=DEFAULT_STATS_CHUNK,
            n_cvt=self.n_cvt,
            pipeline_buffers=chunks.n_buffers,
            compute_chunk_size=chunks.chunk_size,
            n_grid=0,
            uab_iab_gb=(
                chunks.chunk_size
                * lmm_extra_bytes_per_snp(
                    self.n_samples,
                    self.n_cvt,
                    self.dispatch,
                    n_buffers=chunks.n_buffers,
                )
                / 1e9
            ),
        )
        return (
            ledger.lmm_gb
            + array_gb(input_subset_rows, chunks.chunk_size)
            + workspace_gb
        )


def _quote_fits(plan: ExecutableAssociationPlan, *, available_gb: float) -> bool:
    """Apply the same two ceilings as ``memory.require`` to the association phase.

    Narrowing the chunk or the phenotype group moves only that phase. The
    kinship, eigen, and statistics figures are fixed by the sample counts,
    so the preflight gates them once on the whole quote.
    """
    required_gb = plan.price(eigen=None).association_gb
    within_budget = plan.mem_budget_gb is None or required_gb <= plan.mem_budget_gb
    return within_budget and memory.fits(required_gb, available_gb)


def _narrow_chunks_to_fit(
    plan: ExecutableAssociationPlan, *, available_gb: float
) -> ExecutableAssociationPlan:
    """Choose the widest chunk whose completed quote passes both ceilings."""
    chunks = plan.conservative_chunks
    if _quote_fits(plan, available_gb=available_gb):
        return plan

    def candidate(width: int) -> ExecutableAssociationPlan:
        return replace(
            plan,
            conservative_chunks=chunks.cap_width(plan.n_snps_before_filter, width),
        )

    one = candidate(1)
    if not _quote_fits(one, available_gb=available_gb):
        # Preserve the unavoidable one-SNP quote for the final shared gate.
        return one

    upper = chunks.chunk_size
    low = 1
    widest = one
    while low <= upper:
        mid = (low + upper) // 2
        proposed = candidate(mid)
        if _quote_fits(proposed, available_gb=available_gb):
            widest = proposed
            low = mid + 1
        else:
            upper = mid - 1
    return widest


def _select_phenotype_group(
    plan: ExecutableAssociationPlan,
    *,
    n_phenotypes: int,
    available_gb: float,
) -> ExecutableAssociationPlan:
    """Preserve the widest feasible chunk, then fit the largest live group."""
    single = _narrow_chunks_to_fit(
        replace(plan, phenotype_group_size=1), available_gb=available_gb
    )
    if not _quote_fits(single, available_gb=available_gb) or n_phenotypes == 1:
        return single

    low = 1
    high = n_phenotypes
    largest = single
    while low <= high:
        mid = (low + high) // 2
        proposed = replace(single, phenotype_group_size=mid)
        if _quote_fits(proposed, available_gb=available_gb):
            largest = proposed
            low = mid + 1
        else:
            high = mid - 1
    return largest


def plan_association(
    n_samples: int,
    n_snps: int,
    *,
    requested: RequestedBackend = "auto",
    n_cvt: int = 1,
    lmm_mode: LmmMode = 1,
    n_input_samples: int | None = None,
    n_grid: int = 50,
    n_refine: int = 20,
    n_phenotypes: int = 1,
    mem_budget: float | None = None,
    max_chunk_size: int | None = None,
    log_dispatch_choices: bool = False,
    loco: bool = False,
) -> ExecutableAssociationPlan:
    """Select all association policy and conservative geometry once.

    Plans every public backend with either native compute or the real NumPy
    fallback. Streaming is a storage policy and remains available when the C
    extension is absent.

    ``loco=True`` selects the ``loco`` mode regardless of ``requested``: the
    LOCO orchestrator runs the NumPy body per chromosome over disk-read
    chunks, so it is priced like streaming (one chunk plus the
    eigendecomposition), never like batch. ``n_snps`` is then the run total,
    and ``max_chunk_size`` should be the LOCO disk-read chunk width.
    """
    valid_requests = ("auto", "numpy", "numpy-streaming")
    if requested not in valid_requests:
        raise ValueError(
            f"Unknown backend {requested!r}. Must be one of {valid_requests}."
        )

    c_ext_available = accel.available()
    mode = parse_lmm_mode(lmm_mode)
    if n_input_samples is None:
        n_input_samples = n_samples
    if n_input_samples < n_samples:
        raise ValueError("n_input_samples must be >= analysed n_samples")
    if n_phenotypes < 1:
        raise ValueError("n_phenotypes must be >= 1")
    if loco and n_phenotypes != 1:
        raise ValueError("LOCO supports one phenotype per execution plan")
    dispatch = select_dispatch_path(
        n_cvt, mode, accel=c_ext_available, log_choices=log_dispatch_choices
    )
    # The machine is read here, once, so the planner and the pricing stay pure.
    available_gb = memory.available_ram_gb()
    max_workspace_threads = get_c_extension_thread_count(
        c_ext_available, accel.HAS_OPENMP
    )
    workspace = WorkspaceSpec.build(
        dispatch,
        mode,
        n_samples,
        n_input_samples,
        n_cvt,
        n_grid,
        n_refine,
        max_workspace_threads,
    )
    chunks = LmmChunkPlan.plan(
        n_samples,
        n_snps,
        n_cvt,
        dispatch,
        budget_bytes=chunk_budget_bytes(
            mem_budget, available_bytes=int(available_gb * 1e9)
        ),
        blas_controllable=is_blas_controllable(),
        max_chunk_size=max_chunk_size,
        fixed_bytes=workspace.fixed_bytes + n_samples * n_samples * 8,
        output_bytes_per_snp=(
            workspace.bytes_per_snp + max(0, n_input_samples - n_samples) * 8
        ),
    )

    if loco:
        summary = ExecutionPlan("loco", "LOCO per-chromosome NumPy runs")
    elif requested == "numpy-streaming":
        summary = ExecutionPlan("streaming", "Explicit numpy-streaming request")
    elif requested == "numpy":
        summary = ExecutionPlan("batch", "NumPy backend explicitly requested")
    else:
        summary = ExecutionPlan("batch", "Evaluating NumPy batch capacity")

    plan = ExecutableAssociationPlan(
        summary=summary,
        dispatch=dispatch,
        conservative_chunks=chunks,
        n_samples=n_samples,
        n_input_samples=n_input_samples,
        n_snps_before_filter=n_snps,
        n_cvt=n_cvt,
        mem_budget_gb=mem_budget,
        workspace=workspace,
    )
    if requested == "auto" and not loco:
        batch_gb = plan.price(eigen=None).association_gb
        capacity_gb = min(
            available_gb,
            mem_budget if mem_budget is not None else available_gb,
        )
        if _quote_fits(plan, available_gb=available_gb):
            summary = ExecutionPlan(
                "batch", f"{batch_gb:.1f}GB fits in {capacity_gb:.1f}GB capacity"
            )
        else:
            summary = ExecutionPlan(
                "streaming",
                f"{batch_gb:.1f}GB exceeds {capacity_gb:.1f}GB capacity, "
                "using NumPy streaming",
            )
        plan = replace(plan, summary=summary)

    return _select_phenotype_group(
        plan,
        n_phenotypes=n_phenotypes,
        available_gb=available_gb,
    )
