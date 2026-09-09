"""Memory pricing and BLAS ownership for bounded batches of LOCO solves."""

from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from itertools import islice
from typing import NamedTuple

import numpy as np

from jamma.core import memory
from jamma.core.eigen_plan import EigenDriverPlan, square_matrix_gb
from jamma.core.threading import blas_threads
from jamma.kinship.loco import LocoRetainedSet

EigenResult = tuple[str, np.ndarray, np.ndarray]
EigenSolver = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class LocoWorkerPlan(NamedTuple):
    """Selected concurrency and the complete consumer memory reservation."""

    workers: int
    memory_allows: int
    cores: int
    consumer_gb: float
    requested: int
    n_chr: int

    def describe(self) -> str:
        if self.requested == 1:
            return "LOCO workers: 1"
        return (
            f"LOCO workers: {self.workers} (requested {self.requested}; "
            f"{self.n_chr} chromosomes; {self.cores} cores; "
            f"memory allows {self.memory_allows})"
        )


def plan_loco_workers(
    requested: int,
    *,
    n_chr: int,
    n_samples: int,
    retained: LocoRetainedSet,
    eigen_plan: EigenDriverPlan,
    available_gb: float,
    budget_gb: float | None,
    association_gb: float,
    cores: int,
) -> LocoWorkerPlan:
    """Price concurrent solves followed by association over completed pairs.

    The driver peak already includes its owned input. After every solve in
    the batch finishes, association retains only the other eigenvectors,
    not their solver workspaces. The kinship stream's retained set is separate.
    One worker borrows the stream buffer and keeps the sequential reservation.
    The caller gates the one-worker floor if even that cannot fit.
    """
    eigenvectors_gb = square_matrix_gb(n_samples)

    def consumer(workers: int) -> float:
        return max(
            workers * eigen_plan.required_gb,
            association_gb + (workers - 1) * eigenvectors_gb,
        )

    def fits(workers: int) -> bool:
        peak = retained.while_consuming_gb + consumer(workers)
        return (budget_gb is None or peak <= budget_gb) and memory.fits(
            peak, available_gb
        )

    # Monotone search keeps even an unbounded environment request cheap and
    # applies the canonical strict RAM margin at every candidate, including ties.
    low, high = 1, requested + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    workers = max(1, min(low, n_chr, cores))
    return LocoWorkerPlan(workers, low, cores, consumer(workers), requested, n_chr)


def solve_eigen_pairs(
    inputs: Iterable[tuple[str, np.ndarray]],
    solve: EigenSolver,
    *,
    workers: int,
    n_threads: int,
) -> Generator[EigenResult, None, None]:
    """Yield ordered pairs only after all solves in their batch have stopped.

    The caller supplies owned matrices for concurrent work and a solver that
    does not change BLAS limits. Each batch has one process-wide BLAS scope.
    It closes before yielding to association, whose own limits can then apply.
    At most ``workers`` inputs/results are retained. Input preparation finishes
    before workers start, and worker cleanup precedes restoring BLAS state.
    One worker runs inline without copying or creating an executor.
    """

    def solve_named(name: str, K: np.ndarray) -> EigenResult:
        return name, *solve(K)

    if workers == 1:
        for name, K in inputs:
            with blas_threads(n_threads):
                result = solve_named(name, K)
            del K
            yield result
            del result
        return

    pending: deque[Future[EigenResult]] = deque()

    def oldest() -> EigenResult:
        # Keep the yielded arrays out of this generator's local variables.
        return pending.popleft().result()

    source = iter(inputs)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        while batch := list(islice(source, workers)):
            with blas_threads(n_threads):
                try:
                    pending.extend(
                        pool.submit(solve_named, name, K) for name, K in batch
                    )
                    wait(pending)
                except BaseException:
                    # Submission errors and interrupts must stop workers
                    # before the BLAS scope restores the process-wide limit.
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
            # A non-inplace solve returns a separate U: release its input
            # before association, matching the planned retained allocations.
            batch.clear()
            while pending:
                yield oldest()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
