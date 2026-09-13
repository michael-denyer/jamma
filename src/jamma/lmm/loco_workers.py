"""Memory pricing and BLAS ownership for concurrent LOCO eigen solves."""

import queue
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future
from typing import NamedTuple

import numpy as np

from jamma.core import memory
from jamma.core.eigen_plan import EigenDriverPlan
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
    retained: LocoRetainedSet,
    eigen_plan: EigenDriverPlan,
    available_gb: float,
    budget_gb: float | None,
    association_gb: float,
    cores: int,
) -> LocoWorkerPlan:
    """Price ``workers`` solves in flight while association consumes the oldest.

    Each driver peak already includes its owned input, so a worker costs
    ``eigen_plan.required_gb`` and nothing more. The consumer's moment of
    peak is the larger of every worker solving at once and association
    running while the other ``workers - 1`` still solve. The kinship stream's
    retained set sits underneath both. The caller gates the one-worker floor
    if even that cannot fit.
    """

    def consumer(workers: int) -> float:
        return (workers - 1) * eigen_plan.required_gb + max(
            eigen_plan.required_gb, association_gb
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
    """Yield eigenpairs in input order with at most ``workers`` solves in flight.

    One process-wide BLAS scope, entered on the consumer's thread, wraps the
    whole stream. ``solve`` never changes thread limits, so every limit change
    happens on that one thread in nested order: association's own scope opens
    and closes inside this one between pulls. While association runs, solves
    still in flight inherit its limit, which is oversubscription on MKL and
    OpenBLAS, not a stale restore. The scope closes, after the pool has
    stopped, when the stream is drained or the generator is closed early.

    Workers are daemon threads rather than a ``ThreadPoolExecutor``: a
    ``KeyboardInterrupt`` on the consumer's thread propagates without joining
    a solve still inside LAPACK, so Ctrl-C ends the run as promptly as it
    ends GEMMA. Every other exit joins the workers.
    """

    def solve_named(name: str, K: np.ndarray) -> EigenResult:
        return name, *solve(K)

    work: queue.SimpleQueue[tuple[Future[EigenResult], str, np.ndarray] | None] = (
        queue.SimpleQueue()
    )

    def run() -> None:
        while (item := work.get()) is not None:
            future, name, K = item
            del item
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(solve_named(name, K))
                except BaseException as exc:  # noqa: BLE001 — worker thread routes every failure through its Future; the consumer re-raises it in chromosome order
                    future.set_exception(exc)
            del K

    threads = [
        threading.Thread(target=run, name=f"loco-eigen-{i}", daemon=True)
        for i in range(workers)
    ]
    for thread in threads:
        thread.start()
    pending: deque[Future[EigenResult]] = deque()
    interrupted = False
    with blas_threads(n_threads):
        try:
            for name, K in inputs:
                future: Future[EigenResult] = Future()
                pending.append(future)
                work.put((future, name, K))
                del K
                if len(pending) == workers:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            for future in pending:
                future.cancel()
            for _ in threads:
                work.put(None)
            if not interrupted:
                for thread in threads:
                    thread.join()
