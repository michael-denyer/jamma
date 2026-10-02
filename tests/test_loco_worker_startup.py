"""Worker startup failures must release every thread already started."""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager

import numpy as np
import pytest

from jamma.core import threading as core_threading
from jamma.lmm.loco_workers import solve_eigen_pairs

pytestmark = pytest.mark.tier0


def _stop_leaked_workers(threads: list[threading.Thread]) -> None:
    """Keep a failing regression from leaving daemon workers in the test process."""
    for thread in threads:
        if not thread.is_alive():
            continue
        target = thread._target  # type: ignore[missing-attribute]
        for cell in target.__closure__:
            work = cell.cell_contents
            if isinstance(work, queue.SimpleQueue):
                work.put(None)
        thread.join(5)


@pytest.mark.parametrize("fail_on", [1, 2, 3])
def test_start_failure_stops_preceding_workers(monkeypatch, fail_on):
    real_start = threading.Thread.start
    started: list[threading.Thread] = []
    attempts = 0

    def start(thread: threading.Thread) -> None:
        nonlocal attempts
        if thread.name.startswith("loco-eigen-"):
            attempts += 1
            if attempts == fail_on:
                raise RuntimeError("cannot start new thread")
            started.append(thread)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    pairs = solve_eigen_pairs([], np.linalg.eigh, workers=3, n_threads=1)
    try:
        with pytest.raises(RuntimeError, match="cannot start new thread"):
            next(pairs)
        pairs.close()
        assert attempts == fail_on
        assert len(started) == fail_on - 1
        assert all(not thread.is_alive() for thread in started)
    finally:
        pairs.close()
        _stop_leaked_workers(started)


def test_blas_entry_failure_starts_no_workers(monkeypatch):
    real_start = threading.Thread.start
    started: list[threading.Thread] = []

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("loco-eigen-"):
            started.append(thread)
        real_start(thread)

    @contextmanager
    def unavailable_controller(*, limits: int, user_api: str):
        raise RuntimeError("BLAS controller unavailable")
        yield

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(core_threading, "is_blas_controllable", lambda: True)
    monkeypatch.setattr(core_threading, "threadpool_limits", unavailable_controller)
    pairs = solve_eigen_pairs([], np.linalg.eigh, workers=3, n_threads=1)
    try:
        with pytest.raises(RuntimeError, match="BLAS controller unavailable"):
            next(pairs)
        pairs.close()
        assert started == []
    finally:
        pairs.close()
        _stop_leaked_workers(started)
