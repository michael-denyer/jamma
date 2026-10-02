"""Completion rendering must agree with the real progressbar2 lifecycle."""

from __future__ import annotations

import threading

import progressbar
import pytest

from jamma.core.progress import timed_progress

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize(
    "failure", [RuntimeError, MemoryError, SystemExit, KeyboardInterrupt]
)
def test_real_progressbar_does_not_claim_completion_on_worker_failure(
    monkeypatch, failure
):
    updates: list[int | None] = []
    real_bar = progressbar.ProgressBar

    class RecordingBar(real_bar):
        def update(self, value=None, force=False, **kwargs):
            updates.append(value)
            return super().update(value, force=force, **kwargs)

    monkeypatch.setattr(progressbar, "ProgressBar", RecordingBar)

    def fail():
        raise failure("worker failure")

    with pytest.raises(failure, match="worker failure"):
        timed_progress(fail, estimated_seconds=10, poll_interval=0.01)

    assert 100 not in updates


def test_transient_stdout_failure_does_not_claim_completion_before_worker_returns(
    monkeypatch,
):
    release = threading.Event()
    returned = threading.Event()
    premature: list[int] = []
    real_bar = progressbar.ProgressBar
    failed_update = False

    class RecordingBar(real_bar):
        def update(self, value=None, force=False, **kwargs):
            nonlocal failed_update
            # ProgressBar.start also updates. Inject the stdout failure only
            # after startup, in the consumer's timed polling update.
            if value is not None and value > 0 and value < 100 and not failed_update:
                failed_update = True
                raise OSError("transient stdout failure")
            if value == 100 and not returned.is_set():
                premature.append(value)
            return super().update(value, force=force, **kwargs)

        def finish(self, *args, **kwargs):
            try:
                return super().finish(*args, **kwargs)
            finally:
                release.set()

    monkeypatch.setattr(progressbar, "ProgressBar", RecordingBar)

    def work():
        assert release.wait(10), "bar shutdown never released the worker"
        returned.set()
        return "result"

    assert timed_progress(work, estimated_seconds=0.1, poll_interval=0.02) == "result"
    assert failed_update
    assert premature == []


def test_consumer_interrupt_does_not_render_completion_or_join_worker(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    workers: list[threading.Thread] = []
    updates: list[int | None] = []
    real_bar = progressbar.ProgressBar

    class RecordingBar(real_bar):
        def update(self, value=None, force=False, **kwargs):
            updates.append(value)
            if value is not None and 0 < value < 100:
                assert started.wait(5)
                raise KeyboardInterrupt
            return super().update(value, force=force, **kwargs)

    monkeypatch.setattr(progressbar, "ProgressBar", RecordingBar)

    def work():
        workers.append(threading.current_thread())
        started.set()
        assert release.wait(10)
        return "result"

    try:
        with pytest.raises(KeyboardInterrupt):
            timed_progress(work, estimated_seconds=0.1, poll_interval=0.02)
        assert not release.is_set()
        assert workers[0].is_alive(), "consumer interrupt waited for the worker"
        assert 100 not in updates
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
        assert all(not worker.is_alive() for worker in workers)
