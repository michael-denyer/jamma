"""Failed worker startup must revoke the already-enqueued callable's borrow."""

from threading import Thread

import pytest

from jamma.core.thread_pool import JoiningThreadPoolExecutor

pytestmark = pytest.mark.tier0


def test_failed_thread_start_does_not_run_the_failed_submission(monkeypatch):
    borrowed: list[str] = []

    def failed_start(_thread):
        raise OSError("worker startup failed")

    pool = JoiningThreadPoolExecutor(max_workers=1)
    try:
        # Thread startup is the OS boundary. ThreadPoolExecutor has already
        # enqueued this callable before asking the OS to start its first worker.
        with monkeypatch.context() as patch:
            patch.setattr(Thread, "start", failed_start)
            with pytest.raises(OSError, match="worker startup failed"):
                pool.submit(borrowed.append, "failed")

        # Starting another worker drains the old queue entry before this one.
        # The cancelled wrapper must reject its borrow even when dequeued later.
        pool.submit(borrowed.append, "successful").result(timeout=5)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    assert borrowed == ["successful"]
