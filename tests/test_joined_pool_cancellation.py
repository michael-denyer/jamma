"""A signal between Future cancellation and its callbacks cannot strand a borrow."""

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from jamma.core.thread_pool import JoiningThreadPoolExecutor

pytestmark = pytest.mark.tier0


def _cancel_during_signal() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    queued_started = threading.Event()
    signal_sent = threading.Event()

    def running_work() -> None:
        started.set()
        assert release.wait(5)
        finished.set()

    def queued_work() -> None:
        queued_started.set()

    def signal_before_callbacks(frame, event, arg):
        if (
            event == "line"
            and frame.f_code.co_name == "_invoke_callbacks"
            and frame.f_globals.get("__name__") == "concurrent.futures._base"
            and frame.f_locals["self"].cancelled()
            and not signal_sent.is_set()
        ):
            signal_sent.set()
            signal.raise_signal(signal.SIGINT)
        return signal_before_callbacks

    def release_after_signal() -> None:
        if signal_sent.wait(5):
            time.sleep(0.05)
        release.set()

    pool = JoiningThreadPoolExecutor(max_workers=1)
    running = pool.submit(running_work)
    assert started.wait(5)
    queued = pool.submit(queued_work)
    releaser = threading.Thread(target=release_after_signal)
    releaser.start()
    try:
        sys.settrace(signal_before_callbacks)
        with pytest.raises(KeyboardInterrupt):
            pool.shutdown(wait=True, cancel_futures=True)
        assert finished.is_set(), "shutdown returned before the running borrow finished"
        assert signal_sent.is_set(), "SIGINT did not interrupt cancellation callbacks"
        assert queued.cancelled()
        assert not queued_started.is_set()
        running.result(timeout=5)
    finally:
        sys.settrace(None)
        release.set()
        releaser.join(timeout=5)
        assert not releaser.is_alive()
        pool.shutdown(wait=True, cancel_futures=True)


def test_cancellation_callback_interrupt_does_not_hang() -> None:
    # A pre-fix shutdown blocks forever even after running_work finishes. Keep
    # the regression bounded and kill that child without stranding pytest.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _cancel_during_signal()
