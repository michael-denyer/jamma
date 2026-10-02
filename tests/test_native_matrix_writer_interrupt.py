"""SIGINT during executor shutdown must not release borrowed matrix inputs."""

import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from jamma.io._native_matrix_writer import write_native_matrix

pytestmark = pytest.mark.tier0


def test_shutdown_interrupt_waits_for_borrowed_matrix(tmp_path: Path) -> None:
    running = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    signal_sent = threading.Event()
    driver_errors: list[Exception] = []
    main_ident = threading.get_ident()

    def controlled_format(block: np.ndarray, buffer: bytearray) -> int:
        if block[0, 0] == 0:
            assert running.wait(10)
            raise ValueError("first formatter failed")
        running.set()
        assert release.wait(10)
        finished.set()
        return 0

    def interrupt_join() -> None:
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                frame = sys._current_frames()[main_ident]
                while frame is not None:
                    if frame.f_code.co_name == "shutdown" and frame.f_globals.get(
                        "__name__"
                    ) in {"concurrent.futures.thread", "jamma.core.thread_pool"}:
                        os.kill(os.getpid(), signal.SIGINT)
                        signal_sent.set()
                        # Keep the borrow live until the writer has handled SIGINT.
                        time.sleep(0.2)
                        return
                    frame = frame.f_back
                time.sleep(0.001)
            raise AssertionError("writer did not enter executor shutdown")
        except (AssertionError, KeyError, OSError) as error:
            driver_errors.append(error)
        finally:
            release.set()

    output = tmp_path / "matrix.txt"
    output.write_bytes(b"prior output")
    matrix = np.ones((500, 1000))
    matrix[0, 0] = 0
    interrupter = threading.Thread(target=interrupt_join, name="writer-interrupter")
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            write_native_matrix(matrix, output, 2, controlled_format)
        returned_after_completion = finished.is_set()
    finally:
        release.set()
        interrupter.join(timeout=10)
        assert not interrupter.is_alive()
        assert finished.wait(10)
    assert not driver_errors
    assert signal_sent.is_set()
    assert returned_after_completion, "writer returned while formatter borrowed matrix"
    assert output.read_bytes() == b"prior output"
    assert list(tmp_path.iterdir()) == [output]
