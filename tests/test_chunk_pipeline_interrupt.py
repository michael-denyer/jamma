"""Interrupting cleanup must preserve the BLAS limit until rotation finishes."""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.tier0


def test_interrupt_during_cleanup_waits_for_background_rotation():
    # Deliver a real SIGINT in a child: pytest's own interrupt handler and
    # concurrent test workers must not receive the signal.
    script = textwrap.dedent(
        """
        import os
        import signal
        import sys
        import threading
        import time
        from contextlib import contextmanager
        from jamma.lmm import chunk_pipeline

        running = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        interrupted = threading.Event()
        exited = threading.Event()
        active = [False]
        observations = []
        main = threading.get_ident()

        @contextmanager
        def blas(_threads):
            active[0] = True
            try:
                yield
            finally:
                observations.append(('restore', finished.is_set()))
                active[0] = False

        class Engine:
            calls = 0
            def prepare(self):
                self.calls += 1
                if self.calls == 1:
                    return 1
                running.set()
                assert release.wait(5)
                observations.append(('rotation', active[0]))
                finished.set()
                return None
            def compute_and_write(self, _chunk):
                assert running.wait(5)
                raise RuntimeError('compute failed')

        def handle_interrupt(_signum, _frame):
            interrupted.set()
            raise KeyboardInterrupt

        def stack():
            frame = sys._current_frames()[main]
            result = []
            while frame:
                result.append(frame)
                frame = frame.f_back
            return result

        def sender():
            deadline = time.monotonic() + 5
            original_wait = None
            while time.monotonic() < deadline:
                frames = stack()
                if any(f.f_code.co_name == 'shutdown' for f in frames):
                    waits = [f for f in frames if f.f_code.co_name in ('wait', 'join')]
                    if waits:
                        original_wait = waits[0]
                        os.kill(os.getpid(), signal.SIGINT)
                        break
                time.sleep(0.001)
            assert original_wait is not None
            assert interrupted.wait(5)
            # With the fix, cleanup retries its condition wait. With the
            # original executor, the driver escapes before rotation finishes.
            while time.monotonic() < deadline:
                if exited.is_set():
                    break
                frames = stack()
                if any(f.f_code.co_name == 'shutdown' for f in frames) and any(
                    f.f_code.co_name == 'wait' and f is not original_wait
                    for f in frames
                ):
                    break
                time.sleep(0.001)
            release.set()

        signal.signal(signal.SIGINT, handle_interrupt)
        chunk_pipeline.blas_threads = blas
        thread = threading.Thread(target=sender)
        thread.start()
        try:
            chunk_pipeline._drive_pipeline(
                Engine(), n_chunks=2, rotation_threads=1, n_samples=1,
                n_filtered=2, show_progress=False, progress_label='interrupt',
            )
        except KeyboardInterrupt:
            observations.append(('exit', finished.is_set()))
            exited.set()
        finally:
            exited.set()
            release.set()
            assert finished.wait(5)
            thread.join(5)
        assert interrupted.is_set()
        expected = [('rotation', True), ('restore', True), ('exit', True)]
        assert observations == expected, observations
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
