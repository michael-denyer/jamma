"""An exited spawn worker must report an error instead of losing its task."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import traceback

import pytest

from jamma.io._parallel_text import run_spawn_pool

pytestmark = pytest.mark.tier0


def _exit_worker(kind: str) -> None:
    if kind == "system_exit":
        raise SystemExit(7)
    os._exit(int(kind))


def _run_exit_case(kind: str, report) -> None:
    def stop(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        run_spawn_pool(_exit_worker, [kind], error_context="exit test", n_workers=1)
    except RuntimeError as exc:
        report.put(("error", str(exc), len(mp.active_children())))
    except BaseException:  # noqa: BLE001 - report child failures before cleanup
        report.put(("unexpected", traceback.format_exc(), len(mp.active_children())))
    else:
        report.put(("returned", "", len(mp.active_children())))


@pytest.mark.parametrize("kind", ["system_exit", "0", "7"])
def test_spawn_worker_exit_reports_error_and_joins(kind: str) -> None:
    # An outer process lets the test recover from the pre-fix lost result.
    # Kill descendants too if the regression returns to a blocking imap.
    ctx = mp.get_context("spawn")
    report = ctx.Queue()
    parent = ctx.Process(target=_run_exit_case, args=(kind, report))
    parent.start()
    try:
        parent.join(8)
        assert not parent.is_alive(), (
            "exited worker left ordered result waiting forever"
        )
        assert parent.exitcode == 0
        outcome, message, alive = report.get(timeout=2)
        assert outcome == "error", message
        assert "exited unexpectedly" in message
        assert alive == 0, "pool worker survived the error"
    finally:
        if parent.is_alive():
            parent.terminate()
            parent.join(5)
        report.close()
        report.join_thread()
