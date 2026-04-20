"""Shared progress bar utility for JAMMA.

Provides a cross-platform progress iterator that works in both
Databricks interactive notebooks and workflow notebooks.
"""

import contextlib
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import TypeVar

import progressbar

_T = TypeVar("_T")


class SeededETA(progressbar.AdaptiveETA):
    """AdaptiveETA that shows a model-predicted ETA before any chunks complete.

    Once the first chunk finishes, delegates to AdaptiveETA's exponential
    smoothing. This prevents the jarring cold-start overestimate where
    chunk 0 (slow due to thread pool spin-up) inflates the ETA.
    """

    def __init__(self, initial_eta_seconds: float, **kwargs):
        super().__init__(**kwargs)
        self._initial_eta = initial_eta_seconds

    def __call__(self, progress, data, **kwargs):
        if data.get("value", 0) < 1:
            s = int(self._initial_eta)
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            return f"ETA:  {h:02d}:{m:02d}:{s:02d}"
        return super().__call__(progress, data, **kwargs)


def _make_eta_widget(
    initial_eta_seconds: float | None = None,
) -> progressbar.widgets.WidgetBase:
    """Build the ETA widget, seeded with a model estimate if available."""
    if initial_eta_seconds is not None:
        return SeededETA(initial_eta_seconds)
    return progressbar.AdaptiveETA()


def create_progress_bar(
    total: int,
    desc: str = "",
    initial_eta_seconds: float | None = None,
    poll_interval: float | None = None,
) -> progressbar.ProgressBar:
    """Create and start a progressbar with standard JAMMA widgets.

    Use this when you need manual ``bar.update(n)`` / ``bar.finish()``
    control (e.g. pipeline loops).  For simple iterator wrapping, prefer
    :func:`progress_iterator` instead.

    Args:
        total: Total number of items.
        desc: Optional description prefix.
        initial_eta_seconds: Model-predicted total time in seconds. When
            provided, the ETA widget shows this estimate until the first
            chunk completes, then switches to adaptive smoothing.
        poll_interval: Minimum seconds between terminal redraws. Lower
            values make the bar feel more responsive; higher values reduce
            I/O overhead. Defaults to progressbar2's built-in default.

    Returns:
        A started ProgressBar instance. Caller must call ``bar.finish()``.
    """
    widgets = [
        f"{desc}: " if desc else "",
        progressbar.Counter(),
        f"/{total} ",
        progressbar.Percentage(),
        " ",
        progressbar.Bar(),
        " ",
        progressbar.Timer(),
        " ",
        _make_eta_widget(initial_eta_seconds),
    ]
    kwargs = {"max_value": total, "widgets": widgets, "fd": sys.stdout}
    if poll_interval is not None:
        kwargs["poll_interval"] = poll_interval
    bar = progressbar.ProgressBar(**kwargs)
    bar.start()
    return bar


def progress_iterator(
    iterable: Iterator,
    total: int,
    desc: str = "",
    initial_eta_seconds: float | None = None,
    poll_interval: float | None = None,
) -> Iterator:
    """Wrap iterator with progressbar2 progress display.

    Works in both Databricks interactive notebooks and workflow notebooks,
    unlike tqdm which only works in interactive mode. Writes to stdout so
    output is visible in Databricks notebook cells (stderr may be buffered).

    The bar is finalized in a try/finally block so that early breaks or
    exceptions from the caller don't leave terminal output corrupted.

    Args:
        iterable: Iterator to wrap.
        total: Total number of items.
        desc: Optional description prefix.
        initial_eta_seconds: Model-predicted total time in seconds. When
            provided, the ETA widget shows this estimate until the first
            chunk completes, then switches to adaptive smoothing.
        poll_interval: Minimum seconds between terminal redraws. Lower
            values make the bar feel more responsive; higher values reduce
            I/O overhead. Defaults to progressbar2's built-in default.

    Yields:
        Items from the wrapped iterator.
    """
    bar = create_progress_bar(
        total,
        desc,
        initial_eta_seconds=initial_eta_seconds,
        poll_interval=poll_interval,
    )
    try:
        for i, item in enumerate(iterable):
            yield item
            bar.update(i + 1)
    finally:
        bar.finish()


def timed_progress(
    fn: Callable[[], _T],
    estimated_seconds: float,
    desc: str = "",
    poll_interval: float = 1.0,
) -> _T:
    """Run a blocking function with a time-based progress bar.

    Designed for opaque operations like LAPACK eigendecomposition that
    release the GIL but provide no intermediate progress. The bar advances
    based on elapsed time vs the estimate, capping at 99% until the
    function actually returns.

    Args:
        fn: Zero-argument callable to execute. Must release the GIL
            (numpy/LAPACK calls do this) so the main thread can update
            the progress bar while *fn* runs in a background thread.
        estimated_seconds: Model-predicted wall time. The bar reaches
            ~99% at this time, then holds until completion.
        desc: Optional description prefix.
        poll_interval: Seconds between bar updates.

    Returns:
        The return value of *fn*.  Any exception raised by *fn* is
        re-raised after the bar is finalized.
    """
    # Use 100 ticks so the bar shows percentage naturally.
    n_ticks = 100
    widgets = [
        f"{desc}: " if desc else "",
        progressbar.Percentage(),
        " ",
        progressbar.Bar(),
        " ",
        progressbar.Timer(),
    ]
    bar = progressbar.ProgressBar(
        max_value=n_ticks,
        widgets=widgets,
        fd=sys.stdout,
        poll_interval=poll_interval,
    )
    bar.start()

    result: list[_T] = []  # mutable box for cross-thread result passing
    exception: list[BaseException] = []
    done = threading.Event()

    def _worker():
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 — worker thread must catch BaseException (KeyboardInterrupt, SystemExit) and route via exception list; otherwise the main thread hangs on done.wait()
            exception.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    t0 = time.monotonic()
    cancelled = False
    try:
        while not done.wait(timeout=poll_interval):
            elapsed = time.monotonic() - t0
            # Cap at 99 so the bar never claims 100% before fn returns.
            pct = min(int(elapsed / max(estimated_seconds, 0.1) * n_ticks), n_ticks - 1)
            try:
                bar.update(pct)
            except OSError:
                break  # stdout gone; stop updating but still wait for fn
        if not exception:
            with contextlib.suppress(OSError):
                bar.update(n_ticks)
    except KeyboardInterrupt:
        cancelled = True
        raise
    finally:
        with contextlib.suppress(OSError):
            bar.finish()
        # Join with timeout so KeyboardInterrupt is deliverable between
        # iterations.  On cancellation the worker is a daemon thread and
        # will be killed when the process exits — don't block.
        if not cancelled:
            while thread.is_alive():
                thread.join(timeout=0.5)

    if exception:
        raise exception[0]
    return result[0]
