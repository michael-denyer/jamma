"""Shared progress bar utility for JAMMA.

Provides a cross-platform progress iterator that works in both
Databricks interactive notebooks and workflow notebooks.
"""

import sys
from collections.abc import Iterator

import progressbar


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
    bar = progressbar.ProgressBar(max_value=total, widgets=widgets, fd=sys.stdout)
    bar.start()
    return bar


def progress_iterator(
    iterable: Iterator,
    total: int,
    desc: str = "",
    initial_eta_seconds: float | None = None,
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

    Yields:
        Items from the wrapped iterator.
    """
    bar = create_progress_bar(total, desc, initial_eta_seconds=initial_eta_seconds)
    try:
        for i, item in enumerate(iterable):
            yield item
            bar.update(i + 1)
    finally:
        bar.finish()
