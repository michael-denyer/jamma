"""Shared progress bar utility for JAMMA.

Provides a cross-platform progress iterator that works in both
Databricks interactive notebooks and workflow notebooks.
"""

import sys
from collections.abc import Iterator

import progressbar


def create_progress_bar(total: int, desc: str = "") -> progressbar.ProgressBar:
    """Create and start a progressbar with standard JAMMA widgets.

    Use this when you need manual ``bar.update(n)`` / ``bar.finish()``
    control (e.g. pipeline loops).  For simple iterator wrapping, prefer
    :func:`progress_iterator` instead.

    Args:
        total: Total number of items.
        desc: Optional description prefix.

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
        progressbar.ETA(),
    ]
    bar = progressbar.ProgressBar(max_value=total, widgets=widgets, fd=sys.stdout)
    bar.start()
    return bar


def progress_iterator(iterable: Iterator, total: int, desc: str = "") -> Iterator:
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

    Yields:
        Items from the wrapped iterator.
    """
    bar = create_progress_bar(total, desc)
    try:
        for i, item in enumerate(iterable):
            yield item
            bar.update(i + 1)
    finally:
        bar.finish()
