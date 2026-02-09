"""Shared progress bar utility for JAMMA.

Provides a cross-platform progress iterator that works in both
Databricks interactive notebooks and workflow notebooks.
"""

import sys
from collections.abc import Iterator

import progressbar


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
    try:
        for i, item in enumerate(iterable):
            yield item
            bar.update(i + 1)
    finally:
        bar.finish()
