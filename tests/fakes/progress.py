"""Fakes for the third-party ``progressbar`` library.

Replaces ``patch("jamma.core.progress.progressbar")`` + ``MagicMock()``
nests with concrete classes that record calls. Catches signature changes
in ``ProgressBar.update`` / ``finish`` that ``MagicMock`` would silently
absorb.

Usage::

    from tests.fakes import FakeProgressbarModule

    def test_progress_finish_called(monkeypatch):
        fake = FakeProgressbarModule()
        monkeypatch.setattr("jamma.core.progress.progressbar", fake)
        # ... exercise code that uses progress bar ...
        assert fake.last_bar.finish_calls == 1
        assert fake.last_bar.update_calls == [1, 2, 3]
"""

from __future__ import annotations

from typing import Any


class FakeProgressBar:
    """Records update / finish calls. Mirrors progressbar.ProgressBar."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.update_calls: list[int] = []
        self.finish_calls = 0

    def update(self, value: int) -> None:
        self.update_calls.append(value)

    def finish(self) -> None:
        self.finish_calls += 1


class _FakeWidget:
    """Stand-in for progressbar widget classes (Counter, Bar, Timer, ...).

    The real widgets are passed positionally into ``ProgressBar`` and only
    matter for rendering. Tests that don't care about rendering can treat
    them as opaque sentinels.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<FakeWidget {self.name}>"


class FakeProgressbarModule:
    """Module-shaped fake for the ``progressbar`` library.

    Substitute via ``monkeypatch.setattr("jamma.core.progress.progressbar", fake)``.
    The most recently constructed ``FakeProgressBar`` is exposed as
    ``last_bar`` for assertions.
    """

    def __init__(self) -> None:
        self.last_bar: FakeProgressBar | None = None
        # Widget classes return a sentinel; jamma.core.progress passes them
        # positionally into ProgressBar(**kwargs) and never inspects them.
        self.Counter = lambda *a, **kw: _FakeWidget("Counter")
        self.Percentage = lambda *a, **kw: _FakeWidget("Percentage")
        self.Bar = lambda *a, **kw: _FakeWidget("Bar")
        self.Timer = lambda *a, **kw: _FakeWidget("Timer")
        self.AdaptiveETA = lambda *a, **kw: _FakeWidget("AdaptiveETA")

    def ProgressBar(self, **kwargs: Any) -> FakeProgressBar:
        bar = FakeProgressBar(**kwargs)
        self.last_bar = bar
        return bar

    # ``progressbar.widgets`` is referenced by the type annotation in
    # jamma.core.progress; provide a placeholder so attribute access
    # doesn't blow up.
    class widgets:
        WidgetBase = _FakeWidget
