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
        assert fake.last_bar.finished
        assert fake.last_bar.update_calls == [1, 2, 3]
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakeProgressBar:
    """Records start / update / finish calls. Mirrors progressbar.ProgressBar.

    The full real surface is much larger; this fake declares only the
    methods jamma.core.progress actually calls. If production code starts
    calling another method, the test will fail with AttributeError —
    that is the whole point of using a fake instead of MagicMock.

    Lifecycle invariants (real ``progressbar.ProgressBar`` behaves the same
    way in normal use): ``start()`` is called once before the first
    ``update()``, and ``finish()`` is called once at the end. Calling
    ``start()`` or ``finish()`` twice raises ``AssertionError`` so a
    misuse fails loudly instead of being silently absorbed.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.update_calls: list[int] = []
        self.finished = False
        # Optional hook fired after each ``update()`` call. Set by tests
        # that need a deterministic synchronisation point against the
        # real polling loop in ``timed_progress``. The callable receives
        # the value that was just recorded; tests that only need a
        # "tick happened" signal can pass ``threading.Event().set``.
        self.on_update: Any = None

    def start(self) -> FakeProgressBar:
        if self.started:
            raise AssertionError("FakeProgressBar.start() called twice")
        self.started = True
        return self

    def update(self, value: int) -> None:
        self.update_calls.append(value)
        if self.on_update is not None:
            self.on_update(value)

    def finish(self) -> None:
        if self.finished:
            raise AssertionError("FakeProgressBar.finish() called twice")
        self.finished = True


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
        # Optional hook applied to every ``FakeProgressBar`` constructed by
        # this module fake. Tests set this BEFORE the work starts (the bar
        # is constructed inside ``timed_progress``, so per-bar attribute
        # setting is impossible). See ``test_bar_not_set_to_100_on_error``
        # for the deterministic-sync use case.
        self.on_update: Any = None
        # Widget classes return a sentinel; jamma.core.progress passes them
        # positionally into ProgressBar(**kwargs) and never inspects them.
        self.Counter = lambda *a, **kw: _FakeWidget("Counter")
        self.Percentage = lambda *a, **kw: _FakeWidget("Percentage")
        self.Bar = lambda *a, **kw: _FakeWidget("Bar")
        self.Timer = lambda *a, **kw: _FakeWidget("Timer")
        self.AdaptiveETA = lambda *a, **kw: _FakeWidget("AdaptiveETA")

    def ProgressBar(self, **kwargs: Any) -> FakeProgressBar:
        bar = FakeProgressBar(**kwargs)
        bar.on_update = self.on_update
        self.last_bar = bar
        return bar

    # ``progressbar.widgets`` is a submodule on the real package; mimic
    # it as a namespace so ``progressbar.widgets.WidgetBase`` resolves.
    widgets = SimpleNamespace(WidgetBase=_FakeWidget)
