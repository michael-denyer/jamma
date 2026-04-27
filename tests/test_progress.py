"""Tests for progress bar lifecycle management."""

import time

import pytest

from jamma.core.progress import progress_iterator, timed_progress
from tests.fakes import FakeProgressbarModule


@pytest.fixture
def fake_progressbar(monkeypatch: pytest.MonkeyPatch) -> FakeProgressbarModule:
    """Replace ``jamma.core.progress.progressbar`` with a recording fake.

    Catches signature drift in ``ProgressBar.update`` / ``finish`` that
    ``MagicMock`` would silently absorb. See docs/TESTING.md §2.3.
    """
    fake = FakeProgressbarModule()
    monkeypatch.setattr("jamma.core.progress.progressbar", fake)
    return fake


@pytest.mark.tier0
class TestProgressBarLifecycle:
    """Tests that progress bar is finalized correctly in all scenarios."""

    def test_finish_called_on_normal_completion(self, fake_progressbar):
        """bar.finish() is called when iteration completes normally."""
        items = list(range(5))

        collected = list(progress_iterator(iter(items), total=5, desc="test"))

        assert fake_progressbar.last_bar.finish_calls == 1
        assert collected == items

    def test_finish_called_on_early_break(self, fake_progressbar):
        """bar.finish() is called when caller breaks out of loop early."""
        for i, _item in enumerate(
            progress_iterator(iter(range(10)), total=10, desc="test")
        ):
            if i == 2:
                break

        assert fake_progressbar.last_bar.finish_calls == 1

    def test_finish_called_on_exception(self, fake_progressbar):
        """bar.finish() is called when loop body raises an exception."""

        def exploding_items():
            yield 1
            yield 2
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            for _ in progress_iterator(exploding_items(), total=5, desc="test"):
                pass

        assert fake_progressbar.last_bar.finish_calls == 1

    def test_finish_called_on_caller_exception(self, fake_progressbar):
        """bar.finish() is called when exception occurs in caller's loop body."""
        with pytest.raises(ValueError, match="test error"):
            for i, _ in enumerate(
                progress_iterator(iter(range(10)), total=10, desc="test")
            ):
                if i == 3:
                    raise ValueError("test error")

        assert fake_progressbar.last_bar.finish_calls == 1

    def test_update_called_for_each_item(self, fake_progressbar):
        """bar.update() is called once per yielded item with 1-based indices."""
        items = list(range(5))
        for _ in progress_iterator(iter(items), total=5, desc="test"):
            pass

        assert fake_progressbar.last_bar.update_calls == [1, 2, 3, 4, 5]


@pytest.mark.tier0
class TestTimedProgress:
    """Tests for timed_progress() threading and exception propagation."""

    def test_returns_fn_result(self):
        """timed_progress returns the value from fn."""
        result = timed_progress(lambda: 42, estimated_seconds=10.0, poll_interval=0.01)
        assert result == 42

    def test_returns_complex_result(self):
        """timed_progress handles non-trivial return types."""
        expected = {"a": [1, 2, 3], "b": None}
        result = timed_progress(
            lambda: expected, estimated_seconds=10.0, poll_interval=0.01
        )
        assert result == expected

    def test_propagates_exception(self):
        """Exceptions from fn are re-raised on the calling thread."""

        def failing():
            raise ValueError("test error from worker")

        with pytest.raises(ValueError, match="test error from worker"):
            timed_progress(failing, estimated_seconds=10.0, poll_interval=0.01)

    def test_propagates_memory_error(self):
        """MemoryError propagates correctly (critical for eigendecomp)."""

        def oom():
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError, match="out of memory"):
            timed_progress(oom, estimated_seconds=10.0, poll_interval=0.01)

    def test_fast_completion(self):
        """fn completing faster than one poll interval works correctly."""
        result = timed_progress(
            lambda: "fast", estimated_seconds=60.0, poll_interval=10.0
        )
        assert result == "fast"

    def test_bar_finish_called_on_success(self, fake_progressbar):
        """bar.finish() is called on normal completion."""
        timed_progress(lambda: 1, estimated_seconds=10.0, poll_interval=0.01)

        assert fake_progressbar.last_bar.finish_calls == 1

    def test_bar_finish_called_on_exception(self, fake_progressbar):
        """bar.finish() is called even when fn raises."""

        def boom():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            timed_progress(boom, estimated_seconds=10.0, poll_interval=0.01)

        assert fake_progressbar.last_bar.finish_calls == 1

    def test_bar_not_set_to_100_on_error(self, fake_progressbar):
        """Bar should not show 100% when fn fails."""

        def slow_boom():
            time.sleep(0.05)
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            timed_progress(slow_boom, estimated_seconds=10.0, poll_interval=0.01)

        bar = fake_progressbar.last_bar
        assert len(bar.update_calls) > 0, "progress loop must have ticked"
        assert all(v < 100 for v in bar.update_calls)

    def test_slow_fn_caps_at_99(self, fake_progressbar):
        """Bar caps at 99% while fn is still running."""

        def slow():
            time.sleep(0.15)
            return "done"

        result = timed_progress(slow, estimated_seconds=0.01, poll_interval=0.02)
        assert result == "done"

        bar = fake_progressbar.last_bar
        # At least one intermediate + one final update.
        assert len(bar.update_calls) >= 2
        # All intermediate updates capped at 99 (the final update is 100).
        assert all(v <= 99 for v in bar.update_calls[:-1])

    def test_estimated_seconds_zero(self):
        """estimated_seconds=0 is handled gracefully (no ZeroDivisionError)."""
        result = timed_progress(lambda: "ok", estimated_seconds=0.0, poll_interval=0.01)
        assert result == "ok"
