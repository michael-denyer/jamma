"""Tests for progress bar lifecycle management."""

import time
from unittest.mock import MagicMock, patch

import pytest

from jamma.core.progress import progress_iterator, timed_progress


@pytest.mark.tier0
class TestProgressBarLifecycle:
    """Tests that progress bar is finalized correctly in all scenarios."""

    def test_finish_called_on_normal_completion(self):
        """bar.finish() is called when iteration completes normally."""
        items = list(range(5))
        collected = []

        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            for item in progress_iterator(iter(items), total=5, desc="test"):
                collected.append(item)

            mock_bar.finish.assert_called_once()
            assert collected == items

    def test_finish_called_on_early_break(self):
        """bar.finish() is called when caller breaks out of loop early."""
        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            for i, _item in enumerate(
                progress_iterator(iter(range(10)), total=10, desc="test")
            ):
                if i == 2:
                    break

            mock_bar.finish.assert_called_once()

    def test_finish_called_on_exception(self):
        """bar.finish() is called when loop body raises an exception."""

        def exploding_items():
            yield 1
            yield 2
            raise RuntimeError("boom")

        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            with pytest.raises(RuntimeError, match="boom"):
                for _ in progress_iterator(exploding_items(), total=5, desc="test"):
                    pass

            mock_bar.finish.assert_called_once()

    def test_finish_called_on_caller_exception(self):
        """bar.finish() is called when exception occurs in caller's loop body."""
        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            with pytest.raises(ValueError, match="test error"):
                for i, _ in enumerate(
                    progress_iterator(iter(range(10)), total=10, desc="test")
                ):
                    if i == 3:
                        raise ValueError("test error")

            mock_bar.finish.assert_called_once()

    def test_update_called_for_each_item(self):
        """bar.update() is called once per yielded item."""
        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            items = list(range(5))
            for _ in progress_iterator(iter(items), total=5, desc="test"):
                pass

            assert mock_bar.update.call_count == 5
            # Verify called with 1-based indices
            mock_bar.update.assert_any_call(1)
            mock_bar.update.assert_any_call(5)


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

    def test_bar_finish_called_on_success(self):
        """bar.finish() is called on normal completion."""
        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            timed_progress(lambda: 1, estimated_seconds=10.0, poll_interval=0.01)

            mock_bar.finish.assert_called_once()

    def test_bar_finish_called_on_exception(self):
        """bar.finish() is called even when fn raises."""

        def boom():
            raise RuntimeError("boom")

        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            with pytest.raises(RuntimeError):
                timed_progress(boom, estimated_seconds=10.0, poll_interval=0.01)

            mock_bar.finish.assert_called_once()

    def test_bar_not_set_to_100_on_error(self):
        """Bar should not show 100% when fn fails."""

        def slow_boom():
            time.sleep(0.05)
            raise RuntimeError("boom")

        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            with pytest.raises(RuntimeError):
                timed_progress(slow_boom, estimated_seconds=10.0, poll_interval=0.01)

            # Progress loop must have run at least once before the exception.
            assert mock_bar.update.call_count > 0
            for call in mock_bar.update.call_args_list:
                assert call.args[0] < 100

    def test_slow_fn_caps_at_99(self):
        """Bar caps at 99% while fn is still running."""

        def slow():
            time.sleep(0.15)
            return "done"

        with patch("jamma.core.progress.progressbar") as mock_pb:
            mock_bar = MagicMock()
            mock_pb.ProgressBar.return_value = mock_bar

            result = timed_progress(slow, estimated_seconds=0.01, poll_interval=0.02)

            assert result == "done"
            # Must have at least one intermediate + one final update.
            assert len(mock_bar.update.call_args_list) >= 2
            # Any intermediate update should be <= 99
            for call in mock_bar.update.call_args_list[:-1]:
                assert call.args[0] <= 99

    def test_estimated_seconds_zero(self):
        """estimated_seconds=0 is handled gracefully (no ZeroDivisionError)."""
        result = timed_progress(lambda: "ok", estimated_seconds=0.0, poll_interval=0.01)
        assert result == "ok"
