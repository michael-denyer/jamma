"""Tests for progress bar lifecycle management."""

from unittest.mock import MagicMock, patch

import pytest

from jamma.core.progress import progress_iterator


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
