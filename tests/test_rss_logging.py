"""Tests for RSS memory logging."""

from unittest.mock import MagicMock, patch

import pytest

from jamma.utils.logging import log_rss_memory


class TestLogRssMemory:
    """Tests for log_rss_memory function."""

    def test_returns_rss_in_gb(self):
        """Should return RSS memory in GB."""
        # Mock psutil.Process - psutil is imported inside the function
        with patch("psutil.Process") as mock_process_class:
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 12_345_678_901  # ~12.35 GB
            mock_process_class.return_value = mock_process

            rss = log_rss_memory("test_phase", "test_checkpoint")

            assert abs(rss - 12.35) < 0.01

    def test_logs_with_phase_and_checkpoint(self, capfd):
        """Should log message containing phase and checkpoint."""
        from jamma.utils.logging import setup_logging

        # Setup logging to capture output
        setup_logging(verbose=True)

        with patch("psutil.Process") as mock_process_class:
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 5e9  # 5 GB
            mock_process_class.return_value = mock_process

            log_rss_memory("eigendecomp", "before")

            # Check stderr for log message
            captured = capfd.readouterr()
            assert "RSS memory" in captured.err
            assert "eigendecomp" in captured.err
            assert "before" in captured.err

    def test_real_rss_measurement(self):
        """Should measure actual process RSS (sanity check)."""
        # No mocking - verify it actually works
        rss = log_rss_memory("integration", "test")

        # RSS should be positive and reasonable (less than 100GB for a test process)
        assert 0 < rss < 100
