"""Tests for memory gate OOM prevention in PipelineRunner and check_memory_available.

Covers ERRP-05: memory gate code paths in both PipelineRunner.check_memory_requirements
and check_memory_available are tested using mock psutil to simulate low-memory
conditions without requiring actual large allocations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jamma.core.memory import StreamingMemoryBreakdown, check_memory_available
from jamma.pipeline import PipelineConfig, PipelineRunner

FIXTURES = Path(__file__).parent / "fixtures" / "gemma_synthetic"
BFILE = FIXTURES / "test"


@pytest.mark.tier0
class TestMemoryGates:
    """Integration tests for memory gate OOM prevention."""

    def test_budget_exceeded_raises(self):
        """Budget-exceeded path: 1 MB budget raises MemoryError with 'exceeds' message.

        PipelineRunner.check_memory_requirements raises MemoryError when
        est.total_peak_gb > config.mem_budget.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=True, mem_budget=0.001)
        runner = PipelineRunner(config)

        with pytest.raises(MemoryError, match="exceeds"):
            runner.check_memory_requirements(n_samples=100, n_snps=500)

    @patch("jamma.core.memory._check_available", return_value=(0.001, False))
    def test_insufficient_system_memory_raises(self, mock_check):
        """Insufficient system memory raises MemoryError with 'Insufficient' message.

        Mocks _check_available to return (0.001 GB, False), simulating a system
        with nearly no available memory. check_memory_requirements must raise when
        est.sufficient is False.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=True)
        runner = PipelineRunner(config)

        with pytest.raises(MemoryError, match="Insufficient"):
            runner.check_memory_requirements(n_samples=100, n_snps=500)

    @patch("jamma.core.memory._check_available", return_value=(1000.0, True))
    def test_memory_check_passes_when_sufficient(self, mock_check):
        """Sufficient memory (1 TB available) returns StreamingMemoryBreakdown.

        Mocks _check_available to return (1000.0 GB, True), simulating ample
        memory. check_memory_requirements must return the breakdown, not raise.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=True)
        runner = PipelineRunner(config)

        result = runner.check_memory_requirements(n_samples=100, n_snps=500)

        assert result is not None
        assert isinstance(result, StreamingMemoryBreakdown)
        assert result.sufficient is True

    def test_memory_check_disabled_returns_none(self):
        """check_memory=False returns None without performing any memory check.

        When check_memory is disabled, check_memory_requirements must return
        None immediately, even with a tiny (realistic) dataset.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=False)
        runner = PipelineRunner(config)

        result = runner.check_memory_requirements(n_samples=100, n_snps=500)

        assert result is None

    def test_check_memory_available_raises_on_insufficient(self):
        """check_memory_available raises MemoryError when psutil reports 1 MB available.

        Patches psutil.virtual_memory at the import site used by jamma.core.memory
        to return 1 MB available. Requesting 100 GB must raise MemoryError.
        """
        mock_vmem = MagicMock()
        mock_vmem.available = 1_000_000  # 1 MB in bytes

        with patch("jamma.core.memory.psutil.virtual_memory", return_value=mock_vmem):
            with pytest.raises(MemoryError, match="Insufficient memory"):
                check_memory_available(required_gb=100.0, operation="test")

    def test_check_memory_available_passes_when_sufficient(self):
        """check_memory_available returns True when psutil reports 1 TB available.

        Patches psutil.virtual_memory to return 1 TB available. Requesting
        1 GB must succeed without raising.
        """
        mock_vmem = MagicMock()
        mock_vmem.available = 1_000_000_000_000  # 1 TB in bytes

        with patch("jamma.core.memory.psutil.virtual_memory", return_value=mock_vmem):
            result = check_memory_available(required_gb=1.0)

        assert result is True
