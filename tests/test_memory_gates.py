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


@pytest.mark.tier0
class TestBatchPreflightThreadsNcvt:
    """Regression: batch LMM memory preflight must propagate n_cvt.

    The estimator estimate_lmm_memory scales Uab/Iab memory with n_cvt
    (memory.py:_uab_iab_gb). Both batch preflight call sites previously
    omitted n_cvt, so multi-covariate runs (n_cvt > 1) would pass the
    preflight using the n_cvt=1 default and then OOM at real allocation
    time in compute_numpy._run_inner. These tests pin that the dispatch
    sites propagate n_cvt to the estimator.

    Dispatch-site assertions are the right test shape here: the preflight's
    sole job is to delegate to the estimator with correct arguments, so the
    delegation contract IS the observable behavior (see CLAUDE.md testing
    guidance on system-boundary assert_called_once_with).
    """

    def test_pipeline_batch_preflight_passes_n_cvt(self):
        """PipelineRunner batch branch must pass n_cvt to estimate_lmm_memory.

        The batch preflight site at pipeline.py:~1035 previously called
        estimate_lmm_memory(n_valid, n_snps) with no n_cvt, silently
        defaulting to 1 and underestimating multi-covariate runs.
        """
        import numpy as np

        # Use a real 3-column covariate file so n_cvt=3 reaches the
        # preflight through the normal pipeline code path.
        fixtures = Path(__file__).parent / "fixtures" / "gemma_synthetic"
        bfile = fixtures / "test"

        # Read sample count from .fam to build a matching covariate file.
        fam_path = bfile.with_suffix(".fam")
        n_samples = sum(1 for _ in fam_path.open())

        import tempfile

        # Sentinel raised by patched estimator. This stops execution at
        # the preflight site so we never need to mock downstream stages.
        sentinel = RuntimeError("stop-at-preflight-sentinel")

        captured_calls = []

        def capturing_estimator(*args, **kwargs):
            captured_calls.append((args, kwargs))
            raise sentinel

        with tempfile.TemporaryDirectory() as tmpdir:
            cov_path = Path(tmpdir) / "covariates.txt"
            # GEMMA-format covariates: intercept column + 2 real covariates
            # (total 3 columns → n_cvt = 3).
            rng = np.random.default_rng(0)
            cov_data = np.column_stack(
                [
                    np.ones(n_samples),
                    rng.normal(size=n_samples),
                    rng.normal(size=n_samples),
                ]
            )
            np.savetxt(cov_path, cov_data, fmt="%.6f")

            config = PipelineConfig(
                bfile=bfile,
                covariate_file=cov_path,
                check_memory=True,
                backend="numpy",  # force batch branch, not streaming
                output_dir=Path(tmpdir),
                show_progress=False,
            )
            runner = PipelineRunner(config)

            # Patch estimate_lmm_memory at the source module — the batch
            # preflight imports it locally inside _run_inner.
            with patch(
                "jamma.core.memory.estimate_lmm_memory",
                side_effect=capturing_estimator,
            ):
                with pytest.raises(RuntimeError, match="stop-at-preflight-sentinel"):
                    runner.run()

        # The batch preflight must have been called at least once.
        assert captured_calls, (
            "estimate_lmm_memory was not called — batch preflight branch did not run"
        )
        # Every call must propagate n_cvt=3 (from the 3-column covariates).
        for args, kwargs in captured_calls:
            n_cvt_arg = kwargs.get("n_cvt")
            if n_cvt_arg is None and len(args) >= 4:
                # estimate_lmm_memory signature:
                # (n_samples, n_snps, lmm_batch_size=..., n_cvt=1)
                n_cvt_arg = args[3]
            assert n_cvt_arg == 3, (
                f"Batch preflight called estimate_lmm_memory with "
                f"n_cvt={n_cvt_arg!r}, expected 3 (from 3-col covariates). "
                f"Full call: args={args}, kwargs={kwargs}"
            )

    def test_runner_numpy_preflight_passes_n_cvt(self):
        """run_lmm_association_numpy must pass n_cvt to estimate_lmm_memory.

        The batch runner preflight at runner_numpy.py:~610 previously
        called estimate_lmm_memory(n_samples, n_snps) with no n_cvt.
        """
        import numpy as np

        from jamma.lmm.runner_numpy import run_lmm_association_numpy

        n_samples = 100
        n_snps = 20
        expected_n_cvt = 4  # 3 real covariates + intercept column

        rng = np.random.default_rng(0)
        genotypes = rng.normal(size=(n_samples, n_snps)).astype(np.float64)
        phenotypes = rng.normal(size=n_samples).astype(np.float64)
        kinship = np.eye(n_samples, dtype=np.float64)
        covariates = np.column_stack(
            [np.ones(n_samples)] + [rng.normal(size=n_samples) for _ in range(3)]
        ).astype(np.float64)
        # snp_info is not touched before the preflight call, so an empty
        # list is safe — the sentinel estimator raises before any iteration.
        snp_info: list = []

        sentinel = RuntimeError("stop-at-preflight-sentinel")
        captured_calls = []

        def capturing_estimator(*args, **kwargs):
            captured_calls.append((args, kwargs))
            raise sentinel

        with patch(
            "jamma.lmm.runner_numpy.estimate_lmm_memory",
            side_effect=capturing_estimator,
        ):
            with pytest.raises(RuntimeError, match="stop-at-preflight-sentinel"):
                run_lmm_association_numpy(
                    genotypes=genotypes,
                    phenotypes=phenotypes,
                    kinship=kinship,
                    snp_info=snp_info,
                    covariates=covariates,
                    check_memory=True,
                    show_progress=False,
                )

        assert captured_calls, (
            "estimate_lmm_memory was not called — runner_numpy preflight did not run"
        )
        for args, kwargs in captured_calls:
            n_cvt_arg = kwargs.get("n_cvt")
            if n_cvt_arg is None and len(args) >= 4:
                n_cvt_arg = args[3]
            assert n_cvt_arg == expected_n_cvt, (
                f"runner_numpy preflight called estimate_lmm_memory with "
                f"n_cvt={n_cvt_arg!r}, expected {expected_n_cvt}. "
                f"Full call: args={args}, kwargs={kwargs}"
            )
