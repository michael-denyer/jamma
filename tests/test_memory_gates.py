"""Tests for memory gate OOM prevention in PipelineRunner and check_memory_available.

Covers ERRP-05: memory gate code paths in both pipeline_memory.memory_preflight
and check_memory_available are tested using mock psutil to simulate low-memory
conditions without requiring actual large allocations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.memory import check_memory_available
from jamma.lmm.runner import ExecutionPlan
from jamma.lmm.schema import LmmConfig
from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.pipeline_memory import memory_preflight
from tests.fixture_paths import SYNTHETIC

pytestmark = pytest.mark.tier0

BFILE = SYNTHETIC.bfile


class TestMemoryGates:
    """Integration tests for memory gate OOM prevention."""

    def test_budget_exceeded_raises(self):
        """Budget-exceeded path: 1 MB budget raises MemoryError with 'exceeds' message.

        memory_preflight raises MemoryError when
        the plan's peak exceeds config.mem_budget.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=True, mem_budget=0.001)
        runner = PipelineRunner(config)

        with pytest.raises(MemoryError, match="exceeds"):
            memory_preflight(
                runner.config,
                ExecutionPlan(mode="streaming", reason="test"),
                n_valid=100,
                n_snps=500,
                n_cvt=1,
            )

    @patch("jamma.core.memory._check_available", return_value=(0.001, False))
    def test_insufficient_system_memory_raises(self, mock_check):
        """Insufficient system memory raises MemoryError with 'Insufficient' message.

        Mocks _check_available to return (0.001 GB, False), simulating a system
        with nearly no available memory. memory_preflight must raise when
        the plan reports sufficient=False.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=True)
        runner = PipelineRunner(config)

        with pytest.raises(MemoryError, match="Insufficient"):
            memory_preflight(
                runner.config,
                ExecutionPlan(mode="streaming", reason="test"),
                n_valid=100,
                n_snps=500,
                n_cvt=1,
            )

    @patch("jamma.core.memory._check_available", return_value=(1000.0, True))
    def test_memory_check_passes_when_sufficient(self, mock_check):
        """Sufficient memory (1 TB available) returns StreamingMemoryBreakdown.

        Mocks _check_available to return (1000.0 GB, True), simulating ample
        memory. memory_preflight must return the plan, not raise.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=True)
        runner = PipelineRunner(config)

        result = memory_preflight(
            runner.config,
            ExecutionPlan(mode="streaming", reason="test"),
            n_valid=100,
            n_snps=500,
            n_cvt=1,
        )

        assert result is not None
        assert result.mode == "streaming"
        assert result.sufficient is True

    def test_memory_check_disabled_returns_none(self):
        """check_memory=False returns None without performing any memory check.

        When check_memory is disabled, memory_preflight must return
        None immediately, even with a tiny (realistic) dataset.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=False)
        runner = PipelineRunner(config)

        result = memory_preflight(
            runner.config,
            ExecutionPlan(mode="streaming", reason="test"),
            n_valid=100,
            n_snps=500,
            n_cvt=1,
        )

        assert result is None

    def test_check_memory_available_raises_on_insufficient(self):
        """check_memory_available raises MemoryError when psutil reports 1 MB available.

        Pins the machine at 1 MB available through the available_ram_gb seam.
        Requesting 100 GB must raise MemoryError.
        """
        with patch("jamma.core.memory.available_ram_gb", return_value=0.001):
            with pytest.raises(MemoryError, match="Insufficient memory"):
                check_memory_available(required_gb=100.0, operation="test")

    def test_check_memory_available_passes_when_sufficient(self):
        """check_memory_available returns True when psutil reports 1 TB available.

        Pins the machine at 1 TB available through the available_ram_gb seam.
        Requesting 1 GB must succeed without raising.
        """
        with patch("jamma.core.memory.available_ram_gb", return_value=1000.0):
            result = check_memory_available(required_gb=1.0)

        assert result is True


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
        bfile = SYNTHETIC.bfile

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

            # Patch where it is used, not where it is defined. The batch
            # preflight used to import it inside the function, so patching
            # jamma.core.memory worked by accident of import placement; a
            # module-level import there would have silently un-patched this.
            with patch(
                "jamma.pipeline_memory.estimate_lmm_memory",
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
                    config=LmmConfig(check_memory=True, show_progress=False),
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


class TestKinshipOnlyPreflight:
    """The kinship gate must size the kinship phase, not the whole workflow.

    ``-gk`` writes a kinship matrix and never eigendecomposes, but the gate
    inside ``compute_kinship_streaming`` charged callers for
    ``max(kinship, eigendecomp, lmm)``. That refused kinship-only runs on
    machines with ample room for the kinship phase itself.
    """

    def test_streaming_breakdown_exposes_kinship_phase_peak(self):
        """The per-phase kinship peak is reported, not just the workflow max."""
        from jamma.core.memory import estimate_streaming_memory

        est = estimate_streaming_memory(50_000, chunk_size=10_000)

        assert est.peak_kinship_gb == pytest.approx(
            est.kinship_gb + est.chunk_gb + est.dsyrk_scratch_gb
        )
        assert est.peak_kinship_gb < est.total_peak_gb, (
            "eigendecomp phase should dominate the workflow max at this scale"
        )

    def test_kinship_only_run_not_blocked_by_eigendecomp_budget(self):
        """Memory that fits the kinship phase but not eigendecomp must pass.

        50,000 samples: kinship phase needs ~24 GB, the full workflow max is
        ~80 GB. With 40 GB available a kinship-only run fits and must proceed.
        """
        from jamma.core.memory import estimate_streaming_memory
        from jamma.kinship.compute import _preflight_kinship_memory

        est = estimate_streaming_memory(50_000, chunk_size=10_000)
        assert est.peak_kinship_gb < 40.0 < est.total_peak_gb, (
            "test fixture no longer straddles the two budgets"
        )

        with patch("jamma.core.memory.available_ram_gb", return_value=40.0):
            _preflight_kinship_memory(n_samples=50_000, chunk_size=10_000)

    def test_kinship_only_run_still_blocked_when_kinship_does_not_fit(self):
        """The gate still refuses when the kinship phase itself will not fit."""
        from jamma.kinship.compute import _preflight_kinship_memory

        with patch("jamma.core.memory.available_ram_gb", return_value=1.0):
            with pytest.raises(MemoryError, match="Insufficient memory"):
                _preflight_kinship_memory(n_samples=50_000, chunk_size=10_000)


class TestNumpyFallbackKinshipMemory:
    """The NumPy DSYRK fallback must hold no more than it declares.

    ``_preflight_kinship_memory`` budgets the accumulator, one genotype chunk,
    and whatever ``jlinalg.dsyrk_scratch_bytes`` declares. The fallback once
    allocated a full N x N ``np.dot`` result plus the N^2/2 index arrays a
    whole-matrix mirror needs, none of it declared, so the gate could approve a
    run that then OOMs. These tests pin the declaration to the real allocation.

    Measured with ``tracemalloc`` because the claim is about numpy allocations.
    RSS is the wrong instrument: the first matmul faults in ~115 MB of one-time
    Accelerate thread-pool state that no later call repeats, which swamps the
    per-call scratch at these sizes and scales with nothing in the estimate.
    """

    # Python object headers on the transient arrays. Measured at 1416 bytes and
    # flat in n; the budget it guards is expressed in GB.
    _HEADER_SLACK_BYTES = 64 << 10

    @staticmethod
    def _fallback_peak_bytes(monkeypatch, n: int, batch: int) -> tuple[int, int]:
        """Return (measured peak, declared bound) for one fallback accumulation.

        Forces the fallback by swapping the resolved backend, which is BLAS
        detection state rather than numerical behaviour.
        """
        import gc
        import tracemalloc

        from jamma import jlinalg
        from jamma.kinship.compute import _accumulate_kinship

        monkeypatch.setattr(
            jlinalg,
            "_dsyrk_backend",
            jlinalg._dsyrk_numpy_impl,
            # allow-patch: forces the dispatch fallback. _dsyrk_backend is
            # resolved from blas_has_dsyrk at import time, so toggling that
            # flag afterwards would not redirect dispatch.
        )

        K = np.zeros((n, n))
        X = np.ascontiguousarray(np.random.default_rng(1).standard_normal((n, batch)))
        _accumulate_kinship(K, X)  # warm BLAS so its one-time state is excluded

        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            _accumulate_kinship(K, X)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return peak - before, jlinalg.dsyrk_scratch_bytes(n)

    @pytest.mark.parametrize("n", [1000, 3000])
    def test_fallback_holds_no_more_than_it_declares(self, monkeypatch, n):
        """Whatever the fallback allocates must be covered by its declaration."""
        measured, declared = self._fallback_peak_bytes(monkeypatch, n, 200)

        assert declared > 0, "fallback must declare a non-zero scratch bound"
        assert measured <= declared + self._HEADER_SLACK_BYTES, (
            f"n={n}: fallback held {measured / 1e6:.2f} MB but declares "
            f"{declared / 1e6:.2f} MB; the kinship pre-flight budgets the "
            f"declared figure, so the gate would approve a run that OOMs"
        )

    def test_fallback_scratch_stays_far_below_the_accumulator(self, monkeypatch):
        """The declared bound must be a fraction of the matrix, not a multiple."""
        n = 3000
        _measured, declared = self._fallback_peak_bytes(monkeypatch, n, 200)

        assert declared < n * n * 8 // 4, (
            f"fallback declares {declared / 1e6:.2f} MB against a "
            f"{n * n * 8 / 1e6:.0f} MB accumulator; blocking should keep the "
            f"scratch well under a quarter of the output"
        )

    def test_estimator_budgets_the_declared_scratch(self, monkeypatch):
        """The kinship phase peak must include the fallback's declaration."""
        from jamma import jlinalg
        from jamma.core.memory import estimate_streaming_memory

        monkeypatch.setattr(
            jlinalg,
            "_dsyrk_backend",
            jlinalg._dsyrk_numpy_impl,
            # allow-patch: forces the dispatch fallback. _dsyrk_backend is
            # resolved from blas_has_dsyrk at import time, so toggling that
            # flag afterwards would not redirect dispatch.
        )

        est = estimate_streaming_memory(50_000, chunk_size=10_000)

        assert est.dsyrk_scratch_gb == pytest.approx(
            jlinalg.dsyrk_scratch_bytes(50_000) / 1e9
        )
        assert est.peak_kinship_gb == pytest.approx(
            est.kinship_gb + est.chunk_gb + est.dsyrk_scratch_gb
        )

    def test_native_backend_declares_no_scratch(self):
        """The native path accumulates in place, so it budgets nothing extra."""
        from jamma import jlinalg

        if jlinalg._dsyrk_backend is jlinalg._dsyrk_numpy_impl:
            pytest.skip("no native dsyrk on this build")

        assert jlinalg.dsyrk_scratch_bytes(50_000) == 0
