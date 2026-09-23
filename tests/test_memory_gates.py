"""Tests for memory gate OOM prevention in PipelineRunner and the kinship gates.

Covers ERRP-05: the memory gate code paths are tested by pinning
``available_ram_gb`` to simulate low-memory conditions without requiring
actual large allocations.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core import memory
from jamma.core.memory import array_gb, estimate_kinship_memory
from jamma.lmm import accel
from jamma.lmm.association_plan import ExecutableAssociationPlan, plan_association
from jamma.lmm.chunk_sizing import lmm_extra_bytes_per_snp
from jamma.lmm.dispatch import select_dispatch_path
from jamma.lmm.schema import LmmConfig
from jamma.pipeline import PipelineConfig, PipelineRunner
from tests.builders import empty_workspace
from tests.conftest import preflight
from tests.fixture_paths import SYNTHETIC

pytestmark = pytest.mark.tier0

BFILE = SYNTHETIC.bfile


def _streaming_plan(*, mem_budget: float | None = None):  # type: ignore[no-untyped-def]
    return plan_association(
        100, 500, config=LmmConfig(mem_budget=mem_budget), backend="numpy-streaming"
    )


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
            preflight(
                runner.config,
                _streaming_plan(mem_budget=config.mem_budget),
            )

    def test_insufficient_system_memory_raises(self):
        """Insufficient system memory raises MemoryError with 'Insufficient' message.

        Pins the machine at 1 MB available, so memory_preflight must refuse
        even the smallest streaming plan.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=True)
        runner = PipelineRunner(config)

        with patch("jamma.core.memory.available_ram_gb", return_value=0.001):
            with pytest.raises(MemoryError, match="Insufficient"):
                preflight(
                    runner.config,
                    _streaming_plan(),
                )

    def test_memory_check_passes_when_sufficient(self):
        """Sufficient memory (1 TB available) passes the gate."""
        config = PipelineConfig(bfile=BFILE, check_memory=True)
        runner = PipelineRunner(config)
        plan = _streaming_plan()

        with patch("jamma.core.memory.available_ram_gb", return_value=1000.0):
            preflight(runner.config, plan)

        assert memory.fits(plan.price(eigen=None).total_peak_gb, 1000.0)

    def test_memory_check_disabled_returns_none(self):
        """check_memory=False returns None without performing any memory check.

        When check_memory is disabled, memory_preflight must return
        None immediately, even with a tiny (realistic) dataset.
        """
        config = PipelineConfig(bfile=BFILE, check_memory=False)
        runner = PipelineRunner(config)

        result = preflight(
            runner.config,
            _streaming_plan(),
        )

        assert result is None


def _expected_uab_iab_gb(plan: ExecutableAssociationPlan, n_cvt: int) -> float:
    """The Uab/Iab figure a correct batch quote carries for this plan's chunk."""
    dispatch = select_dispatch_path(n_cvt, 1, accel=accel.available())
    per_snp = lmm_extra_bytes_per_snp(plan.n_samples, n_cvt, dispatch)
    return plan.conservative_chunks.chunk_size * per_snp / 1e9


def _priced_uab_iab_gb(plan: ExecutableAssociationPlan) -> float:
    """The batch quote left once U, genotypes, rotation, and workspace are removed."""
    n = plan.n_samples
    chunks = plan.conservative_chunks
    bare = replace(
        plan,
        workspace=empty_workspace(plan.dispatch, n, plan.n_input_samples, plan.n_cvt),
    )
    return bare.price(eigen=None).association_gb - (
        array_gb(n, n)
        + array_gb(plan.n_input_samples, plan.n_snps_before_filter)
        + chunks.n_buffers * array_gb(n, chunks.chunk_size)
    )


def _recording_plan_association(recorded: list[ExecutableAssociationPlan]):  # type: ignore[no-untyped-def]
    """Wrap the real planner: keep the plan it built, then stop the run there."""

    def record(*args, **kwargs):  # type: ignore[no-untyped-def]
        recorded.append(plan_association(*args, **kwargs))
        raise RuntimeError("stop-at-plan-sentinel")

    return record


class TestBatchPreflightPricesItsDispatchPath:
    """Regression: the batch quote must carry the run's own Uab/Iab figure.

    Both batch preflight call sites previously omitted n_cvt, so
    multi-covariate runs passed the preflight on the n_cvt=1 figure and then
    OOMed at real allocation time in compute_numpy._run_inner.

    ``price()`` reads n_cvt and dispatch from the plan it prices, so each
    call site's plan is the contract: the tests keep the plan the real
    planner builds for the call site and check the Uab/Iab its quote carries.
    """

    def test_pipeline_batch_preflight_prices_its_dispatch_path(self):
        """PipelineRunner batch branch must price the run's own dispatch path.

        The batch preflight site previously priced with no n_cvt, silently
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

        plans: list[ExecutableAssociationPlan] = []

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

            with patch(
                "jamma.pipeline.plan_association",
                side_effect=_recording_plan_association(plans),
            ):
                with pytest.raises(RuntimeError, match="stop-at-plan-sentinel"):
                    runner.run()

        (plan,) = plans
        assert plan.summary.mode == "batch"
        assert plan.n_cvt == 3, "n_cvt from the 3-column covariate file"
        assert _priced_uab_iab_gb(plan) == pytest.approx(
            _expected_uab_iab_gb(plan, 3), abs=1e-12
        )

    def test_runner_numpy_preflight_prices_its_dispatch_path(self):
        """run_lmm_association_numpy must price the run's own dispatch path.

        The batch runner preflight previously priced with no n_cvt.
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
        # snp_info is not touched before planning, so an empty list is
        # safe: the sentinel raises before any iteration.
        snp_info: list = []
        plans: list[ExecutableAssociationPlan] = []

        with patch(
            "jamma.lmm.runner_numpy.plan_association",
            side_effect=_recording_plan_association(plans),
        ):
            with pytest.raises(RuntimeError, match="stop-at-plan-sentinel"):
                run_lmm_association_numpy(
                    genotypes=genotypes,
                    phenotypes=phenotypes,
                    kinship=kinship,
                    snp_info=snp_info,
                    covariates=covariates,
                    config=LmmConfig(check_memory=True, show_progress=False),
                )

        (plan,) = plans
        assert plan.summary.mode == "batch"
        assert plan.n_cvt == expected_n_cvt
        assert _priced_uab_iab_gb(plan) == pytest.approx(
            _expected_uab_iab_gb(plan, expected_n_cvt), abs=1e-12
        )


class TestKinshipOnlyPreflight:
    """The kinship gate must size the kinship phase, not the whole workflow.

    ``-gk`` writes a kinship matrix and never eigendecomposes, but the gate
    inside ``compute_kinship_streaming`` charged callers for
    ``max(kinship, eigendecomp, lmm)``. That refused kinship-only runs on
    machines with ample room for the kinship phase itself.
    """

    def test_quote_exposes_kinship_phase_peak(self):
        """The per-phase kinship peak is reported, not just the workflow max."""
        from jamma.core.eigen_plan import EigenDriver, EigenDriverPlan, _dsyevd_peak_gb
        from jamma.lmm.association_plan import KinshipShape
        from tests.builders import association_price_plan

        plan = replace(
            association_price_plan(
                "streaming", n_samples=50_000, n_snps=10_000, chunk_size=10_000
            ),
            kinship=KinshipShape(50_000, loaded=False),
        )
        eigen = EigenDriverPlan(EigenDriver.DSYEVD, _dsyevd_peak_gb(50_000), "test")
        quote = plan.price(eigen=eigen)

        assert quote.kinship_gb < quote.total_peak_gb, (
            "eigendecomp phase should dominate the workflow max at this scale"
        )
        assert quote.total_peak_gb == quote.eigen_gb

    def test_kinship_only_run_not_blocked_by_eigendecomp_budget(self):
        """Memory that fits the kinship phase but not eigendecomp must pass.

        50,000 samples: kinship phase needs ~24 GB, the full workflow max is
        ~80 GB. With 40 GB available a kinship-only run fits and must proceed.
        """
        from jamma.core.eigen_plan import _dsyevd_peak_gb
        from jamma.kinship.stream import _preflight_kinship_memory

        kinship_gb = estimate_kinship_memory(
            n_input_samples=50_000,
            n_output_samples=50_000,
            n_snps=10_000,
            chunk_size=10_000,
        )
        assert kinship_gb < 40.0 < _dsyevd_peak_gb(50_000), (
            "test fixture no longer straddles the two budgets"
        )

        with patch("jamma.core.memory.available_ram_gb", return_value=40.0):
            _preflight_kinship_memory(
                n_input_samples=50_000,
                n_output_samples=50_000,
                n_snps=10_000,
                chunk_size=10_000,
                mem_budget=None,
            )

    def test_kinship_only_run_still_blocked_when_kinship_does_not_fit(self):
        """The gate still refuses when the kinship phase itself will not fit."""
        from jamma.kinship.stream import _preflight_kinship_memory

        with patch("jamma.core.memory.available_ram_gb", return_value=1.0):
            with pytest.raises(MemoryError, match="Insufficient memory"):
                _preflight_kinship_memory(
                    n_input_samples=50_000,
                    n_output_samples=50_000,
                    n_snps=10_000,
                    chunk_size=10_000,
                    mem_budget=None,
                )


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
        from jamma.kinship.accumulation import accumulate_kinship

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
        accumulate_kinship(K, X)  # warm BLAS so its one-time state is excluded

        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            accumulate_kinship(K, X)
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
        """The kinship phase peak must include the fallback's declaration.

        The expected side is the pure formula, not ``dsyrk_scratch_bytes``
        under the same patch: that would hold whether or not the patch took.
        """
        import importlib

        from jamma.jlinalg._dsyrk import numpy_impl, scratch_bytes

        jlinalg = importlib.import_module("jamma.jlinalg")
        monkeypatch.setattr(
            jlinalg,
            "_dsyrk_backend",
            numpy_impl,
            # allow-patch: forces the dispatch fallback. _dsyrk_backend is
            # resolved from blas_has_dsyrk at import time, so toggling that
            # flag afterwards would not redirect dispatch.
        )

        kinship_gb = estimate_kinship_memory(
            n_input_samples=50_000,
            n_output_samples=50_000,
            n_snps=10_000,
            chunk_size=10_000,
        )

        declared = scratch_bytes(50_000, numpy_impl)
        assert declared > 0
        assert kinship_gb == pytest.approx(
            array_gb(50_000, 50_000) + 3.25 * array_gb(50_000, 10_000) + declared / 1e9
        )

    def test_native_backend_declares_no_scratch(self):
        """The native path accumulates in place, so it budgets nothing extra."""
        from jamma import jlinalg

        if jlinalg._dsyrk_backend is jlinalg._dsyrk_numpy_impl:
            pytest.skip("no native dsyrk on this build")

        assert jlinalg.dsyrk_scratch_bytes(50_000) == 0
