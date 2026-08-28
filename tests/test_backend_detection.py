"""Tests for compute backend detection and information."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from jamma.core.backend import get_backend_info
from jamma.core.memory import MemoryBreakdown
from jamma.lmm.runner import ExecutionPlan, select_execution_mode

pytestmark = pytest.mark.tier0

# Stands in for a loaded extension. Only `is not None` is read on
# the paths under test, so the object's identity is all that matters.
_EXTENSION_LOADED = object()


class TestBackendInfo:
    """Tests for get_backend_info function."""

    def test_returns_dict(self):
        """Should return a dictionary with expected keys."""
        info = get_backend_info()

        assert isinstance(info, dict)
        assert "selected" in info
        assert set(info.keys()) == {"selected"}

    def test_selected_is_numpy(self):
        """Selected backend should always be 'numpy'."""
        info = get_backend_info()
        assert info["selected"] == "numpy"


def test_import_jamma_succeeds():
    """Smoke test: importing jamma should not raise."""
    import jamma

    assert hasattr(jamma, "__version__")


def _make_sufficient_estimate() -> MemoryBreakdown:
    """Build a MemoryBreakdown with sufficient=True."""
    return MemoryBreakdown(
        kinship_gb=0.1,
        genotypes_gb=0.1,
        eigenvectors_gb=0.1,
        eigendecomp_workspace_gb=0.1,
        lmm_rotated_gb=0.1,
        lmm_batch_gb=0.1,
        total_gb=1.0,
        available_gb=100.0,
        sufficient=True,
    )


def _make_insufficient_estimate() -> MemoryBreakdown:
    """Build a MemoryBreakdown with sufficient=False."""
    return MemoryBreakdown(
        kinship_gb=100.0,
        genotypes_gb=100.0,
        eigenvectors_gb=100.0,
        eigendecomp_workspace_gb=100.0,
        lmm_rotated_gb=50.0,
        lmm_batch_gb=50.0,
        total_gb=500.0,
        available_gb=10.0,
        sufficient=False,
    )


class TestExecutionMode:
    """Tests for ExecutionPlan and select_execution_mode in runner.py."""

    # -- Auto selection --

    def test_auto_c_ext_memory_sufficient_returns_numpy_batch(self):
        """auto + C ext + memory sufficient -> numpy-batch."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.compute_numpy._accel", _EXTENSION_LOADED),
        ):
            plan = select_execution_mode(100, 1000)
        assert plan.mode == "batch"

    def test_auto_c_ext_memory_insufficient_returns_numpy_streaming(self):
        """auto + C ext + memory insufficient -> numpy-streaming."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_insufficient_estimate(),
            ),
            patch("jamma.lmm.compute_numpy._accel", _EXTENSION_LOADED),
        ):
            plan = select_execution_mode(200_000, 100_000)
        assert plan.mode == "streaming"

    def test_auto_no_c_ext_returns_numpy_batch(self):
        """auto + no C ext -> numpy-batch fallback."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.compute_numpy._accel", None),
        ):
            plan = select_execution_mode(100, 1000)
        assert plan.mode == "batch"

    def test_no_c_ext_warns_for_large_dataset(self):
        """No C extension + insufficient memory logs warning."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_insufficient_estimate(),
            ),
            patch("jamma.lmm.compute_numpy._accel", None),
            patch("jamma.lmm.runner.logger") as mock_logger,
        ):
            plan = select_execution_mode(n_samples=100, n_snps=1000)
        assert plan.mode == "batch"
        mock_logger.warning.assert_called_once()

    # -- Explicit backend selection --

    def test_explicit_numpy_returns_numpy_batch(self):
        """explicit 'numpy' -> numpy-batch regardless of memory."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_insufficient_estimate(),
            ),
            patch("jamma.lmm.compute_numpy._accel", _EXTENSION_LOADED),
        ):
            plan = select_execution_mode(200_000, 100_000, requested="numpy")
        assert plan.mode == "batch"

    def test_explicit_numpy_bypasses_auto(self):
        """Explicit backend='numpy' in PipelineConfig bypasses auto-selection."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(bfile=Path("/tmp/test"), backend="numpy")
        assert config.backend == "numpy"

    # -- Compound backend requests --

    def test_explicit_numpy_streaming_returns_numpy_streaming(self):
        """explicit 'numpy-streaming' -> numpy-streaming directly."""
        with patch("jamma.lmm.compute_numpy._accel", _EXTENSION_LOADED):
            plan = select_execution_mode(100, 1000, requested="numpy-streaming")
        assert plan.mode == "streaming"

    def test_explicit_numpy_streaming_no_c_ext_raises(self):
        """explicit 'numpy-streaming' + no C extension -> ValueError."""
        with patch("jamma.lmm.compute_numpy._accel", None):
            with pytest.raises(ValueError, match="C extension"):
                select_execution_mode(100, 1000, requested="numpy-streaming")

    # -- Input validation --

    def test_invalid_requested_backend_raises(self):
        """Unknown requested backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            select_execution_mode(100, 1000, requested="gpu")  # type: ignore[bad-argument-type]

    def test_jax_requested_raises(self):
        """Requesting 'jax' backend raises ValueError (removed backend)."""
        with pytest.raises(ValueError, match="Unknown backend"):
            select_execution_mode(100, 1000, requested="jax")  # type: ignore[bad-argument-type]

    # -- ExecutionPlan invariants --

    def test_runner_name_property(self):
        """ExecutionPlan.runner_name returns 'numpy-{mode}'."""
        plan = ExecutionPlan(mode="batch", reason="test")
        assert plan.runner_name == "numpy-batch"

        plan2 = ExecutionPlan(mode="streaming", reason="test")
        assert plan2.runner_name == "numpy-streaming"

    def test_numpy_streaming_is_valid(self):
        """ExecutionPlan accepts numpy-streaming (numpy streaming runner available)."""
        plan = ExecutionPlan(mode="streaming", reason="test")
        assert plan.mode == "streaming"
        assert plan.runner_name == "numpy-streaming"

    def test_empty_reason_is_invalid(self):
        """ExecutionPlan rejects empty reason string."""
        with pytest.raises(ValueError, match="reason must be non-empty"):
            ExecutionPlan(mode="batch", reason="")

    def test_equality_excludes_reason(self):
        """Plans with same backend/mode but different reasons are equal."""
        plan_a = ExecutionPlan("batch", "reason A")
        plan_b = ExecutionPlan("batch", "reason B")
        assert plan_a == plan_b

    def test_inequality_on_different_mode(self):
        """Plans with different modes are not equal."""
        plan_a = ExecutionPlan("batch", "test")
        plan_b = ExecutionPlan("streaming", "test")
        assert plan_a != plan_b

    def test_hashable_ignores_reason(self):
        """Plans with same backend/mode hash identically despite different reasons."""
        plan_a = ExecutionPlan("batch", "reason A")
        plan_b = ExecutionPlan("batch", "reason B")
        assert hash(plan_a) == hash(plan_b)
        assert len({plan_a, plan_b}) == 1

    def test_plan_reevaluation_mode_change_allowed(self):
        """Mode change (same backend) during re-evaluation is allowed."""
        sufficient = _make_sufficient_estimate()
        insufficient = _make_insufficient_estimate()

        # First call returns numpy-batch, second returns numpy-streaming
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                side_effect=[sufficient, insufficient],
            ),
            patch("jamma.lmm.compute_numpy._accel", _EXTENSION_LOADED),
        ):
            plan1 = select_execution_mode(100, 1000)
            plan2 = select_execution_mode(100, 1000)
        assert plan1.mode == "batch"
        assert plan2.mode == "streaming"

    # -- n_cvt-aware selection --

    def test_n_cvt_affects_memory(self):
        """select_execution_mode passes n_cvt to estimate_lmm_memory."""
        calls = []

        def capturing_estimate(n_samples, n_snps, **kwargs):
            calls.append(kwargs)
            return _make_sufficient_estimate()

        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                side_effect=capturing_estimate,
            ),
            patch("jamma.lmm.compute_numpy._accel", _EXTENSION_LOADED),
        ):
            select_execution_mode(1000, 10000, n_cvt=4)

        # At least one call should have n_cvt=4
        assert any(c.get("n_cvt") == 4 for c in calls), (
            f"n_cvt=4 not passed to estimate_lmm_memory; calls={calls}"
        )

    def test_no_c_general_falls_to_numpy_batch_for_n_cvt_gt1(self):
        """No C general + n_cvt>1 -> numpy-batch fallback."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.compute_numpy._accel", _EXTENSION_LOADED),
            patch(
                "jamma.lmm.compute_numpy._accel",
                None,
                create=True,
            ),
        ):
            plan_n_cvt4 = select_execution_mode(1000, 10000, n_cvt=4)
            plan_n_cvt1 = select_execution_mode(1000, 10000, n_cvt=1)

        # n_cvt=4 without C general -> falls through to numpy-batch fallback
        assert plan_n_cvt4.mode == "batch"
        # n_cvt=1 still uses numpy-batch (C extension handles n_cvt=1)
        assert plan_n_cvt1.mode == "batch"

    def test_c_general_n_cvt_numpy_batch(self):
        """C general available + n_cvt>1 + sufficient -> numpy-batch."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.compute_numpy._accel", _EXTENSION_LOADED),
            patch(
                "jamma.lmm.compute_numpy._accel",
                _EXTENSION_LOADED,
                create=True,
            ),
        ):
            plan = select_execution_mode(1000, 10000, n_cvt=4)

        assert plan.mode == "batch"
