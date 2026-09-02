"""Tests for compute backend detection and information."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from jamma.lmm.association_plan import ExecutionPlan, plan_association

pytestmark = pytest.mark.tier0


def _select_mode(*args, **kwargs) -> ExecutionPlan:
    """Plan and keep only the selected-mode summary, as the pipeline does."""
    return plan_association(*args, **kwargs).summary


# Stands in for a loaded extension. Only `is not None` is read on
# the paths under test, so the object's identity is all that matters.
_EXTENSION_LOADED = object()


def test_import_jamma_succeeds():
    """Smoke test: importing jamma should not raise."""
    import jamma

    assert hasattr(jamma, "__version__")


# A machine with room for anything the small shapes below price, and one with
# effectively none. Pinning the read lets the real ``memory.fits`` drive the
# batch-versus-streaming decision.
AMPLE_GB = 1000.0
SCARCE_GB = 0.001

# Shapes whose real price straddles AMPLE_GB: 100 samples x 1000 SNPs costs
# well under a GB, 200k x 1M holds 1.6TB of genotypes alone.
FITS_SHAPE = (100, 1000)
OVERFLOWS_SHAPE = (200_000, 1_000_000)


def _pin_ram(available_gb: float):
    """Pin what every memory decision in the planner sees the machine report."""
    return patch("jamma.core.memory.available_ram_gb", return_value=available_gb)


class TestExecutionMode:
    """Tests for ExecutionPlan and plan_association mode selection."""

    # -- Auto selection --

    def test_auto_c_ext_memory_sufficient_returns_numpy_batch(self):
        """auto + C ext + memory sufficient -> numpy-batch."""
        with _pin_ram(AMPLE_GB), patch("jamma.lmm.accel._accel", _EXTENSION_LOADED):
            plan = _select_mode(*FITS_SHAPE)
        assert plan.mode == "batch"

    def test_auto_c_ext_memory_insufficient_returns_numpy_streaming(self):
        """auto + C ext + memory insufficient -> numpy-streaming."""
        with _pin_ram(AMPLE_GB), patch("jamma.lmm.accel._accel", _EXTENSION_LOADED):
            plan = _select_mode(*OVERFLOWS_SHAPE)
        assert plan.mode == "streaming"

    def test_auto_no_c_ext_returns_numpy_batch(self):
        """auto + no C ext -> numpy-batch fallback."""
        with _pin_ram(AMPLE_GB), patch("jamma.lmm.accel._accel", None):
            plan = _select_mode(*FITS_SHAPE)
        assert plan.mode == "batch"

    def test_loco_selects_loco_mode_whatever_was_requested(self):
        """loco=True plans the loco mode and prices one chunk, not the matrix."""
        with _pin_ram(AMPLE_GB), patch("jamma.lmm.accel._accel", _EXTENSION_LOADED):
            loco = plan_association(*OVERFLOWS_SHAPE, loco=True)
            batch = plan_association(*OVERFLOWS_SHAPE, requested="numpy")
        assert loco.summary.mode == "loco"
        assert loco.summary.runner_name == "numpy-loco"
        assert loco.price().total_peak_gb < batch.price().total_peak_gb

    def test_no_c_ext_warns_for_large_dataset(self):
        """No C extension + insufficient memory logs warning."""
        with (
            _pin_ram(AMPLE_GB),
            patch("jamma.lmm.accel._accel", None),
            patch("jamma.lmm.association_plan.logger") as mock_logger,
        ):
            plan = _select_mode(*OVERFLOWS_SHAPE)
        assert plan.mode == "batch"
        mock_logger.warning.assert_called_once()

    # -- Explicit backend selection --

    def test_explicit_numpy_returns_numpy_batch(self):
        """explicit 'numpy' -> numpy-batch regardless of memory."""
        with _pin_ram(AMPLE_GB), patch("jamma.lmm.accel._accel", _EXTENSION_LOADED):
            plan = _select_mode(*OVERFLOWS_SHAPE, requested="numpy")
        assert plan.mode == "batch"

    def test_explicit_numpy_bypasses_auto(self):
        """Explicit backend='numpy' in PipelineConfig bypasses auto-selection."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(bfile=Path("/tmp/test"), backend="numpy")
        assert config.backend == "numpy"

    # -- Compound backend requests --

    def test_explicit_numpy_streaming_returns_numpy_streaming(self):
        """explicit 'numpy-streaming' -> numpy-streaming directly."""
        with patch("jamma.lmm.accel._accel", _EXTENSION_LOADED):
            plan = _select_mode(100, 1000, requested="numpy-streaming")
        assert plan.mode == "streaming"

    def test_explicit_numpy_streaming_no_c_ext_selects_streaming(self):
        """The planner plans the requested mode even without the C extension.

        Refusing the combination is the pipeline boundary's policy, tested
        below; the streaming runner itself works without the extension.
        """
        with patch("jamma.lmm.accel._accel", None):
            plan = _select_mode(100, 1000, requested="numpy-streaming")
        assert plan.mode == "streaming"

    def test_pipeline_rejects_numpy_streaming_without_c_ext(self):
        """The pipeline refuses an explicit numpy-streaming request without
        the C extension, before touching any input file."""
        from jamma.pipeline import PipelineConfig, PipelineRunner

        config = PipelineConfig(
            bfile=Path("/nonexistent/prefix"), backend="numpy-streaming"
        )
        with patch("jamma.lmm.accel._accel", None):
            with pytest.raises(ValueError, match="C extension"):
                PipelineRunner(config).run()

    # -- Input validation --

    def test_invalid_requested_backend_raises(self):
        """Unknown requested backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            _select_mode(100, 1000, requested="gpu")  # type: ignore[bad-argument-type]

    def test_jax_requested_raises(self):
        """Requesting 'jax' backend raises ValueError (removed backend)."""
        with pytest.raises(ValueError, match="Unknown backend"):
            _select_mode(100, 1000, requested="jax")  # type: ignore[bad-argument-type]

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
        # The same shape planned twice on a machine that lost its memory
        # between the calls: numpy-batch first, numpy-streaming second.
        with patch("jamma.lmm.accel._accel", _EXTENSION_LOADED):
            with _pin_ram(AMPLE_GB):
                plan1 = _select_mode(*FITS_SHAPE)
            with _pin_ram(SCARCE_GB):
                plan2 = _select_mode(*FITS_SHAPE)
        assert plan1.mode == "batch"
        assert plan2.mode == "streaming"

    # -- n_cvt-aware selection --

    def test_n_cvt_affects_memory(self):
        """plan_association passes n_cvt to estimate_lmm_memory."""
        calls = []

        def capturing_estimate(n_samples, n_snps, **kwargs):
            calls.append(kwargs)
            return 1.0

        with (
            _pin_ram(AMPLE_GB),
            patch(
                "jamma.lmm.association_plan.estimate_lmm_memory",
                side_effect=capturing_estimate,
            ),
            patch("jamma.lmm.accel._accel", _EXTENSION_LOADED),
        ):
            _select_mode(1000, 10000, n_cvt=4)

        # At least one call should have n_cvt=4
        assert any(c.get("n_cvt") == 4 for c in calls), (
            f"n_cvt=4 not passed to estimate_lmm_memory; calls={calls}"
        )

    def test_no_c_general_falls_to_numpy_batch_for_n_cvt_gt1(self):
        """No C general + n_cvt>1 -> numpy-batch fallback."""
        with _pin_ram(AMPLE_GB), patch("jamma.lmm.accel._accel", None):
            plan_n_cvt4 = _select_mode(1000, 10000, n_cvt=4)
            plan_n_cvt1 = _select_mode(1000, 10000, n_cvt=1)

        # n_cvt=4 without C general -> falls through to numpy-batch fallback
        assert plan_n_cvt4.mode == "batch"
        # n_cvt=1 still uses numpy-batch (C extension handles n_cvt=1)
        assert plan_n_cvt1.mode == "batch"

    def test_c_general_n_cvt_numpy_batch(self):
        """C general available + n_cvt>1 + sufficient -> numpy-batch."""
        with _pin_ram(AMPLE_GB), patch("jamma.lmm.accel._accel", _EXTENSION_LOADED):
            plan = _select_mode(1000, 10000, n_cvt=4)

        assert plan.mode == "batch"
