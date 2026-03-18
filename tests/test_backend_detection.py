"""Tests for compute backend detection and information."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jamma.core.backend import detect_backend, get_backend_info
from jamma.lmm.runner import ExecutionPlan, select_execution_mode


@pytest.mark.tier0
class TestBackendInfo:
    """Tests for get_backend_info function."""

    def test_returns_dict(self):
        """Should return a dictionary with expected keys."""
        info = get_backend_info()

        assert isinstance(info, dict)
        assert "selected" in info
        assert "jax_available" in info
        assert set(info.keys()) == {"selected", "jax_available"}

    def test_selected_is_valid_backend(self):
        """Selected backend should be 'jax' or 'numpy'."""
        info = get_backend_info()
        assert info["selected"] in ("jax", "numpy")

    def test_jax_available_is_bool(self):
        """jax_available should be a boolean."""
        info = get_backend_info()
        assert isinstance(info["jax_available"], bool)


@pytest.mark.tier0
class TestDetectBackend:
    """Tests for detect_backend() function."""

    @pytest.mark.requires_jax
    def test_auto_returns_jax_when_available(self):
        """detect_backend('auto') returns 'jax' in dev env where JAX is installed."""
        result = detect_backend("auto")
        assert result == "jax"

    def test_numpy_always_returns_numpy(self):
        """detect_backend('numpy') always returns 'numpy', regardless of JAX."""
        result = detect_backend("numpy")
        assert result == "numpy"

    @pytest.mark.requires_jax
    def test_jax_returns_jax_when_available(self):
        """detect_backend('jax') returns 'jax' in dev env where JAX is installed."""
        result = detect_backend("jax")
        assert result == "jax"

    def test_invalid_backend_raises(self):
        """detect_backend with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            detect_backend("invalid")

    def test_env_var_overrides_requested(self, monkeypatch):
        """JAMMA_BACKEND=numpy overrides detect_backend('jax')."""
        monkeypatch.setenv("JAMMA_BACKEND", "numpy")
        result = detect_backend("jax")
        assert result == "numpy"

    @pytest.mark.requires_jax
    def test_env_var_jax_overrides_auto(self, monkeypatch):
        """JAMMA_BACKEND=jax overrides detect_backend('auto') explicitly."""
        monkeypatch.setenv("JAMMA_BACKEND", "jax")
        result = detect_backend("auto")
        assert result == "jax"

    def test_env_var_invalid_raises(self, monkeypatch):
        """JAMMA_BACKEND with invalid value raises ValueError."""
        monkeypatch.setenv("JAMMA_BACKEND", "spark")
        with pytest.raises(ValueError, match="Unknown backend"):
            detect_backend("auto")

    def test_compound_numpy_streaming_resolves(self):
        """detect_backend('numpy-streaming') resolves to 'numpy'."""
        result = detect_backend("numpy-streaming")
        assert result == "numpy"

    def test_compound_jax_streaming_resolves(self):
        """detect_backend('jax-streaming') resolves to base backend."""
        # jax-streaming resolves to 'jax' — but JAX may not be installed,
        # so this may raise ValueError about JAX not installed (which is fine).
        try:
            result = detect_backend("jax-streaming")
            assert result == "jax"
        except ValueError as e:
            assert "JAX is not installed" in str(e)


@pytest.mark.tier0
class TestDetectBackendJaxAbsent:
    """T1: detect_backend('jax') raises ValueError when JAX is absent."""

    def test_jax_requested_when_absent_raises(self, monkeypatch):
        """detect_backend('jax') raises ValueError when JAX import fails."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jax":
                raise ImportError("mock: jax not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Clear the has_jax cache so detect_backend re-probes
        from jamma.core import backend

        backend.has_jax.cache_clear()
        try:
            with pytest.raises(ValueError, match="JAX is not installed"):
                detect_backend("jax")
        finally:
            backend.has_jax.cache_clear()


@pytest.mark.tier0
def test_import_jamma_succeeds():
    """Smoke test: importing jamma should not raise even if JAX is available."""
    import jamma

    assert hasattr(jamma, "__version__")


def _make_sufficient_estimate():
    """Build a MemoryBreakdown-like mock with sufficient=True."""
    m = MagicMock()
    m.sufficient = True
    m.total_gb = 1.0
    m.available_gb = 100.0
    return m


def _make_insufficient_estimate():
    """Build a MemoryBreakdown-like mock with sufficient=False."""
    m = MagicMock()
    m.sufficient = False
    m.total_gb = 500.0
    m.available_gb = 10.0
    return m


@pytest.mark.tier0
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
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=True),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            plan = select_execution_mode(100, 1000)
        assert plan.backend == "numpy"
        assert plan.mode == "batch"

    def test_auto_c_ext_memory_insufficient_returns_numpy_streaming(self):
        """auto + C ext + memory insufficient -> numpy-streaming."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_insufficient_estimate(),
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=True),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            plan = select_execution_mode(200_000, 100_000)
        assert plan.backend == "numpy"
        assert plan.mode == "streaming"

    def test_auto_no_c_ext_memory_insufficient_returns_jax_streaming(self):
        """auto + no C ext + JAX + memory insufficient -> jax-streaming."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_insufficient_estimate(),
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=False),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            plan = select_execution_mode(200_000, 100_000)
        assert plan.backend == "jax"
        assert plan.mode == "streaming"

    def test_auto_jax_memory_sufficient_no_c_ext_returns_jax_batch(self):
        """auto + JAX + memory sufficient + no C ext -> jax-batch."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=False),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            plan = select_execution_mode(100, 1000)
        assert plan.backend == "jax"
        assert plan.mode == "batch"

    def test_auto_no_jax_no_c_ext_returns_numpy_batch(self):
        """auto + no JAX + no C ext -> numpy-batch fallback."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=False),
            patch("jamma.lmm.runner.has_jax", return_value=False),
        ):
            plan = select_execution_mode(100, 1000)
        assert plan.backend == "numpy"
        assert plan.mode == "batch"

    def test_no_c_ext_no_jax_warns_for_large_dataset(self):
        """No C extension + no JAX + insufficient memory logs warning."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_insufficient_estimate(),
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=False),
            patch("jamma.lmm.runner.has_jax", return_value=False),
            patch("jamma.lmm.runner.logger") as mock_logger,
        ):
            plan = select_execution_mode(n_samples=100, n_snps=1000)
        assert plan.backend == "numpy"
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
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=True),
        ):
            plan = select_execution_mode(200_000, 100_000, requested="numpy")
        assert plan.backend == "numpy"
        assert plan.mode == "batch"

    def test_explicit_jax_memory_sufficient_returns_jax_batch(self):
        """explicit 'jax' + memory sufficient -> jax-batch."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            plan = select_execution_mode(100, 1000, requested="jax")
        assert plan.backend == "jax"
        assert plan.mode == "batch"

    def test_explicit_jax_memory_insufficient_returns_jax_streaming(self):
        """explicit 'jax' + memory insufficient -> jax-streaming."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_insufficient_estimate(),
            ),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            plan = select_execution_mode(200_000, 100_000, requested="jax")
        assert plan.backend == "jax"
        assert plan.mode == "streaming"

    def test_explicit_jax_absent_raises(self):
        """explicit 'jax' + JAX not installed -> ValueError."""
        with patch("jamma.lmm.runner.has_jax", return_value=False):
            with pytest.raises(ValueError, match="JAX is not installed"):
                select_execution_mode(100, 1000, requested="jax")

    def test_explicit_numpy_bypasses_auto(self):
        """Explicit backend='numpy' in PipelineConfig bypasses auto-selection."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(bfile="/tmp/test", backend="numpy")
        assert config.backend == "numpy"

    def test_explicit_jax_bypasses_auto(self):
        """Explicit backend='jax' in PipelineConfig bypasses auto-selection."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(bfile="/tmp/test", backend="jax")
        assert config.backend == "jax"

    # -- Compound backend requests --

    def test_explicit_numpy_streaming_returns_numpy_streaming(self):
        """explicit 'numpy-streaming' -> numpy-streaming directly."""
        with patch("jamma.lmm.runner.is_c_extension_usable", return_value=True):
            plan = select_execution_mode(100, 1000, requested="numpy-streaming")
        assert plan.backend == "numpy"
        assert plan.mode == "streaming"

    def test_explicit_numpy_streaming_no_c_ext_raises(self):
        """explicit 'numpy-streaming' + no C extension -> ValueError."""
        with patch("jamma.lmm.runner.is_c_extension_usable", return_value=False):
            with pytest.raises(ValueError, match="C extension"):
                select_execution_mode(100, 1000, requested="numpy-streaming")

    def test_explicit_jax_streaming_returns_jax_streaming(self):
        """explicit 'jax-streaming' -> jax-streaming directly."""
        with patch("jamma.lmm.runner.has_jax", return_value=True):
            plan = select_execution_mode(100, 1000, requested="jax-streaming")
        assert plan.backend == "jax"
        assert plan.mode == "streaming"

    def test_explicit_jax_streaming_absent_raises(self):
        """explicit 'jax-streaming' + JAX not installed -> ValueError."""
        with patch("jamma.lmm.runner.has_jax", return_value=False):
            with pytest.raises(ValueError, match="JAX is not installed"):
                select_execution_mode(100, 1000, requested="jax-streaming")

    # -- Input validation --

    def test_invalid_requested_backend_raises(self):
        """Unknown requested backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            select_execution_mode(100, 1000, requested="gpu")

    # -- ExecutionPlan invariants --

    def test_runner_name_property(self):
        """ExecutionPlan.runner_name returns 'backend-mode'."""
        plan = ExecutionPlan(backend="jax", mode="streaming", reason="test")
        assert plan.runner_name == "jax-streaming"

        plan2 = ExecutionPlan(backend="numpy", mode="batch", reason="test")
        assert plan2.runner_name == "numpy-batch"

    def test_numpy_streaming_is_valid(self):
        """ExecutionPlan accepts numpy-streaming (numpy streaming runner available)."""
        plan = ExecutionPlan(backend="numpy", mode="streaming", reason="test")
        assert plan.backend == "numpy"
        assert plan.mode == "streaming"
        assert plan.runner_name == "numpy-streaming"

    def test_empty_reason_is_invalid(self):
        """ExecutionPlan rejects empty reason string."""
        with pytest.raises(ValueError, match="reason must be non-empty"):
            ExecutionPlan(backend="numpy", mode="batch", reason="")

    def test_equality_excludes_reason(self):
        """Plans with same backend/mode but different reasons are equal."""
        plan_a = ExecutionPlan("numpy", "batch", "reason A")
        plan_b = ExecutionPlan("numpy", "batch", "reason B")
        assert plan_a == plan_b

    def test_inequality_on_different_backend(self):
        """Plans with different backends are not equal."""
        plan_a = ExecutionPlan("numpy", "batch", "test")
        plan_b = ExecutionPlan("jax", "batch", "test")
        assert plan_a != plan_b

    def test_inequality_on_different_mode(self):
        """Plans with different modes are not equal."""
        plan_a = ExecutionPlan("jax", "batch", "test")
        plan_b = ExecutionPlan("jax", "streaming", "test")
        assert plan_a != plan_b

    def test_hashable_ignores_reason(self):
        """Plans with same backend/mode hash identically despite different reasons."""
        plan_a = ExecutionPlan("jax", "batch", "reason A")
        plan_b = ExecutionPlan("jax", "batch", "reason B")
        assert hash(plan_a) == hash(plan_b)
        assert len({plan_a, plan_b}) == 1

    def test_plan_reevaluation_mode_change_allowed(self):
        """Mode change (same backend) during re-evaluation is allowed."""
        sufficient = _make_sufficient_estimate()
        insufficient = _make_insufficient_estimate()

        # First call returns jax-batch, second returns jax-streaming
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                side_effect=[sufficient, insufficient],
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=False),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            plan1 = select_execution_mode(100, 1000)
            plan2 = select_execution_mode(100, 1000)
        assert plan1.backend == plan2.backend == "jax"
        assert plan1.mode == "batch"
        assert plan2.mode == "streaming"
        # Same backend → no RuntimeError from pipeline re-evaluation guard

    def test_plan_reevaluation_backend_change_detected(self):
        """Backend change during re-evaluation is detectable via != comparison."""
        sufficient = _make_sufficient_estimate()

        # First call: no C ext → jax-batch; second call: C ext → numpy-batch
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=sufficient,
            ),
            patch(
                "jamma.lmm.runner.is_c_extension_usable",
                side_effect=[False, True],
            ),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            plan1 = select_execution_mode(100, 1000)
            plan2 = select_execution_mode(100, 1000)
        assert plan1.backend == "jax"
        assert plan2.backend == "numpy"
        assert plan1 != plan2  # Pipeline guard would raise RuntimeError

    # -- n_cvt-aware selection (BCKAUTO-01, -02, -03) --

    def test_n_cvt_affects_memory(self):
        """BCKAUTO-01: select_execution_mode passes n_cvt to estimate_lmm_memory."""
        calls = []

        def capturing_estimate(n_samples, n_snps, **kwargs):
            calls.append(kwargs)
            m = MagicMock()
            m.sufficient = True
            m.total_gb = 1.0
            m.available_gb = 100.0
            return m

        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                side_effect=capturing_estimate,
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=True),
            patch("jamma.lmm.runner.has_jax", return_value=True),
        ):
            select_execution_mode(1000, 10000, n_cvt=4)

        # At least one call should have n_cvt=4
        assert any(c.get("n_cvt") == 4 for c in calls), (
            f"n_cvt=4 not passed to estimate_lmm_memory; calls={calls}"
        )

    def test_no_c_general_prefers_jax_for_n_cvt_gt1(self):
        """BCKAUTO-02: No C general + n_cvt>1 -> jax-batch (not numpy-batch)."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=True),
            patch("jamma.lmm.runner.has_jax", return_value=True),
            patch(
                "jamma.lmm.compute_numpy._C_GENERAL_AVAILABLE",
                False,
                create=True,
            ),
        ):
            plan_n_cvt4 = select_execution_mode(1000, 10000, n_cvt=4)
            plan_n_cvt1 = select_execution_mode(1000, 10000, n_cvt=1)

        # n_cvt=4 without C general -> fall through to JAX
        assert plan_n_cvt4.backend == "jax"
        assert plan_n_cvt4.mode == "batch"
        # n_cvt=1 still uses numpy-batch (C extension handles n_cvt=1)
        assert plan_n_cvt1.backend == "numpy"
        assert plan_n_cvt1.mode == "batch"

    def test_no_c_general_n_cvt_gt1_insufficient_falls_to_jax_streaming(self):
        """C ext + no C general + n_cvt>1 + insufficient -> jax-streaming."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_insufficient_estimate(),
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=True),
            patch("jamma.lmm.runner.has_jax", return_value=True),
            patch(
                "jamma.lmm.compute_numpy._C_GENERAL_AVAILABLE",
                False,
                create=True,
            ),
        ):
            plan = select_execution_mode(200_000, 100_000, n_cvt=3)
        assert plan.backend == "jax"
        assert plan.mode == "streaming"

    def test_c_general_n_cvt_numpy_batch(self):
        """BCKAUTO-03: C general available + n_cvt>1 + sufficient -> numpy-batch."""
        with (
            patch(
                "jamma.lmm.runner.estimate_lmm_memory",
                return_value=_make_sufficient_estimate(),
            ),
            patch("jamma.lmm.runner.is_c_extension_usable", return_value=True),
            patch("jamma.lmm.runner.has_jax", return_value=True),
            patch(
                "jamma.lmm.compute_numpy._C_GENERAL_AVAILABLE",
                True,
                create=True,
            ),
        ):
            plan = select_execution_mode(1000, 10000, n_cvt=4)

        assert plan.backend == "numpy"
        assert plan.mode == "batch"

    # -- run_lmm n_cvt forwarding (BCKAUTO-05) --

    def test_run_lmm_forwards_n_cvt(self):
        """BCKAUTO-05: run_lmm auto-selection forwards n_cvt from covariates."""
        from jamma.lmm.runner import run_lmm

        calls = []
        original_sem = select_execution_mode

        def capturing_sem(n_samples, n_snps, **kwargs):
            calls.append(kwargs)
            return original_sem(n_samples, n_snps, **kwargs)

        import numpy as np

        geno = np.zeros((10, 5))
        pheno = np.zeros(10)
        cov = np.zeros((10, 3))  # 3 covariates

        with patch(
            "jamma.lmm.runner.select_execution_mode",
            side_effect=capturing_sem,
        ):
            try:
                run_lmm(genotypes=geno, phenotypes=pheno, covariates=cov)
            except Exception:
                pass  # We only care about the select_execution_mode call

        assert any(c.get("n_cvt") == 3 for c in calls), (
            f"n_cvt=3 not passed to select_execution_mode; calls={calls}"
        )
