"""Tests for run_lmm() dispatcher routing to correct runner.

These tests mock all underlying runners and do not require JAX.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jamma.lmm.runner import ExecutionPlan, run_lmm
from jamma.lmm.schema import LmmRunResult
from jamma.lmm.stats import AssocResult


@pytest.mark.tier0
class TestUnifiedDispatcher:
    """Tests for run_lmm() dispatcher routing to correct runner."""

    def _stub_assoc(self) -> AssocResult:
        """Build a minimal stub AssocResult."""
        return AssocResult(
            chr="1",
            rs="rs1",
            ps=100,
            n_miss=0,
            allele1="A",
            allele0="T",
            af=0.3,
            beta=0.1,
            se=0.05,
            p_wald=0.01,
        )

    def _stub_run_result(self, n_assocs: int = 5) -> LmmRunResult:
        """Build a stub LmmRunResult with n_assocs associations."""
        return LmmRunResult(
            associations=[self._stub_assoc() for _ in range(n_assocs)],
            pve=0.5,
        )

    def test_numpy_batch_calls_numpy_runner(self):
        """numpy-batch plan calls run_lmm_association_numpy."""
        plan = ExecutionPlan(backend="numpy", mode="batch", reason="test")
        stub_result = self._stub_run_result(5)

        with patch(
            "jamma.lmm.runner_numpy.run_lmm_association_numpy",
            return_value=stub_result,
        ) as mock_numpy:
            result, n_tested = run_lmm(
                execution_plan=plan,
                genotypes=np.zeros((10, 5)),
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
                snp_info=[{}] * 5,
            )

        mock_numpy.assert_called_once()
        assert isinstance(result, LmmRunResult)
        assert n_tested == 5

    def test_jax_batch_calls_jax_runner(self):
        """jax-batch plan calls run_lmm_association_jax."""
        plan = ExecutionPlan(backend="jax", mode="batch", reason="test")
        stub_result = self._stub_run_result(5)

        with patch(
            "jamma.lmm.runner_jax.run_lmm_association_jax",
            return_value=stub_result,
        ) as mock_jax:
            result, n_tested = run_lmm(
                execution_plan=plan,
                genotypes=np.zeros((10, 5)),
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
                snp_info=[{}] * 5,
            )

        mock_jax.assert_called_once()
        assert isinstance(result, LmmRunResult)
        assert n_tested == 5

    def test_jax_streaming_calls_streaming_runner(self):
        """jax-streaming plan calls run_lmm_association_streaming."""
        plan = ExecutionPlan(backend="jax", mode="streaming", reason="test")
        stub_result = self._stub_run_result(3)

        with patch(
            "jamma.lmm.runner_jax_streaming.run_lmm_association_streaming",
            return_value=(stub_result, 3),
        ) as mock_stream:
            result, n_tested = run_lmm(
                execution_plan=plan,
                bed_path=Path("/tmp/test"),
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
            )

        mock_stream.assert_called_once()
        assert isinstance(result, LmmRunResult)
        assert n_tested == 3

    def test_no_plan_auto_selects(self):
        """No execution_plan -> auto-selects via select_execution_mode."""
        stub_result = self._stub_run_result(5)
        plan = ExecutionPlan(backend="numpy", mode="batch", reason="auto")

        with (
            patch(
                "jamma.lmm.runner.select_execution_mode",
                return_value=plan,
            ) as mock_select,
            patch(
                "jamma.lmm.runner_numpy.run_lmm_association_numpy",
                return_value=stub_result,
            ),
        ):
            result, n_tested = run_lmm(
                genotypes=np.zeros((10, 5)),
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
                snp_info=[{}] * 5,
            )

        mock_select.assert_called_once()
        assert isinstance(result, LmmRunResult)

    def test_numpy_batch_but_no_genotypes_raises(self):
        """numpy-batch but genotypes=None -> ValueError."""
        plan = ExecutionPlan(backend="numpy", mode="batch", reason="test")

        with pytest.raises(ValueError, match="genotypes"):
            run_lmm(
                execution_plan=plan,
                genotypes=None,
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
                snp_info=[{}] * 5,
            )

    def test_jax_batch_but_no_genotypes_raises(self):
        """jax-batch but genotypes=None -> ValueError."""
        plan = ExecutionPlan(backend="jax", mode="batch", reason="test")

        with pytest.raises(ValueError, match="genotypes"):
            run_lmm(
                execution_plan=plan,
                genotypes=None,
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
                snp_info=[{}] * 5,
            )

    def test_jax_streaming_but_no_bed_path_raises(self):
        """jax-streaming but bed_path=None -> ValueError."""
        plan = ExecutionPlan(backend="jax", mode="streaming", reason="test")

        with pytest.raises(ValueError, match="bed_path"):
            run_lmm(
                execution_plan=plan,
                bed_path=None,
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
            )

    def test_auto_select_bed_path_no_genotypes_raises(self):
        """Auto-select with bed_path but no genotypes gives clear error."""
        with pytest.raises(ValueError, match="ambiguous"):
            run_lmm(
                bed_path=Path("/tmp/test"),
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
            )

    def test_numpy_streaming_calls_numpy_streaming_runner(self):
        """numpy-streaming plan calls run_lmm_association_numpy_streaming."""
        plan = ExecutionPlan(backend="numpy", mode="streaming", reason="test")
        stub_result = self._stub_run_result(3)

        with patch(
            "jamma.lmm.runner_numpy_streaming.run_lmm_association_numpy_streaming",
            return_value=(stub_result, 3),
        ) as mock_np_stream:
            result, n_tested = run_lmm(
                execution_plan=plan,
                bed_path=Path("/tmp/test"),
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
            )

        mock_np_stream.assert_called_once()
        assert isinstance(result, LmmRunResult)
        assert n_tested == 3

    def test_numpy_streaming_no_bed_path_raises(self):
        """numpy-streaming but bed_path=None -> ValueError."""
        plan = ExecutionPlan(backend="numpy", mode="streaming", reason="test")

        with pytest.raises(ValueError, match="bed_path"):
            run_lmm(
                execution_plan=plan,
                bed_path=None,
                phenotypes=np.zeros(10),
                kinship=np.eye(10),
            )

    def test_auto_select_no_data_raises(self):
        """Auto-select with no genotypes and no phenotypes gives clear error."""
        with pytest.raises(ValueError, match="at least genotypes or phenotypes"):
            run_lmm(
                genotypes=None,
                phenotypes=None,
                kinship=np.eye(10),
            )
