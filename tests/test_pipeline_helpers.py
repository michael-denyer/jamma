"""Direct unit tests for the extracted PipelineRunner helpers.

Covers the helpers extracted out of ``_run_inner``:

- ``pipeline_memory.memory_preflight`` (streaming / batch / batch-with-budget /
  insufficient)
- ``_load_phenotypes_and_intersect_masks`` (happy, disjoint, shrink-warning,
  unreadable .fam)
- ``_run_loco`` (delegation contract: LocoResult fields map to
  PipelineResult fields, timing is non-negative, covariates drive n_cvt).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.core.memory import MemoryBreakdown
from jamma.lmm.runner import ExecutionPlan
from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.pipeline_memory import memory_preflight

pytestmark = pytest.mark.tier0


def _make_runner(tmp_path: Path, **overrides) -> PipelineRunner:  # type: ignore[no-untyped-def]
    """Construct a PipelineRunner with a dummy bfile and any config overrides."""
    bfile = tmp_path / "dummy"
    overrides.setdefault("check_memory", False)
    config = PipelineConfig(bfile=bfile, **overrides)
    return PipelineRunner(config)


def _memory_breakdown(
    *, total_gb: float, available_gb: float, sufficient: bool
) -> MemoryBreakdown:
    """Build a real MemoryBreakdown for test doubles.

    Uses real types (not SimpleNamespace / MagicMock) so schema drift in
    MemoryBreakdown surfaces as a test-time TypeError rather than being
    silently accepted. Non-asserted fields are zeroed — the helpers under
    test only read total_gb/available_gb/sufficient.
    """
    return MemoryBreakdown(
        kinship_gb=0.0,
        genotypes_gb=0.0,
        eigenvectors_gb=0.0,
        eigendecomp_workspace_gb=0.0,
        lmm_rotated_gb=0.0,
        lmm_batch_gb=0.0,
        total_gb=total_gb,
        available_gb=available_gb,
        sufficient=sufficient,
    )


class TestMemoryPreflightStreaming:
    """Streaming mode builds and returns one MemoryPlan."""

    def test_returns_streaming_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The returned plan carries the chunk the engine will size and the
        driver-aware eigendecomposition figure."""
        from jamma.core import memory

        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)
        runner = _make_runner(tmp_path, check_memory=True)
        plan = ExecutionPlan(mode="streaming", reason="test")

        mem_plan = memory_preflight(
            runner.config, plan, n_valid=1000, n_snps=50_000, n_cvt=3
        )

        assert mem_plan is not None
        assert mem_plan.mode == "streaming"
        assert mem_plan.disk_chunk_size == 10_000
        assert mem_plan.compute_chunk_size is not None
        assert mem_plan.compute_chunk_size >= 100
        assert mem_plan.eigen is not None
        assert mem_plan.eigen.required_gb > 0
        assert mem_plan.sufficient

    def test_streaming_check_memory_false_logs_skip(self, tmp_path: Path) -> None:
        """Streaming path with check_memory=False must log the skip with the
        runner label so the log stream shows why no preflight ran.
        Paired with the batch counterpart in TestMemoryPreflightBatch.
        """
        from loguru import logger

        runner = _make_runner(tmp_path, check_memory=False)
        plan = ExecutionPlan(mode="streaming", reason="test")

        records: list[str] = []
        handler_id = logger.add(lambda m: records.append(str(m)), level="INFO")
        try:
            result = memory_preflight(
                runner.config, plan, n_valid=1000, n_snps=50_000, n_cvt=3
            )
        finally:
            logger.remove(handler_id)

        assert result is None
        assert any(
            "Memory preflight skipped (numpy-streaming)" in r for r in records
        ), f"streaming skip must log the runner label; got {records!r}"


class TestMemoryPreflightBatch:
    """Batch mode paths: check_memory=False short-circuit, budget, sufficiency."""

    def test_check_memory_false_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When check_memory=False the estimator must not run AND the skip
        must be logged — the log is the only observable signal that the
        preflight was intentionally bypassed (closes the silent-skip
        asymmetry between batch and streaming).
        """
        from loguru import logger

        runner = _make_runner(tmp_path, check_memory=False)
        called = False

        def fake_estimate(*_args: object, **_kw: object) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr("jamma.pipeline_memory.estimate_lmm_memory", fake_estimate)
        plan = ExecutionPlan(mode="batch", reason="test")

        records: list[str] = []
        handler_id = logger.add(lambda m: records.append(str(m)), level="INFO")
        try:
            memory_preflight(runner.config, plan, n_valid=1000, n_snps=100, n_cvt=1)
        finally:
            logger.remove(handler_id)

        assert not called, "estimator must not run when check_memory=False"
        assert any("Memory preflight skipped" in r for r in records), (
            f"batch skip must log intent; got {records!r}"
        )
        # Batch log must include the runner_name, not the literal "streaming".
        assert any("numpy" in r for r in records if "Memory preflight" in r), (
            f"batch skip log must include runner name; got {records!r}"
        )

    def test_budget_exceeded_raises_before_sufficiency_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If total_gb > mem_budget, raise even when sufficient=True.

        Ordering is load-bearing: a generous budget on a large machine must
        not mask a user-set cap.
        """
        runner = _make_runner(tmp_path, check_memory=True, mem_budget=8.0)
        est = _memory_breakdown(total_gb=16.0, available_gb=128.0, sufficient=True)
        monkeypatch.setattr(
            "jamma.pipeline_memory.estimate_lmm_memory", lambda *a, **k: est
        )
        plan = ExecutionPlan(mode="batch", reason="test")

        with pytest.raises(MemoryError, match=r"exceeds .*budget \(8\.0GB\)"):
            memory_preflight(runner.config, plan, n_valid=1000, n_snps=100, n_cvt=1)

    def test_insufficient_memory_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _make_runner(tmp_path, check_memory=True)
        est = _memory_breakdown(total_gb=200.0, available_gb=64.0, sufficient=False)
        monkeypatch.setattr(
            "jamma.pipeline_memory.estimate_lmm_memory", lambda *a, **k: est
        )
        plan = ExecutionPlan(mode="batch", reason="test")

        with pytest.raises(MemoryError, match=r"Insufficient memory"):
            memory_preflight(runner.config, plan, n_valid=1000, n_snps=100, n_cvt=1)

    def test_sufficient_passes_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _make_runner(tmp_path, check_memory=True)
        est = _memory_breakdown(total_gb=32.0, available_gb=128.0, sufficient=True)
        monkeypatch.setattr(
            "jamma.pipeline_memory.estimate_lmm_memory", lambda *a, **k: est
        )
        plan = ExecutionPlan(mode="batch", reason="test")

        memory_preflight(runner.config, plan, n_valid=1000, n_snps=100, n_cvt=1)


def _write_fam(path: Path, rows: list[list[str]]) -> None:
    """Write a minimal .fam-style file (space-separated)."""
    path.write_text("\n".join(" ".join(r) for r in rows) + "\n")


class TestLoadPhenotypesAndIntersectMasks:
    """Multi-phenotype loading + mask intersection."""

    def _runner_with_fam(
        self, tmp_path: Path, pheno_cols: list[list[str]]
    ) -> PipelineRunner:
        """Build a runner whose .fam has FID/IID/PID/MID/SEX plus pheno cols."""
        n_samples = len(pheno_cols[0])
        rows = []
        for i in range(n_samples):
            row = [f"F{i}", f"I{i}", "0", "0", "1"] + [col[i] for col in pheno_cols]
            rows.append(row)
        bfile = tmp_path / "dummy"
        _write_fam(Path(f"{bfile}.fam"), rows)
        config = PipelineConfig(bfile=bfile, check_memory=False)
        return PipelineRunner(config)

    def test_happy_path_multi_column_intersection(self, tmp_path: Path) -> None:
        # 4 samples, 2 phenotypes, all valid.
        runner = self._runner_with_fam(
            tmp_path,
            pheno_cols=[
                ["1.0", "2.0", "3.0", "4.0"],
                ["0.5", "1.5", "2.5", "3.5"],
            ],
        )
        all_pheno, mask, n_valid = runner._load_phenotypes_and_intersect_masks(
            pheno_columns=[1, 2], covariates=None
        )
        assert n_valid == 4
        assert mask.tolist() == [True, True, True, True]
        assert set(all_pheno) == {1, 2}

    def test_intersection_is_elementwise_and(self, tmp_path: Path) -> None:
        # col1: missing at sample 0; col2: missing at sample 1.
        # Intersection keeps only samples 2,3.
        runner = self._runner_with_fam(
            tmp_path,
            pheno_cols=[
                ["NA", "2.0", "3.0", "4.0"],
                ["0.5", "NA", "2.5", "3.5"],
            ],
        )
        all_pheno, mask, n_valid = runner._load_phenotypes_and_intersect_masks(
            pheno_columns=[1, 2], covariates=None
        )
        assert mask.tolist() == [False, False, True, True]
        assert n_valid == 2

    def test_disjoint_masks_raises_with_per_column_counts(self, tmp_path: Path) -> None:
        # col1 valid at samples 0,1; col2 valid at samples 2,3. Intersection empty.
        runner = self._runner_with_fam(
            tmp_path,
            pheno_cols=[
                ["1.0", "2.0", "NA", "NA"],
                ["NA", "NA", "3.0", "4.0"],
            ],
        )
        with pytest.raises(ValueError) as excinfo:
            runner._load_phenotypes_and_intersect_masks(
                pheno_columns=[1, 2], covariates=None
            )
        msg = str(excinfo.value)
        assert "No samples have valid values" in msg
        # Per-column counts must appear so users can diagnose.
        assert "1: 2" in msg
        assert "2: 2" in msg

    def test_shrink_warning_when_intersection_reduces(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the intersection is smaller than every column, emit a warning."""
        from loguru import logger

        records: list[str] = []
        handler_id = logger.add(lambda m: records.append(str(m)), level="WARNING")
        try:
            # col1 valid at 0,1,2 (3); col2 valid at 1,2,3 (3); intersection 1,2 (2).
            runner = self._runner_with_fam(
                tmp_path,
                pheno_cols=[
                    ["1.0", "2.0", "3.0", "NA"],
                    ["NA", "1.5", "2.5", "3.5"],
                ],
            )
            _, mask, n_valid = runner._load_phenotypes_and_intersect_masks(
                pheno_columns=[1, 2], covariates=None
            )
        finally:
            logger.remove(handler_id)

        assert n_valid == 2
        assert mask.tolist() == [False, True, True, False]
        assert any("intersection" in r for r in records), (
            f"expected intersection shrink warning, got: {records}"
        )

    def test_missing_fam_raises_with_path(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        # No .fam file exists at tmp_path/dummy.fam
        with pytest.raises(ValueError, match=r"Failed to read \.fam file .*dummy\.fam"):
            runner._load_phenotypes_and_intersect_masks(
                pheno_columns=[1], covariates=None
            )

    def test_covariates_narrow_the_mask(self, tmp_path: Path) -> None:
        """A NaN covariate row must be excluded from the valid mask."""
        runner = self._runner_with_fam(
            tmp_path,
            pheno_cols=[["1.0", "2.0", "3.0", "4.0"]],
        )
        # Covariate NaN at sample 2.
        covariates = np.array([[1.0], [1.0], [np.nan], [1.0]], dtype=np.float64)
        _, mask, n_valid = runner._load_phenotypes_and_intersect_masks(
            pheno_columns=[1], covariates=covariates
        )
        assert mask.tolist() == [True, True, False, True]
        assert n_valid == 3


class TestRunLoco:
    """Direct tests for the extracted _run_loco helper.

    Replaces transitive coverage via test_loco_numpy.py / test_pipeline.py.
    run() hands _run_loco the loaded phenotype, covariates and valid mask, so
    the tests pass those directly, stub run_lmm_loco, and assert on the
    observable PipelineResult returned.
    """

    def _build_loco_runner(
        self,
        tmp_path: Path,
        *,
        loco_result,  # type: ignore[no-untyped-def]
        monkeypatch: pytest.MonkeyPatch,
    ) -> PipelineRunner:
        """Construct a runner and stub out the LOCO orchestrator."""
        runner = _make_runner(tmp_path)
        monkeypatch.setattr(runner, "_emit_telemetry", lambda *a, **k: None)

        from jamma import lmm as lmm_pkg

        monkeypatch.setattr(lmm_pkg, "run_lmm_loco", lambda **_kw: loco_result)
        return runner

    @staticmethod
    def _call(
        runner: PipelineRunner,
        tmp_path: Path,
        phenotypes: np.ndarray,
        covariates: np.ndarray | None,
    ):  # type: ignore[no-untyped-def]
        """Invoke _run_loco the way run() does, with the mask run() would build."""
        from jamma.lmm.prepare_common import compute_valid_mask

        return runner._run_loco(
            t_start=0.0,
            plan=ExecutionPlan(mode="batch", reason="loco"),
            phenotypes=phenotypes,
            covariates=covariates,
            valid_mask=compute_valid_mask(phenotypes, covariates),
            n_snps=0,
            assoc_path=tmp_path / "out.assoc.txt",
            snps_indices=None,
            ksnps_indices=None,
        )

    def test_loco_result_fields_map_to_pipeline_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """n_tested, associations, pve, pve_se from LocoResult reach PipelineResult."""
        from jamma.lmm.schema import LocoResult

        # 4 samples, one NaN — valid mask has 3 True.
        phenos = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
        covs = np.array(
            [[1.0], [1.0], [1.0], [1.0]], dtype=np.float64
        )  # intercept only
        loco = LocoResult(
            associations=["snp1", "snp2", "snp3"],  # sentinel strings for ordering
            n_tested=3,
            pve=0.42,
            pve_se=0.05,
        )
        runner = self._build_loco_runner(
            tmp_path, loco_result=loco, monkeypatch=monkeypatch
        )

        result = self._call(runner, tmp_path, phenos, covs)

        assert result.n_snps_tested == 3
        assert result.associations == ["snp1", "snp2", "snp3"]
        assert result.pve_estimate == 0.42
        assert result.pve_se == 0.05
        assert result.assoc_path == tmp_path / "out.assoc.txt"
        assert result.assoc_paths == [tmp_path / "out.assoc.txt"]
        # 3 valid samples after NaN filtering (observable n_valid).
        assert result.n_samples == 3

    def test_n_covariates_reflects_loaded_covariates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """n_covariates in the result matches covariates.shape[1].

        Regression guard: the extracted helper must not hard-code n_cvt=1
        when multi-covariate LOCO runs arrive here.
        """
        from jamma.lmm.schema import LocoResult

        phenos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        # 3 covariate columns.
        covs = np.ones((3, 3), dtype=np.float64)
        loco = LocoResult(associations=[], n_tested=0, pve=None, pve_se=None)
        runner = self._build_loco_runner(
            tmp_path, loco_result=loco, monkeypatch=monkeypatch
        )

        result = self._call(runner, tmp_path, phenos, covs)

        assert result.n_covariates == 3

    def test_n_covariates_defaults_to_one_without_covariates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No covariates -> n_covariates=1 (intercept only)."""
        from jamma.lmm.schema import LocoResult

        phenos = np.array([1.0, 2.0], dtype=np.float64)
        loco = LocoResult(associations=[], n_tested=0, pve=None, pve_se=None)
        runner = self._build_loco_runner(
            tmp_path, loco_result=loco, monkeypatch=monkeypatch
        )

        result = self._call(runner, tmp_path, phenos, None)

        assert result.n_covariates == 1

    def test_timing_has_lmm_and_total_nonnegative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Timing dict is populated with lmm_s and total_s; kinship_s/load_s
        are zero (LOCO owns its own kinship/load; the pipeline does not).
        """
        from jamma.lmm.schema import LocoResult

        phenos = np.array([1.0, 2.0], dtype=np.float64)
        covs = np.ones((2, 1), dtype=np.float64)
        loco = LocoResult(associations=[], n_tested=0)
        runner = self._build_loco_runner(
            tmp_path, loco_result=loco, monkeypatch=monkeypatch
        )

        result = self._call(runner, tmp_path, phenos, covs)

        assert result.timing.kinship_s == 0.0
        assert result.timing.load_s == 0.0
        assert result.timing.lmm_s >= 0.0
        assert result.timing.total_s >= 0.0

    def test_lmm_config_handed_to_runner_is_the_shared_projection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_run_loco must build its LmmConfig via PipelineConfig.lmm_config().

        Regression guard for config drift. _run_loco once wrote the nine
        LmmConfig fields out by hand, because LOCO needs check_memory passed
        through where the batch and streaming paths force it off. That left two
        copies of one projection, so a field added to LmmConfig could reach
        _run_batch and miss LOCO.

        Asserted by dataclass equality rather than field by field: a tenth
        LmmConfig field that a re-inlined literal forgot to set would take its
        default and break equality here.
        """
        from jamma.lmm.schema import LocoResult

        captured: dict[str, object] = {}
        phenos = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        def _capturing_loco(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return LocoResult(associations=[], n_tested=0)

        # Every knob off its default, so a projection that dropped one shows up.
        runner = _make_runner(
            tmp_path,
            check_memory=True,
            maf=0.02,
            miss=0.1,
            lmm_mode=4,
            show_progress=False,
            l_min=1e-4,
            l_max=1e4,
            n_grid=17,
            n_refine=23,
            loco=True,
        )
        monkeypatch.setattr(runner, "_emit_telemetry", lambda *a, **k: None)

        from jamma import lmm as lmm_pkg

        monkeypatch.setattr(lmm_pkg, "run_lmm_loco", _capturing_loco)

        self._call(runner, tmp_path, phenos, None)

        assert captured["config"] == runner.config.lmm_config(check_memory=True)
        # The distinguishing field: LOCO returns before memory_preflight, so
        # unlike _run_batch it must not force the runner's memory gate off.
        assert captured["config"].check_memory is True  # type: ignore[union-attr]

    def test_propagates_loco_runner_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If run_lmm_loco raises, _run_loco must propagate — no swallowing."""
        phenos = np.array([1.0, 2.0], dtype=np.float64)
        runner = _make_runner(tmp_path)
        monkeypatch.setattr(runner, "_emit_telemetry", lambda *a, **k: None)

        from jamma import lmm as lmm_pkg

        def _raising_loco(**_kw):
            raise RuntimeError("sentinel: LOCO failed")

        monkeypatch.setattr(lmm_pkg, "run_lmm_loco", _raising_loco)

        with pytest.raises(RuntimeError, match="sentinel: LOCO failed"):
            self._call(runner, tmp_path, phenos, None)
