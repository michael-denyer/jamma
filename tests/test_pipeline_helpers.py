"""Direct unit tests for the extracted PipelineRunner helpers.

Covers the helpers extracted out of ``_run_inner``:

- ``pipeline_memory.memory_preflight`` (streaming / batch / batch-with-budget /
  insufficient)
- ``pipeline_samples.load_analysed_samples`` (happy, disjoint, shrink-warning,
  unreadable .fam, covariate row count, appended intercept)
- ``_associate_loco`` (delegation contract: LmmRunResult fields map to
  PipelineResult fields, only lmm_s is timed, config and errors pass through).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest

import jamma.pipeline as pipeline_mod
from jamma.core import memory
from jamma.lmm.association_plan import plan_association
from jamma.lmm.genotype_source import SampleBasis
from jamma.lmm.prepare_common import compute_valid_mask, with_intercept
from jamma.lmm.schema import LmmConfig, LmmRunResult
from jamma.pipeline import PipelineConfig, PipelineResult, PipelineRunner
from jamma.pipeline_plan import LocoAnalysisPlan, resolve_analysis_plan
from jamma.pipeline_samples import AnalysedSamples, load_analysed_samples
from tests.conftest import preflight

if TYPE_CHECKING:
    from jamma.io.plink import PlinkMetadata
    from jamma.lmm.assoc_output import AssocResult

pytestmark = pytest.mark.tier0


def _association_plan(
    mode: Literal["batch", "streaming"],
    *,
    n_valid: int,
    n_snps: int,
    n_cvt: int,
    mem_budget: float | None = None,
):  # type: ignore[no-untyped-def]
    return plan_association(
        n_valid,
        n_snps,
        config=LmmConfig(mem_budget=mem_budget),
        backend="numpy-streaming" if mode == "streaming" else "numpy",
        n_cvt=n_cvt,
    )


def _make_runner(tmp_path: Path, **overrides) -> PipelineRunner:  # type: ignore[no-untyped-def]
    """Construct a PipelineRunner with a dummy bfile and any config overrides."""
    bfile = tmp_path / "dummy"
    overrides.setdefault("check_memory", False)
    config = PipelineConfig(bfile=bfile, **overrides)
    return PipelineRunner(config)


class TestMemoryPreflightStreaming:
    """Streaming mode prices the plan's own geometry and passes the gate."""

    def test_prices_streaming_geometry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The quote carries the chunk the engine will size and the
        driver-aware eigendecomposition figure, and the gate passes."""
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)
        runner = _make_runner(tmp_path, check_memory=True)
        plan = _association_plan("streaming", n_valid=1000, n_snps=50_000, n_cvt=3)

        eigen = preflight(runner.config, plan)  # gate must pass, not raise

        assert eigen is not None
        assert eigen.required_gb > 0
        quote = plan.price(eigen=eigen)
        assert quote.compute_chunk_size >= 100
        assert memory.fits(quote.total_peak_gb, 64.0)

    def test_streaming_check_memory_false_logs_skip(self, tmp_path: Path) -> None:
        """Streaming path with check_memory=False must log the skip with the
        runner label so the log stream shows why no preflight ran.
        Paired with the batch counterpart in TestMemoryPreflightBatch.
        """
        from loguru import logger

        runner = _make_runner(tmp_path, check_memory=False)
        plan = _association_plan("streaming", n_valid=1000, n_snps=50_000, n_cvt=3)

        records: list[str] = []
        handler_id = logger.add(lambda m: records.append(str(m)), level="INFO")
        try:
            result = preflight(runner.config, plan)
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
        """When check_memory=False preflight must not reprice AND the skip
        must be logged — the log is the only observable signal that the
        preflight was intentionally bypassed (closes the silent-skip
        asymmetry between batch and streaming).
        """
        from loguru import logger

        runner = _make_runner(tmp_path, check_memory=False)
        plan = _association_plan("batch", n_valid=1000, n_snps=100, n_cvt=1)
        called = False

        def fake_estimate(*_args: object, **_kw: object) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(
            "jamma.lmm.association_plan.estimate_lmm_memory", fake_estimate
        )
        records: list[str] = []
        handler_id = logger.add(lambda m: records.append(str(m)), level="INFO")
        try:
            preflight(runner.config, plan)
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
        """A peak above mem_budget raises even when the machine has room.

        Ordering is load-bearing: a generous budget on a large machine must
        not mask a user-set cap.
        """
        runner = _make_runner(tmp_path, check_memory=True, mem_budget=8.0)
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 128.0)
        monkeypatch.setattr(
            "jamma.lmm.association_plan.estimate_lmm_memory", lambda *a, **k: 16.0
        )
        plan = _association_plan(
            "batch", n_valid=1000, n_snps=100, n_cvt=1, mem_budget=8.0
        )

        with pytest.raises(MemoryError, match=r"exceeds .*budget \(8\.0GB\)"):
            preflight(runner.config, plan)

    def test_insufficient_memory_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _make_runner(tmp_path, check_memory=True)
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)
        monkeypatch.setattr(
            "jamma.lmm.association_plan.estimate_lmm_memory", lambda *a, **k: 200.0
        )
        plan = _association_plan("batch", n_valid=1000, n_snps=100, n_cvt=1)

        with pytest.raises(MemoryError, match=r"Insufficient memory"):
            preflight(runner.config, plan)

    def test_sufficient_passes_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _make_runner(tmp_path, check_memory=True)
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 128.0)
        monkeypatch.setattr(
            "jamma.lmm.association_plan.estimate_lmm_memory", lambda *a, **k: 32.0
        )
        plan = _association_plan("batch", n_valid=1000, n_snps=100, n_cvt=1)

        preflight(runner.config, plan)


def _write_fam(path: Path, rows: list[list[str]]) -> None:
    """Write a minimal .fam-style file (space-separated)."""
    path.write_text("\n".join(" ".join(r) for r in rows) + "\n")


class TestLoadAnalysedSamples:
    """Multi-phenotype loading, covariate loading, and mask intersection."""

    def _config_with_fam(
        self,
        tmp_path: Path,
        pheno_cols: list[list[str]],
        *,
        phenotype_columns: list[int] | None = None,
        covariate_rows: list[str] | None = None,
    ) -> PipelineConfig:
        """Build a config whose .fam has FID/IID/PID/MID/SEX plus pheno cols."""
        n_samples = len(pheno_cols[0])
        rows = []
        for i in range(n_samples):
            row = [f"F{i}", f"I{i}", "0", "0", "1"] + [col[i] for col in pheno_cols]
            rows.append(row)
        bfile = tmp_path / "dummy"
        _write_fam(Path(f"{bfile}.fam"), rows)
        covariate_file = None
        if covariate_rows is not None:
            covariate_file = tmp_path / "cov.txt"
            covariate_file.write_text("\n".join(covariate_rows) + "\n")
        return PipelineConfig(
            bfile=bfile,
            check_memory=False,
            phenotype_columns=phenotype_columns or [1],
            covariate_file=covariate_file,
        )

    def test_happy_path_multi_column_intersection(self, tmp_path: Path) -> None:
        # 4 samples, 2 phenotypes, all valid.
        config = self._config_with_fam(
            tmp_path,
            pheno_cols=[
                ["1.0", "2.0", "3.0", "4.0"],
                ["0.5", "1.5", "2.5", "3.5"],
            ],
            phenotype_columns=[1, 2],
        )
        samples = load_analysed_samples(config, n_samples=4)
        assert samples.basis.analyzed_sample_count == 4
        assert samples.valid_mask.tolist() == [True, True, True, True]
        assert set(samples.phenotypes) == {1, 2}
        assert samples.filter_indices is None

    def test_intersection_is_elementwise_and(self, tmp_path: Path) -> None:
        # col1: missing at sample 0; col2: missing at sample 1.
        # Intersection keeps only samples 2,3.
        config = self._config_with_fam(
            tmp_path,
            pheno_cols=[
                ["NA", "2.0", "3.0", "4.0"],
                ["0.5", "NA", "2.5", "3.5"],
            ],
            phenotype_columns=[1, 2],
        )
        samples = load_analysed_samples(config, n_samples=4)
        assert samples.valid_mask.tolist() == [False, False, True, True]
        assert samples.basis.analyzed_sample_count == 2
        assert samples.filter_indices is not None
        assert samples.filter_indices.tolist() == [2, 3]

    def test_disjoint_masks_raises_with_per_column_counts(self, tmp_path: Path) -> None:
        # col1 valid at samples 0,1; col2 valid at samples 2,3. Intersection empty.
        config = self._config_with_fam(
            tmp_path,
            pheno_cols=[
                ["1.0", "2.0", "NA", "NA"],
                ["NA", "NA", "3.0", "4.0"],
            ],
            phenotype_columns=[1, 2],
        )
        with pytest.raises(ValueError) as excinfo:
            load_analysed_samples(config, n_samples=4)
        msg = str(excinfo.value)
        assert "No samples have valid values" in msg
        # Per-column counts must appear so users can diagnose.
        assert "1: 2" in msg
        assert "2: 2" in msg

    def test_shrink_warning_when_intersection_reduces(self, tmp_path: Path) -> None:
        """When the intersection is smaller than every column, emit a warning."""
        from loguru import logger

        records: list[str] = []
        handler_id = logger.add(lambda m: records.append(str(m)), level="WARNING")
        try:
            # col1 valid at 0,1,2 (3); col2 valid at 1,2,3 (3); intersection 1,2 (2).
            config = self._config_with_fam(
                tmp_path,
                pheno_cols=[
                    ["1.0", "2.0", "3.0", "NA"],
                    ["NA", "1.5", "2.5", "3.5"],
                ],
                phenotype_columns=[1, 2],
            )
            samples = load_analysed_samples(config, n_samples=4)
        finally:
            logger.remove(handler_id)

        assert samples.basis.analyzed_sample_count == 2
        assert samples.valid_mask.tolist() == [False, True, True, False]
        assert any("intersection" in r for r in records), (
            f"expected intersection shrink warning, got: {records}"
        )

    def test_missing_fam_raises_with_path(self, tmp_path: Path) -> None:
        config = PipelineConfig(bfile=tmp_path / "dummy", check_memory=False)
        # No .fam file exists at tmp_path/dummy.fam
        with pytest.raises(ValueError, match=r"Failed to read \.fam file .*dummy\.fam"):
            load_analysed_samples(config, n_samples=4)

    def test_covariates_narrow_the_mask(self, tmp_path: Path) -> None:
        """A NaN covariate row must be excluded from the valid mask."""
        config = self._config_with_fam(
            tmp_path,
            pheno_cols=[["1.0", "2.0", "3.0", "4.0"]],
            # Covariate NaN at sample 2, and a constant column so no intercept
            # is appended.
            covariate_rows=["1 0.5", "1 1.5", "1 NA", "1 3.5"],
        )
        samples = load_analysed_samples(config, n_samples=4)
        assert samples.valid_mask.tolist() == [True, True, False, True]
        assert samples.basis.analyzed_sample_count == 3
        assert samples.n_covariates == 2

    def test_covariate_row_count_mismatch_raises(self, tmp_path: Path) -> None:
        """A covariate file that does not span every sample is named as such."""
        config = self._config_with_fam(
            tmp_path,
            pheno_cols=[["1.0", "2.0", "3.0", "4.0"]],
            covariate_rows=["1 0.5", "1 1.5", "1 2.5"],
        )
        with pytest.raises(ValueError, match="3 rows but PLINK data has 4 samples"):
            load_analysed_samples(config, n_samples=4)

    def test_intercept_appended_when_no_column_is_constant(
        self, tmp_path: Path
    ) -> None:
        """with_intercept runs after masking, so n_covariates counts the ones column."""
        config = self._config_with_fam(
            tmp_path,
            pheno_cols=[["1.0", "2.0", "3.0", "4.0"]],
            covariate_rows=["0 1", "1 3", "2 5", "3 7"],
        )
        samples = load_analysed_samples(config, n_samples=4)
        assert samples.n_covariates == 3
        assert samples.covariates is not None
        assert np.array_equal(samples.covariates[:, 2], np.ones(4))


class TestAssociateLoco:
    """Direct tests for the LOCO branch of ``PipelineRunner.run``.

    run() hands ``_associate_loco`` its ``AnalysedSamples``, so the tests
    build those directly, stub the LOCO body, and assert on the
    ``PipelineResult`` the branch's records derive.
    """

    def _build_loco_runner(
        self,
        tmp_path: Path,
        *,
        loco_result,  # type: ignore[no-untyped-def]
        monkeypatch: pytest.MonkeyPatch,
    ) -> PipelineRunner:
        """Construct a runner and stub out the LOCO orchestrator."""
        runner = _make_runner(tmp_path, loco=True)
        monkeypatch.setattr(pipeline_mod, "run_loco", lambda *_a, **_kw: loco_result)
        return runner

    @staticmethod
    def _call(
        runner: PipelineRunner,
        tmp_path: Path,
        phenotypes: np.ndarray,
        covariates: np.ndarray | None,
    ) -> PipelineResult:
        """Run the LOCO branch on the samples run() would build."""
        analysis = resolve_analysis_plan(
            runner.config,
            execution=plan_association(4, 1, backend="numpy"),
            snps_indices=None,
            ksnps_indices=None,
        )
        assert isinstance(analysis, LocoAnalysisPlan)
        valid_mask = compute_valid_mask(phenotypes, covariates)
        samples = AnalysedSamples(
            phenotypes={runner.config.phenotype_columns[0]: phenotypes},
            covariates=with_intercept(covariates, valid_mask),
            valid_mask=valid_mask,
            basis=SampleBasis.from_mask(valid_mask),
        )
        records, timing = runner._associate_loco(
            analysis,
            samples,
            cast("PlinkMetadata", None),
            tmp_path / "out.assoc.txt",
            None,
        )
        return PipelineResult(
            phenotype_results=records,
            n_samples=samples.basis.analyzed_sample_count,
            timing=timing,
        )

    def test_loco_result_fields_map_to_pipeline_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """n_tested, associations, pve, pve_se from the LOCO run reach the result."""
        phenos = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
        covs = np.ones((4, 1), dtype=np.float64)
        loco = LmmRunResult(
            associations=cast("list[AssocResult]", ["snp1", "snp2", "snp3"]),
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
        assert result.n_samples == 3

    def test_timing_has_lmm_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LOCO owns its kinship and load, so only lmm_s is timed here."""
        phenos = np.array([1.0, 2.0], dtype=np.float64)
        loco = LmmRunResult(associations=[], n_tested=0)
        runner = self._build_loco_runner(
            tmp_path, loco_result=loco, monkeypatch=monkeypatch
        )

        result = self._call(runner, tmp_path, phenos, None)

        assert result.timing.kinship_s == 0.0
        assert result.timing.load_s == 0.0
        assert result.timing.lmm_s >= 0.0

    def test_lmm_config_handed_to_runner_is_the_shared_projection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The LOCO branch must build its LmmConfig via PipelineConfig.lmm_config().

        Regression guard for config drift. The branch once wrote the nine
        LmmConfig fields out by hand, because LOCO needs check_memory passed
        through where the batch and streaming paths force it off.

        Asserted by dataclass equality rather than field by field: a tenth
        LmmConfig field that a re-inlined literal forgot to set would take its
        default and break equality here.
        """
        captured: dict[str, object] = {}
        phenos = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        def _capturing_loco(run, _output_path):  # type: ignore[no-untyped-def]
            captured["config"] = run.config
            return LmmRunResult(associations=[], n_tested=0)

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
        monkeypatch.setattr(pipeline_mod, "run_loco", _capturing_loco)

        self._call(runner, tmp_path, phenos, None)

        assert captured["config"] == runner.config.lmm_config(check_memory=True)
        assert captured["config"].check_memory is True  # type: ignore[union-attr]

    def test_propagates_loco_runner_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the LOCO body raises, the branch must propagate it."""
        phenos = np.array([1.0, 2.0], dtype=np.float64)
        runner = _make_runner(tmp_path, loco=True)

        def _raising_loco(*_args, **_kw):
            raise RuntimeError("sentinel: LOCO failed")

        monkeypatch.setattr(pipeline_mod, "run_loco", _raising_loco)

        with pytest.raises(RuntimeError, match="sentinel: LOCO failed"):
            self._call(runner, tmp_path, phenos, None)
