"""Direct unit tests for the extracted PipelineRunner helpers.

Covers the three helpers extracted out of ``_run_inner``:

- ``_memory_preflight`` (streaming / batch / batch-with-budget / insufficient)
- ``_load_phenotypes_and_intersect_masks`` (happy, disjoint, shrink-warning,
  unreadable .fam)

``_run_loco`` is exercised transitively by ``tests/test_loco_numpy.py`` and
``tests/test_pipeline.py``; adding a direct test requires the LOCO PLINK
fixture to be wired through ``PipelineRunner.run()`` which lives outside
the scope of these helper unit tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jamma.lmm.runner import ExecutionPlan
from jamma.pipeline import PipelineConfig, PipelineRunner


def _make_runner(tmp_path: Path, **overrides) -> PipelineRunner:  # type: ignore[no-untyped-def]
    """Construct a PipelineRunner with a dummy bfile and any config overrides."""
    bfile = tmp_path / "dummy"
    overrides.setdefault("check_memory", False)
    config = PipelineConfig(bfile=bfile, **overrides)
    return PipelineRunner(config)


@pytest.mark.tier0
class TestMemoryPreflightStreaming:
    """Streaming mode delegates to check_memory_requirements."""

    def test_delegates_with_n_cvt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _make_runner(tmp_path, check_memory=True)
        calls: list[tuple[int, int, int]] = []

        def fake_check(n_valid: int, n_snps: int, *, n_cvt: int = 1) -> None:
            calls.append((n_valid, n_snps, n_cvt))

        monkeypatch.setattr(runner, "check_memory_requirements", fake_check)
        plan = ExecutionPlan(backend="numpy", mode="streaming", reason="test")

        runner._memory_preflight(plan, n_valid=1000, n_snps=50_000, n_cvt=3)

        assert calls == [(1000, 50_000, 3)]


@pytest.mark.tier0
class TestMemoryPreflightBatch:
    """Batch mode paths: check_memory=False short-circuit, budget, sufficiency."""

    def test_check_memory_false_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _make_runner(tmp_path, check_memory=False)
        called = False

        def fake_estimate(*_args: object, **_kw: object) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr("jamma.core.memory.estimate_lmm_memory", fake_estimate)
        plan = ExecutionPlan(backend="numpy", mode="batch", reason="test")

        runner._memory_preflight(plan, n_valid=1000, n_snps=100, n_cvt=1)

        assert not called, "estimator must not run when check_memory=False"

    def test_budget_exceeded_raises_before_sufficiency_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If total_gb > mem_budget, raise even when sufficient=True.

        Ordering is load-bearing: a generous budget on a large machine must
        not mask a user-set cap.
        """
        runner = _make_runner(tmp_path, check_memory=True, mem_budget=8.0)
        est = SimpleNamespace(total_gb=16.0, available_gb=128.0, sufficient=True)
        monkeypatch.setattr(
            "jamma.core.memory.estimate_lmm_memory", lambda *a, **k: est
        )
        plan = ExecutionPlan(backend="numpy", mode="batch", reason="test")

        with pytest.raises(MemoryError, match=r"exceeds .*budget \(8\.0GB\)"):
            runner._memory_preflight(plan, n_valid=1000, n_snps=100, n_cvt=1)

    def test_insufficient_memory_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _make_runner(tmp_path, check_memory=True)
        est = SimpleNamespace(total_gb=200.0, available_gb=64.0, sufficient=False)
        monkeypatch.setattr(
            "jamma.core.memory.estimate_lmm_memory", lambda *a, **k: est
        )
        plan = ExecutionPlan(backend="numpy", mode="batch", reason="test")

        with pytest.raises(MemoryError, match=r"Insufficient memory"):
            runner._memory_preflight(plan, n_valid=1000, n_snps=100, n_cvt=1)

    def test_sufficient_passes_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _make_runner(tmp_path, check_memory=True)
        est = SimpleNamespace(total_gb=32.0, available_gb=128.0, sufficient=True)
        monkeypatch.setattr(
            "jamma.core.memory.estimate_lmm_memory", lambda *a, **k: est
        )
        plan = ExecutionPlan(backend="numpy", mode="batch", reason="test")

        runner._memory_preflight(plan, n_valid=1000, n_snps=100, n_cvt=1)


def _write_fam(path: Path, rows: list[list[str]]) -> None:
    """Write a minimal .fam-style file (space-separated)."""
    path.write_text("\n".join(" ".join(r) for r in rows) + "\n")


@pytest.mark.tier0
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
