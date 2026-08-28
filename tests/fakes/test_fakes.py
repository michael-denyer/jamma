"""Self-tests for the fakes package.

If a fake breaks, every test that uses it fails with a confusing message.
These tests fail with a clear message so the regression is obvious.

The ``TestFakeProductionDrift`` class compares each fake method's
signature against the real production method it shadows. This catches
the failure mode the rest of the package can't: when production *adds* a
parameter to a method the fake also implements, calling the fake with
the new arg works only because the fake doesn't declare it — tests that
exercise the new parameter would silently call into a stale stub. The
signature check fails loudly in that case.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import psutil
import pytest

import jamma.jlinalg
from jamma.lmm import eigen
from jamma.lmm.io import IncrementalAssocWriter
from jamma.pipeline import PipelineConfig, PipelineResult, PipelineRunner
from tests.fakes import (
    FakeAssocWriter,
    FakeJlinalg,
    FakeMemoryInfo,
    FakePipelineRunner,
    FakePipelineRunnerFactory,
    FakeProgressBar,
    FakeProgressbarModule,
    FakeVirtualMemory,
)

pytestmark = pytest.mark.tier0


def _make_config(phenotype_columns: list[int]) -> PipelineConfig:
    return PipelineConfig(
        bfile=Path("/tmp/jamma_fake_bfile"),
        phenotype_columns=phenotype_columns,
    )


def _make_result() -> PipelineResult:
    return PipelineResult(
        associations=[],
        n_samples=0,
        n_snps_tested=0,
        assoc_path=Path("/tmp/jamma_fake.assoc.txt"),
        assoc_paths=[Path("/tmp/jamma_fake.assoc.txt")],
        timing={"total_s": 0.0},
        n_covariates=1,
    )


class TestFakeAssocWriter:
    def test_records_batches(self) -> None:
        writer = FakeAssocWriter()
        snp_indices = np.array([0, 1, 2])
        snp_info = [{"chr": "1", "rs": "rs1", "pos": 1, "a1": "A", "a0": "G"}]
        afs = np.array([0.3])
        miss = np.array([0])
        arrays = {"betas": np.array([0.5])}

        writer.write_arrays_batch(1, snp_indices, snp_info, afs, miss, arrays)
        writer.write_arrays_batch(2, snp_indices, snp_info, afs, miss, arrays)

        assert len(writer.batches) == 2
        assert writer.batches[0][0] == 1
        assert writer.batches[1][0] == 2

    def test_unknown_attribute_raises(self) -> None:
        """Unlike MagicMock, accessing an unknown attribute fails loudly."""
        writer = FakeAssocWriter()
        with pytest.raises(AttributeError):
            writer.flush_to_disk_async()  # type: ignore[attr-defined]


class TestFakeProgressBar:
    def test_records_lifecycle(self) -> None:
        bar = FakeProgressBar(max_value=10)

        bar.start()
        bar.update(1)
        bar.update(5)
        bar.finish()

        assert bar.kwargs == {"max_value": 10}
        assert bar.started
        assert bar.update_calls == [1, 5]
        assert bar.finished

    def test_double_start_raises(self) -> None:
        bar = FakeProgressBar()
        bar.start()
        with pytest.raises(AssertionError, match=r"start.*twice"):
            bar.start()

    def test_double_finish_raises(self) -> None:
        bar = FakeProgressBar()
        bar.finish()
        with pytest.raises(AssertionError, match=r"finish.*twice"):
            bar.finish()

    def test_unknown_attribute_raises(self) -> None:
        bar = FakeProgressBar()
        with pytest.raises(AttributeError):
            bar.set_color("red")  # type: ignore[attr-defined]


class TestFakeProgressbarModule:
    def test_progressbar_factory_records_last_bar(self) -> None:
        module = FakeProgressbarModule()
        assert module.last_bar is None

        bar1 = module.ProgressBar(max_value=5)
        assert module.last_bar is bar1

        bar2 = module.ProgressBar(max_value=10)
        assert module.last_bar is bar2

    def test_widget_classes_return_sentinels(self) -> None:
        """Widget classes are passed positionally and never inspected."""
        module = FakeProgressbarModule()
        for name in ("Counter", "Percentage", "Bar", "Timer", "AdaptiveETA"):
            widget = getattr(module, name)()
            assert widget.name == name


class TestFakePipelineRunnerFactory:
    def test_captures_config_and_returns_result(self) -> None:
        result = _make_result()
        factory = FakePipelineRunnerFactory(result=result)

        config = _make_config([1, 2, 3])
        runner = factory(config)

        assert runner.config is config
        assert runner.run() is result
        assert runner.ran_at_least_once

    def test_last_config_tracks_most_recent_call(self) -> None:
        factory = FakePipelineRunnerFactory(result=_make_result())
        factory(_make_config([1]))
        last = _make_config([2, 3])
        factory(last)

        assert len(factory.runners) == 2
        assert factory.last_config is last
        assert factory.last_config.phenotype_columns == [2, 3]

    def test_last_config_before_any_call_raises(self) -> None:
        factory = FakePipelineRunnerFactory(result=_make_result())
        with pytest.raises(AssertionError, match="never constructed"):
            factory.last_config  # noqa: B018 — property access raises

    def test_runner_unknown_attribute_raises(self) -> None:
        runner = FakePipelineRunner(config=_make_config([1]), result=_make_result())
        with pytest.raises(AttributeError):
            runner.cancel()  # type: ignore[attr-defined]

    def test_dict_config_rejected_by_type_checker(self) -> None:
        """Real PipelineConfig only — dicts would silently accept stale fields.

        This test documents the contract; dict construction now causes a
        type error at the call site (Pyrefly / mypy will flag it).
        """
        factory = FakePipelineRunnerFactory(result=_make_result())
        config = _make_config([1])
        factory(config)
        # Adding a required field to PipelineConfig must break this line:
        assert isinstance(factory.last_config, PipelineConfig)


class TestFakeProductionDrift:
    """Compare fake method signatures to the real production methods.

    These tests catch the failure mode an ``AttributeError`` check can't:
    a method's *parameter list* diverging between fake and real. If
    production adds a parameter, the fake's stale signature would silently
    accept old call sites and miss the new arg entirely.

    Allowed delta: a fake's ``__init__`` may take *more* parameters than
    the real one (the test seam, e.g. ``result=`` for the fake runner).
    Methods called by production code must match exactly.
    """

    @staticmethod
    def _param_names(callable_obj: Callable[..., object]) -> list[str]:
        return [
            p.name
            for p in inspect.signature(callable_obj).parameters.values()
            if p.name != "self"
        ]

    def test_fake_assoc_writer_write_arrays_batch_matches(self) -> None:
        real = self._param_names(IncrementalAssocWriter.write_arrays_batch)
        fake = self._param_names(FakeAssocWriter.write_arrays_batch)
        assert fake == real, (
            f"FakeAssocWriter.write_arrays_batch drift:\n"
            f"  real: {real}\n  fake: {fake}\n"
            f"Production added or renamed a parameter; update the fake."
        )

    def test_fake_pipeline_runner_run_matches(self) -> None:
        real = self._param_names(PipelineRunner.run)
        fake = self._param_names(FakePipelineRunner.run)
        assert fake == real, (
            f"FakePipelineRunner.run drift:\n  real: {real}\n  fake: {fake}"
        )

    def test_fake_pipeline_runner_init_accepts_real_config(self) -> None:
        """FakePipelineRunner.__init__ must accept what production accepts.

        The real ``PipelineRunner.__init__`` parameter set must be a
        prefix of the fake's (the fake adds ``result=`` for test setup).
        If production adds a new positional parameter to ``__init__``,
        every consumer of the fake will still work but the new parameter
        won't be captured — fail here so it's caught.
        """
        real = self._param_names(PipelineRunner.__init__)
        fake = self._param_names(FakePipelineRunner.__init__)
        assert real == fake[: len(real)], (
            f"FakePipelineRunner.__init__ drift:\n"
            f"  real prefix: {real}\n  fake: {fake}\n"
            f"The fake must accept every real __init__ parameter "
            f"(extra trailing test-seam params like 'result' are fine)."
        )


class TestFakeJlinalg:
    def test_eigh_computes_real_decomposition_by_default(self) -> None:
        K = np.diag([1.0, 2.0, 3.0])
        w, v = FakeJlinalg().eigh(K)
        np.testing.assert_allclose(w, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(v @ np.diag(w) @ v.T, K)

    def test_eigh_raises_the_configured_error(self) -> None:
        with pytest.raises(MemoryError, match="boom"):
            FakeJlinalg(eigh_error=MemoryError("boom")).eigh(np.eye(2))

    def test_eigh_returns_the_configured_result(self) -> None:
        result = (np.ones(2), np.eye(2))
        assert FakeJlinalg(eigh_result=result).eigh(np.eye(2)) is result

    def test_undeclared_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = FakeJlinalg().dsyrk  # type: ignore[attr-defined]

    def test_declares_only_names_the_real_module_has(self) -> None:
        declared = {f.name for f in dataclasses.fields(FakeJlinalg)} | {"eigh"}
        declared -= {"eigh_result", "eigh_error"}
        missing = declared - set(dir(jamma.jlinalg))
        assert not missing, f"FakeJlinalg declares names jamma.jlinalg lacks: {missing}"

    def test_declares_every_name_eigen_reads(self) -> None:
        """A new ``jlinalg.<name>`` read in eigen.py must be added to the fake."""
        tree = ast.parse(Path(eigen.__file__).read_text())
        read = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "jlinalg"
        }
        declared = {f.name for f in dataclasses.fields(FakeJlinalg)} | {"eigh"}
        missing = read - declared
        assert not missing, f"eigen.py reads jlinalg names the fake lacks: {missing}"

    def test_eigh_signature_matches_real(self) -> None:
        real = TestFakeProductionDrift._param_names(jamma.jlinalg.eigh)
        fake = TestFakeProductionDrift._param_names(FakeJlinalg.eigh)
        assert fake == real, f"FakeJlinalg.eigh drift:\n  real: {real}\n  fake: {fake}"


class TestFakePsutil:
    def test_virtual_memory_fields_exist_on_psutil(self) -> None:
        declared = {f.name for f in dataclasses.fields(FakeVirtualMemory)}
        missing = declared - set(psutil.virtual_memory()._fields)
        assert not missing, f"psutil.virtual_memory() has no {missing}"

    def test_memory_info_fields_exist_on_psutil(self) -> None:
        declared = {f.name for f in dataclasses.fields(FakeMemoryInfo)}
        missing = declared - set(psutil.Process().memory_info()._fields)
        assert not missing, f"psutil memory_info() has no {missing}"

    def test_use_fake_psutil_pins_both_reads(self, monkeypatch) -> None:
        from tests.fakes import use_fake_psutil

        use_fake_psutil(monkeypatch, available=5.0, total=7.0, rss=1.0, vms=2.0)
        assert psutil.virtual_memory().available == 5.0
        assert psutil.virtual_memory().total == 7.0
        assert psutil.Process().memory_info().rss == 1.0
        assert psutil.Process().memory_info().vms == 2.0

    def test_undeclared_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = FakeVirtualMemory(available=1.0, total=1.0).percent  # type: ignore[attr-defined]
