"""Self-tests for the fakes package.

If a fake breaks, every test that uses it fails with a confusing message.
These tests fail with a clear message so the regression is obvious.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fakes import (
    FakeAssocWriter,
    FakePipelineRunner,
    FakePipelineRunnerFactory,
    FakeProgressBar,
    FakeProgressbarModule,
)

pytestmark = pytest.mark.tier0


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

        assert writer.call_count == 2
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
        assert bar.start_calls == 1
        assert bar.update_calls == [1, 5]
        assert bar.finish_calls == 1

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
        sentinel_result = object()
        factory = FakePipelineRunnerFactory(result=sentinel_result)

        config = {"phenotype_columns": [1, 2, 3]}
        runner = factory(config)

        assert runner.config is config
        assert runner.run() is sentinel_result
        assert runner.run_calls == 1

    def test_last_config_tracks_most_recent_call(self) -> None:
        factory = FakePipelineRunnerFactory(result=None)
        factory({"phenotype_columns": [1]})
        factory({"phenotype_columns": [2, 3]})

        assert factory.call_count == 2
        assert factory.last_config == {"phenotype_columns": [2, 3]}

    def test_last_config_before_any_call_raises(self) -> None:
        factory = FakePipelineRunnerFactory(result=None)
        with pytest.raises(AssertionError, match="never constructed"):
            _ = factory.last_config

    def test_runner_unknown_attribute_raises(self) -> None:
        runner = FakePipelineRunner(config={}, result=None)
        with pytest.raises(AttributeError):
            runner.cancel()  # type: ignore[attr-defined]
