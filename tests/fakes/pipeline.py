"""Fake for jamma.pipeline.PipelineRunner.

Replaces ``patch("jamma.cli.PipelineRunner")`` with a recording fake that
captures the ``PipelineConfig`` passed in and returns a pre-set
``PipelineResult`` from ``.run()``. Catches signature drift on either
constructor or ``run`` that ``MagicMock`` would absorb silently.
"""

from __future__ import annotations

from jamma.pipeline import PipelineConfig, PipelineResult


class FakePipelineRunner:
    """Single-call recorder. ``run()`` returns a pre-set result."""

    def __init__(self, config: PipelineConfig, result: PipelineResult) -> None:
        self.config = config
        self._result = result
        self.ran_at_least_once = False

    def run(self) -> PipelineResult:
        if self.ran_at_least_once:
            raise AssertionError(
                "FakePipelineRunner.run() called twice; PipelineRunner is single-use"
            )
        self.ran_at_least_once = True
        return self._result


class FakePipelineRunnerFactory:
    """Substitute for ``jamma.cli.PipelineRunner``.

    Use via::

        factory = FakePipelineRunnerFactory(result=pipeline_result)
        monkeypatch.setattr("jamma.cli.PipelineRunner", factory)
        # ... invoke CLI ...
        assert factory.last_config.phenotype_columns == [1, 2, 3]

    Records every constructed runner on ``self.runners`` and exposes the
    most recent config on ``self.last_config`` for ergonomic assertions.
    Tests check ``len(factory.runners)`` for the number of constructions
    rather than asserting on a separate counter (project rule: assert on
    observable outputs, not internal call counts).
    """

    def __init__(self, result: PipelineResult) -> None:
        self._result = result
        self.runners: list[FakePipelineRunner] = []

    def __call__(self, config: PipelineConfig) -> FakePipelineRunner:
        runner = FakePipelineRunner(config, self._result)
        self.runners.append(runner)
        return runner

    @property
    def last_config(self) -> PipelineConfig:
        if not self.runners:
            raise AssertionError("FakePipelineRunner was never constructed")
        return self.runners[-1].config
