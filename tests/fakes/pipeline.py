"""Fake for jamma.pipeline.PipelineRunner.

Replaces ``patch("jamma.cli.PipelineRunner")`` with a recording fake that
captures the ``PipelineConfig`` passed in and returns a pre-set
``PipelineResult`` from ``.run()``. Catches signature drift on either
constructor or ``run`` that ``MagicMock`` would absorb silently.
"""

from __future__ import annotations

from typing import Any


class FakePipelineRunner:
    """Single-call recorder. ``run()`` returns a pre-set result."""

    def __init__(self, config: Any, result: Any) -> None:
        self.config = config
        self._result = result
        self.run_calls = 0

    def run(self) -> Any:
        self.run_calls += 1
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
    """

    def __init__(self, result: Any) -> None:
        self._result = result
        self.runners: list[FakePipelineRunner] = []

    def __call__(self, config: Any) -> FakePipelineRunner:
        runner = FakePipelineRunner(config, self._result)
        self.runners.append(runner)
        return runner

    @property
    def last_config(self) -> Any:
        if not self.runners:
            raise AssertionError("FakePipelineRunner was never constructed")
        return self.runners[-1].config

    @property
    def call_count(self) -> int:
        return len(self.runners)
