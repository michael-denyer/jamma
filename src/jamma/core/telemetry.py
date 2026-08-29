"""Benchmark telemetry for JAMMA runs.

Provides :class:`BenchmarkRecord` and :func:`append_benchmark_record` for
appending structured run data to ``~/.jamma/benchmarks.jsonl``.

Telemetry is on by default.  ``JAMMA_NO_TELEMETRY`` set to any non-empty
value (e.g. ``JAMMA_NO_TELEMETRY=1``) or ``DO_NOT_TRACK=1`` opts out.

``JAMMA_NO_TELEMETRY`` is parsed once at the boundary — the CLI's
``--no-telemetry`` flag and the env var both resolve to
``PipelineConfig.no_telemetry``, which callers pass in as ``no_telemetry``
below, so this module never reads ``os.environ`` itself for it. ``"0"`` and
``"false"`` disable telemetry, matching :func:`jamma.core.constants.env_flag`.
``DO_NOT_TRACK`` follows the
`consoledonottrack.com <https://consoledonottrack.com/>`_ convention instead:
only ``DO_NOT_TRACK=1`` opts out; ``DO_NOT_TRACK=0`` explicitly opts in. That
different truthiness rule is why it stays a direct env read here rather than
a field on :class:`jamma.core.constants.Env`.

Write failures are logged as warnings and never propagate — telemetry must
never abort a GWAS run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

from loguru import logger

__all__ = ["BenchmarkRecord", "append_benchmark_record"]

_DEFAULT_BENCH_FILE: Path | None = None


def _default_bench_file() -> Path:
    """Return (and cache) the default benchmark file path.

    Deferred so ``Path.home()`` is not evaluated at import time, which
    would raise ``RuntimeError`` (Python 3.12+) or ``KeyError`` (older
    Pythons) in environments without ``HOME`` set.
    """
    global _DEFAULT_BENCH_FILE
    if _DEFAULT_BENCH_FILE is None:
        _DEFAULT_BENCH_FILE = Path.home() / ".jamma" / "benchmarks.jsonl"
    return _DEFAULT_BENCH_FILE


class _BenchmarkRequired(TypedDict):
    """Required fields — always available at every call site."""

    timestamp: str  # ISO 8601 UTC
    jamma_version: str
    n_samples: int
    n_snps: int
    backend: str  # see ExecutionPlan.runner_name


class BenchmarkRecord(_BenchmarkRequired, total=False):
    """One run's worth of benchmark data for telemetry.

    Required fields (from ``_BenchmarkRequired``): ``timestamp``,
    ``jamma_version``, ``n_samples``, ``n_snps``, ``backend``.
    All other fields are optional so callers can omit fields that are
    unavailable for their configuration.
    """

    n_cvt: int
    lmm_mode: int
    loco: bool
    n_chunks: int
    eigendecomp_s: float
    kinship_s: float
    lmm_s: float
    total_s: float
    rotation_s: float
    peak_memory_gb: float
    cpu_model: str
    blas_backend: str
    blas_threads: int
    total_ram_gb: float
    numpy_version: str
    platform: str


def append_benchmark_record(
    record: BenchmarkRecord,
    *,
    no_telemetry: bool = False,
    path: Path | None = None,
) -> None:
    """Append one record to the benchmark JSONL file.

    Never raises.  Write failures are logged as warnings.

    When ``no_telemetry`` is True, or ``DO_NOT_TRACK`` is set to ``"1"``,
    this function returns immediately without writing or creating
    directories.

    Args:
        record: Benchmark data to append.
        no_telemetry: Resolved ``JAMMA_NO_TELEMETRY`` / ``--no-telemetry``
            state. The caller resolves this once (``PipelineConfig.no_telemetry``)
            rather than this module reading ``os.environ`` itself.
        path: Override the default file path.  Default: ``~/.jamma/benchmarks.jsonl``.
    """
    if no_telemetry or os.environ.get("DO_NOT_TRACK") == "1":
        return

    try:
        dest = path if path is not None else _default_bench_file()
    except (RuntimeError, KeyError) as exc:
        logger.warning(f"Cannot determine benchmark file path: {exc}")
        return

    try:
        line = json.dumps(record) + "\n"
    except (TypeError, ValueError) as exc:
        logger.warning(
            f"Benchmark record is not JSON-serializable: {exc}. "
            f"Record keys: {list(record.keys())}"
        )
        return

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        logger.warning(
            f"Could not write benchmark record to {dest}: {exc}. "
            "Set JAMMA_NO_TELEMETRY=1 or DO_NOT_TRACK=1 to disable telemetry."
        )
