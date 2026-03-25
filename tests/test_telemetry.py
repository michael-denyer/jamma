"""Unit tests for jamma.core.telemetry.

Covers TEL-01 through TEL-04 plus edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# TEL-01: append_benchmark_record() creates file and writes valid JSON
# ---------------------------------------------------------------------------


def test_append_creates_file(tmp_path: Path) -> None:
    """TEL-01: append_benchmark_record creates the file and writes valid JSONL."""
    from jamma.core.telemetry import append_benchmark_record

    dest = tmp_path / "benchmarks.jsonl"
    assert not dest.exists()

    append_benchmark_record({"n_samples": 100, "backend": "numpy-batch"}, path=dest)

    assert dest.exists()
    lines = dest.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["n_samples"] == 100
    assert record["backend"] == "numpy-batch"


def test_append_creates_parent_dirs(tmp_path: Path) -> None:
    """TEL-01: append_benchmark_record creates parent directories if absent."""
    from jamma.core.telemetry import append_benchmark_record

    dest = tmp_path / "nested" / "dir" / "benchmarks.jsonl"
    assert not dest.parent.exists()

    append_benchmark_record({"n_samples": 50}, path=dest)

    assert dest.exists()
    record = json.loads(dest.read_text().strip())
    assert record["n_samples"] == 50


# ---------------------------------------------------------------------------
# TEL-02: Write failure is logged as warning, never raised
# ---------------------------------------------------------------------------


def test_write_failure_warns_not_raises(tmp_path: Path) -> None:
    """TEL-02: OSError during write logs a warning, does not raise."""
    from loguru import logger

    from jamma.core.telemetry import append_benchmark_record

    # Point to a path where the parent cannot be created (file as parent dir)
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory")
    dest = blocker / "benchmarks.jsonl"  # parent is a file — mkdir will fail

    with patch.object(logger, "warning") as mock_warning:
        # Must not raise
        append_benchmark_record({"n_samples": 10}, path=dest)
        assert mock_warning.call_count == 1
        warning_msg = str(mock_warning.call_args)
        assert "benchmark" in warning_msg.lower() or str(dest) in warning_msg


def test_write_failure_on_readonly_file(tmp_path: Path) -> None:
    """TEL-02: OSError on read-only file logs a warning, does not raise."""
    from loguru import logger

    from jamma.core.telemetry import append_benchmark_record

    dest = tmp_path / "benchmarks.jsonl"
    dest.write_text("")  # create the file
    dest.chmod(0o444)  # read-only

    with patch.object(logger, "warning") as mock_warning:
        append_benchmark_record({"n_samples": 10}, path=dest)
        assert mock_warning.call_count == 1

    # cleanup so tmp_path can be removed
    dest.chmod(0o644)


# ---------------------------------------------------------------------------
# TEL-03: Multiple calls produce multiple JSONL lines
# ---------------------------------------------------------------------------


def test_multiple_records_appended(tmp_path: Path) -> None:
    """TEL-03: Two sequential calls produce two independently parseable JSON lines."""
    from jamma.core.telemetry import append_benchmark_record

    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 1000, "backend": "numpy-batch"}, path=dest)
    append_benchmark_record(
        {"n_samples": 2000, "backend": "numpy-streaming"}, path=dest
    )

    lines = dest.read_text().splitlines()
    assert len(lines) == 2

    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["n_samples"] == 1000
    assert rec2["n_samples"] == 2000
    assert rec1["backend"] == "numpy-batch"
    assert rec2["backend"] == "numpy-streaming"


def test_each_line_independently_parseable(tmp_path: Path) -> None:
    """TEL-03: Each JSONL line must be independently parseable."""
    from jamma.core.telemetry import append_benchmark_record

    dest = tmp_path / "benchmarks.jsonl"
    for i in range(5):
        append_benchmark_record({"n_samples": i * 100, "n_snps": i * 1000}, path=dest)

    for line in dest.read_text().splitlines():
        record = json.loads(line)  # must not raise
        assert isinstance(record, dict)


# ---------------------------------------------------------------------------
# TEL-04: JAMMA_NO_TELEMETRY=1 skips write entirely
# ---------------------------------------------------------------------------


def test_opt_out_env_var_skips_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-04: When JAMMA_NO_TELEMETRY=1, no file is created."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.setenv("JAMMA_NO_TELEMETRY", "1")
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 999}, path=dest)

    assert not dest.exists()


def test_opt_out_env_var_any_truthy_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-04: Any truthy value for JAMMA_NO_TELEMETRY skips write."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.setenv("JAMMA_NO_TELEMETRY", "true")
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 999}, path=dest)

    assert not dest.exists()


def test_opt_out_env_var_not_set_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-04: Without JAMMA_NO_TELEMETRY, file IS created normally."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 42}, path=dest)

    assert dest.exists()


# ---------------------------------------------------------------------------
# Edge: json.dumps failure (non-serializable value)
# ---------------------------------------------------------------------------


def test_non_serializable_value_warns_not_raises(tmp_path: Path) -> None:
    """Edge: Non-serializable value in record logs a warning, does not raise."""
    from loguru import logger

    from jamma.core.telemetry import append_benchmark_record

    dest = tmp_path / "benchmarks.jsonl"
    bad_record = {"bad_value": object()}  # type: ignore[dict-item]

    with patch.object(logger, "warning") as mock_warning:
        append_benchmark_record(bad_record, path=dest)  # type: ignore[arg-type]
        assert mock_warning.call_count == 1


# ---------------------------------------------------------------------------
# Exports check
# ---------------------------------------------------------------------------


def test_module_exports() -> None:
    """BenchmarkRecord and append_benchmark_record must be exported."""
    from jamma.core import telemetry

    assert hasattr(telemetry, "BenchmarkRecord")
    assert hasattr(telemetry, "append_benchmark_record")
    assert callable(telemetry.append_benchmark_record)
