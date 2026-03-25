"""Unit tests for jamma.core.telemetry.

Covers TEL-01 through TEL-06 plus edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# TEL-01: append_benchmark_record() creates file and writes valid JSON
# ---------------------------------------------------------------------------


def test_append_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TEL-01: append_benchmark_record creates the file and writes valid JSONL."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    dest = tmp_path / "benchmarks.jsonl"
    assert not dest.exists()

    append_benchmark_record({"n_samples": 100, "backend": "numpy-batch"}, path=dest)

    assert dest.exists()
    lines = dest.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["n_samples"] == 100
    assert record["backend"] == "numpy-batch"


def test_append_creates_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-01: append_benchmark_record creates parent directories if absent."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    dest = tmp_path / "nested" / "dir" / "benchmarks.jsonl"
    assert not dest.parent.exists()

    append_benchmark_record({"n_samples": 50}, path=dest)

    assert dest.exists()
    record = json.loads(dest.read_text().strip())
    assert record["n_samples"] == 50


# ---------------------------------------------------------------------------
# TEL-02: Write failure is logged as warning, never raised
# ---------------------------------------------------------------------------


def test_write_failure_warns_not_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-02: OSError during write logs a warning, does not raise."""
    from loguru import logger

    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
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


def test_write_failure_on_readonly_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-02: OSError on read-only file logs a warning, does not raise."""
    from loguru import logger

    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
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


def test_multiple_records_appended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-03: Two sequential calls produce two independently parseable JSON lines."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
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


def test_each_line_independently_parseable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-03: Each JSONL line must be independently parseable."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
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
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 999}, path=dest)

    assert not dest.exists()


def test_opt_out_env_var_any_truthy_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-04: Any non-empty value for JAMMA_NO_TELEMETRY skips write."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.setenv("JAMMA_NO_TELEMETRY", "true")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 999}, path=dest)

    assert not dest.exists()


def test_opt_out_env_var_not_set_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-04: Without JAMMA_NO_TELEMETRY or DO_NOT_TRACK, file IS created normally."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 42}, path=dest)

    assert dest.exists()


# ---------------------------------------------------------------------------
# TEL-05: DO_NOT_TRACK=1 skips write entirely
# ---------------------------------------------------------------------------


def test_do_not_track_env_var_skips_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-05: When DO_NOT_TRACK=1, no file is created."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 999}, path=dest)

    assert not dest.exists()


def test_do_not_track_zero_allows_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-05: DO_NOT_TRACK=0 means explicit opt-in per consoledonottrack.com."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.setenv("DO_NOT_TRACK", "0")
    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 999}, path=dest)

    assert dest.exists()


def test_do_not_track_non_one_allows_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-05: DO_NOT_TRACK values other than '1' do not disable telemetry."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.setenv("DO_NOT_TRACK", "true")
    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 999}, path=dest)

    assert dest.exists()


def test_do_not_track_unset_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEL-05: Without DO_NOT_TRACK, file IS created normally."""
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    dest = tmp_path / "benchmarks.jsonl"

    append_benchmark_record({"n_samples": 42}, path=dest)

    assert dest.exists()


# ---------------------------------------------------------------------------
# TEL-06: --no-telemetry CLI flag sets JAMMA_NO_TELEMETRY
# ---------------------------------------------------------------------------


def test_no_telemetry_flag_sets_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """TEL-06: --no-telemetry sets JAMMA_NO_TELEMETRY=1 in the environment."""
    import os

    from click.testing import CliRunner

    from jamma.cli import main

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)

    env_captured: dict[str, str | None] = {}

    def capture_env(**_kwargs: object) -> None:
        env_captured["JAMMA_NO_TELEMETRY"] = os.environ.get("JAMMA_NO_TELEMETRY")

    # Patch _run_lmm so we don't need real data files — just need to reach
    # the point after os.environ is set.
    with patch("jamma.cli._run_lmm", side_effect=capture_env):
        runner = CliRunner()
        runner.invoke(
            main,
            ["--no-telemetry", "-lmm", "1", "-bfile", "dummy", "-k", "dummy.cXX.txt"],
        )

    assert env_captured.get("JAMMA_NO_TELEMETRY") == "1"


# ---------------------------------------------------------------------------
# Edge: _default_bench_file() failure (no HOME)
# ---------------------------------------------------------------------------


def test_no_home_warns_not_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edge: When Path.home() fails, logs a warning, does not raise."""
    from loguru import logger

    from jamma.core import telemetry
    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.setattr(telemetry, "_DEFAULT_BENCH_FILE", None)
    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)

    with (
        patch("pathlib.Path.home", side_effect=RuntimeError("no home dir")),
        patch.object(logger, "warning") as mock_warning,
    ):
        append_benchmark_record({"n_samples": 10})  # no path= override
        assert mock_warning.call_count == 1
        assert "path" in str(mock_warning.call_args).lower()


# ---------------------------------------------------------------------------
# Edge: json.dumps failure (non-serializable value)
# ---------------------------------------------------------------------------


def test_non_serializable_value_warns_not_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edge: Non-serializable value in record logs a warning, does not raise."""
    from loguru import logger

    from jamma.core.telemetry import append_benchmark_record

    monkeypatch.delenv("JAMMA_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
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
