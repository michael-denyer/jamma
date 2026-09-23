"""Tests for the large-N comparison benchmark protocol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0


def _load_script_module():
    """Import the benchmark script without making ``scripts`` a package."""
    script_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        import bench_large_n_stages
    finally:
        if sys.path and sys.path[0] == str(script_dir):
            sys.path.pop(0)
    return bench_large_n_stages


def test_every_second_block_reverses_both_orders() -> None:
    """Do not repeat process allocation bias under one revision label."""
    benchmark = _load_script_module()

    assert benchmark.balanced_schedule(4) == [
        ["A", "B", "B", "A"],
        ["B", "A", "A", "B"],
        ["A", "B", "B", "A"],
        ["B", "A", "A", "B"],
    ]
    assert [benchmark._worker_start_order(index) for index in range(4)] == [
        ("A", "B"),
        ("B", "A"),
        ("A", "B"),
        ("B", "A"),
    ]


def test_stage_parser_rejects_duplicate_stage() -> None:
    """A duplicate stage would make the reported measurement count misleading."""
    benchmark = _load_script_module()

    with pytest.raises(Exception, match="must not contain duplicates"):
        benchmark._parse_stages("eigen,eigen")


def test_bench_common_leaves_jamma_unimported() -> None:
    """A/B workers import it before pinning jamma to their own source tree."""
    script_dir = Path(__file__).resolve().parent.parent / "scripts"
    probe = "import sys, _bench_common; print('jamma' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=script_dir,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"
