"""Tests for the temp-dir lifetime shared by the parallel matrix reader and writer."""

from pathlib import Path

import pytest

from jamma.io._parallel_text import temp_dir_beside

pytestmark = pytest.mark.tier0


def test_temp_dir_removed_with_untracked_contents_on_exit(tmp_path: Path) -> None:
    """Leaving the block removes the directory and every file inside it."""
    target = tmp_path / "matrix.txt"
    with temp_dir_beside(target, prefix=".jamma_mtest_") as tmp_dir:
        assert tmp_dir.parent == tmp_path
        (tmp_dir / "untracked.dat").write_bytes(b"x")

    assert list(tmp_path.glob(".jamma_mtest_*")) == []


def test_temp_dir_removed_when_block_raises(tmp_path: Path) -> None:
    """An exception inside the block still removes the directory."""
    target = tmp_path / "matrix.txt"
    with pytest.raises(KeyboardInterrupt):
        with temp_dir_beside(target, prefix=".jamma_mtest_") as tmp_dir:
            (tmp_dir / "chunk_000000.txt").write_bytes(b"x")
            raise KeyboardInterrupt

    assert list(tmp_path.glob(".jamma_mtest_*")) == []
