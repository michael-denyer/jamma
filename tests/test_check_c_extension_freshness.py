"""Tests for scripts/check_c_extension_freshness.py.

The freshness checker is called from tests/conftest.py (warning) and
from the pre-push hook (blocking). These tests verify its observable
behavior with synthetic file trees so a regression in the detector
won't silently pass through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _load_script_module():
    """Import check_c_extension_freshness as a module under test."""
    script_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        import check_c_extension_freshness as freshness
    finally:
        if sys.path and sys.path[0] == str(script_dir):
            sys.path.pop(0)
    return freshness


@pytest.mark.tier0
def test_check_extension_reports_fresh_when_so_is_newer(tmp_path: Path) -> None:
    """so_mtime > source_mtime => is_stale False."""
    freshness = _load_script_module()

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    c_file = src_dir / "foo.c"
    c_file.write_text("int main(){}")

    so_path = tmp_path / "foo.so"
    so_path.write_bytes(b"fake")
    # Force so mtime to be AFTER c_file mtime.
    import os

    os.utime(c_file, (c_file.stat().st_atime, so_path.stat().st_mtime - 100))

    spec = freshness.ExtensionSpec(
        label="fake",
        so_path=so_path,
        source_globs=((src_dir, "*.c"),),
        rebuild_command="fake-rebuild",
    )
    result = freshness._check_extension(spec)

    assert result.so_exists is True
    assert result.is_stale is False


@pytest.mark.tier0
def test_check_extension_reports_stale_when_source_is_newer(tmp_path: Path) -> None:
    """source_mtime > so_mtime => is_stale True, newest_source populated."""
    freshness = _load_script_module()

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    c_file = src_dir / "foo.c"
    c_file.write_text("int main(){}")

    so_path = tmp_path / "foo.so"
    so_path.write_bytes(b"fake")
    import os

    os.utime(so_path, (so_path.stat().st_atime, c_file.stat().st_mtime - 100))

    spec = freshness.ExtensionSpec(
        label="fake",
        so_path=so_path,
        source_globs=((src_dir, "*.c"),),
        rebuild_command="fake-rebuild",
    )
    result = freshness._check_extension(spec)

    assert result.so_exists is True
    assert result.is_stale is True
    assert result.newest_source == c_file


@pytest.mark.tier0
def test_check_extension_reports_missing_so_as_not_stale(tmp_path: Path) -> None:
    """Missing .so is not a drift failure — nothing has been built yet."""
    freshness = _load_script_module()

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "foo.c").write_text("int main(){}")

    spec = freshness.ExtensionSpec(
        label="fake",
        so_path=tmp_path / "foo.so",  # not created
        source_globs=((src_dir, "*.c"),),
        rebuild_command="fake-rebuild",
    )
    result = freshness._check_extension(spec)

    assert result.so_exists is False
    assert result.is_stale is False


@pytest.mark.tier0
def test_check_extension_picks_newest_of_multiple_sources(tmp_path: Path) -> None:
    """When multiple sources exist, newest one wins (and determines staleness)."""
    freshness = _load_script_module()

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    old_file = src_dir / "old.c"
    new_file = src_dir / "new.c"
    old_file.write_text("old")
    new_file.write_text("new")

    so_path = tmp_path / "foo.so"
    so_path.write_bytes(b"fake")
    import os

    # so is newer than old.c but older than new.c.
    so_mtime = so_path.stat().st_mtime
    os.utime(old_file, (old_file.stat().st_atime, so_mtime - 100))
    os.utime(new_file, (new_file.stat().st_atime, so_mtime + 100))

    spec = freshness.ExtensionSpec(
        label="fake",
        so_path=so_path,
        source_globs=((src_dir, "*.c"),),
        rebuild_command="fake-rebuild",
    )
    result = freshness._check_extension(spec)

    assert result.is_stale is True
    assert result.newest_source == new_file


@pytest.mark.tier0
def test_check_extension_scans_multiple_source_globs(tmp_path: Path) -> None:
    """Extensions with both .c and .h source globs must check both."""
    freshness = _load_script_module()

    c_dir = tmp_path / "src"
    h_dir = tmp_path / "include"
    c_dir.mkdir()
    h_dir.mkdir()
    (c_dir / "foo.c").write_text("int main(){}")
    h_file = h_dir / "foo.h"
    h_file.write_text("#pragma once")

    so_path = tmp_path / "foo.so"
    so_path.write_bytes(b"fake")
    import os

    # Header is newer than .so.
    so_mtime = so_path.stat().st_mtime
    os.utime(h_file, (h_file.stat().st_atime, so_mtime + 100))

    spec = freshness.ExtensionSpec(
        label="fake",
        so_path=so_path,
        source_globs=((c_dir, "*.c"), (h_dir, "*.h")),
        rebuild_command="fake-rebuild",
    )
    result = freshness._check_extension(spec)

    assert result.is_stale is True
    assert result.newest_source == h_file


@pytest.mark.tier0
def test_discover_extensions_returns_known_targets() -> None:
    """Smoke test: the shipped discovery returns exactly the two real
    extensions JAMMA builds. Guards against an accidental rename or
    deletion that would silently skip a drift check in production."""
    freshness = _load_script_module()
    exts = freshness._discover_extensions()
    labels = {e.label for e in exts}
    assert labels == {"_lmm_accel", "_jlinalg"}


@pytest.mark.tier0
def test_check_extension_degrades_to_not_stale_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stat()`` / ``glob()`` failures (unreadable .so, raced glob,
    permission issues on locked-down hosts) must NOT propagate — the
    freshness check is called unguarded from pytest_configure and an
    uncaught OSError there aborts the entire test session.

    Contract: any OSError inside _check_extension returns a
    FreshnessResult with is_stale=False, letting callers treat a
    transient FS failure as "can't tell; carry on". The pre-push hook
    re-runs in a clean environment to catch real drift.
    """
    freshness = _load_script_module()

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "foo.c").write_text("int main(){}")
    so_path = tmp_path / "foo.so"
    so_path.write_bytes(b"fake")

    spec = freshness.ExtensionSpec(
        label="fake",
        so_path=so_path,
        source_globs=((src_dir, "*.c"),),
        rebuild_command="fake-rebuild",
    )

    real_stat = Path.stat

    def _raise_on_so_stat(self, *args, **kwargs):
        # Fail only on the .so stat so we prove the except catches it
        # mid-function rather than just short-circuiting at exists().
        if self == so_path:
            raise PermissionError("simulated unreadable .so")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _raise_on_so_stat)

    result = freshness._check_extension(spec)

    assert result.is_stale is False, (
        "OSError during stat must degrade to is_stale=False, not propagate — "
        "the checker is called unguarded from pytest_configure"
    )
    # check_all() wraps the same call; verify it also returns a list
    # rather than raising, preserving the pytest_configure contract.
    monkeypatch.setattr(
        freshness,
        "_discover_extensions",
        lambda: [spec],
    )
    results = freshness.check_all()
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].is_stale is False


@pytest.mark.tier0
def test_main_returns_zero_on_real_tree() -> None:
    """The real tree should pass (the dev venv already built .so files).

    If this ever fails locally, the developer forgot to rebuild before
    running tests — exactly the condition the checker is designed to
    catch. Skips if either .so is missing (fresh checkout, pre-build).
    """
    freshness = _load_script_module()
    results = freshness.check_all()
    if any(not r.so_exists for r in results):
        pytest.skip("C extension(s) not yet built in this checkout")
    stale = [r for r in results if r.is_stale]
    assert not stale, (
        f"Real-tree drift detected: {[r.spec.label for r in stale]}. "
        f"Rebuild with the commands printed by check_c_extension_freshness.py."
    )
