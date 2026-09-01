"""Tests for the shared atomic-publish primitives.

These pin the two properties callers depend on and that a rename would
otherwise silently break: the temp path is a sibling of the destination (so
os.replace is atomic), and it is unique per writer (so concurrent publishers
do not clobber each other). Assertions are on structure, never on the exact
generated name.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jamma.utils.atomic_publish import atomic_output, publish_temp_path, unlink_quietly


@pytest.mark.tier0
class TestPublishTempPath:
    """publish_temp_path builds a unique sibling of its destination."""

    def test_temp_is_sibling_of_destination(self) -> None:
        """os.replace() is only atomic within one filesystem, so stay beside."""
        target = Path("/data/run/result.cXX.txt")
        assert publish_temp_path(target).parent == target.parent

    def test_temp_differs_from_destination(self) -> None:
        """A temp equal to the target would make the write non-atomic."""
        target = Path("/data/run/result.cXX.txt")
        assert publish_temp_path(target) != target

    def test_temp_is_unique_per_call(self) -> None:
        """Two writers publishing one destination must not share a temp file."""
        target = Path("/data/run/result.cXX.txt")
        names = {publish_temp_path(target).name for _ in range(100)}
        assert len(names) == 100

    def test_temp_carries_pid(self) -> None:
        """The pid narrows a stray temp to the process that abandoned it."""
        target = Path("/data/run/result.cXX.txt")
        assert f".{os.getpid()}." in publish_temp_path(target).name

    def test_temp_is_hidden(self) -> None:
        """A dotted name keeps an in-flight publish out of a casual listing."""
        assert publish_temp_path(Path("/data/run/result.txt")).name.startswith(".")

    def test_suffix_is_appended_when_requested(self) -> None:
        """np.save appends .npy when absent, so .npy callers ask for it here."""
        tmp = publish_temp_path(Path("/data/run/result.eigenD.npy"), suffix=".npy")
        assert tmp.suffix == ".npy"

    def test_suffix_replaces_the_destination_extension(self) -> None:
        """With a suffix the stem is used, so the extension is not doubled."""
        tmp = publish_temp_path(Path("/data/run/result.eigenD.npy"), suffix=".npy")
        assert not tmp.name.endswith(".npy.npy")
        assert "result.eigenD" in tmp.name

    def test_without_suffix_the_full_name_is_kept(self) -> None:
        """Text callers keep the whole name so the temp is traceable."""
        tmp = publish_temp_path(Path("/data/run/result.cXX.txt"))
        assert "result.cXX.txt" in tmp.name


@pytest.mark.tier0
class TestUnlinkQuietly:
    """unlink_quietly removes a file without raising on a failure path."""

    def test_removes_an_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "leftover.tmp"
        target.write_text("x")

        unlink_quietly(target)

        assert not target.exists()

    def test_absent_file_is_not_an_error(self, tmp_path: Path) -> None:
        """Cleanup runs on paths that may never have been created."""
        unlink_quietly(tmp_path / "never-created.tmp")

    def test_does_not_raise_when_removal_fails(self, tmp_path: Path) -> None:
        """Cleanup must never mask the real error it is unwinding from.

        A directory raises OSError (not FileNotFoundError) from Path.unlink,
        which is the branch that has to stay quiet.
        """
        blocked = tmp_path / "a-directory"
        blocked.mkdir()

        unlink_quietly(blocked)

        assert blocked.exists()

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        """Callers pass raw memmap/chunk path strings straight through."""
        target = tmp_path / "leftover.tmp"
        target.write_text("x")

        unlink_quietly(str(target))

        assert not target.exists()


@pytest.mark.tier0
class TestAtomicOutput:
    """atomic_output owns the ordinary publish and cleanup lifecycle."""

    def test_success_publishes_bytes_and_removes_temp(self, tmp_path: Path) -> None:
        target = tmp_path / "result.txt"
        with atomic_output(target) as tmp_path_for_write:
            tmp_path_for_write.write_text("new contents")
            assert not target.exists()

        assert target.read_text() == "new contents"
        assert not list(tmp_path.glob(".result.txt.tmp.*"))

    def test_writer_failure_preserves_existing_target(self, tmp_path: Path) -> None:
        target = tmp_path / "result.txt"
        target.write_text("old contents")

        with pytest.raises(RuntimeError, match="injected write failure"):
            with atomic_output(target) as tmp_path_for_write:
                tmp_path_for_write.write_text("partial")
                raise RuntimeError("injected write failure")

        assert target.read_text() == "old contents"
        assert not list(tmp_path.glob(".result.txt.tmp.*"))

    def test_keyboard_interrupt_cleans_temp(self, tmp_path: Path) -> None:
        target = tmp_path / "result.txt"

        with pytest.raises(KeyboardInterrupt):
            with atomic_output(target) as tmp_path_for_write:
                tmp_path_for_write.write_text("partial")
                raise KeyboardInterrupt

        assert not target.exists()
        assert not list(tmp_path.glob(".result.txt.tmp.*"))

    def test_replace_failure_preserves_existing_target_and_cleans_temp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "result.txt"
        target.write_text("old contents")

        def raise_on_replace(self: Path, target: object) -> None:
            raise OSError("injected replace failure")

        monkeypatch.setattr(Path, "replace", raise_on_replace)

        with pytest.raises(OSError, match="injected replace failure"):
            with atomic_output(target) as tmp_path_for_write:
                tmp_path_for_write.write_text("new contents")

        assert target.read_text() == "old contents"
        assert not list(tmp_path.glob(".result.txt.tmp.*"))

    def test_suffix_is_used_for_numpy_style_writers(self, tmp_path: Path) -> None:
        target = tmp_path / "result.eigenD.npy"
        with atomic_output(target, suffix=".npy") as tmp_path_for_write:
            assert tmp_path_for_write.suffix == ".npy"
            tmp_path_for_write.write_bytes(b"binary")

        assert target.read_bytes() == b"binary"
        assert not list(tmp_path.glob(".result.eigenD.tmp.*.npy"))
