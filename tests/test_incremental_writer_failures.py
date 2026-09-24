"""The association writer refuses to publish after any failure while writing."""

import contextlib
import errno
from pathlib import Path

import pytest

from jamma.lmm.assoc_output import IncrementalAssocWriter
from jamma.lmm.schema import MODE_SPECS
from tests.assoc_writer_helpers import (
    SampleBatch,
    _captured_warnings,
    _inject_write_failure,
    open_handle,
    single_snp_batch,
    ten_snp_batch,
)

pytestmark = pytest.mark.tier0


@pytest.fixture
def sample_result() -> SampleBatch:
    return single_snp_batch()


@pytest.fixture
def sample_results() -> SampleBatch:
    return ten_snp_batch()


class TestWriterPoisonedAfterFailure:
    """A writer that lost a batch never publishes, even when the error is swallowed."""

    def test_swallowed_write_failure_poisons_later_writes_and_clean_exit(
        self, tmp_path: Path, sample_results: SampleBatch
    ):
        """Later writes name the original failure and a clean exit raises."""
        destination = tmp_path / "result.assoc.txt"
        destination.write_bytes(b"previous completed run\n")

        with pytest.raises(RuntimeError, match="Read-only file system"):
            with IncrementalAssocWriter(destination, MODE_SPECS[1]) as writer:
                with _inject_write_failure(
                    writer,
                    message="Read-only file system",
                    always=True,
                    tab_only=True,
                    errno_code=errno.EROFS,
                ):
                    with pytest.raises(OSError):
                        writer.write_arrays_batch(
                            *sample_results.slice_one(0).as_call_args()
                        )
                with pytest.raises(RuntimeError, match="Read-only file system"):
                    writer.write_arrays_batch(
                        *sample_results.slice_one(1).as_call_args()
                    )

        assert destination.read_bytes() == b"previous completed run\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["result.assoc.txt"]

    def test_swallowed_tell_failure_never_publishes_a_gap(
        self, tmp_path: Path, sample_results: SampleBatch
    ):
        """A batch lost to a tell() failure blocks publication of the rest."""
        destination = tmp_path / "result.assoc.txt"
        destination.write_bytes(b"previous completed run\n")
        raised: BaseException | None = None

        try:
            with IncrementalAssocWriter(destination, MODE_SPECS[1]) as writer:
                handle = open_handle(writer)
                original_tell = handle.tell
                tell_calls = 0

                def tell_fails_on_second_batch() -> int:
                    nonlocal tell_calls
                    tell_calls += 1
                    if tell_calls == 2:
                        raise OSError(errno.ENOSPC, "No space left on device")
                    return original_tell()

                handle.tell = tell_fails_on_second_batch
                for i in range(3):
                    # A caller that logs and carries on.
                    with contextlib.suppress(OSError, RuntimeError):
                        writer.write_arrays_batch(
                            *sample_results.slice_one(i).as_call_args()
                        )
        except RuntimeError as error:
            raised = error

        assert destination.read_bytes() == b"previous completed run\n", (
            "rs10001 was never written, so the run must not publish"
        )
        assert raised is not None
        assert "No space left on device" in str(raised)

    def test_interrupt_after_discard_does_not_claim_a_retained_path(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """The loco.py ExitStack shape: write error, then Ctrl-C during cleanup."""
        destination = tmp_path / "result.assoc.txt"

        def interrupted_cleanup() -> None:
            raise KeyboardInterrupt

        with _captured_warnings() as messages:
            with pytest.raises(KeyboardInterrupt):
                with contextlib.ExitStack() as stack:
                    writer = stack.enter_context(
                        IncrementalAssocWriter(destination, MODE_SPECS[1])
                    )
                    stack.callback(interrupted_cleanup)
                    with _inject_write_failure(
                        writer,
                        message="Read-only file system",
                        always=True,
                        tab_only=True,
                        errno_code=errno.EROFS,
                    ):
                        writer.write_arrays_batch(*sample_result.as_call_args())

        claimed = [
            m.rsplit("retained at ", 1)[1].strip()
            for m in messages
            if "retained at " in m
        ]
        assert all(Path(p).exists() for p in claimed), (
            f"log names a retained path that does not exist: {claimed}"
        )
        assert any("no partial output retained" in m for m in messages), messages
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    @pytest.mark.parametrize("error", [KeyboardInterrupt, MemoryError])
    def test_interrupt_mid_write_retains_only_complete_rows(
        self, tmp_path: Path, sample_results: SampleBatch, error
    ):
        """A partial cut mid-row is truncated to the last complete row."""
        destination = tmp_path / "result.assoc.txt"

        with _captured_warnings() as messages:
            with pytest.raises(error):
                with IncrementalAssocWriter(destination, MODE_SPECS[1]) as writer:
                    handle = open_handle(writer)
                    original_write = handle.write
                    data_writes = 0

                    def half_row_then_interrupt(data: str) -> int:
                        nonlocal data_writes
                        if "\t" in data:
                            data_writes += 1
                            if data_writes == 2:
                                original_write(data[: len(data) // 2])
                                raise error()
                        return original_write(data)

                    handle.write = half_row_then_interrupt
                    for i in range(3):
                        writer.write_arrays_batch(
                            *sample_results.slice_one(i).as_call_args()
                        )

        partial = (tmp_path / "result.assoc.txt.partial").read_text()
        assert partial.endswith("\n"), f"partial ends mid-row: {partial!r}"
        rows = partial.splitlines()[1:]
        assert [row.split("\t")[1] for row in rows] == ["rs10000"]
        assert f"after {len(rows)} results written" in messages[0]
