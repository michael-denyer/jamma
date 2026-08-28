"""Tests for incremental association result writer."""

import errno
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.schema import SnpMeta

pytestmark = pytest.mark.tier0


def open_handle(writer: IncrementalAssocWriter) -> TextIOWrapper:
    """Return an open writer's file handle, for injecting I/O failures.

    The tests below drive the retry and rollback paths by replacing methods on
    the live handle, which means reaching past the public API. ``_file`` is
    None until ``__enter__`` runs, so going through here asserts the writer is
    open once instead of at every call site.
    """
    handle = writer._file
    assert handle is not None, "writer must be open before patching its handle"
    return handle


@dataclass
class SampleBatch:
    """One or more SNPs as write_arrays_batch's array arguments (Wald mode).

    write_arrays_batch is the only writer entry point left, so tests drive
    it with a small batch of pre-built arrays rather than AssocResult
    objects. ``lmm_mode=1`` (Wald) throughout; ``arrays`` keys match
    RESULT_FIELDS[1]'s array_key names (betas, ses, logls, lambdas, pwalds).
    """

    snp_indices: np.ndarray
    snp_info: SnpMeta
    afs: np.ndarray
    miss_counts: np.ndarray
    arrays: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.snp_indices)

    def as_call_args(self) -> tuple:
        """Positional args for writer.write_arrays_batch(*batch.as_call_args())."""
        return (
            1,
            self.snp_indices,
            self.snp_info,
            self.afs,
            self.miss_counts,
            self.arrays,
        )

    def slice_one(self, i: int) -> "SampleBatch":
        """Return a one-SNP batch holding position i's values, index reset to 0."""
        return SampleBatch(
            snp_indices=np.array([0]),
            snp_info=SnpMeta(
                chr=self.snp_info.chr[[i]],
                rs=self.snp_info.rs[[i]],
                pos=self.snp_info.pos[[i]],
                a1=self.snp_info.a1[[i]],
                a0=self.snp_info.a0[[i]],
            ),
            afs=self.afs[[i]],
            miss_counts=self.miss_counts[[i]],
            arrays={k: v[[i]] for k, v in self.arrays.items()},
        )


@pytest.fixture
def sample_result() -> SampleBatch:
    """Create a single-SNP batch for testing."""
    return SampleBatch(
        snp_indices=np.array([0]),
        snp_info=SnpMeta.from_dicts(
            [{"chr": "1", "rs": "rs12345", "pos": 100000, "a1": "A", "a0": "G"}]
        ),
        afs=np.array([0.25]),
        miss_counts=np.array([5]),
        arrays={
            "betas": np.array([0.123456]),
            "ses": np.array([0.0234567]),
            "logls": np.array([-1234.567]),
            "lambdas": np.array([0.456789]),
            "pwalds": np.array([0.00123456]),
        },
    )


@pytest.fixture
def sample_results() -> SampleBatch:
    """Create a ten-SNP batch for testing."""
    n = 10
    return SampleBatch(
        snp_indices=np.arange(n),
        snp_info=SnpMeta.from_dicts(
            [
                {
                    "chr": str((i % 22) + 1),
                    "rs": f"rs{10000 + i}",
                    "pos": 100000 + i * 1000,
                    "a1": "A",
                    "a0": "G",
                }
                for i in range(n)
            ]
        ),
        afs=np.array([0.1 + i * 0.05 for i in range(n)]),
        miss_counts=np.arange(n),
        arrays={
            "betas": np.array([0.1 * (i + 1) for i in range(n)]),
            "ses": np.array([0.01 * (i + 1) for i in range(n)]),
            "logls": np.array([-1000.0 - i for i in range(n)]),
            "lambdas": np.array([0.5 + i * 0.1 for i in range(n)]),
            "pwalds": np.array([0.05 / (i + 1) for i in range(n)]),
        },
    )


class TestIncrementalAssocWriter:
    """Tests for IncrementalAssocWriter class."""

    def test_writes_header(self, tmp_path: Path):
        """Should write GEMMA-compatible header on open."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path):
            pass  # Just open and close

        content = output_path.read_text()
        assert content.startswith("chr\trs\tps\tn_miss")
        assert "beta" in content
        assert "p_wald" in content

    def test_writes_single_result(self, tmp_path: Path, sample_result: SampleBatch):
        """Should write single result correctly."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write_arrays_batch(*sample_result.as_call_args())

        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # Header + 1 result
        assert "rs12345" in lines[1]
        assert writer.count == 1

    def test_writes_multiple_results(self, tmp_path: Path, sample_results: SampleBatch):
        """Should write multiple results correctly."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write_arrays_batch(*sample_results.as_call_args())

        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 11  # Header + 10 results
        assert writer.count == 10

    def test_write_arrays_batch_method(
        self, tmp_path: Path, sample_results: SampleBatch
    ):
        """write_arrays_batch should write all results at once."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write_arrays_batch(*sample_results.as_call_args())

        assert writer.count == len(sample_results)

    def test_write_arrays_batch_single_flush(
        self, tmp_path: Path, sample_results: SampleBatch
    ):
        """write_arrays_batch should have far fewer flushes than N results.

        CPython's TextIOWrapper.tell() internally calls flush(), so the
        exact count is 2 (tell-induced + explicit). The key invariant is
        that it's constant per batch, not proportional to len(results).
        The old per-SNP path would have produced len(results) flushes.
        """
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            original_flush = open_handle(writer).flush
            flush_count = 0

            def counting_flush():
                nonlocal flush_count
                flush_count += 1
                return original_flush()

            open_handle(writer).flush = counting_flush
            writer.write_arrays_batch(*sample_results.as_call_args())
            # Count flushes only during write_arrays_batch, not __exit__
            batch_flush_count = flush_count

        # Constant flush count per batch (tell + explicit), not N per-SNP
        n = len(sample_results)
        assert batch_flush_count <= 2, f"Expected <=2 flushes, got {batch_flush_count}"
        assert batch_flush_count < n, (
            f"Batch flush count ({batch_flush_count}) should be far less than "
            f"result count ({n})"
        )

    def test_write_arrays_batch_empty_indices(self, tmp_path: Path):
        """write_arrays_batch with empty snp_indices is a no-op: count stays 0."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write_arrays_batch(
                1,
                np.array([], dtype=int),
                SnpMeta.from_dicts([]),
                np.array([]),
                np.array([], dtype=int),
                {},
            )

        assert writer.count == 0
        # File should only contain header
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1  # Header only

    def test_creates_parent_directories(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Should create parent directories if needed."""
        output_path = tmp_path / "deep" / "nested" / "dir" / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write_arrays_batch(*sample_result.as_call_args())

        assert output_path.exists()

    def test_raises_if_not_opened(self, sample_result: SampleBatch):
        """Should raise error if write_arrays_batch called without context manager."""
        writer = IncrementalAssocWriter(Path("dummy.txt"))

        with pytest.raises(RuntimeError, match="not opened"):
            writer.write_arrays_batch(*sample_result.as_call_args())

    def test_retries_on_oserror(self, tmp_path: Path, sample_result: SampleBatch):
        """Should retry on OSError and succeed on second attempt."""
        output_path = tmp_path / "test.assoc.txt"
        call_count = 0

        with IncrementalAssocWriter(output_path) as writer:
            original_write = open_handle(writer).write

            def flaky_write(data):
                nonlocal call_count
                call_count += 1
                # First call to this patched write is the data line -- fail it.
                # Second call (retry) succeeds.
                if call_count == 1:
                    raise OSError("Disk full")
                return original_write(data)

            open_handle(writer).write = flaky_write

            with patch("jamma.lmm.io.time.sleep") as mock_sleep:
                writer.write_arrays_batch(*sample_result.as_call_args())
                # Should have slept once with 0.1s (first retry backoff)
                mock_sleep.assert_called_once_with(0.1)

        assert writer.count == 1

    def test_deletes_partial_on_final_failure(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Should delete partial file after exhausting all retries."""
        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep"):
            with pytest.raises(OSError):
                with IncrementalAssocWriter(output_path) as writer:
                    # Make every write after header fail
                    original_write = open_handle(writer).write

                    def always_fail(data):
                        if "\t" in data:  # data lines have tabs
                            raise OSError("Disk full")
                        return original_write(data)

                    open_handle(writer).write = always_fail
                    writer.write_arrays_batch(*sample_result.as_call_args())

        # Partial file should be cleaned up
        assert not output_path.exists(), "Partial output file should be deleted"

    def test_no_retry_on_non_oserror(self, tmp_path: Path, sample_result: SampleBatch):
        """Non-OSError exceptions propagate immediately without retries."""
        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep") as mock_sleep:
            with pytest.raises(TypeError):
                with IncrementalAssocWriter(output_path) as writer:
                    # Patch spec lookup to raise TypeError before any bytes
                    # are formatted, mirroring a non-OSError formatting failure.
                    with patch(
                        "jamma.lmm.io.get_spec",
                        side_effect=TypeError("bad format"),
                    ):
                        writer.write_arrays_batch(*sample_result.as_call_args())

            # time.sleep should NOT have been called
            mock_sleep.assert_not_called()

    def test_cleanup_on_context_exit_with_error(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Partial file deleted when OSError propagates through context exit."""
        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep"):
            with pytest.raises(OSError):
                with IncrementalAssocWriter(output_path) as writer:
                    original_write = open_handle(writer).write

                    def always_fail(data):
                        if "\t" in data:
                            raise OSError("Disk full")
                        return original_write(data)

                    open_handle(writer).write = always_fail
                    writer.write_arrays_batch(*sample_result.as_call_args())

        # File should be cleaned up by __exit__
        assert not output_path.exists(), (
            "Partial file should be deleted on context exit with OSError"
        )

    def test_retains_partial_on_keyboard_interrupt(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Partial file retained (not deleted) on KeyboardInterrupt."""
        output_path = tmp_path / "test.assoc.txt"

        with pytest.raises(KeyboardInterrupt):
            with IncrementalAssocWriter(output_path) as writer:
                writer.write_arrays_batch(*sample_result.as_call_args())
                raise KeyboardInterrupt()

        assert output_path.exists(), "Partial file should be retained on interrupt"
        content = output_path.read_text()
        assert "rs12345" in content

    def test_retains_partial_on_system_exit(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Partial file retained (not deleted) on SystemExit (e.g. SIGTERM)."""
        output_path = tmp_path / "test.assoc.txt"

        with pytest.raises(SystemExit):
            with IncrementalAssocWriter(output_path) as writer:
                writer.write_arrays_batch(*sample_result.as_call_args())
                raise SystemExit(1)

        assert output_path.exists(), "Partial file should be retained on SystemExit"
        content = output_path.read_text()
        assert "rs12345" in content

    def test_cleanup_on_computation_error(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Partial file deleted for any Exception subclass (computation error)."""
        output_path = tmp_path / "test.assoc.txt"

        with pytest.raises(ValueError, match="bad data"):
            with IncrementalAssocWriter(output_path) as writer:
                writer.write_arrays_batch(*sample_result.as_call_args())
                raise ValueError("bad data")

        assert not output_path.exists(), (
            "Partial file should be deleted on computation error"
        )

    def test_write_arrays_batch_retries_on_flaky_write(
        self, tmp_path: Path, sample_results: SampleBatch
    ):
        """write_arrays_batch retries on first OSError, succeeds on the second."""
        output_path = tmp_path / "test.assoc.txt"
        call_count = 0

        with IncrementalAssocWriter(output_path) as writer:
            original_write = open_handle(writer).write

            def flaky_write(data):
                nonlocal call_count
                call_count += 1
                # First write of data (not header) fails; retry succeeds
                if call_count == 1 and "\t" in data:
                    raise OSError("Transient I/O error")
                return original_write(data)

            open_handle(writer).write = flaky_write

            with patch("jamma.lmm.io.time.sleep") as mock_sleep:
                writer.write_arrays_batch(*sample_results.as_call_args())
                mock_sleep.assert_called_once_with(0.1)

        assert writer.count == len(sample_results)

        # Verify file content is complete and not duplicated
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1 + len(sample_results), (
            f"Expected header + {len(sample_results)} results, got {len(lines)} lines"
        )

    def test_write_arrays_batch_raises_if_not_opened(self, sample_results: SampleBatch):
        """write_arrays_batch on unopened writer raises RuntimeError."""
        writer = IncrementalAssocWriter(Path("dummy.txt"))

        with pytest.raises(RuntimeError, match="not opened"):
            writer.write_arrays_batch(*sample_results.as_call_args())

    def test_write_retries_on_transient_eio(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Verify retry with exponential backoff on retryable EIO errno.

        Injects a transient errno.EIO on first write attempt. The writer
        should retry and succeed on the second attempt.
        """
        import errno as errno_mod

        output_path = tmp_path / "test.assoc.txt"
        call_count = 0

        with IncrementalAssocWriter(output_path) as writer:
            original_write = open_handle(writer).write

            def eio_write(data):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    err = OSError("I/O error")
                    err.errno = errno_mod.EIO
                    raise err
                return original_write(data)

            open_handle(writer).write = eio_write

            with patch("jamma.lmm.io.time.sleep") as mock_sleep:
                writer.write_arrays_batch(*sample_result.as_call_args())
                mock_sleep.assert_called_once_with(0.1)

        assert writer.count == 1
        # Verify file has header + 1 result
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_write_fails_immediately_on_eacces(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Verify immediate failure on non-retryable EACCES errno.

        EACCES is not in _RETRYABLE_ERRNOS, so the writer should fail
        after a single attempt with no retry backoff.
        """
        import errno as errno_mod

        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep") as mock_sleep:
            with pytest.raises(OSError):
                with IncrementalAssocWriter(output_path) as writer:
                    original_write = open_handle(writer).write

                    def eacces_write(data):
                        if "\t" in data:
                            err = OSError("Permission denied")
                            err.errno = errno_mod.EACCES
                            raise err
                        return original_write(data)

                    open_handle(writer).write = eacces_write
                    writer.write_arrays_batch(*sample_result.as_call_args())

            # No retry sleeps should have occurred
            mock_sleep.assert_not_called()

        # Partial file should be cleaned up
        assert not output_path.exists(), "Partial output file should be deleted"

    def test_write_rollback_preserves_first_result(
        self, tmp_path: Path, sample_results: SampleBatch
    ):
        """Verify seek+truncate rollback on second write preserves first result.

        Writes first result successfully, then injects a permanent OSError
        on the second write. After all retries exhausted, verifies the file
        is cleaned up (partial file deleted).
        """
        output_path = tmp_path / "test.assoc.txt"
        write_call_count = 0

        with patch("jamma.lmm.io.time.sleep"):
            with pytest.raises(OSError):
                with IncrementalAssocWriter(output_path) as writer:
                    # Write first result successfully
                    writer.write_arrays_batch(
                        *sample_results.slice_one(0).as_call_args()
                    )
                    assert writer.count == 1

                    # Now make all subsequent writes fail permanently
                    def fail_after_first(data):
                        nonlocal write_call_count
                        write_call_count += 1
                        raise OSError("Disk full")

                    open_handle(writer).write = fail_after_first
                    # This should fail after retries and clean up
                    writer.write_arrays_batch(
                        *sample_results.slice_one(1).as_call_args()
                    )

        # _cleanup_partial deletes the file after exhausting retries
        assert not output_path.exists(), (
            "Partial output file should be deleted after exhausting retries"
        )

    def test_retains_partial_on_memory_error(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """Partial file retained on MemoryError (valid data, resource exhaustion)."""
        output_path = tmp_path / "test.assoc.txt"

        with pytest.raises(MemoryError):
            with IncrementalAssocWriter(output_path) as writer:
                writer.write_arrays_batch(*sample_result.as_call_args())
                raise MemoryError("OOM at 90% completion")

        assert output_path.exists(), "Partial file should be retained on MemoryError"
        content = output_path.read_text()
        assert "rs12345" in content

    def test_retains_partial_on_generator_exit(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """GeneratorExit (other BaseException) retains partial file."""
        output_path = tmp_path / "test.assoc.txt"

        with pytest.raises(GeneratorExit):
            with IncrementalAssocWriter(output_path) as writer:
                writer.write_arrays_batch(*sample_result.as_call_args())
                raise GeneratorExit()

        assert output_path.exists(), "Partial file should be retained on GeneratorExit"
        content = output_path.read_text()
        assert "rs12345" in content

    def test_flush_failure_on_normal_exit_raises(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """OSError during flush on normal exit propagates (not silently swallowed)."""
        output_path = tmp_path / "test.assoc.txt"

        with pytest.raises(OSError, match="Disk full"):
            with IncrementalAssocWriter(output_path) as writer:
                writer.write_arrays_batch(*sample_result.as_call_args())

                def failing_flush():
                    raise OSError("Disk full")

                open_handle(writer).flush = failing_flush

        # Partial file should be cleaned up
        assert not output_path.exists(), (
            "Partial output should be deleted after flush failure"
        )

    def test_keyboard_interrupt_with_close_failure(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """OSError during file.close() on interrupt does not mask KeyboardInterrupt."""
        output_path = tmp_path / "test.assoc.txt"

        with pytest.raises(KeyboardInterrupt):
            with IncrementalAssocWriter(output_path) as writer:
                writer.write_arrays_batch(*sample_result.as_call_args())
                original_close = open_handle(writer).close

                def failing_close():
                    original_close()
                    raise OSError("Stale NFS file handle")

                open_handle(writer).close = failing_close
                raise KeyboardInterrupt()

    def test_write_buf_retries_enospc_and_succeeds(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """_write_buf retries on ENOSPC flush failure and succeeds on second attempt.

        Patches _file.flush to raise OSError(ENOSPC) on the explicit post-write
        flush call inside _write_buf. TextIOWrapper.tell() also calls flush()
        internally, so the counter targets the second flush call (first = tell-induced,
        second = explicit post-write flush). ENOSPC is in _RETRYABLE_ERRNOS, so
        _write_buf must retry. Patches time.sleep to avoid actual delays.
        """
        output_path = tmp_path / "test.assoc.txt"
        # TextIOWrapper.tell() triggers an internal flush before returning position.
        # _write_buf sequence per attempt:
        #   tell() [flush #N] → write() → flush() [flush #N+1]
        # We want to fail the first explicit post-write flush: flush call #2.
        flush_call_count = 0
        failed_once = False

        with IncrementalAssocWriter(output_path) as writer:
            original_flush = open_handle(writer).flush

            def flaky_flush():
                nonlocal flush_call_count, failed_once
                flush_call_count += 1
                # Call 1: tell()-induced flush (let it succeed so pos is assigned).
                # Call 2: first explicit post-write flush — fail with ENOSPC.
                # Subsequent calls: succeed (retry path).
                if flush_call_count == 2 and not failed_once:
                    failed_once = True
                    err = OSError("No space left on device")
                    err.errno = errno.ENOSPC
                    raise err
                return original_flush()

            open_handle(writer).flush = flaky_flush

            with patch("jamma.lmm.io.time.sleep") as mock_sleep:
                writer.write_arrays_batch(*sample_result.as_call_args())
                # Should have slept once with first retry backoff delay
                mock_sleep.assert_called_once_with(0.1)

        assert writer.count == 1
        content = output_path.read_text()
        assert "rs12345" in content

    def test_write_buf_deletes_partial_on_non_retryable_rollback_failure(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """_write_buf deletes partial file when EPERM + rollback (seek) both fail.

        EPERM is not in _RETRYABLE_ERRNOS, so _write_buf breaks out immediately.
        Patching seek to also raise OSError simulates rollback failure, which
        triggers _cleanup_partial to delete the partial output file.
        """
        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep"):
            with pytest.raises(OSError):
                with IncrementalAssocWriter(output_path) as writer:
                    original_write = open_handle(writer).write

                    def eperm_write(data):
                        # Only fail on data lines (which contain tabs), not the header
                        if "\t" in data:
                            err = OSError("Operation not permitted")
                            err.errno = errno.EPERM
                            raise err
                        return original_write(data)

                    open_handle(writer).write = eperm_write

                    # Also make seek fail to trigger _cleanup_partial
                    def failing_seek(cookie: int, whence: int = 0) -> int:
                        raise OSError("Seek failed")

                    open_handle(writer).seek = failing_seek

                    writer.write_arrays_batch(*sample_result.as_call_args())

        # _cleanup_partial should have deleted the partial file
        assert not output_path.exists(), (
            "Partial output file should be deleted when EPERM + rollback both fail"
        )

    def test_write_buf_tell_failure_surfaces_as_oserror(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """An OSError from tell() propagates as OSError, not UnboundLocalError.

        tell() supplies the rollback position. Reading it inside the retry
        try-block left the position unbound when tell() itself failed, so the
        rollback in the except-handler raised UnboundLocalError -- which is not
        an OSError, so it escaped write()'s documented contract and skipped the
        partial-file cleanup entirely.
        """
        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep"):
            with pytest.raises(OSError, match="Illegal seek"):
                with IncrementalAssocWriter(output_path) as writer:

                    def failing_tell():
                        err = OSError("Illegal seek")
                        err.errno = errno.ESPIPE
                        raise err

                    open_handle(writer).tell = failing_tell

                    writer.write_arrays_batch(*sample_result.as_call_args())

        assert not output_path.exists(), (
            "Partial output file should be deleted when tell() fails"
        )

    def test_write_buf_rollback_discards_debris_before_retry(
        self, tmp_path: Path, sample_result: SampleBatch
    ):
        """A failed attempt is rolled back, so the retry writes over clean state.

        The first attempt puts more bytes into the stream than the retry will,
        then fails with a retryable errno. That is what the seek/truncate pair
        in _write_buf exists for, and each half is load-bearing:

        - Without ``seek(pos)``, ``truncate()`` cuts at the position left after
          the debris, so the retried line lands *after* it.
        - Without ``truncate()``, ``seek(pos)`` rewinds but the retried line is
          shorter than the debris, so the tail of the debris survives past the
          end of the line.

        Either way the file is malformed while the writer reports success, so
        this asserts the file contents rather than the call sequence.
        """
        output_path = tmp_path / "test.assoc.txt"
        debris = "X" * 500

        with patch("jamma.lmm.io.time.sleep"):
            with IncrementalAssocWriter(output_path) as writer:
                handle = open_handle(writer)
                original_write = handle.write
                data_attempts = 0

                def debris_then_succeed(data: str) -> int:
                    nonlocal data_attempts
                    if "\t" not in data:  # header, not a result line
                        return original_write(data)
                    data_attempts += 1
                    if data_attempts == 1:
                        # Longer than the real line, so a missing truncate
                        # leaves a tail behind that overwriting cannot cover.
                        original_write(debris)
                        raise OSError(errno.EIO, "I/O error")
                    return original_write(data)

                handle.write = debris_then_succeed
                writer.write_arrays_batch(*sample_result.as_call_args())

                assert data_attempts == 2, "expected exactly one retry"

        content = output_path.read_text()
        assert "X" not in content, (
            f"Rollback must discard the failed attempt's bytes; got {content!r}"
        )
        lines = content.strip().split("\n")
        assert len(lines) == 2, f"Expected header + one result, got {lines}"
        assert lines[0].startswith("chr\trs\tps")
        assert lines[1].startswith("1\trs12345\t")
        assert writer.count == 1
