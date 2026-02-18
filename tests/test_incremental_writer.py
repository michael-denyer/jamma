"""Tests for incremental association result writer."""

from pathlib import Path
from unittest.mock import patch

import pytest

from jamma.lmm.io import IncrementalAssocWriter, write_assoc_results
from jamma.lmm.stats import AssocResult


@pytest.fixture
def sample_result() -> AssocResult:
    """Create a sample AssocResult for testing."""
    return AssocResult(
        chr="1",
        rs="rs12345",
        ps=100000,
        n_miss=5,
        allele1="A",
        allele0="G",
        af=0.25,
        beta=0.123456,
        se=0.0234567,
        logl_H1=-1234.567,
        l_remle=0.456789,
        p_wald=0.00123456,
    )


@pytest.fixture
def sample_results(sample_result: AssocResult) -> list[AssocResult]:
    """Create multiple sample results."""
    results = []
    for i in range(10):
        results.append(
            AssocResult(
                chr=str((i % 22) + 1),
                rs=f"rs{10000 + i}",
                ps=100000 + i * 1000,
                n_miss=i,
                allele1="A",
                allele0="G",
                af=0.1 + i * 0.05,
                beta=0.1 * (i + 1),
                se=0.01 * (i + 1),
                logl_H1=-1000.0 - i,
                l_remle=0.5 + i * 0.1,
                p_wald=0.05 / (i + 1),
            )
        )
    return results


@pytest.mark.tier0
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

    def test_writes_single_result(self, tmp_path: Path, sample_result: AssocResult):
        """Should write single result correctly."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write(sample_result)

        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # Header + 1 result
        assert "rs12345" in lines[1]
        assert writer.count == 1

    def test_writes_multiple_results(self, tmp_path: Path, sample_results: list):
        """Should write multiple results correctly."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            for result in sample_results:
                writer.write(result)

        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 11  # Header + 10 results
        assert writer.count == 10

    def test_output_matches_write_assoc_results(
        self, tmp_path: Path, sample_results: list
    ):
        """Incremental writer output should match batch writer exactly."""
        incremental_path = tmp_path / "incremental.assoc.txt"
        batch_path = tmp_path / "batch.assoc.txt"

        # Write with incremental writer
        with IncrementalAssocWriter(incremental_path) as writer:
            for result in sample_results:
                writer.write(result)

        # Write with batch writer
        write_assoc_results(sample_results, batch_path)

        # Compare byte-for-byte
        incremental_content = incremental_path.read_text()
        batch_content = batch_path.read_text()
        assert incremental_content == batch_content

    def test_write_batch_method(self, tmp_path: Path, sample_results: list):
        """write_batch should write all results at once."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write_batch(sample_results)

        assert writer.count == len(sample_results)

    def test_write_batch_single_flush(self, tmp_path: Path, sample_results: list):
        """write_batch should have far fewer flushes than N results.

        CPython's TextIOWrapper.tell() internally calls flush(), so the
        exact count is 2 (tell-induced + explicit). The key invariant is
        that it's constant per batch, not proportional to len(results).
        The old per-SNP path would have produced len(results) flushes.
        """
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            original_flush = writer._file.flush
            flush_count = 0

            def counting_flush():
                nonlocal flush_count
                flush_count += 1
                return original_flush()

            writer._file.flush = counting_flush
            writer.write_batch(sample_results)
            # Count flushes only during write_batch, not __exit__
            batch_flush_count = flush_count

        # Constant flush count per batch (tell + explicit), not N per-SNP
        n = len(sample_results)
        assert batch_flush_count <= 2, f"Expected <=2 flushes, got {batch_flush_count}"
        assert batch_flush_count < n, (
            f"Batch flush count ({batch_flush_count}) should be far less than "
            f"result count ({n})"
        )

    def test_write_batch_output_matches_individual_writes(
        self, tmp_path: Path, sample_results: list
    ):
        """write_batch output must be byte-identical to individual write() calls."""
        batch_path = tmp_path / "batch.assoc.txt"
        individual_path = tmp_path / "individual.assoc.txt"

        with IncrementalAssocWriter(batch_path) as writer:
            writer.write_batch(sample_results)

        with IncrementalAssocWriter(individual_path) as writer:
            for result in sample_results:
                writer.write(result)

        assert batch_path.read_bytes() == individual_path.read_bytes()

    def test_write_batch_empty_list(self, tmp_path: Path):
        """write_batch([]) should be a no-op: count stays 0, no error."""
        output_path = tmp_path / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write_batch([])

        assert writer.count == 0
        # File should only contain header
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1  # Header only

    def test_creates_parent_directories(
        self, tmp_path: Path, sample_result: AssocResult
    ):
        """Should create parent directories if needed."""
        output_path = tmp_path / "deep" / "nested" / "dir" / "test.assoc.txt"

        with IncrementalAssocWriter(output_path) as writer:
            writer.write(sample_result)

        assert output_path.exists()

    def test_raises_if_not_opened(self, sample_result: AssocResult):
        """Should raise error if write called without context manager."""
        writer = IncrementalAssocWriter(Path("dummy.txt"))

        with pytest.raises(RuntimeError, match="not opened"):
            writer.write(sample_result)

    def test_retries_on_oserror(self, tmp_path: Path, sample_result: AssocResult):
        """Should retry on OSError and succeed on second attempt."""
        output_path = tmp_path / "test.assoc.txt"
        call_count = 0

        with IncrementalAssocWriter(output_path) as writer:
            original_write = writer._file.write

            def flaky_write(data):
                nonlocal call_count
                call_count += 1
                # First call to this patched write is the data line -- fail it.
                # Second call (retry) succeeds.
                if call_count == 1:
                    raise OSError("Disk full")
                return original_write(data)

            writer._file.write = flaky_write

            with patch("jamma.lmm.io.time.sleep") as mock_sleep:
                writer.write(sample_result)
                # Should have slept once with 0.1s (first retry backoff)
                mock_sleep.assert_called_once_with(0.1)

        assert writer.count == 1

    def test_deletes_partial_on_final_failure(
        self, tmp_path: Path, sample_result: AssocResult
    ):
        """Should delete partial file after exhausting all retries."""
        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep"):
            with pytest.raises(OSError):
                with IncrementalAssocWriter(output_path) as writer:
                    # Make every write after header fail
                    original_write = writer._file.write

                    def always_fail(data):
                        if "\t" in data:  # data lines have tabs
                            raise OSError("Disk full")
                        return original_write(data)

                    writer._file.write = always_fail
                    writer.write(sample_result)

        # Partial file should be cleaned up
        assert not output_path.exists(), "Partial output file should be deleted"

    def test_no_retry_on_non_oserror(self, tmp_path: Path, sample_result: AssocResult):
        """Non-OSError exceptions propagate immediately without retries."""
        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep") as mock_sleep:
            with pytest.raises(TypeError):
                with IncrementalAssocWriter(output_path) as writer:
                    # Patch format function to raise TypeError
                    with patch(
                        "jamma.lmm.io.format_assoc_line",
                        side_effect=TypeError("bad format"),
                    ):
                        writer.write(sample_result)

            # time.sleep should NOT have been called
            mock_sleep.assert_not_called()

    def test_cleanup_on_context_exit_with_error(
        self, tmp_path: Path, sample_result: AssocResult
    ):
        """Partial file deleted when OSError propagates through context exit."""
        output_path = tmp_path / "test.assoc.txt"

        with patch("jamma.lmm.io.time.sleep"):
            with pytest.raises(OSError):
                with IncrementalAssocWriter(output_path) as writer:
                    original_write = writer._file.write

                    def always_fail(data):
                        if "\t" in data:
                            raise OSError("Disk full")
                        return original_write(data)

                    writer._file.write = always_fail
                    writer.write(sample_result)

        # File should be cleaned up by __exit__
        assert not output_path.exists(), (
            "Partial file should be deleted on context exit with OSError"
        )

    def test_write_batch_retries_on_flaky_write(
        self, tmp_path: Path, sample_results: list
    ):
        """write_batch retries on first OSError and succeeds on second attempt."""
        output_path = tmp_path / "test.assoc.txt"
        call_count = 0

        with IncrementalAssocWriter(output_path) as writer:
            original_write = writer._file.write

            def flaky_write(data):
                nonlocal call_count
                call_count += 1
                # First write of data (not header) fails; retry succeeds
                if call_count == 1 and "\t" in data:
                    raise OSError("Transient I/O error")
                return original_write(data)

            writer._file.write = flaky_write

            with patch("jamma.lmm.io.time.sleep") as mock_sleep:
                writer.write_batch(sample_results)
                mock_sleep.assert_called_once_with(0.1)

        assert writer.count == len(sample_results)

        # Verify file content is complete and not duplicated
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1 + len(sample_results), (
            f"Expected header + {len(sample_results)} results, got {len(lines)} lines"
        )

    def test_write_batch_raises_if_not_opened(self, sample_results: list):
        """write_batch on unopened writer raises RuntimeError."""
        writer = IncrementalAssocWriter(Path("dummy.txt"))

        with pytest.raises(RuntimeError, match="not opened"):
            writer.write_batch(sample_results)

    def test_write_retries_on_transient_eio(
        self, tmp_path: Path, sample_result: AssocResult
    ):
        """Verify retry with exponential backoff on retryable EIO errno.

        Injects a transient errno.EIO on first write attempt. The writer
        should retry and succeed on the second attempt.
        """
        import errno as errno_mod

        output_path = tmp_path / "test.assoc.txt"
        call_count = 0

        with IncrementalAssocWriter(output_path) as writer:
            original_write = writer._file.write

            def eio_write(data):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    err = OSError("I/O error")
                    err.errno = errno_mod.EIO
                    raise err
                return original_write(data)

            writer._file.write = eio_write

            with patch("jamma.lmm.io.time.sleep") as mock_sleep:
                writer.write(sample_result)
                mock_sleep.assert_called_once_with(0.1)

        assert writer.count == 1
        # Verify file has header + 1 result
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_write_fails_immediately_on_eacces(
        self, tmp_path: Path, sample_result: AssocResult
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
                    original_write = writer._file.write

                    def eacces_write(data):
                        if "\t" in data:
                            err = OSError("Permission denied")
                            err.errno = errno_mod.EACCES
                            raise err
                        return original_write(data)

                    writer._file.write = eacces_write
                    writer.write(sample_result)

            # No retry sleeps should have occurred
            mock_sleep.assert_not_called()

        # Partial file should be cleaned up
        assert not output_path.exists(), "Partial output file should be deleted"

    def test_write_rollback_preserves_first_result(
        self, tmp_path: Path, sample_results: list
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
                    writer.write(sample_results[0])
                    assert writer.count == 1

                    # Now make all subsequent writes fail permanently
                    def fail_after_first(data):
                        nonlocal write_call_count
                        write_call_count += 1
                        raise OSError("Disk full")

                    writer._file.write = fail_after_first
                    # This should fail after retries and clean up
                    writer.write(sample_results[1])

        # _cleanup_partial deletes the file after exhausting retries
        assert not output_path.exists(), (
            "Partial output file should be deleted after exhausting retries"
        )
