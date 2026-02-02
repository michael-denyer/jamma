"""Tests for incremental association result writer."""

from pathlib import Path

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
