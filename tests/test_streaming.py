"""Tests for streaming genotype I/O and memory estimation."""

from pathlib import Path

import numpy as np
import pytest

from jamma.core.memory import StreamingMemoryBreakdown, estimate_streaming_memory
from jamma.io.plink import (
    get_plink_metadata,
    load_plink_binary,
    stream_genotype_chunks,
)
from jamma.kinship.compute import compute_centered_kinship, compute_kinship_streaming


class TestStreamGenotypeChunks:
    """Tests for stream_genotype_chunks generator."""

    def test_yields_correct_shapes(self, sample_plink_data: Path) -> None:
        """Verify chunk dimensions match expected (n_samples, chunk_size)."""
        chunk_size = 5000
        chunks = list(
            stream_genotype_chunks(
                sample_plink_data, chunk_size=chunk_size, show_progress=False
            )
        )

        # First chunk should be full size
        assert chunks[0][0].shape == (1940, 5000)

        # All chunks except possibly last should have chunk_size SNPs
        for chunk, _start, _end in chunks[:-1]:
            assert chunk.shape[1] == chunk_size
            assert chunk.shape[0] == 1940

    def test_covers_all_snps(self, sample_plink_data: Path) -> None:
        """Verify no SNPs are missed - indices cover full range."""
        chunk_size = 5000
        chunks = list(
            stream_genotype_chunks(
                sample_plink_data, chunk_size=chunk_size, show_progress=False
            )
        )

        # Total SNPs from indices should match known count
        total_snps = sum(end - start for _, start, end in chunks)
        assert total_snps == 12226

        # Verify contiguous coverage
        expected_start = 0
        for _, start, end in chunks:
            assert start == expected_start
            expected_start = end

        # Final end should be total SNP count
        assert chunks[-1][2] == 12226

    def test_matches_full_load(self, sample_plink_data: Path) -> None:
        """Verify streamed data matches PlinkData full load."""
        # Load full data for reference
        full_data = load_plink_binary(sample_plink_data)

        # Stream and reassemble
        chunk_size = 4000  # Use non-divisor to test last chunk handling
        reassembled = []
        for chunk, _start, _end in stream_genotype_chunks(
            sample_plink_data, chunk_size=chunk_size, show_progress=False
        ):
            reassembled.append(chunk)

        reassembled = np.concatenate(reassembled, axis=1)

        # Should match exactly
        np.testing.assert_array_equal(
            reassembled,
            full_data.genotypes,
            err_msg="Streamed data should match full load",
        )

    def test_dtype_respected(self, sample_plink_data: Path) -> None:
        """Verify dtype parameter is honored."""
        # Default float32
        for chunk, _, _ in stream_genotype_chunks(
            sample_plink_data, chunk_size=12226, show_progress=False
        ):
            assert chunk.dtype == np.float32

        # Explicit float64
        for chunk, _, _ in stream_genotype_chunks(
            sample_plink_data, chunk_size=12226, dtype=np.float64, show_progress=False
        ):
            assert chunk.dtype == np.float64

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="PLINK .bed file not found"):
            list(stream_genotype_chunks(nonexistent, show_progress=False))


class TestGetPlinkMetadata:
    """Tests for get_plink_metadata function."""

    def test_returns_correct_dimensions(self, sample_plink_data: Path) -> None:
        """Verify n_samples and n_snps match known values."""
        meta = get_plink_metadata(sample_plink_data)

        assert meta["n_samples"] == 1940
        assert meta["n_snps"] == 12226

    def test_returns_all_fields(self, sample_plink_data: Path) -> None:
        """Verify all expected metadata fields are present."""
        meta = get_plink_metadata(sample_plink_data)

        expected_keys = {
            "n_samples",
            "n_snps",
            "iid",
            "sid",
            "chromosome",
            "bp_position",
            "allele_1",
            "allele_2",
        }
        assert set(meta.keys()) == expected_keys

    def test_metadata_lengths_match_dimensions(self, sample_plink_data: Path) -> None:
        """Verify metadata array lengths match n_samples/n_snps."""
        meta = get_plink_metadata(sample_plink_data)

        assert len(meta["iid"]) == meta["n_samples"]
        assert len(meta["sid"]) == meta["n_snps"]
        assert len(meta["chromosome"]) == meta["n_snps"]
        assert len(meta["bp_position"]) == meta["n_snps"]
        assert len(meta["allele_1"]) == meta["n_snps"]
        assert len(meta["allele_2"]) == meta["n_snps"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="PLINK .bed file not found"):
            get_plink_metadata(nonexistent)


class TestEstimateStreamingMemory:
    """Tests for estimate_streaming_memory function."""

    def test_returns_breakdown(self) -> None:
        """Verify function returns StreamingMemoryBreakdown with all fields."""
        est = estimate_streaming_memory(1_000, 10_000)

        assert isinstance(est, StreamingMemoryBreakdown)
        assert isinstance(est.kinship_gb, float)
        assert isinstance(est.eigenvectors_gb, float)
        assert isinstance(est.eigendecomp_workspace_gb, float)
        assert isinstance(est.chunk_gb, float)
        assert isinstance(est.rotation_buffer_gb, float)
        assert isinstance(est.total_peak_gb, float)
        assert isinstance(est.available_gb, float)
        assert isinstance(est.sufficient, bool)

    def test_peak_is_eigendecomp(self) -> None:
        """Verify peak memory is dominated by eigendecomp phase.

        For large n_samples, eigendecomp requires both kinship (input) and
        eigenvectors (output) simultaneously, which exceeds other phases.
        """
        est = estimate_streaming_memory(100_000, 95_000, chunk_size=10_000)

        # Eigendecomp peak: kinship + eigenvectors + workspace
        eigendecomp_peak = (
            est.kinship_gb + est.eigenvectors_gb + est.eigendecomp_workspace_gb
        )

        # Verify peak equals eigendecomp phase (within floating point tolerance)
        assert abs(est.total_peak_gb - eigendecomp_peak) < 1e-6, (
            f"Peak {est.total_peak_gb:.2f}GB should equal "
            f"eigendecomp {eigendecomp_peak:.2f}GB"
        )

    def test_200k_memory_estimate(self) -> None:
        """Verify memory estimates for 200k sample scale."""
        est = estimate_streaming_memory(200_000, 95_000, chunk_size=10_000)

        # Kinship: 200k^2 * 8 / 1e9 = 320GB
        assert (
            319 < est.kinship_gb < 321
        ), f"Expected ~320GB kinship, got {est.kinship_gb}"

        # Eigenvectors: same as kinship = 320GB
        assert 319 < est.eigenvectors_gb < 321

        # Workspace: O(n) ~50MB for 200k
        assert est.eigendecomp_workspace_gb < 1.0

        # Chunk: 200k * 10k * 8 / 1e9 = 16GB
        assert 15 < est.chunk_gb < 17, f"Expected ~16GB chunk, got {est.chunk_gb}"

        # Peak should be eigendecomp: ~640GB (kinship + eigenvectors)
        assert (
            600 < est.total_peak_gb < 700
        ), f"Expected ~640GB peak, got {est.total_peak_gb}"

    def test_chunk_size_affects_chunk_gb(self) -> None:
        """Verify chunk_size parameter affects chunk memory."""
        est_small = estimate_streaming_memory(100_000, 95_000, chunk_size=5_000)
        est_large = estimate_streaming_memory(100_000, 95_000, chunk_size=20_000)

        # Larger chunk size should require more chunk memory
        assert est_large.chunk_gb > est_small.chunk_gb
        assert abs(est_large.chunk_gb / est_small.chunk_gb - 4.0) < 0.01

    def test_n_snps_does_not_affect_peak(self) -> None:
        """Verify n_snps parameter doesn't affect peak memory.

        n_snps is only for logging; streaming memory doesn't scale with total SNPs.
        """
        est_few = estimate_streaming_memory(100_000, 10_000, chunk_size=10_000)
        est_many = estimate_streaming_memory(100_000, 1_000_000, chunk_size=10_000)

        # Peak should be identical regardless of n_snps
        assert est_few.total_peak_gb == est_many.total_peak_gb

    def test_sufficient_flag_correct(self) -> None:
        """Verify sufficient flag reflects available vs required."""
        # Tiny estimate should always be sufficient
        est = estimate_streaming_memory(100, 100)
        assert est.sufficient is True

        # 200k estimate requires ~640GB - not sufficient on typical machines
        est = estimate_streaming_memory(200_000, 95_000)
        assert (
            est.sufficient is False
        ), "200k sample workflow should exceed available memory"


class TestComputeKinshipStreaming:
    """Tests for compute_kinship_streaming function."""

    def test_compute_kinship_streaming_matches_full_load(
        self, sample_plink_data: Path
    ) -> None:
        """Verify streaming kinship matches full-load kinship computation."""
        # Load genotypes for full computation
        data = load_plink_binary(sample_plink_data)

        # Compute kinship via full-load method
        K_full = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Compute kinship via streaming
        K_stream = compute_kinship_streaming(
            sample_plink_data, chunk_size=5000, check_memory=False, show_progress=False
        )

        # Should match within numerical precision
        np.testing.assert_allclose(
            K_stream,
            K_full,
            rtol=1e-10,
            atol=1e-14,
            err_msg="Streaming kinship should match full-load kinship",
        )

    def test_compute_kinship_streaming_is_symmetric(
        self, sample_plink_data: Path
    ) -> None:
        """Verify streaming kinship produces symmetric matrix."""
        K = compute_kinship_streaming(
            sample_plink_data, chunk_size=4000, check_memory=False, show_progress=False
        )

        # Kinship matrix must be symmetric
        np.testing.assert_allclose(K, K.T, err_msg="Kinship matrix should be symmetric")

    def test_compute_kinship_streaming_different_chunk_sizes(
        self, sample_plink_data: Path
    ) -> None:
        """Verify different chunk sizes produce identical results."""
        chunk_sizes = [1000, 5000, 10000]

        results = [
            compute_kinship_streaming(
                sample_plink_data,
                chunk_size=cs,
                check_memory=False,
                show_progress=False,
            )
            for cs in chunk_sizes
        ]

        # All chunk sizes should produce identical kinship
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[i],
                results[0],
                rtol=1e-10,
                atol=1e-14,
                err_msg=f"Chunk size {chunk_sizes[i]} should match {chunk_sizes[0]}",
            )

    def test_compute_kinship_streaming_memory_check(
        self, sample_plink_data: Path
    ) -> None:
        """Verify memory check behavior."""
        # With check_memory=False, should always succeed
        K = compute_kinship_streaming(
            sample_plink_data, check_memory=False, show_progress=False
        )
        assert K.shape == (1940, 1940)

        # Mock low available memory to test MemoryError
        # For this small dataset, we don't actually expect MemoryError
        # Just verify the function works with check_memory=True
        K_checked = compute_kinship_streaming(
            sample_plink_data, check_memory=True, show_progress=False
        )
        assert K_checked.shape == (1940, 1940)

    def test_compute_kinship_streaming_missing_file_raises(
        self, tmp_path: Path
    ) -> None:
        """Verify FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="PLINK .bed file not found"):
            compute_kinship_streaming(nonexistent, show_progress=False)
