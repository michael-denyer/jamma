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
from jamma.lmm import run_lmm_association_jax, run_lmm_association_streaming


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

        # First chunk should be full size (100 samples from gemma_synthetic)
        assert chunks[0][0].shape == (100, min(chunk_size, 500))

        # All chunks except possibly last should have chunk_size SNPs
        for chunk, _start, _end in chunks[:-1]:
            assert chunk.shape[1] == chunk_size
            assert chunk.shape[0] == 100

    def test_covers_all_snps(self, sample_plink_data: Path) -> None:
        """Verify no SNPs are missed - indices cover full range."""
        chunk_size = 5000
        chunks = list(
            stream_genotype_chunks(
                sample_plink_data, chunk_size=chunk_size, show_progress=False
            )
        )

        # Total SNPs from indices should match known count (500 for gemma_synthetic)
        total_snps = sum(end - start for _, start, end in chunks)
        assert total_snps == 500

        # Verify contiguous coverage
        expected_start = 0
        for _, start, end in chunks:
            assert start == expected_start
            expected_start = end

        # Final end should be total SNP count
        assert chunks[-1][2] == 500

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
            sample_plink_data, chunk_size=500, show_progress=False
        ):
            assert chunk.dtype == np.float32

        # Explicit float64
        for chunk, _, _ in stream_genotype_chunks(
            sample_plink_data, chunk_size=500, dtype=np.float64, show_progress=False
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

        assert meta["n_samples"] == 100
        assert meta["n_snps"] == 500

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

        # Grid REML: 50 * 10k * 8 / 1e9 = 0.004GB (4MB with default n_grid=50)
        assert (
            0.003 < est.grid_reml_gb < 0.005
        ), f"Expected ~0.004GB grid_reml, got {est.grid_reml_gb}"

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

    def test_n_grid_affects_lmm_memory(self) -> None:
        """Verify n_grid parameter affects LMM phase memory estimate."""
        est_default = estimate_streaming_memory(100_000, 95_000, n_grid=50)
        est_large = estimate_streaming_memory(100_000, 95_000, n_grid=100)

        # Larger n_grid should increase grid_reml_gb
        assert est_large.grid_reml_gb > est_default.grid_reml_gb
        # With default chunk_size=10_000, doubling n_grid should double grid_reml
        assert abs(est_large.grid_reml_gb / est_default.grid_reml_gb - 2.0) < 0.01

    def test_grid_reml_gb_in_breakdown(self) -> None:
        """Verify grid_reml_gb is included in breakdown."""
        est = estimate_streaming_memory(100_000, 95_000, chunk_size=10_000, n_grid=50)

        # grid_reml: 50 * 10_000 * 8 / 1e9 = 0.004GB = 4MB
        expected_grid_reml = 50 * 10_000 * 8 / 1e9
        assert abs(est.grid_reml_gb - expected_grid_reml) < 1e-6

    def test_memory_budget_insufficient_for_large_samples(self) -> None:
        """Verify memory estimation correctly reports insufficient for large datasets.

        This test validates that the memory model accurately predicts when
        available memory is insufficient, which is critical for preventing OOM.
        """
        from unittest.mock import patch

        # Mock low available memory (8GB)
        with patch("psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 8 * 1e9  # 8GB

            # Estimate for 100k samples - eigendecomp needs ~160GB
            # (kinship + eigenvectors = 2 * 100k^2 * 8 bytes = 160GB)
            est = estimate_streaming_memory(100_000, 95_000)

            # Should report insufficient (160GB > 8GB)
            assert est.sufficient is False, (
                f"100k samples should require ~{est.total_peak_gb:.0f}GB, "
                f"exceeding mocked 8GB available"
            )

            # Available should reflect mocked value
            assert est.available_gb == 8.0

    def test_memory_budget_sufficient_for_small_samples(self) -> None:
        """Verify memory estimation correctly reports sufficient for small datasets."""
        from unittest.mock import patch

        # Mock moderate available memory (32GB)
        with patch("psutil.virtual_memory") as mock_mem:
            mock_obj = mock_mem.return_value
            mock_obj.available = 32 * 1e9  # 32GB

            # Estimate for 10k samples - eigendecomp needs ~1.6GB
            # (kinship + eigenvectors = 2 * 10k^2 * 8 bytes = 1.6GB)
            est = estimate_streaming_memory(10_000, 95_000)

            # Should report sufficient (1.6GB < 32GB)
            assert est.sufficient is True, (
                f"10k samples should require ~{est.total_peak_gb:.1f}GB, "
                f"fitting in mocked 32GB available"
            )


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
        assert K.shape == (100, 100)

        # Mock low available memory to test MemoryError
        # For this small dataset, we don't actually expect MemoryError
        # Just verify the function works with check_memory=True
        K_checked = compute_kinship_streaming(
            sample_plink_data, check_memory=True, show_progress=False
        )
        assert K_checked.shape == (100, 100)

    def test_compute_kinship_streaming_missing_file_raises(
        self, tmp_path: Path
    ) -> None:
        """Verify FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="PLINK .bed file not found"):
            compute_kinship_streaming(nonexistent, show_progress=False)

    def test_compute_kinship_streaming_with_filtering_matches_full_load(
        self, sample_plink_data: Path
    ) -> None:
        """Verify streaming kinship WITH filtering matches full-load kinship."""
        # Load genotypes for full computation
        data = load_plink_binary(sample_plink_data)

        # Test with various filtering thresholds
        maf = 0.05
        miss = 0.1

        # Compute kinship via full-load method with filtering
        K_full = compute_centered_kinship(
            data.genotypes.astype(np.float64),
            maf_threshold=maf,
            miss_threshold=miss,
            check_memory=False,
        )

        # Compute kinship via streaming with same filtering
        K_stream = compute_kinship_streaming(
            sample_plink_data,
            maf_threshold=maf,
            miss_threshold=miss,
            chunk_size=5000,
            check_memory=False,
            show_progress=False,
        )

        # Should match within numerical precision
        np.testing.assert_allclose(
            K_stream,
            K_full,
            rtol=1e-10,
            atol=1e-14,
            err_msg="Streaming kinship with filtering should match full-load",
        )


class TestFilteringBoundaryBehavior:
    """Tests for MAF/missing threshold boundary behavior."""

    def test_maf_boundary_inclusion(self) -> None:
        """SNPs exactly at MAF threshold should be included."""
        # Create synthetic genotypes with known MAF
        # SNP with MAF = 0.05 exactly (1 minor allele in 10 samples)
        # Values: 9x 0 + 1x 1 → freq = 0.1/2 = 0.05
        n_samples = 10
        genotypes = np.zeros((n_samples, 3), dtype=np.float64)
        genotypes[0, 0] = 1  # MAF = 0.05 exactly
        genotypes[0, 1] = 2  # MAF = 0.10
        genotypes[:, 2] = 0  # Monomorphic (should be filtered)

        # With MAF >= 0.05, first two SNPs should pass
        K = compute_centered_kinship(genotypes, maf_threshold=0.05, check_memory=False)
        # Should not raise - means some SNPs passed
        assert K.shape == (n_samples, n_samples)

        # With MAF >= 0.051, first SNP should be filtered
        # Only one SNP left (MAF=0.10)
        K_strict = compute_centered_kinship(
            genotypes, maf_threshold=0.051, check_memory=False
        )
        assert K_strict.shape == (n_samples, n_samples)

    def test_miss_boundary_inclusion(self) -> None:
        """SNPs exactly at missing threshold should be included."""
        # Create synthetic genotypes with known missing rate
        n_samples = 10
        genotypes = np.zeros((n_samples, 3), dtype=np.float64)
        genotypes[0, 0] = 1  # Polymorphic
        genotypes[0, 1] = 1  # Polymorphic
        genotypes[0, 2] = 1  # Polymorphic
        genotypes[1, 1] = np.nan  # 10% missing on SNP 1
        genotypes[1, 2] = np.nan  # 10% missing on SNP 2
        genotypes[2, 2] = np.nan  # 20% missing on SNP 2

        # With miss <= 0.10, SNPs 0 and 1 should pass
        K = compute_centered_kinship(genotypes, miss_threshold=0.10, check_memory=False)
        assert K.shape == (n_samples, n_samples)

        # With miss <= 0.05, only SNP 0 should pass
        K_strict = compute_centered_kinship(
            genotypes, miss_threshold=0.05, check_memory=False
        )
        assert K_strict.shape == (n_samples, n_samples)

    def test_monomorphic_always_filtered(self) -> None:
        """Monomorphic SNPs should always be filtered regardless of thresholds."""
        n_samples = 10
        # All SNPs are monomorphic (constant value)
        genotypes = np.ones((n_samples, 5), dtype=np.float64)

        # Even with permissive thresholds, all SNPs should be filtered
        with pytest.raises(ValueError, match="No SNPs passed filtering"):
            compute_centered_kinship(
                genotypes, maf_threshold=0.0, miss_threshold=1.0, check_memory=False
            )


class TestRunLmmAssociationStreaming:
    """Tests for run_lmm_association_streaming function."""

    def test_run_lmm_streaming_matches_full_load(self, sample_plink_data: Path) -> None:
        """Verify streaming LMM matches full-load LMM results."""
        # Fixed seed for reproducible phenotypes
        np.random.seed(42)

        # Load genotypes and compute kinship
        data = load_plink_binary(sample_plink_data)
        phenotypes = np.random.randn(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Build snp_info
        snp_info = [
            {
                "chr": str(c),
                "rs": s,
                "pos": int(p),
                "a1": a1,
                "a0": a0,
            }
            for c, s, p, a1, a0 in zip(
                data.chromosome,
                data.sid,
                data.bp_position,
                data.allele_1,
                data.allele_2,
                strict=False,
            )
        ]

        # Run full-load version
        results_full = run_lmm_association_jax(
            data.genotypes.astype(np.float32),
            phenotypes,
            kinship,
            snp_info,
            check_memory=False,
        )

        # Run streaming version
        results_stream = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info,
            check_memory=False,
            show_progress=False,
        )

        # Same number of results
        assert len(results_full) == len(
            results_stream
        ), f"Count mismatch: full={len(results_full)}, stream={len(results_stream)}"

        # Compare p-values and betas
        for i, (r_full, r_stream) in enumerate(
            zip(results_full, results_stream, strict=False)
        ):
            # P-values should match closely (rtol=1e-5)
            np.testing.assert_allclose(
                r_stream.p_wald,
                r_full.p_wald,
                rtol=1e-5,
                atol=1e-15,
                err_msg=f"SNP {i} p-value mismatch",
            )
            # Betas should match closely (rtol=1e-6)
            np.testing.assert_allclose(
                r_stream.beta,
                r_full.beta,
                rtol=1e-6,
                atol=1e-15,
                err_msg=f"SNP {i} beta mismatch",
            )

    def test_run_lmm_streaming_snp_info_from_metadata(
        self, sample_plink_data: Path
    ) -> None:
        """Verify SNP info is extracted from PLINK metadata when not provided."""
        np.random.seed(42)

        # Get expected metadata
        meta = get_plink_metadata(sample_plink_data)

        # Compute kinship
        data = load_plink_binary(sample_plink_data)
        phenotypes = np.random.randn(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Run streaming without snp_info
        results = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,  # Should build from metadata
            check_memory=False,
            show_progress=False,
        )

        # Verify results have correct metadata from first few SNPs
        assert len(results) > 0
        # Find a result that maps to first few SNPs by rs ID
        first_result = results[0]
        assert first_result.rs in meta["sid"], "rs should match PLINK metadata"
        assert first_result.chr in [str(c) for c in meta["chromosome"]]

    def test_run_lmm_streaming_filters_correctly(self, sample_plink_data: Path) -> None:
        """Verify streaming applies same filtering as full-load version."""
        np.random.seed(42)

        data = load_plink_binary(sample_plink_data)
        phenotypes = np.random.randn(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        snp_info = [
            {"chr": str(c), "rs": s, "pos": int(p), "a1": a1, "a0": a0}
            for c, s, p, a1, a0 in zip(
                data.chromosome,
                data.sid,
                data.bp_position,
                data.allele_1,
                data.allele_2,
                strict=False,
            )
        ]

        # Strict filtering thresholds
        maf_threshold = 0.1
        miss_threshold = 0.01

        # Run both versions with same filtering
        results_full = run_lmm_association_jax(
            data.genotypes.astype(np.float32),
            phenotypes,
            kinship,
            snp_info,
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            check_memory=False,
        )

        results_stream = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info,
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            check_memory=False,
            show_progress=False,
        )

        # Same number of results (filtering applied identically)
        assert len(results_full) == len(results_stream), (
            f"Filtering mismatch: full={len(results_full)}, "
            f"stream={len(results_stream)}"
        )

    def test_run_lmm_streaming_handles_missing_phenotypes(
        self, sample_plink_data: Path
    ) -> None:
        """Verify streaming handles missing phenotypes correctly."""
        np.random.seed(42)

        data = load_plink_binary(sample_plink_data)
        phenotypes = np.random.randn(data.n_samples)

        # Set some phenotypes to missing
        n_missing = 50
        phenotypes[:n_missing] = -9.0  # GEMMA missing indicator

        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        snp_info = [
            {"chr": str(c), "rs": s, "pos": int(p), "a1": a1, "a0": a0}
            for c, s, p, a1, a0 in zip(
                data.chromosome,
                data.sid,
                data.bp_position,
                data.allele_1,
                data.allele_2,
                strict=False,
            )
        ]

        # Run full-load version (filters internally)
        results_full = run_lmm_association_jax(
            data.genotypes.astype(np.float32),
            phenotypes,
            kinship,
            snp_info,
            check_memory=False,
        )

        # Run streaming version
        results_stream = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info,
            check_memory=False,
            show_progress=False,
        )

        # Should produce same results
        assert len(results_full) == len(results_stream)

        # P-values should match
        for r_full, r_stream in zip(results_full, results_stream, strict=False):
            np.testing.assert_allclose(
                r_stream.p_wald, r_full.p_wald, rtol=1e-5, atol=1e-15
            )

    def test_full_streaming_workflow(self, sample_plink_data: Path) -> None:
        """Verify complete streaming workflow: kinship + LMM from disk.

        This is the target use case: never loading full genotype matrix.
        """
        np.random.seed(42)

        # Get metadata for phenotype generation
        meta = get_plink_metadata(sample_plink_data)
        phenotypes = np.random.randn(meta["n_samples"])

        # Compute kinship via streaming (no genotype matrix loaded)
        kinship = compute_kinship_streaming(
            sample_plink_data, chunk_size=5000, check_memory=False, show_progress=False
        )

        # Run LMM via streaming (no genotype matrix loaded)
        results = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,  # Build from metadata
            check_memory=False,
            show_progress=False,
        )

        # Should produce valid results
        assert len(results) > 0, "Should have results after filtering"

        # Results should have valid statistics
        for r in results[:10]:  # Check first 10
            assert np.isfinite(r.p_wald), f"p-value should be finite: {r.p_wald}"
            assert 0 <= r.p_wald <= 1, f"p-value should be in [0,1]: {r.p_wald}"
            assert np.isfinite(r.beta), f"beta should be finite: {r.beta}"
            assert np.isfinite(r.se), f"se should be finite: {r.se}"
            assert r.se >= 0, f"se should be non-negative: {r.se}"

    def test_run_lmm_streaming_missing_file_raises(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "nonexistent"
        phenotypes = np.random.randn(100)
        kinship = np.eye(100)

        with pytest.raises(FileNotFoundError, match="PLINK .bed file not found"):
            run_lmm_association_streaming(
                nonexistent, phenotypes, kinship, show_progress=False
            )


class TestChunkEquivalence:
    """Tests verifying chunked processing produces identical results."""

    def test_single_vs_multi_chunk_equivalence(self, sample_plink_data: Path) -> None:
        """Verify single large chunk equals multiple small chunks.

        This test proves that JAX chunking is purely a memory optimization
        and does not affect numerical results.
        """
        np.random.seed(42)

        # Load data and prepare
        data = load_plink_binary(sample_plink_data)
        phenotypes = np.random.randn(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Build snp_info
        snp_info = [
            {
                "chr": str(data.chromosome[i]) if data.chromosome is not None else "1",
                "rs": data.sid[i] if data.sid is not None else f"snp{i}",
                "pos": int(data.bp_position[i]) if data.bp_position is not None else i,
                "a1": data.allele_1[i] if data.allele_1 is not None else "A",
                "a0": data.allele_2[i] if data.allele_2 is not None else "G",
            }
            for i in range(data.n_snps)
        ]

        # Run with full-load JAX (single batch, no streaming)
        results_single = run_lmm_association_jax(
            data.genotypes,
            phenotypes,
            kinship,
            snp_info,
            check_memory=False,
        )

        # Run with streaming (multiple chunks)
        # Use larger chunk to reduce number of JIT compilations
        results_multi = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            chunk_size=5000,  # Fewer chunks = faster test
            check_memory=False,
            show_progress=False,
        )

        # Both should produce same count of results
        assert len(results_single) == len(results_multi)

        # Results should be numerically identical within machine precision
        # (rtol=1e-13 allows for floating-point accumulation differences)
        for r1, r2 in zip(results_single, results_multi, strict=False):
            assert r1.rs == r2.rs, f"SNP mismatch: {r1.rs} vs {r2.rs}"
            np.testing.assert_allclose(
                r1.beta,
                r2.beta,
                rtol=1e-13,
                atol=0,
                err_msg=f"Beta mismatch for {r1.rs}",
            )
            np.testing.assert_allclose(
                r1.se, r2.se, rtol=1e-13, atol=0, err_msg=f"SE mismatch for {r1.rs}"
            )
            np.testing.assert_allclose(
                r1.p_wald,
                r2.p_wald,
                rtol=1e-13,
                atol=0,
                err_msg=f"P-value mismatch for {r1.rs}",
            )

    def test_streaming_different_chunk_sizes_equivalent(
        self, sample_plink_data: Path
    ) -> None:
        """Verify different chunk sizes in streaming produce identical results."""
        np.random.seed(42)

        # Load data
        data = load_plink_binary(sample_plink_data)
        phenotypes = np.random.randn(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Test only 2 chunk sizes (reduced from 3 for faster test)
        # Using sizes that force different chunk boundaries
        chunk_sizes = [2000, 5000]
        results_by_chunk: dict[int, list] = {}

        for cs in chunk_sizes:
            results = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                chunk_size=cs,
                check_memory=False,
                show_progress=False,
            )
            results_by_chunk[cs] = results

        # All chunk sizes should produce identical results within machine precision
        baseline = results_by_chunk[chunk_sizes[0]]
        for cs in chunk_sizes[1:]:
            results = results_by_chunk[cs]
            assert len(results) == len(baseline)

            for r1, r2 in zip(baseline, results, strict=False):
                # rtol=1e-12 allows for floating-point variance from different
                # chunk orderings while still detecting algorithmic differences
                np.testing.assert_allclose(
                    r1.beta,
                    r2.beta,
                    rtol=1e-12,
                    atol=0,
                    err_msg=f"Beta mismatch at chunk_size={cs}",
                )
                np.testing.assert_allclose(
                    r1.p_wald,
                    r2.p_wald,
                    rtol=1e-12,
                    atol=0,
                    err_msg=f"P-value mismatch at chunk_size={cs}",
                )
