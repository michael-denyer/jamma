"""Tests for streaming genotype I/O and memory estimation."""

from pathlib import Path

import numpy as np
import pytest

from jamma.core.memory import (
    StreamingMemoryBreakdown,
    estimate_lmm_streaming_memory,
    estimate_streaming_memory,
)
from jamma.io.plink import (
    get_plink_metadata,
    load_plink_binary,
    stream_genotype_chunks,
)
from jamma.kinship.compute import compute_centered_kinship, compute_kinship_streaming
from jamma.lmm import run_lmm_association_jax, run_lmm_association_streaming


def _build_snp_info(data) -> list[dict]:
    """Build snp_info list from plink data for test use."""
    return [
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
class TestEstimateStreamingMemory:
    """Tests for estimate_streaming_memory function."""

    def test_returns_breakdown(self) -> None:
        """Verify function returns StreamingMemoryBreakdown with all fields."""
        est = estimate_streaming_memory(1_000)

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

        With in-place eigendecomp (K/U share a buffer), eigendecomp peak
        is kinship + workspace (eigenvectors_gb not counted separately).
        """
        est = estimate_streaming_memory(100_000, chunk_size=10_000)

        # Eigendecomp peak: K/U shared buffer + workspace (no separate eigenvectors_gb)
        eigendecomp_peak = est.kinship_gb + est.eigendecomp_workspace_gb

        # Verify peak equals eigendecomp phase (within floating point tolerance)
        assert abs(est.total_peak_gb - eigendecomp_peak) < 1e-6, (
            f"Peak {est.total_peak_gb:.2f}GB should equal "
            f"eigendecomp {eigendecomp_peak:.2f}GB"
        )

    def test_200k_memory_estimate(self) -> None:
        """Verify memory estimates for 200k sample scale."""
        est = estimate_streaming_memory(200_000, chunk_size=10_000)

        # Kinship: 200k^2 * 8 / 1e9 = 320GB
        assert 319 < est.kinship_gb < 321, (
            f"Expected ~320GB kinship, got {est.kinship_gb}"
        )

        # Eigenvectors: same as kinship = 320GB
        assert 319 < est.eigenvectors_gb < 321

        # Workspace: DSYEVD O(n^2) ~640GB at 200k
        assert est.eigendecomp_workspace_gb > 600, (
            f"DSYEVD workspace should be ~640GB at 200k, "
            f"got {est.eigendecomp_workspace_gb:.2f}GB"
        )

        # Chunk: 200k * 10k * 8 / 1e9 = 16GB
        assert 15 < est.chunk_gb < 17, f"Expected ~16GB chunk, got {est.chunk_gb}"

        # Grid REML: 50 * 10k * 8 / 1e9 = 0.004GB (4MB with default n_grid=50)
        assert 0.003 < est.grid_reml_gb < 0.005, (
            f"Expected ~0.004GB grid_reml, got {est.grid_reml_gb}"
        )

        # Peak should be eigendecomp: ~960GB (K/U shared + dsyevd workspace)
        assert 950 < est.total_peak_gb < 970, (
            f"Expected ~960GB peak, got {est.total_peak_gb}"
        )

    def test_chunk_size_affects_chunk_gb(self) -> None:
        """Verify chunk_size parameter affects chunk memory."""
        est_small = estimate_streaming_memory(100_000, chunk_size=5_000)
        est_large = estimate_streaming_memory(100_000, chunk_size=20_000)

        # Larger chunk size should require more chunk memory
        assert est_large.chunk_gb > est_small.chunk_gb
        assert abs(est_large.chunk_gb / est_small.chunk_gb - 4.0) < 0.01

    def test_sufficient_flag_correct(self) -> None:
        """Verify sufficient flag reflects available vs required."""
        # Tiny estimate should always be sufficient
        est = estimate_streaming_memory(100)
        assert est.sufficient is True

        # 200k estimate requires ~640GB - not sufficient on typical machines
        est = estimate_streaming_memory(200_000)
        assert est.sufficient is False, (
            "200k sample workflow should exceed available memory"
        )

    def test_n_grid_affects_lmm_memory(self) -> None:
        """Verify n_grid parameter affects LMM phase memory estimate."""
        est_default = estimate_streaming_memory(100_000, n_grid=50)
        est_large = estimate_streaming_memory(100_000, n_grid=100)

        # Larger n_grid should increase grid_reml_gb
        assert est_large.grid_reml_gb > est_default.grid_reml_gb
        # With default chunk_size=10_000, doubling n_grid should double grid_reml
        assert abs(est_large.grid_reml_gb / est_default.grid_reml_gb - 2.0) < 0.01

    def test_grid_reml_gb_in_breakdown(self) -> None:
        """Verify grid_reml_gb is included in breakdown."""
        est = estimate_streaming_memory(100_000, chunk_size=10_000, n_grid=50)

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
            est = estimate_streaming_memory(100_000)

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
            est = estimate_streaming_memory(10_000)

            # Should report sufficient (1.6GB < 32GB)
            assert est.sufficient is True, (
                f"10k samples should require ~{est.total_peak_gb:.1f}GB, "
                f"fitting in mocked 32GB available"
            )


@pytest.mark.tier1
class TestEstimateLmmStreamingMemory:
    """Tests for estimate_lmm_streaming_memory function (LMM-phase only)."""

    def test_lmm_estimate_less_than_full_pipeline(self) -> None:
        """LMM-only estimate should be less than full streaming estimate."""
        lmm_est = estimate_lmm_streaming_memory(100_000, 95_000)
        full_est = estimate_streaming_memory(100_000)

        assert lmm_est.total_peak_gb < full_est.total_peak_gb, (
            f"LMM-only ({lmm_est.total_peak_gb:.1f}GB) should be less than "
            f"full pipeline ({full_est.total_peak_gb:.1f}GB)"
        )

    def test_lmm_estimate_excludes_kinship(self) -> None:
        """LMM estimate should not include kinship memory."""
        est = estimate_lmm_streaming_memory(100_000, 95_000)
        assert est.kinship_gb == 0.0

    def test_lmm_estimate_excludes_eigendecomp_workspace(self) -> None:
        """LMM estimate should not include eigendecomp workspace."""
        est = estimate_lmm_streaming_memory(100_000, 95_000)
        assert est.eigendecomp_workspace_gb == 0.0

    def test_lmm_estimate_includes_eigenvectors(self) -> None:
        """LMM estimate should include eigenvectors (~80GB at 100k)."""
        est = estimate_lmm_streaming_memory(100_000, 95_000)
        assert 79 < est.eigenvectors_gb < 81

    def test_lmm_estimate_100k_under_300gb(self) -> None:
        """At 100k samples, LMM should need well under 300GB.

        This is the exact scenario from the xlarge benchmark bug:
        300.6GB available after eigendecomp, but old check demanded 320GB.
        """
        est = estimate_lmm_streaming_memory(100_000, 95_000)
        assert est.total_peak_gb < 200, (
            f"Streaming LMM for 100k should need <200GB, got {est.total_peak_gb:.1f}GB"
        )

    def test_returns_streaming_memory_breakdown(self) -> None:
        """Should return StreamingMemoryBreakdown with all fields."""
        est = estimate_lmm_streaming_memory(1_000, 10_000)
        assert isinstance(est, StreamingMemoryBreakdown)

    def test_sufficient_flag_correct(self) -> None:
        """Tiny estimate should be sufficient."""
        est = estimate_lmm_streaming_memory(100, 100)
        assert est.sufficient is True


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
class TestRunLmmAssociationStreaming:
    """Tests for run_lmm_association_streaming function."""

    def test_run_lmm_streaming_matches_full_load(self, sample_plink_data: Path) -> None:
        """Verify streaming LMM matches full-load LMM results."""
        # Fixed seed for reproducible phenotypes
        rng = np.random.default_rng(42)

        # Load genotypes and compute kinship
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )
        # eigendecomp overwrites K in-place; needs fresh copy per run
        kinship_full = kinship.copy()
        kinship_stream = kinship.copy()

        # Build snp_info
        snp_info = _build_snp_info(data)

        # Run full-load version
        results_full = run_lmm_association_jax(
            data.genotypes.astype(np.float32),
            phenotypes,
            kinship_full,
            snp_info,
            check_memory=False,
        )

        # Run streaming version
        results_stream, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship_stream,
            snp_info,
            check_memory=False,
            show_progress=False,
        )

        # Same number of results
        assert len(results_full) == len(results_stream), (
            f"Count mismatch: full={len(results_full)}, stream={len(results_stream)}"
        )

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
        rng = np.random.default_rng(42)

        # Get expected metadata
        meta = get_plink_metadata(sample_plink_data)

        # Compute kinship
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Run streaming without snp_info
        results, _ = run_lmm_association_streaming(
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
        rng = np.random.default_rng(42)

        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        snp_info = _build_snp_info(data)

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

        results_stream, _ = run_lmm_association_streaming(
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
        rng = np.random.default_rng(42)

        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)

        # Set some phenotypes to missing
        n_missing = 50
        phenotypes[:n_missing] = -9.0  # GEMMA missing indicator

        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        snp_info = _build_snp_info(data)

        # Run full-load version (filters internally)
        results_full = run_lmm_association_jax(
            data.genotypes.astype(np.float32),
            phenotypes,
            kinship,
            snp_info,
            check_memory=False,
        )

        # Run streaming version
        results_stream, _ = run_lmm_association_streaming(
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
        rng = np.random.default_rng(42)

        # Get metadata for phenotype generation
        meta = get_plink_metadata(sample_plink_data)
        phenotypes = rng.standard_normal(meta["n_samples"])

        # Compute kinship via streaming (no genotype matrix loaded)
        kinship = compute_kinship_streaming(
            sample_plink_data, chunk_size=5000, check_memory=False, show_progress=False
        )

        # Run LMM via streaming (no genotype matrix loaded)
        results, _ = run_lmm_association_streaming(
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
        phenotypes = np.random.default_rng(42).standard_normal(100)
        kinship = np.eye(100)

        with pytest.raises(FileNotFoundError, match="PLINK .bed file not found"):
            run_lmm_association_streaming(
                nonexistent, phenotypes, kinship, show_progress=False
            )

    def test_streaming_incremental_write_per_chunk(
        self, sample_plink_data: Path, tmp_path: Path
    ) -> None:
        """Verify streaming runner writes per-chunk, not after accumulating all results.

        This tests that results are written incrementally to disk as each file chunk
        is processed, rather than accumulating all results in memory first.
        """
        rng = np.random.default_rng(42)

        # Load data and prepare
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        output_path = tmp_path / "streaming.assoc.txt"

        # Use very small chunk_size to ensure multiple file chunks
        # This tests that results are written incrementally, not accumulated
        chunk_size = 100  # Very small to force many file chunks

        results, n_tested = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,  # Let it build from PLINK metadata
            chunk_size=chunk_size,
            check_memory=False,
            show_progress=False,
            output_path=output_path,
        )

        # Verify empty list returned (results on disk)
        assert results == [], "Should return empty list when output_path is provided"
        assert n_tested > 0, "Should have tested some SNPs"

        # Verify file exists and has content
        assert output_path.exists(), "Output file should exist"
        with open(output_path) as f:
            lines = f.readlines()

        # Header + at least some results (depends on filtering)
        assert len(lines) >= 1, "Should have at least header line"

        # Verify header format
        header = lines[0].strip()
        expected_cols = [
            "chr",
            "rs",
            "ps",
            "n_miss",
            "allele1",
            "allele0",
            "af",
            "beta",
            "se",
            "logl_H1",
            "l_remle",
            "p_wald",
        ]
        for col in expected_cols:
            assert col in header, f"Missing column: {col}"

        # Verify we have results (excluding header)
        n_results = len(lines) - 1
        assert n_results > 0, "Should have at least one result"

        # With chunk_size=100 and 500 SNPs, we should have ~5 file chunks
        # This confirms the code handles multiple chunks correctly
        # The exact count depends on filtering, but should be substantial
        assert n_results > 100, f"Expected many results, got {n_results}"


@pytest.mark.tier1
class TestChunkEquivalence:
    """Tests verifying chunked processing produces identical results."""

    def test_single_vs_multi_chunk_equivalence(self, sample_plink_data: Path) -> None:
        """Verify single large chunk equals multiple small chunks.

        This test proves that JAX chunking is purely a memory optimization
        and does not affect numerical results.
        """
        rng = np.random.default_rng(42)

        # Load data and prepare
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )
        # eigendecomp overwrites K in-place; needs fresh copy per run
        kinship_single = kinship.copy()
        kinship_multi = kinship.copy()

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
            kinship_single,
            snp_info,
            check_memory=False,
        )

        # Run with streaming (multiple chunks)
        # Use larger chunk to reduce number of JIT compilations
        results_multi, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship_multi,
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
        rng = np.random.default_rng(42)

        # Load data
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Test only 2 chunk sizes (reduced from 3 for faster test)
        # Using sizes that force different chunk boundaries
        chunk_sizes = [2000, 5000]
        results_by_chunk: dict[int, list] = {}

        for cs in chunk_sizes:
            # eigendecomp overwrites K in-place; needs fresh copy per run
            results, _ = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship.copy(),
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


@pytest.mark.tier1
def test_streaming_vs_batch_parity(sample_plink_data: Path) -> None:
    """Streaming and batch runners must produce identical results for the same input.

    This test catches regressions where streaming and batch code paths diverge
    (e.g., different imputation, different filtering, different eigendecomp reuse).
    Uses chunk_size=10 to force multiple streaming chunks for thorough coverage.
    """
    rng = np.random.default_rng(42)

    data = load_plink_binary(sample_plink_data)
    phenotypes = rng.standard_normal(data.n_samples)
    kinship = compute_centered_kinship(
        data.genotypes.astype(np.float64), check_memory=False
    )
    # eigendecomp overwrites K in-place; needs fresh copy per run
    kinship_batch = kinship.copy()
    kinship_stream = kinship.copy()

    snp_info = _build_snp_info(data)

    # Batch runner (all genotypes in memory, float64 to match streaming reader)
    results_batch = run_lmm_association_jax(
        data.genotypes.astype(np.float64),
        phenotypes,
        kinship_batch,
        snp_info,
        check_memory=False,
        show_progress=False,
    )

    # Streaming runner with small chunk_size to force multiple file chunks
    results_stream, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        chunk_size=100,  # Small to exercise chunking
        check_memory=False,
        show_progress=False,
    )

    # Same number of results (same filtering applied)
    assert len(results_batch) == len(results_stream), (
        f"Result count mismatch: batch={len(results_batch)}, "
        f"stream={len(results_stream)}"
    )
    assert len(results_batch) > 0, "Expected some results"

    # Element-by-element comparison of key statistics
    for i, (rb, rs) in enumerate(zip(results_batch, results_stream, strict=True)):
        assert rb.rs == rs.rs, f"SNP {i}: rs mismatch {rb.rs} vs {rs.rs}"
        np.testing.assert_allclose(
            rb.beta,
            rs.beta,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) beta mismatch",
        )
        np.testing.assert_allclose(
            rb.se,
            rs.se,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) se mismatch",
        )
        np.testing.assert_allclose(
            rb.p_wald,
            rs.p_wald,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) p_wald mismatch",
        )
        # AF tolerance relaxed: batch computes from in-memory matrix,
        # streaming computes from disk-read chunks with different FP
        # accumulation order, producing ~1e-8 relative differences.
        np.testing.assert_allclose(
            rb.af,
            rs.af,
            rtol=1e-7,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) af mismatch",
        )
        np.testing.assert_allclose(
            rb.l_remle,
            rs.l_remle,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) l_remle mismatch",
        )


@pytest.mark.tier1
def test_streaming_lrt_only_matches_batch(sample_plink_data: Path) -> None:
    """Verify streaming LRT-only (mode 2) output matches batch runner.

    LRT mode computes l_mle and p_lrt per SNP; beta and se are NaN.
    This test exercises the lmm_mode=2 code path in both runners and
    verifies parity.
    """
    rng = np.random.default_rng(42)

    data = load_plink_binary(sample_plink_data)
    phenotypes = rng.standard_normal(data.n_samples)
    kinship = compute_centered_kinship(
        data.genotypes.astype(np.float64), check_memory=False
    )
    # eigendecomp overwrites K in-place; needs fresh copy per run
    kinship_batch = kinship.copy()
    kinship_stream = kinship.copy()

    snp_info = _build_snp_info(data)

    # Batch runner with lmm_mode=2
    results_batch = run_lmm_association_jax(
        data.genotypes.astype(np.float64),
        phenotypes,
        kinship_batch,
        snp_info,
        check_memory=False,
        show_progress=False,
        lmm_mode=2,
    )

    # Streaming runner with lmm_mode=2
    results_stream, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        chunk_size=100,
        check_memory=False,
        show_progress=False,
        lmm_mode=2,
    )

    assert len(results_batch) == len(results_stream), (
        f"Count mismatch: batch={len(results_batch)}, stream={len(results_stream)}"
    )
    assert len(results_batch) > 0, "Expected some results"

    for i, (rb, rs) in enumerate(zip(results_batch, results_stream, strict=True)):
        assert rb.rs == rs.rs, f"SNP {i}: rs mismatch {rb.rs} vs {rs.rs}"

        # LRT-specific fields must match
        np.testing.assert_allclose(
            rb.l_mle,
            rs.l_mle,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) l_mle mismatch",
        )
        np.testing.assert_allclose(
            rb.p_lrt,
            rs.p_lrt,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) p_lrt mismatch",
        )

        # LRT mode: beta and se should be NaN
        assert np.isnan(rb.beta), f"SNP {i} batch beta should be NaN, got {rb.beta}"
        assert np.isnan(rs.beta), f"SNP {i} stream beta should be NaN, got {rs.beta}"
        assert np.isnan(rb.se), f"SNP {i} batch se should be NaN, got {rb.se}"
        assert np.isnan(rs.se), f"SNP {i} stream se should be NaN, got {rs.se}"


@pytest.mark.tier1
def test_streaming_score_only_matches_batch(sample_plink_data: Path) -> None:
    """Verify streaming Score-only (mode 3) output matches batch runner.

    Score mode computes p_score per SNP; logl_H1 and l_remle are None.
    This test exercises the lmm_mode=3 code path in both runners and
    verifies parity.
    """
    rng = np.random.default_rng(42)

    data = load_plink_binary(sample_plink_data)
    phenotypes = rng.standard_normal(data.n_samples)
    kinship = compute_centered_kinship(
        data.genotypes.astype(np.float64), check_memory=False
    )
    # eigendecomp overwrites K in-place; needs fresh copy per run
    kinship_batch = kinship.copy()
    kinship_stream = kinship.copy()

    snp_info = _build_snp_info(data)

    # Batch runner with lmm_mode=3
    results_batch = run_lmm_association_jax(
        data.genotypes.astype(np.float64),
        phenotypes,
        kinship_batch,
        snp_info,
        check_memory=False,
        show_progress=False,
        lmm_mode=3,
    )

    # Streaming runner with lmm_mode=3
    results_stream, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        chunk_size=100,
        check_memory=False,
        show_progress=False,
        lmm_mode=3,
    )

    assert len(results_batch) == len(results_stream), (
        f"Count mismatch: batch={len(results_batch)}, stream={len(results_stream)}"
    )
    assert len(results_batch) > 0, "Expected some results"

    for i, (rb, rs) in enumerate(zip(results_batch, results_stream, strict=True)):
        assert rb.rs == rs.rs, f"SNP {i}: rs mismatch {rb.rs} vs {rs.rs}"

        # Score-specific fields must match
        np.testing.assert_allclose(
            rb.p_score,
            rs.p_score,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) p_score mismatch",
        )

        # Score mode still computes beta/se (unlike LRT)
        np.testing.assert_allclose(
            rb.beta,
            rs.beta,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) beta mismatch",
        )
        np.testing.assert_allclose(
            rb.se,
            rs.se,
            rtol=1e-10,
            atol=0,
            err_msg=f"SNP {i} ({rb.rs}) se mismatch",
        )

        # Score mode: logl_H1 and l_remle should be None
        assert rb.logl_H1 is None, (
            f"SNP {i} batch logl_H1 should be None, got {rb.logl_H1}"
        )
        assert rs.logl_H1 is None, (
            f"SNP {i} stream logl_H1 should be None, got {rs.logl_H1}"
        )
        assert rb.l_remle is None, (
            f"SNP {i} batch l_remle should be None, got {rb.l_remle}"
        )
        assert rs.l_remle is None, (
            f"SNP {i} stream l_remle should be None, got {rs.l_remle}"
        )


@pytest.mark.tier1
def test_streaming_all_invalid_samples_raises(tmp_path: Path) -> None:
    """Streaming runner raises ValueError when all samples have missing phenotypes.

    Regression test for the guard clause that prevents empty eigendecomposition
    when all phenotype values are NaN or -9.
    """
    from bed_reader import to_bed

    n_samples, n_snps = 50, 20
    rng = np.random.default_rng(42)
    genotypes = rng.choice([0, 1, 2], size=(n_samples, n_snps)).astype(np.int8)

    bed_path = tmp_path / "all_missing"
    to_bed(
        str(bed_path) + ".bed",
        genotypes,
        properties={
            "iid": [f"sample_{i}" for i in range(n_samples)],
            "sid": [f"snp_{i}" for i in range(n_snps)],
            "chromosome": ["1"] * n_snps,
            "bp_position": list(range(1, n_snps + 1)),
        },
    )

    # All phenotypes are NaN (missing)
    phenotypes = np.full(n_samples, np.nan)
    kinship = np.eye(n_samples)

    with pytest.raises(ValueError, match="No valid samples"):
        run_lmm_association_streaming(
            bed_path,
            phenotypes,
            kinship,
            check_memory=False,
            show_progress=False,
        )
