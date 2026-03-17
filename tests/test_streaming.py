"""Tests for streaming genotype I/O and memory estimation."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from jamma.core.memory import (
    StreamingMemoryBreakdown,
    estimate_lmm_streaming_memory,
    estimate_streaming_memory,
)
from jamma.io.plink import (
    get_plink_metadata,
    load_plink_binary,
    prefetch_iterator,
    stream_genotype_chunks,
)
from jamma.kinship.compute import compute_centered_kinship, compute_kinship_streaming
from jamma.lmm import run_lmm_association_jax, run_lmm_association_streaming

pytestmark = pytest.mark.requires_jax


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


def _assert_results_match(
    results_a: list,
    results_b: list,
    fields: tuple[str, ...] = ("beta", "se", "p_wald"),
    rtol: float = 1e-10,
    atol: float = 0,
) -> None:
    """Assert two AssocResult lists match on the given fields."""
    assert len(results_a) == len(results_b), (
        f"Count mismatch: {len(results_a)} vs {len(results_b)}"
    )
    assert len(results_a) > 0, "Expected some results"

    for i, (ra, rb) in enumerate(zip(results_a, results_b, strict=True)):
        assert ra.rs == rb.rs, f"SNP {i}: rs mismatch {ra.rs} vs {rb.rs}"
        for field in fields:
            np.testing.assert_allclose(
                getattr(ra, field),
                getattr(rb, field),
                rtol=rtol,
                atol=atol,
                err_msg=f"SNP {i} ({ra.rs}) {field} mismatch",
            )


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

    def test_peak_phase_correct(self) -> None:
        """Verify eigendecomp is peak phase (DSYEVD workspace dominates).

        Estimates default to DSYEVD O(N^2) workspace. At 100k samples:
        eigendecomp peak = K (80GB) + DSYEVD workspace (~160GB) = ~240GB,
        which dominates streaming LMM (~144GB).
        """
        est = estimate_streaming_memory(100_000, chunk_size=10_000)

        # DSYEVD in-place: K/U shared + O(N^2) workspace
        eigendecomp_peak = est.kinship_gb + est.eigendecomp_workspace_gb

        # Total should be at least the eigendecomp peak
        assert est.total_peak_gb >= eigendecomp_peak - 1e-6, (
            f"Peak {est.total_peak_gb:.2f}GB should be >= "
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

        # Eigendecomp workspace: always DSYEVD O(n^2) ~640GB at 200k
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

        # Peak: K + U + K_work + DSYEVD workspace = 320+320+320+640 = ~1600GB
        assert 1590 < est.total_peak_gb < 1610, (
            f"Expected ~1600GB peak (K+U+K_work+workspace), got {est.total_peak_gb}"
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

    def test_lmm_estimate_at_most_full_pipeline(self) -> None:
        """LMM-only estimate should be <= full streaming estimate.

        With DSYEVR, eigendecomp workspace is tiny so streaming LMM phase
        dominates — LMM-only equals full pipeline. With DSYEVD, eigendecomp
        dominates so LMM-only is strictly less.
        """
        lmm_est = estimate_lmm_streaming_memory(100_000, 95_000)
        full_est = estimate_streaming_memory(100_000)

        assert lmm_est.total_peak_gb <= full_est.total_peak_gb, (
            f"LMM-only ({lmm_est.total_peak_gb:.1f}GB) should be <= "
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
        run_result = run_lmm_association_jax(
            data.genotypes.astype(np.float32),
            phenotypes,
            kinship_full,
            snp_info,
            check_memory=False,
        )
        results_full = run_result.associations

        # Run streaming version
        run_result, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship_stream,
            snp_info,
            check_memory=False,
            show_progress=False,
        )
        results_stream = run_result.associations

        # PVE should be populated
        assert run_result.pve is not None, "Streaming runner should return PVE"
        assert 0 < run_result.pve < 1, f"PVE out of range: {run_result.pve}"
        assert run_result.pve_se is None or run_result.pve_se > 0, (
            f"PVE SE should be None or positive, got {run_result.pve_se}"
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
        run_result, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,  # Should build from metadata
            check_memory=False,
            show_progress=False,
        )
        results = run_result.associations

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
        run_result = run_lmm_association_jax(
            data.genotypes.astype(np.float32),
            phenotypes,
            kinship,
            snp_info,
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            check_memory=False,
        )
        results_full = run_result.associations

        run_result, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info,
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            check_memory=False,
            show_progress=False,
        )
        results_stream = run_result.associations

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
        run_result = run_lmm_association_jax(
            data.genotypes.astype(np.float32),
            phenotypes,
            kinship,
            snp_info,
            check_memory=False,
        )
        results_full = run_result.associations

        # Run streaming version
        run_result, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info,
            check_memory=False,
            show_progress=False,
        )
        results_stream = run_result.associations

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
        run_result, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,  # Build from metadata
            check_memory=False,
            show_progress=False,
        )
        results = run_result.associations

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

        run_result, n_tested = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,  # Let it build from PLINK metadata
            chunk_size=chunk_size,
            check_memory=False,
            show_progress=False,
            output_path=output_path,
        )
        results = run_result.associations

        # Verify empty list returned (results on disk)
        assert len(results) == 0, (
            "Should return empty results when output_path is provided"
        )
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

    def test_streaming_output_flushes_each_jax_subchunk(
        self, sample_plink_data: Path, tmp_path: Path
    ) -> None:
        """Disk-write path flushes each JAX sub-chunk instead of batching on device."""
        import math
        from unittest.mock import patch

        from jamma.lmm.io import IncrementalAssocWriter

        rng = np.random.default_rng(123)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        batch_sizes: list[int] = []
        original_write_arrays_batch = IncrementalAssocWriter.write_arrays_batch

        def _spy_write_arrays_batch(
            self,
            lmm_mode,
            snp_indices,
            snp_info,
            afs,
            miss_counts,
            arrays,
        ):
            batch_sizes.append(len(snp_indices))
            return original_write_arrays_batch(
                self, lmm_mode, snp_indices, snp_info, afs, miss_counts, arrays
            )

        with (
            patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=50),
            patch.object(
                IncrementalAssocWriter,
                "write_arrays_batch",
                new=_spy_write_arrays_batch,
            ),
        ):
            run_result, n_tested = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info=None,
                chunk_size=500,
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                output_path=tmp_path / "subchunks.assoc.txt",
            )

        assert run_result.associations == []
        assert n_tested > 50, "Need multiple JAX sub-chunks for this regression test"
        assert len(batch_sizes) > 1, "Expected multiple sub-chunk writes"
        assert sum(batch_sizes) == n_tested
        assert len(batch_sizes) == math.ceil(n_tested / 50)

    def test_streaming_in_memory_materializes_each_jax_subchunk(
        self, sample_plink_data: Path
    ) -> None:
        """In-memory path converts each JAX sub-chunk before advancing."""
        import math
        from unittest.mock import patch

        from jamma.lmm import runner_streaming

        rng = np.random.default_rng(321)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        chunk_sizes: list[int] = []
        original_yield_chunk_results = runner_streaming._yield_chunk_results

        def _spy_yield_chunk_results(
            lmm_mode,
            filtered_indices,
            snp_indices,
            filtered_afs,
            filtered_miss,
            snp_info,
            arrays,
        ):
            chunk_sizes.append(len(filtered_indices))
            yield from original_yield_chunk_results(
                lmm_mode,
                filtered_indices,
                snp_indices,
                filtered_afs,
                filtered_miss,
                snp_info,
                arrays,
            )

        with (
            patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=50),
            patch.object(
                runner_streaming,
                "_yield_chunk_results",
                new=_spy_yield_chunk_results,
            ),
        ):
            run_result, n_tested = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info=None,
                chunk_size=500,
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
            )

        assert len(run_result.associations) == n_tested
        assert n_tested > 50, "Need multiple JAX sub-chunks for this regression test"
        assert len(chunk_sizes) > 1, "Expected multiple sub-chunk materializations"
        assert sum(chunk_sizes) == n_tested
        assert len(chunk_sizes) == math.ceil(n_tested / 50)


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
        run_result = run_lmm_association_jax(
            data.genotypes,
            phenotypes,
            kinship_single,
            snp_info,
            check_memory=False,
        )
        results_single = run_result.associations

        # Run with streaming (multiple chunks)
        # Use larger chunk to reduce number of JIT compilations
        run_result, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship_multi,
            chunk_size=5000,  # Fewer chunks = faster test
            check_memory=False,
            show_progress=False,
        )
        results_multi = run_result.associations

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
            run_result, _ = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship.copy(),
                chunk_size=cs,
                check_memory=False,
                show_progress=False,
            )
            results = run_result.associations
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
    run_result = run_lmm_association_jax(
        data.genotypes.astype(np.float64),
        phenotypes,
        kinship_batch,
        snp_info,
        check_memory=False,
        show_progress=False,
    )
    results_batch = run_result.associations

    # Streaming runner with small chunk_size to force multiple file chunks
    run_result, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        chunk_size=100,  # Small to exercise chunking
        check_memory=False,
        show_progress=False,
    )
    results_stream = run_result.associations

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
    run_result = run_lmm_association_jax(
        data.genotypes.astype(np.float64),
        phenotypes,
        kinship_batch,
        snp_info,
        check_memory=False,
        show_progress=False,
        lmm_mode=2,
    )
    results_batch = run_result.associations

    # Streaming runner with lmm_mode=2
    run_result, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        chunk_size=100,
        check_memory=False,
        show_progress=False,
        lmm_mode=2,
    )
    results_stream = run_result.associations

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
    run_result = run_lmm_association_jax(
        data.genotypes.astype(np.float64),
        phenotypes,
        kinship_batch,
        snp_info,
        check_memory=False,
        show_progress=False,
        lmm_mode=3,
    )
    results_batch = run_result.associations

    # Streaming runner with lmm_mode=3
    run_result, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        chunk_size=100,
        check_memory=False,
        show_progress=False,
        lmm_mode=3,
    )
    results_stream = run_result.associations

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
class TestExposedRotationDiagnostic:
    """Tests for the UT@G exposed rotation timing diagnostic in runner_streaming."""

    def test_single_chunk_exposed_equals_total(self, sample_plink_data: Path) -> None:
        """Single-chunk run: exposed rotation should equal total rotation.

        When chunk_size > n_snps, only one file-chunk and one JAX sub-chunk
        are processed. The first (and only) rotation has no prior compute to
        overlap with, so exposed == total.
        """
        from jamma.lmm.runner_streaming import last_run_timing

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # chunk_size >> n_snps ensures single file-chunk and single JAX sub-chunk
        _, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,
            chunk_size=99_999,  # Larger than all 500 SNPs: single chunk
            check_memory=False,
            show_progress=False,
        )

        assert "rotation_s" in last_run_timing, (
            "last_run_timing must contain 'rotation_s'"
        )
        assert "rotation_exposed_s" in last_run_timing, (
            "last_run_timing must contain 'rotation_exposed_s'"
        )
        assert last_run_timing["rotation_exposed_s"] == pytest.approx(
            last_run_timing["rotation_s"], abs=1e-6
        ), (
            f"Single-chunk: exposed ({last_run_timing['rotation_exposed_s']:.6f}s) "
            f"should equal total ({last_run_timing['rotation_s']:.6f}s)"
        )

    def test_multi_chunk_exposed_leq_total(self, sample_plink_data: Path) -> None:
        """Multi-chunk run: exposed rotation cannot exceed total rotation.

        Whether or not actual overlap occurs, the invariant exposed <= total
        must always hold. Uses a small chunk_size to force multiple chunks.
        """
        from jamma.lmm.runner_streaming import last_run_timing

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Small chunk_size forces multiple disk-read chunks
        _, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,
            chunk_size=100,
            check_memory=False,
            show_progress=False,
        )

        assert "rotation_s" in last_run_timing
        assert "rotation_exposed_s" in last_run_timing
        # Invariant: exposed cannot exceed total (with small float tolerance)
        assert last_run_timing["rotation_exposed_s"] <= (
            last_run_timing["rotation_s"] + 1e-6
        ), (
            f"Exposed ({last_run_timing['rotation_exposed_s']:.6f}s) must be "
            f"<= total ({last_run_timing['rotation_s']:.6f}s)"
        )

    def test_timing_keys_present_after_run(self, sample_plink_data: Path) -> None:
        """last_run_timing dict contains all four expected timing keys."""
        from jamma.lmm.runner_streaming import last_run_timing

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        _, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,
            check_memory=False,
            show_progress=False,
        )

        expected_keys = {
            "rotation_s",
            "rotation_exposed_s",
            "jax_compute_s",
            "result_write_s",
        }
        assert set(last_run_timing.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(last_run_timing.keys())}"
        )
        for key, val in last_run_timing.items():
            assert val >= 0.0, f"Timing value for '{key}' must be >= 0, got {val}"


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


@pytest.mark.tier1
def test_stream_genotype_chunks_filtered(sample_plink_data: Path) -> None:
    """stream_genotype_chunks with snp_indices reads only specified columns (RUN-04)."""
    # Read full data for comparison
    chunks_full = list(
        stream_genotype_chunks(sample_plink_data, chunk_size=5000, show_progress=False)
    )
    n_snps_total = sum(end - start for _, start, end in chunks_full)

    # Read filtered subset
    rng = np.random.default_rng(42)
    snp_indices = np.sort(
        rng.choice(n_snps_total, size=min(200, n_snps_total), replace=False)
    )

    chunks_filtered = list(
        stream_genotype_chunks(
            sample_plink_data,
            chunk_size=80,
            show_progress=False,
            snp_indices=snp_indices,
        )
    )

    # Total filtered SNPs should match snp_indices length
    total_filtered = sum(end - start for _, start, end in chunks_filtered)
    assert total_filtered == len(snp_indices), (
        f"Expected {len(snp_indices)} filtered SNPs, got {total_filtered}"
    )

    # Each chunk should have the right number of columns
    for chunk, start, end in chunks_filtered:
        assert chunk.shape[1] == end - start, (
            f"Chunk shape mismatch: got {chunk.shape[1]} cols, expected {end - start}"
        )

    # Verify filtered data matches corresponding columns from full read
    full_matrix = np.concatenate([c for c, _, _ in chunks_full], axis=1)
    filtered_matrix = np.concatenate([c for c, _, _ in chunks_filtered], axis=1)
    np.testing.assert_array_equal(
        filtered_matrix,
        full_matrix[:, snp_indices].astype(filtered_matrix.dtype),
        err_msg="Filtered read values do not match corresponding full-read columns",
    )


@pytest.mark.tier1
def test_stream_genotype_chunks_unsorted_snp_indices_raises(
    sample_plink_data: Path,
) -> None:
    """Unsorted snp_indices raises ValueError."""
    unsorted = np.array([5, 2, 8, 1])
    with pytest.raises(ValueError, match="sorted in strictly ascending order"):
        list(
            stream_genotype_chunks(
                sample_plink_data, snp_indices=unsorted, show_progress=False
            )
        )


@pytest.mark.tier1
def test_stream_genotype_chunks_duplicate_snp_indices_raises(
    sample_plink_data: Path,
) -> None:
    """Duplicate snp_indices raises ValueError."""
    duplicates = np.array([1, 3, 3, 5])
    with pytest.raises(ValueError, match="sorted in strictly ascending order"):
        list(
            stream_genotype_chunks(
                sample_plink_data, snp_indices=duplicates, show_progress=False
            )
        )


@pytest.mark.tier1
def test_stream_genotype_chunks_oob_snp_indices_raises(
    sample_plink_data: Path,
) -> None:
    """Out-of-bounds snp_indices raises ValueError."""
    oob = np.array([0, 1, 999_999])
    with pytest.raises(ValueError, match="out of bounds"):
        list(
            stream_genotype_chunks(
                sample_plink_data, snp_indices=oob, show_progress=False
            )
        )


@pytest.mark.tier1
def test_prefetch_iterator_correctness() -> None:
    """prefetch_iterator yields identical results to sequential iteration (RUN-09)."""

    def make_chunks():
        for i in range(5):
            chunk = np.arange(10 * i, 10 * (i + 1), dtype=np.float64).reshape(2, 5)
            yield chunk, i * 5, (i + 1) * 5

    # Sequential
    sequential = list(make_chunks())

    # Prefetched
    prefetched = list(prefetch_iterator(make_chunks()))

    assert len(prefetched) == len(sequential)
    for (c1, s1, e1), (c2, s2, e2) in zip(sequential, prefetched, strict=True):
        np.testing.assert_array_equal(c1, c2)
        assert s1 == s2
        assert e1 == e2


@pytest.mark.tier1
def test_prefetch_iterator_empty() -> None:
    """prefetch_iterator handles empty iterator correctly (RUN-09)."""
    result = list(prefetch_iterator(iter([])))
    assert result == []


@pytest.mark.tier1
def test_prefetch_iterator_single_item() -> None:
    """prefetch_iterator handles single-item iterator correctly (RUN-09)."""

    def single():
        yield np.array([[1.0, 2.0]]), 0, 2

    result = list(prefetch_iterator(single()))
    assert len(result) == 1
    np.testing.assert_array_equal(result[0][0], np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# ThreadPoolExecutor rotation-compute overlap tests for streaming runner (Plan 54-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.requires_jax
class TestThreadPoolExecutorOverlapStreaming:
    """Tests for ThreadPoolExecutor-based rotation-compute overlap in runner_streaming.

    Verifies that BLAS rotation for JAX sub-chunk N+1 runs concurrently
    with JAX compute for sub-chunk N using a background thread.
    """

    def test_rotation_overlap_multi_subchunk_exposed_leq_total(
        self, sample_plink_data: Path
    ) -> None:
        """Multi-sub-chunk: rotation_exposed_s <= rotation_s.

        Uses a small jax_chunk_size to force multiple JAX sub-chunks within
        a single BED file chunk, exercising the inner ThreadPoolExecutor loop.
        """
        from unittest.mock import patch

        from jamma.lmm.runner_streaming import last_run_timing

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Force small jax_chunk_size so 500 SNPs → many sub-chunks
        with patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=50):
            _, _ = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info=None,
                chunk_size=500,  # Single BED file chunk covering all SNPs
                check_memory=False,
                show_progress=False,
            )

        assert "rotation_s" in last_run_timing
        assert "rotation_exposed_s" in last_run_timing
        assert last_run_timing["rotation_exposed_s"] <= (
            last_run_timing["rotation_s"] + 1e-6
        ), (
            f"Exposed ({last_run_timing['rotation_exposed_s']:.6f}s) must be "
            f"<= total ({last_run_timing['rotation_s']:.6f}s)"
        )

    def test_rotation_overlap_single_subchunk_exposed_equals_total(
        self, sample_plink_data: Path
    ) -> None:
        """Single-sub-chunk: rotation_exposed_s == rotation_s.

        When jax_chunk_size >= all SNPs in the file chunk, there is only one
        JAX sub-chunk and no overlap is possible.
        """
        from jamma.lmm.runner_streaming import last_run_timing

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # chunk_size > n_snps ensures single BED chunk and single JAX sub-chunk
        _, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,
            chunk_size=99_999,
            check_memory=False,
            show_progress=False,
        )

        assert last_run_timing["rotation_exposed_s"] == pytest.approx(
            last_run_timing["rotation_s"], abs=1e-6
        ), (
            f"Single-sub-chunk: exposed ({last_run_timing['rotation_exposed_s']:.6f}s) "
            f"should equal total ({last_run_timing['rotation_s']:.6f}s)"
        )

    def test_rotation_overlap_numerical_correctness(
        self, sample_plink_data: Path
    ) -> None:
        """ThreadPoolExecutor overlap produces numerically identical results.

        Results with multiple JAX sub-chunks (overlap active) must match
        results with a single sub-chunk (no overlap) to rtol=1e-12.
        """
        from unittest.mock import patch

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Reference: single JAX sub-chunk (no overlap)
        run_result, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship.copy(),
            snp_info=None,
            chunk_size=99_999,
            check_memory=False,
            show_progress=False,
        )
        reference = run_result.associations

        # Test: multiple JAX sub-chunks (overlap active)
        with patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=50):
            run_result, _ = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship.copy(),
                snp_info=None,
                chunk_size=500,
                check_memory=False,
                show_progress=False,
            )
            results = run_result.associations

        assert len(results) == len(reference), (
            f"Expected {len(reference)} results, got {len(results)}"
        )

        ref_by_rs = {r.rs: r for r in reference}
        for r in results:
            ref = ref_by_rs[r.rs]
            if not np.isnan(r.beta):
                np.testing.assert_allclose(
                    r.beta,
                    ref.beta,
                    rtol=1e-12,
                    err_msg=f"beta mismatch for {r.rs}",
                )
                np.testing.assert_allclose(
                    r.se,
                    ref.se,
                    rtol=1e-12,
                    err_msg=f"se mismatch for {r.rs}",
                )
                np.testing.assert_allclose(
                    r.p_wald,
                    ref.p_wald,
                    rtol=1e-12,
                    err_msg=f"p_wald mismatch for {r.rs}",
                )

    def test_streaming_pipeline_buffers_passed(self, sample_plink_data: Path) -> None:
        """_compute_chunk_size is called with pipeline_buffers=2 in streaming runner."""
        from unittest.mock import patch

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        with patch(
            "jamma.lmm.runner_streaming._compute_chunk_size", return_value=1000
        ) as mock_chunk:
            _, _ = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info=None,
                check_memory=False,
                show_progress=False,
            )

        calls_with_pipeline = [
            c
            for c in mock_chunk.call_args_list
            if c.kwargs.get("pipeline_buffers") == 2
            or (len(c.args) >= 5 and c.args[4] == 2)
        ]
        assert len(calls_with_pipeline) >= 1, (
            f"Expected _compute_chunk_size called with pipeline_buffers=2. "
            f"Actual calls: {mock_chunk.call_args_list}"
        )

    def test_threadpoolexecutor_used_in_streaming_runner(self) -> None:
        """runner_streaming imports and uses ThreadPoolExecutor."""
        import inspect

        from jamma.lmm import runner_streaming

        source = inspect.getsource(runner_streaming)
        assert "ThreadPoolExecutor" in source, (
            "runner_streaming must use ThreadPoolExecutor for rotation-compute overlap"
        )
        assert "executor.submit" in source, (
            "runner_streaming must submit rotation work to background thread"
        )

    def test_background_rotation_failure_propagates(
        self, sample_plink_data: Path
    ) -> None:
        """Background rotation failure raises RuntimeError with exception chain.

        When prepare_utg_chunk raises in the background thread, the streaming
        runner must wrap it in a RuntimeError with 'from exc' so the original
        traceback is preserved.
        """
        from unittest.mock import patch

        rng = np.random.default_rng(777)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        call_count = 0
        from jamma.lmm import prepare

        original_prepare = prepare.prepare_utg_chunk

        def _failing_prepare(*args, **kwargs):
            """Succeed on first call, fail on second (background thread)."""
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ValueError("Simulated BLAS failure in background rotation")
            return original_prepare(*args, **kwargs)

        with (
            patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=50),
            patch(
                "jamma.lmm.runner_streaming.prepare_utg_chunk",
                side_effect=_failing_prepare,
            ),
        ):
            with pytest.raises(
                RuntimeError, match="Background rotation failed"
            ) as exc_info:
                run_lmm_association_streaming(
                    sample_plink_data,
                    phenotypes,
                    kinship,
                    snp_info=None,
                    chunk_size=500,
                    check_memory=False,
                    show_progress=False,
                )

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "Simulated BLAS failure" in str(exc_info.value.__cause__)

    def test_background_rotation_memoryerror_propagates_directly(
        self, sample_plink_data: Path
    ) -> None:
        """MemoryError from background rotation is NOT wrapped in RuntimeError.

        The except MemoryError: raise clause must fire before the generic
        except Exception handler, allowing OOM to propagate directly.
        """
        from unittest.mock import patch

        rng = np.random.default_rng(777)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        call_count = 0
        from jamma.lmm import prepare

        original_prepare = prepare.prepare_utg_chunk

        def _oom_prepare(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise MemoryError("Simulated OOM in background rotation")
            return original_prepare(*args, **kwargs)

        with (
            patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=50),
            patch(
                "jamma.lmm.runner_streaming.prepare_utg_chunk",
                side_effect=_oom_prepare,
            ),
        ):
            with pytest.raises(MemoryError, match="Simulated OOM"):
                run_lmm_association_streaming(
                    sample_plink_data,
                    phenotypes,
                    kinship,
                    snp_info=None,
                    chunk_size=500,
                    check_memory=False,
                    show_progress=False,
                )

    def test_clear_caches_runs_on_failure(self, sample_plink_data: Path) -> None:
        """Failure paths still clear JAX caches before unwinding."""
        from unittest.mock import patch

        rng = np.random.default_rng(777)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        call_count = 0
        from jamma.lmm import prepare

        original_prepare = prepare.prepare_utg_chunk

        def _failing_prepare(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ValueError("Simulated BLAS failure in background rotation")
            return original_prepare(*args, **kwargs)

        with (
            patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=50),
            patch(
                "jamma.lmm.runner_streaming.prepare_utg_chunk",
                side_effect=_failing_prepare,
            ),
            patch("jamma.lmm.runner_streaming.jax.clear_caches") as mock_clear,
        ):
            with pytest.raises(RuntimeError, match="Background rotation failed"):
                run_lmm_association_streaming(
                    sample_plink_data,
                    phenotypes,
                    kinship,
                    snp_info=None,
                    chunk_size=500,
                    check_memory=False,
                    show_progress=False,
                )

        mock_clear.assert_called_once()

    def test_multi_file_chunk_prev_compute_end_handoff(
        self, sample_plink_data: Path
    ) -> None:
        """prev_compute_end persists across BED file-chunk boundaries.

        Uses chunk_size=100 (5 BED file chunks for 500 SNPs) and small
        jax_chunk_size to force multiple JAX sub-chunks within each BED chunk.
        The exposed rotation timing must still satisfy exposed <= total,
        verifying that prev_compute_end correctly bridges file-chunk boundaries.
        """
        from unittest.mock import patch

        from jamma.lmm.runner_streaming import last_run_timing

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # chunk_size=100 → 5 BED file chunks for 500 SNPs
        # jax_chunk_size=25 → 4 JAX sub-chunks per BED file chunk
        with patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=25):
            run_result, n_tested = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info=None,
                chunk_size=100,
                check_memory=False,
                show_progress=False,
            )
            results = run_result.associations

        assert n_tested > 0, "Should have tested some SNPs"
        assert len(results) > 0, "Should have results"

        # Timing invariant must hold across file-chunk boundaries
        assert "rotation_s" in last_run_timing
        assert "rotation_exposed_s" in last_run_timing
        assert last_run_timing["rotation_exposed_s"] <= (
            last_run_timing["rotation_s"] + 1e-6
        ), (
            f"Multi-file-chunk: exposed ({last_run_timing['rotation_exposed_s']:.6f}s) "
            f"must be <= total ({last_run_timing['rotation_s']:.6f}s)"
        )


# ---------------------------------------------------------------------------
# Rotation overlap effectiveness tests for streaming runner (Plan 54-03)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.requires_jax
class TestRotationOverlapEffectivenessStreaming:
    """Tests that rotation-compute overlap is measurably effective in streaming runner.

    Verifies that on multi-sub-chunk runs, exposed rotation time is strictly
    less than total rotation time (overlap hides meaningful rotation work).
    These tests complement Plan 54-02 invariant tests with effectiveness checks.
    """

    def test_streaming_overlap_effectiveness(self, sample_plink_data: Path) -> None:
        """Multi-sub-chunk: rotation_exposed_s < 0.95 * rotation_s (overlap hides >=5%).

        Forces multiple JAX sub-chunks within a BED file chunk so the
        ThreadPoolExecutor overlap can hide rotation work behind JAX compute.
        The 5% threshold is conservative — real large datasets see 80%+ hiding.
        """
        from unittest.mock import patch

        from jamma.lmm.runner_streaming import last_run_timing

        rng = np.random.default_rng(54)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Force jax_chunk_size=25 so 500 SNPs → 20 JAX sub-chunks
        with patch("jamma.lmm.runner_streaming._compute_chunk_size", return_value=25):
            _, _ = run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info=None,
                chunk_size=500,  # Single BED file chunk covering all SNPs
                check_memory=False,
                show_progress=False,
            )

        rot_total = last_run_timing["rotation_s"]
        rot_exposed = last_run_timing["rotation_exposed_s"]

        assert rot_total > 0, "rotation_s must be > 0 (rotation occurred)"
        assert rot_exposed >= 0, f"rotation_exposed_s must be >= 0, got {rot_exposed}"

        # Overlap effectiveness: exposed must be < 95% of total (at least 5% hidden)
        assert rot_exposed < 0.95 * rot_total, (
            f"Expected overlap to hide at least 5% of rotation time "
            f"on 20-sub-chunk run. "
            f"total={rot_total:.6f}s, exposed={rot_exposed:.6f}s, "
            f"ratio={rot_exposed / max(rot_total, 1e-10):.3f} (threshold: 0.95). "
            f"ThreadPoolExecutor overlap in streaming runner may not be active."
        )

    def test_streaming_timing_keys_present(self, sample_plink_data: Path) -> None:
        """All four timing keys are present and non-negative after a streaming run.

        Verifies that last_run_timing is fully populated with all expected
        keys and all values are valid (float, >= 0).
        """
        from jamma.lmm.runner_streaming import last_run_timing

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        _, _ = run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,
            check_memory=False,
            show_progress=False,
        )

        expected_keys = {
            "rotation_s",
            "rotation_exposed_s",
            "jax_compute_s",
            "result_write_s",
        }
        assert set(last_run_timing.keys()) >= expected_keys, (
            f"last_run_timing missing keys. "
            f"Expected (at least): {expected_keys}, "
            f"Got: {set(last_run_timing.keys())}"
        )
        for key in expected_keys:
            val = last_run_timing[key]
            assert isinstance(val, float), (
                f"last_run_timing['{key}'] must be float, got {type(val)}"
            )
            assert val >= 0.0, f"last_run_timing['{key}'] must be >= 0, got {val}"


@pytest.mark.tier1
class TestComputeKinshipStreamingSinglePass:
    """Tests for the single-pass kinship optimization in compute_kinship_streaming."""

    def test_single_pass_matches_two_pass_result(self, sample_plink_data: Path) -> None:
        """Single-pass (default filters) matches the full-load reference result."""
        data = load_plink_binary(sample_plink_data)
        K_ref = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )

        # Default filters trigger single-pass path
        K_single = compute_kinship_streaming(
            sample_plink_data,
            maf_threshold=0.0,
            miss_threshold=1.0,
            ksnps_indices=None,
            check_memory=False,
            show_progress=False,
        )

        np.testing.assert_allclose(
            K_single,
            K_ref,
            rtol=1e-10,
            atol=1e-14,
            err_msg="Single-pass kinship must match full-load reference (rtol=1e-10)",
        )

    def test_two_pass_preserved_for_maf_filter(self, sample_plink_data: Path) -> None:
        """Two-pass path is used when maf_threshold > 0.0."""
        data = load_plink_binary(sample_plink_data)
        K_ref = compute_centered_kinship(
            data.genotypes.astype(np.float64),
            maf_threshold=0.05,
            check_memory=False,
        )

        K_filtered = compute_kinship_streaming(
            sample_plink_data,
            maf_threshold=0.05,
            miss_threshold=1.0,
            ksnps_indices=None,
            check_memory=False,
            show_progress=False,
        )

        np.testing.assert_allclose(
            K_filtered,
            K_ref,
            rtol=1e-10,
            atol=1e-14,
            err_msg="Two-pass kinship with maf_threshold=0.05 must match full-load",
        )

    def test_two_pass_preserved_for_miss_filter(self, sample_plink_data: Path) -> None:
        """Two-pass path is used when miss_threshold < 1.0."""
        data = load_plink_binary(sample_plink_data)
        K_ref = compute_centered_kinship(
            data.genotypes.astype(np.float64),
            miss_threshold=0.9,
            check_memory=False,
        )

        K_filtered = compute_kinship_streaming(
            sample_plink_data,
            maf_threshold=0.0,
            miss_threshold=0.9,
            ksnps_indices=None,
            check_memory=False,
            show_progress=False,
        )

        np.testing.assert_allclose(
            K_filtered,
            K_ref,
            rtol=1e-10,
            atol=1e-14,
            err_msg="Two-pass kinship with miss_threshold=0.9 must match full-load",
        )

    def test_two_pass_preserved_for_ksnps_restriction(
        self, sample_plink_data: Path
    ) -> None:
        """Two-pass path is used when ksnps_indices is provided."""
        data = load_plink_binary(sample_plink_data)
        n_snps = data.genotypes.shape[1]
        # Restrict to first 200 SNPs
        ksnps_indices = np.arange(min(200, n_snps))

        K_ksnps = compute_kinship_streaming(
            sample_plink_data,
            maf_threshold=0.0,
            miss_threshold=1.0,
            ksnps_indices=ksnps_indices,
            check_memory=False,
            show_progress=False,
        )

        # Compute reference by subsetting genotypes
        K_ref = compute_centered_kinship(
            data.genotypes[:, ksnps_indices].astype(np.float64),
            check_memory=False,
        )

        np.testing.assert_allclose(
            K_ksnps,
            K_ref,
            rtol=1e-10,
            atol=1e-14,
            err_msg="Two-pass kinship with ksnps_indices must match reference",
        )

    def test_single_pass_symmetry(self, sample_plink_data: Path) -> None:
        """Single-pass kinship produces a symmetric matrix."""
        K = compute_kinship_streaming(
            sample_plink_data,
            maf_threshold=0.0,
            miss_threshold=1.0,
            ksnps_indices=None,
            check_memory=False,
            show_progress=False,
        )
        np.testing.assert_allclose(
            K,
            K.T,
            err_msg="Single-pass kinship must be symmetric",
        )


# ---------------------------------------------------------------------------
# SC-02: Streaming runner mode 4, all-filtered, and covariate tests
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_streaming_all_tests_matches_batch(sample_plink_data: Path) -> None:
    """Verify streaming all-tests (mode 4) output matches batch runner mode 4.

    Mode 4 populates p_wald, p_lrt, and p_score. This test exercises the
    lmm_mode=4 code path in both runners and verifies field-by-field parity.
    """
    rng = np.random.default_rng(42)

    data = load_plink_binary(sample_plink_data)
    geno = data.genotypes.astype(np.float64)
    phenotypes = rng.standard_normal(data.n_samples)
    kinship = compute_centered_kinship(geno, check_memory=False)
    # eigendecomp overwrites K in-place; needs fresh copy per run
    kinship_batch = kinship.copy()
    kinship_stream = kinship.copy()

    snp_info = _build_snp_info(data)

    run_result = run_lmm_association_jax(
        geno,
        phenotypes,
        kinship_batch,
        snp_info,
        check_memory=False,
        show_progress=False,
        lmm_mode=4,
    )
    results_batch = run_result.associations

    run_result, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        chunk_size=100,
        check_memory=False,
        show_progress=False,
        lmm_mode=4,
    )
    results_stream = run_result.associations

    _assert_results_match(
        results_batch,
        results_stream,
        fields=("beta", "se", "p_wald", "p_lrt", "p_score"),
    )


@pytest.mark.tier1
def test_streaming_unaligned_chunk_size_matches_batch(sample_plink_data: Path) -> None:
    """Streaming with chunk_size that doesn't divide n_snps matches batch runner.

    Uses chunk_size=7 with 500 SNPs (71 full chunks + 3 remainder) to verify
    the streaming runner handles chunk boundaries correctly.
    """
    rng = np.random.default_rng(42)

    data = load_plink_binary(sample_plink_data)
    geno = data.genotypes.astype(np.float64)
    phenotypes = rng.standard_normal(data.n_samples)
    kinship = compute_centered_kinship(geno, check_memory=False)
    kinship_batch = kinship.copy()
    kinship_stream = kinship.copy()

    snp_info = _build_snp_info(data)

    run_result = run_lmm_association_jax(
        geno,
        phenotypes,
        kinship_batch,
        snp_info,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
    )
    results_batch = run_result.associations

    run_result, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        chunk_size=7,  # Does not divide 500 evenly
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
    )
    results_stream = run_result.associations

    _assert_results_match(results_batch, results_stream)


@pytest.mark.tier1
def test_streaming_all_snps_filtered_returns_zero(sample_plink_data: Path) -> None:
    """Streaming runner with impossible MAF threshold returns zero tested SNPs.

    maf_threshold=1.0 is impossible (MAF is at most 0.5), so all SNPs are
    filtered out. The runner must handle this gracefully: return empty results
    and n_tested == 0, without raising or hanging.
    """
    data = load_plink_binary(sample_plink_data)
    phenotypes = np.random.default_rng(42).standard_normal(data.n_samples)
    # Kinship is never consumed (all SNPs filtered before eigendecomp),
    # so use a cheap identity matrix instead of computing the real one.
    kinship = np.eye(data.n_samples)
    snp_info = _build_snp_info(data)

    run_result, n_tested = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship,
        snp_info,
        maf_threshold=1.0,  # Impossible threshold; all SNPs filtered
        check_memory=False,
        show_progress=False,
    )
    results = run_result.associations

    assert len(results) == 0, (
        f"Expected empty results when all SNPs are filtered, got {len(results)}"
    )
    assert n_tested == 0, (
        f"Expected n_tested=0 when all SNPs are filtered, got {n_tested}"
    )


@pytest.mark.tier1
def test_streaming_with_covariates_matches_batch(sample_plink_data: Path) -> None:
    """Streaming runner with covariates produces results consistent with batch runner.

    Uses the gemma_synthetic fixture (100 samples) with a synthetic covariate
    matrix (intercept + 1 random covariate). Both batch and streaming runners
    are called with the same covariates and results compared field-by-field.
    """
    rng = np.random.default_rng(42)

    data = load_plink_binary(sample_plink_data)
    geno = data.genotypes.astype(np.float64)
    n_samples = data.n_samples
    phenotypes = rng.standard_normal(n_samples)
    kinship = compute_centered_kinship(geno, check_memory=False)
    # eigendecomp overwrites K in-place; needs fresh copy per run
    kinship_batch = kinship.copy()
    kinship_stream = kinship.copy()

    snp_info = _build_snp_info(data)

    # Synthetic covariate: intercept + 1 random covariate
    covariates = np.column_stack(
        [
            np.ones(n_samples),
            rng.standard_normal(n_samples),
        ]
    )

    run_result = run_lmm_association_jax(
        geno,
        phenotypes,
        kinship_batch,
        snp_info,
        covariates=covariates,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
    )
    results_batch = run_result.associations

    # Streaming runner with same covariates
    run_result, _ = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship_stream,
        snp_info,
        covariates=covariates,
        chunk_size=100,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
    )
    results_stream = run_result.associations

    _assert_results_match(results_batch, results_stream)


@pytest.mark.tier0
def test_streaming_all_snps_filtered_mode4_returns_zero(
    sample_plink_data: Path,
) -> None:
    """Streaming runner mode 4 with impossible MAF threshold returns zero tested SNPs.

    Verifies that mode 4 (all tests: Wald + LRT + Score) handles the all-filtered
    edge case correctly for all-tests mode (lmm_mode=4). maf_threshold=1.0
    is impossible (MAF is at most 0.5), so all SNPs are filtered out. The runner must
    handle this gracefully: return empty results and n_tested == 0.
    """
    data = load_plink_binary(sample_plink_data)
    phenotypes = np.random.default_rng(42).standard_normal(data.n_samples)
    # Kinship is never consumed (all SNPs filtered before eigendecomp),
    # so use a cheap identity matrix instead of computing the real one.
    kinship = np.eye(data.n_samples)
    snp_info = _build_snp_info(data)

    run_result, n_tested = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship,
        snp_info,
        maf_threshold=1.0,  # Impossible threshold; all SNPs filtered
        check_memory=False,
        show_progress=False,
        lmm_mode=4,
    )
    results = run_result.associations

    assert len(results) == 0, (
        f"Expected empty results when all SNPs filtered (mode 4), got {len(results)}"
    )
    assert n_tested == 0, (
        f"Expected n_tested=0 when all SNPs filtered (mode 4), got {n_tested}"
    )


@pytest.mark.tier1
def test_streaming_output_path_returns_empty_list(
    sample_plink_data: Path,
    tmp_path: Path,
) -> None:
    """Streaming runner with output_path returns empty list and writes results to disk.

    Verifies that when output_path is provided, the runner:
    - Returns an empty list (not the actual results)
    - Returns n_tested > 0 (some SNPs passed filtering)
    - Writes results to the output file
    - The file contains the expected number of rows
    """
    from jamma.validation import load_gemma_assoc

    data = load_plink_binary(sample_plink_data)
    geno = data.genotypes.astype(np.float64)
    n_samples = data.n_samples
    phenotypes = np.random.default_rng(42).standard_normal(n_samples)
    # Use real kinship so SNPs pass MAF filtering and are actually tested.
    kinship = compute_centered_kinship(geno, check_memory=False)
    snp_info = _build_snp_info(data)

    output_path = tmp_path / "results.assoc.txt"

    run_result, n_tested = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship,
        snp_info,
        output_path=output_path,
        check_memory=False,
        show_progress=False,
    )
    results = run_result.associations

    assert len(results) == 0, (
        f"Expected empty results when output_path is set, got {len(results)} results"
    )
    assert n_tested > 0, "Expected some SNPs to be tested (output_path mode)"
    assert output_path.exists(), f"Expected output file to exist at {output_path}"

    disk_results = load_gemma_assoc(output_path)
    assert len(disk_results) == n_tested, (
        f"Expected {n_tested} rows in output file, got {len(disk_results)}"
    )


@pytest.mark.tier1
def test_streaming_output_path_mode4_writes_all_columns(
    sample_plink_data: Path,
    tmp_path: Path,
) -> None:
    """Streaming runner mode 4 with output_path writes Wald+LRT+Score columns to disk.

    Verifies that the mode 4 disk-write path through IncrementalAssocWriter
    produces the correct column layout (all three test types), not just mode 1.
    """
    from jamma.validation import load_gemma_assoc

    data = load_plink_binary(sample_plink_data)
    geno = data.genotypes.astype(np.float64)
    n_samples = data.n_samples
    phenotypes = np.random.default_rng(42).standard_normal(n_samples)
    kinship = compute_centered_kinship(geno, check_memory=False)
    snp_info = _build_snp_info(data)

    output_path = tmp_path / "results_mode4.assoc.txt"

    run_result, n_tested = run_lmm_association_streaming(
        sample_plink_data,
        phenotypes,
        kinship,
        snp_info,
        output_path=output_path,
        check_memory=False,
        show_progress=False,
        lmm_mode=4,
    )
    results = run_result.associations

    assert len(results) == 0, (
        "Expected empty results when output_path is set "
        f"(mode 4), got {len(results)} results"
    )
    assert n_tested > 0, "Expected some SNPs to be tested (mode 4 output_path)"
    assert output_path.exists(), f"Expected output file at {output_path}"

    disk_results = load_gemma_assoc(output_path)
    assert len(disk_results) == n_tested, (
        f"Expected {n_tested} rows, got {len(disk_results)}"
    )
    # Mode 4 should produce all three test types' fields
    first = disk_results[0]
    assert first.p_wald is not None, "Mode 4 should include p_wald"
    assert first.p_lrt is not None, "Mode 4 should include p_lrt"
    assert first.p_score is not None, "Mode 4 should include p_score"


@pytest.mark.tier1
class TestNaNDiagnostics:
    """Tests for NaN diagnostic warnings in the streaming runner."""

    def test_nan_warning_emitted_for_nan_pvalues(self, sample_plink_data: Path) -> None:
        """When chunk results contain NaN p-values, a warning is emitted."""
        import re
        from unittest.mock import patch

        import jamma.lmm.runner_streaming as streaming_module

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )
        snp_info = _build_snp_info(data)

        original_fn = streaming_module._chunk_result_to_numpy

        def _inject_nan(*args, **kwargs):
            arrays = original_fn(*args, **kwargs)
            if "pwalds" in arrays:
                arr = arrays["pwalds"].copy()
                arr[0] = np.nan
                arrays["pwalds"] = arr
            return arrays

        logged_warnings: list[str] = []
        original_warning = streaming_module.logger.warning

        def capture_warning(msg, *args, **kwargs):
            logged_warnings.append(str(msg))
            return original_warning(msg, *args, **kwargs)

        with (
            patch.object(
                streaming_module,
                "_chunk_result_to_numpy",
                side_effect=_inject_nan,
            ),
            patch.object(
                streaming_module.logger,
                "warning",
                side_effect=capture_warning,
            ),
        ):
            run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info,
                check_memory=False,
                show_progress=False,
            )

        nan_warnings = [m for m in logged_warnings if "SNPs have NaN pwalds" in m]
        assert len(nan_warnings) > 0, (
            "Expected NaN warning for pwalds but none found. "
            f"All warnings: {logged_warnings}"
        )
        # Verify it contains the count format
        assert re.search(r"\d+/\d+ SNPs have NaN pwalds", nan_warnings[0])

    def test_no_nan_warning_when_clean(self, sample_plink_data: Path) -> None:
        """When no NaN values are present, no NaN warning is emitted."""
        from unittest.mock import patch

        import jamma.lmm.runner_streaming as streaming_module

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )
        snp_info = _build_snp_info(data)

        logged_warnings: list[str] = []
        original_warning = streaming_module.logger.warning

        def capture_warning(msg, *args, **kwargs):
            logged_warnings.append(str(msg))
            return original_warning(msg, *args, **kwargs)

        with patch.object(
            streaming_module.logger,
            "warning",
            side_effect=capture_warning,
        ):
            run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info,
                check_memory=False,
                show_progress=False,
            )

        nan_warnings = [m for m in logged_warnings if "SNPs have NaN" in m]
        assert len(nan_warnings) == 0, (
            f"Expected no NaN warnings for clean data, got: {nan_warnings}"
        )

    def test_nan_warning_format_matches_batch_runners(
        self, sample_plink_data: Path
    ) -> None:
        """NaN warning format: '{n_nan}/{n_filtered} SNPs have NaN {key}'."""
        import re
        from unittest.mock import patch

        import jamma.lmm.runner_streaming as streaming_module

        rng = np.random.default_rng(42)
        data = load_plink_binary(sample_plink_data)
        phenotypes = rng.standard_normal(data.n_samples)
        kinship = compute_centered_kinship(
            data.genotypes.astype(np.float64), check_memory=False
        )
        snp_info = _build_snp_info(data)

        original_fn = streaming_module._chunk_result_to_numpy

        def _inject_nan(*args, **kwargs):
            arrays = original_fn(*args, **kwargs)
            if "pwalds" in arrays:
                arr = arrays["pwalds"].copy()
                arr[:2] = np.nan
                arrays["pwalds"] = arr
            return arrays

        logged_warnings: list[str] = []
        original_warning = streaming_module.logger.warning

        def capture_warning(msg, *args, **kwargs):
            logged_warnings.append(str(msg))
            return original_warning(msg, *args, **kwargs)

        with (
            patch.object(
                streaming_module,
                "_chunk_result_to_numpy",
                side_effect=_inject_nan,
            ),
            patch.object(
                streaming_module.logger,
                "warning",
                side_effect=capture_warning,
            ),
        ):
            run_lmm_association_streaming(
                sample_plink_data,
                phenotypes,
                kinship,
                snp_info,
                check_memory=False,
                show_progress=False,
            )

        nan_warnings = [m for m in logged_warnings if "SNPs have NaN" in m]
        assert len(nan_warnings) > 0, "Expected at least one NaN warning"

        # Verify format: "{n_nan}/{n_filtered} SNPs have NaN {key} — ..."
        pattern = re.compile(r"^\d+/\d+ SNPs have NaN \w+")
        for msg in nan_warnings:
            assert pattern.match(msg), (
                f"NaN warning format mismatch. Expected "
                f"'{{n}}/{{total}} SNPs have NaN {{key}} ...' but got: {msg}"
            )
