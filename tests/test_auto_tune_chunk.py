"""Tests for chunk size computation invariants.

Verifies that _compute_chunk_size and auto_tune_chunk_size respect
MAX_SAFE_CHUNK cap, clamp constraints, and device alignment contracts.
"""

import pytest

from jamma.lmm.chunk import (
    MAX_SAFE_CHUNK,
    _compute_chunk_size,
    auto_tune_chunk_size,
)


@pytest.mark.tier0
class TestAutoTuneChunkSize:
    """Tests for auto_tune_chunk_size() safe capping."""

    def test_max_safe_chunk_constant_exists(self):
        """MAX_SAFE_CHUNK constant should be defined."""
        assert MAX_SAFE_CHUNK == 50_000

    def test_respects_max_chunk_default(self):
        """Should not exceed MAX_SAFE_CHUNK even with high memory budget."""
        # Very high memory budget would suggest huge chunk without cap
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=1_000_000,  # Million SNPs
            mem_budget_gb=1000.0,  # Unrealistically high budget
        )

        assert result <= MAX_SAFE_CHUNK

    def test_respects_custom_max_chunk(self):
        """Should respect custom max_chunk when provided."""
        custom_max = 10_000

        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=1_000_000,
            mem_budget_gb=1000.0,
            max_chunk=custom_max,
        )

        assert result <= custom_max

    def test_still_respects_n_filtered_when_smaller(self):
        """When n_filtered < max_chunk, should use n_filtered."""
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=5000,  # Smaller than max_chunk
            mem_budget_gb=100.0,
        )

        assert result <= 5000

    def test_still_respects_memory_budget_when_smaller(self):
        """When memory budget limits chunk size, should use that limit."""
        result = auto_tune_chunk_size(
            n_samples=100_000,  # Large samples means high memory per SNP
            n_filtered=1_000_000,
            mem_budget_gb=0.1,  # Very low budget
        )

        # Should be constrained by memory, not max_chunk
        assert result < MAX_SAFE_CHUNK

    def test_min_chunk_still_enforced(self):
        """min_chunk should be the floor when n_filtered allows it."""
        result = auto_tune_chunk_size(
            n_samples=100_000,
            n_filtered=50_000,
            mem_budget_gb=0.0001,  # Tiny budget
            min_chunk=1000,
        )
        assert result >= 1000

    def test_n_filtered_caps_below_min_chunk(self):
        """n_filtered takes precedence when smaller than min_chunk."""
        result = auto_tune_chunk_size(
            n_samples=100_000,
            n_filtered=500,  # Fewer SNPs than min_chunk
            mem_budget_gb=0.0001,
            min_chunk=1000,
        )
        assert result <= 500

    def test_typical_gwas_scale(self):
        """Smoke test: typical GWAS should get reasonable chunk size."""
        result = auto_tune_chunk_size(
            n_samples=10_000,
            n_filtered=500_000,
            mem_budget_gb=4.0,
        )

        # Should be reasonable: between 1000 and 50000
        assert 1000 <= result <= MAX_SAFE_CHUNK

    def test_backward_compatibility_default_args(self):
        """Existing calls without max_chunk should still work."""
        # This would fail if we broke the signature
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=10000,
        )

        assert result > 0

    def test_n_devices_greater_than_max_chunk(self):
        """n_devices > max_chunk should not exceed max_chunk."""
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=100_000,
            max_chunk=500,
            n_devices=1024,
        )
        assert result <= 500

    def test_n_devices_greater_than_n_filtered(self):
        """n_devices > n_filtered should not exceed n_filtered."""
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=50,
            n_devices=128,
        )
        assert result <= 50

    def test_alignment_does_not_drop_below_min_chunk_significantly(self):
        """Alignment rounding should not produce zero or negative."""
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=100_000,
            min_chunk=1000,
            n_devices=128,
        )
        assert result > 0


@pytest.mark.tier0
class TestComputeChunkSize:
    """Tests for _compute_chunk_size with MAX_SAFE_CHUNK cap."""

    def test_small_dataset_no_chunking(self):
        """When n_snps < MAX_SAFE_CHUNK, return n_snps."""
        result = _compute_chunk_size(n_samples=1000, n_snps=5000)
        assert result == 5000

    def test_large_dataset_caps_at_max_safe(self):
        """When n_snps > MAX_SAFE_CHUNK, cap at MAX_SAFE_CHUNK."""
        result = _compute_chunk_size(n_samples=1000, n_snps=500_000)
        assert result == MAX_SAFE_CHUNK

    def test_75k_samples_gets_large_chunks(self):
        """At 75k samples, chunk size is MAX_SAFE_CHUNK (no int32 guard).

        The old int32 guard produced 3,817 SNPs/chunk at 75k samples,
        fragmenting 95k SNPs into 25 tiny chunks. With memory-based
        sizing, we get MAX_SAFE_CHUNK (50k) — just 2 chunks.
        """
        chunk = _compute_chunk_size(n_samples=75_000, n_snps=95_000)
        assert chunk == MAX_SAFE_CHUNK

    def test_125k_samples_gets_large_chunks(self):
        """At 125k samples, chunk size is MAX_SAFE_CHUNK (no int32 guard)."""
        chunk = _compute_chunk_size(n_samples=125_000, n_snps=95_000)
        assert chunk == MAX_SAFE_CHUNK

    @pytest.mark.parametrize("n_devices", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_device_alignment(self, n_devices):
        """Chunk is device-aligned when n_devices > 1 and chunking occurs."""
        result = _compute_chunk_size(
            n_samples=50_000,
            n_snps=500_000,
            n_devices=n_devices,
        )
        if n_devices > 1 and result < 500_000:
            assert result % n_devices == 0, (
                f"Chunk {result} is not aligned to {n_devices} devices"
            )

    def test_never_returns_zero(self):
        """Chunk size must always be >= 1."""
        result = _compute_chunk_size(n_samples=0, n_snps=100)
        assert result >= 1


@pytest.mark.tier0
class TestChunkSizingAtDatabricksScale:
    """Chunk sizing at Databricks-relevant scale (100k+ samples, many devices).

    Verifies _compute_chunk_size and auto_tune_chunk_size produce valid,
    device-aligned chunks at the scale where JAMMA actually runs.
    """

    @pytest.mark.parametrize("n_devices", [1, 8, 16, 24, 48])
    def test_chunk_positive_at_scale(self, n_devices):
        """At 125k samples, chunk size is always positive."""
        result = _compute_chunk_size(
            n_samples=125_000,
            n_snps=95_000,
            n_devices=n_devices,
        )
        assert result > 0, f"Chunk size must be positive, got {result}"

    @pytest.mark.parametrize("n_devices", [1, 8, 16, 24, 48])
    def test_chunk_device_alignment_at_scale(self, n_devices):
        """Chunk is a multiple of n_devices when n_devices > 1."""
        n_samples = 125_000
        n_snps = 95_000

        result = _compute_chunk_size(
            n_samples=n_samples,
            n_snps=n_snps,
            n_devices=n_devices,
        )

        if n_devices > 1 and result < n_snps:
            assert result % n_devices == 0, (
                f"Chunk {result} is not aligned to {n_devices} devices"
            )

    @pytest.mark.parametrize("n_devices", [1, 8, 16, 24, 48])
    def test_auto_tune_databricks_scale(self, n_devices):
        """auto_tune_chunk_size at 125k samples, 4GB budget, various device counts."""
        result = auto_tune_chunk_size(
            n_samples=125_000,
            n_filtered=95_000,
            mem_budget_gb=4.0,
            n_devices=n_devices,
        )
        assert result > 0
        assert result <= MAX_SAFE_CHUNK
        assert result <= 95_000

        if n_devices > 1 and result > n_devices:
            assert result % n_devices == 0, (
                f"auto_tune result {result} not aligned to {n_devices} devices"
            )


@pytest.mark.tier0
class TestChunkSizeNeverZero:
    """_compute_chunk_size must never return 0."""

    def test_extreme_n_cvt_does_not_return_zero(self):
        """Very high n_cvt should still return >= 1."""
        result = _compute_chunk_size(
            n_samples=10_000,
            n_snps=500,
            n_cvt=1000,
        )
        assert result >= 1, f"Chunk size must be >= 1, got {result}"

    def test_extreme_n_samples_does_not_return_zero(self):
        """Very large n_samples should still return >= 1."""
        result = _compute_chunk_size(
            n_samples=5_000_000,
            n_snps=1000,
            n_cvt=2,
        )
        assert result >= 1, f"Chunk size must be >= 1, got {result}"

    @pytest.mark.parametrize("n_devices", [1, 8, 64, 256])
    def test_zero_safe_bound_with_devices(self, n_devices):
        """Edge case doesn't interact badly with device alignment."""
        result = _compute_chunk_size(
            n_samples=10_000,
            n_snps=500,
            n_cvt=1000,
            n_devices=n_devices,
        )
        assert result >= 1
