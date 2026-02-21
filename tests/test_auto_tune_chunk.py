"""Tests for chunk size computation invariants.

Verifies that _compute_chunk_size and auto_tune_chunk_size respect int32
safe bounds, clamp constraints, and device alignment contracts.
"""

import pytest

from jamma.lmm.chunk import (
    _MAX_BUFFER_ELEMENTS,
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
class TestComputeChunkSizeInvariants:
    """Tests for _compute_chunk_size int32 safe bound invariants."""

    def test_never_exceeds_safe_bound_large_samples_high_devices(self):
        """Chunk must never exceed int32 safe bound for large n_samples."""
        n_samples = 120_000
        n_snps = 500_000
        n_devices = 64
        n_cvt = 1
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        elements_per_snp = n_samples * n_index
        safe_bound = _MAX_BUFFER_ELEMENTS // elements_per_snp

        result = _compute_chunk_size(
            n_samples=n_samples,
            n_snps=n_snps,
            n_devices=n_devices,
        )
        assert result <= safe_bound

    def test_never_exceeds_safe_bound_alignment_forces_minimum(self):
        """When alignment rounds to zero, result stays within safe bound."""
        # Construct case where safe_bound < n_devices
        # Use huge n_samples so elements_per_snp is large and safe_bound is small
        n_samples = 300_000
        n_cvt = 2  # n_index = 10
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        elements_per_snp = n_samples * n_index
        safe_bound = _MAX_BUFFER_ELEMENTS // elements_per_snp
        # Pick n_devices larger than safe_bound to trigger the edge case
        n_devices = max(safe_bound + 1, 2)

        result = _compute_chunk_size(
            n_samples=n_samples,
            n_snps=1_000_000,
            n_cvt=n_cvt,
            n_devices=n_devices,
        )
        assert result <= safe_bound

    def test_floor_100_does_not_exceed_safe_bound(self):
        """The min-100 floor must not push result above the safe bound."""
        # Large n_samples so safe_bound < 100
        n_samples = 500_000
        n_cvt = 2
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        elements_per_snp = n_samples * n_index
        safe_bound = _MAX_BUFFER_ELEMENTS // elements_per_snp

        if safe_bound < 100:
            result = _compute_chunk_size(
                n_samples=n_samples,
                n_snps=1_000_000,
                n_cvt=n_cvt,
            )
            assert result <= safe_bound

    @pytest.mark.parametrize("n_devices", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_safe_bound_invariant_across_device_counts(self, n_devices):
        """Safe bound invariant holds across a range of device counts."""
        n_samples = 50_000
        n_cvt = 1
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        elements_per_snp = n_samples * n_index
        safe_bound = _MAX_BUFFER_ELEMENTS // elements_per_snp

        result = _compute_chunk_size(
            n_samples=n_samples,
            n_snps=500_000,
            n_devices=n_devices,
        )
        assert result <= safe_bound
