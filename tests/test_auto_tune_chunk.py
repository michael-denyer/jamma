"""Tests for auto_tune_chunk_size() chunk capping behavior.

Verifies that auto_tune_chunk_size respects max_chunk parameter to prevent
excessive memory allocation and int32 overflow.
"""

from jamma.lmm.chunk import MAX_SAFE_CHUNK, auto_tune_chunk_size


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
        """min_chunk should still be the floor."""
        result = auto_tune_chunk_size(
            n_samples=100_000,
            n_filtered=500,  # Fewer SNPs than min_chunk default
            mem_budget_gb=0.0001,  # Tiny budget
            min_chunk=1000,
        )

        assert result >= 1000

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
