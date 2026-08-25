"""Tests for chunk size computation invariants.

Verifies that _compute_chunk_size respects the MAX_SAFE_CHUNK cap and
clamp constraints, and that estimate_streaming_memory prices the
pipeline's double buffering.
"""

import pytest

from jamma.core.chunk import MAX_SAFE_CHUNK, _compute_chunk_size


@pytest.mark.tier0
class TestComputeChunkSize:
    """Tests for _compute_chunk_size with MAX_SAFE_CHUNK cap."""

    def test_small_dataset_no_chunking(self):
        """When n_snps < MAX_SAFE_CHUNK, return n_snps."""
        result = _compute_chunk_size(n_snps=5000)
        assert result == 5000

    def test_large_dataset_caps_at_max_safe(self):
        """When n_snps > MAX_SAFE_CHUNK, cap at MAX_SAFE_CHUNK."""
        result = _compute_chunk_size(n_snps=500_000)
        assert result == MAX_SAFE_CHUNK

    def test_gwas_scale_caps_at_max_safe(self):
        """At GWAS scale (95k SNPs), chunk is MAX_SAFE_CHUNK."""
        chunk = _compute_chunk_size(n_snps=95_000)
        assert chunk == MAX_SAFE_CHUNK

    def test_never_returns_zero(self):
        """Chunk size must always be >= 1, even for degenerate input."""
        assert _compute_chunk_size(n_snps=0) >= 1
        assert _compute_chunk_size(n_snps=1) >= 1

    def test_n_snps_equals_max_safe_chunk(self):
        """When n_snps == MAX_SAFE_CHUNK, return exactly MAX_SAFE_CHUNK."""
        result = _compute_chunk_size(n_snps=MAX_SAFE_CHUNK)
        assert result == MAX_SAFE_CHUNK


@pytest.mark.tier0
@pytest.mark.tier1
def test_compute_chunk_size_with_n_samples():
    """_compute_chunk_size uses memory-aware sizing when n_samples > 0."""
    chunk = _compute_chunk_size(n_snps=1_000_000, n_samples=10_000)
    # At minimum, it should be at least 1000 (the floor)
    assert chunk >= 1000
    # Should not exceed n_snps
    assert chunk <= 1_000_000


@pytest.mark.tier1
def test_compute_chunk_size_backward_compatible():
    """_compute_chunk_size without n_samples uses MAX_SAFE_CHUNK cap (legacy)."""
    chunk = _compute_chunk_size(n_snps=100_000)
    assert chunk == MAX_SAFE_CHUNK  # Falls back to cap without n_samples


@pytest.mark.tier0
class TestComputeChunkSizePipelineBuffers:
    """Tests for _compute_chunk_size pipeline_buffers parameter."""

    def test_pipeline_buffers_halves_budget(self):
        """pipeline_buffers=2 produces at most 60% of pipeline_buffers=1 chunk size.

        This verifies the double-buffer memory accounting is working: when two
        live UtG arrays are needed (current + next), the effective budget halves,
        resulting in a smaller chunk size.

        Uses n_samples=10_000 to keep memory-based sizing above the min clamp
        even with a halved budget, while staying below MAX_SAFE_CHUNK.
        If both hit the min clamp (extreme memory pressure), the test is skipped.
        """
        chunk_1 = _compute_chunk_size(
            n_snps=50_000, n_samples=10_000, pipeline_buffers=1
        )
        chunk_2 = _compute_chunk_size(
            n_snps=50_000, n_samples=10_000, pipeline_buffers=2
        )
        # Both must be positive
        assert chunk_1 >= 1
        assert chunk_2 >= 1

        # If both hit the min clamp, memory pressure is extreme — skip halving check
        min_clamp = 1000
        if chunk_1 == min_clamp and chunk_2 == min_clamp:
            pytest.skip(
                "Both chunks at min clamp — memory too constrained to verify halving"
            )

        # pipeline_buffers=2 should yield a meaningfully smaller chunk
        assert chunk_2 <= chunk_1 * 0.6, (
            f"Expected pipeline_buffers=2 chunk ({chunk_2}) to be at most 60% "
            f"of pipeline_buffers=1 chunk ({chunk_1})"
        )

    def test_pipeline_buffers_default_matches_explicit_one(self):
        """Omitting pipeline_buffers gives the same result as pipeline_buffers=1."""
        chunk_default = _compute_chunk_size(n_snps=50_000, n_samples=1000)
        chunk_explicit = _compute_chunk_size(
            n_snps=50_000, n_samples=1000, pipeline_buffers=1
        )
        # Allow ±1 tolerance: available memory can shift between the two calls,
        # causing a rounding boundary difference in the chunk size calculation.
        assert abs(chunk_default - chunk_explicit) <= 1

    def test_pipeline_buffers_small_snps_never_zero(self):
        """pipeline_buffers=2 with tiny n_snps must return at least 1."""
        result = _compute_chunk_size(n_snps=100, n_samples=1000, pipeline_buffers=2)
        assert result >= 1

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_pipeline_buffers_invalid_raises(self, bad_value):
        """pipeline_buffers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="pipeline_buffers must be >= 1"):
            _compute_chunk_size(
                n_snps=50_000, n_samples=1000, pipeline_buffers=bad_value
            )

    @pytest.mark.parametrize("bad_value", [1.0, 2.0, "2", None])
    def test_pipeline_buffers_type_error_raises(self, bad_value):
        """pipeline_buffers must be int, not float/str/None."""
        with pytest.raises(TypeError, match="pipeline_buffers must be an int"):
            _compute_chunk_size(
                n_snps=50_000, n_samples=1000, pipeline_buffers=bad_value
            )


@pytest.mark.tier0
class TestStreamingMemoryPipelineBuffers:
    """Tests for pipeline_buffers parameter in streaming memory estimators."""

    def test_streaming_memory_double_buffer_rotation_doubles(self):
        """rotation_buffer_gb doubles when pipeline_buffers=2."""
        from jamma.core.memory import estimate_streaming_memory

        est_1 = estimate_streaming_memory(1000, pipeline_buffers=1)
        est_2 = estimate_streaming_memory(1000, pipeline_buffers=2)
        assert est_2.rotation_buffer_gb == pytest.approx(
            2 * est_1.rotation_buffer_gb, rel=1e-10
        )
        assert est_2.total_peak_gb > est_1.total_peak_gb

    def test_streaming_memory_default_matches_single_buffer(self):
        """Omitting pipeline_buffers gives the same total_peak_gb as pipeline_buffers=1.

        Backward compatibility: default call must equal explicit pipeline_buffers=1.
        """
        from jamma.core.memory import estimate_streaming_memory

        est_default = estimate_streaming_memory(1000)
        est_explicit = estimate_streaming_memory(1000, pipeline_buffers=1)
        assert est_default.total_peak_gb == pytest.approx(
            est_explicit.total_peak_gb, rel=1e-10
        )

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_streaming_memory_pipeline_buffers_invalid_raises(self, bad_value):
        """pipeline_buffers < 1 raises ValueError in memory estimators."""
        from jamma.core.memory import estimate_streaming_memory

        with pytest.raises(ValueError, match="pipeline_buffers must be >= 1"):
            estimate_streaming_memory(1000, pipeline_buffers=bad_value)

    @pytest.mark.parametrize("bad_value", [1.0, "2", None])
    def test_streaming_memory_pipeline_buffers_type_error(self, bad_value):
        """pipeline_buffers must be int in memory estimators."""
        from jamma.core.memory import estimate_streaming_memory

        with pytest.raises(TypeError, match="pipeline_buffers must be an int"):
            estimate_streaming_memory(1000, pipeline_buffers=bad_value)

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_numpy_chunk_size_pipeline_buffers_invalid_raises(self, bad_value):
        """pipeline_buffers < 1 raises ValueError in NumPy chunk sizer."""
        from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
        from jamma.lmm.dispatch import DispatchPath

        with pytest.raises(ValueError, match="pipeline_buffers must be >= 1"):
            compute_chunk_size_numpy(
                n_samples=1000,
                n_filtered=50_000,
                dispatch=DispatchPath.FUSED,
                pipeline_buffers=bad_value,
            )

    @pytest.mark.parametrize("bad_value", [1.0, "2", None])
    def test_numpy_chunk_size_pipeline_buffers_type_error(self, bad_value):
        """pipeline_buffers must be int in NumPy chunk sizer."""
        from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
        from jamma.lmm.dispatch import DispatchPath

        with pytest.raises(TypeError, match="pipeline_buffers must be an int"):
            compute_chunk_size_numpy(
                n_samples=1000,
                n_filtered=50_000,
                dispatch=DispatchPath.FUSED,
                pipeline_buffers=bad_value,
            )
