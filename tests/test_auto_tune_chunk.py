"""Tests for chunk-size and streaming-memory pipeline_buffers invariants.

The association-pass chunk sizer is ``compute_chunk_size_numpy``; these tests
pin its pipeline_buffers guards and the estimator's double-buffer pricing.
"""

import pytest


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
