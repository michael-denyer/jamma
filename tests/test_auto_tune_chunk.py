"""Tests for chunk size computation invariants.

Verifies that _compute_chunk_size and auto_tune_chunk_size respect
MAX_SAFE_CHUNK cap, clamp constraints, and device alignment contracts.
"""

import numpy as np
import pytest

from jamma.lmm.chunk import (
    MAX_SAFE_CHUNK,
    _compute_chunk_size,
    auto_tune_chunk_size,
    compute_subchunk_starts,
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

    @pytest.mark.parametrize("n_devices", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_device_alignment(self, n_devices):
        """Chunk is device-aligned when n_devices > 1 and chunking occurs."""
        result = _compute_chunk_size(n_snps=500_000, n_devices=n_devices)
        if n_devices > 1 and result < 500_000:
            assert result % n_devices == 0, (
                f"Chunk {result} is not aligned to {n_devices} devices"
            )

    def test_never_returns_zero(self):
        """Chunk size must always be >= 1, even for degenerate input."""
        assert _compute_chunk_size(n_snps=0) >= 1
        assert _compute_chunk_size(n_snps=1) >= 1

    def test_n_snps_equals_max_safe_chunk(self):
        """When n_snps == MAX_SAFE_CHUNK, return exactly MAX_SAFE_CHUNK."""
        result = _compute_chunk_size(n_snps=MAX_SAFE_CHUNK)
        assert result == MAX_SAFE_CHUNK


@pytest.mark.tier0
class TestChunkSizingAtDatabricksScale:
    """Chunk sizing at Databricks-relevant scale (100k+ samples, many devices).

    Verifies _compute_chunk_size and auto_tune_chunk_size produce valid,
    device-aligned chunks at the scale where JAMMA actually runs.
    """

    @pytest.mark.parametrize("n_devices", [1, 8, 16, 24, 48])
    def test_chunk_device_alignment_at_scale(self, n_devices):
        """Chunk is a multiple of n_devices when n_devices > 1."""
        n_snps = 95_000

        result = _compute_chunk_size(n_snps=n_snps, n_devices=n_devices)

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


@pytest.mark.tier1
def test_clear_caches_not_in_chunk_loop():
    """jax.clear_caches() must be at end-of-run only, not inside chunk loop (RUN-07)."""
    import ast
    from pathlib import Path

    runner_files = [
        Path("src/jamma/lmm/runner_jax.py"),
        Path("src/jamma/lmm/runner_streaming.py"),
    ]

    for fpath in runner_files:
        source = fpath.read_text()
        tree = ast.parse(source)

        # Check that no jax.clear_caches() call is inside a for loop
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        func = inner.func
                        if (
                            isinstance(func, ast.Attribute)
                            and func.attr == "clear_caches"
                            and isinstance(func.value, ast.Name)
                            and func.value.id == "jax"
                        ):
                            pytest.fail(
                                f"jax.clear_caches() inside for loop in {fpath}"
                                f" line {inner.lineno}"
                            )


@pytest.mark.tier1
def test_compute_chunk_size_with_n_samples():
    """_compute_chunk_size uses memory-aware sizing when n_samples > 0."""
    chunk = _compute_chunk_size(n_snps=1_000_000, n_devices=1, n_samples=10_000)
    # At minimum, it should be at least 1000 (the floor)
    assert chunk >= 1000
    # Should not exceed n_snps
    assert chunk <= 1_000_000


@pytest.mark.tier1
def test_compute_chunk_size_backward_compatible():
    """_compute_chunk_size without n_samples uses MAX_SAFE_CHUNK cap (legacy)."""
    chunk = _compute_chunk_size(n_snps=100_000, n_devices=1)
    assert chunk == MAX_SAFE_CHUNK  # Falls back to cap without n_samples


@pytest.mark.tier1
def test_get_device_budget_bytes_cpu_returns_none():
    """_get_device_budget_bytes returns None on CPU (no device memory stats)."""
    from jamma.lmm.chunk import _get_device_budget_bytes

    # On CPU-only machines, device.memory_stats() returns None
    # The function should gracefully return None
    result = _get_device_budget_bytes()
    # On CPU, returns None; on GPU, returns an int — both are valid
    assert result is None or isinstance(result, int)


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
            n_snps=50_000, n_devices=1, n_samples=10_000, pipeline_buffers=1
        )
        chunk_2 = _compute_chunk_size(
            n_snps=50_000, n_devices=1, n_samples=10_000, pipeline_buffers=2
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
        chunk_default = _compute_chunk_size(n_snps=50_000, n_devices=1, n_samples=1000)
        chunk_explicit = _compute_chunk_size(
            n_snps=50_000, n_devices=1, n_samples=1000, pipeline_buffers=1
        )
        assert chunk_default == chunk_explicit

    def test_pipeline_buffers_small_snps_never_zero(self):
        """pipeline_buffers=2 with tiny n_snps must return at least 1."""
        result = _compute_chunk_size(
            n_snps=100, n_devices=1, n_samples=1000, pipeline_buffers=2
        )
        assert result >= 1

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_pipeline_buffers_invalid_raises(self, bad_value):
        """pipeline_buffers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="pipeline_buffers must be >= 1"):
            _compute_chunk_size(
                n_snps=50_000, n_devices=1, n_samples=1000, pipeline_buffers=bad_value
            )

    @pytest.mark.parametrize("bad_value", [1.0, 2.0, "2", None])
    def test_pipeline_buffers_type_error_raises(self, bad_value):
        """pipeline_buffers must be int, not float/str/None."""
        with pytest.raises(TypeError, match="pipeline_buffers must be an int"):
            _compute_chunk_size(
                n_snps=50_000, n_devices=1, n_samples=1000, pipeline_buffers=bad_value
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

    def test_lmm_streaming_memory_double_buffer(self):
        """estimate_lmm_streaming_memory(pipeline_buffers=2).total_peak_gb is higher."""
        from jamma.core.memory import estimate_lmm_streaming_memory

        est_1 = estimate_lmm_streaming_memory(1000, n_snps=10000, pipeline_buffers=1)
        est_2 = estimate_lmm_streaming_memory(1000, n_snps=10000, pipeline_buffers=2)
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
    def test_lmm_streaming_memory_pipeline_buffers_invalid_raises(self, bad_value):
        """pipeline_buffers < 1 raises ValueError in LMM memory estimator."""
        from jamma.core.memory import estimate_lmm_streaming_memory

        with pytest.raises(ValueError, match="pipeline_buffers must be >= 1"):
            estimate_lmm_streaming_memory(
                1000, n_snps=10000, pipeline_buffers=bad_value
            )

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_numpy_chunk_size_pipeline_buffers_invalid_raises(self, bad_value):
        """pipeline_buffers < 1 raises ValueError in NumPy chunk sizer."""
        from jamma.lmm.runner_numpy import _compute_chunk_size_numpy

        with pytest.raises(ValueError, match="pipeline_buffers must be >= 1"):
            _compute_chunk_size_numpy(
                n_samples=1000, n_filtered=50_000, pipeline_buffers=bad_value
            )

    @pytest.mark.parametrize("bad_value", [1.0, "2", None])
    def test_numpy_chunk_size_pipeline_buffers_type_error(self, bad_value):
        """pipeline_buffers must be int in NumPy chunk sizer."""
        from jamma.lmm.runner_numpy import _compute_chunk_size_numpy

        with pytest.raises(TypeError, match="pipeline_buffers must be an int"):
            _compute_chunk_size_numpy(
                n_samples=1000, n_filtered=50_000, pipeline_buffers=bad_value
            )


@pytest.mark.tier0
class TestComputeSubchunkStarts:
    """Tests for compute_subchunk_starts() tail merging."""

    def test_no_tail_issue_single_device(self):
        """Single device never merges — no sharding constraint."""
        starts = compute_subchunk_starts(50000, 49992, n_devices=1)
        assert starts == [0, 49992]

    def test_tail_smaller_than_n_devices_is_merged(self):
        """Tail of 8 SNPs with 24 devices must be merged into previous chunk."""
        # 50000 SNPs, chunk_size=49992 → tail=8, 8 < 24 → merge
        starts = compute_subchunk_starts(50000, 49992, n_devices=24)
        assert starts == [0], "Tail of 8 should be merged into first sub-chunk"

    def test_tail_equal_to_n_devices_not_merged(self):
        """Tail exactly equal to n_devices is valid — no merge needed."""
        # 49992 + 24 = 50016 → tail=24, 24 >= 24 → keep
        starts = compute_subchunk_starts(50016, 49992, n_devices=24)
        assert starts == [0, 49992]

    def test_tail_larger_than_n_devices_not_merged(self):
        """Tail larger than n_devices is valid — no merge needed."""
        starts = compute_subchunk_starts(50100, 49992, n_devices=24)
        assert starts == [0, 49992]

    def test_single_chunk_no_tail(self):
        """When n_subset <= chunk_size, only one sub-chunk exists."""
        starts = compute_subchunk_starts(45000, 49992, n_devices=24)
        assert starts == [0]

    def test_exact_multiple_no_tail(self):
        """When n_subset is exact multiple of chunk_size, no tail exists."""
        starts = compute_subchunk_starts(99984, 49992, n_devices=24)
        assert starts == [0, 49992]

    def test_many_chunks_with_small_tail(self):
        """Multiple chunks where last tail is too small."""
        # 3 * 49992 = 149976, total = 149980, tail = 4 < 24 → merge
        starts = compute_subchunk_starts(149980, 49992, n_devices=24)
        assert starts == [0, 49992, 99984], "Tail of 4 merged into third sub-chunk"

    def test_derived_ends_cover_all_snps(self):
        """Ends derived from starts must cover all n_subset SNPs.

        Regression test for bug where starts.pop() merged the tail but
        _prepare_jax_chunk still capped at start + chunk_size, silently
        dropping the tail SNPs.
        """
        n_subset = 50000
        chunk_size = 49992
        n_devices = 24
        starts = compute_subchunk_starts(n_subset, chunk_size, n_devices)
        # Derive ends the same way runners do
        ends = [
            starts[i + 1] if i + 1 < len(starts) else n_subset
            for i in range(len(starts))
        ]
        # Total coverage must equal n_subset — no dropped SNPs
        total = sum(e - s for s, e in zip(starts, ends, strict=True))
        assert total == n_subset, (
            f"Derived ranges cover {total} SNPs, expected {n_subset}. "
            f"starts={starts}, ends={ends}"
        )

    def test_derived_ends_multi_chunk(self):
        """Multi-chunk case: ends cover everything including merged tail."""
        n_subset = 149980
        chunk_size = 49992
        n_devices = 24
        starts = compute_subchunk_starts(n_subset, chunk_size, n_devices)
        ends = [
            starts[i + 1] if i + 1 < len(starts) else n_subset
            for i in range(len(starts))
        ]
        total = sum(e - s for s, e in zip(starts, ends, strict=True))
        assert total == n_subset


jax = pytest.importorskip("jax")


@pytest.mark.requires_jax
@pytest.mark.tier1
class TestShardingTailRegression:
    """Regression test for IndivisibleError with multi-device tail sub-chunks.

    When jax_chunk_size doesn't divide the file chunk evenly, the tail
    sub-chunk can have fewer SNPs than n_devices. Without the
    compute_subchunk_starts fix, this causes:
        IndivisibleError: shape=[8, 3] is incompatible with
        mesh_shape=OrderedDict({'snps': 24})

    These tests verify the fix by running the full JAX compute pipeline
    on deliberately small sub-chunks that would have triggered the error.
    """

    def test_tail_subchunk_with_multi_device_sharding(self):
        """Padded tail sub-chunk must not cause IndivisibleError.

        Simulates the exact scenario: 8 SNPs padded to n_devices,
        sharded across the mesh, then processed through the full
        Uab → Iab → golden section pipeline.
        """
        import jax.numpy as jnp
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from jamma.lmm.likelihood_jax import (
            batch_compute_iab,
            batch_compute_uab,
            golden_section_optimize_lambda,
        )
        from jamma.lmm.prepare import DevicePlacement, prepare_utg_chunk

        n_devices = len(jax.devices("cpu"))
        if n_devices < 2:
            pytest.skip("Need >= 2 JAX CPU devices for sharding regression test")

        mesh = Mesh(np.array(jax.devices("cpu")), ("snps",))
        snp_spec = NamedSharding(mesh, P(None, "snps"))
        rep_spec = NamedSharding(mesh, P())
        placement = DevicePlacement(snp=snp_spec, rep=rep_spec, n_devices=n_devices)

        n_samples = 100
        # Simulate tail: fewer actual SNPs than n_devices
        n_actual = max(1, n_devices - 2)
        geno_chunk = np.random.default_rng(42).standard_normal((n_samples, n_actual))
        U = np.eye(n_samples)

        # prepare_utg_chunk pads to n_devices multiple
        UtG_np, actual_len = prepare_utg_chunk(
            geno_chunk, U, placement, rotation_threads=1
        )
        assert actual_len == n_actual
        assert UtG_np.shape[1] % n_devices == 0, (
            f"UtG not padded to device multiple: {UtG_np.shape[1]}"
        )

        # device_put with sharding — this is where the old bug manifested
        UtG_jax = jax.device_put(UtG_np, snp_spec)

        eigenvalues = jax.device_put(jnp.ones(n_samples, dtype=jnp.float64), rep_spec)
        UtW = jax.device_put(jnp.ones((n_samples, 1), dtype=jnp.float64), rep_spec)
        Uty = jax.device_put(jnp.ones(n_samples, dtype=jnp.float64), rep_spec)

        # Full compute pipeline — would throw IndivisibleError without fix
        Uab = batch_compute_uab(1, UtW, Uty, UtG_jax)
        Iab = batch_compute_iab(1, Uab)
        lambdas, logls = golden_section_optimize_lambda(1, eigenvalues, Uab, Iab)

        assert lambdas.shape[0] == UtG_np.shape[1]
        assert logls.shape[0] == UtG_np.shape[1]

    def test_compute_subchunk_starts_prevents_indivisible_tail(self):
        """Verify compute_subchunk_starts prevents the exact failing scenario.

        With 24 devices and jax_chunk_size=49992, a file chunk of 50000
        leaves a tail of 8 — too small for 24-way sharding.
        """
        # The exact parameters from the production failure
        starts = compute_subchunk_starts(n_subset=50000, chunk_size=49992, n_devices=24)
        # Tail was 8, which is < 24, so it must be merged
        assert len(starts) == 1
        assert starts == [0]

        # The merged sub-chunk processes all 50000 SNPs
        # prepare_utg_chunk will pad 50000 to 50016 (next multiple of 24)
        assert 50000 % 24 != 0  # confirms padding is needed
        next_multiple = ((50000 + 23) // 24) * 24
        assert next_multiple == 50016
