"""Tests for CPU device sharding infrastructure.

Tests for:
- configure_jax() multi-device setup (via get_jax_info observability)
- get_blas_thread_count() JAX-device-aware thread reduction
- _compute_chunk_size() device-count alignment
- auto_tune_chunk_size() device-count alignment pass-through
- Zero-padding round-trip correctness for multi-device distribution
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np

from jamma.core.jax_config import _configure_cpu_devices, get_jax_info
from jamma.core.threading import get_blas_thread_count
from jamma.lmm.chunk import _compute_chunk_size, auto_tune_chunk_size
from jamma.lmm.compute import _compute_lmm_chunk
from jamma.lmm.likelihood_jax import batch_compute_uab
from jamma.lmm.prepare import _setup_cpu_sharding

pytestmark = pytest.mark.requires_jax


@pytest.mark.tier0
class TestJaxDeviceConfiguration:
    """Tests for configure_jax() device sharding infrastructure."""

    def test_get_jax_info_includes_n_cpu_devices(self):
        """get_jax_info() must report n_cpu_devices for observability."""
        info = get_jax_info()
        assert "n_cpu_devices" in info
        assert isinstance(info["n_cpu_devices"], int)
        assert info["n_cpu_devices"] >= 1

    def test_n_cpu_devices_matches_jax_devices(self):
        """n_cpu_devices in get_jax_info() must match len(jax.devices('cpu'))."""
        info = get_jax_info()
        assert info["n_cpu_devices"] == len(jax.devices("cpu"))

    def test_auto_config_produces_at_least_one_device(self):
        """Auto-config must produce at least 1 device regardless of hardware."""
        # conftest.py calls ensure_jax_configured() which triggers auto-config.
        # This test verifies the result is sane.
        assert len(jax.devices("cpu")) >= 1


@pytest.mark.tier0
class TestBlasThreadJaxAwareness:
    """Tests for get_blas_thread_count() JAX device-aware reduction."""

    def test_returns_positive_with_multiple_jax_devices(self):
        """Thread count must be positive even when JAX has many devices."""
        # Use current JAX state (conftest auto-configured)
        n = get_blas_thread_count()
        assert n >= 1

    def test_env_override_takes_priority_over_jax_devices(self, monkeypatch):
        """JAMMA_BLAS_THREADS env var must win over JAX device reduction."""
        monkeypatch.setenv("JAMMA_BLAS_THREADS", "3")
        # Even with multiple JAX devices, env var should win
        assert get_blas_thread_count() == 3

    def test_reduces_threads_for_multiple_jax_devices(self, monkeypatch):
        """Thread count must reduce proportionally to JAX device count."""
        import jax
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or 4

        # Mock jax.devices to return a list of N device objects.
        # We need n_devices > 1 to trigger the reduction path.
        # Patch the jax module directly since threading.py does `import jax` inside
        # get_blas_thread_count(), then calls jax.devices("cpu").
        n_devices = 4
        mock_devices = [object() for _ in range(n_devices)]

        with patch.object(jax, "devices", return_value=mock_devices):
            n = get_blas_thread_count()

        expected = max(1, physical_cores // n_devices)
        assert n == expected

    def test_single_jax_device_uses_physical_cores(self, monkeypatch):
        """With 1 JAX device, threads should equal physical core count."""
        import os

        import jax
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
        max_threads = os.cpu_count() or 64

        mock_devices = [object()]  # single device

        with patch.object(jax, "devices", return_value=mock_devices):
            n = get_blas_thread_count()

        expected = max(1, min(physical_cores, max_threads))
        assert n == expected


@pytest.mark.tier0
class TestComputeChunkSizeAlignment:
    """Tests for _compute_chunk_size() n_devices alignment."""

    def test_default_n_devices_is_backward_compatible(self):
        """n_devices=1 must produce same result as calling without it."""
        result_default = _compute_chunk_size(10768)
        result_explicit = _compute_chunk_size(10768, n_devices=1)
        assert result_default == result_explicit

    def test_n_devices_4_result_is_multiple_of_4(self):
        """Result with n_devices=4 must be a multiple of 4."""
        result = _compute_chunk_size(10768, n_devices=4)
        assert result % 4 == 0, f"Expected multiple of 4, got {result}"

    def test_n_devices_8_result_is_multiple_of_8(self):
        """Result with n_devices=8 must be a multiple of 8."""
        result = _compute_chunk_size(100000, n_devices=8)
        if result < 100000:  # only check when chunking actually occurs
            assert result % 8 == 0, f"Expected multiple of 8, got {result}"

    def test_n_devices_1_no_alignment_applied(self):
        """n_devices=1 must not modify the chunk size calculation."""
        result_no_devices = _compute_chunk_size(50000, n_devices=1)
        assert result_no_devices == 50000

    def test_small_n_snps_with_device_alignment(self):
        """Small n_snps returns n_snps even with device alignment."""
        result = _compute_chunk_size(n_snps=10, n_devices=4)
        assert result == 10

    def test_no_chunking_needed_returns_n_snps(self):
        """When n_snps < MAX_SAFE_CHUNK, n_snps is returned unchanged."""
        result = _compute_chunk_size(100, n_devices=4)
        assert result == 100

    def test_n_devices_2_mouse_hs1940_scale(self):
        """Test alignment at mouse_hs1940 scale (10768 SNPs)."""
        result = _compute_chunk_size(10768, n_devices=2)
        assert result == 10768


@pytest.mark.tier0
class TestAutoTuneChunkSizeDeviceAlignment:
    """Tests for auto_tune_chunk_size() n_devices pass-through."""

    def test_default_n_devices_is_backward_compatible(self):
        """Calling without n_devices must match n_devices=1."""
        result_default = auto_tune_chunk_size(10000, 50000)
        result_explicit = auto_tune_chunk_size(10000, 50000, n_devices=1)
        assert result_default == result_explicit

    def test_n_devices_4_result_is_multiple_of_4(self):
        """When n_devices=4, chunk must be a multiple of 4."""
        result = auto_tune_chunk_size(
            n_samples=10000,
            n_filtered=100000,
            mem_budget_gb=4.0,
            n_devices=4,
        )
        assert result % 4 == 0, f"Expected multiple of 4, got {result}"

    def test_min_chunk_still_enforced_with_n_devices(self):
        """min_chunk floor applies when n_filtered allows it."""
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=50_000,
            mem_budget_gb=0.001,
            min_chunk=1000,
            n_devices=4,
        )
        assert result >= 1000

    def test_n_filtered_caps_below_min_chunk_with_n_devices(self):
        """n_filtered ceiling takes precedence over min_chunk floor."""
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=500,
            mem_budget_gb=0.001,
            min_chunk=1000,
            n_devices=4,
        )
        assert result <= 500

    def test_min_chunk_result_is_device_aligned(self):
        """When min_chunk forces the result up, it must still be device-aligned.

        Device alignment takes precedence: if min_chunk=1001 and n_devices=4,
        the result is rounded down to 1000 (nearest multiple of 4).
        """
        result = auto_tune_chunk_size(
            n_samples=1000,
            n_filtered=50000,
            mem_budget_gb=0.001,
            min_chunk=1001,  # not a multiple of 4
            n_devices=4,
        )
        assert result % 4 == 0, f"Expected multiple of 4, got {result}"
        assert result >= 1000  # rounded down from min_chunk=1001


@pytest.mark.tier0
class TestConfigureCpuDevices:
    """Tests for _configure_cpu_devices() priority chain and validation."""

    def test_invalid_env_var_raises_value_error(self, monkeypatch):
        """Non-integer JAMMA_JAX_DEVICES must raise ValueError."""
        monkeypatch.setenv("JAMMA_JAX_DEVICES", "four")
        with pytest.raises(ValueError, match="not a valid integer"):
            _configure_cpu_devices(None)

    def test_negative_env_var_raises_value_error(self, monkeypatch):
        """Negative JAMMA_JAX_DEVICES must raise ValueError."""
        monkeypatch.setenv("JAMMA_JAX_DEVICES", "-1")
        with pytest.raises(ValueError, match="must be >= 1"):
            _configure_cpu_devices(None)

    def test_zero_env_var_raises_value_error(self, monkeypatch):
        """JAMMA_JAX_DEVICES=0 must raise ValueError."""
        monkeypatch.setenv("JAMMA_JAX_DEVICES", "0")
        with pytest.raises(ValueError, match="must be >= 1"):
            _configure_cpu_devices(None)

    def test_env_var_strips_whitespace(self, monkeypatch):
        """Trailing whitespace in JAMMA_JAX_DEVICES must be handled."""
        monkeypatch.setenv("JAMMA_JAX_DEVICES", " 1 ")
        # Should not raise — int(" 1 ".strip()) = 1
        _configure_cpu_devices(None)

    def test_env_var_1_does_not_call_config_update(self, monkeypatch):
        """JAMMA_JAX_DEVICES=1 must not call jax.config.update."""
        monkeypatch.setenv("JAMMA_JAX_DEVICES", "1")
        with patch.object(jax.config, "update") as mock_update:
            _configure_cpu_devices(None)
        # n=1 should NOT call jax.config.update (leaves JAX default)
        for call in mock_update.call_args_list:
            assert call[0][0] != "jax_num_cpu_devices"

    def test_n_cpu_devices_zero_raises_value_error(self):
        """n_cpu_devices=0 must raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            _configure_cpu_devices(n_cpu_devices=0)

    def test_n_cpu_devices_negative_raises_value_error(self):
        """n_cpu_devices=-3 must raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            _configure_cpu_devices(n_cpu_devices=-3)

    def test_env_var_overrides_argument(self, monkeypatch):
        """JAMMA_JAX_DEVICES env var must take priority over n_cpu_devices arg."""
        monkeypatch.setenv("JAMMA_JAX_DEVICES", "1")
        # Even with n_cpu_devices=8, the env var should win
        with patch.object(jax.config, "update") as mock_update:
            _configure_cpu_devices(n_cpu_devices=8)
        # n=1 from env var → should NOT set jax_num_cpu_devices
        for call in mock_update.call_args_list:
            assert call[0][0] != "jax_num_cpu_devices"


@pytest.mark.tier0
class TestSetupCpuSharding:
    """Tests for _setup_cpu_sharding()."""

    def test_single_device_returns_none_none(self):
        """Single CPU device must return (None, None)."""
        with patch.object(jax, "devices", return_value=[MagicMock()]):
            snp_spec, rep_spec = _setup_cpu_sharding()
        assert snp_spec is None
        assert rep_spec is None

    def test_returns_none_none_on_actual_single_device(self):
        """On actual test machine with 1 device, returns (None, None)."""
        cpu_devices = jax.devices("cpu")
        if len(cpu_devices) > 1:
            pytest.skip("Test requires single CPU device")
        snp_spec, rep_spec = _setup_cpu_sharding()
        assert snp_spec is None
        assert rep_spec is None

    def test_multi_device_returns_sharding_specs(self):
        """Multiple CPU devices must return valid NamedSharding specs."""
        cpu_devices = jax.devices("cpu")
        if len(cpu_devices) <= 1:
            pytest.skip("Test requires multiple CPU devices")
        snp_spec, rep_spec = _setup_cpu_sharding()
        assert snp_spec is not None
        assert rep_spec is not None

    def test_sharding_failure_falls_back_to_none(self):
        """If Mesh construction fails, must return (None, None) with warning."""

        mock_devices = [MagicMock(), MagicMock()]
        with (
            patch.object(jax, "devices", return_value=mock_devices),
            patch("jamma.lmm.prepare.Mesh", side_effect=RuntimeError("test error")),
        ):
            snp_spec, rep_spec = _setup_cpu_sharding()
        assert snp_spec is None
        assert rep_spec is None


@pytest.mark.tier0
class TestMultiDevicePaddingRoundTrip:
    """Verify zero-padding for device alignment preserves computation correctness.

    When UtG columns aren't evenly divisible by n_devices, the runner pads
    with zero columns before device_put and strips padding from results via
    actual_len slicing. These tests verify the padding is mathematically
    transparent: padded results[:actual_len] == unpadded results exactly.
    """

    @pytest.fixture
    def synthetic_lmm_inputs(self):
        """Create small synthetic LMM inputs for padding tests."""
        rng = np.random.default_rng(42)
        n_samples = 50
        n_cvt = 1

        # Rotated covariates and phenotype (as if U.T @ W, U.T @ y)
        UtW = jnp.array(rng.standard_normal((n_samples, n_cvt)), dtype=jnp.float64)
        Uty = jnp.array(rng.standard_normal(n_samples), dtype=jnp.float64)

        return n_samples, n_cvt, UtW, Uty

    def test_padded_uab_matches_unpadded(self, synthetic_lmm_inputs):
        """batch_compute_uab on zero-padded UtG must match unpadded for real columns."""
        n_samples, n_cvt, UtW, Uty = synthetic_lmm_inputs
        rng = np.random.default_rng(123)
        n_snps = 7  # deliberately not divisible by 2, 4, or 8
        n_devices = 4

        UtG = jnp.array(rng.standard_normal((n_samples, n_snps)), dtype=jnp.float64)

        # Unpadded reference
        Uab_ref = batch_compute_uab(n_cvt, UtW, Uty, UtG)
        Uab_ref_np = np.asarray(Uab_ref)

        # Pad to n_devices multiple (same logic as runner_jax._prepare_chunk)
        dev_pad = n_devices - (n_snps % n_devices)
        UtG_padded = jnp.pad(UtG, ((0, 0), (0, dev_pad)), mode="constant")
        assert UtG_padded.shape[1] % n_devices == 0

        # Padded computation
        Uab_padded = batch_compute_uab(n_cvt, UtW, Uty, UtG_padded)
        Uab_padded_np = np.asarray(Uab_padded)

        # Slice to actual_len — must be bitwise identical (zero-padding
        # cannot affect independent per-column computations)
        np.testing.assert_array_equal(
            Uab_padded_np[:n_snps],
            Uab_ref_np,
            err_msg="Padded Uab[:actual_len] differs from unpadded reference",
        )

    def test_padding_columns_produce_deterministic_zeros(self, synthetic_lmm_inputs):
        """Zero-padded columns must produce deterministic Uab (no NaN/Inf)."""
        n_samples, n_cvt, UtW, Uty = synthetic_lmm_inputs
        n_devices = 4
        n_snps_padded = n_devices * 3  # 12 columns, all zero

        UtG_zeros = jnp.zeros((n_samples, n_snps_padded), dtype=jnp.float64)
        Uab_zeros = batch_compute_uab(n_cvt, UtW, Uty, UtG_zeros)
        Uab_np = np.asarray(Uab_zeros)

        assert np.all(np.isfinite(Uab_np)), "Zero-column Uab contains NaN or Inf"

    @pytest.mark.parametrize(
        "n_snps,n_devices",
        [
            (1, 2),  # 1 real + 1 pad
            (3, 4),  # 3 real + 1 pad
            (5, 4),  # 5 real + 3 pad
            (7, 8),  # 7 real + 1 pad
            (13, 4),  # 13 real + 3 pad
            (16, 4),  # 16 real + 0 pad (exact multiple, no padding)
        ],
    )
    def test_various_padding_sizes(self, synthetic_lmm_inputs, n_snps, n_devices):
        """Padding correctness across various SNP count / device count combos."""
        n_samples, n_cvt, UtW, Uty = synthetic_lmm_inputs
        rng = np.random.default_rng(n_snps * 100 + n_devices)

        UtG = jnp.array(rng.standard_normal((n_samples, n_snps)), dtype=jnp.float64)

        # Reference: no padding
        Uab_ref = np.asarray(batch_compute_uab(n_cvt, UtW, Uty, UtG))

        # Padded version
        remainder = n_snps % n_devices
        if remainder != 0:
            dev_pad = n_devices - remainder
            UtG_padded = jnp.pad(UtG, ((0, 0), (0, dev_pad)), mode="constant")
        else:
            UtG_padded = UtG

        assert UtG_padded.shape[1] % n_devices == 0
        Uab_padded = np.asarray(batch_compute_uab(n_cvt, UtW, Uty, UtG_padded))

        np.testing.assert_array_equal(
            Uab_padded[:n_snps],
            Uab_ref,
            err_msg=f"Padding mismatch: n_snps={n_snps}, n_devices={n_devices}",
        )


@pytest.mark.tier0
class TestComputeLmmChunkPaddingRoundTrip:
    """Verify _compute_lmm_chunk results are unaffected by device-alignment padding.

    This extends TestMultiDevicePaddingRoundTrip to cover the full computation
    pipeline (Wald stats, golden section optimization, result slicing), not just
    batch_compute_uab. Ensures padded results[:actual_len] == unpadded results.
    """

    @pytest.fixture
    def synthetic_pipeline_inputs(self):
        """Create synthetic inputs for the full _compute_lmm_chunk pipeline."""
        rng = np.random.default_rng(99)
        n_samples = 50
        n_cvt = 1

        # Simulate eigenvalues (positive, decreasing)
        eigenvalues_np = np.sort(rng.uniform(0.1, 10.0, size=n_samples))[::-1]
        eigenvalues = jnp.array(eigenvalues_np, dtype=jnp.float64)

        # Rotated covariates and phenotype
        UtW = jnp.array(rng.standard_normal((n_samples, n_cvt)), dtype=jnp.float64)
        Uty = jnp.array(rng.standard_normal(n_samples), dtype=jnp.float64)

        return n_samples, n_cvt, eigenvalues, UtW, Uty

    def test_padded_lmm_chunk_matches_unpadded(self, synthetic_pipeline_inputs):
        """_compute_lmm_chunk (Wald) on padded UtG must match unpadded for real SNPs."""
        n_samples, n_cvt, eigenvalues, UtW, Uty = synthetic_pipeline_inputs
        lmm_mode = 1  # Wald — no null model dependency
        rng = np.random.default_rng(lmm_mode * 1000)
        n_snps = 7
        n_devices = 4

        UtG = jnp.array(rng.standard_normal((n_samples, n_snps)), dtype=jnp.float64)

        # Unpadded reference
        Uab_ref = batch_compute_uab(n_cvt, UtW, Uty, UtG)
        ref_result = _compute_lmm_chunk(
            lmm_mode, n_cvt, eigenvalues, Uab_ref, n_samples
        )

        # Padded version
        dev_pad = n_devices - (n_snps % n_devices)
        UtG_padded = jnp.pad(UtG, ((0, 0), (0, dev_pad)), mode="constant")
        Uab_padded = batch_compute_uab(n_cvt, UtW, Uty, UtG_padded)
        padded_result = _compute_lmm_chunk(
            lmm_mode, n_cvt, eigenvalues, Uab_padded, n_samples
        )

        # Compare real SNP results (slice to actual_len)
        for key in ref_result:
            if ref_result[key] is None:
                continue
            ref_arr = np.asarray(ref_result[key])
            padded_arr = np.asarray(padded_result[key][:n_snps])
            np.testing.assert_array_equal(
                padded_arr,
                ref_arr,
                err_msg=f"mode={lmm_mode}, key={key}: padded differs from unpadded",
            )
