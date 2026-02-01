"""Tests for JAMMA validation framework."""

import numpy as np
import pytest

from jamma.validation import (
    ComparisonResult,
    ToleranceConfig,
    compare_arrays,
    compare_kinship_matrices,
    load_gemma_kinship,
)


class TestToleranceConfig:
    """Tests for ToleranceConfig dataclass."""

    def test_tolerance_config_defaults(self):
        """Verify default tolerance values match empirical observations.

        These tolerances are based on JAMMA vs GEMMA validation tests.
        """
        config = ToleranceConfig()

        # Beta: max observed 8.5e-3 due to lambda sensitivity
        assert config.beta_rtol == 1e-2
        # SE: follows beta sensitivity pattern
        assert config.se_rtol == 1e-5
        # P-values: CDF implementation differences (max observed: 4.1e-5)
        assert config.pvalue_rtol == 1e-4
        # Kinship: direct matrix computation
        assert config.kinship_rtol == 1e-8
        # Log-likelihood: max observed 3.2e-7
        assert config.logl_rtol == 1e-6
        # Lambda: Brent optimization convergence (max observed: 1.2e-5)
        assert config.lambda_rtol == 2e-5
        # AF: JAMMA reports MAF, GEMMA reports AF (max diff: 0.04)
        assert config.af_rtol == 0.05
        assert config.atol == 1e-12

    def test_tolerance_config_strict(self):
        """Verify strict factory produces tighter tolerances."""
        default = ToleranceConfig()
        strict = ToleranceConfig.strict()

        # Strict should be at least 10x tighter than default
        assert strict.beta_rtol < default.beta_rtol
        assert strict.se_rtol < default.se_rtol
        assert strict.pvalue_rtol < default.pvalue_rtol
        assert strict.kinship_rtol < default.kinship_rtol
        assert strict.atol < default.atol

        # Verify specific strict values
        assert strict.kinship_rtol == 1e-10

    def test_tolerance_config_relaxed(self):
        """Verify relaxed factory produces looser tolerances."""
        default = ToleranceConfig()
        relaxed = ToleranceConfig.relaxed()

        # Relaxed should be looser than default
        assert relaxed.beta_rtol > default.beta_rtol
        assert relaxed.se_rtol > default.se_rtol
        assert relaxed.pvalue_rtol > default.pvalue_rtol
        assert relaxed.kinship_rtol > default.kinship_rtol
        assert relaxed.atol > default.atol


class TestCompareArrays:
    """Tests for compare_arrays function."""

    def test_compare_arrays_pass_identical(self):
        """Two identical arrays should pass comparison."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = compare_arrays(a, b, rtol=1e-6, atol=1e-12, name="test")

        assert result.passed is True
        assert result.max_abs_diff == 0.0
        assert result.max_rel_diff == 0.0
        assert result.worst_location is None

    def test_compare_arrays_within_tolerance(self):
        """Arrays within tolerance should pass."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0 + 1e-8, 2.0 - 1e-8, 3.0 + 1e-8])

        result = compare_arrays(a, b, rtol=1e-6, atol=1e-12, name="test")

        assert result.passed is True
        assert result.max_abs_diff < 1e-7

    def test_compare_arrays_fail_outside_tolerance(self):
        """Arrays outside tolerance should fail with informative message."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])  # Last element differs by 1.0

        result = compare_arrays(a, b, rtol=1e-6, atol=1e-12, name="test")

        assert result.passed is False
        assert result.max_abs_diff == pytest.approx(1.0)
        assert result.worst_location == (2,)
        assert "test comparison failed" in result.message
        assert "(2,)" in result.message

    def test_compare_arrays_shape_mismatch(self):
        """Shape mismatch should fail with appropriate message."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])

        result = compare_arrays(a, b, rtol=1e-6, atol=1e-12, name="test")

        assert result.passed is False
        assert "shape mismatch" in result.message

    def test_compare_arrays_2d(self):
        """2D array comparison should work correctly."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [3.0, 5.0]])  # [1,1] differs

        result = compare_arrays(a, b, rtol=1e-6, atol=1e-12, name="matrix")

        assert result.passed is False
        assert result.worst_location == (1, 1)

    def test_compare_arrays_near_zero(self):
        """Values near zero should use absolute tolerance."""
        a = np.array([0.0, 1e-13, 2e-13])
        b = np.array([1e-13, 0.0, 1e-13])

        # These small differences should pass with atol=1e-12
        result = compare_arrays(a, b, rtol=1e-6, atol=1e-12, name="test")

        assert result.passed is True


class TestCompareKinshipMatrices:
    """Tests for compare_kinship_matrices function."""

    def test_compare_kinship_identical(self):
        """Identical kinship matrices should pass."""
        K = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.3], [0.25, 0.3, 1.0]])

        result = compare_kinship_matrices(K, K.copy())

        assert result.passed is True
        assert result.max_abs_diff == 0.0

    def test_compare_kinship_within_tolerance(self):
        """Kinship matrices with small differences should pass."""
        K1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        K2 = K1 + 1e-10  # Within default kinship_rtol of 1e-8

        result = compare_kinship_matrices(K1, K2)

        assert result.passed is True

    def test_compare_kinship_outside_tolerance(self):
        """Kinship matrices with large differences should fail."""
        K1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        K2 = np.array([[1.0, 0.6], [0.6, 1.0]])  # 0.1 difference in off-diagonals

        result = compare_kinship_matrices(K1, K2)

        assert result.passed is False
        assert result.max_abs_diff == pytest.approx(0.1)

    def test_compare_kinship_custom_config(self):
        """Custom tolerance config should be respected."""
        K1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        K2 = K1 + 1e-7  # Outside default (1e-8), but within relaxed (1e-6)

        # Default config should fail (1e-7 > 1e-8 rtol)
        result_default = compare_kinship_matrices(K1, K2)
        assert result_default.passed is False

        # Relaxed config should pass (1e-7 < 1e-6 rtol)
        relaxed_config = ToleranceConfig.relaxed()
        result_relaxed = compare_kinship_matrices(K1, K2, config=relaxed_config)
        assert result_relaxed.passed is True


class TestLoadGemmaKinship:
    """Tests for load_gemma_kinship function."""

    def test_load_gemma_kinship_valid_file(self, tmp_path):
        """Load kinship matrix from valid file."""
        # Create a test kinship matrix file
        K = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.3], [0.25, 0.3, 1.0]])
        kinship_path = tmp_path / "test.cXX.txt"
        np.savetxt(kinship_path, K)

        # Load and verify
        loaded = load_gemma_kinship(kinship_path)

        assert loaded.shape == (3, 3)
        np.testing.assert_allclose(loaded, K)

    def test_load_gemma_kinship_preserves_precision(self, tmp_path):
        """Kinship values should be loaded with full precision."""
        K = np.array([[1.123456789012345, 0.5], [0.5, 0.987654321098765]])
        kinship_path = tmp_path / "precision.txt"
        np.savetxt(kinship_path, K, fmt="%.15e")

        loaded = load_gemma_kinship(kinship_path)

        # Should match within machine precision
        np.testing.assert_allclose(loaded, K, rtol=1e-14)

    def test_load_gemma_kinship_file_not_found(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            load_gemma_kinship(nonexistent)


class TestComparisonResult:
    """Tests for ComparisonResult dataclass structure."""

    def test_comparison_result_structure(self):
        """ComparisonResult should have expected fields."""
        result = ComparisonResult(
            passed=True,
            max_abs_diff=1e-10,
            max_rel_diff=1e-8,
            worst_location=None,
            message="test passed",
        )

        assert hasattr(result, "passed")
        assert hasattr(result, "max_abs_diff")
        assert hasattr(result, "max_rel_diff")
        assert hasattr(result, "worst_location")
        assert hasattr(result, "message")

    def test_comparison_result_failed_structure(self):
        """Failed ComparisonResult should include location info."""
        result = ComparisonResult(
            passed=False,
            max_abs_diff=0.5,
            max_rel_diff=0.1,
            worst_location=(2, 3),
            message="comparison failed at (2, 3)",
        )

        assert result.passed is False
        assert result.worst_location == (2, 3)
        assert "(2, 3)" in result.message
