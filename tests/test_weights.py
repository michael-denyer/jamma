"""Tests for weight file reading and kinship weight application."""

from pathlib import Path

import numpy as np
import pytest

from jamma.io.weight import apply_individual_weights, read_weight_file

pytestmark = pytest.mark.tier0


class TestReadWeightFile:
    """Tests for read_weight_file."""

    def test_read_weight_file(self, tmp_path: Path) -> None:
        """Read a valid weight file with 3 entries."""
        weight_file = tmp_path / "weights.txt"
        weight_file.write_text("1.0\n2.0\n3.0\n")

        weights = read_weight_file(weight_file)

        assert weights.shape == (3,)
        np.testing.assert_array_equal(weights, [1.0, 2.0, 3.0])

    def test_read_weight_file_empty(self, tmp_path: Path) -> None:
        """Empty weight file raises ValueError."""
        weight_file = tmp_path / "empty.txt"
        weight_file.write_text("")

        with pytest.raises(ValueError, match="Weight file is empty"):
            read_weight_file(weight_file)

    def test_read_weight_file_whitespace_lines(self, tmp_path: Path) -> None:
        """Weight file with leading/trailing whitespace still parses."""
        weight_file = tmp_path / "weights.txt"
        weight_file.write_text("  1.5  \n  2.5  \n")

        weights = read_weight_file(weight_file)

        assert weights.shape == (2,)
        np.testing.assert_array_equal(weights, [1.5, 2.5])

    def test_read_weight_file_multi_column_rejected(self, tmp_path: Path) -> None:
        """Multi-column weight file raises ValueError instead of silently flattening.

        Regression test: previously a 2x2 file would be .ravel()'d to 4 weights,
        causing silent weight misalignment if the flattened length matched n_samples.
        """
        weight_file = tmp_path / "weights.txt"
        weight_file.write_text("1.0 2.0\n3.0 4.0\n")

        with pytest.raises(ValueError, match="2 columns but expected 1"):
            read_weight_file(weight_file)

    def test_read_weight_file_single_column_matrix(self, tmp_path: Path) -> None:
        """Single-column file that numpy reads as (n,1) is accepted and flattened."""
        weight_file = tmp_path / "weights.txt"
        weight_file.write_text("1.0\n2.0\n3.0\n")

        weights = read_weight_file(weight_file)

        assert weights.shape == (3,)
        np.testing.assert_array_equal(weights, [1.0, 2.0, 3.0])

    def test_read_weight_file_nan_rejected(self, tmp_path: Path) -> None:
        """NaN values in weight file raise ValueError.

        Regression test: NaN weights would silently bypass scaling in
        apply_individual_weights (NaN comparisons are always False), leaving
        rows unscaled instead of zeroed or rejected.
        """
        weight_file = tmp_path / "weights.txt"
        weight_file.write_text("1.0\nnan\n3.0\n")

        with pytest.raises(ValueError, match="1 NaN value"):
            read_weight_file(weight_file)

    def test_read_weight_file_unparsable_includes_path(self, tmp_path: Path) -> None:
        """Unparsable weight file includes file path in error message."""
        weight_file = tmp_path / "weights.txt"
        weight_file.write_text("1.0\nnot_a_number\n3.0\n")

        with pytest.raises(ValueError, match=str(weight_file)):
            read_weight_file(weight_file)


class TestApplyIndividualWeights:
    """Tests for apply_individual_weights."""

    def test_apply_individual_weights_basic(self) -> None:
        """Verify K[i,j] /= sqrt(w_i * w_j) for known values."""
        # 3x3 matrix with known off-diagonals
        K = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 6.0, 9.0],
            ]
        )
        weights = np.array([1.0, 4.0, 9.0])

        result = apply_individual_weights(K, weights)

        # K[0,1] /= sqrt(1*4) = /2 -> 2.0/2 = 1.0
        assert result[0, 1] == pytest.approx(1.0)
        # K[0,2] /= sqrt(1*9) = /3 -> 3.0/3 = 1.0
        assert result[0, 2] == pytest.approx(1.0)
        # K[1,2] /= sqrt(4*9) = /6 -> 6.0/6 = 1.0
        assert result[1, 2] == pytest.approx(1.0)
        # Diagonal: K[0,0] /= sqrt(1*1) = 1.0
        assert result[0, 0] == pytest.approx(1.0)
        # K[1,1] /= sqrt(4*4) = /4 -> 4.0/4 = 1.0
        assert result[1, 1] == pytest.approx(1.0)
        # K[2,2] /= sqrt(9*9) = /9 -> 9.0/9 = 1.0
        assert result[2, 2] == pytest.approx(1.0)

        # Verify in-place modification
        assert result is K

    def test_apply_individual_weights_zero_weight(self) -> None:
        """Zero weight zeros out corresponding row and column."""
        K = np.ones((3, 3), dtype=np.float64)
        weights = np.array([1.0, 0.0, 1.0])

        result = apply_individual_weights(K, weights)

        # Row 1 and column 1 should be zeroed
        np.testing.assert_array_equal(result[1, :], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[:, 1], [0.0, 0.0, 0.0])
        # Other entries should be unchanged (weights are 1.0)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 2] == pytest.approx(1.0)
        assert result[2, 0] == pytest.approx(1.0)
        assert result[2, 2] == pytest.approx(1.0)

    def test_apply_individual_weights_negative_weight(self) -> None:
        """Negative weights also zero out entries (GEMMA treats <= 0 the same)."""
        K = np.ones((3, 3), dtype=np.float64)
        weights = np.array([1.0, -2.0, 1.0])

        result = apply_individual_weights(K, weights)

        # Row 1 and column 1 should be zeroed (negative weight)
        np.testing.assert_array_equal(result[1, :], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[:, 1], [0.0, 0.0, 0.0])
        # Other entries unchanged
        assert result[0, 0] == pytest.approx(1.0)
        assert result[2, 2] == pytest.approx(1.0)

    def test_apply_individual_weights_all_ones(self) -> None:
        """Weights of all 1.0 should not change K."""
        K_orig = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]
        )
        K = K_orig.copy()
        weights = np.ones(3)

        apply_individual_weights(K, weights)

        np.testing.assert_allclose(K, K_orig)

    def test_apply_individual_weights_symmetry(self) -> None:
        """Result must remain symmetric when input is symmetric."""
        rng = np.random.default_rng(42)
        n = 10
        A = rng.standard_normal((n, n))
        K = A @ A.T  # Symmetric positive definite
        weights = rng.uniform(0.5, 2.0, size=n)

        apply_individual_weights(K, weights)

        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_apply_individual_weights_shape_mismatch(self) -> None:
        """Mismatched weights and K dimensions raise ValueError."""
        K = np.ones((3, 3))
        weights = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Weight array has 2 entries"):
            apply_individual_weights(K, weights)
