"""Tests for validate_runner_inputs() and LmmConfig validation.

Covers all error branches in validate_runner_inputs() and LmmConfig.__post_init__,
plus the happy path with sample filtering.
"""

import numpy as np
import pytest

from jamma.core.constants import PHENOTYPE_MISSING
from jamma.lmm.prepare_common import (
    EigenPairs,
    KinshipMatrix,
    RunnerSetup,
    parse_eigen_input,
    validate_runner_inputs,
)
from jamma.lmm.schema import MIN_N_REFINE, LmmConfig

pytestmark = pytest.mark.tier0

# ── validate_runner_inputs ─────────────────────────────────────────


class TestValidateRunnerInputsErrors:
    """Error branches in validate_runner_inputs."""

    def test_only_eigenvalues_raises(self):
        """Providing eigenvalues without eigenvectors raises ValueError."""
        K = np.eye(3)
        with pytest.raises(ValueError, match="both eigenvalues and eigenvectors"):
            parse_eigen_input(K, np.ones(3), None)

    def test_only_eigenvectors_raises(self):
        """Providing eigenvectors without eigenvalues raises ValueError."""
        K = np.eye(3)
        with pytest.raises(ValueError, match="both eigenvalues and eigenvectors"):
            parse_eigen_input(K, None, np.eye(3))

    def test_no_kinship_no_eigen_raises(self):
        """Neither kinship nor eigenvalues raises ValueError."""
        with pytest.raises(ValueError, match="Either kinship or pre-computed"):
            parse_eigen_input(None, None, None)

    def test_all_phenotypes_missing_raises(self):
        """All phenotypes NaN raises ValueError."""
        y = np.array([np.nan, np.nan, np.nan])
        K = np.eye(3)
        with pytest.raises(ValueError, match="No valid samples"):
            validate_runner_inputs(y, KinshipMatrix(K), None)

    def test_all_phenotypes_sentinel_raises(self):
        """All phenotypes equal to PHENOTYPE_MISSING raises ValueError."""
        y = np.full(3, PHENOTYPE_MISSING, dtype=np.float64)
        K = np.eye(3)
        with pytest.raises(ValueError, match="No valid samples"):
            validate_runner_inputs(y, KinshipMatrix(K), None)

    def test_eigenpair_dimension_mismatch_eigenvalues(self):
        """Eigenvalue length mismatch after filtering raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        # Eigenvalues are length 4 but 3 samples → mismatch
        evals = np.ones(4)
        evecs = np.eye(4)
        with pytest.raises(ValueError, match="eigenvalues length"):
            validate_runner_inputs(y, EigenPairs(evals, evecs), None)

    def test_eigenpair_dimension_mismatch_eigenvectors(self):
        """Eigenvector shape mismatch raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        evals = np.ones(3)
        evecs = np.eye(4)  # Wrong shape
        with pytest.raises(ValueError, match="eigenvectors shape"):
            validate_runner_inputs(y, EigenPairs(evals, evecs), None)


class TestValidateRunnerInputsHappyPath:
    """Happy-path behaviour of validate_runner_inputs."""

    def test_returns_runner_setup(self):
        """Valid inputs return a RunnerSetup."""
        y = np.array([1.0, 2.0, 3.0])
        K = np.eye(3)
        result = validate_runner_inputs(y, KinshipMatrix(K), None)
        assert isinstance(result, RunnerSetup)
        assert result.n_samples == 3

    def test_no_copy_when_all_valid(self):
        """When all samples are valid, arrays are not copied."""
        y = np.array([1.0, 2.0, 3.0])
        K = np.eye(3)
        result = validate_runner_inputs(y, KinshipMatrix(K), None)
        # Same object — no copy made
        assert result.phenotypes is y
        assert isinstance(result.eigen_input, KinshipMatrix)
        assert result.eigen_input.value is K

    def test_filters_nan_phenotypes(self):
        """NaN phenotypes are filtered out."""
        y = np.array([1.0, np.nan, 3.0])
        K = np.eye(3)
        result = validate_runner_inputs(y, KinshipMatrix(K), None)
        assert result.n_samples == 2
        np.testing.assert_array_equal(result.phenotypes, [1.0, 3.0])

    def test_filters_sentinel_phenotypes(self):
        """PHENOTYPE_MISSING (-9) phenotypes are filtered out."""
        y = np.array([1.0, PHENOTYPE_MISSING, 3.0], dtype=np.float64)
        K = np.eye(3)
        result = validate_runner_inputs(y, KinshipMatrix(K), None)
        assert result.n_samples == 2

    def test_filters_nan_covariates(self):
        """Samples with NaN covariates are excluded."""
        y = np.array([1.0, 2.0, 3.0])
        K = np.eye(3)
        cov = np.array([[1.0], [np.nan], [1.0]])
        result = validate_runner_inputs(y, KinshipMatrix(K), cov)
        assert result.n_samples == 2
        np.testing.assert_array_equal(result.phenotypes, [1.0, 3.0])

    def test_kinship_filtered_symmetrically(self):
        """Kinship is subsetted via np.ix_ when samples are removed."""
        y = np.array([1.0, np.nan, 3.0])
        K = np.arange(9, dtype=np.float64).reshape(3, 3)
        result = validate_runner_inputs(y, KinshipMatrix(K), None)
        expected = K[np.ix_([True, False, True], [True, False, True])]
        assert isinstance(result.eigen_input, KinshipMatrix)
        np.testing.assert_array_equal(result.eigen_input.value, expected)

    def test_valid_mask_shape_matches_original(self):
        """valid_mask has the original (pre-filter) length."""
        y = np.array([1.0, np.nan, 3.0, 4.0])
        K = np.eye(4)
        result = validate_runner_inputs(y, KinshipMatrix(K), None)
        assert result.valid_mask.shape == (4,)
        np.testing.assert_array_equal(result.valid_mask, [True, False, True, True])

    def test_precomputed_eigenpairs_accepted(self):
        """Pre-computed eigenvalues + eigenvectors are passed through."""
        y = np.array([1.0, 2.0, 3.0])
        evals = np.ones(3)
        evecs = np.eye(3)
        result = validate_runner_inputs(y, EigenPairs(evals, evecs), None)
        assert isinstance(result.eigen_input, EigenPairs)
        assert result.eigen_input.values is evals
        assert result.eigen_input.vectors is evecs

    def test_complete_eigenpairs_take_precedence_over_kinship(self):
        kinship = np.eye(3)
        evals = np.ones(3)
        evecs = np.eye(3)

        result = parse_eigen_input(kinship, evals, evecs)

        assert isinstance(result, EigenPairs)
        assert result.values is evals
        assert result.vectors is evecs


# ── LmmConfig ──────────────────────────────────────────────────────


class TestLmmConfigValidation:
    """LmmConfig __post_init__ validation."""

    def test_defaults_are_valid(self):
        """Default LmmConfig() succeeds."""
        config = LmmConfig()
        assert config.lmm_mode == 1
        assert config.maf_threshold == 0.01
        assert config.miss_threshold == 0.05

    @pytest.mark.parametrize("mode", [0, -1, 5, 99])
    def test_invalid_lmm_mode(self, mode):
        with pytest.raises(ValueError, match="lmm_mode must be"):
            LmmConfig(lmm_mode=mode)

    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_valid_lmm_modes(self, mode):
        assert LmmConfig(lmm_mode=mode).lmm_mode == mode

    def test_maf_negative_raises(self):
        with pytest.raises(ValueError, match="maf_threshold"):
            LmmConfig(maf_threshold=-0.1)

    def test_maf_above_half_raises(self):
        with pytest.raises(ValueError, match="maf_threshold"):
            LmmConfig(maf_threshold=0.6)

    def test_maf_boundaries_accepted(self):
        assert LmmConfig(maf_threshold=0.0).maf_threshold == 0.0
        assert LmmConfig(maf_threshold=0.5).maf_threshold == 0.5

    def test_miss_negative_raises(self):
        with pytest.raises(ValueError, match="miss_threshold"):
            LmmConfig(miss_threshold=-0.1)

    def test_miss_above_one_raises(self):
        with pytest.raises(ValueError, match="miss_threshold"):
            LmmConfig(miss_threshold=1.1)

    def test_miss_boundaries_accepted(self):
        assert LmmConfig(miss_threshold=0.0).miss_threshold == 0.0
        assert LmmConfig(miss_threshold=1.0).miss_threshold == 1.0

    def test_l_min_zero_raises(self):
        with pytest.raises(ValueError, match="l_min must be positive"):
            LmmConfig(l_min=0.0)

    def test_l_min_negative_raises(self):
        with pytest.raises(ValueError, match="l_min must be positive"):
            LmmConfig(l_min=-1.0)

    def test_l_max_less_than_l_min_raises(self):
        with pytest.raises(ValueError, match=r"l_max.*must be greater than l_min"):
            LmmConfig(l_min=100.0, l_max=1.0)

    def test_l_max_equal_to_l_min_raises(self):
        with pytest.raises(ValueError, match=r"l_max.*must be greater than l_min"):
            LmmConfig(l_min=1.0, l_max=1.0)

    @pytest.mark.parametrize("n_grid", [-5, 0, 1])
    def test_n_grid_below_two_raises(self, n_grid):
        with pytest.raises(ValueError, match="n_grid must be >= 2"):
            LmmConfig(n_grid=n_grid)

    @pytest.mark.parametrize("n_refine", [0, 1, 10, 19])
    def test_n_refine_below_minimum_is_raised_not_rejected(self, n_refine):
        """A low n_refine is raised to MIN_N_REFINE here, the one place that does."""
        assert LmmConfig(n_refine=n_refine).n_refine == MIN_N_REFINE

    @pytest.mark.parametrize("n_refine", [20, 25])
    def test_n_refine_at_or_above_minimum_is_kept(self, n_refine):
        assert LmmConfig(n_refine=n_refine).n_refine == n_refine

    def test_valid_l_range(self):
        config = LmmConfig(l_min=1e-5, l_max=1e5)
        assert config.l_min == 1e-5
        assert config.l_max == 1e5
