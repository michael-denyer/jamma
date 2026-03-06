"""Unit tests for JAX likelihood functions and edge cases.

Tests cover: negative P_yy handling, degenerate SNPs (Wald/LRT),
batch assembly correctness, and near-zero eigenvalue stability.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp
from jax import vmap

from jamma.lmm.likelihood import (
    calc_pab,
    compute_Uab,
    reml_log_likelihood,
)
from jamma.lmm.likelihood_jax import (
    _batch_grid_reml_ncvt1,
    batch_calc_score_stats,
    batch_calc_wald_stats,
    batch_compute_iab,
    batch_compute_uab,
    build_index_table,
    classify_uab_columns,
    golden_section_optimize_lambda,
    mle_log_likelihood_jax,
    reml_log_likelihood_jax,
)
from jamma.lmm.stats import calc_wald_test

pytestmark = pytest.mark.requires_jax


def _make_test_data(n_samples=50, n_cvt=1, rng_seed=42):
    """Create synthetic eigenvalues, UtW, Uty, Utx for testing."""
    rng = np.random.default_rng(rng_seed)

    # Eigenvalues from a PSD kinship matrix (all positive)
    eigenvalues = np.sort(rng.exponential(1.0, size=n_samples))[::-1]

    # Covariates (intercept only for n_cvt=1)
    W = np.ones((n_samples, n_cvt))
    # Random phenotype
    y = rng.standard_normal(n_samples)
    # Random genotype
    x = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=[0.25, 0.5, 0.25])

    # Simulate U.T @ vectors (as if we've eigendecomposed)
    U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]
    UtW = U.T @ W
    Uty = U.T @ y
    Utx = U.T @ x

    return eigenvalues, UtW, Uty, Utx


@pytest.mark.tier0
class TestNegativePyyHandling:
    """Tests for negative P_yy clamping in JAX REML and MLE paths."""

    def test_reml_negative_pyy_returns_nan(self):
        """REML with contrived Uab that produces negative P_yy returns NaN."""
        n_cvt = 1
        eigenvalues = jnp.ones(10, dtype=jnp.float64)
        table = build_index_table(n_cvt)
        n_index = table["n_index"]

        # Create Uab where the P_yy projection will go negative
        Uab = jnp.zeros((10, n_index), dtype=jnp.float64)
        # Set values such that recursive Pab projection produces negative P_yy
        # by making the W-W diagonal tiny and W-Y cross term large
        Uab = Uab.at[:, 0].set(1e-15)  # WW: near zero
        Uab = Uab.at[:, 2].set(100.0)  # WY: large
        Uab = Uab.at[:, 5].set(-1.0)  # YY: negative → forces negative P_yy

        lambda_val = jnp.float64(1.0)
        result = reml_log_likelihood_jax(n_cvt, lambda_val, eigenvalues, Uab)
        assert jnp.isnan(result), f"Expected NaN for negative P_yy, got {result}"

    def test_mle_negative_pyy_returns_nan(self):
        """MLE with contrived Uab that produces negative P_yy returns NaN."""
        n_cvt = 1
        eigenvalues = jnp.ones(10, dtype=jnp.float64)
        table = build_index_table(n_cvt)
        n_index = table["n_index"]

        Uab = jnp.zeros((10, n_index), dtype=jnp.float64)
        Uab = Uab.at[:, 0].set(1e-15)
        Uab = Uab.at[:, 2].set(100.0)
        Uab = Uab.at[:, 5].set(-1.0)

        lambda_val = jnp.float64(1.0)
        result = mle_log_likelihood_jax(n_cvt, lambda_val, eigenvalues, Uab)
        assert jnp.isnan(result), f"Expected NaN for negative P_yy, got {result}"

    def test_reml_normal_data_produces_finite(self):
        """REML on well-conditioned data produces finite likelihood."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        eigenvalues_jax = jnp.array(eigenvalues)
        Uab_jax = jnp.array(Uab)
        lambda_val = jnp.float64(1.0)

        result = reml_log_likelihood_jax(1, lambda_val, eigenvalues_jax, Uab_jax)
        assert jnp.isfinite(result), f"Expected finite result, got {result}"


@pytest.mark.tier0
class TestJaxNumpyConsistency:
    """JAX and NumPy likelihood paths should agree on normal data."""

    def test_reml_jax_matches_numpy(self):
        """JAX REML matches NumPy REML on well-conditioned data."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        lambda_val = 1.0
        numpy_result = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt=1)

        jax_result = float(
            reml_log_likelihood_jax(
                1, jnp.float64(lambda_val), jnp.array(eigenvalues), jnp.array(Uab)
            )
        )

        np.testing.assert_allclose(jax_result, numpy_result, rtol=1e-10)

    @pytest.mark.parametrize("lambda_val", [1e-5, 0.01, 1.0, 100.0, 1e5])
    def test_reml_jax_matches_numpy_across_lambda(self, lambda_val):
        """JAX and NumPy REML agree across lambda range including boundaries."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        numpy_result = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt=1)
        jax_result = float(
            reml_log_likelihood_jax(
                1, jnp.float64(lambda_val), jnp.array(eigenvalues), jnp.array(Uab)
            )
        )

        np.testing.assert_allclose(jax_result, numpy_result, rtol=1e-10)


@pytest.mark.tier0
class TestWaldDegenerateSNP:
    """Wald test should return NaN for degenerate (constant) genotypes."""

    def test_wald_constant_genotype_returns_nan(self):
        """Constant genotype (no variance) produces NaN beta/se/p."""
        eigenvalues, UtW, Uty, _ = _make_test_data()

        # Constant genotype → zero Utx after rotation
        Utx_const = np.zeros(len(eigenvalues))
        Uab = compute_Uab(UtW, Uty, Utx_const)

        lambda_val = 1.0
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(1, Hi_eval, Uab)

        beta, se, p = calc_wald_test(Pab, n_cvt=1, ni_test=len(eigenvalues))
        assert np.isnan(beta), f"Expected NaN beta for constant genotype, got {beta}"
        assert np.isnan(se), f"Expected NaN se, got {se}"
        assert np.isnan(p), f"Expected NaN p, got {p}"

    def test_wald_nearly_constant_genotype(self):
        """Near-constant genotype (tiny variance) produces finite but large SE."""
        eigenvalues, UtW, Uty, _ = _make_test_data()
        n = len(eigenvalues)

        # One non-zero value in an otherwise constant genotype
        Utx_near_const = np.zeros(n)
        Utx_near_const[0] = 1e-6
        Uab = compute_Uab(UtW, Uty, Utx_near_const)

        lambda_val = 1.0
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(1, Hi_eval, Uab)

        beta, se, p = calc_wald_test(Pab, n_cvt=1, ni_test=n)
        # Should produce finite results (genotype has some variance)
        assert np.isfinite(beta), f"Expected finite beta, got {beta}"


@pytest.mark.tier0
class TestNearZeroEigenvalues:
    """REML should handle near-machine-epsilon eigenvalues correctly."""

    def test_reml_with_zero_eigenvalues(self):
        """Eigenvalues exactly 0 should produce finite REML likelihood."""
        eigenvalues, UtW, Uty, Utx = _make_test_data(n_samples=20)
        # Zero out half the eigenvalues (simulating rank-deficient kinship)
        eigenvalues[:10] = 0.0

        Uab = compute_Uab(UtW, Uty, Utx)

        result = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt=1)
        assert np.isfinite(result), (
            f"Expected finite REML with zero eigenvalues, got {result}"
        )

    def test_reml_jax_with_zero_eigenvalues(self):
        """JAX REML handles zero eigenvalues same as NumPy."""
        eigenvalues, UtW, Uty, Utx = _make_test_data(n_samples=20)
        eigenvalues[:10] = 0.0
        Uab = compute_Uab(UtW, Uty, Utx)

        numpy_result = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt=1)
        jax_result = float(
            reml_log_likelihood_jax(
                1, jnp.float64(1.0), jnp.array(eigenvalues), jnp.array(Uab)
            )
        )
        np.testing.assert_allclose(jax_result, numpy_result, rtol=1e-10)

    def test_reml_with_tiny_eigenvalues(self):
        """Eigenvalues near machine epsilon don't produce inf/nan."""
        eigenvalues, UtW, Uty, Utx = _make_test_data(n_samples=20)
        eigenvalues[:5] = 1e-15  # Near machine epsilon

        Uab = compute_Uab(UtW, Uty, Utx)

        result = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt=1)
        assert np.isfinite(result), f"Expected finite REML, got {result}"


@pytest.mark.tier0
class TestLambdaBounds:
    """REML should behave sensibly at lambda optimization boundaries."""

    def test_reml_at_lower_bound(self):
        """Lambda = 1e-5 (lower bound) produces finite likelihood."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        result = reml_log_likelihood(1e-5, eigenvalues, Uab, n_cvt=1)
        assert np.isfinite(result), f"Expected finite at lower bound, got {result}"

    def test_reml_at_upper_bound(self):
        """Lambda = 1e5 (upper bound) produces finite likelihood."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        result = reml_log_likelihood(1e5, eigenvalues, Uab, n_cvt=1)
        assert np.isfinite(result), f"Expected finite at upper bound, got {result}"

    def test_reml_monotonic_near_optimum(self):
        """REML should be smooth (no discontinuities) around typical lambda values."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        lambdas = np.logspace(-2, 2, 50)
        results = [
            reml_log_likelihood(lam, eigenvalues, Uab, n_cvt=1) for lam in lambdas
        ]

        assert all(np.isfinite(r) for r in results), (
            "All lambda values should give finite REML"
        )
        # Check no huge jumps (>100x) between adjacent values
        for i in range(1, len(results)):
            ratio = abs(results[i] / results[i - 1]) if results[i - 1] != 0 else 1.0
            assert ratio < 100, (
                f"Discontinuity at lambda={lambdas[i]}: {results[i - 1]} → {results[i]}"
            )


@pytest.mark.tier0
class TestCovariateRankValidation:
    """Test that rank-deficient covariates are caught."""

    def test_linearly_dependent_covariates_rejected(self):
        """Covariates with duplicate columns should raise ValueError."""
        from jamma.lmm.prepare import _build_covariate_matrix

        # Column 2 = 2 * Column 1 (intercept) → rank 1 instead of 2
        covariates = np.column_stack([np.ones(20), 2 * np.ones(20)])

        with pytest.raises(ValueError, match="rank-deficient"):
            _build_covariate_matrix(covariates, n_samples=20)

    def test_full_rank_covariates_accepted(self):
        """Covariates with full rank pass validation."""
        from jamma.lmm.prepare import _build_covariate_matrix

        rng = np.random.default_rng(42)
        covariates = np.column_stack([np.ones(20), rng.standard_normal(20)])

        W, n_cvt = _build_covariate_matrix(covariates, n_samples=20)
        assert n_cvt == 2
        assert W.shape == (20, 2)


@pytest.mark.tier0
class TestKinshipSymmetryCheck:
    """Test that non-symmetric kinship matrices produce warnings."""

    def test_asymmetric_kinship_warns(self):
        """Non-symmetric kinship produces loguru warning during eigendecomposition."""
        from io import StringIO

        from loguru import logger

        from jamma.lmm.eigen import eigendecompose_kinship

        K = np.eye(10)
        K[0, 1] = 0.5  # Asymmetric
        K[1, 0] = 0.3

        buf = StringIO()
        sink_id = logger.add(buf, level="WARNING", format="{message}")
        try:
            eigendecompose_kinship(K, check_memory=False)
        finally:
            logger.remove(sink_id)

        assert "not symmetric" in buf.getvalue(), (
            f"Expected 'not symmetric' warning, got: {buf.getvalue()[:200]}"
        )

    def test_symmetric_kinship_no_warning(self):
        """Symmetric kinship produces no symmetry warning."""
        from io import StringIO

        from loguru import logger

        from jamma.lmm.eigen import eigendecompose_kinship

        K = np.eye(10)

        buf = StringIO()
        sink_id = logger.add(buf, level="WARNING", format="{message}")
        try:
            eigendecompose_kinship(K, check_memory=False)
        finally:
            logger.remove(sink_id)

        assert "not symmetric" not in buf.getvalue()


@pytest.mark.tier0
class TestNcvt1FastPathParity:
    """n_cvt=1 fast path must match general path exactly."""

    def _make_batch_data(self, n_samples=50, n_snps=20, rng_seed=42):
        """Create Uab_batch and Iab_batch for n_cvt=1."""
        rng = np.random.default_rng(rng_seed)
        eigenvalues = np.sort(rng.exponential(1.0, size=n_samples))[::-1]

        W = np.ones((n_samples, 1))
        y = rng.standard_normal(n_samples)
        U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]
        UtW = jnp.array(U.T @ W)
        Uty = jnp.array(U.T @ y)

        G = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.25, 0.5, 0.25])
        UtG = jnp.array(U.T @ G)

        evals = jnp.array(eigenvalues)
        Uab = batch_compute_uab(1, UtW, Uty, UtG)
        Iab = batch_compute_iab(1, Uab)

        return evals, Uab, Iab

    def test_grid_reml_ncvt1_matches_general(self):
        """_batch_grid_reml_ncvt1 matches scalar _reml_with_precomputed_iab."""
        from jax import vmap

        from jamma.lmm.likelihood_jax import _reml_with_precomputed_iab

        evals, Uab, Iab = self._make_batch_data()
        lambdas = jnp.logspace(-5, 5, 50)

        # Fast path (n_cvt=1 specialized)
        fast = _batch_grid_reml_ncvt1(lambdas, evals, Uab, Iab)

        # General path: manually vmap the scalar REML evaluator
        # vmap over lambdas (outer), then over SNPs (inner)
        def reml_for_lambda(lam):
            return vmap(lambda u, i: _reml_with_precomputed_iab(1, lam, evals, u, i))(
                Uab, Iab
            )

        general = vmap(reml_for_lambda)(lambdas)

        np.testing.assert_allclose(
            np.array(fast),
            np.array(general),
            rtol=1e-10,
            err_msg="n_cvt=1 fast path diverges from general REML evaluator",
        )

    def test_golden_section_ncvt1_produces_valid_results(self):
        """golden_section_optimize_lambda n_cvt=1 produces valid lambdas and logls."""
        evals, Uab, Iab = self._make_batch_data()

        # Run golden section with n_cvt=1 fast path
        lambdas, logls = golden_section_optimize_lambda(
            1,
            evals,
            Uab,
            Iab,
            n_grid=50,
            n_iter=20,
        )

        lambdas_np = np.array(lambdas)
        logls_np = np.array(logls)

        # Lambdas should be in valid range
        assert np.all(lambdas_np >= 1e-5), "Lambdas below l_min"
        assert np.all(lambdas_np <= 1e5), "Lambdas above l_max"

        # Log-likelihoods should be finite (no NaN from fast path)
        assert np.all(np.isfinite(logls_np)), "Non-finite log-likelihoods"

        # Optimized logls should be close to the best coarse grid point.
        # Golden section refines within a bracket, so it can be marginally
        # worse than the grid's best point when the optimum is very flat.
        grid_logls = _batch_grid_reml_ncvt1(jnp.logspace(-5, 5, 50), evals, Uab, Iab)
        best_grid = np.array(jnp.max(grid_logls, axis=0))
        np.testing.assert_allclose(
            logls_np,
            best_grid,
            rtol=1e-2,
            err_msg="Golden section diverges from grid optimum",
        )


@pytest.mark.tier0
class TestJaxBatchDegenerateSNP:
    """JAX batch likelihood path produces NaN stats for degenerate (zero) SNPs."""

    def _make_batch_data_with_degen(self, n_samples=50, rng_seed=42):
        """Create Uab_batch with one valid and one degenerate (all-zero) SNP column.

        Returns:
            (eigenvalues_jax, Uab_batch, Iab_batch, n_samples) as JAX arrays.
        """
        rng = np.random.default_rng(rng_seed)
        eigenvalues = np.sort(rng.exponential(1.0, size=n_samples))[::-1]

        W = np.ones((n_samples, 1))
        y = rng.standard_normal(n_samples)
        U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]

        UtW = jnp.array(U.T @ W)
        Uty = jnp.array(U.T @ y)

        # Two SNPs: column 0 is valid, column 1 is degenerate (constant genotype)
        x_valid = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=[0.25, 0.5, 0.25])
        x_degen = np.zeros(n_samples)
        G = np.column_stack([x_valid, x_degen])
        UtG = jnp.array(U.T @ G)

        evals = jnp.array(eigenvalues)
        Uab = batch_compute_uab(1, UtW, Uty, UtG)
        Iab = batch_compute_iab(1, Uab)

        return evals, Uab, Iab, n_samples

    def test_jax_batch_degenerate_snp_returns_nan_pvalues(self):
        """JAX batch path: degenerate SNP produces NaN beta/se/p_wald.

        An all-zero genotype column (P_XX=0) should propagate NaN through the
        Wald test stats, producing NaN beta, se, and p_wald for that SNP.
        The valid SNP in the same batch should remain unaffected.
        """
        evals, Uab_batch, Iab_batch, n_samples = self._make_batch_data_with_degen()

        # Run batch optimizer
        lambdas, logls = golden_section_optimize_lambda(
            1, evals, Uab_batch, Iab_batch, n_grid=50, n_iter=20
        )

        # Run batch Wald stats
        betas, ses, pwalds = batch_calc_wald_stats(
            1, lambdas, evals, Uab_batch, n_samples
        )

        betas_np = np.array(betas)
        ses_np = np.array(ses)
        pwalds_np = np.array(pwalds)

        # Valid SNP (index 0): should have finite stats
        assert np.isfinite(betas_np[0]), (
            f"Valid SNP beta should be finite, got {betas_np[0]}"
        )
        assert np.isfinite(ses_np[0]), f"Valid SNP se should be finite, got {ses_np[0]}"
        assert np.isfinite(pwalds_np[0]), (
            f"Valid SNP p_wald should be finite, got {pwalds_np[0]}"
        )

        # Degenerate SNP (index 1): zero genotype → NaN Wald stats
        assert np.isnan(betas_np[1]), (
            f"Degenerate SNP beta should be NaN, got {betas_np[1]}"
        )
        assert np.isnan(ses_np[1]), f"Degenerate SNP se should be NaN, got {ses_np[1]}"
        assert np.isnan(pwalds_np[1]), (
            f"Degenerate SNP p_wald should be NaN, got {pwalds_np[1]}"
        )

    def test_jax_batch_all_degenerate_snps_wald_all_nan(self):
        """JAX batch path: all-degenerate SNPs (UtG=0) produce all-NaN Wald stats.

        When every SNP is constant (UtG=0), P_XX=0 for every SNP.  The
        optimizer may find any lambda (REML depends only on phenotype
        variance when genotype is zero), but the critical downstream behavior
        is that all Wald stats must be NaN.  Lambdas must be finite and
        within [l_min, l_max].
        """
        n_samples = 50
        n_snps = 5
        l_min = 1e-5
        l_max = 1e5
        rng = np.random.default_rng(7)

        eigenvalues = np.sort(rng.exponential(1.0, size=n_samples))[::-1]
        U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]
        W = np.ones((n_samples, 1))
        y = rng.standard_normal(n_samples)

        UtW = jnp.array(U.T @ W)
        Uty = jnp.array(U.T @ y)
        UtG_degen = jnp.zeros((n_samples, n_snps))  # constant genotype

        evals = jnp.array(eigenvalues)
        Uab = batch_compute_uab(1, UtW, Uty, UtG_degen)
        Iab = batch_compute_iab(1, Uab)

        lambdas, logls = golden_section_optimize_lambda(
            1, evals, Uab, Iab, l_min=l_min, l_max=l_max, n_grid=50, n_iter=20
        )

        lambdas_np = np.array(lambdas)
        assert lambdas_np.shape == (n_snps,), (
            f"Expected ({n_snps},), got {lambdas_np.shape}"
        )
        # Lambdas must be finite and within [l_min, l_max]
        assert np.all(np.isfinite(lambdas_np)), (
            f"All lambdas should be finite for degenerate batch, got {lambdas_np}"
        )
        assert np.all(lambdas_np >= l_min * 0.99), f"Lambdas below l_min: {lambdas_np}"
        assert np.all(lambdas_np <= l_max * 1.01), f"Lambdas above l_max: {lambdas_np}"

        # Wald stats: P_XX=0 → all NaN (critical downstream behavior)
        betas, ses, pwalds = batch_calc_wald_stats(1, lambdas, evals, Uab, n_samples)
        assert np.all(np.isnan(np.array(betas))), (
            f"All betas should be NaN for degenerate batch, got {np.array(betas)}"
        )
        assert np.all(np.isnan(np.array(ses))), (
            f"All ses should be NaN for degenerate batch, got {np.array(ses)}"
        )
        assert np.all(np.isnan(np.array(pwalds))), (
            f"All p_walds should be NaN for degenerate batch, got {np.array(pwalds)}"
        )


# ---------------------------------------------------------------------------
# Degenerate SNP tests — JAX Score stats path
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.requires_jax
class TestJaxScoreStatsDegenerateSNP:
    """JAX batch Score stats path produces NaN for degenerate SNPs."""

    def test_batch_score_stats_degenerate_snp_returns_nan(self):
        """batch_calc_score_stats: degenerate SNP (P_XX=0) → NaN beta/se/p_score.

        Score test uses a fixed null-model Hi_eval (lambda_null).  When a SNP
        is constant (UtG=0), P_xx=0 after projection.  The guard `is_valid =
        P_xx > 0` must propagate NaN through beta, se, and p_score for that
        SNP without affecting the valid SNP in the same batch.
        """

        n_samples = 50
        rng = np.random.default_rng(13)

        eigenvalues = np.sort(rng.exponential(1.0, size=n_samples))[::-1]
        U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]
        W = np.ones((n_samples, 1))
        y = rng.standard_normal(n_samples)

        UtW = jnp.array(U.T @ W)
        Uty = jnp.array(U.T @ y)

        x_valid = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=[0.25, 0.5, 0.25])
        x_degen = np.zeros(n_samples)
        G = np.column_stack([x_valid, x_degen])
        UtG = jnp.array(U.T @ G)

        evals = jnp.array(eigenvalues)
        Uab = batch_compute_uab(1, UtW, Uty, UtG)

        # Null-model lambda: use l_min as a simple fixed null
        lambda_null = 1e-5
        Hi_eval_null = 1.0 / (lambda_null * evals + 1.0)

        betas, ses, pscores = batch_calc_score_stats(1, Hi_eval_null, Uab, n_samples)

        betas_np = np.array(betas)
        ses_np = np.array(ses)
        pscores_np = np.array(pscores)

        # Valid SNP (index 0): finite stats
        assert np.isfinite(betas_np[0]), (
            f"Valid SNP beta should be finite, got {betas_np[0]}"
        )
        assert np.isfinite(ses_np[0]), f"Valid SNP se should be finite, got {ses_np[0]}"
        assert np.isfinite(pscores_np[0]), (
            f"Valid SNP p_score should be finite, got {pscores_np[0]}"
        )

        # Degenerate SNP (index 1): P_XX=0 → NaN
        assert np.isnan(betas_np[1]), (
            f"Degenerate SNP beta should be NaN, got {betas_np[1]}"
        )
        assert np.isnan(ses_np[1]), f"Degenerate SNP se should be NaN, got {ses_np[1]}"
        assert np.isnan(pscores_np[1]), (
            f"Degenerate SNP p_score should be NaN, got {pscores_np[1]}"
        )

    def test_batch_score_stats_all_degenerate_all_nan(self):
        """batch_calc_score_stats: all-degenerate batch → all NaN.

        Feeding only constant-genotype SNPs (UtG=0) should produce all-NaN
        beta/se/p_score arrays.  This tests the P_yy clamping path and the
        P_XX guard with no valid SNPs present.
        """

        n_samples = 50
        n_snps = 5
        rng = np.random.default_rng(31)

        eigenvalues = np.sort(rng.exponential(1.0, size=n_samples))[::-1]
        U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]
        W = np.ones((n_samples, 1))
        y = rng.standard_normal(n_samples)

        UtW = jnp.array(U.T @ W)
        Uty = jnp.array(U.T @ y)
        UtG_degen = jnp.zeros((n_samples, n_snps))

        evals = jnp.array(eigenvalues)
        Uab = batch_compute_uab(1, UtW, Uty, UtG_degen)

        lambda_null = 1e-5
        Hi_eval_null = 1.0 / (lambda_null * evals + 1.0)

        betas, ses, pscores = batch_calc_score_stats(1, Hi_eval_null, Uab, n_samples)

        assert np.all(np.isnan(np.array(betas))), (
            f"All betas should be NaN for all-degenerate batch, got {np.array(betas)}"
        )
        assert np.all(np.isnan(np.array(ses))), (
            f"All ses should be NaN for all-degenerate batch, got {np.array(ses)}"
        )
        assert np.all(np.isnan(np.array(pscores))), (
            f"All p_scores should be NaN for all-degenerate batch, "
            f"got {np.array(pscores)}"
        )


# ---------------------------------------------------------------------------
# classify_uab_columns + general split tests
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestClassifyUabColumns:
    """classify_uab_columns correctly identifies invariant vs varying columns."""

    def test_classify_uab_columns_ncvt1(self):
        """n_cvt=1: invariant=[0,2,5] (ww,wy,yy), varying=[1,3,4] (wx,xx,xy)."""
        invariant, varying = classify_uab_columns(1)
        assert invariant == (0, 2, 5), f"Expected (0,2,5), got {invariant}"
        assert varying == (1, 3, 4), f"Expected (1,3,4), got {varying}"

    def test_classify_uab_columns_ncvt4(self):
        """n_cvt=4: 15 invariant + 6 varying, all varying involve genotype index 4."""
        invariant, varying = classify_uab_columns(4)
        assert len(invariant) == 15, f"Expected 15 invariant, got {len(invariant)}"
        assert len(varying) == 6, f"Expected 6 varying, got {len(varying)}"

        # All varying columns must involve genotype (0-based index = n_cvt = 4)
        table = build_index_table(4)
        genotype_idx = 4  # 0-based index of X in vectors array
        for idx in varying:
            # Find the (a_col, b_col) pair for this linear index
            pair = next(
                (a, b) for a, b, lin_idx in table["uab_pairs"] if lin_idx == idx
            )
            assert pair[0] == genotype_idx or pair[1] == genotype_idx, (
                f"Varying column {idx} with pair {pair} doesn't involve genotype"
            )

        # Invariant columns must NOT involve genotype
        for idx in invariant:
            pair = next(
                (a, b) for a, b, lin_idx in table["uab_pairs"] if lin_idx == idx
            )
            assert pair[0] != genotype_idx and pair[1] != genotype_idx, (
                f"Invariant column {idx} with pair {pair} involves genotype"
            )


@pytest.mark.tier0
class TestGeneralSplitParity:
    """General n_cvt split path must match the existing general vmap path."""

    def _make_multi_cvt_data(self, n_cvt=2, n_samples=50, n_snps=10, rng_seed=42):
        """Create test data for arbitrary n_cvt."""
        rng = np.random.default_rng(rng_seed)
        eigenvalues = np.sort(rng.exponential(1.0, size=n_samples))[::-1]

        W = np.column_stack(
            [np.ones(n_samples)]
            + [rng.standard_normal(n_samples) for _ in range(n_cvt - 1)]
        )
        y = rng.standard_normal(n_samples)
        U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]

        UtW = jnp.array(U.T @ W)
        Uty = jnp.array(U.T @ y)

        G = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.25, 0.5, 0.25])
        UtG = jnp.array(U.T @ G)

        evals = jnp.array(eigenvalues)
        Uab = batch_compute_uab(n_cvt, UtW, Uty, UtG)
        Iab = batch_compute_iab(n_cvt, Uab)

        return evals, Uab, Iab

    def test_grid_reml_general_split_matches_general(self):
        """_batch_grid_reml_general matches the old general vmap path for n_cvt=2."""
        from jamma.lmm.likelihood_jax import (
            _batch_grid_reml_general,
            _reml_with_precomputed_iab,
        )

        evals, Uab, Iab = self._make_multi_cvt_data(n_cvt=2)
        lambdas = jnp.logspace(-5, 5, 50)

        # New split path
        split_result = _batch_grid_reml_general(2, lambdas, evals, Uab, Iab)

        # Old general path: vmap over lambdas, then over SNPs
        def reml_for_lambda(lam):
            return vmap(lambda u, i: _reml_with_precomputed_iab(2, lam, evals, u, i))(
                Uab, Iab
            )

        general_result = vmap(reml_for_lambda)(lambdas)

        np.testing.assert_allclose(
            np.array(split_result),
            np.array(general_result),
            rtol=1e-10,
            err_msg="General split path diverges from general vmap path (n_cvt=2)",
        )

    @pytest.mark.parametrize("n_cvt", [2, 3, 4])
    def test_grid_reml_general_split_multiple_ncvt(self, n_cvt):
        """_batch_grid_reml_general matches for n_cvt=2,3,4."""
        from jamma.lmm.likelihood_jax import (
            _batch_grid_reml_general,
            _reml_with_precomputed_iab,
        )

        evals, Uab, Iab = self._make_multi_cvt_data(n_cvt=n_cvt)
        lambdas = jnp.logspace(-5, 5, 30)

        split_result = _batch_grid_reml_general(n_cvt, lambdas, evals, Uab, Iab)

        def reml_for_lambda(lam):
            return vmap(
                lambda u, i: _reml_with_precomputed_iab(n_cvt, lam, evals, u, i)
            )(Uab, Iab)

        general_result = vmap(reml_for_lambda)(lambdas)

        np.testing.assert_allclose(
            np.array(split_result),
            np.array(general_result),
            rtol=1e-10,
            err_msg=f"General split diverges from general vmap (n_cvt={n_cvt})",
        )

    def test_golden_section_general_split_matches_general(self):
        """Golden section general path with split matches existing path for n_cvt=2."""
        evals, Uab, Iab = self._make_multi_cvt_data(n_cvt=2)

        # The golden_section_optimize_lambda now uses the split path internally
        # for n_cvt>1. We compare against the old general path result by
        # computing the old way manually.
        from jamma.lmm.likelihood_jax import _reml_with_precomputed_iab

        # New path (uses split internally)
        lambdas_new, logls_new = golden_section_optimize_lambda(
            2, evals, Uab, Iab, n_grid=50, n_iter=20
        )

        # Verify results are valid
        lambdas_np = np.array(lambdas_new)
        logls_np = np.array(logls_new)
        assert np.all(np.isfinite(lambdas_np)), f"Non-finite lambdas: {lambdas_np}"
        assert np.all(np.isfinite(logls_np)), f"Non-finite logls: {logls_np}"
        assert np.all(lambdas_np >= 1e-5), f"Lambdas below l_min: {lambdas_np}"
        assert np.all(lambdas_np <= 1e5), f"Lambdas above l_max: {lambdas_np}"

        # Cross-check: evaluate REML at the found lambdas using the old path
        old_logls = vmap(
            lambda lam, u, i: _reml_with_precomputed_iab(2, lam, evals, u, i)
        )(lambdas_new, Uab, Iab)

        np.testing.assert_allclose(
            logls_np,
            np.array(old_logls),
            rtol=1e-10,
            err_msg="Golden section split logls don't match old path evaluation",
        )
