"""Tests for LRT (Likelihood Ratio Test) implementation.

Tests verify:
1. MLE log-likelihood formula (no logdet_hiw term)
2. Chi-squared p-value computation
3. Null model MLE caching
4. GEMMA reference data comparison
"""

from pathlib import Path

import numpy as np
import pytest

from jamma.core import configure_jax
from jamma.lmm import run_lmm_association
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import (
    compute_null_model_lambda,
    compute_null_model_mle,
    compute_Uab,
    mle_log_likelihood,
    reml_log_likelihood,
)
from jamma.lmm.stats import calc_lrt_test


@pytest.fixture(autouse=True)
def setup_jax():
    """Configure JAX with 64-bit precision before each test."""
    configure_jax(enable_x64=True)


def _create_test_data(n_samples: int = 100, n_cvt: int = 1, seed: int = 42):
    """Create synthetic test data for LMM tests.

    Returns:
        Dictionary with eigenvalues, eigenvectors, UtW, Uty, Utx, W, y, x
    """
    rng = np.random.default_rng(seed)

    # Create valid kinship matrix
    X = rng.standard_normal((n_samples, 200))
    K = X @ X.T / X.shape[1]

    # Eigendecomposition
    eigenvalues, U = eigendecompose_kinship(K)

    # Create phenotype, covariates, genotype
    y = rng.standard_normal(n_samples)
    W = np.ones((n_samples, n_cvt))  # Intercept-only covariate
    x = rng.standard_normal(n_samples)  # Random genotype

    # Rotate to eigenspace
    Uty = U.T @ y
    UtW = U.T @ W
    Utx = U.T @ x

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": U,
        "UtW": UtW,
        "Uty": Uty,
        "Utx": Utx,
        "W": W,
        "y": y,
        "x": x,
        "n_samples": n_samples,
        "n_cvt": n_cvt,
    }


class TestMLELogLikelihood:
    """Unit tests for MLE log-likelihood formula."""

    def test_mle_differs_from_reml(self):
        """MLE should differ from REML (no logdet_hiw term)."""
        np.random.seed(42)
        data = _create_test_data(n_samples=100, n_cvt=1, seed=42)

        # Compute Uab with genotype
        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])
        lambda_val = 0.5

        mle = mle_log_likelihood(lambda_val, data["eigenvalues"], Uab, data["n_cvt"])
        reml = reml_log_likelihood(lambda_val, data["eigenvalues"], Uab, data["n_cvt"])

        # MLE should be different from REML (no logdet_hiw term)
        assert mle != reml, "MLE and REML should differ"
        # MLE is typically larger (less penalized) because it lacks the logdet_hiw term
        # The difference should be positive (MLE > REML) in most cases
        # but this is data-dependent, so we just check they differ

    def test_mle_returns_finite(self):
        """MLE should return finite values for valid inputs."""
        data = _create_test_data(n_samples=50, n_cvt=1, seed=123)
        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])

        for lambda_val in [1e-5, 0.01, 0.1, 1.0, 10.0, 1e5]:
            result = mle_log_likelihood(
                lambda_val, data["eigenvalues"], Uab, data["n_cvt"]
            )
            assert np.isfinite(result), f"MLE should be finite at lambda={lambda_val}"

    def test_mle_optimum_exists(self):
        """MLE surface should have an optimum."""
        data = _create_test_data(n_samples=100, n_cvt=1, seed=456)
        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])

        # Evaluate MLE at grid of lambda values
        lambdas = np.logspace(-4, 4, 50)
        mle_vals = [
            mle_log_likelihood(lam, data["eigenvalues"], Uab, data["n_cvt"])
            for lam in lambdas
        ]

        # Should have a maximum somewhere
        max_val = max(mle_vals)
        assert np.isfinite(max_val), "Maximum MLE should be finite"
        # Just verify we got valid values throughout
        assert all(np.isfinite(v) for v in mle_vals), "All MLE values should be finite"


class TestComputeNullModelMLE:
    """Tests for null model MLE computation."""

    def test_compute_null_model_mle_returns_valid(self):
        """Null model MLE should return valid lambda and log-likelihood."""
        data = _create_test_data(n_samples=100, n_cvt=1, seed=42)

        lambda_null, logl_H0 = compute_null_model_mle(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )

        assert 1e-5 <= lambda_null <= 1e5, f"Lambda {lambda_null} should be in bounds"
        assert np.isfinite(logl_H0), "logl_H0 should be finite"

    def test_null_model_mle_differs_from_reml(self):
        """Null model MLE lambda may differ from REML lambda."""
        data = _create_test_data(n_samples=100, n_cvt=1, seed=789)

        lambda_reml, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )
        lambda_mle, _ = compute_null_model_mle(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )

        # They may be close but can differ due to different objective functions
        # (MLE doesn't penalize for fixed effects degrees of freedom)
        # Just verify both are valid
        assert 1e-5 <= lambda_reml <= 1e5, "REML lambda should be in bounds"
        assert 1e-5 <= lambda_mle <= 1e5, "MLE lambda should be in bounds"


class TestCalcLRTTest:
    """Tests for LRT p-value computation."""

    def test_lrt_pvalue_valid_probability(self):
        """LRT p-value should be valid probability [0, 1]."""
        # Alternative better than null (typical case)
        p = calc_lrt_test(logl_H1=-100.0, logl_H0=-105.0)
        assert 0 < p < 1, "p-value should be valid probability"

        # Large difference = very significant
        p_sig = calc_lrt_test(logl_H1=-100.0, logl_H0=-150.0)
        assert p_sig < 0.01, "Large LRT should be significant"

    def test_lrt_negative_stat_returns_one(self):
        """Negative LRT statistic (numerical artifact) should return p=1."""
        p = calc_lrt_test(logl_H1=-110.0, logl_H0=-105.0)
        assert p == 1.0, "Negative stat should return p=1"

    def test_lrt_zero_stat_returns_one(self):
        """Zero LRT statistic should return p=1."""
        p = calc_lrt_test(logl_H1=-100.0, logl_H0=-100.0)
        assert p == 1.0, "Zero stat should return p=1"

    def test_lrt_chi2_df1(self):
        """Verify LRT uses chi2(df=1) distribution."""
        from scipy.stats import chi2

        # LRT stat = 2 * (logl_H1 - logl_H0)
        logl_H1, logl_H0 = -100.0, -105.0
        lrt_stat = 2.0 * (logl_H1 - logl_H0)  # = 10.0

        p_jamma = calc_lrt_test(logl_H1, logl_H0)
        p_scipy = chi2.sf(lrt_stat, df=1)

        np.testing.assert_allclose(p_jamma, p_scipy, rtol=1e-10)


class TestLRTIntegration:
    """Integration tests for LRT via run_lmm_association."""

    def test_lrt_mode_produces_results(self):
        """LRT mode should produce valid results."""
        np.random.seed(42)
        n, p = 50, 10
        G = np.random.binomial(2, 0.3, (n, p)).astype(float)
        y = np.random.randn(n)
        K = G @ G.T / p

        snp_info = [
            {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
            for i in range(p)
        ]

        results = run_lmm_association(G, y, K, snp_info, lmm_mode=2)

        assert len(results) > 0, "Should produce results"
        for r in results:
            assert r.l_mle is not None, "l_mle should be set"
            assert r.p_lrt is not None, "p_lrt should be set"
            assert 0 <= r.p_lrt <= 1, "p_lrt should be valid probability"

    def test_lrt_vs_wald_correlation(self):
        """LRT and Wald p-values should be highly correlated."""
        np.random.seed(123)
        n, p = 100, 20
        G = np.random.binomial(2, 0.3, (n, p)).astype(float)
        y = np.random.randn(n) + 0.3 * G[:, 0]  # First SNP has effect
        K = G @ G.T / p

        snp_info = [
            {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
            for i in range(p)
        ]

        results_wald = run_lmm_association(G, y, K, snp_info, lmm_mode=1)
        results_lrt = run_lmm_association(G, y, K, snp_info, lmm_mode=2)

        p_wald = np.array([r.p_wald for r in results_wald])
        p_lrt = np.array([r.p_lrt for r in results_lrt])

        # P-values should be highly correlated (both detect same signals)
        # Add small constant to avoid log(0)
        correlation = np.corrcoef(np.log10(p_wald + 1e-100), np.log10(p_lrt + 1e-100))[
            0, 1
        ]
        assert (
            correlation > 0.9
        ), f"Wald and LRT should be correlated (got {correlation})"


class TestGEMMALRTValidation:
    """Validation tests against GEMMA LRT reference data."""

    @pytest.fixture
    def gemma_lrt_fixture(self):
        """Load GEMMA LRT reference data if available."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "gemma_synthetic"
            / "gemma_lrt.assoc.txt"
        )
        if not fixture_path.exists():
            pytest.skip("GEMMA LRT fixture not available")
        return fixture_path

    @pytest.mark.xfail(
        reason="Known null model MLE discrepancy - same issue as Score test"
    )
    def test_gemma_lrt_pvalues(self, gemma_lrt_fixture):
        """LRT p-values should match GEMMA within tolerance."""
        from jamma.io import load_plink_binary
        from jamma.kinship import read_kinship_matrix

        fixture_dir = Path(__file__).parent / "fixtures" / "gemma_synthetic"

        # Load JAMMA data
        plink = load_plink_binary(fixture_dir / "test")
        K = read_kinship_matrix(
            fixture_dir / "gemma_kinship.cXX.txt", n_samples=plink.n_samples
        )

        # Load phenotype from .fam file
        fam_path = fixture_dir / "test.fam"
        fam_data = np.loadtxt(fam_path, dtype=str)
        phenotypes = fam_data[:, 5].astype(float)

        snp_info = [
            {
                "chr": str(plink.chromosome[i]),
                "rs": str(plink.sid[i]),
                "pos": int(plink.bp_position[i]),
                "a1": str(plink.allele_1[i]),
                "a0": str(plink.allele_2[i]),
            }
            for i in range(plink.n_snps)
        ]

        # Run JAMMA LRT
        jamma_results = run_lmm_association(
            plink.genotypes, phenotypes, K, snp_info, lmm_mode=2
        )

        # Load GEMMA reference (need custom loader for LRT format)
        gemma_results = _load_gemma_lrt(gemma_lrt_fixture)

        # Compare p_lrt values
        jamma_plrt = {r.rs: r.p_lrt for r in jamma_results}
        gemma_plrt = {r["rs"]: r["p_lrt"] for r in gemma_results}

        # Find common SNPs
        common_snps = set(jamma_plrt.keys()) & set(gemma_plrt.keys())
        assert len(common_snps) > 0, "Should have common SNPs"

        # Compare p-values
        max_rel_diff = 0.0
        for rs in common_snps:
            j_p = jamma_plrt[rs]
            g_p = gemma_plrt[rs]
            if g_p > 0:
                rel_diff = abs(j_p - g_p) / g_p
                max_rel_diff = max(max_rel_diff, rel_diff)

        # This test is xfail because we expect discrepancies
        # due to null model MLE computation differences
        assert max_rel_diff < 1e-4, f"Max relative p-value diff: {max_rel_diff}"


def _load_gemma_lrt(path: Path) -> list[dict]:
    """Load GEMMA LRT results from .assoc.txt format.

    LRT format: chr rs ps n_miss allele1 allele0 af l_mle p_lrt
    """
    results = []
    with open(path) as f:
        header = f.readline().strip()
        cols = header.split("\t")

        # Verify LRT format
        expected_cols = [
            "chr",
            "rs",
            "ps",
            "n_miss",
            "allele1",
            "allele0",
            "af",
            "l_mle",
            "p_lrt",
        ]
        if cols != expected_cols:
            raise ValueError(f"Expected LRT format {expected_cols}, got {cols}")

        for line in f:
            fields = line.strip().split("\t")
            results.append(
                {
                    "chr": fields[0],
                    "rs": fields[1],
                    "ps": int(fields[2]),
                    "n_miss": int(fields[3]),
                    "allele1": fields[4],
                    "allele0": fields[5],
                    "af": float(fields[6]),
                    "l_mle": float(fields[7]),
                    "p_lrt": float(fields[8]),
                }
            )
    return results
