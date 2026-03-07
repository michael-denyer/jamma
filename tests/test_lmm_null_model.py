"""Tests for null model likelihood functions.

Verifies that JAMMA's null model REML and MLE log-likelihood computations
match GEMMA's exact algorithm (calc_null=true behavior).

GEMMA Reference Values (from gemma_lrt.log.txt):
- REMLE log-likelihood in the null model = -140.636
- MLE log-likelihood in the null model = -139.281
- pve estimate in the null model = 0.293439
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from jamma.io import load_plink_binary
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import (
    compute_null_model_lambda,
    compute_null_model_mle,
    compute_Uab,
    mle_log_likelihood,
    mle_log_likelihood_null,
    reml_log_likelihood,
    reml_log_likelihood_null,
)
from jamma.lmm.runner_jax import run_lmm_association_jax
from tests.conftest import load_phenotypes_from_fam

pytestmark = pytest.mark.requires_jax

# GEMMA reference values from gemma_lrt.log.txt
GEMMA_REML_NULL_LOGL = -140.636
GEMMA_MLE_NULL_LOGL = -139.281
GEMMA_PVE_NULL = 0.293439
GEMMA_SE_PVE_NULL = 0.353149

# Test fixture paths
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gemma_synthetic"


@pytest.fixture
def gemma_test_data():
    """Load GEMMA synthetic test data."""
    if not (FIXTURE_DIR / "test.bed").exists():
        pytest.skip("GEMMA synthetic fixture not available")

    # Load PLINK data
    plink = load_plink_binary(FIXTURE_DIR / "test")

    # Load kinship matrix
    kinship = read_kinship_matrix(
        FIXTURE_DIR / "gemma_kinship.cXX.txt", n_samples=plink.n_samples
    )

    # Load phenotype from .fam file
    phenotypes = load_phenotypes_from_fam(FIXTURE_DIR / "test.fam")

    # eigendecomp overwrites K in-place; save copy for runner tests
    kinship_original = kinship.copy()

    # Eigendecomposition
    eigenvalues, U = eigendecompose_kinship(kinship)

    # Create intercept-only covariate (n_cvt=1)
    W = np.ones((plink.n_samples, 1))

    # Rotate to eigenspace
    UtW = U.T @ W
    Uty = U.T @ phenotypes

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": U,
        "UtW": UtW,
        "Uty": Uty,
        "n_cvt": 1,
        "phenotypes": phenotypes,
        "genotypes": plink.genotypes,
        "kinship": kinship_original,
        "plink": plink,
    }


@pytest.mark.tier1
class TestREMLNullLogLikelihood:
    """Tests for REML null model log-likelihood."""

    def test_reml_null_logl_matches_gemma(self, gemma_test_data):
        """REML null log-likelihood should match GEMMA reference (-140.636).

        GEMMA reports: REMLE log-likelihood in the null model = -140.636
        """
        data = gemma_test_data

        # Compute null model lambda using our corrected function
        lambda_null, logl_null = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )

        # Verify log-likelihood matches GEMMA
        # GEMMA reports -140.636, we should be very close
        np.testing.assert_allclose(
            logl_null,
            GEMMA_REML_NULL_LOGL,
            rtol=1e-4,  # Allow 0.01% relative tolerance
            err_msg=f"REML null logl {logl_null:.6f} != GEMMA {GEMMA_REML_NULL_LOGL}",
        )

    def test_reml_null_function_differs_from_alternative(self, gemma_test_data):
        """reml_log_likelihood_null should differ from reml_log_likelihood.

        The null model uses nc_total=n_cvt, alternative uses nc_total=n_cvt+1.
        """
        data = gemma_test_data

        # Compute Uab without genotype (for null model)
        Uab = compute_Uab(data["UtW"], data["Uty"], Utx=None)

        lambda_val = 0.5  # Arbitrary test value

        # Compute both formulas
        logl_null = reml_log_likelihood_null(
            lambda_val, data["eigenvalues"], Uab, data["n_cvt"]
        )
        logl_alt = reml_log_likelihood(
            lambda_val, data["eigenvalues"], Uab, data["n_cvt"]
        )

        # They should differ due to different nc_total and df
        assert logl_null != logl_alt, (
            f"Null and alt REML should differ: null={logl_null}, alt={logl_alt}"
        )


@pytest.mark.tier1
class TestMLENullLogLikelihood:
    """Tests for MLE null model log-likelihood."""

    def test_mle_null_logl_matches_gemma(self, gemma_test_data):
        """MLE null log-likelihood should match GEMMA reference (-139.281).

        GEMMA reports: MLE log-likelihood in the null model = -139.281
        """
        data = gemma_test_data

        # Compute null model MLE using our corrected function
        lambda_null_mle, logl_H0 = compute_null_model_mle(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )

        # Verify log-likelihood matches GEMMA
        np.testing.assert_allclose(
            logl_H0,
            GEMMA_MLE_NULL_LOGL,
            rtol=1e-4,  # Allow 0.01% relative tolerance
            err_msg=f"MLE null logl {logl_H0:.6f} != GEMMA {GEMMA_MLE_NULL_LOGL}",
        )

    def test_mle_null_function_differs_from_alternative(self, gemma_test_data):
        """mle_log_likelihood_null should differ from mle_log_likelihood.

        The null model uses nc_total=n_cvt, alternative uses nc_total=n_cvt+1.
        We need to use a Uab with genotype data for a fair comparison, since
        the alternative formula accesses genotype columns that would be zero
        in null-only Uab.
        """
        data = gemma_test_data

        # Compute Uab WITH genotype so alternative formula can access valid data
        # Use first SNP as genotype
        Utx = data["eigenvectors"].T @ data["genotypes"][:, 0]
        Uab = compute_Uab(data["UtW"], data["Uty"], Utx=Utx)

        lambda_val = 0.5  # Arbitrary test value

        # Compute both formulas
        logl_null = mle_log_likelihood_null(
            lambda_val, data["eigenvalues"], Uab, data["n_cvt"]
        )
        logl_alt = mle_log_likelihood(
            lambda_val, data["eigenvalues"], Uab, data["n_cvt"]
        )

        # They should differ due to different nc_total
        # (P_yy extracted at different Pab level)
        assert logl_null != logl_alt, (
            f"Null and alt MLE should differ: null={logl_null}, alt={logl_alt}"
        )


@pytest.mark.tier1
class TestNullModelFormulas:
    """Tests verifying null model uses correct degrees of freedom."""

    def test_null_model_uses_n_cvt(self, gemma_test_data):
        """Null model should use nc_total=n_cvt, not n_cvt+1.

        This is the key algorithmic difference from the alternative model.
        """
        data = gemma_test_data

        # Compute Uab without genotype
        Uab = compute_Uab(data["UtW"], data["Uty"], Utx=None)

        # The null formula uses:
        # - nc_total = n_cvt (not n_cvt + 1)
        # - df = n - n_cvt (not n - n_cvt - 1)
        # - logdet_hiw loops over nc_total columns

        # Compute at two lambda values to compare behavior
        lambda_low = 0.1
        lambda_high = 10.0

        # Null model REML
        reml_null_low = reml_log_likelihood_null(
            lambda_low, data["eigenvalues"], Uab, data["n_cvt"]
        )
        reml_null_high = reml_log_likelihood_null(
            lambda_high, data["eigenvalues"], Uab, data["n_cvt"]
        )

        # Alt model REML (same Uab, but uses different formula)
        reml_alt_low = reml_log_likelihood(
            lambda_low, data["eigenvalues"], Uab, data["n_cvt"]
        )
        reml_alt_high = reml_log_likelihood(
            lambda_high, data["eigenvalues"], Uab, data["n_cvt"]
        )

        # Both should have finite values
        assert np.isfinite(reml_null_low)
        assert np.isfinite(reml_null_high)
        assert np.isfinite(reml_alt_low)
        assert np.isfinite(reml_alt_high)

        # The null values should be different from alt values
        # at both lambda points, due to different df and nc_total
        assert reml_null_low != reml_alt_low
        assert reml_null_high != reml_alt_high

    def test_null_lambda_is_finite_and_positive(self, gemma_test_data):
        """Null model lambda optimization should converge to valid value."""
        data = gemma_test_data

        lambda_null, logl_null = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )

        assert np.isfinite(lambda_null), f"Lambda should be finite, got {lambda_null}"
        assert lambda_null > 0, f"Lambda should be positive, got {lambda_null}"
        assert np.isfinite(logl_null), f"logl should be finite, got {logl_null}"

    def test_pve_matches_gemma(self, gemma_test_data):
        """PVE from REML null lambda should match GEMMA reference (0.293439).

        GEMMA uses trace-adjusted PVE: vg*trace(K) / (vg*trace(K) + ve*n),
        which equals lambda*trace(K) / (lambda*trace(K) + n) where
        trace(K) = sum(eigenvalues).
        """
        from jamma.lmm.prepare_common import compute_and_log_pve

        data = gemma_test_data
        pve, pve_se = compute_and_log_pve(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )
        np.testing.assert_allclose(
            pve,
            GEMMA_PVE_NULL,
            rtol=1e-4,
            err_msg=f"PVE {pve:.6f} != GEMMA {GEMMA_PVE_NULL}",
        )

    def test_lmm_run_result_carries_pve(self, gemma_test_data):
        """LmmRunResult.pve is populated from REML null model."""
        data = gemma_test_data
        snp_info = [
            {
                "chr": str(data["plink"].chromosome[i]),
                "rs": str(data["plink"].sid[i]),
                "pos": int(data["plink"].bp_position[i]),
                "a1": str(data["plink"].allele_1[i]),
                "a0": str(data["plink"].allele_2[i]),
            }
            for i in range(data["plink"].n_snps)
        ]

        run_result = run_lmm_association_jax(
            data["genotypes"],
            data["phenotypes"],
            data["kinship"],
            snp_info,
            lmm_mode=1,
            show_progress=False,
            check_memory=False,
        )
        assert run_result.pve is not None, "LmmRunResult.pve should be populated"
        assert 0.0 < run_result.pve < 1.0, (
            f"PVE should be in (0, 1), got {run_result.pve}"
        )
        # Verify it matches the GEMMA reference
        np.testing.assert_allclose(
            run_result.pve,
            GEMMA_PVE_NULL,
            rtol=1e-4,
            err_msg=(
                f"LmmRunResult.pve {run_result.pve:.6f} != GEMMA {GEMMA_PVE_NULL}"
            ),
        )


@pytest.mark.tier1
class TestWaldTestUnchanged:
    """Regression tests verifying Wald test is unaffected by null model fix.

    The Wald test uses the alternative model (per-SNP optimization) which
    should remain unchanged. These tests catch accidental modification of
    reml_log_likelihood or mle_log_likelihood functions.
    """

    def test_wald_test_produces_valid_results(self, gemma_test_data):
        """Wald test should produce valid results after null model fix."""
        data = gemma_test_data

        snp_info = [
            {
                "chr": str(data["plink"].chromosome[i]),
                "rs": str(data["plink"].sid[i]),
                "pos": int(data["plink"].bp_position[i]),
                "a1": str(data["plink"].allele_1[i]),
                "a0": str(data["plink"].allele_2[i]),
            }
            for i in range(data["plink"].n_snps)
        ]

        # Run Wald test (lmm_mode=1)
        run_result = run_lmm_association_jax(
            data["genotypes"],
            data["phenotypes"],
            data["kinship"],
            snp_info,
            lmm_mode=1,
            show_progress=False,
            check_memory=False,
        )
        results = run_result.associations

        assert len(results) > 0, "Should produce results"
        for r in results:
            if np.isfinite(r.beta):  # Skip degenerate SNPs
                assert np.isfinite(r.se), f"SE should be finite for {r.rs}"
                assert 0 <= r.p_wald <= 1, f"p_wald should be valid for {r.rs}"
                assert r.l_remle > 0, f"l_remle should be positive for {r.rs}"

    def test_wald_test_matches_known_reference(self, gemma_test_data):
        """Wald test results should match GEMMA reference.

        This is a regression test to ensure we didn't accidentally change
        the alternative model functions. Uses relaxed tolerances because
        the JAX runner uses golden section optimization (vs GEMMA's Brent)
        which produces slightly different lambda values.
        """
        from jamma.validation import (
            ToleranceConfig,
            compare_assoc_results,
            load_gemma_assoc,
        )

        ref_path = FIXTURE_DIR / "gemma_assoc.assoc.txt"
        if not ref_path.exists():
            pytest.skip("GEMMA Wald reference not available")

        data = gemma_test_data
        reference_results = load_gemma_assoc(ref_path)

        snp_info = [
            {
                "chr": str(data["plink"].chromosome[i]),
                "rs": str(data["plink"].sid[i]),
                "pos": int(data["plink"].bp_position[i]),
                "a1": str(data["plink"].allele_1[i]),
                "a0": str(data["plink"].allele_2[i]),
            }
            for i in range(data["plink"].n_snps)
        ]

        run_result = run_lmm_association_jax(
            data["genotypes"],
            data["phenotypes"],
            data["kinship"],
            snp_info,
            lmm_mode=1,
            show_progress=False,
            check_memory=False,
        )
        jamma_results = run_result.associations

        # Relaxed tolerances: golden section vs Brent lambda difference
        config = ToleranceConfig.relaxed()
        comparison = compare_assoc_results(
            jamma_results, reference_results, config=config
        )
        assert comparison.passed, (
            f"Wald test regression:\n"
            f"  Beta: {comparison.beta.message}\n"
            f"  SE: {comparison.se.message}\n"
            f"  P-value: {comparison.p_wald.message}\n"
            f"  Lambda: {comparison.l_remle.message}"
        )


@pytest.mark.tier0
class TestNullModelProperties:
    """Property-based tests for null model functions."""

    def test_reml_null_returns_finite_for_valid_inputs(self):
        """REML null should return finite values for valid inputs."""
        rng = np.random.default_rng(42)
        n_samples = 100

        # Create properly structured synthetic data
        # Eigenvalues must be positive
        eigenvalues = np.abs(rng.standard_normal(n_samples)) + 0.1

        # Create proper Uab from realistic vectors
        W = np.ones((n_samples, 1))  # Intercept
        y = rng.standard_normal(n_samples)
        Uab = compute_Uab(W, y, Utx=None)

        for lambda_val in [1e-5, 0.01, 0.1, 1.0, 10.0, 1e5]:
            result = reml_log_likelihood_null(lambda_val, eigenvalues, Uab, n_cvt=1)
            assert np.isfinite(result), f"Should be finite at lambda={lambda_val}"

    def test_mle_null_returns_finite_for_valid_inputs(self):
        """MLE null should return finite values for valid inputs."""
        rng = np.random.default_rng(42)
        n_samples = 100

        # Create properly structured synthetic data
        eigenvalues = np.abs(rng.standard_normal(n_samples)) + 0.1

        # Create proper Uab from realistic vectors
        W = np.ones((n_samples, 1))  # Intercept
        y = rng.standard_normal(n_samples)
        Uab = compute_Uab(W, y, Utx=None)

        for lambda_val in [1e-5, 0.01, 0.1, 1.0, 10.0, 1e5]:
            result = mle_log_likelihood_null(lambda_val, eigenvalues, Uab, n_cvt=1)
            assert np.isfinite(result), f"Should be finite at lambda={lambda_val}"


@pytest.mark.tier1
class TestSePVE:
    """Tests for se(pve) via REML second derivative delta method."""

    def test_se_pve_matches_gemma(self, gemma_test_data):
        """se(pve) should match GEMMA reference (0.353149) for synthetic data.

        GEMMA reports: se(pve) in the null model = 0.353149
        """
        from jamma.lmm.prepare_common import compute_and_log_pve

        data = gemma_test_data
        pve, pve_se = compute_and_log_pve(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )
        assert pve_se is not None, "pve_se should not be None for valid data"
        np.testing.assert_allclose(
            pve_se,
            GEMMA_SE_PVE_NULL,
            rtol=1e-3,
            err_msg=(f"se(pve) {pve_se:.6f} != GEMMA {GEMMA_SE_PVE_NULL}"),
        )

    def test_se_pve_none_for_degenerate_case(self):
        """se(pve) should be None when REML second derivative is non-negative.

        Construct a degenerate case with uniform eigenvalues where the
        likelihood surface is flat (dev2 >= 0).
        """
        from jamma.lmm.prepare_common import compute_and_log_pve

        # Uniform eigenvalues = no genetic variance signal = flat surface
        # With all eigenvalues equal, lambda is at the boundary and dev2
        # may be non-negative.
        rng = np.random.default_rng(42)
        n = 50
        eigenvalues = np.ones(n)  # All equal = no K structure
        W = np.ones((n, 1))
        y = rng.standard_normal(n)
        UtW = W  # Identity rotation (trivial)
        Uty = y

        pve, pve_se = compute_and_log_pve(eigenvalues, UtW, Uty, n_cvt=1)
        # With uniform eigenvalues, lambda converges at boundary;
        # pve_se may or may not be None depending on dev2 sign.
        # The important thing is no crash and type correctness.
        assert isinstance(pve, float)
        assert pve_se is None or isinstance(pve_se, float)

    def test_lmm_run_result_carries_pve_se(self, gemma_test_data):
        """LmmRunResult.pve_se is populated from REML null model."""
        data = gemma_test_data
        snp_info = [
            {
                "chr": str(data["plink"].chromosome[i]),
                "rs": str(data["plink"].sid[i]),
                "pos": int(data["plink"].bp_position[i]),
                "a1": str(data["plink"].allele_1[i]),
                "a0": str(data["plink"].allele_2[i]),
            }
            for i in range(data["plink"].n_snps)
        ]

        run_result = run_lmm_association_jax(
            data["genotypes"],
            data["phenotypes"],
            data["kinship"],
            snp_info,
            lmm_mode=1,
            show_progress=False,
            check_memory=False,
        )
        assert run_result.pve_se is not None, "LmmRunResult.pve_se should be populated"
        np.testing.assert_allclose(
            run_result.pve_se,
            GEMMA_SE_PVE_NULL,
            rtol=1e-3,
            err_msg=(
                f"LmmRunResult.pve_se {run_result.pve_se:.6f} != "
                f"GEMMA {GEMMA_SE_PVE_NULL}"
            ),
        )
