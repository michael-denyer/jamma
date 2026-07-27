"""Parity tests for SoA-native Score/LRT dispatch vs full-Uab equivalents.

Validates that the split SoA C kernels produce identical results to the
full-Uab batch C kernels, ensuring that eliminating reconstruct_uab_from_soa
in the runner paths does not change computed statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _compute_lrt_numpy,
    _compute_score_numpy,
)
from jamma.lmm.likelihood import compute_null_model_mle
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)

pytestmark = pytest.mark.tier0

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURE_ROOT = Path(__file__).parent / "fixtures"
MOUSE_HS1940_DIR = _FIXTURE_ROOT / "mouse_hs1940"
MOUSE_HS1940_DATA = MOUSE_HS1940_DIR / "mouse_hs1940"
MOUSE_HS1940_KINSHIP = MOUSE_HS1940_DIR / "mouse_hs1940_kinship.cXX.txt"


@pytest.fixture(scope="module")
def mouse_data():
    """Load mouse_hs1940 fixture and prepare eigendecomposition + Uab arrays.

    Uses a synthetic phenotype with known signal to ensure a well-conditioned
    MLE null model (finite logl_H0). The mouse_hs1940 column-1 phenotype
    produces a degenerate MLE landscape (NaN logl_H0 at boundary lambda).
    """
    plink_data = load_plink_binary(MOUSE_HS1940_DATA)
    genotypes = plink_data.genotypes
    K = read_kinship_matrix(MOUSE_HS1940_KINSHIP)

    n_samples = genotypes.shape[0]
    n_cvt = 1

    # Eigendecomposition
    eigenvalues, U = np.linalg.eigh(K)

    # Generate synthetic phenotype with genetic signal (ensures finite MLE null)
    rng = np.random.default_rng(42)
    # y = K @ beta + noise, where beta is random — ensures non-degenerate null
    phenotypes = K @ rng.standard_normal(n_samples) * 0.5 + rng.standard_normal(
        n_samples
    )

    # Build intercept-only covariate matrix
    W = np.ones((n_samples, 1))
    UtW = U.T @ W
    Uty = U.T @ phenotypes

    # Get a small subset of SNPs for parity tests (first 50)
    n_test_snps = min(50, genotypes.shape[1])
    geno_subset = genotypes[:, :n_test_snps].astype(np.float64)

    # Mean-impute
    col_means = np.nanmean(geno_subset, axis=0)
    missing = np.isnan(geno_subset)
    if missing.any():
        geno_subset = np.where(missing, col_means[None, :], geno_subset)

    # Rotate
    UtG = U.T @ geno_subset

    # Build full Uab
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)

    # Build SoA split
    uab_var_soa = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)

    # Null model MLE for Score/LRT
    lambda_null_mle, logl_H0 = compute_null_model_mle(eigenvalues, UtW, Uty, n_cvt)
    Hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues + 1.0)

    return {
        "eigenvalues": eigenvalues,
        "n_samples": n_samples,
        "n_cvt": n_cvt,
        "Uab_batch": Uab_batch,
        "uab_var_soa": uab_var_soa,
        "uab_inv_soa": uab_inv_soa,
        "Hi_eval_null": Hi_eval_null,
        "logl_H0": logl_H0,
        "lambda_null_mle": lambda_null_mle,
    }


@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension unavailable")
class TestScoreSplitParity:
    """SoA Score split produces identical results to full-Uab Score."""

    def test_score_split_parity(self, mouse_data):
        """Score split C vs full-Uab C: betas, ses, p_scores must match."""
        from jamma.lmm.compute_numpy import _compute_score_split_numpy

        d = mouse_data
        # Full-Uab path
        full_result = _compute_score_numpy(
            d["n_cvt"],
            d["eigenvalues"],
            d["Hi_eval_null"],
            d["Uab_batch"],
            d["n_samples"],
            n_threads=1,
        )

        # SoA split path
        split_result = _compute_score_split_numpy(
            d["n_cvt"],
            d["eigenvalues"],
            d["Hi_eval_null"],
            d["uab_var_soa"],
            d["uab_inv_soa"],
            d["n_samples"],
            n_threads=1,
        )

        np.testing.assert_allclose(
            split_result["betas"],
            full_result["betas"],
            rtol=1e-12,
            err_msg="Score split betas differ from full-Uab",
        )
        np.testing.assert_allclose(
            split_result["ses"],
            full_result["ses"],
            rtol=1e-12,
            err_msg="Score split ses differ from full-Uab",
        )
        np.testing.assert_allclose(
            split_result["p_scores"],
            full_result["p_scores"],
            rtol=1e-12,
            err_msg="Score split p_scores differ from full-Uab",
        )


@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension unavailable")
class TestLrtSplitParity:
    """SoA LRT split produces identical results to full-Uab LRT."""

    def test_lrt_split_parity(self, mouse_data):
        """LRT split C vs full-Uab C: lambdas_mle and p_lrts must match."""
        from jamma.lmm.compute_numpy import _compute_lrt_split_numpy

        d = mouse_data
        l_min, l_max = 1e-5, 1e5
        n_grid, n_refine = 50, 20

        # Full-Uab path
        full_result = _compute_lrt_numpy(
            d["n_cvt"],
            d["eigenvalues"],
            d["Uab_batch"],
            l_min,
            l_max,
            n_grid,
            n_refine,
            d["logl_H0"],
            n_threads=1,
        )

        # SoA split path
        split_result = _compute_lrt_split_numpy(
            d["n_cvt"],
            d["eigenvalues"],
            d["uab_var_soa"],
            d["uab_inv_soa"],
            d["n_samples"],
            l_min,
            l_max,
            n_grid,
            n_refine,
            d["logl_H0"],
            n_threads=1,
        )

        np.testing.assert_allclose(
            split_result["lambdas_mle"],
            full_result["lambdas_mle"],
            rtol=5e-5,
            err_msg="LRT split lambdas_mle differ from full-Uab",
        )
        np.testing.assert_allclose(
            split_result["p_lrts"],
            full_result["p_lrts"],
            rtol=5e-3,
            err_msg="LRT split p_lrts differ from full-Uab",
        )


@pytest.fixture(scope="module")
def degenerate_data(mouse_data):
    """Extend mouse_data with constant-genotype (degenerate) SNP columns.

    Injects 3 constant columns (all-0, all-1, all-2) into the genotype
    matrix, rebuilds Uab in both full and SoA layouts. Degenerate SNPs
    have zero variance after covariate projection (P_xx <= 0), so
    beta/se/p-values must be NaN.
    """
    d = mouse_data
    n_samples = d["n_samples"]

    # Load the original rotated genotypes for the first 10 well-conditioned SNPs
    plink_data = load_plink_binary(MOUSE_HS1940_DATA)
    genotypes = plink_data.genotypes
    K = read_kinship_matrix(MOUSE_HS1940_KINSHIP)
    eigenvalues, U = np.linalg.eigh(K)

    geno_subset = genotypes[:, :10].astype(np.float64)
    col_means = np.nanmean(geno_subset, axis=0)
    missing = np.isnan(geno_subset)
    if missing.any():
        geno_subset = np.where(missing, col_means[None, :], geno_subset)

    # Inject degenerate columns: all-zero genotype produces P_xx=0 exactly
    # (zero vector after eigenrotation). Also add a near-zero-variance column.
    const_cols = np.column_stack(
        [
            np.zeros(n_samples),  # zero vector → P_xx = 0 → NaN
            np.zeros(n_samples),  # second zero column for robustness
            np.zeros(n_samples),  # third zero column
        ]
    )
    geno_with_degen = np.column_stack([geno_subset, const_cols])

    W = np.ones((n_samples, 1))
    UtW = U.T @ W
    Uty = U.T @ (
        K @ np.random.default_rng(42).standard_normal(n_samples) * 0.5
        + np.random.default_rng(42).standard_normal(n_samples)
    )
    UtG = U.T @ geno_with_degen

    n_cvt = 1
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    lambda_null_mle, logl_H0 = compute_null_model_mle(eigenvalues, UtW, Uty, n_cvt)
    Hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues + 1.0)

    return {
        "eigenvalues": eigenvalues,
        "n_samples": n_samples,
        "n_cvt": n_cvt,
        "Uab_batch": Uab_batch,
        "uab_var_soa": uab_var_soa,
        "uab_inv_soa": uab_inv_soa,
        "Hi_eval_null": Hi_eval_null,
        "logl_H0": logl_H0,
        "n_normal": 10,  # first 10 are well-conditioned
        "n_degenerate": 3,  # last 3 are constant
    }


@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension unavailable")
class TestDegenerateSplitParity:
    """Split and batch paths agree on NaN output for degenerate SNPs."""

    def test_score_degenerate_nan_parity(self, degenerate_data):
        """Score split produces NaN for constant-genotype SNPs, matching batch."""
        from jamma.lmm.compute_numpy import _compute_score_split_numpy

        d = degenerate_data
        full_result = _compute_score_numpy(
            d["n_cvt"],
            d["eigenvalues"],
            d["Hi_eval_null"],
            d["Uab_batch"],
            d["n_samples"],
            n_threads=1,
        )
        split_result = _compute_score_split_numpy(
            d["n_cvt"],
            d["eigenvalues"],
            d["Hi_eval_null"],
            d["uab_var_soa"],
            d["uab_inv_soa"],
            d["n_samples"],
            n_threads=1,
        )

        # Well-conditioned SNPs must match
        n = d["n_normal"]
        np.testing.assert_allclose(
            split_result["p_scores"][:n],
            full_result["p_scores"][:n],
            rtol=1e-12,
            err_msg="Score normal SNPs: split vs batch mismatch",
        )

        # Degenerate SNPs must be NaN in both paths
        assert np.all(np.isnan(full_result["p_scores"][n:])), (
            "Batch Score should return NaN for constant-genotype SNPs"
        )
        assert np.all(np.isnan(split_result["p_scores"][n:])), (
            "Split Score should return NaN for constant-genotype SNPs"
        )

    def test_lrt_degenerate_parity(self, degenerate_data):
        """LRT split matches batch for degenerate SNPs (zero-signal → p≈1)."""
        from jamma.lmm.compute_numpy import _compute_lrt_split_numpy

        d = degenerate_data
        l_min, l_max = 1e-5, 1e5
        n_grid, n_refine = 50, 20

        full_result = _compute_lrt_numpy(
            d["n_cvt"],
            d["eigenvalues"],
            d["Uab_batch"],
            l_min,
            l_max,
            n_grid,
            n_refine,
            d["logl_H0"],
            n_threads=1,
        )
        split_result = _compute_lrt_split_numpy(
            d["n_cvt"],
            d["eigenvalues"],
            d["uab_var_soa"],
            d["uab_inv_soa"],
            d["n_samples"],
            l_min,
            l_max,
            n_grid,
            n_refine,
            d["logl_H0"],
            n_threads=1,
        )

        # Well-conditioned SNPs must match
        n = d["n_normal"]
        np.testing.assert_allclose(
            split_result["p_lrts"][:n],
            full_result["p_lrts"][:n],
            rtol=5e-3,
            err_msg="LRT normal SNPs: split vs batch mismatch",
        )

        # Degenerate SNPs: LRT finds no signal (LR≈0, p≈1). Both paths
        # must agree — NaN pattern and finite values must match exactly.
        np.testing.assert_allclose(
            split_result["p_lrts"][n:],
            full_result["p_lrts"][n:],
            rtol=5e-3,
            equal_nan=True,
            err_msg="LRT degenerate SNPs: split vs batch mismatch",
        )
        np.testing.assert_allclose(
            split_result["lambdas_mle"][n:],
            full_result["lambdas_mle"][n:],
            rtol=5e-5,
            equal_nan=True,
            err_msg="LRT degenerate lambdas_mle: split vs batch mismatch",
        )
