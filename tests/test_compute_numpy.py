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
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)
from jamma.lmm.prepare_common import _compute_null_model_common
from tests.conftest import load_phenotypes_from_fam

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

    Returns dict with eigenvalues, eigenvectors, UtW, Uty, genotypes,
    col_means, uab_invariant_soa, and a small Uab batch for parity tests.
    """
    plink_data = load_plink_binary(str(MOUSE_HS1940_DATA))
    genotypes = plink_data.genotypes
    phenotypes = load_phenotypes_from_fam(Path(str(MOUSE_HS1940_DATA) + ".fam"))
    K = read_kinship_matrix(str(MOUSE_HS1940_KINSHIP))

    n_samples = genotypes.shape[0]
    n_cvt = 1

    # Eigendecomposition
    eigenvalues, U = np.linalg.eigh(K)

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
    uab_var_soa = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG)
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)

    # Null model for Score/LRT (lmm_mode=4 computes all needed fields)
    logl_H0, lambda_null_mle, Hi_eval_null = _compute_null_model_common(
        lmm_mode=4,
        eigenvalues_np=eigenvalues,
        UtW=UtW,
        Uty=Uty,
        n_cvt=n_cvt,
        show_progress=False,
    )

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
