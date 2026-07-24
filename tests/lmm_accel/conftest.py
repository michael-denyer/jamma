"""Fixtures shared across the _lmm_accel kernel-family test modules.

Held in a conftest so pytest supplies them by name. The modules do not
import them, which is what made the previous single-file layout awkward
to split.
"""

import numpy as np
import pytest

from jamma.lmm.likelihood_numpy import (
    golden_section_optimize_lambda_mle_numpy,
)
from tests.lmm_accel._helpers import _make_general_score_lrt_data


@pytest.fixture
def synthetic_wald_data():
    """Deterministic synthetic Wald test data for n_cvt=1.

    Returns:
        Tuple of (eigenvalues, Uab_batch, n_samples).
    """
    rng = np.random.default_rng(42)
    n_samples, n_snps = 200, 50
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    # Build physically meaningful Uab from w, x, y vectors so columns have
    # proper cross-product structure and Pab recursion is well-conditioned.
    Uab_batch = np.zeros((n_snps, n_samples, 6), dtype=np.float64)
    for i in range(n_snps):
        w = np.abs(rng.standard_normal(n_samples)) + 1.0  # positive ww
        x = np.abs(rng.standard_normal(n_samples)) + 0.5  # positive xx
        y = rng.standard_normal(n_samples)
        Uab_batch[i, :, 0] = w * w  # ww
        Uab_batch[i, :, 1] = w * x  # wx
        Uab_batch[i, :, 2] = w * y  # wy
        Uab_batch[i, :, 3] = x * x  # xx
        Uab_batch[i, :, 4] = x * y  # xy
        Uab_batch[i, :, 5] = y * y  # yy
    return eigenvalues, Uab_batch, n_samples


@pytest.fixture
def split_wald_data():
    """Deterministic data for split-Uab tests.

    Builds physically meaningful Uab from w, x, y vectors so the split
    construction matches the full construction exactly.

    Returns:
        Tuple of (eigenvalues, UtW, Uty, UtG, n_samples, n_snps).
    """
    rng = np.random.default_rng(42)
    n_samples, n_snps = 200, 50
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))
    return eigenvalues, UtW, Uty, UtG, n_samples, n_snps


@pytest.fixture
def score_lrt_data(synthetic_wald_data):
    """Extends synthetic_wald_data with null-model Hi_eval and logl_H0.

    Computes the null-model MLE lambda via golden section on the null Uab
    (no genotype), then derives Hi_eval_null = 1/(lambda_null*eval + 1)
    and logl_H0 (null MLE log-likelihood).
    """
    eigenvalues, Uab_batch, n_samples = synthetic_wald_data
    n_cvt = 1

    # Build null Uab: zero genotype columns (wx=0, xx=0, xy=0).
    # The null model only uses ww, wy, yy columns.
    Uab_null = np.zeros((1, n_samples, 6), dtype=np.float64)
    Uab_null[0, :, 0] = Uab_batch[0, :, 0]  # ww (invariant)
    Uab_null[0, :, 2] = Uab_batch[0, :, 2]  # wy (invariant)
    Uab_null[0, :, 5] = Uab_batch[0, :, 5]  # yy (invariant)

    # Null MLE lambda optimization
    lambdas_null, logls_null = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_null,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    lambda_null = float(lambdas_null[0])
    logl_H0 = float(logls_null[0])

    # Hi_eval_null: Score test uses this fixed weight vector
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

    return eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0


@pytest.fixture
def general_score_lrt_ncvt2(synthetic_covariate_data_ncvt2):
    """Score/LRT data for n_cvt=2."""
    return _make_general_score_lrt_data(synthetic_covariate_data_ncvt2)
