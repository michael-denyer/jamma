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
from tests.builders import (
    covariate_lmm_inputs,
    gram_uab_batch,
    rotated_lmm_inputs,
)
from tests.lmm_accel._helpers import GeneralCase


@pytest.fixture
def synthetic_covariate_data_ncvt2() -> GeneralCase:
    """200 samples, 50 SNPs, 2 covariates."""
    return GeneralCase(covariate_lmm_inputs(n_cvt=2, seed=42))


@pytest.fixture
def synthetic_covariate_data_ncvt4() -> GeneralCase:
    """200 samples, 50 SNPs, 4 covariates."""
    return GeneralCase(covariate_lmm_inputs(n_cvt=4, seed=99))


@pytest.fixture
def fused_data():
    from jamma.lmm.uab import compute_uab_invariant_soa

    d = rotated_lmm_inputs(200, 50, eig_range=(0.1, 2.0), intercept=False)
    return (
        d.eigenvalues,
        d.UtW[:, 0].copy(),
        d.Uty,
        np.ascontiguousarray(d.UtG.T),
        compute_uab_invariant_soa(d.UtW, d.Uty, 1),
        d.n_samples,
    )


@pytest.fixture
def score_lrt_data():
    """Build gram_uab_batch() with its null-model Hi_eval and logl_H0.

    Computes the null-model MLE lambda via golden section on the null Uab
    (no genotype), then derives Hi_eval_null = 1/(lambda_null*eval + 1)
    and logl_H0 (null MLE log-likelihood).
    """
    eigenvalues, Uab_batch = gram_uab_batch()
    n_samples = eigenvalues.shape[0]
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
