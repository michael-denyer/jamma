"""_lmm_accel C extension tests for the identity-Pab optimisation.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from tests.conftest import requires_c
from tests.lmm_accel._helpers import (
    _fused_general_wald,
    _run_general_ncvt_c_vs_python,
)

pytestmark = pytest.mark.tier0


@requires_c
def test_general_wald_identity_pab_optimization(synthetic_covariate_data_ncvt2):
    """C-GEN-OPT-01: logdet_from_row0 helper produces identical Wald results.

    The C extension uses logdet_from_row0 to deduplicate the identity Pab
    prepass across the fused general Wald and fused general mode-4 kernels. If
    the helper introduces any numerical divergence, it shows up in the Wald
    results compared against the NumPy reference, because logdet_iab feeds the
    REML log-likelihood and from there lambda, beta, SE and the p-values.
    """
    _run_general_ncvt_c_vs_python(synthetic_covariate_data_ncvt2)

    betas = _fused_general_wald(synthetic_covariate_data_ncvt2)["betas"]
    assert np.sum(~np.isnan(betas)) > 0, "No valid SNPs, so the test is vacuous"


@requires_c
def test_general_ncvt_reml_wald_ncvt20():
    """C extension Wald matches Python for n_cvt=20 (previous MAX_N_CVT limit).

    Verifies that n_cvt=20 — the old limit before MAX_N_CVT was raised to
    100 — works correctly through the full REML+CalcPab+Wald pipeline.
    Uses small matrices (150 samples, 15 SNPs) to keep execution fast.
    """
    from tests.conftest import _build_synthetic_covariate_data

    data = _build_synthetic_covariate_data(
        n_cvt=20, n_samples=150, n_snps=15, seed=2020
    )
    _run_general_ncvt_c_vs_python(data)


@requires_c
def test_general_ncvt_reml_wald_ncvt50():
    """C extension Wald matches Python for n_cvt=50 (well beyond old limit).

    Exercises n_cvt=50, which requires Pab tables with n_index=1431 and
    n_rows=52. Validates that the raised MAX_N_CVT=100 works at a midpoint.
    Uses small matrices (150 samples, 10 SNPs) to keep execution fast.
    """
    from tests.conftest import _build_synthetic_covariate_data

    data = _build_synthetic_covariate_data(
        n_cvt=50, n_samples=150, n_snps=10, seed=5050
    )
    _run_general_ncvt_c_vs_python(data)
