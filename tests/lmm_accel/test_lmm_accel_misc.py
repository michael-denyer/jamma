"""_lmm_accel C extension tests: identity-Pab optimisation and n_cvt bounds.

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
def test_ncvt_101_rejected_by_c_extension():
    """C extension raises ValueError for n_cvt=101 (exceeds MAX_N_CVT=100).

    Uses create_workspace_general_c as the representative entry point: it
    parses the Pab table, which carries n_cvt, before anything else.
    """
    from jamma.lmm._lmm_accel import create_workspace_general_c
    from jamma.lmm.pab import build_pab_table_for_c

    n_cvt = 101
    n_samples = 200

    rng = np.random.default_rng(777)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))[::-1]
    pab_table = build_pab_table_for_c(n_cvt)._asdict()

    with pytest.raises(ValueError, match=r"n_cvt must be 1\.\.100, got 101"):
        create_workspace_general_c(
            eigenvalues,
            np.zeros((pab_table["n_inv"], n_samples), dtype=np.float64),
            np.zeros((n_samples, n_cvt), dtype=np.float64),
            np.zeros(n_samples, dtype=np.float64),
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
            pab_table,
            lmm_mode=1,
        )


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
