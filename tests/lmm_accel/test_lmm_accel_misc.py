"""_lmm_accel C extension tests: identity-Pab optimisation and n_cvt bounds.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import _compute_wald_numpy
from tests.lmm_accel._helpers import (
    _run_general_ncvt_c_vs_python,
    _score_general_c_available,
)


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_general_wald_identity_pab_optimization(synthetic_covariate_data_ncvt2):
    """C-GEN-OPT-01: logdet_from_row0 helper produces identical Wald results.

    Verifies the identity Pab prepass optimization (logdet_from_row0 helper)
    produces numerically identical results to the Python reference. This test
    specifically targets the logdet_iab computation path which flows through
    to REML log-likelihood and ultimately to Wald beta/SE/p-values.

    The C extension uses logdet_from_row0 to deduplicate the identity Pab
    prepass across compute_lmm_chunk_general_c, fused general Wald, and
    fused general mode-4. If the helper introduces any numerical divergence,
    it will show up in the Wald results compared to the Python reference.
    """

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]

    # Python reference (force fallback — no C extension)
    orig = compute_numpy._accel
    try:
        compute_numpy._accel = None
        result_py = _compute_wald_numpy(
            n_cvt,
            eigenvalues,
            Uab_batch,
            n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
        )
    finally:
        compute_numpy._accel = orig

    # C extension path (uses logdet_from_row0 helper internally)
    result_c = _compute_wald_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        n_threads=1,
    )

    # logdet_iab affects REML logl which flows to lambda, beta, SE, p-values.
    # Any divergence from the optimization would show up here.
    for key in ("lambdas", "logls", "betas", "ses"):
        np.testing.assert_allclose(
            result_c[key],
            result_py[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=(
                f"{key}: C (logdet_from_row0) vs Python mismatch — "
                f"identity Pab optimization may have diverged"
            ),
        )
    np.testing.assert_allclose(
        result_c["pwalds"],
        result_py["pwalds"],
        rtol=1e-6,
        atol=1e-14,
        equal_nan=True,
        err_msg=(
            "pwalds: C (logdet_from_row0) vs Python mismatch — "
            "identity Pab optimization may have diverged"
        ),
    )

    # Verify we actually tested SNPs (not an empty batch)
    valid = ~np.isnan(result_c["betas"])
    assert np.sum(valid) > 0, "No valid SNPs — test is vacuous"


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_ncvt_101_rejected_by_c_extension():
    """C extension raises ValueError for n_cvt=101 (exceeds MAX_N_CVT=100).

    Uses compute_score_batch_general_c as a representative entry point since
    it takes n_cvt as a direct parameter (not hidden in a workspace).
    """
    if not _score_general_c_available():
        pytest.skip("compute_score_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_score_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    n_cvt = 101
    n_samples = 200
    n_snps = 5

    rng = np.random.default_rng(777)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))[::-1]

    # Build minimal arrays — they won't be used because validation fails first.
    # Use n_cvt=100 for the pab table (build_pab_table_for_c is pure Python,
    # no limit), then pass n_cvt=101 to the C function to trigger the check.
    pab_table_dict = build_pab_table_for_c(100)
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    Uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    Hi_eval_null = np.ones(n_samples, dtype=np.float64)

    with pytest.raises(ValueError, match=r"n_cvt must be 1\.\.100, got 101"):
        compute_score_batch_general_c(
            eigenvalues,
            Uab_batch,
            Hi_eval_null,
            n_samples,
            n_cvt,
            pab_table_dict,
            1,  # n_threads
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
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


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
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
