"""_lmm_accel C extension tests: Score and LRT kernels at n_cvt>1.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _C_GENERAL_AVAILABLE,
)
from jamma.lmm.likelihood_numpy import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    golden_section_optimize_lambda_mle_numpy,
)
from tests.lmm_accel._helpers import (
    _make_general_score_lrt_data,
    _score_general_c_available,
)


@pytest.fixture
def general_score_lrt_ncvt4(synthetic_covariate_data_ncvt4):
    """Score/LRT data for n_cvt=4."""
    return _make_general_score_lrt_data(synthetic_covariate_data_ncvt4)


def _lrt_general_c_available() -> bool:
    """Check if compute_lrt_batch_general_c is available from the C extension."""
    if not _C_ACCEL_AVAILABLE:
        return False
    try:
        from jamma.lmm._lmm_accel import compute_lrt_batch_general_c  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_score_batch_general_ncvt2(general_score_lrt_ncvt2):
    """C-70-01: compute_score_batch_general_c matches Python for n_cvt=2."""
    if not _score_general_c_available():
        pytest.skip("compute_score_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_score_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = general_score_lrt_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    Hi_eval_null = data["Hi_eval_null"]

    pab_table_dict = build_pab_table_for_c(n_cvt)

    # C path
    result_c = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,  # n_threads
    )

    # Python reference path
    betas_py, ses_py, p_scores_py = batch_calc_score_stats_numpy(
        n_cvt,
        Hi_eval_null,
        Uab_batch,
        n_samples,
    )

    np.testing.assert_allclose(
        result_c["betas"],
        betas_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: C vs Python mismatch for n_cvt=2 Score",
    )
    np.testing.assert_allclose(
        result_c["ses"],
        ses_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: C vs Python mismatch for n_cvt=2 Score",
    )
    np.testing.assert_allclose(
        result_c["p_scores"],
        p_scores_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: C vs Python mismatch for n_cvt=2 Score",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_score_batch_general_ncvt4(general_score_lrt_ncvt4):
    """C-70-02: compute_score_batch_general_c matches Python for n_cvt=4."""
    if not _score_general_c_available():
        pytest.skip("compute_score_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_score_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = general_score_lrt_ncvt4
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    Hi_eval_null = data["Hi_eval_null"]

    pab_table_dict = build_pab_table_for_c(n_cvt)

    result_c = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    betas_py, ses_py, p_scores_py = batch_calc_score_stats_numpy(
        n_cvt,
        Hi_eval_null,
        Uab_batch,
        n_samples,
    )

    np.testing.assert_allclose(
        result_c["betas"],
        betas_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: C vs Python mismatch for n_cvt=4 Score",
    )
    np.testing.assert_allclose(
        result_c["ses"],
        ses_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: C vs Python mismatch for n_cvt=4 Score",
    )
    np.testing.assert_allclose(
        result_c["p_scores"],
        p_scores_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: C vs Python mismatch for n_cvt=4 Score",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_score_batch_general_degenerate_snps(synthetic_covariate_data_ncvt2):
    """C-70-03: Degenerate SNPs produce NaN for Score general n_cvt."""
    if not _score_general_c_available():
        pytest.skip("compute_score_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_score_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    n_samples = data["n_samples"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"].copy()

    # Zero out all genotype-containing columns in one SNP to make it degenerate
    from jamma.lmm.likelihood import classify_uab_columns

    _, var_indices = classify_uab_columns(n_cvt)
    for idx in var_indices:
        Uab_batch[0, :, idx] = 0.0

    lambda_val = 1.0
    Hi_eval_null = 1.0 / (lambda_val * eigenvalues + 1.0)
    pab_table_dict = build_pab_table_for_c(n_cvt)

    result_c = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    # Degenerate SNP 0 should produce NaN
    assert np.isnan(result_c["betas"][0]), "Expected NaN beta for degenerate SNP"
    assert np.isnan(result_c["ses"][0]), "Expected NaN se for degenerate SNP"
    assert np.isnan(result_c["p_scores"][0]), "Expected NaN p_score for degenerate SNP"

    # Non-degenerate SNPs should have finite values
    finite_mask = np.isfinite(result_c["betas"][1:])
    assert finite_mask.sum() > 0, "Expected some finite betas for non-degenerate SNPs"


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_lrt_batch_general_ncvt2(general_score_lrt_ncvt2):
    """C-70-04: compute_lrt_batch_general_c matches Python for n_cvt=2."""
    if not _lrt_general_c_available():
        pytest.skip("compute_lrt_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_lrt_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = general_score_lrt_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    logl_H0 = data["logl_H0"]

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    pab_table_dict = build_pab_table_for_c(n_cvt)

    # C path
    result_c = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,  # n_threads
    )

    # Python reference path
    lambdas_py, logls_py = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts_py = _batch_lrt_pvalues_numpy(logls_py, logl_H0)

    np.testing.assert_allclose(
        result_c["lambdas_mle"],
        lambdas_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: C vs Python mismatch for n_cvt=2 LRT",
    )
    np.testing.assert_allclose(
        result_c["p_lrts"],
        p_lrts_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: C vs Python mismatch for n_cvt=2 LRT",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_lrt_batch_general_ncvt4(general_score_lrt_ncvt4):
    """C-70-05: compute_lrt_batch_general_c matches Python for n_cvt=4."""
    if not _lrt_general_c_available():
        pytest.skip("compute_lrt_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_lrt_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = general_score_lrt_ncvt4
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    logl_H0 = data["logl_H0"]

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    pab_table_dict = build_pab_table_for_c(n_cvt)

    result_c = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    lambdas_py, logls_py = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts_py = _batch_lrt_pvalues_numpy(logls_py, logl_H0)

    np.testing.assert_allclose(
        result_c["lambdas_mle"],
        lambdas_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: C vs Python mismatch for n_cvt=4 LRT",
    )
    np.testing.assert_allclose(
        result_c["p_lrts"],
        p_lrts_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: C vs Python mismatch for n_cvt=4 LRT",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_lrt_batch_general_degenerate_snps(synthetic_covariate_data_ncvt2):
    """C-70-06: LRT general C matches Python on degenerate SNPs.

    Unlike Score (which checks P_XX <= 0 and returns NaN), LRT's golden
    section MLE optimizer can still converge to a finite lambda on degenerate
    SNPs. This test verifies C-vs-Python parity on a batch containing a
    degenerate SNP.
    """
    if not _lrt_general_c_available():
        pytest.skip("compute_lrt_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_lrt_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    n_samples = data["n_samples"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"].copy()

    # Zero out all genotype-containing columns in one SNP to make it degenerate
    _, var_indices = classify_uab_columns(n_cvt)
    for idx in var_indices:
        Uab_batch[0, :, idx] = 0.0

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    logl_H0 = -100.0  # arbitrary finite value for null model
    pab_table_dict = build_pab_table_for_c(n_cvt)

    result_c = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,  # n_threads
    )

    # Python reference path
    lambdas_py, logls_py = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts_py = _batch_lrt_pvalues_numpy(logls_py, logl_H0)

    np.testing.assert_allclose(
        result_c["lambdas_mle"],
        lambdas_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: C vs Python mismatch on batch with degenerate SNP",
    )
    np.testing.assert_allclose(
        result_c["p_lrts"],
        p_lrts_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: C vs Python mismatch on batch with degenerate SNP",
    )


def _score_split_general_c_available() -> bool:
    """Check if compute_score_split_general_c is available from the C extension."""
    if not _C_ACCEL_AVAILABLE:
        return False
    try:
        from jamma.lmm._lmm_accel import compute_score_split_general_c  # noqa: F401

        return True
    except ImportError:
        return False


def _lrt_split_general_c_available() -> bool:
    """Check if compute_lrt_split_general_c is available from the C extension."""
    if not _C_ACCEL_AVAILABLE:
        return False
    try:
        from jamma.lmm._lmm_accel import compute_lrt_split_general_c  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_score_split_ncvt2(general_score_lrt_ncvt2):
    """C-105-01: compute_score_split_general_c matches reconstruct+batch for n_cvt=2."""
    if not _score_split_general_c_available():
        pytest.skip("compute_score_split_general_c not compiled yet")

    from jamma.lmm._lmm_accel import (
        compute_score_batch_general_c,
        compute_score_split_general_c,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c
    from jamma.lmm.likelihood_numpy import (
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
    )

    data = general_score_lrt_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    Hi_eval_null = data["Hi_eval_null"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    pab_table_dict = build_pab_table_for_c(n_cvt)

    # Reference: reconstruct + batch general C
    ref = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    # SoA split path
    uab_inv = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    uab_var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    result = compute_score_split_general_c(
        eigenvalues,
        uab_var,
        uab_inv,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    # SoA split accumulates dot products column-by-column (outer loop=column,
    # inner=samples), while batch general accumulates row-by-row (outer=samples,
    # inner=columns). Different FP accumulation order gives machine-epsilon
    # differences for n_cvt>=2. Use tight allclose instead of bitwise equality.
    np.testing.assert_allclose(
        result["betas"],
        ref["betas"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: split general vs batch general mismatch for n_cvt=2",
    )
    np.testing.assert_allclose(
        result["ses"],
        ref["ses"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: split general vs batch general mismatch for n_cvt=2",
    )
    np.testing.assert_allclose(
        result["p_scores"],
        ref["p_scores"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: split general vs batch general mismatch for n_cvt=2",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_lrt_split_ncvt2(general_score_lrt_ncvt2):
    """C-105-02: compute_lrt_split_general_c matches reconstruct+batch for n_cvt=2."""
    if not _lrt_split_general_c_available():
        pytest.skip("compute_lrt_split_general_c not compiled yet")

    from jamma.lmm._lmm_accel import (
        compute_lrt_batch_general_c,
        compute_lrt_split_general_c,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c
    from jamma.lmm.likelihood_numpy import (
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
    )

    data = general_score_lrt_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    logl_H0 = data["logl_H0"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    pab_table_dict = build_pab_table_for_c(n_cvt)

    # Reference: reconstruct + batch general C
    ref = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    # SoA split path
    uab_inv = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    uab_var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    result = compute_lrt_split_general_c(
        eigenvalues,
        uab_var,
        uab_inv,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    np.testing.assert_allclose(
        result["lambdas_mle"],
        ref["lambdas_mle"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: split general vs batch general mismatch for n_cvt=2",
    )
    np.testing.assert_allclose(
        result["p_lrts"],
        ref["p_lrts"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: split general vs batch general mismatch for n_cvt=2",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_score_split_ncvt4(general_score_lrt_ncvt4):
    """C-105-03: compute_score_split_general_c matches reconstruct+batch for n_cvt=4."""
    if not _score_split_general_c_available():
        pytest.skip("compute_score_split_general_c not compiled yet")

    from jamma.lmm._lmm_accel import (
        compute_score_batch_general_c,
        compute_score_split_general_c,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c
    from jamma.lmm.likelihood_numpy import (
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
    )

    data = general_score_lrt_ncvt4
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    Hi_eval_null = data["Hi_eval_null"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    pab_table_dict = build_pab_table_for_c(n_cvt)

    ref = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    uab_inv = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    uab_var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    result = compute_score_split_general_c(
        eigenvalues,
        uab_var,
        uab_inv,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    np.testing.assert_allclose(
        result["betas"],
        ref["betas"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: split general vs batch general mismatch for n_cvt=4",
    )
    np.testing.assert_allclose(
        result["ses"],
        ref["ses"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: split general vs batch general mismatch for n_cvt=4",
    )
    np.testing.assert_allclose(
        result["p_scores"],
        ref["p_scores"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: split general vs batch general mismatch for n_cvt=4",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_lrt_split_ncvt4(general_score_lrt_ncvt4):
    """C-105-04: compute_lrt_split_general_c matches reconstruct+batch for n_cvt=4."""
    if not _lrt_split_general_c_available():
        pytest.skip("compute_lrt_split_general_c not compiled yet")

    from jamma.lmm._lmm_accel import (
        compute_lrt_batch_general_c,
        compute_lrt_split_general_c,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c
    from jamma.lmm.likelihood_numpy import (
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
    )

    data = general_score_lrt_ncvt4
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    logl_H0 = data["logl_H0"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    pab_table_dict = build_pab_table_for_c(n_cvt)

    ref = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    uab_inv = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    uab_var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    result = compute_lrt_split_general_c(
        eigenvalues,
        uab_var,
        uab_inv,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    np.testing.assert_allclose(
        result["lambdas_mle"],
        ref["lambdas_mle"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: split general vs batch general mismatch for n_cvt=4",
    )
    np.testing.assert_allclose(
        result["p_lrts"],
        ref["p_lrts"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: split general vs batch general mismatch for n_cvt=4",
    )
