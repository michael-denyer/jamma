"""_lmm_accel C extension tests: batch Score and LRT kernels at n_cvt=1.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _compute_lrt_batch_c,
    _compute_score_batch_c,
)
from jamma.lmm.likelihood_numpy import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    golden_section_optimize_lambda_mle_numpy,
)

_score_c_available = _C_ACCEL_AVAILABLE and _compute_score_batch_c is not None

_lrt_c_available = _C_ACCEL_AVAILABLE and _compute_lrt_batch_c is not None


@pytest.mark.tier0
@pytest.mark.skipif(not _score_c_available, reason="Score C extension not available")
def test_score_c_vs_python_parity(score_lrt_data):
    """C compute_score_batch_c matches Python batch_calc_score_stats_numpy."""
    eigenvalues, Uab_batch, n_samples, Hi_eval_null, _ = score_lrt_data
    n_cvt = 1

    # C path
    result_c = _compute_score_batch_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        1,
    )

    # Python path
    betas_py, ses_py, p_scores_py = batch_calc_score_stats_numpy(
        n_cvt,
        Hi_eval_null,
        Uab_batch,
        n_samples,
    )

    np.testing.assert_allclose(result_c["betas"], betas_py, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(result_c["ses"], ses_py, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(
        result_c["p_scores"], p_scores_py, rtol=1e-10, atol=1e-14
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _lrt_c_available, reason="LRT C extension not available")
def test_lrt_c_vs_python_parity(score_lrt_data):
    """C compute_lrt_batch_c matches Python golden_section_optimize_lambda_mle_numpy."""
    eigenvalues, Uab_batch, n_samples, _, logl_H0 = score_lrt_data
    n_cvt = 1
    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20

    # C path
    result_c = _compute_lrt_batch_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    # Python path
    lambdas_mle_py, logls_mle_py = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts_py = _batch_lrt_pvalues_numpy(logls_mle_py, logl_H0)

    np.testing.assert_allclose(
        result_c["lambdas_mle"],
        lambdas_mle_py,
        rtol=1e-6,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result_c["p_lrts"],
        p_lrts_py,
        rtol=1e-4,
        atol=1e-14,
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _score_c_available, reason="Score C extension not available")
def test_score_c_degenerate_snps(score_lrt_data):
    """Score C returns NaN for constant genotypes (P_xx <= 0)."""
    eigenvalues, Uab_batch, n_samples, Hi_eval_null, _ = score_lrt_data

    # Create degenerate Uab: constant genotype -> wx=0, xx=0, xy=0
    Uab_degen = Uab_batch.copy()
    Uab_degen[0, :, 1] = 0.0  # wx = 0
    Uab_degen[0, :, 3] = 0.0  # xx = 0
    Uab_degen[0, :, 4] = 0.0  # xy = 0

    result = _compute_score_batch_c(
        eigenvalues,
        Uab_degen,
        Hi_eval_null,
        n_samples,
        1,
    )

    # First SNP is degenerate: should have NaN beta/se/p_score
    assert np.isnan(result["betas"][0]), "degenerate SNP should have NaN beta"
    assert np.isnan(result["ses"][0]), "degenerate SNP should have NaN se"
    assert np.isnan(result["p_scores"][0]), "degenerate SNP should have NaN p_score"

    # Remaining SNPs should still be finite
    assert np.all(np.isfinite(result["betas"][1:])), (
        "non-degenerate SNPs should be finite"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _lrt_c_available, reason="LRT C extension not available")
def test_lrt_c_degenerate_snps(score_lrt_data):
    """LRT C handles degenerate SNPs: p_lrt ~ 1.0 (no signal)."""
    eigenvalues, Uab_batch, n_samples, _, logl_H0 = score_lrt_data

    # Create degenerate Uab: constant genotype
    Uab_degen = Uab_batch.copy()
    Uab_degen[0, :, 1] = 0.0  # wx = 0
    Uab_degen[0, :, 3] = 0.0  # xx = 0
    Uab_degen[0, :, 4] = 0.0  # xy = 0

    result = _compute_lrt_batch_c(
        eigenvalues,
        Uab_degen,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        1,
    )

    # Degenerate SNP: MLE logl_H1 ~ logl_H0, so LRT stat ~ 0, p ~ 1.0
    assert result["p_lrts"][0] >= 0.99, (
        f"degenerate SNP should have p_lrt ~ 1.0, got {result['p_lrts'][0]}"
    )

    # Remaining SNPs should be finite
    assert np.all(np.isfinite(result["p_lrts"][1:])), (
        "non-degenerate SNPs should be finite"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _score_c_available, reason="Score C extension not available")
def test_score_c_multithreaded(score_lrt_data):
    """Score C with n_threads=4 produces identical output to n_threads=1."""
    eigenvalues, Uab_batch, n_samples, Hi_eval_null, _ = score_lrt_data

    result_1t = _compute_score_batch_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        1,
    )
    result_4t = _compute_score_batch_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        4,
    )

    np.testing.assert_array_equal(result_1t["betas"], result_4t["betas"])
    np.testing.assert_array_equal(result_1t["ses"], result_4t["ses"])
    np.testing.assert_array_equal(result_1t["p_scores"], result_4t["p_scores"])


@pytest.mark.tier0
@pytest.mark.skipif(not _lrt_c_available, reason="LRT C extension not available")
def test_lrt_c_multithreaded(score_lrt_data):
    """LRT C with n_threads=4 produces identical output to n_threads=1."""
    eigenvalues, Uab_batch, n_samples, _, logl_H0 = score_lrt_data

    result_1t = _compute_lrt_batch_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        1,
    )
    result_4t = _compute_lrt_batch_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        4,
    )

    np.testing.assert_array_equal(result_1t["lambdas_mle"], result_4t["lambdas_mle"])
    np.testing.assert_array_equal(result_1t["p_lrts"], result_4t["p_lrts"])
