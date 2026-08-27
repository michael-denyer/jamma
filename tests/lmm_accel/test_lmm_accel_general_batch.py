"""_lmm_accel C extension tests: Score and LRT kernels at n_cvt>1.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.

The kernel under test is the SoA-split general pair, which is what
``DispatchPath.SOA_SPLIT`` reaches for modes 2 and 3 at n_cvt>=2. These checks
previously drove ``compute_score_batch_general_c`` and
``compute_lrt_batch_general_c``, the full-Uab siblings, which no dispatch path
selects. The reference on each was already NumPy, and the two kernels agree with
each other to 1e-12, so moving the subject onto the live kernel needed no change
to any tolerance.
"""

import numpy as np
import pytest

from jamma.lmm import compute_numpy
from jamma.lmm._lmm_accel import (
    compute_lrt_split_general_c,
    compute_score_split_general_c,
)
from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns
from jamma.lmm.likelihood_numpy import golden_section_optimize_lambda_mle_numpy
from jamma.lmm.stats import _batch_lrt_pvalues_numpy, batch_calc_score_stats_numpy
from jamma.lmm.uab import batch_compute_uab_varying_soa_numpy, compute_uab_invariant_soa
from tests.lmm_accel._helpers import _make_general_score_lrt_data

_L_MIN, _L_MAX, _N_GRID, _N_REFINE = 1e-5, 1e5, 50, 20

# Score statistics are a closed-form evaluation at a fixed lambda, so C and
# NumPy differ only by accumulation order. Measured worst case 9.3e-13.
_SCORE_RTOL = 1e-10
# The MLE lambda is an argmin on a surface that is flat for weak-signal SNPs, so
# the two golden-section implementations land further apart than the value they
# are optimising: 3.8e-5 relative at n_cvt=2, while p_lrts still agrees to
# 1.3e-12. See CLAUDE.md on lambda_rtol.
_LRT_RTOL = 5e-5


@pytest.fixture
def general_score_lrt_ncvt4(synthetic_covariate_data_ncvt4):
    """Score/LRT data for n_cvt=4."""
    return _make_general_score_lrt_data(synthetic_covariate_data_ncvt4)


def _soa_inputs(data: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split the fixture's rotated inputs into the SoA form the kernels take."""
    n_cvt = data["n_cvt"]
    return (
        batch_compute_uab_varying_soa_numpy(
            n_cvt, data["UtW"], data["Uty"], data["UtG"].T
        ),
        compute_uab_invariant_soa(data["UtW"], data["Uty"], n_cvt=n_cvt),
        build_pab_table_for_c(n_cvt)._asdict(),
    )


def _degenerate(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Zero SNP 0's genotype columns in both the SoA and full-Uab views.

    Both sides have to see the same degenerate SNP, or the comparison is
    between two different problems rather than two implementations of one.
    """
    uab_var, uab_inv, pab = _soa_inputs(data)
    uab_var = uab_var.copy()
    uab_var[0] = 0.0
    Uab_batch = data["Uab_batch"].copy()
    _, var_indices = classify_uab_columns(data["n_cvt"])
    for idx in var_indices:
        Uab_batch[0, :, idx] = 0.0
    return uab_var, uab_inv, Uab_batch, pab


def _assert_close(got, ref, rtol, label):
    np.testing.assert_allclose(
        got,
        ref,
        rtol=rtol,
        atol=1e-14,
        equal_nan=True,
        err_msg=f"{label}: C vs NumPy mismatch",
    )


def _check_score(data, uab_var, uab_inv, pab, Uab_batch, label):
    result = compute_score_split_general_c(
        data["eigenvalues"],
        uab_var,
        uab_inv,
        data["Hi_eval_null"],
        data["n_samples"],
        data["n_cvt"],
        pab,
        1,
    )
    betas, ses, p_scores = batch_calc_score_stats_numpy(
        data["n_cvt"], data["Hi_eval_null"], Uab_batch, data["n_samples"]
    )
    for key, ref in (("betas", betas), ("ses", ses), ("p_scores", p_scores)):
        _assert_close(result[key], ref, _SCORE_RTOL, f"{label} {key}")
    return result


def _check_lrt(data, uab_var, uab_inv, pab, Uab_batch, label):
    result = compute_lrt_split_general_c(
        data["eigenvalues"],
        uab_var,
        uab_inv,
        data["n_samples"],
        data["n_cvt"],
        pab,
        _L_MIN,
        _L_MAX,
        _N_GRID,
        _N_REFINE,
        data["logl_H0"],
        1,
    )
    lambdas, logls = golden_section_optimize_lambda_mle_numpy(
        data["n_cvt"],
        data["eigenvalues"],
        Uab_batch,
        l_min=_L_MIN,
        l_max=_L_MAX,
        n_grid=_N_GRID,
        n_iter=_N_REFINE,
    )
    _assert_close(result["lambdas_mle"], lambdas, _LRT_RTOL, f"{label} lambdas_mle")
    _assert_close(
        result["p_lrts"],
        _batch_lrt_pvalues_numpy(logls, data["logl_H0"]),
        _LRT_RTOL,
        f"{label} p_lrts",
    )
    return result


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
@pytest.mark.parametrize("n_cvt", [2, 4])
def test_score_split_general_matches_numpy(n_cvt, request):
    """C-70-01: compute_score_split_general_c matches NumPy for n_cvt 2 and 4."""
    data = request.getfixturevalue(f"general_score_lrt_ncvt{n_cvt}")
    uab_var, uab_inv, pab = _soa_inputs(data)
    _check_score(data, uab_var, uab_inv, pab, data["Uab_batch"], f"n_cvt={n_cvt} Score")


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
@pytest.mark.parametrize("n_cvt", [2, 4])
def test_lrt_split_general_matches_numpy(n_cvt, request):
    """C-70-02: compute_lrt_split_general_c matches NumPy for n_cvt 2 and 4."""
    data = request.getfixturevalue(f"general_score_lrt_ncvt{n_cvt}")
    uab_var, uab_inv, pab = _soa_inputs(data)
    _check_lrt(data, uab_var, uab_inv, pab, data["Uab_batch"], f"n_cvt={n_cvt} LRT")


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_score_split_general_degenerate_snps(general_score_lrt_ncvt2):
    """C-70-03: a degenerate SNP produces NaN, and the rest stay finite."""
    data = general_score_lrt_ncvt2
    uab_var, uab_inv, Uab_batch, pab = _degenerate(data)
    result = _check_score(
        data, uab_var, uab_inv, pab, Uab_batch, "degenerate n_cvt=2 Score"
    )

    for key in ("betas", "ses", "p_scores"):
        assert np.isnan(result[key][0]), f"expected NaN {key} for the degenerate SNP"
    assert np.isfinite(result["betas"][1:]).any(), (
        "expected finite betas for the non-degenerate SNPs"
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_lrt_split_general_degenerate_snps(general_score_lrt_ncvt2):
    """C-70-04: a degenerate SNP carries no signal, so its p_lrt sits at 1."""
    data = general_score_lrt_ncvt2
    uab_var, uab_inv, Uab_batch, pab = _degenerate(data)
    result = _check_lrt(
        data, uab_var, uab_inv, pab, Uab_batch, "degenerate n_cvt=2 LRT"
    )

    assert result["p_lrts"][0] >= 0.99, (
        f"degenerate SNP should have p_lrt near 1, got {result['p_lrts'][0]}"
    )
    assert np.isfinite(result["p_lrts"][1:]).all(), (
        "non-degenerate SNPs should have finite p_lrts"
    )
