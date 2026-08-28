"""Regression test for the golden-section bracket the general workspace uses.

The split LRT entry (``compute_lrt_split_general_c``) computes ``log_l_min``
and ``step`` directly from ``l_min``/``l_max``. The general workspace's fused
compute (``compute_lmm_chunk_fused_general_c``) used to re-derive the same
two scalars from ``log(lambda_grid[0])`` and ``log(lambda_grid[-1])``, which
is ``log(exp(log_l_min))`` and can differ from ``log_l_min`` by an ulp. That
ulp shifts the golden-section bracket endpoints and, on this fixture, 46 of
50 ``lambdas_mle`` and 7 of 50 ``p_lrts`` differ between the two code paths
that are supposed to compute the same bracket.

D1 stores the bracket in the workspace at creation time instead of
re-deriving it, so both paths read the same ``log_l_min``/``step`` and the
two code paths agree bit for bit.
"""

import numpy as np
import pytest

from jamma.lmm.compute_numpy import _c
from tests.conftest import _build_synthetic_covariate_data
from tests.lmm_accel._helpers import (
    _fused_general_mode4_workspace,
    _make_general_score_lrt_data,
    _prepare_fused_general_data,
)

pytestmark = pytest.mark.tier0


def test_lrt_split_matches_mode4_workspace_bracket():
    """lambdas_mle and p_lrts are byte-equal between split and workspace LRT paths."""
    data = _prepare_fused_general_data(
        _make_general_score_lrt_data(_build_synthetic_covariate_data(n_cvt=2, seed=42))
    )
    n_samples = data["n_samples"]
    pab_dict = data["pab_c"]._asdict()

    ws = _fused_general_mode4_workspace(data, n_threads=1)
    ws_result = _c().compute_lmm_chunk_fused_general_c(ws, data["utg_t"], 1)

    split_result = _c().compute_lrt_split_general_c(
        data["eigenvalues"],
        data["uab_var_soa"],
        data["uab_inv_soa"],
        n_samples,
        data["n_cvt"],
        pab_dict,
        1e-5,
        1e5,
        50,
        20,
        data["logl_H0"],
        1,
    )

    np.testing.assert_array_equal(
        split_result["lambdas_mle"],
        ws_result["lambdas_mle"],
        err_msg="lambdas_mle must be byte-equal between split and workspace LRT paths",
    )
    np.testing.assert_array_equal(
        split_result["p_lrts"],
        ws_result["p_lrts"],
        err_msg="p_lrts must be byte-equal between split and workspace LRT paths",
    )
