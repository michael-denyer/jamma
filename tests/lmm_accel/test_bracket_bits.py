"""Regression test for the golden-section bracket the general workspace uses.

D1 found that the split LRT entry (``compute_lrt_split_general_c``, since
deleted by D2) computed ``log_l_min`` and ``step`` directly from
``l_min``/``l_max``, while the general workspace's fused compute
(``compute_lmm_chunk_fused_general_c``) re-derived the same two scalars from
``log(lambda_grid[0])`` and ``log(lambda_grid[-1])``, which is
``log(exp(log_l_min))`` and can differ from ``log_l_min`` by an ulp. That ulp
shifted the golden-section bracket endpoints and, on this fixture, 46 of 50
``lambdas_mle`` and 7 of 50 ``p_lrts`` differed between the two code paths
that were supposed to compute the same bracket. D1 stored the bracket in the
workspace at creation time instead of re-deriving it, fixing the drift.

D2 deleted the split entry and gave the general workspace's one compute a
standalone lmm_mode=2 (LRT-only), so the split-vs-workspace comparison this
file used to run live can no longer call both sides. The caller-level gate
for "did mode 2 inherit the same bracket the deleted split entry used to
compute" is the CI fingerprint job (``scripts/run-fingerprint.sh`` at the
merge-base and at the head, compared key for key): the deleted entry's keys
become base-only and are not compared, but every general-workspace key the
fingerprint records is bit-identical between the two sides, which is exactly
what a silent drift on deletion would break. A first version of this file
tried to pin that with a hardcoded literal array recorded on one machine;
golden-section MLE lambda is an argmin on a surface flat enough for
weak-signal SNPs that gcc-on-Linux and clang-on-macOS land up to 3.8e-5
relative apart, so a cross-platform hardcoded reference is not a valid gate
and broke CI. The test below stays same-run, same-platform: it compares two
computes from the one loaded extension, so no cross-compiler drift enters.
"""

import numpy as np
import pytest

from jamma.lmm import accel
from tests.conftest import _build_synthetic_covariate_data, requires_c
from tests.lmm_accel._helpers import (
    _fused_general_mode4_workspace,
    _make_general_score_lrt_data,
    _prepare_fused_general_data,
)

pytestmark = [pytest.mark.tier0, requires_c]


def _general_score_lrt_fixture():
    return _prepare_fused_general_data(
        _make_general_score_lrt_data(_build_synthetic_covariate_data(n_cvt=2, seed=42))
    )


def test_mode2_workspace_matches_mode4_workspace_bracket():
    """A standalone LRT workspace (lmm_mode=2) and the mode-4 workspace's LRT
    block compute the same bracket, so their outputs are byte-equal."""
    data = _general_score_lrt_fixture()
    n_samples = data["n_samples"]
    pab_dict = data["pab_c"]._asdict()

    ws4 = _fused_general_mode4_workspace(data, n_threads=1)
    result4 = accel.require().compute_lmm_chunk_fused_general_c(ws4, data["utg_t"], 1)

    ws2 = accel.require().create_workspace_general_c(
        data["eigenvalues"],
        data["uab_inv_soa"],
        data["UtW"],
        data["Uty"],
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        pab_dict,
        lmm_mode=2,
        logl_H0=data["logl_H0"],
    )
    result2 = accel.require().compute_lmm_chunk_fused_general_c(ws2, data["utg_t"], 1)

    np.testing.assert_array_equal(
        result2["lambdas_mle"],
        result4["lambdas_mle"],
        err_msg="lambdas_mle must be byte-equal between mode-2 and mode-4 workspaces",
    )
    np.testing.assert_array_equal(
        result2["p_lrts"],
        result4["p_lrts"],
        err_msg="p_lrts must be byte-equal between mode-2 and mode-4 workspaces",
    )
