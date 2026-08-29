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
file used to run live can no longer call both sides. ``_BASE_LAMBDAS_MLE``
and ``_BASE_P_LRTS`` are the split entry's output on this exact fixture,
recorded once at the merge-base (`df598f3`) before the deletion, with
``uv run python -c`` calling ``compute_lrt_split_general_c`` directly. The
general workspace's mode 2 must still reproduce them bit for bit: the
fingerprint recorder keys by entry point and cannot see a deleted entry's
callers move, so this fixed recording is the caller-level gate that proves
the general workspace inherited the split entry's numerics rather than
silently drifting when the split path went away.
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

# compute_lrt_split_general_c's lambdas_mle/p_lrts on the fixture below,
# recorded at df598f3 (D2's merge-base, the last commit before the split
# entry point was deleted).
_BASE_LAMBDAS_MLE = np.array(
    [
        4.8674762550977170e-02,
        3.1636263863440810e-02,
        1.0000347319647849e-05,
        1.4518669590942931e-01,
        3.0519479734518969e-02,
        3.8929645106874820e-02,
        1.0711629895165515e-02,
        4.2054216082287994e-02,
        4.3643388411604249e-02,
        1.9513121430620776e-02,
        2.5904761775752048e-02,
        1.9272605482734351e-02,
        2.8438006196077076e-02,
        6.1303593422025862e-02,
        4.5092813800369412e-02,
        6.0370347863809078e-02,
        3.1367967705062651e-02,
        1.2458665313585448e-02,
        5.0853147899013795e-02,
        1.7163518077465818e-02,
        4.0961340354389662e-02,
        4.2573124337633196e-02,
        9.9361872748033112e-02,
        5.5513149835130442e-02,
        1.7207427715968126e-02,
        5.4452416345898702e-02,
        4.4166486739891882e-02,
        4.7282287833335268e-02,
        4.0820466213795044e-02,
        3.5276835683190365e-02,
        4.2120676559554530e-02,
        8.6743536046489204e-03,
        3.8698245881209604e-02,
        6.6959649577850486e-02,
        3.9524257451820165e-02,
        3.9642250541090669e-02,
        6.2819465421247375e-02,
        1.6999804909902404e-02,
        4.0794884059192918e-02,
        4.2000525170915834e-02,
        4.5937793791672732e-02,
        3.6503200305765073e-02,
        6.1836524717059277e-02,
        5.1912034806351280e-02,
        4.1287991319183495e-02,
        4.3315516747705361e-02,
        5.6498739522702048e-02,
        4.4277778190892375e-02,
        1.0000155324577450e-05,
        2.1040485100264392e-02,
    ]
)

_BASE_P_LRTS = np.array(
    [
        0.632166767525889,
        0.5932173585221345,
        0.2587761300878967,
        0.23523748691977442,
        0.5004229046186532,
        0.7202017330618957,
        0.5068341171170879,
        0.9724377382447653,
        0.8624817490681823,
        0.5467770973018344,
        0.37508912804899125,
        0.46520184181587826,
        0.5911061364003247,
        0.032730554822917805,
        0.47551177765490477,
        0.5223121353800724,
        0.06744069333854517,
        0.37897787694129315,
        0.6221050973175796,
        0.2329282291867404,
        0.6154690471419315,
        0.8680335024698259,
        0.5454213839246456,
        0.6026124683522167,
        0.46839534618243617,
        0.6897212536564604,
        0.5497672643864608,
        0.8928500519438479,
        0.8002898795258363,
        0.8068748334920328,
        0.9008521915616312,
        0.1098054791189342,
        0.2481721366701657,
        0.33177681493945943,
        0.9542508031847553,
        0.20821473405897795,
        0.5050199600900748,
        0.1512788305928548,
        0.8980069335504054,
        0.9624215377659485,
        0.6547897779764192,
        0.059077704395812045,
        0.1842794694307016,
        0.16415913807203064,
        0.985034785414292,
        0.5182634165823625,
        0.030601828154617382,
        0.6166720947647029,
        0.05645976890766074,
        0.3900264185406169,
    ]
)


def _general_score_lrt_fixture():
    return _prepare_fused_general_data(
        _make_general_score_lrt_data(_build_synthetic_covariate_data(n_cvt=2, seed=42))
    )


def test_mode2_workspace_matches_recorded_split_path_bracket():
    """The standalone LRT workspace (lmm_mode=2) reproduces the deleted split
    entry point's output bit for bit, against the recording above."""
    data = _general_score_lrt_fixture()
    n_samples = data["n_samples"]
    pab_dict = data["pab_c"]._asdict()

    ws = _c().create_workspace_general_c(
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
    result = _c().compute_lmm_chunk_fused_general_c(ws, data["utg_t"], 1)

    np.testing.assert_array_equal(
        result["lambdas_mle"],
        _BASE_LAMBDAS_MLE,
        err_msg="lambdas_mle must be byte-equal to the recorded split-path output",
    )
    np.testing.assert_array_equal(
        result["p_lrts"],
        _BASE_P_LRTS,
        err_msg="p_lrts must be byte-equal to the recorded split-path output",
    )


def test_mode2_workspace_matches_mode4_workspace_bracket():
    """A standalone LRT workspace (lmm_mode=2) and the mode-4 workspace's LRT
    block compute the same bracket, so their outputs are byte-equal."""
    data = _general_score_lrt_fixture()
    n_samples = data["n_samples"]
    pab_dict = data["pab_c"]._asdict()

    ws4 = _fused_general_mode4_workspace(data, n_threads=1)
    result4 = _c().compute_lmm_chunk_fused_general_c(ws4, data["utg_t"], 1)

    ws2 = _c().create_workspace_general_c(
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
    result2 = _c().compute_lmm_chunk_fused_general_c(ws2, data["utg_t"], 1)

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
