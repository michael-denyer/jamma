"""General native workspaces derive their packed layout from covariate count."""

import numpy as np
import pytest

from jamma.lmm import accel
from tests.conftest import _build_synthetic_covariate_data, requires_c
from tests.lmm_accel._helpers import _prepare_fused_general_data

pytestmark = [pytest.mark.tier0, requires_c]


def test_general_workspace_accepts_covariate_count():
    data = _prepare_fused_general_data(
        _build_synthetic_covariate_data(n_cvt=2, seed=42)
    )
    workspace = accel.require().create_workspace_general_c(
        data["eigenvalues"],
        data["uab_inv_soa"],
        data["UtW"],
        data["Uty"],
        data["n_samples"],
        1e-5,
        1e5,
        50,
        20,
        1,
        2,
        lmm_mode=1,
    )
    result = accel.require().compute_lmm_chunk_fused_general_c(
        workspace, data["utg_t"], 1
    )
    assert np.all(np.isfinite(result["betas"]))
    assert np.all((result["pwalds"] >= 0) & (result["pwalds"] <= 1))


@pytest.mark.parametrize("n_cvt", [0, 101])
def test_general_workspace_rejects_unsupported_covariate_count(n_cvt):
    data = _prepare_fused_general_data(
        _build_synthetic_covariate_data(n_cvt=2, seed=42)
    )
    with pytest.raises(ValueError, match=r"n_cvt must be 1\.\.100"):
        accel.require().create_workspace_general_c(
            data["eigenvalues"],
            data["uab_inv_soa"],
            data["UtW"],
            data["Uty"],
            data["n_samples"],
            1e-5,
            1e5,
            50,
            20,
            1,
            n_cvt,
            lmm_mode=1,
        )
