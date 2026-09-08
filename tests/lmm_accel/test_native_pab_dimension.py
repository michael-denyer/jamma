"""General native workspaces derive their packed layout from covariate count."""

import pytest

from jamma.lmm import accel, compute_numpy
from tests.conftest import requires_c
from tests.lmm_accel._helpers import _prepare_fused_general_data

pytestmark = [pytest.mark.tier0, requires_c]


@pytest.mark.parametrize("n_cvt", [0, compute_numpy.MAX_C_N_CVT + 1])
def test_general_workspace_rejects_unsupported_covariate_count(
    n_cvt, synthetic_covariate_data_ncvt2
):
    """Nothing in Python bounds n_cvt, so the kernel's own guard is the enforcement.

    The creator reads the covariate count before any array, so a
    two-covariate fixture reaches the guard whatever count it is handed.
    """
    data = _prepare_fused_general_data(synthetic_covariate_data_ncvt2)
    with pytest.raises(ValueError, match=rf"n_cvt must be 1\.\.100, got {n_cvt}"):
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
