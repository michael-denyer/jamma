"""Independent checks for the mode-specific meaning of ``logl_H1``."""

from __future__ import annotations

import numpy as np
import pytest

from jamma.lmm import accel
from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy
from jamma.lmm.likelihood import compute_null_model_mle
from jamma.lmm.pab import build_pab_table_for_c
from jamma.lmm.schema import LmmMode
from jamma.lmm.uab import batch_compute_uab_numpy, compute_uab_invariant_soa
from tests.builders import rotated_lmm_inputs
from tests.independent_lmm_oracle import dense_lmm_log_likelihood

pytestmark = pytest.mark.tier0


def _compute(data, mode: LmmMode, backend: str):
    uab = batch_compute_uab_numpy(
        data.n_cvt, data.UtW, data.Uty, np.ascontiguousarray(data.UtG.T)
    )
    lambda_null, logl_H0 = compute_null_model_mle(
        data.eigenvalues, data.UtW, data.Uty, data.n_cvt
    )
    hi_eval_null = 1.0 / (lambda_null * data.eigenvalues + 1.0)
    if backend == "numpy":
        return compute_lmm_chunk_numpy(
            mode,
            data.n_cvt,
            data.eigenvalues,
            uab,
            data.n_samples,
            Hi_eval_null=hi_eval_null,
            logl_H0=logl_H0,
        )

    if not accel.available():
        pytest.skip("C accelerator is unavailable")
    invariant = compute_uab_invariant_soa(data.UtW, data.Uty, n_cvt=data.n_cvt)
    if data.n_cvt == 1:
        workspace = accel.require().create_workspace_ncvt1_c(
            data.eigenvalues,
            invariant,
            data.UtW[:, 0],
            data.Uty,
            data.n_samples,
            1e-5,
            1e5,
            50,
            20,
            lmm_mode=mode,
            **({"hi_eval_null": hi_eval_null, "logl_H0": logl_H0} if mode == 4 else {}),
        )
        return accel.require().compute_lmm_chunk_ncvt1_c(
            workspace, np.ascontiguousarray(data.UtG.T), 1
        )

    workspace = accel.require().create_workspace_general_c(
        data.eigenvalues,
        invariant,
        data.UtW,
        data.Uty,
        data.n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        build_pab_table_for_c(data.n_cvt)._asdict(),
        lmm_mode=mode,
        **({"hi_eval_null": hi_eval_null, "logl_H0": logl_H0} if mode == 4 else {}),
    )
    return accel.require().compute_lmm_chunk_fused_general_c(
        workspace, np.ascontiguousarray(data.UtG.T), 1
    )


@pytest.mark.parametrize("backend", ["numpy", "c"])
@pytest.mark.parametrize("n_cvt", [1, 2])
def test_logl_h1_uses_reml_in_mode1_and_mle_in_mode4(backend, n_cvt):
    """Every backend evaluates the likelihood named by GEMMA's mode contract."""
    data = rotated_lmm_inputs(40, 6, n_cvt=n_cvt, seed=910 + n_cvt)
    wald = _compute(data, 1, backend)
    all_tests = _compute(data, 4, backend)

    assert wald["lambdas"] is not None
    assert wald["logls"] is not None
    assert all_tests["lambdas_mle"] is not None
    assert all_tests["logls"] is not None

    expected_reml = np.array(
        [
            dense_lmm_log_likelihood(
                data.eigenvalues,
                data.UtW,
                data.Uty,
                data.UtG[:, snp],
                wald["lambdas"][snp],
                restricted=True,
            )
            for snp in range(data.n_snps)
        ]
    )
    expected_mle = np.array(
        [
            dense_lmm_log_likelihood(
                data.eigenvalues,
                data.UtW,
                data.Uty,
                data.UtG[:, snp],
                all_tests["lambdas_mle"][snp],
                restricted=False,
            )
            for snp in range(data.n_snps)
        ]
    )
    np.testing.assert_allclose(wald["logls"], expected_reml, rtol=2e-13)
    np.testing.assert_allclose(all_tests["logls"], expected_mle, rtol=2e-13)
