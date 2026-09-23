"""Independent dense checks for the shared-invariant n_cvt=1 Wald route."""

from __future__ import annotations

import numpy as np
import pytest

from jamma.lmm import accel
from jamma.lmm.compute_numpy import compute_wald_numpy
from jamma.lmm.uab import batch_compute_uab_numpy
from tests.math_validation.dense_oracle import evaluate
from tests.support import requires_c

pytestmark = pytest.mark.tier0


def _valid_shared_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260904)
    n_samples = 48
    eigenvalues = np.sort(rng.uniform(0.05, 1.5, n_samples))
    UtW = rng.normal(size=n_samples)
    Uty = rng.normal(size=n_samples)
    UtG = rng.normal(size=(n_samples, 4))
    return eigenvalues, UtW, Uty, UtG


def _assert_matches_oracle(
    result: dict[str, np.ndarray],
    eigenvalues: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> None:
    kinship = np.diag(eigenvalues)
    for snp, lambda_value in enumerate(result["lambdas"]):
        oracle = evaluate(kinship, UtW, UtG[:, snp], Uty, lambda_value)
        assert result["logls"][snp] == pytest.approx(
            oracle["reml"], rel=1e-10, abs=1e-12
        )
        assert result["betas"][snp] == pytest.approx(
            oracle["beta"], rel=1e-10, abs=1e-12
        )
        assert result["ses"][snp] == pytest.approx(oracle["se"], rel=1e-10, abs=1e-12)
        assert result["pwalds"][snp] == pytest.approx(
            oracle["p_wald"], rel=1e-10, abs=1e-12
        )


def test_numpy_wald_matches_independent_dense_oracle() -> None:
    eigenvalues, UtW, Uty, UtG = _valid_shared_case()
    n_samples = eigenvalues.size
    uab = batch_compute_uab_numpy(1, UtW[:, None], Uty, UtG.T)
    numpy_result = compute_wald_numpy(1, eigenvalues, uab, n_samples, 1e-5, 1e5, 50, 20)
    _assert_matches_oracle(numpy_result, eigenvalues, UtW, Uty, UtG)


@requires_c
def test_native_wald_matches_numpy_on_valid_shared_inputs() -> None:
    eigenvalues, UtW, Uty, UtG = _valid_shared_case()
    n_samples = eigenvalues.size
    uab = batch_compute_uab_numpy(1, UtW[:, None], Uty, UtG.T)
    numpy_result = compute_wald_numpy(1, eigenvalues, uab, n_samples, 1e-5, 1e5, 50, 20)

    invariant = np.stack((UtW * UtW, UtW * Uty, Uty * Uty))
    workspace = accel.require().create_workspace_c(
        eigenvalues,
        invariant,
        UtW[:, None],
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        1,
        lmm_mode=1,
    )
    native_result = accel.require().compute_lmm_chunk_c(
        workspace, np.ascontiguousarray(UtG.T), 1
    )

    for field in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            native_result[field], numpy_result[field], rtol=1e-6, atol=1e-12
        )


def test_dense_oracle_lambda_zero_matches_closed_form_ols() -> None:
    rng = np.random.default_rng(7)
    n_samples = 24
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    UtW = rng.normal(size=(n_samples, 3))
    Uty = rng.normal(size=n_samples)
    Utg = rng.normal(size=n_samples)

    design = np.column_stack((UtW, Utg))
    residual = Uty - design @ np.linalg.lstsq(design, Uty, rcond=None)[0]
    residual_ss = float(residual @ residual)
    df = n_samples - design.shape[1]
    _, logdet_design = np.linalg.slogdet(design.T @ design)
    expected_reml = 0.5 * df * (
        np.log(df) - np.log(2.0 * np.pi) - 1.0
    ) - 0.5 * df * np.log(residual_ss)
    expected_mle = 0.5 * n_samples * (
        np.log(n_samples) - np.log(2.0 * np.pi) - 1.0
    ) - 0.5 * n_samples * np.log(residual_ss)

    oracle = evaluate(np.diag(eigenvalues), UtW, Utg, Uty, 0.0)

    # At lambda zero the REML determinant ratio is log|X'X|-log|X'X| = 0.
    assert np.isfinite(logdet_design)
    assert oracle["reml"] == pytest.approx(expected_reml, rel=1e-12)
    assert oracle["mle"] == pytest.approx(expected_mle, rel=1e-12)
