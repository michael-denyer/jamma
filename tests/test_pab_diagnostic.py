"""Shared-input numerical parity through public native and NumPy entry points."""

import numpy as np
import pytest

from tests.conftest import requires_c
from tests.math_validation.dense_oracle import evaluate
from tests.math_validation.native_observer import native_wald
from tests.math_validation.pab_cases import (
    benchmark_inputs,
    numpy_routes,
    reduced_inputs,
)

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("shape", [(4, 2), (32, 7), (500, 2000)])
def test_shared_input_numpy_parity(shape):
    actual, expected = numpy_routes(*benchmark_inputs(*shape, shared=True))
    for field in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            actual[field],
            expected[field],
            rtol=1e-6,
            atol=1e-4,
            equal_nan=True,
            err_msg=field,
        )


@requires_c
@pytest.mark.parametrize("shape", [(4, 2), (32, 7), (500, 2000)])
def test_actual_native_shared_input_parity(shape):
    ev, w, x, y = benchmark_inputs(*shape, shared=True)
    _, expected = numpy_routes(ev, w, x, y)
    actual = native_wald(ev, w[0], x, y[0])
    for field in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            actual[field],
            expected[field],
            rtol=1e-6,
            atol=1e-4,
            equal_nan=True,
            err_msg=field,
        )


@requires_c
@pytest.mark.parametrize("snp", [0, 1])
def test_native_and_numpy_at_reported_lambda_match_dense_oracle(snp):
    ev, w, x, y = reduced_inputs()
    native = native_wald(ev, w[snp], x[snp : snp + 1], y[snp])
    _, generic = numpy_routes(ev, w, x, y)
    for result, index in ((native, 0), (generic, snp)):
        oracle = evaluate(
            np.diag(ev), w[snp, :, None], x[snp], y[snp], result["lambdas"][index]
        )
        for field, name in (
            ("betas", "beta"),
            ("ses", "se"),
            ("pwalds", "p_wald"),
            ("logls", "reml"),
        ):
            np.testing.assert_allclose(
                result[field][index], oracle[name], rtol=1e-6, atol=1e-4
            )


def test_three_samples_leave_a_flat_reml_profile():
    ev, w, x, y = reduced_inputs()
    values = [
        evaluate(np.diag(ev[:3]), w[0, :3, None], x[0, :3], y[0, :3], lam)["reml"]
        for lam in (1e-5, 1, 1e5)
    ]
    np.testing.assert_allclose(values, values[0], rtol=1e-10, atol=1e-12)
