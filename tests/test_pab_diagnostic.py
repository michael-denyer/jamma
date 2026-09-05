"""Numerical coverage extracted from the benchmark; invalid input stays visible."""

import numpy as np
import pytest

from jamma.lmm.uab import batch_compute_iab_numpy
from tests.conftest import requires_c
from tests.math_validation.dense_oracle import evaluate
from tests.math_validation.native_observer import native_wald
from tests.math_validation.pab_cases import (
    benchmark_inputs,
    gram_products,
    numpy_routes,
    reduced_inputs,
)
from tests.math_validation.pab_trace import diagnose, observe

pytestmark = pytest.mark.tier0


@pytest.mark.xfail(
    strict=True,
    reason="Historical benchmark violates shared w/y; --runxfail reproduces it",
)
def test_reduced_original_benchmark_parity():
    actual, expected = numpy_routes(*reduced_inputs())
    for field in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            actual[field],
            expected[field],
            rtol=1e-6,
            atol=1e-4,
            equal_nan=True,
            err_msg=field,
        )


def test_first_divergence_is_reused_invariant_columns():
    ev, w, x, y = reduced_inputs()
    (_, generic), calls, stages = observe(lambda: numpy_routes(ev, w, x, y))
    assert calls["golden_section_optimize_lambda_split_ncvt1_numpy"] == 1
    assert "compute_lmm_chunk_ncvt1_c" not in calls
    observed = stages["_compute_wald_numpy"][0]["uab_invariant_soa"]
    uab = gram_products(w, x, y)
    np.testing.assert_array_equal(observed, uab[0][:, [0, 2, 5]].T)
    assert not np.array_equal(observed, uab[1][:, [0, 2, 5]].T)
    hybrid = uab.copy()
    hybrid[:, :, [0, 2, 5]] = observed.T
    # Each original sample is a valid rank-one Gram matrix. Reusing only three
    # columns breaks this invariant before the first Pab recursion.
    for i in range(4):
        v = np.array([w[1, i], x[1, i], y[1, i]])
        assert np.linalg.eigvalsh(np.outer(v, v)).min() > -1e-12
    assert batch_compute_iab_numpy(1, hybrid)[1, 1, 3] < 0
    assert generic["lambdas"][1] > 99990


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
    actual, calls, _ = observe(lambda: native_wald(ev, w[0], x, y[0]))
    assert calls["compute_lmm_chunk_ncvt1_c"] == 1
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


@requires_c
def test_pab_diagnostic_observations_and_negative_controls():
    result = diagnose(include_original=False)
    assert result["status"] == "VERIFIED", result["failure_ids"]
    assert all(
        control["status"] == "VERIFIED"
        for control in result["observer_negative_controls"]
    )
