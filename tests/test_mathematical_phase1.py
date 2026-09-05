"""Acceptance checks for Phase 1 likelihood and REML-boundary evidence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jamma.validation.tolerances import ToleranceConfig
from tests.builders import rotated_lmm_inputs
from tests.math_validation.phase1 import (
    MODE1_FIELDS,
    MODE4_ORACLE_FIELDS,
    boundary_trace,
    compare_phase1,
    mode4_evidence,
    objective_tolerance,
    phase1_evidence,
)

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("backend", ["numpy", "native"])
@pytest.mark.parametrize("n_cvt", [1, 2])
def test_modes_1_2_4_and_all_mode4_fields_have_independent_evidence(backend, n_cvt):
    evidence = mode4_evidence(backend=backend, n_cvt=n_cvt, seed=910 + n_cvt)
    assert evidence["mode2_logl_H1_emitted"] is True
    assert evidence["mode2_mode4_logl_H1_equal"] is True
    assert evidence["mode2_mode4_l_mle_equal"] is True
    assert evidence["mode2_mode4_p_lrt_equal"] is True
    assert evidence["small_gemma_mode4_logl_H1"] == (
        "absent_in_raw_gemma_0.98.5_layout"
    )
    tolerances = ToleranceConfig()
    field_rtol = {
        "beta": tolerances.beta_rtol,
        "se": tolerances.se_rtol,
        "logl_H1": tolerances.logl_rtol,
        "l_remle": tolerances.lambda_rtol,
        "l_mle": tolerances.lambda_rtol,
        "p_wald": tolerances.pvalue_rtol,
        "p_lrt": tolerances.p_lrt_rtol,
        "p_score": tolerances.pvalue_rtol,
    }
    for record in evidence["records"]:
        for field in MODE4_ORACLE_FIELDS:
            assert record["actual"][field] == pytest.approx(
                record["oracle"][field],
                rel=field_rtol[field],
                abs=1e-12,
            ), f"{backend}/{n_cvt}/{record['snp']}:{field}"
        for field in MODE1_FIELDS:
            assert record["mode1_oracle_field_checks"][field], (
                f"{backend}/{n_cvt}/{record['snp']}:mode1:{field}"
            )
    assert all(item["passed"] for item in evidence["named_detectors"].values())


def test_mode_specific_likelihoods_are_separated_beyond_print_rounding():
    evidence = mode4_evidence()
    for record in evidence["records"]:
        values = (record["mode1_reml_logl"], record["mode4_mle_logl"])
        # ``.6e`` has six digits after the decimal and seven significant digits.
        half_print_units = [
            0.5 * 10.0 ** (int(np.floor(np.log10(abs(value)))) - 6) for value in values
        ]
        assert record["mle_reml_separation"] > 100 * max(half_print_units)


def test_matching_lower_boundary_retains_distance_curve_and_objective_gap():
    data = rotated_lmm_inputs(20, 1, seed=1)
    trace = boundary_trace(
        np.diag(data.eigenvalues),
        data.UtW,
        data.UtG[:, 0],
        data.Uty,
        1e-5 * (1 + 1.5e-5),
        1e-5,
    )
    assert trace["classes"] == ["lower", "lower"]
    assert trace["passed"] is True
    assert trace["absolute_lambda_distance"] > 0
    assert trace["relative_lambda_distance"] > 0
    assert len(trace["grid"]["log_lambda"]) == 129
    assert len(trace["grid"]["reml"]) == 129
    assert trace["objective_losses"][0] <= trace["objective_tolerance"]
    json.dumps(trace, allow_nan=False)


def test_steep_valid_same_boundary_perturbation_fails_objective_policy():
    """A valid high-spectrum GLS case separates class from objective quality."""
    data = rotated_lmm_inputs(20, 1, seed=2, eig_range=(0.1, 100.0))
    trace = boundary_trace(
        np.diag(data.eigenvalues),
        data.UtW,
        data.UtG[:, 0],
        data.Uty,
        1e-5 * (1 + 1.9e-5),
        1e-5,
    )
    assert trace["classes"] == ["lower", "lower"]
    assert trace["objective_losses"][0] > trace["objective_tolerance"]
    assert trace["passed"] is False
    assert trace["detector"] == "matching-boundary-objective-degradation"


def test_objective_policy_constants_are_predeclared_not_fixture_fitted():
    assert objective_tolerance(1.0) == 1e-8
    large_scale = 1e9
    assert objective_tolerance(large_scale) == pytest.approx(
        128 * np.finfo(np.float64).eps * large_scale
    )


@pytest.mark.tier1
def test_callable_phase1_bundle_is_json_safe_and_native_is_required(math_evidence_dir):
    bundle = compare_phase1(math_evidence_dir)
    evidence = bundle["evidence"]
    backends = (
        ["numpy"] if bundle["environment"]["forced_numpy"] else ["numpy", "native"]
    )
    assert evidence["status"] == "VERIFIED"
    assert evidence["routes"] == {
        "declared": backends,
        "observed": backends,
    }
    assert {(item["backend"], item["n_cvt"]) for item in evidence["mode4"]} == {
        (backend, n_cvt) for backend in backends for n_cvt in (1, 2)
    }
    assert all(item["passed"] for item in evidence["boundary"])
    assert {item["snp"] for item in evidence["raw_gemma_boundary"]} == {
        "snp0",
        "snp2",
    }
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize("backends", [(), ("bogus",)])
def test_callable_evidence_rejects_missing_or_unknown_routes(backends):
    with pytest.raises(ValueError, match="nonempty subset"):
        phase1_evidence(backends=backends)
