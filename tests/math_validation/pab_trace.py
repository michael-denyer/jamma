"""A rerunnable route and intermediate-value diagnostic for the Pab report."""

import copy
import sys
from collections import Counter
from typing import Any

import numpy as np

from tests.math_validation.dense_oracle import evaluate, optimize, projection_products
from tests.math_validation.native_observer import (
    native_grid_bracket,
    native_objectives,
    native_wald,
)
from tests.math_validation.pab_cases import (
    benchmark_inputs,
    gram_products,
    numpy_routes,
    reduced_inputs,
)


def observe(call):
    """Trace live calls and returned NumPy locals; never replace computation."""
    calls = Counter()
    stages = {}
    wanted = {
        "_compute_wald_numpy",
        "golden_section_optimize_lambda_split_ncvt1_numpy",
        "golden_section_optimize_lambda_numpy",
        "_batch_golden_section_bracket_numpy",
    }

    def profiler(frame, event, arg):
        if event == "call" and "/src/jamma/" in frame.f_code.co_filename:
            calls[frame.f_code.co_name] += 1
        elif (
            event == "c_call"
            and getattr(arg, "__module__", "") == "jamma.lmm._lmm_accel"
        ):
            calls[arg.__name__] += 1

    def tracer(frame, event, arg):
        if event == "return" and frame.f_code.co_name in wanted:
            values = {
                k: v.copy()
                for k, v in frame.f_locals.items()
                if isinstance(v, np.ndarray)
            }
            stages.setdefault(frame.f_code.co_name, []).append(values)
        return tracer

    old_profile, old_trace = sys.getprofile(), sys.gettrace()
    try:
        sys.setprofile(profiler)
        sys.settrace(tracer)
        result = call()
    finally:
        sys.setprofile(old_profile)
        sys.settrace(old_trace)
    return result, dict(calls), stages


def summarize(a, b):
    return {
        key: {
            "max_abs": float(np.nanmax(np.abs(a[key] - b[key]))),
            "failed_ids": np.flatnonzero(
                ~np.isclose(a[key], b[key], rtol=1e-6, atol=1e-4, equal_nan=True)
            ).tolist(),
            "actual_nan_ids": np.flatnonzero(np.isnan(a[key])).tolist(),
            "reference_nan_ids": np.flatnonzero(np.isnan(b[key])).tolist(),
        }
        for key in b
    }


def diagnose(*, include_original=True) -> dict[str, Any]:
    ev, w, x, y = reduced_inputs()
    (split, generic), calls, stages = observe(lambda: numpy_routes(ev, w, x, y))
    grid = np.exp(np.linspace(np.log(1e-5), np.log(1e5), 50))
    from jamma.lmm.likelihood_numpy import _batch_grid_pab_numpy

    uab = gram_products(w, x, y)
    numpy_pab, _, _ = _batch_grid_pab_numpy(1, grid, ev, uab)
    hybrid = uab.copy()
    hybrid[:, :, [0, 2, 5]] = uab[0, :, [0, 2, 5]].T
    records = []
    for j in range(2):
        vectors = np.column_stack((w[j], x[j], y[j]))
        native, native_calls, _ = observe(
            lambda j=j: native_wald(ev, w[j], x[j : j + 1], y[j])
        )
        c_obj, c_pab, c_iab = native_objectives(ev, uab[j], grid)
        h_obj, h_pab, h_iab = native_objectives(ev, hybrid[j], grid)
        dense = np.array(
            [projection_products(np.diag(ev), vectors, lam) for lam in grid]
        )
        oracle = optimize(np.diag(ev), w[j, :, None], x[j], y[j])
        oracle_grid = [
            evaluate(np.diag(ev), w[j, :, None], x[j], y[j], lam) for lam in grid
        ]
        best, c_grid = native_grid_bracket(ev, uab[j])
        bracket = np.log(c_grid[[max(0, best - 1), min(49, best + 1)]])
        records.append(
            {
                "snp_id": j,
                "native": native,
                "native_calls": native_calls,
                "oracle": oracle,
                "uab": uab[j],
                "hybrid_uab": hybrid[j],
                "native_iab": c_iab,
                "hybrid_iab": h_iab,
                "native_pab": c_pab,
                "numpy_pab": numpy_pab[:, j],
                "hybrid_pab": h_pab,
                "dense_projection_products": dense,
                "native_grid_objectives": c_obj,
                "hybrid_grid_objectives": h_obj,
                "oracle_grid": oracle_grid,
                "native_bracket_log": bracket,
                "native_selected_grid_index": best,
                "native_refinement_lambdas": [
                    float(
                        native_wald(ev, w[j], x[j : j + 1], y[j], n_refine=n)[
                            "lambdas"
                        ][0]
                    )
                    for n in range(1, 21)
                ],
                "native_final_log_interval": None,
                "interval_method": (
                    "Not exposed by native API. Returned lambdas include "
                    "safeguarded REML score refinement, so golden-section width "
                    "does not identify a final interval."
                ),
            }
        )
    valid_w, valid_y = np.broadcast_to(w[0], w.shape), np.broadcast_to(y[0], y.shape)
    (valid_split, valid_generic), valid_calls, valid_stages = observe(
        lambda: numpy_routes(ev, valid_w, x, valid_y)
    )
    valid_native, native_calls, _ = observe(lambda: native_wald(ev, w[0], x, y[0]))
    report = {
        "status": "INCONCLUSIVE",
        "verdict": "Invalid benchmark shared-input contract; mislabeled NumPy route",
        "first_divergence": (
            "_compute_wald_numpy selects SNP 0 ww/wy/yy for all SNPs before Iab/Pab"
        ),
        "original_assertion_status": "NOT VERIFIED",
        "production_correction_required": False,
        "reduction": {
            "original_shape": [500, 2000, 1],
            "shape": [4, 2, 1],
            "sample_indices": [0, 1, 498, 499],
            "snp_indices": [0, 1],
            "minimum_reason": (
                "One SNP cannot violate cross-SNP sharing; n<=3 "
                "leaves at most one residual contrast and flat REML."
            ),
        },
        "eigenvalues": ev,
        "w": w,
        "x": x,
        "y": y,
        "grid": grid,
        "calls": calls,
        "stages": stages,
        "split": split,
        "generic": generic,
        "records": records,
        "reduced_errors": summarize(split, generic),
        "shared_control": {
            "split": valid_split,
            "generic": valid_generic,
            "native": valid_native,
            "oracle": [optimize(np.diag(ev), w[0, :, None], xx, y[0]) for xx in x],
            "calls": valid_calls,
            "native_calls": native_calls,
            "stages": valid_stages,
            "numpy_errors": summarize(valid_split, valid_generic),
            "native_errors": summarize(valid_native, valid_generic),
        },
    }
    if include_original:
        original = benchmark_inputs()
        a, b = numpy_routes(*original)
        report["original_errors"] = summarize(a, b)
        report["original_outputs"] = {"split": a, "generic": b}
        shared = benchmark_inputs(shared=True)
        a, b = numpy_routes(*shared)
        c = native_wald(shared[0], shared[1][0], shared[2], shared[3][0])
        report["shared_2000_outputs"] = {"split": a, "generic": b, "native": c}
        report["shared_2000_errors"] = {
            "numpy": summarize(a, b),
            "native": summarize(c, b),
        }
    failures = validate_diagnosis(report)
    controls = observer_negative_controls(report)
    report["observer_negative_controls"] = controls
    failures.extend(
        f"control:{c['mutation']}" for c in controls if c["status"] != "VERIFIED"
    )
    report["failure_ids"] = failures
    report["status"] = "VERIFIED" if not failures else "NOT VERIFIED"
    return report


def validate_diagnosis(report):
    """Derive the diagnosis verdict from observations, including negative controls."""
    failures = []
    if report["calls"].get("compute_lmm_chunk_ncvt1_c", 0):
        failures.append("old_route_unexpectedly_native")
    if not report["calls"].get("golden_section_optimize_lambda_split_ncvt1_numpy", 0):
        failures.append("split_route_not_observed")
    if report["reduced_errors"]["lambdas"]["failed_ids"] != [1]:
        failures.append("reduced_failure_not_reproduced")
    for key in ("numpy_errors", "native_errors"):
        for field, error in report["shared_control"][key].items():
            if error["failed_ids"]:
                failures.append(f"shared_control:{key}:{field}")
    for j, oracle in enumerate(report["shared_control"]["oracle"]):
        for actual, external in (
            ("betas", "beta"),
            ("ses", "se"),
            ("pwalds", "p_wald"),
            ("logls", "logl_H1"),
        ):
            if not np.isclose(
                report["shared_control"]["native"][actual][j],
                oracle[external],
                rtol=1e-6,
                atol=1e-4,
            ):
                failures.append(f"shared_oracle:{j}:{external}")
    for group, fields in report.get("shared_2000_errors", {}).items():
        for field, error in fields.items():
            if error["failed_ids"]:
                failures.append(f"shared_2000:{group}:{field}")
    for record in report["records"]:
        j = record["snp_id"]
        if record["native_calls"].get("compute_lmm_chunk_ncvt1_c") != 1:
            failures.append(f"{j}:native_route")
        expected = np.array([row["reml"] for row in record["oracle_grid"]])
        if not np.allclose(
            record["native_grid_objectives"], expected, rtol=1e-10, atol=1e-12
        ):
            failures.append(f"{j}:native_objective")
        if not np.allclose(
            record["native_pab"], record["numpy_pab"], rtol=1e-10, atol=1e-12
        ):
            failures.append(f"{j}:native_numpy_pab")
        if record["native_selected_grid_index"] != int(np.argmax(expected)):
            failures.append(f"{j}:native_grid_bracket")
        for level in range(3):
            for col, (a, b) in enumerate(
                ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
            ):
                if (
                    a >= level
                    and b >= level
                    and not np.allclose(
                        record["native_pab"][:, level, col],
                        record["dense_projection_products"][:, level, a, b],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                ):
                    failures.append(f"{j}:pab:{level}:{a}:{b}")
        for actual, external in (
            ("lambdas", "l_remle"),
            ("logls", "logl_H1"),
            ("betas", "beta"),
            ("ses", "se"),
            ("pwalds", "p_wald"),
        ):
            # Existing benchmark comparison policy, plus the existing lambda rtol
            # for the independently optimized optimum (not a new boundary policy).
            rtol = 2e-5 if actual == "lambdas" else 1e-6
            if not np.allclose(
                record["native"][actual],
                record["oracle"][external],
                rtol=rtol,
                atol=1e-4,
            ):
                failures.append(f"{j}:oracle:{external}")
    if (
        "original_errors" in report
        and not report["original_errors"]["lambdas"]["failed_ids"]
    ):
        failures.append("invalid_shared_input_failure_not_reproduced")
    return failures


def observer_negative_controls(report):
    """The evidence checker must reject corrupted observations for named reasons."""
    controls = []
    for mutation, detector in (
        ("native_pab", "1:pab:0:0:0"),
        ("native_objective", "1:native_objective"),
        ("native_route", "old_route_unexpectedly_native"),
    ):
        changed = copy.deepcopy(report)
        if mutation == "native_pab":
            changed["records"][1]["native_pab"][0, 0, 0] = 999.0
        elif mutation == "native_objective":
            changed["records"][1]["native_grid_objectives"][0] = 999.0
        else:
            changed["calls"]["compute_lmm_chunk_ncvt1_c"] = 1
        failures = validate_diagnosis(changed)
        controls.append(
            {
                "mutation": mutation,
                "expected_detector": detector,
                "failure_ids": failures,
                "status": "VERIFIED" if detector in failures else "NOT VERIFIED",
            }
        )
    return controls
