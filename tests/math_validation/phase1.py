"""Phase 1 evidence for likelihood semantics and constrained REML optima."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from jamma.lmm import accel
from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy
from jamma.lmm.likelihood import compute_null_model_mle
from jamma.lmm.pab import build_pab_table_for_c
from jamma.lmm.stats import AssocResult
from jamma.lmm.uab import batch_compute_uab_numpy, compute_uab_invariant_soa
from jamma.validation.compare import _classify_lambdas, compare_assoc_results
from jamma.validation.tolerances import LambdaBoundaryPolicy, ToleranceConfig
from tests.builders import rotated_lmm_inputs
from tests.math_validation import dense_oracle

L_MIN = 1e-5
L_MAX = 1e5
OBJECTIVE_ABSOLUTE_FLOOR = 1e-8
OBJECTIVE_ROUNDOFF_MULTIPLIER = 128
MODE4_FIELDS = ("logl_H1", "l_remle", "l_mle", "p_wald", "p_lrt", "p_score")
MODE4_ORACLE_FIELDS = ("beta", "se", *MODE4_FIELDS)
MODE1_FIELDS = ("beta", "se", "logl_H1", "l_remle", "p_wald")
ARRAY_FIELDS = {
    "logl_H1": "logls",
    "l_remle": "lambdas",
    "l_mle": "lambdas_mle",
    "p_wald": "pwalds",
    "p_lrt": "p_lrts",
    "p_score": "p_scores",
}


class ModeRecord(TypedDict):
    snp: str
    actual: dict[str, float]
    oracle: dict[str, float]
    mode1_reml_logl: float
    mode4_mle_logl: float
    mle_reml_separation: float
    oracle_comparison_passed: bool
    oracle_field_checks: dict[str, bool]
    mode1_actual: dict[str, float]
    mode1_oracle: dict[str, float]
    mode1_oracle_field_checks: dict[str, bool]


def _required_array(result: dict[str, np.ndarray | None], key: str) -> np.ndarray:
    value = result[key]
    if value is None:
        raise AssertionError(f"production mode omitted required array {key}")
    return value


def _production_results(data, backend) -> dict[int, dict[str, np.ndarray | None]]:
    uab = batch_compute_uab_numpy(
        data.n_cvt, data.UtW, data.Uty, np.ascontiguousarray(data.UtG.T)
    )
    lambda_null, logl_h0 = compute_null_model_mle(
        data.eigenvalues, data.UtW, data.Uty, data.n_cvt
    )
    hi_null = 1.0 / (lambda_null * data.eigenvalues + 1.0)
    common = {
        "n_cvt": data.n_cvt,
        "eigenvalues": data.eigenvalues,
        "n_samples": data.n_samples,
        "Hi_eval_null": hi_null,
        "logl_H0": logl_h0,
    }
    if backend == "numpy":
        return cast(
            dict[int, dict[str, np.ndarray | None]],
            {
                mode: compute_lmm_chunk_numpy(mode, Uab_batch=uab, **common)
                for mode in (1, 2, 4)
            },
        )

    module = accel.require()
    invariant = compute_uab_invariant_soa(data.UtW, data.Uty, data.n_cvt)
    results = {}
    for mode in (1, 2, 4):
        optional = {"logl_H0": logl_h0} if mode == 2 else {}
        if mode == 4:
            optional = {"logl_H0": logl_h0, "hi_eval_null": hi_null}
        if data.n_cvt == 1:
            workspace = module.create_workspace_ncvt1_c(
                data.eigenvalues,
                invariant,
                data.UtW[:, 0],
                data.Uty,
                data.n_samples,
                L_MIN,
                L_MAX,
                50,
                20,
                lmm_mode=mode,
                **optional,
            )
            results[mode] = module.compute_lmm_chunk_ncvt1_c(
                workspace, np.ascontiguousarray(data.UtG.T), 1
            )
        else:
            workspace = module.create_workspace_general_c(
                data.eigenvalues,
                invariant,
                data.UtW,
                data.Uty,
                data.n_samples,
                L_MIN,
                L_MAX,
                50,
                20,
                1,
                build_pab_table_for_c(data.n_cvt)._asdict(),
                lmm_mode=mode,
                **optional,
            )
            results[mode] = module.compute_lmm_chunk_fused_general_c(
                workspace, np.ascontiguousarray(data.UtG.T), 1
            )
    return cast(dict[int, dict[str, np.ndarray | None]], results)


def _row(values):
    return AssocResult(
        chr="1",
        rs="phase1-snp",
        ps=1,
        n_miss=0,
        allele1="A",
        allele0="G",
        af=0.25,
        beta=float(values["beta"]),
        se=float(values["se"]),
        **{field: float(values[field]) for field in MODE4_FIELDS},
    )


def _mode1_row(values):
    return AssocResult(
        chr="1",
        rs="phase1-snp",
        ps=1,
        n_miss=0,
        allele1="A",
        allele0="G",
        af=0.25,
        beta=float(values["beta"]),
        se=float(values["se"]),
        logl_H1=float(values["logl_H1"]),
        l_remle=float(values["l_remle"]),
        p_wald=float(values["p_wald"]),
    )


def _named_detectors(expected, row_factory, fields, prefix):
    detectors = {}
    config = ToleranceConfig()
    for field in fields:
        changed = dict(expected)
        value = changed[field]
        changed[field] = value + max(1.0, abs(value)) * 0.1
        comparison = compare_assoc_results(
            [row_factory(changed)], [row_factory(expected)], config
        )
        field_result = getattr(comparison, field)
        detectors[field] = {
            "detector": f"{prefix}:{field}",
            "passed": not comparison.passed and not field_result.passed,
        }
    return detectors


def mode4_evidence(*, backend="numpy", n_cvt=1, seed=911, mutation=None):
    """Return JSON-safe evidence for modes 1, 2, and 4 on identical inputs."""
    data = rotated_lmm_inputs(40, 6, n_cvt=n_cvt, seed=seed)
    results = _production_results(data, backend)
    kinship = np.diag(data.eigenvalues)
    records: list[ModeRecord] = []
    for snp in range(data.n_snps):
        oracle = dense_oracle.all_test_statistics(
            kinship, data.UtW, data.UtG[:, snp], data.Uty
        )
        actual = {
            field: float(_required_array(results[4], ARRAY_FIELDS[field])[snp])
            for field in MODE4_FIELDS
        }
        actual.update(
            beta=float(_required_array(results[4], "betas")[snp]),
            se=float(_required_array(results[4], "ses")[snp]),
        )
        mode1_actual = {
            "beta": float(_required_array(results[1], "betas")[snp]),
            "se": float(_required_array(results[1], "ses")[snp]),
            "logl_H1": float(_required_array(results[1], "logls")[snp]),
            "l_remle": float(_required_array(results[1], "lambdas")[snp]),
            "p_wald": float(_required_array(results[1], "pwalds")[snp]),
        }
        mode1_oracle = {
            "beta": float(oracle["beta"]),
            "se": float(oracle["se"]),
            "logl_H1": float(oracle["reml_log_likelihood"]),
            "l_remle": float(oracle["l_remle"]),
            "p_wald": float(oracle["p_wald"]),
        }
        mutation_mode, _, mutation_field = (mutation or "").partition(":")
        if snp == 0:
            if mutation_mode == "mode4" and mutation_field in MODE4_ORACLE_FIELDS:
                actual[mutation_field] += max(1.0, abs(actual[mutation_field])) * 0.1
            elif mutation_mode == "mode1" and mutation_field in MODE1_FIELDS:
                mode1_actual[mutation_field] += (
                    max(1.0, abs(mode1_actual[mutation_field])) * 0.1
                )
            elif mutation in MODE4_ORACLE_FIELDS:
                actual[mutation] += max(1.0, abs(actual[mutation])) * 0.1
        comparison = compare_assoc_results([_row(actual)], [_row(oracle)])
        field_checks = {
            field: bool(getattr(comparison, field).passed)
            for field in MODE4_ORACLE_FIELDS
        }
        mode1_comparison = compare_assoc_results(
            [_mode1_row(mode1_actual)], [_mode1_row(mode1_oracle)]
        )
        mode1_field_checks = {
            field: bool(getattr(mode1_comparison, field).passed)
            for field in MODE1_FIELDS
        }
        records.append(
            {
                "snp": f"synthetic-{snp}",
                "actual": actual,
                "oracle": {
                    field: float(oracle[field]) for field in MODE4_ORACLE_FIELDS
                },
                "mode1_actual": mode1_actual,
                "mode1_oracle": mode1_oracle,
                "mode1_oracle_field_checks": mode1_field_checks,
                "mode1_reml_logl": float(_required_array(results[1], "logls")[snp]),
                "mode4_mle_logl": float(_required_array(results[4], "logls")[snp]),
                "mle_reml_separation": abs(
                    float(
                        _required_array(results[4], "logls")[snp]
                        - _required_array(results[1], "logls")[snp]
                    )
                ),
                "oracle_comparison_passed": comparison.passed,
                "oracle_field_checks": field_checks,
            }
        )
    fixture = Path(__file__).parents[1] / "fixtures/gemma_all_tests/gemma_all.assoc.txt"
    header = fixture.read_text().splitlines()[0].split("\t")
    first_oracle = records[0]["oracle"]
    separation_passed = all(
        record["mle_reml_separation"]
        > 100
        * max(
            0.5 * 10.0 ** (int(np.floor(np.log10(abs(value)))) - 6)
            for value in (
                record["mode1_reml_logl"],
                record["mode4_mle_logl"],
            )
        )
        for record in records
    )
    detectors = {
        **_named_detectors(first_oracle, _row, MODE4_ORACLE_FIELDS, "mode4"),
        **{
            f"mode1:{field}": value
            for field, value in _named_detectors(
                records[0]["mode1_oracle"], _mode1_row, MODE1_FIELDS, "mode1"
            ).items()
        },
    }
    checks = {
        "mode2_logl_H1_present": results[2].get("logls") is not None,
        "mode2_mode4_logl_H1_equal": bool(
            np.array_equal(
                _required_array(results[2], "logls"),
                _required_array(results[4], "logls"),
            )
        ),
        "mode2_mode4_l_mle_equal": bool(
            np.array_equal(
                _required_array(results[2], "lambdas_mle"),
                _required_array(results[4], "lambdas_mle"),
            )
        ),
        "mode2_mode4_p_lrt_equal": bool(
            np.array_equal(
                _required_array(results[2], "p_lrts"),
                _required_array(results[4], "p_lrts"),
            )
        ),
        "mode1_mode4_likelihood_separation": separation_passed,
        "all_named_mutations_detected": all(
            item["passed"] for item in detectors.values()
        ),
    }
    checks.update(
        {
            f"oracle_{field}": all(
                record["oracle_field_checks"][field] for record in records
            )
            for field in MODE4_ORACLE_FIELDS
        }
    )
    checks.update(
        {
            f"mode1_oracle_{field}": all(
                record["mode1_oracle_field_checks"][field] for record in records
            )
            for field in MODE1_FIELDS
        }
    )
    return {
        "backend": backend,
        "n_cvt": n_cvt,
        "status": "VERIFIED" if all(checks.values()) else "NOT VERIFIED",
        "checks": checks,
        "failure_ids": [name for name, passed in checks.items() if not passed],
        "mode2_logl_H1_emitted": checks["mode2_logl_H1_present"],
        "mode2_mode4_logl_H1_equal": checks["mode2_mode4_logl_H1_equal"],
        "mode2_mode4_l_mle_equal": checks["mode2_mode4_l_mle_equal"],
        "mode2_mode4_p_lrt_equal": checks["mode2_mode4_p_lrt_equal"],
        "small_gemma_mode4_logl_H1": (
            "present" if "logl_H1" in header else "absent_in_raw_gemma_0.98.5_layout"
        ),
        "records": records,
        "named_detectors": detectors,
    }


def objective_tolerance(term_scale):
    """A priori log-likelihood loss limit from the Phase 1 audit.

    The fixed 1e-8 limit bounds twice the log-likelihood change by 2e-8.
    The second term allows 128 binary64 rounding units across the absolute
    terms of the independently evaluated REML expression. Neither constant
    was fitted to observed optimizer differences.
    """
    return max(
        OBJECTIVE_ABSOLUTE_FLOOR,
        OBJECTIVE_ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * term_scale,
    )


def boundary_trace(
    kinship,
    covariates,
    genotype,
    phenotype,
    actual_lambda,
    reference_lambda,
    *,
    policy=None,
):
    """Evaluate a matching-boundary pair without hiding its lambda distance."""
    if policy is None:
        policy = LambdaBoundaryPolicy()
    candidates = np.array([actual_lambda, reference_lambda], dtype=float)
    classes = _classify_lambdas(candidates, policy).tolist()
    if "invalid" in classes:
        return {
            "actual_lambda": float(actual_lambda),
            "reference_lambda": float(reference_lambda),
            "classes": classes,
            "class_match": False,
            "passed": False,
            "detector": "invalid-boundary-lambda",
        }
    optimum = dense_oracle.optimize(
        kinship,
        covariates,
        genotype,
        phenotype,
        objective="reml",
        bounds=(policy.lower, policy.upper),
    )
    optimum_lambda = float(optimum["l_remle"])
    optimum_fit = dense_oracle.evaluate(
        kinship, covariates, genotype, phenotype, optimum_lambda
    )
    grid_x = np.linspace(np.log(policy.lower), np.log(policy.upper), 129)
    grid_y = np.array(
        [
            dense_oracle.evaluate(
                kinship, covariates, genotype, phenotype, float(np.exp(point))
            )["reml"]
            for point in grid_x
        ]
    )
    h = 1e-4
    x = np.log(optimum_lambda)

    def at_log_lambda(point):
        return dense_oracle.evaluate(
            kinship, covariates, genotype, phenotype, float(np.exp(point))
        )["reml"]

    if np.isclose(x, grid_x[0], atol=1e-10):
        nearby = [at_log_lambda(x + step * h) for step in range(4)]
        curvature = (2 * nearby[0] - 5 * nearby[1] + 4 * nearby[2] - nearby[3]) / (
            h * h
        )
        curvature_stencil = "forward-four-point"
    elif np.isclose(x, grid_x[-1], atol=1e-10):
        nearby = [at_log_lambda(x - step * h) for step in range(4)]
        curvature = (2 * nearby[0] - 5 * nearby[1] + 4 * nearby[2] - nearby[3]) / (
            h * h
        )
        curvature_stencil = "backward-four-point"
    else:
        nearby = [at_log_lambda(x + step * h) for step in (-1, 0, 1)]
        curvature = (nearby[0] - 2 * nearby[1] + nearby[2]) / (h * h)
        curvature_stencil = "central-three-point"
    objectives = [
        dense_oracle.evaluate(kinship, covariates, genotype, phenotype, value)["reml"]
        for value in candidates
    ]
    exponent = int(np.floor(np.log10(abs(reference_lambda))))
    rounding_half_width = 0.5 * 10.0 ** (exponent - 6)
    rounding_interval = (
        max(policy.lower, reference_lambda - rounding_half_width),
        min(policy.upper, reference_lambda + rounding_half_width),
    )
    reference_best = max(
        dense_oracle.evaluate(kinship, covariates, genotype, phenotype, value)["reml"]
        for value in rounding_interval
    )
    tolerance = objective_tolerance(optimum_fit["reml_term_scale"])
    losses = [
        max(0.0, optimum_fit["reml"] - objectives[0]),
        max(0.0, optimum_fit["reml"] - reference_best),
    ]
    class_match = classes[0] == classes[1] and classes[0] in {"lower", "upper"}
    return {
        "actual_lambda": float(actual_lambda),
        "reference_lambda": float(reference_lambda),
        "classes": classes,
        "class_match": class_match,
        "absolute_lambda_distance": float(abs(actual_lambda - reference_lambda)),
        "relative_lambda_distance": float(
            abs(actual_lambda - reference_lambda) / abs(reference_lambda)
        ),
        "oracle_optimum_lambda": optimum_lambda,
        "objective_values": [float(value) for value in objectives],
        "objective_losses": [float(value) for value in losses],
        "reference_rounding_interval": [float(value) for value in rounding_interval],
        "objective_tolerance": float(tolerance),
        "local_curvature_log_lambda": float(curvature),
        "curvature_stencil": curvature_stencil,
        "grid": {
            "log_lambda": grid_x.tolist(),
            "reml": grid_y.tolist(),
        },
        "passed": bool(class_match and max(losses) <= tolerance),
        "detector": "matching-boundary-objective-degradation",
        "policy": {
            "absolute_floor": OBJECTIVE_ABSOLUTE_FLOOR,
            "roundoff_multiplier": OBJECTIVE_ROUNDOFF_MULTIPLIER,
            "binary64_epsilon": float(np.finfo(np.float64).eps),
            "term_scale": float(optimum_fit["reml_term_scale"]),
        },
    }


def _raw_gemma_boundary_evidence():
    directory = (
        Path(__file__).parents[1]
        / "fixtures/mathematical_validation/tiny-wald-supplied-kinship"
    )
    model = json.loads((directory / "model.json").read_text())
    with (directory / "gemma.assoc.txt").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    kinship = np.asarray(model["kinship"], dtype=float)
    genotypes = np.asarray(model["genotypes"], dtype=float)
    phenotype = np.asarray(model["phenotype"], dtype=float)
    covariates = np.ones((len(phenotype), 1))
    records = []
    for index in (0, 2):
        oracle = dense_oracle.optimize(
            kinship, covariates, genotypes[:, index], phenotype, objective="reml"
        )
        trace = boundary_trace(
            kinship,
            covariates,
            genotypes[:, index],
            phenotype,
            float(oracle["l_remle"]),
            float(rows[index]["l_remle"]),
        )
        trace["snp"] = rows[index]["rs"]
        trace["comparison"] = "independent-oracle_vs_raw-gemma-0.98.5"
        records.append(trace)
    return records


def phase1_evidence(*, backends=("numpy", "native"), mutation=None):
    """Callable JSON-safe Phase 1 evidence bundle."""
    if not backends or any(backend not in {"numpy", "native"} for backend in backends):
        raise ValueError("backends must be a nonempty subset of ('numpy', 'native')")
    mode = [
        mode4_evidence(
            backend=backend,
            n_cvt=n_cvt,
            seed=910 + n_cvt,
            mutation=mutation,
        )
        for backend in backends
        for n_cvt in (1, 2)
    ]
    data = rotated_lmm_inputs(20, 8, seed=1)
    boundary = []
    for backend in backends:
        results = _production_results(data, backend)[1]
        lambdas = _required_array(results, "lambdas")
        classes = _classify_lambdas(lambdas, LambdaBoundaryPolicy())
        indices = np.flatnonzero(classes == "lower")
        if not len(indices):
            raise AssertionError("deterministic Phase 1 case has no lower-bound SNP")
        index = int(indices[0])
        trace = boundary_trace(
            np.diag(data.eigenvalues),
            data.UtW,
            data.UtG[:, index],
            data.Uty,
            float(lambdas[index]),
            L_MIN,
        )
        trace["backend"] = backend
        boundary.append(trace)
    raw_gemma = _raw_gemma_boundary_evidence()
    failures = [
        f"mode4:{item['backend']}:ncvt{item['n_cvt']}:{failure}"
        for item in mode
        for failure in item["failure_ids"]
    ]
    failures.extend(
        f"boundary:{item.get('backend', item.get('snp'))}"
        for item in [*boundary, *raw_gemma]
        if not item["passed"]
    )
    return {
        "status": "VERIFIED" if not failures else "NOT VERIFIED",
        "failure_ids": failures,
        "routes": {"declared": list(backends), "observed": list(backends)},
        "mode4": mode,
        "boundary": boundary,
        "raw_gemma_boundary": raw_gemma,
    }
