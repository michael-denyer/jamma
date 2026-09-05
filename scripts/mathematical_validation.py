#!/usr/bin/env python3
"""Generate external references, compare immutable cases, or trace the Pab defect.

Run with PYTHONPATH=src python scripts/mathematical_validation.py --help.
No reference output is created by the compare command.
"""

import argparse
import csv
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tests.math_validation.fixtures import (
    EXTERNAL_HEADERS,
    MANIFEST,
    REFERENCE,
    ROOT,
    digest,
    generate_reference,
    load_manifest,
    run_command,
    verify_reference,
)


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def environment():
    import scipy

    from jamma import jlinalg
    from jamma.lmm import accel

    expected_blas = os.environ.get("EXPECTED_BLAS_BACKEND")
    if expected_blas and jlinalg.blas_backend != expected_blas:
        raise RuntimeError(
            f"active BLAS {jlinalg.blas_backend!r} "
            f"differs from expected {expected_blas!r}"
        )

    with redirect_stdout(io.StringIO()) as config:
        np.show_config()
    compiler = subprocess.run(
        ["cc", "--version"], capture_output=True, text=True, check=True
    ).stdout
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": compiler,
        "numpy_config": config.getvalue(),
        "active_blas": jlinalg.blas_backend,
        "expected_blas": expected_blas,
        "ilp64": bool(jlinalg.blas_is_ilp64),
        "lapack_dsyevd": bool(jlinalg.blas_has_dsyevd),
        "lapack_dsyevr": bool(jlinalg.blas_has_dsyevr),
        "forced_numpy": os.environ.get("JAMMA_FORCE_NUMPY_FALLBACK") == "1",
        "native_binary": accel.require().__file__ if accel.available() else None,
        "native_sha256": digest(accel.require().__file__)
        if accel.available()
        else None,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {
            str(p.relative_to(ROOT)): digest(p)
            for parent in [ROOT / "src/jamma", ROOT / "tests/math_validation"]
            for p in sorted(parent.rglob("*"))
            if p.suffix in {".py", ".c", ".h", ".json"} and "evidence" not in p.parts
        },
        "driver_sha256": digest(Path(__file__)),
    }


def read_rows(path, mode=1, *, optional_logl=False):
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        header = tuple(reader.fieldnames or [])
        expected = EXTERNAL_HEADERS[mode]
        allowed: list[tuple[str, ...]] = [expected]
        if optional_logl and mode == 4:
            allowed.append(tuple(field for field in expected if field != "logl_H1"))
        if header not in allowed:
            raise ValueError(f"unexpected mode {mode} header: {reader.fieldnames}")
        rows = list(reader)
    if len({row["rs"] for row in rows}) != len(rows):
        raise ValueError("duplicate SNP IDs")
    return rows


def compare_files(
    actual,
    reference,
    *,
    af_contract="counted-allele",
    mode=1,
    reference_optional_logl=False,
):
    # Consume the repaired R7 comparator; do not create another lambda policy.
    from jamma.validation.compare import compare_assoc_results, load_gemma_assoc

    if af_contract != "counted-allele":
        raise ValueError(f"unknown allele-frequency contract: {af_contract}")

    a_rows = read_rows(actual, mode)
    b_rows = read_rows(reference, mode, optional_logl=reference_optional_logl)
    result = compare_assoc_results(
        load_gemma_assoc(actual), load_gemma_assoc(reference)
    )
    errors = []
    if [r["rs"] for r in a_rows] != [r["rs"] for r in b_rows]:
        errors.append("ordered SNP IDs")
    for a, b in zip(a_rows, b_rows, strict=False):
        for field in ("chr", "rs", "ps", "n_miss", "allele1", "allele0"):
            if a[field] != b[field]:
                errors.append(f"{b['rs']}:{field}")
        # Both current writers emit BIM A1 dosage frequency. Keep its direction:
        # folding to MAF would hide flips. Two .3f values have at most 1e-3
        # combined rounding uncertainty; this supplements the existing gate.
        # Compare the printed decimals exactly at the formatting limit.
        # Binary subtraction makes 0.538 - 0.537 slightly greater than 0.001.
        actual_af = Decimal(a["af"])
        reference_af = Decimal(b["af"])
        if not (
            actual_af.is_finite()
            and reference_af.is_finite()
            and 0 <= actual_af <= 1
            and abs(actual_af - reference_af) <= Decimal("0.001")
        ):
            errors.append(f"{b['rs']}:af_orientation")
    failures = []
    for a, b in zip(
        load_gemma_assoc(actual), load_gemma_assoc(reference), strict=False
    ):
        item = compare_assoc_results([a], [b])
        for field, value in asdict(item).items():
            if isinstance(value, dict) and not value["passed"]:
                failures.append(f"{b.rs}:{field}")
    return {
        "status": "VERIFIED" if result.passed and not errors else "NOT VERIFIED",
        "fields": asdict(result),
        "failure_ids": errors + failures,
        "af_contract": af_contract,
        "reference_absent_fields": [
            field
            for field in EXTERNAL_HEADERS[mode]
            if b_rows and field not in b_rows[0]
        ],
    }


def oracle_output(model, destination, mode=1):
    from tests.math_validation.dense_oracle import all_test_statistics, optimize

    k, x, y = (np.array(model[key]) for key in ("kinship", "genotypes", "phenotype"))
    w = np.array(model.get("covariates", np.ones((len(y), 1))))
    rows, details = [], []
    for j, snp in enumerate(model["snp_ids"]):
        fit = optimize(k, w, x[:, j], y)
        if mode != 1:
            fit.update(all_test_statistics(k, w, x[:, j], y))
        if mode == 3:
            fit["beta"], fit["se"] = fit["score_beta"], fit["score_se"]
        details.append({"rs": snp, **fit})
        rows.append(
            {
                "chr": 1,
                "rs": snp,
                "ps": 100 + j,
                "n_miss": 0,
                "allele1": "A",
                "allele0": "G",
                "af": f"{x[:, j].mean() / 2:.3f}",
                **{field: fit[field] for field in EXTERNAL_HEADERS[mode][7:]},
            }
        )
    with destination.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=EXTERNAL_HEADERS[mode], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    return details


def check_boundary_coverage(case, records, expectations):
    """Require the boundary SNPs and classes declared before comparison."""
    expected = expectations.get(case["id"], {})
    observed = {record["rs"]: record["classes"][0] for record in records}
    passed = bool(expected) and observed == expected
    return {
        "status": "VERIFIED" if passed else "NOT VERIFIED",
        "expected": expected,
        "observed": observed,
        "failure_ids": [] if passed else [f"{case['id']}:boundary-coverage"],
    }


def compare(manifest, reference, destination):
    destination.mkdir(parents=True, exist_ok=False)
    bundle = {
        "schema_version": 1,
        "status": "INCONCLUSIVE",
        "manifest": manifest,
        "environment": environment(),
        "invocation": sys.argv,
        "cases": [],
        "untested": manifest["untested"],
    }
    statuses = []
    try:
        for case in manifest["cases"]:
            source, provenance = verify_reference(case, reference)
            target = destination / case["id"]
            shutil.copytree(source, target)
            model = json.loads((target / "model.json").read_text())
            mode = case["mode"]
            oracle = oracle_output(model, target / "oracle.assoc.txt", mode)
            command = run_command(
                [
                    sys.executable,
                    "-m",
                    "jamma",
                    "-bfile",
                    "tiny",
                    "-k",
                    "kinship.txt",
                    *case["jamma_args"],
                    "-outdir",
                    ".",
                    "-o",
                    "jamma",
                ],
                target,
                "jamma",
            )
            checks = {
                "jamma_gemma": compare_files(
                    target / "jamma.assoc.txt",
                    target / "gemma.assoc.txt",
                    mode=mode,
                    reference_optional_logl=True,
                ),
                "oracle_gemma": compare_files(
                    target / "oracle.assoc.txt",
                    target / "gemma.assoc.txt",
                    af_contract="counted-allele",
                    mode=mode,
                    reference_optional_logl=True,
                ),
                "jamma_oracle": compare_files(
                    target / "jamma.assoc.txt",
                    target / "oracle.assoc.txt",
                    mode=mode,
                ),
            }
            boundary_records = []
            boundary_coverage = None
            if mode in (1, 4):
                from jamma.validation.compare import _classify_lambdas
                from jamma.validation.tolerances import LambdaBoundaryPolicy
                from tests.math_validation.phase1 import boundary_trace

                actual_rows = read_rows(target / "jamma.assoc.txt", mode)
                reference_rows = read_rows(
                    target / "gemma.assoc.txt", mode, optional_logl=True
                )
                for j, (actual, reference_row) in enumerate(
                    zip(actual_rows, reference_rows, strict=True)
                ):
                    pair = [float(row["l_remle"]) for row in (actual, reference_row)]
                    classes = _classify_lambdas(np.array(pair), LambdaBoundaryPolicy())
                    if classes[0] == classes[1] and classes[0] in {"lower", "upper"}:
                        trace = boundary_trace(
                            np.array(model["kinship"]),
                            np.array(
                                model.get(
                                    "covariates", np.ones((len(model["phenotype"]), 1))
                                )
                            ),
                            np.array(model["genotypes"])[:, j],
                            np.array(model["phenotype"]),
                            *pair,
                        )
                        boundary_records.append({"rs": actual["rs"], **trace})
                boundary_coverage = check_boundary_coverage(
                    case, boundary_records, manifest["boundary_expectations"]
                )
                statuses.append(boundary_coverage["status"])
                statuses.extend(
                    "VERIFIED" if trace["passed"] else "NOT VERIFIED"
                    for trace in boundary_records
                )
            expected_n = case["n_samples"]
            for tool in ("gemma", "jamma"):
                log = (target / f"{tool}.stdout.txt").read_text()
                if f"number of analyzed individuals = {expected_n}" not in log:
                    raise ValueError(f"{tool} sample count differs from manifest")
            statuses.extend(c["status"] for c in checks.values())
            bundle["cases"].append(
                {
                    "id": case["id"],
                    "reference_provenance": provenance,
                    "command": command,
                    "oracle": oracle,
                    "comparisons": checks,
                    "boundary_records": boundary_records,
                    "boundary_coverage": boundary_coverage,
                    "selected_sample_ids": model["sample_ids"],
                    "sample_selection_basis": (
                        "all FAM rows valid; sample counts checked in both raw logs"
                    ),
                    "selected_snp_ids": [
                        r["rs"] for r in read_rows(target / "jamma.assoc.txt", mode)
                    ],
                    "execution_log": [
                        line
                        for line in (target / "jamma.stdout.txt")
                        .read_text()
                        .splitlines()
                        if "Execution plan:" in line
                        or "Pipeline:" in line
                        or "Eigendecomp:" in line
                    ],
                    "files": {
                        p.name: digest(p) for p in target.iterdir() if p.is_file()
                    },
                }
            )
        bundle["status"] = (
            "VERIFIED"
            if statuses and all(status == "VERIFIED" for status in statuses)
            else "NOT VERIFIED"
        )
    except Exception as exc:
        bundle["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(destination / "bundle.json", bundle)
    return bundle


def summarize_evidence(root, destination):
    """Generate the review summary from immutable raw bundles, never hand maxima."""
    tiny_path, pab_path = root / "tiny/bundle.json", root / "pab/pab.json"
    tiny = json.loads(tiny_path.read_text())
    pab = json.loads(pab_path.read_text())
    diagnostic = pab["diagnostic"]
    summary = {
        "schema_version": 1,
        "tiny_status": tiny["status"],
        "pab_diagnosis_status": diagnostic["status"],
        "historical_parity_status": diagnostic["original_assertion_status"],
        "environment": {
            key: pab["environment"][key]
            for key in (
                "revision",
                "platform",
                "python",
                "numpy",
                "scipy",
                "active_blas",
                "native_sha256",
            )
        },
        "case_ids": [case["id"] for case in tiny["cases"]],
        "tiny_comparisons": {case["id"]: case["comparisons"] for case in tiny["cases"]},
        "first_divergence": diagnostic["first_divergence"],
        "reduction": diagnostic["reduction"],
        "original_errors": {
            field: {
                "max_abs": value["max_abs"],
                "failure_count": len(value["failed_ids"]),
            }
            for field, value in diagnostic["original_errors"].items()
        },
        "shared_2000_errors": diagnostic["shared_2000_errors"],
        "observer_negative_controls": diagnostic["observer_negative_controls"],
        "bundles": {
            "tiny": {"path": str(tiny_path), "sha256": digest(tiny_path)},
            "pab": {"path": str(pab_path), "sha256": digest(pab_path)},
        },
        "untested": tiny["untested"],
    }
    write_json(destination, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "generate",
            "generate-pipeline",
            "generate-loco",
            "generate-weights",
            "compare",
            "pipeline",
            "loco",
            "weights",
            "pab",
            "phase1",
            "summarize",
        ),
    )
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gemma", default=str(Path.home() / ".local/bin/gemma"))
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command in {"generate-weights", "weights"}:
        from tests.math_validation.weight_contract import (
            compare_weights,
            generate_external,
        )

        if args.command == "generate-weights":
            generate_external(args.output, args.gemma)
            return 0
        result = compare_weights(args.output)
        print(result["status"])
        if result["default_optimizer_status"] != "VERIFIED":
            print("default optimizer: NOT VERIFIED")
        return int(result["status"] != "VERIFIED")
    if args.command in {"generate-loco", "loco"}:
        from tests.math_validation.loco_cases import compare_loco, generate_external

        if args.command == "generate-loco":
            generate_external(args.output, args.gemma)
            return 0
        result = compare_loco(args.output)
        print(result["status"])
        return int(result["status"] != "VERIFIED")
    if args.command in {"generate-pipeline", "pipeline"}:
        from tests.math_validation.pipeline_cases import (
            compare_pipeline,
            generate_external,
        )

        if args.command == "generate-pipeline":
            generate_external(args.output, args.gemma)
            return 0
        result = compare_pipeline(args.output)
        print(result["status"])
        return int(result["status"] != "VERIFIED")
    if args.command == "phase1":
        from tests.math_validation.phase1 import phase1_evidence

        args.output.mkdir(parents=True, exist_ok=False)
        observed_environment = environment()
        backends = (
            ("numpy",) if observed_environment["forced_numpy"] else ("numpy", "native")
        )
        result = {
            "schema_version": 1,
            "environment": observed_environment,
            "invocation": sys.argv,
            "evidence": phase1_evidence(backends=backends),
        }
        write_json(args.output / "phase1.json", result)
        print(result["evidence"]["status"])
        return int(result["evidence"]["status"] != "VERIFIED")
    if args.command == "summarize":
        if args.evidence_root is None:
            parser.error("summarize requires --evidence-root")
        summarize_evidence(args.evidence_root, args.output)
    elif args.command == "generate":
        generate_reference(manifest, args.output, args.gemma)
    elif args.command == "compare":
        result = compare(manifest, args.reference, args.output)
        print(result["status"])
        return int(result["status"] != "VERIFIED")
    else:
        from tests.math_validation.pab_trace import diagnose

        args.output.mkdir(parents=True, exist_ok=False)
        result = {
            "schema_version": 1,
            "environment": environment(),
            "invocation": sys.argv,
            "diagnostic": diagnose(),
        }
        write_json(args.output / "pab.json", result)
        print(result["diagnostic"]["status"], result["diagnostic"]["verdict"])
        return int(result["diagnostic"]["status"] != "VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
