"""Compare supplied-kinship cases with external GEMMA and dense GLS."""

import json
import shutil
import sys

import numpy as np

from tests.math_validation.compare import (
    check_boundary_coverage,
    compare_files,
    read_rows,
)
from tests.math_validation.evidence import bundle_status, environment, write_json
from tests.math_validation.fixtures import verify_reference
from tests.math_validation.oracle_io import write_oracle_assoc
from tests.math_validation.reference import run_command, snapshot_files


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
            oracle = write_oracle_assoc(model, target / "oracle.assoc.txt", mode)
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
                    "files": snapshot_files(target),
                }
            )
        bundle["status"] = bundle_status(statuses)
    except Exception as exc:
        bundle["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(destination / "bundle.json", bundle)
    return bundle
