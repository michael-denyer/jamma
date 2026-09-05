"""External GEMMA contract for per-individual residual weights."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tests.math_validation.compare import compare_files, read_rows
from tests.math_validation.dense_oracle import evaluate
from tests.math_validation.evidence import (
    bundle_status,
    environment,
    run_pipeline,
    write_json,
)
from tests.math_validation.oracle_io import write_oracle_assoc
from tests.math_validation.reference import (
    copy_reference,
    digest,
    gemma_binary,
    run_command,
    snapshot_files,
    verify_reference_dir,
    write_provenance,
)

if TYPE_CHECKING:
    pass

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests/fixtures/mathematical_weights/mode4-missing-covariates"
NONPOSITIVE_REFERENCE = (
    ROOT / "tests/fixtures/mathematical_weights/mode4-nonpositive-supplied"
)


def load_weight_cases() -> list[dict]:
    return [
        {"id": f"{kinship}-{backend}", "backend": backend, "kinship": kinship}
        for kinship in ("computed-k", "supplied-k")
        for backend in ("numpy", "numpy-streaming")
    ]


def load_nonpositive_weight_cases() -> list[dict]:
    return [
        {
            "id": "nonpositive-supplied-k-numpy",
            "backend": "numpy",
            "kinship": "supplied-k",
            "nonpositive": True,
        }
    ]


def _generate_positive_reference(destination: Path, gemma: Path | str) -> None:
    """Generate the positive-weight case inside an absent directory."""
    from tests.math_validation.pipeline_cases import (
        _arrays,
        _selected_model,
        _write_plink,
    )

    binary, _ = gemma_binary(gemma)
    destination.mkdir(parents=True, exist_ok=False)
    arrays = _arrays()
    _write_plink(destination, arrays)
    (destination / "weights.txt").write_text(
        "".join(f"{value:.17g}\n" for value in np.linspace(0.5, 2.0, 40))
    )
    kinship_run = run_command(
        [
            str(binary),
            "-bfile",
            "tiny",
            "-c",
            "covariates.txt",
            "-gk",
            "1",
            "-maf",
            "0.1",
            "-miss",
            "0.1",
            "-outdir",
            ".",
            "-o",
            "gemma_kinship",
        ],
        destination,
        "gemma_kinship",
    )
    kinship = np.loadtxt(destination / "gemma_kinship.cXX.txt")
    model = _selected_model(arrays, kinship, 0.1, 0.1)
    (destination / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    association_run = run_command(
        [
            str(binary),
            "-bfile",
            "tiny",
            "-k",
            "gemma_kinship.cXX.txt",
            "-c",
            "covariates.txt",
            "-widv",
            "weights.txt",
            "-lmm",
            "4",
            "-maf",
            "0.1",
            "-miss",
            "0.1",
            "-outdir",
            ".",
            "-o",
            "gemma",
        ],
        destination,
        "gemma",
    )
    write_provenance(
        destination,
        schema_version=1,
        case="mode4-missing-covariates",
        gemma={
            "version": "0.98.5",
            "binary": str(binary),
            "binary_sha256": digest(binary),
            "source_repository": "https://github.com/genetics-statistics/GEMMA",
            "source_revision": "c37b0445f820b682836a1d20009ce1817546493a",
        },
        commands={"kinship": kinship_run, "association": association_run},
    )


def generate_nonpositive_reference(destination: Path, gemma: Path | str) -> None:
    """Generate GEMMA evidence for one negative and one zero analyzed weight."""
    _generate_positive_reference(destination, gemma)
    weights_path = destination / "weights.txt"
    weights = np.loadtxt(weights_path)
    weights[2] = -0.5
    weights[3] = 0.0
    weights_path.write_text("".join(f"{value:.17g}\n" for value in weights))
    provenance_path = destination / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    binary = Path(provenance["gemma"]["binary"])
    association_run = run_command(
        [
            str(binary),
            "-bfile",
            "tiny",
            "-k",
            "gemma_kinship.cXX.txt",
            "-c",
            "covariates.txt",
            "-widv",
            "weights.txt",
            "-lmm",
            "4",
            "-maf",
            "0.1",
            "-miss",
            "0.1",
            "-outdir",
            ".",
            "-o",
            "gemma",
        ],
        destination,
        "gemma",
    )
    provenance["case"] = "mode4-nonpositive-supplied"
    provenance["contract"] = {
        "weights": "analyzed sample F2 weight -0.5 and F3 weight 0.0",
        "source_behavior": "GEMMA zeros K and eigenvector rows for weights <= 0",
    }
    provenance["commands"]["association"] = association_run
    write_provenance(
        destination,
        **{key: value for key, value in provenance.items() if key != "files"},
    )


def generate_external(destination: Path, gemma: Path | str) -> None:
    """Generate positive and nonpositive immutable weighted references."""
    destination.mkdir(parents=True, exist_ok=False)
    _generate_positive_reference(destination / REFERENCE.name, gemma)
    generate_nonpositive_reference(destination / NONPOSITIVE_REFERENCE.name, gemma)


def require_reference() -> tuple[Path, dict]:
    provenance = verify_reference_dir(REFERENCE, label="weight reference")
    return REFERENCE, provenance


def require_nonpositive_reference() -> tuple[Path, dict]:
    provenance = verify_reference_dir(
        NONPOSITIVE_REFERENCE, label="nonpositive weight reference"
    )
    return NONPOSITIVE_REFERENCE, provenance


def _weighted_model(model_override: dict | None = None) -> dict:
    source, _ = require_reference()
    model = (
        json.loads((source / "model.json").read_text())
        if model_override is None
        else model_override
    )
    valid = np.asarray(model["valid_mask"])
    k = np.loadtxt(source / "gemma_kinship.cXX.txt")[np.ix_(valid, valid)]
    k = k - k.mean(axis=0)[None, :] - k.mean(axis=1)[:, None] + k.mean()
    root_weight = np.sqrt(np.loadtxt(source / "weights.txt")[valid])
    weighted_k = k / root_weight[:, None] / root_weight[None, :]
    weighted_y = np.asarray(model["phenotype"]) * root_weight
    weighted_w = np.asarray(model["covariates"]) * root_weight[:, None]
    weighted_x = np.asarray(model["genotypes"]) * root_weight[:, None]
    return {
        **model,
        "kinship": weighted_k,
        "covariates": weighted_w,
        "genotypes": weighted_x,
        "phenotype": weighted_y,
    }


def fixed_lambda_differences() -> dict[str, float]:
    """Compare dense GLS at GEMMA's reported per-marker REML lambda."""
    source, _ = require_reference()
    model = _weighted_model()
    k, w, x, y = (
        model[key] for key in ("kinship", "covariates", "genotypes", "phenotype")
    )
    rows = list(csv.DictReader((source / "gemma.assoc.txt").open(), delimiter="\t"))
    beta = []
    se = []
    p_wald = []
    for index, row in enumerate(rows):
        fit = evaluate(k, w, x[:, index], y, float(row["l_remle"]))
        beta.append(abs(fit["beta"] - float(row["beta"])))
        se.append(abs(fit["se"] - float(row["se"])))
        p_wald.append(abs(fit["p_wald"] - float(row["p_wald"])))
    return {"beta": max(beta), "se": max(se), "p_wald": max(p_wald)}


def write_oracle(path: Path, *, model_override: dict | None = None) -> None:
    source, _ = require_reference()
    model = _weighted_model(model_override)
    bim = {
        fields[1]: fields
        for line in (source / "tiny.bim").read_text().splitlines()
        if (fields := line.split())
    }

    def metadata(index, snp):
        if snp not in bim:
            raise ValueError(f"selected model SNP is absent from raw BIM: {snp}")
        fields = bim[snp]
        return {
            "chr": fields[0],
            "rs": snp,
            "ps": fields[3],
            "n_miss": model["selected_n_miss"][index],
            "allele1": fields[4],
            "allele0": fields[5],
            "af": model["selected_af"][index] / 2,
        }

    write_oracle_assoc(model, path, 4, metadata=metadata)


def _l_mle_extrema(actual: Path, reference: Path) -> dict:
    actual_rows = read_rows(actual, 4)
    reference_rows = read_rows(reference, 4)
    values = []
    for observed, expected in zip(actual_rows, reference_rows, strict=True):
        actual_value = float(observed["l_mle"])
        reference_value = float(expected["l_mle"])
        absolute = abs(actual_value - reference_value)
        values.append(
            {
                "rs": expected["rs"],
                "actual": actual_value,
                "reference": reference_value,
                "absolute": absolute,
                "relative_to_reference": absolute / abs(reference_value),
            }
        )
    return {
        "max_absolute": max(values, key=lambda value: value["absolute"]),
        "max_relative": max(values, key=lambda value: value["relative_to_reference"]),
    }


def compare_weights(destination: Path, case_ids: tuple[str, ...] | None = None) -> dict:
    """Compare declared weighted routes, recording default and refined results."""
    declared = [*load_weight_cases(), *load_nonpositive_weight_cases()]
    requested = {case["id"] for case in declared} if case_ids is None else set(case_ids)
    if not requested or requested - {case["id"] for case in declared}:
        raise ValueError("case_ids must name at least one declared weight case")
    destination.mkdir(parents=True, exist_ok=False)
    bundle = {
        "schema_version": 1,
        "status": "INCONCLUSIVE",
        "environment": environment(),
        "invocation": sys.argv,
        "references": {},
        "cases": [],
    }
    refined_statuses, default_statuses, default_gaps = [], [], []
    for case in declared:
        if case["id"] not in requested:
            continue
        nonpositive = case.get("nonpositive", False)
        source, provenance = (
            require_nonpositive_reference() if nonpositive else require_reference()
        )
        if source.name not in bundle["references"]:
            copy_reference(source, destination / source.name, provenance)
            bundle["references"][source.name] = provenance
        model = json.loads((source / "model.json").read_text())
        if not nonpositive and "dense" not in bundle:
            oracle_path = destination / "oracle.assoc.txt"
            write_oracle(oracle_path)
            oracle_gemma = compare_files(
                oracle_path,
                source / "gemma.assoc.txt",
                mode=4,
                reference_optional_logl=True,
            )
            bundle["dense"] = {
                "optimized_gemma": oracle_gemma,
                "fixed_lambda_max_abs": fixed_lambda_differences(),
            }
            refined_statuses.append(oracle_gemma["status"])
        fam_ids = [
            f"{fields[0]}:{fields[1]}"
            for line in (source / "tiny.fam").read_text().splitlines()
            if (fields := line.split())
        ]
        runs = []
        refinements = (
            ((30, "refined"),) if nonpositive else ((20, "default"), (30, "refined"))
        )
        for n_refine, label in refinements:
            out = destination / case["id"] / label
            result, logs, config = run_pipeline(
                source,
                out,
                kinship_file=source / "gemma_kinship.cXX.txt"
                if case["kinship"] == "supplied-k"
                else None,
                covariate_file=source / "covariates.txt",
                weight_file=source / "weights.txt",
                lmm_mode=4,
                maf=0.1,
                miss=0.1,
                n_refine=n_refine,
                backend=case["backend"],
            )
            comparison = compare_files(
                result.assoc_path,
                source / "gemma.assoc.txt",
                mode=4,
                reference_optional_logl=True,
            )
            actual_snp_ids = [row["rs"] for row in read_rows(result.assoc_path, 4)]
            actual_indices = result.analyzed_sample_indices
            actual_samples = [fam_ids[index] for index in actual_indices]
            stages = {
                "selected-sample-ids": actual_samples == model["selected_sample_ids"],
                "selected-snp-ids": actual_snp_ids == model["selected_snp_ids"],
                "sample-count": result.n_samples == len(model["selected_sample_ids"]),
                "snp-count": result.n_snps_tested == len(model["selected_snp_ids"]),
            }
            comparison["failure_ids"].extend(
                f"stage:actual-{stage}"
                for stage, passed in stages.items()
                if not passed
            )
            if not all(stages.values()):
                comparison["status"] = "NOT VERIFIED"
            run = {
                "label": label,
                "n_refine": n_refine,
                "status": comparison["status"],
                "comparison": comparison,
                "l_mle_extrema": _l_mle_extrema(
                    result.assoc_path, source / "gemma.assoc.txt"
                ),
                "pipeline_config": config,
                "actual": {
                    "n_samples": result.n_samples,
                    "n_snps": result.n_snps_tested,
                    "selected_snp_ids": actual_snp_ids,
                    "selected_sample_ids": actual_samples,
                    "valid_indices": actual_indices,
                },
                "logs": logs,
                "files": snapshot_files(out),
            }
            runs.append(run)
            (default_statuses if label == "default" else refined_statuses).append(
                comparison["status"]
            )
            if label == "default" and comparison["status"] != "VERIFIED":
                default_gaps.append(
                    {
                        "case": case["id"],
                        "failure_ids": comparison["failure_ids"],
                        "l_mle_extrema": run["l_mle_extrema"],
                    }
                )
        bundle["cases"].append({**case, "runs": runs})
    bundle["status"] = bundle_status(refined_statuses)
    if default_statuses:
        bundle["default_optimizer_status"] = bundle_status(default_statuses)
        bundle["default_optimizer_gaps"] = default_gaps
    write_json(destination / "bundle.json", bundle)
    return bundle
