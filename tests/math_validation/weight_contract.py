"""External GEMMA contract for per-individual residual weights."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.mathematical_validation import (
    compare_files,
    environment,
    read_rows,
    write_json,
)
from tests.math_validation.dense_oracle import all_test_statistics, evaluate
from tests.math_validation.fixtures import (
    EXTERNAL_HEADERS,
    copy_reference,
    digest,
    run_command,
)

if TYPE_CHECKING:
    from jamma.pipeline_config import BackendRequest

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
    return [{"id": "nonpositive-supplied-k-numpy"}]


def _generate_positive_reference(destination: Path, gemma: Path | str) -> None:
    """Generate the positive-weight case inside an absent directory."""
    from tests.math_validation.pipeline_cases import (
        _arrays,
        _selected_model,
        _write_plink,
    )

    binary = Path(shutil.which(str(gemma)) or gemma).expanduser().resolve(strict=True)
    version_run = subprocess.run(
        [str(binary), "-h"], capture_output=True, text=True, check=True, timeout=15
    )
    version = version_run.stdout + version_run.stderr
    if not re.search(r"GEMMA\s+0\.98\.5(?:\s|$)", version):
        raise ValueError("reference executable must identify itself as GEMMA 0.98.5")
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
    provenance = {
        "schema_version": 1,
        "case": "mode4-missing-covariates",
        "gemma": {
            "version": "0.98.5",
            "binary": str(binary),
            "binary_sha256": digest(binary),
            "source_repository": "https://github.com/genetics-statistics/GEMMA",
            "source_revision": "c37b0445f820b682836a1d20009ce1817546493a",
        },
        "commands": {"kinship": kinship_run, "association": association_run},
        "files": {
            path.name: digest(path)
            for path in sorted(destination.iterdir())
            if path.is_file()
        },
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
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
    provenance["files"] = {
        path.name: digest(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "provenance.json"
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


def generate_external(destination: Path, gemma: Path | str) -> None:
    """Generate positive and nonpositive immutable weighted references."""
    destination.mkdir(parents=True, exist_ok=False)
    _generate_positive_reference(destination / REFERENCE.name, gemma)
    generate_nonpositive_reference(destination / NONPOSITIVE_REFERENCE.name, gemma)


def require_reference() -> tuple[Path, dict]:
    provenance = json.loads((REFERENCE / "provenance.json").read_text())
    for name, expected in provenance["files"].items():
        if Path(name).name != name or digest(REFERENCE / name) != expected:
            raise ValueError(f"weight reference hash mismatch: {name}")
    return REFERENCE, provenance


def require_nonpositive_reference() -> tuple[Path, dict]:
    provenance = json.loads((NONPOSITIVE_REFERENCE / "provenance.json").read_text())
    for name, expected in provenance["files"].items():
        if Path(name).name != name or digest(NONPOSITIVE_REFERENCE / name) != expected:
            raise ValueError(f"nonpositive weight reference hash mismatch: {name}")
    return NONPOSITIVE_REFERENCE, provenance


def compare_nonpositive_weight_contract(destination: Path) -> dict:
    """Compare GEMMA's zero/negative ``-widv`` branch through the pipeline."""
    from jamma.pipeline import PipelineRunner
    from jamma.pipeline_config import PipelineConfig

    source, _ = require_nonpositive_reference()
    result = PipelineRunner(
        PipelineConfig(
            bfile=source / "tiny",
            kinship_file=source / "gemma_kinship.cXX.txt",
            covariate_file=source / "covariates.txt",
            weight_file=source / "weights.txt",
            lmm_mode=4,
            maf=0.1,
            miss=0.1,
            n_refine=30,
            backend="numpy",
            output_dir=destination,
            output_prefix="jamma",
            legacy_text=True,
            check_memory=False,
            show_progress=False,
            no_telemetry=True,
        )
    ).run()
    return compare_files(
        result.assoc_path,
        source / "gemma.assoc.txt",
        af_contract="counted-allele",
        mode=4,
        reference_optional_logl=True,
    )


def _weighted_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    source, _ = require_reference()
    model = json.loads((source / "model.json").read_text())
    valid = np.asarray(model["valid_mask"])
    k = np.loadtxt(source / "gemma_kinship.cXX.txt")[np.ix_(valid, valid)]
    k = k - k.mean(axis=0)[None, :] - k.mean(axis=1)[:, None] + k.mean()
    root_weight = np.sqrt(np.loadtxt(source / "weights.txt")[valid])
    weighted_k = k / root_weight[:, None] / root_weight[None, :]
    weighted_y = np.asarray(model["phenotype"]) * root_weight
    weighted_w = np.asarray(model["covariates"]) * root_weight[:, None]
    weighted_x = np.asarray(model["genotypes"]) * root_weight[:, None]
    return weighted_k, weighted_w, weighted_x, weighted_y, model


def fixed_lambda_differences() -> dict[str, float]:
    """Compare dense GLS at GEMMA's reported per-marker REML lambda."""
    source, _ = require_reference()
    k, w, x, y, _ = _weighted_model()
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
    k, w, x, y, model = _weighted_model()
    if model_override is not None:
        model = model_override
        valid = np.asarray(model["valid_mask"])
        root_weight = np.sqrt(np.loadtxt(source / "weights.txt")[valid])
        x = np.asarray(model["genotypes"]) * root_weight[:, None]
        y = np.asarray(model["phenotype"]) * root_weight
        w = np.asarray(model["covariates"]) * root_weight[:, None]
    bim = {
        fields[1]: fields
        for line in (source / "tiny.bim").read_text().splitlines()
        if (fields := line.split())
    }
    headers = EXTERNAL_HEADERS[4]
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for index, snp in enumerate(model["selected_snp_ids"]):
            if snp not in bim:
                raise ValueError(f"selected model SNP is absent from raw BIM: {snp}")
            fit = all_test_statistics(k, w, x[:, index], y)
            fields = bim[snp]
            row = {
                "chr": fields[0],
                "rs": snp,
                "ps": fields[3],
                "n_miss": model["selected_n_miss"][index],
                "allele1": fields[4],
                "allele0": fields[5],
                "af": model["selected_af"][index] / 2,
            }
            for field in (
                "beta",
                "se",
                "logl_H1",
                "l_remle",
                "p_wald",
                "l_mle",
                "p_lrt",
                "p_score",
            ):
                row[field] = fit[field]
            writer.writerow(row)


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


def compare_weight_contract(
    destination: Path,
    backend: BackendRequest = "numpy",
    *,
    supplied_kinship: bool = False,
    n_refine: int = 30,
) -> dict:
    """Run JAMMA's real pipeline and compare every mode-4 output field."""
    from jamma.pipeline import PipelineRunner
    from jamma.pipeline_config import PipelineConfig

    source, _ = require_reference()
    destination.mkdir(parents=True, exist_ok=False)
    result = PipelineRunner(
        PipelineConfig(
            bfile=source / "tiny",
            kinship_file=(
                source / "gemma_kinship.cXX.txt" if supplied_kinship else None
            ),
            covariate_file=source / "covariates.txt",
            weight_file=source / "weights.txt",
            lmm_mode=4,
            maf=0.1,
            miss=0.1,
            n_refine=n_refine,
            backend=backend,
            output_dir=destination,
            output_prefix="jamma",
            legacy_text=True,
            check_memory=False,
            show_progress=False,
            no_telemetry=True,
        )
    ).run()
    return compare_files(
        result.assoc_path,
        source / "gemma.assoc.txt",
        af_contract="counted-allele",
        mode=4,
        reference_optional_logl=True,
    )


@contextmanager
def _capture_logs():
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink)


@contextmanager
def _observe_pipeline_selection():
    """Capture the sample indices used by the real kinship pipeline seam."""
    observed: dict[str, np.ndarray] = {}
    prior = sys.getprofile()

    def profile(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "_load_kinship_from_source":
            indices = frame.f_locals["valid_indices"]
            observed["valid_indices"] = (
                np.arange(frame.f_locals["n_samples"])
                if indices is None
                else np.array(indices, copy=True)
            )

    sys.setprofile(profile)
    try:
        yield observed
    finally:
        sys.setprofile(prior)


def compare_weights(destination: Path) -> dict:
    """Build a self-contained four-route weighted evidence bundle."""
    from jamma.pipeline import PipelineRunner
    from jamma.pipeline_config import PipelineConfig

    source, provenance = require_reference()
    nonpositive_source, nonpositive_provenance = require_nonpositive_reference()
    destination.mkdir(parents=True, exist_ok=False)
    reference_copy = destination / "reference"
    copy_reference(source, reference_copy, provenance)
    nonpositive_copy = destination / "nonpositive-reference"
    copy_reference(nonpositive_source, nonpositive_copy, nonpositive_provenance)
    nonpositive_comparison = compare_nonpositive_weight_contract(
        destination / "nonpositive-supplied-k-numpy"
    )
    oracle_path = destination / "oracle.assoc.txt"
    write_oracle(oracle_path)
    oracle_gemma = compare_files(
        oracle_path,
        source / "gemma.assoc.txt",
        af_contract="counted-allele",
        mode=4,
        reference_optional_logl=True,
    )
    cases = []
    model = json.loads((source / "model.json").read_text())
    fam_ids = [
        f"{fields[0]}:{fields[1]}"
        for line in (source / "tiny.fam").read_text().splitlines()
        if (fields := line.split())
    ]
    refined_statuses = [oracle_gemma["status"], nonpositive_comparison["status"]]
    default_statuses = []
    default_optimizer_gaps = []
    for backend in ("numpy", "numpy-streaming"):
        typed_backend: BackendRequest = backend
        for supplied_kinship in (False, True):
            case_id = f"{'supplied-k' if supplied_kinship else 'computed-k'}-{backend}"
            runs = []
            for n_refine, label in ((20, "default"), (30, "refined")):
                out = destination / case_id / label
                config = PipelineConfig(
                    bfile=source / "tiny",
                    kinship_file=(
                        source / "gemma_kinship.cXX.txt" if supplied_kinship else None
                    ),
                    covariate_file=source / "covariates.txt",
                    weight_file=source / "weights.txt",
                    lmm_mode=4,
                    maf=0.1,
                    miss=0.1,
                    n_refine=n_refine,
                    backend=typed_backend,
                    output_dir=out,
                    output_prefix="jamma",
                    legacy_text=True,
                    check_memory=False,
                    show_progress=False,
                    no_telemetry=True,
                )
                with _capture_logs() as logs, _observe_pipeline_selection() as observed:
                    result = PipelineRunner(config).run()
                comparison = compare_files(
                    result.assoc_path,
                    source / "gemma.assoc.txt",
                    af_contract="counted-allele",
                    mode=4,
                    reference_optional_logl=True,
                )
                rows = read_rows(result.assoc_path, 4)
                actual_snp_ids = [row["rs"] for row in rows]
                actual_indices = observed.get("valid_indices", np.array([], dtype=int))
                actual_sample_ids = [fam_ids[index] for index in actual_indices]
                stage_failures = []
                if actual_sample_ids != model["selected_sample_ids"]:
                    stage_failures.append("stage:actual-selected-sample-ids")
                if actual_snp_ids != model["selected_snp_ids"]:
                    stage_failures.append("stage:actual-selected-snp-ids")
                if result.n_samples != len(model["selected_sample_ids"]):
                    stage_failures.append("stage:actual-sample-count")
                if result.n_snps_tested != len(model["selected_snp_ids"]):
                    stage_failures.append("stage:actual-snp-count")
                if stage_failures:
                    comparison["status"] = "NOT VERIFIED"
                    comparison["failure_ids"].extend(stage_failures)
                run = {
                    "label": label,
                    "n_refine": n_refine,
                    "status": comparison["status"],
                    "comparison": comparison,
                    "l_mle_extrema": _l_mle_extrema(
                        result.assoc_path, source / "gemma.assoc.txt"
                    ),
                    "pipeline_config": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in asdict(config).items()
                    },
                    "actual": {
                        "n_samples": result.n_samples,
                        "n_snps": result.n_snps_tested,
                        "selected_snp_ids": actual_snp_ids,
                        "selected_sample_ids": actual_sample_ids,
                        "valid_indices": actual_indices,
                    },
                    "logs": logs,
                    "files": {
                        str(path.relative_to(out)): digest(path)
                        for path in sorted(out.rglob("*"))
                        if path.is_file()
                    },
                }
                runs.append(run)
                (default_statuses if label == "default" else refined_statuses).append(
                    comparison["status"]
                )
                if label == "default" and comparison["status"] != "VERIFIED":
                    default_optimizer_gaps.append(
                        {
                            "case": case_id,
                            "failure_ids": comparison["failure_ids"],
                            "l_mle_extrema": run["l_mle_extrema"],
                        }
                    )
            cases.append(
                {
                    "id": case_id,
                    "backend": backend,
                    "kinship": "supplied" if supplied_kinship else "computed",
                    "runs": runs,
                }
            )
    bundle = {
        "schema_version": 1,
        "status": (
            "VERIFIED"
            if refined_statuses
            and all(status == "VERIFIED" for status in refined_statuses)
            else "NOT VERIFIED"
        ),
        "default_optimizer_status": (
            "VERIFIED"
            if default_statuses
            and all(status == "VERIFIED" for status in default_statuses)
            else "NOT VERIFIED"
        ),
        "default_optimizer_gaps": default_optimizer_gaps,
        "bounded_limitations": (
            [
                "The default n_refine=20 failures are recorded with separate "
                "maximum-absolute and maximum-relative rows; n_refine=30 "
                "verifies without changing tolerances."
            ]
            if default_optimizer_gaps
            else []
        ),
        "environment": environment(),
        "invocation": sys.argv,
        "reference": {
            "source": str(source),
            "copied_to": str(reference_copy),
            "provenance": provenance,
            "files": provenance["files"],
        },
        "nonpositive_reference": {
            "source": str(nonpositive_source),
            "copied_to": str(nonpositive_copy),
            "provenance": nonpositive_provenance,
            "files": nonpositive_provenance["files"],
            "comparison": nonpositive_comparison,
        },
        "source_hashes": {
            "weight_contract": digest(Path(__file__)),
            "pipeline": digest(ROOT / "src/jamma/pipeline.py"),
            "weight_io": digest(ROOT / "src/jamma/io/weight.py"),
        },
        "dense": {
            "optimized_gemma": oracle_gemma,
            "fixed_lambda_max_abs": fixed_lambda_differences(),
        },
        "cases": cases,
    }
    write_json(destination / "bundle.json", bundle)
    return bundle
