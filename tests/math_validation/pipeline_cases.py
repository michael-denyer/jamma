"""Independent raw-input fixtures and Phase 2 pipeline comparisons."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.mathematical_validation import (
    compare_files,
    environment,
    read_rows,
    write_json,
)
from tests.math_validation.dense_oracle import all_test_statistics, optimize
from tests.math_validation.fixtures import (
    EXTERNAL_HEADERS,
    copy_reference,
    digest,
    run_command,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("pipeline_manifest.json")
REFERENCE = ROOT / "tests/fixtures/mathematical_pipeline"


def load_pipeline_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != 1 or manifest["reference_version"] != "0.98.5":
        raise ValueError("unsupported pipeline manifest")
    ids = [case["id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"[a-z0-9-]+", i) for i in ids
    ):
        raise ValueError("pipeline case IDs must be unique safe directory names")
    if {case["mode"] for case in manifest["cases"]} != set(EXTERNAL_HEADERS):
        raise ValueError("pipeline manifest must contain modes 1 through 4")
    return manifest


def _arrays() -> dict[str, np.ndarray | list[str]]:
    """Make boundary cases explicit so fixture intent survives RNG changes."""
    n, m = 40, 12
    rng = np.random.default_rng(20260905)
    x = np.tile(np.array([0.0, 1.0, 2.0, 1.0]), (n, 3))
    rng.shuffle(x, axis=0)
    analysed = np.arange(2, n)
    # MAF is calculated over the 38 analysed rows. Seven copies fails 0.1;
    # eight copies passes. Missingness 3/38 passes and 4/38 fails.
    x[:, 0] = 0
    x[analysed[:7], 0] = 1
    x[:2, 0] = 2
    x[:, 1] = 0
    x[analysed[:8], 1] = 1
    x[:, 2] = np.resize([0, 1, 2, 1], n)
    x[analysed[:3], 2] = np.nan
    x[:, 3] = np.resize([0, 1, 2, 1], n)
    x[analysed[:4], 3] = np.nan
    x[:, 4] = 1
    for j in range(5, m):
        x[:, j] = rng.integers(0, 3, size=n)
    cov = np.column_stack((np.ones(n), np.linspace(-1.5, 1.5, n)))
    phenotype = 0.65 * np.nan_to_num(x[:, 5], nan=1.0) - 0.3 * cov[:, 1]
    phenotype += rng.normal(0, 0.7, n)
    phenotype[0] = -9
    cov[1, 1] = np.nan
    return {
        "genotypes": x,
        "phenotype": phenotype,
        "covariates": cov,
        "sample_ids": [f"F{i}:I{i}" for i in range(n)],
        "snp_ids": [f"boundary{j}" for j in range(m)],
    }


def _write_plink(directory: Path, arrays: dict) -> None:
    prefix = directory / "tiny"
    x = arrays["genotypes"]
    y = arrays["phenotype"]
    prefix.with_suffix(".fam").write_text(
        "".join(f"F{i}\tI{i}\t0\t0\t0\t{float(y[i]):.17g}\n" for i in range(len(y)))
    )
    prefix.with_suffix(".bim").write_text(
        "".join(f"1\tboundary{j}\t0\t{100 + j}\tA\tG\n" for j in range(x.shape[1]))
    )
    bed = bytearray([0x6C, 0x1B, 0x01])
    codes = {0: 3, 1: 2, 2: 0}
    for j in range(x.shape[1]):
        for start in range(0, x.shape[0], 4):
            byte = 0
            for offset, dosage in enumerate(x[start : start + 4, j]):
                code = 1 if np.isnan(dosage) else codes[int(dosage)]
                byte |= code << (2 * offset)
            bed.append(byte)
    prefix.with_suffix(".bed").write_bytes(bed)
    covariates = np.asarray(arrays["covariates"])
    (directory / "covariates.txt").write_text(
        "".join(
            "\t".join("NA" if np.isnan(value) else f"{value:.17g}" for value in row)
            + "\n"
            for row in covariates
        )
    )


def _double_center(matrix: np.ndarray) -> np.ndarray:
    centered = matrix.copy()
    row_mean = centered.mean(axis=1)
    grand_mean = centered.mean()
    centered -= row_mean[:, None]
    centered -= row_mean[None, :]
    centered += grand_mean
    return centered


def _selected_model(
    arrays: dict, gemma_kinship: np.ndarray, maf: float, miss: float
) -> dict:
    x = np.asarray(arrays["genotypes"], dtype=float)
    y = np.asarray(arrays["phenotype"], dtype=float)
    w = np.asarray(arrays["covariates"], dtype=float)
    valid = (y != -9) & np.isfinite(y) & np.all(np.isfinite(w), axis=1)
    xa = x[valid]
    af = np.nanmean(xa, axis=0) / 2
    selected = (
        (np.minimum(af, 1 - af) >= maf)
        & (np.isnan(xa).mean(axis=0) <= miss)
        & (np.nanvar(xa, axis=0) > 0)
    )
    association_means = np.nanmean(xa[:, selected], axis=0)
    imputed = np.where(np.isnan(xa[:, selected]), association_means, xa[:, selected])
    full_columns = x[:, selected]
    kinship_means = np.nanmean(full_columns, axis=0)
    centered_columns = (
        np.where(np.isnan(full_columns), kinship_means, full_columns) - kinship_means
    )
    raw_full_k = centered_columns @ centered_columns.T / selected.sum()
    analysed_k = _double_center(raw_full_k[np.ix_(valid, valid)])
    gemma_analysed_k = _double_center(gemma_kinship[np.ix_(valid, valid)])
    if not np.allclose(analysed_k, gemma_analysed_k, rtol=1e-8, atol=1e-10):
        raise ValueError("independent raw-array kinship does not match GEMMA")
    return {
        "valid_mask": valid.tolist(),
        "selected_mask": selected.tolist(),
        "selected_sample_ids": [
            s for s, keep in zip(arrays["sample_ids"], valid, strict=True) if keep
        ],
        "selected_snp_ids": [
            s for s, keep in zip(arrays["snp_ids"], selected, strict=True) if keep
        ],
        "association_imputation_means": association_means.tolist(),
        "kinship_imputation_means": kinship_means.tolist(),
        "selected_n_miss": np.isnan(xa[:, selected]).sum(axis=0).tolist(),
        "selected_af": np.nanmean(xa[:, selected], axis=0).tolist(),
        "genotypes": imputed.tolist(),
        "phenotype": y[valid].tolist(),
        "covariates": w[valid].tolist(),
        "kinship": analysed_k.tolist(),
        "gemma_centered_analysis_kinship": gemma_analysed_k.tolist(),
    }


def generate_external(destination: Path, gemma: Path | str) -> None:
    """Generate immutable references; comparison never calls this function."""
    manifest = load_pipeline_manifest()
    binary = Path(shutil.which(str(gemma)) or gemma).expanduser().resolve(strict=True)
    version_run = subprocess.run(
        [str(binary), "-h"], capture_output=True, text=True, check=True, timeout=15
    )
    version = version_run.stdout + version_run.stderr
    if not re.search(r"GEMMA\s+0\.98\.5(?:\s|$)", version):
        raise ValueError("reference executable must identify itself as GEMMA 0.98.5")
    destination.mkdir(parents=True, exist_ok=False)
    arrays = _arrays()
    for case in manifest["cases"]:
        directory = destination / case["id"]
        directory.mkdir()
        _write_plink(directory, arrays)
        gk = run_command(
            [
                str(binary),
                "-bfile",
                "tiny",
                "-c",
                "covariates.txt",
                "-gk",
                "1",
                "-maf",
                str(manifest["maf"]),
                "-miss",
                str(manifest["miss"]),
                "-outdir",
                ".",
                "-o",
                "gemma_kinship",
            ],
            directory,
            "gemma_kinship",
        )
        kinship_path = directory / "gemma_kinship.cXX.txt"
        if not kinship_path.is_file():
            raise RuntimeError("GEMMA did not produce centered kinship")
        kinship = np.loadtxt(kinship_path)
        model = _selected_model(arrays, kinship, manifest["maf"], manifest["miss"])
        (directory / "model.json").write_text(json.dumps(model, indent=2) + "\n")
        assoc = run_command(
            [
                str(binary),
                "-bfile",
                "tiny",
                "-k",
                "gemma_kinship.cXX.txt",
                "-c",
                "covariates.txt",
                "-lmm",
                str(case["mode"]),
                "-maf",
                str(manifest["maf"]),
                "-miss",
                str(manifest["miss"]),
                "-outdir",
                ".",
                "-o",
                "gemma",
            ],
            directory,
            "gemma",
        )
        provenance = {
            "schema_version": 1,
            "executable": str(binary),
            "executable_sha256": digest(binary),
            "version": version,
            "case": case,
            "manifest": manifest,
            "commands": {"kinship": gk, "association": assoc},
            "generator_sha256": digest(Path(__file__)),
            "files": {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
        }
        (directory / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )


def require_pipeline_reference(case: dict, root: Path = REFERENCE) -> tuple[Path, dict]:
    directory = root / case["id"]
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["case"] != case or provenance["manifest"] != load_pipeline_manifest():
        raise ValueError("pipeline case differs from immutable provenance")
    for name, expected in provenance["files"].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError(f"pipeline reference hash mismatch: {name}")
    return directory, provenance


def _oracle_rows(model: dict, mode: int, destination: Path) -> list[dict]:
    import csv

    k = np.asarray(model["kinship"])
    x = np.asarray(model["genotypes"])
    y = np.asarray(model["phenotype"])
    w = np.asarray(model["covariates"])
    rows = []
    for j, snp in enumerate(model["selected_snp_ids"]):
        fit = optimize(k, w, x[:, j], y)
        if mode != 1:
            fit.update(all_test_statistics(k, w, x[:, j], y))
        if mode == 3:
            fit["beta"], fit["se"] = fit["score_beta"], fit["score_se"]
        row = {
            "chr": "1",
            "rs": snp,
            "ps": str(100 + int(snp.removeprefix("boundary"))),
            "n_miss": str(model["selected_n_miss"][j]),
            "allele1": "A",
            "allele0": "G",
            "af": model["selected_af"][j] / 2,
        }
        row.update({field: fit[field] for field in EXTERNAL_HEADERS[mode][7:]})
        rows.append(row)
    with destination.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=EXTERNAL_HEADERS[mode], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


@contextmanager
def _observe_pipeline_kinship():
    """Capture the matrix and sample indices returned to the real pipeline."""
    observed: dict[str, np.ndarray] = {}
    prior = sys.getprofile()

    def profile(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "_load_kinship_from_source":
            observed["kinship"] = np.array(arg, copy=True)
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


def compare_pipeline(
    destination: Path,
    backends: tuple[str, ...] | None = None,
    forced_global: bool | None = None,
    case_ids: tuple[str, ...] | None = None,
) -> dict:
    """Compare mandatory pipeline routes with immutable external evidence."""
    from loguru import logger

    from jamma.pipeline import PipelineRunner
    from jamma.pipeline_config import PipelineConfig

    manifest = load_pipeline_manifest()
    backends = ("numpy", "numpy-streaming") if backends is None else tuple(backends)
    if not backends or set(backends) - {"numpy", "numpy-streaming"}:
        raise ValueError("backends must name at least one supported pipeline route")
    selected_cases = [
        case for case in manifest["cases"] if case_ids is None or case["id"] in case_ids
    ]
    if not selected_cases or (
        case_ids is not None
        and {case["id"] for case in selected_cases} != set(case_ids)
    ):
        raise ValueError("case_ids must name at least one declared pipeline case")
    destination.mkdir(parents=True, exist_ok=False)
    bundle = {
        "schema_version": 1,
        "status": "INCONCLUSIVE",
        "environment": environment(),
        "manifest": manifest,
        "backends": list(backends),
        "forced_global": bool(
            forced_global or os.environ.get("JAMMA_FORCE_NUMPY_FALLBACK") == "1"
        ),
        "invocation": sys.argv,
        "cases": [],
        "untested": manifest["untested"],
    }
    statuses = []
    for case in selected_cases:
        source, provenance = require_pipeline_reference(case)
        copy_reference(source, destination / f"{case['id']}-reference", provenance)
        model = json.loads((source / "model.json").read_text())
        oracle_path = destination / f"{case['id']}.oracle.assoc.txt"
        _oracle_rows(model, case["mode"], oracle_path)
        external_oracle = compare_files(
            oracle_path,
            source / "gemma.assoc.txt",
            af_contract="counted-allele",
            mode=case["mode"],
            reference_optional_logl=True,
        )
        statuses.append(external_oracle["status"])
        runs = []
        for backend in backends:
            for save_kinship in (False, True):
                route = f"{backend}-save-{str(save_kinship).lower()}"
                out = destination / f"{case['id']}-{route}"
                config = PipelineConfig(
                    bfile=source / "tiny",
                    covariate_file=source / "covariates.txt",
                    lmm_mode=case["mode"],
                    maf=manifest["maf"],
                    miss=manifest["miss"],
                    backend="numpy" if backend == "numpy" else "numpy-streaming",
                    output_dir=out,
                    output_prefix="jamma",
                    save_kinship=save_kinship,
                    legacy_text=True,
                    check_memory=False,
                    show_progress=False,
                    no_telemetry=True,
                )
                messages: list[str] = []
                sink = logger.add(
                    lambda message, sink_messages=messages: sink_messages.append(
                        str(message)
                    ),
                    format="{message}",
                )
                try:
                    with _observe_pipeline_kinship() as observed:
                        result = PipelineRunner(config).run()
                finally:
                    logger.remove(sink)
                comparison = compare_files(
                    result.assoc_path,
                    source / "gemma.assoc.txt",
                    af_contract="counted-allele",
                    mode=case["mode"],
                    reference_optional_logl=True,
                )
                actual_ids = [
                    row["rs"] for row in read_rows(result.assoc_path, case["mode"])
                ]
                actual_indices = observed.get("valid_indices", np.array([], dtype=int))
                actual_samples = [f"F{i}:I{i}" for i in actual_indices]
                actual_k = observed.get("kinship")
                expected_k = np.asarray(model["kinship"])
                gemma_k = np.asarray(model["gemma_centered_analysis_kinship"])
                stage_ok = (
                    actual_k is not None
                    and actual_samples == model["selected_sample_ids"]
                    and actual_ids == model["selected_snp_ids"]
                    and result.n_samples == len(model["selected_sample_ids"])
                    and np.allclose(actual_k, expected_k, rtol=1e-8, atol=1e-10)
                    and np.allclose(actual_k, gemma_k, rtol=1e-8, atol=1e-10)
                )
                if not stage_ok:
                    comparison["status"] = "NOT VERIFIED"
                    comparison["failure_ids"].append(
                        "stage:actual-pipeline-samples-snps-or-kinship"
                    )
                statuses.append(comparison["status"])
                runs.append(
                    {
                        "route": route,
                        "backend": backend,
                        "save_kinship": save_kinship,
                        "pipeline_config": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in asdict(config).items()
                        },
                        "association": comparison,
                        "stage_boundaries": {
                            "actual_selected_sample_ids": actual_samples,
                            "actual_selected_snp_ids": actual_ids,
                            "actual_pipeline_centered_kinship": (
                                "VERIFIED" if stage_ok else "NOT VERIFIED"
                            ),
                            "reconstructed_association_imputation_means": model[
                                "association_imputation_means"
                            ],
                            "reconstructed_kinship_imputation_means": model[
                                "kinship_imputation_means"
                            ],
                        },
                        "logs": messages,
                        "files": {
                            str(path.relative_to(out)): digest(path)
                            for path in sorted(out.rglob("*"))
                            if path.is_file()
                        },
                    }
                )
        bundle["cases"].append(
            {
                "id": case["id"],
                "mode": case["mode"],
                "reference": {
                    "directory": str(source),
                    "provenance": provenance,
                    "files": {
                        path.name: digest(path)
                        for path in sorted(source.iterdir())
                        if path.is_file()
                    },
                },
                "oracle_gemma": external_oracle,
                "runs": runs,
            }
        )
    bundle["status"] = (
        "VERIFIED"
        if statuses and all(status == "VERIFIED" for status in statuses)
        else "NOT VERIFIED"
    )
    write_json(destination / "bundle.json", bundle)
    return bundle
