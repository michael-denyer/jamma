"""Independent raw-input fixtures and Phase 2 pipeline comparisons."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from tests.math_validation.compare import compare_files, read_rows
from tests.math_validation.evidence import (
    bundle_status,
    environment,
    run_pipeline,
    write_json,
)
from tests.math_validation.fixtures import (
    EXTERNAL_HEADERS,
)
from tests.math_validation.oracle_io import write_oracle_assoc
from tests.math_validation.reference import (
    copy_reference,
    digest,
    gemma_binary,
    run_command,
    snapshot_files,
    validate_manifest,
    verify_reference_dir,
    write_plink,
    write_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("pipeline_manifest.json")
REFERENCE = ROOT / "tests/fixtures/mathematical_pipeline"


def load_pipeline_manifest(path: Path = MANIFEST) -> dict:
    return validate_manifest(
        path, label="pipeline", required_modes=set(EXTERNAL_HEADERS)
    )


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
    write_plink(
        directory,
        genotypes=np.asarray(arrays["genotypes"]),
        phenotype=np.asarray(arrays["phenotype"]),
        snp_ids=arrays["snp_ids"],
        covariates=np.asarray(arrays["covariates"]),
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
    binary, version = gemma_binary(gemma)
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
        write_provenance(
            directory,
            schema_version=1,
            executable=str(binary),
            executable_sha256=digest(binary),
            version=version,
            case=case,
            manifest=manifest,
            commands={"kinship": gk, "association": assoc},
            generator_sha256=digest(Path(__file__)),
        )


def require_pipeline_reference(case: dict, root: Path = REFERENCE) -> tuple[Path, dict]:
    directory = root / case["id"]
    provenance = verify_reference_dir(
        directory,
        expected={"case": case, "manifest": load_pipeline_manifest()},
        label="pipeline reference",
    )
    return directory, provenance


def _oracle_rows(model: dict, mode: int, destination: Path) -> list[dict]:
    def metadata(j, snp):
        return {
            "chr": "1",
            "rs": snp,
            "ps": str(100 + int(snp.removeprefix("boundary"))),
            "n_miss": str(model["selected_n_miss"][j]),
            "allele1": "A",
            "allele0": "G",
            "af": model["selected_af"][j] / 2,
        }

    return write_oracle_assoc(model, destination, mode, metadata=metadata)


def compare_pipeline(
    destination: Path,
    backends: tuple[str, ...] | None = None,
    forced_global: bool | None = None,
    case_ids: tuple[str, ...] | None = None,
) -> dict:
    """Compare mandatory pipeline routes with immutable external evidence."""
    from jamma.lmm.eigen_io import read_eigen_files

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
                result, messages, serialized_config = run_pipeline(
                    source,
                    out,
                    covariate_file=source / "covariates.txt",
                    lmm_mode=case["mode"],
                    maf=manifest["maf"],
                    miss=manifest["miss"],
                    backend="numpy" if backend == "numpy" else "numpy-streaming",
                    save_kinship=save_kinship,
                    write_eigen=True,
                )
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
                actual_indices = np.asarray(result.analyzed_sample_indices)
                actual_samples = [f"F{i}:I{i}" for i in actual_indices]
                eigenvalues, eigenvectors = read_eigen_files(
                    out / "jamma.eigenD.txt",
                    out / "jamma.eigenU.txt",
                    n_samples=result.n_samples,
                )
                actual_k = (eigenvectors * eigenvalues) @ eigenvectors.T
                expected_k = np.asarray(model["kinship"])
                gemma_k = np.asarray(model["gemma_centered_analysis_kinship"])
                stage_ok = (
                    actual_samples == model["selected_sample_ids"]
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
                        "pipeline_config": serialized_config,
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
                        "files": snapshot_files(out),
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
    bundle["status"] = bundle_status(statuses)
    write_json(destination / "bundle.json", bundle)
    return bundle
