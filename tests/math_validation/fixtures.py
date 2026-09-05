"""Independent raw fixture construction and external GEMMA provenance.

This module deliberately imports no JAMMA code. Fixture generation is a separate
command from comparison; the latter only reads hash-verified reference files.
"""

import json
from pathlib import Path

import numpy as np

from tests.math_validation.reference import (
    digest,
    gemma_binary,
    run_command,
    validate_manifest,
    verify_reference_dir,
    write_plink,
    write_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("manifest.json")
REFERENCE = ROOT / "tests/fixtures/mathematical_validation"
WALD_HEADER = (
    "chr",
    "rs",
    "ps",
    "n_miss",
    "allele1",
    "allele0",
    "af",
    "beta",
    "se",
    "logl_H1",
    "l_remle",
    "p_wald",
)
EXTERNAL_HEADERS = {
    1: WALD_HEADER,
    2: (*WALD_HEADER[:7], "logl_H1", "l_mle", "p_lrt"),
    3: (*WALD_HEADER[:7], "beta", "se", "p_score"),
    4: (
        *WALD_HEADER[:7],
        "beta",
        "se",
        "logl_H1",
        "l_remle",
        "l_mle",
        "p_wald",
        "p_lrt",
        "p_score",
    ),
}


def load_manifest(path=MANIFEST):
    manifest = validate_manifest(Path(path), label="supplied-kinship")
    for case in manifest["cases"]:
        if case["recipe"] != "tiny-plink-v1" or case["mode"] not in EXTERNAL_HEADERS:
            raise ValueError("unsupported tiny recipe or LMM mode")
        if not case["reason"] or not 4 <= case["n_samples"] <= 100:
            raise ValueError("tiny case needs a reason and 4..100 samples")
        if not 1 <= case["n_snps"] <= 20:
            raise ValueError("tiny case requires 1..20 SNPs")
        if not 1 <= case.get("n_covariates", 1) <= 3:
            raise ValueError("tiny case requires 1..3 covariates including intercept")
    return manifest


def materialize(case, directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(case["seed"])
    n, m = case["n_samples"], case["n_snps"]
    x = rng.integers(0, 3, size=(n, m))
    a = rng.normal(size=(n, n))
    a -= a.mean(axis=0)
    k = a @ a.T / n
    y = 0.7 * x[:, 0] + rng.multivariate_normal(np.zeros(n), np.eye(n) + 2 * k)
    write_plink(
        directory,
        genotypes=x,
        phenotype=y,
        snp_ids=[f"snp{j}" for j in range(m)],
    )
    np.savetxt(directory / "kinship.txt", k, fmt="%.17g")
    # Explicit raw arrays let the oracle avoid JAMMA's PLINK loader.
    model = {
        "kinship": k.tolist(),
        "genotypes": x.tolist(),
        "phenotype": y.tolist(),
        "sample_ids": [f"F{i}:I{i}" for i in range(n)],
        "snp_ids": [f"snp{j}" for j in range(m)],
    }
    if case.get("n_covariates", 1) > 1:
        covariates = np.column_stack(
            [np.ones(n), rng.normal(size=(n, case["n_covariates"] - 1))]
        )
        np.savetxt(directory / "covariates.txt", covariates, fmt="%.17g")
        model["covariates"] = covariates.tolist()
    (directory / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    return model


def generate_reference(manifest, destination, gemma):
    binary, version = gemma_binary(gemma)
    destination.mkdir(parents=True, exist_ok=False)
    for case in manifest["cases"]:
        directory = destination / case["id"]
        materialize(case, directory)
        command = run_command(
            [
                str(binary),
                "-bfile",
                "tiny",
                "-k",
                "kinship.txt",
                *case["gemma_args"],
                "-outdir",
                ".",
                "-o",
                "gemma",
            ],
            directory,
            "gemma",
        )
        if not (directory / "gemma.assoc.txt").is_file():
            raise RuntimeError("GEMMA did not produce association output")
        write_provenance(
            directory,
            schema_version=1,
            executable=str(binary),
            executable_sha256=digest(binary),
            version=version,
            command=command,
            case=case,
            generator_sha256=digest(Path(__file__)),
        )


def verify_reference(case, root=REFERENCE):
    directory = root / case["id"]
    provenance = verify_reference_dir(
        directory, expected={"case": case}, label="reference"
    )
    return directory, provenance
