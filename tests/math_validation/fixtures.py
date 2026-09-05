"""Independent raw fixture construction and external GEMMA provenance.

This module deliberately imports no JAMMA code. Fixture generation is a separate
command from comparison; the latter only reads hash-verified reference files.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

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


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_reference(source, destination, provenance):
    """Keep raw external inputs and outputs inside a self-contained run bundle."""
    destination.mkdir(parents=True, exist_ok=False)
    for name in (*provenance["files"], "provenance.json"):
        if Path(name).name != name:
            raise ValueError(f"unsafe reference member: {name}")
        shutil.copy2(source / name, destination / name)


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    if manifest["schema_version"] != 1 or manifest["reference_version"] != "0.98.5":
        raise ValueError("unsupported case schema or GEMMA version")
    ids = []
    for case in manifest["cases"]:
        if not re.fullmatch(r"[a-z0-9-]+", case["id"]):
            raise ValueError("case ID must be a safe directory name")
        if case["recipe"] != "tiny-plink-v1" or case["mode"] not in EXTERNAL_HEADERS:
            raise ValueError("unsupported tiny recipe or LMM mode")
        if not case["reason"] or not 4 <= case["n_samples"] <= 100:
            raise ValueError("tiny case needs a reason and 4..100 samples")
        if not 1 <= case["n_snps"] <= 20:
            raise ValueError("tiny case requires 1..20 SNPs")
        if not 1 <= case.get("n_covariates", 1) <= 3:
            raise ValueError("tiny case requires 1..3 covariates including intercept")
        ids.append(case["id"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("case IDs must be nonempty and unique")
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
    prefix = directory / "tiny"
    prefix.with_suffix(".fam").write_text(
        "".join(f"F{i}\tI{i}\t0\t0\t0\t{y[i]:.17g}\n" for i in range(n))
    )
    prefix.with_suffix(".bim").write_text(
        "".join(f"1\tsnp{j}\t0\t{100 + j}\tA\tG\n" for j in range(m))
    )
    bed = bytearray([0x6C, 0x1B, 0x01])
    for j in range(m):
        for start in range(0, n, 4):
            byte = 0
            for offset, dosage in enumerate(x[start : start + 4, j]):
                byte |= {0: 3, 1: 2, 2: 0}[int(dosage)] << (2 * offset)
            bed.append(byte)
    prefix.with_suffix(".bed").write_bytes(bed)
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


def run_command(argv, directory, name):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        argv,
        cwd=directory,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    (directory / f"{name}.stdout.txt").write_text(result.stdout)
    (directory / f"{name}.stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; see {directory}")
    return {"argv": argv, "cwd": str(directory), "exit_code": result.returncode}


def generate_reference(manifest, destination, gemma):
    binary = Path(shutil.which(str(gemma)) or gemma).expanduser().resolve(strict=True)
    version = subprocess.run(
        [str(binary), "-h"], capture_output=True, text=True, check=True, timeout=15
    ).stdout
    if not re.search(r"GEMMA\s+0\.98\.5(?:\s|$)", version):
        raise ValueError("reference executable must identify itself as GEMMA 0.98.5")
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
        provenance = {
            "schema_version": 1,
            "executable": str(binary),
            "executable_sha256": digest(binary),
            "version": version,
            "command": command,
            "case": case,
            "generator_sha256": digest(Path(__file__)),
            "files": {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
        }
        (directory / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )


def verify_reference(case, root=REFERENCE):
    directory = root / case["id"]
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["case"] != case:
        raise ValueError(
            "case differs from reference provenance; regenerate separately"
        )
    for name, expected in provenance["files"].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError(f"reference hash mismatch: {name}")
    return directory, provenance
