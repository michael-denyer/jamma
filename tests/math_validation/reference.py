"""Shared construction and verification for external GEMMA references.

This module deliberately imports no JAMMA code. Reference generation must stay
independent from the implementation it validates.
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
GEMMA_VERSION = "0.98.5"


def digest(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gemma_binary(candidate: Path | str) -> tuple[Path, str]:
    """Resolve GEMMA and require the version used by committed references."""
    binary = (
        Path(shutil.which(str(candidate)) or candidate)
        .expanduser()
        .resolve(strict=True)
    )
    run = subprocess.run(
        [str(binary), "-h"], capture_output=True, text=True, check=True, timeout=15
    )
    version = run.stdout + run.stderr
    if not re.search(rf"GEMMA\s+{re.escape(GEMMA_VERSION)}(?:\s|$)", version):
        raise ValueError(
            f"reference executable must identify itself as GEMMA {GEMMA_VERSION}"
        )
    return binary, version


def validate_manifest(
    path: Path,
    *,
    label: str,
    required_modes: set[int] | None = None,
) -> dict:
    manifest = json.loads(path.read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["reference_version"] != GEMMA_VERSION
    ):
        raise ValueError(f"unsupported {label} manifest")
    ids = [case["id"] for case in manifest["cases"]]
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(not re.fullmatch(r"[a-z0-9-]+", case_id) for case_id in ids)
    ):
        raise ValueError(
            f"{label} case IDs must be nonempty unique safe directory names"
        )
    if (
        required_modes is not None
        and {case["mode"] for case in manifest["cases"]} != required_modes
    ):
        raise ValueError(f"{label} manifest must contain modes 1 through 4")
    return manifest


def write_plink(
    directory: Path,
    *,
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    snp_ids: list[str],
    chromosomes: list[str] | None = None,
    covariates: np.ndarray | None = None,
) -> None:
    """Write a tiny PLINK v1 data set from explicit raw arrays."""
    x = np.asarray(genotypes)
    chromosomes = ["1"] * x.shape[1] if chromosomes is None else chromosomes
    prefix = directory / "tiny"
    prefix.with_suffix(".fam").write_text(
        "".join(
            f"F{i}\tI{i}\t0\t0\t0\t{float(value):.17g}\n"
            for i, value in enumerate(phenotype)
        )
    )
    prefix.with_suffix(".bim").write_text(
        "".join(
            f"{chromosome}\t{snp}\t0\t{100 + j}\tA\tG\n"
            for j, (chromosome, snp) in enumerate(
                zip(chromosomes, snp_ids, strict=True)
            )
        )
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
    if covariates is not None:
        (directory / "covariates.txt").write_text(
            "".join(
                "\t".join("NA" if np.isnan(value) else f"{value:.17g}" for value in row)
                + "\n"
                for row in np.asarray(covariates)
            )
        )


def run_command(argv: list[str], directory: Path, name: str) -> dict:
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


def snapshot_files(directory: Path, *, exclude: frozenset[str] = frozenset()) -> dict:
    return {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in exclude
    }


def write_provenance(directory: Path, **fields) -> dict:
    provenance = {
        **fields,
        "files": snapshot_files(directory, exclude=frozenset({"provenance.json"})),
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def verify_reference_dir(
    directory: Path, *, expected: dict | None = None, label: str = "reference"
) -> dict:
    provenance = json.loads((directory / "provenance.json").read_text())
    if expected is not None:
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise ValueError(f"{label} differs from immutable provenance")
    for name, expected_digest in provenance["files"].items():
        if Path(name).name != name or digest(directory / name) != expected_digest:
            raise ValueError(f"{label} hash mismatch: {name}")
    return provenance


def copy_reference(source: Path, destination: Path, provenance: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in (*provenance["files"], "provenance.json"):
        if Path(name).name != name:
            raise ValueError(f"unsafe reference member: {name}")
        shutil.copy2(source / name, destination / name)
