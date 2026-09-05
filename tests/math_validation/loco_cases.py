"""Independent GEMMA references for LOCO cold and warm cache routes."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.mathematical_validation import (
    compare_files,
    environment,
    read_rows,
    write_json,
)
from tests.math_validation.fixtures import (
    EXTERNAL_HEADERS,
    copy_reference,
    digest,
    run_command,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("loco_manifest.json")
REFERENCE = ROOT / "tests/fixtures/mathematical_loco"


def load_loco_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != 1 or manifest["reference_version"] != "0.98.5":
        raise ValueError("unsupported LOCO manifest")
    ids = [case["id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"[a-z0-9-]+", case_id) for case_id in ids
    ):
        raise ValueError("LOCO case IDs must be unique safe directory names")
    if {case["mode"] for case in manifest["cases"]} != set(EXTERNAL_HEADERS):
        raise ValueError("LOCO manifest must contain modes 1 through 4")
    return manifest


def _arrays() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(20260906)
    x = rng.integers(0, 3, size=(40, 9)).astype(float)
    # Keep every chromosome informative, including the singleton chromosome.
    x[:, 8] = np.resize([0, 1, 2, 1], 40)
    y = 0.7 * x[:, 0] - 0.45 * x[:, 8] + rng.normal(0, 0.8, 40)
    chromosomes = ["1"] * 4 + ["2"] * 4 + ["3"]
    snps = [f"loco{j}" for j in range(9)]
    return x, y, chromosomes, snps


def _write_inputs(directory: Path) -> dict:
    x, y, chromosomes, snps = _arrays()
    prefix = directory / "tiny"
    prefix.with_suffix(".fam").write_text(
        "".join(f"F{i}\tI{i}\t0\t0\t0\t{y[i]:.17g}\n" for i in range(len(y)))
    )
    prefix.with_suffix(".bim").write_text(
        "".join(
            f"{chromosome}\t{snp}\t0\t{100 + j}\tA\tG\n"
            for j, (chromosome, snp) in enumerate(zip(chromosomes, snps, strict=True))
        )
    )
    bed = bytearray([0x6C, 0x1B, 0x01])
    codes = {0: 3, 1: 2, 2: 0}
    for j in range(x.shape[1]):
        for start in range(0, x.shape[0], 4):
            byte = 0
            for offset, dosage in enumerate(x[start : start + 4, j]):
                byte |= codes[int(dosage)] << (2 * offset)
            bed.append(byte)
    prefix.with_suffix(".bed").write_bytes(bed)
    by_chr = {
        chromosome: [
            snp
            for snp, chrom in zip(snps, chromosomes, strict=True)
            if chrom == chromosome
        ]
        for chromosome in ("1", "2", "3")
    }
    for chromosome, chromosome_snps in by_chr.items():
        (directory / f"chr{chromosome}.snps.txt").write_text(
            "\n".join(chromosome_snps) + "\n"
        )
        outside = [snp for snp in snps if snp not in chromosome_snps]
        (directory / f"chr{chromosome}.ksnps.txt").write_text("\n".join(outside) + "\n")
    return {
        "genotypes": x.tolist(),
        "phenotype": y.tolist(),
        "chromosomes": chromosomes,
        "snp_ids": snps,
        "snps_by_chromosome": by_chr,
    }


def _independent_loco_kinship(model: dict, chromosome: str) -> np.ndarray:
    x = np.asarray(model["genotypes"])
    outside = np.array(
        [value != chromosome for value in model["chromosomes"]], dtype=bool
    )
    columns = x[:, outside]
    centered = columns - columns.mean(axis=0)
    return centered @ centered.T / columns.shape[1]


def _combine_outputs(directory: Path, mode: int) -> None:
    lines: list[str] = []
    for chromosome in ("1", "2", "3"):
        path = directory / f"gemma_chr{chromosome}.assoc.txt"
        rows = path.read_text().splitlines()
        if tuple(rows[0].split("\t")) != EXTERNAL_HEADERS[mode]:
            raise ValueError(f"unexpected GEMMA mode {mode} header for chr{chromosome}")
        if len(rows) - 1 != len(
            json.loads((directory / "model.json").read_text())["snps_by_chromosome"][
                chromosome
            ]
        ):
            raise ValueError(f"GEMMA omitted a declared chromosome-{chromosome} SNP")
        if not lines:
            lines.append(rows[0])
        lines.extend(rows[1:])
    (directory / "gemma.assoc.txt").write_text("\n".join(lines) + "\n")


def generate_external(destination: Path, gemma: Path | str) -> None:
    """Generate per-chromosome GEMMA kinships and association references."""
    manifest = load_loco_manifest()
    binary = Path(shutil.which(str(gemma)) or gemma).expanduser().resolve(strict=True)
    version_run = subprocess.run(
        [str(binary), "-h"], capture_output=True, text=True, check=True, timeout=15
    )
    version = version_run.stdout + version_run.stderr
    if not re.search(r"GEMMA\s+0\.98\.5(?:\s|$)", version):
        raise ValueError("reference executable must identify itself as GEMMA 0.98.5")
    destination.mkdir(parents=True, exist_ok=False)
    for case in manifest["cases"]:
        directory = destination / case["id"]
        directory.mkdir()
        model = _write_inputs(directory)
        (directory / "model.json").write_text(json.dumps(model, indent=2) + "\n")
        commands = []
        for chromosome in ("1", "2", "3"):
            kinship_name = f"gemma_k_chr{chromosome}"
            commands.append(
                run_command(
                    [
                        str(binary),
                        "-bfile",
                        "tiny",
                        "-gk",
                        "1",
                        "-snps",
                        f"chr{chromosome}.ksnps.txt",
                        "-maf",
                        "0",
                        "-miss",
                        "1",
                        "-outdir",
                        ".",
                        "-o",
                        kinship_name,
                    ],
                    directory,
                    kinship_name,
                )
            )
            gemma_k = np.loadtxt(directory / f"{kinship_name}.cXX.txt")
            independent_k = _independent_loco_kinship(model, chromosome)
            if not np.allclose(gemma_k, independent_k, rtol=1e-8, atol=1e-10):
                raise ValueError(
                    f"independent chromosome-{chromosome} K differs from GEMMA"
                )
            commands.append(
                run_command(
                    [
                        str(binary),
                        "-bfile",
                        "tiny",
                        "-k",
                        f"{kinship_name}.cXX.txt",
                        "-snps",
                        f"chr{chromosome}.snps.txt",
                        "-lmm",
                        str(case["mode"]),
                        "-maf",
                        "0",
                        "-miss",
                        "1",
                        "-outdir",
                        ".",
                        "-o",
                        f"gemma_chr{chromosome}",
                    ],
                    directory,
                    f"gemma_chr{chromosome}",
                )
            )
        _combine_outputs(directory, case["mode"])
        provenance = {
            "schema_version": 1,
            "executable": str(binary),
            "executable_sha256": digest(binary),
            "version": version,
            "manifest": manifest,
            "case": case,
            "commands": commands,
            "generator_sha256": digest(Path(__file__)),
            "files": {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
        }
        (directory / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )


def require_loco_reference(case: dict, root: Path = REFERENCE) -> tuple[Path, dict]:
    directory = root / case["id"]
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["case"] != case or provenance["manifest"] != load_loco_manifest():
        raise ValueError("LOCO case differs from immutable provenance")
    for name, expected in provenance["files"].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError(f"LOCO reference hash mismatch: {name}")
    return directory, provenance


def compare_loco(destination: Path, case_ids: tuple[str, ...] | None = None) -> dict:
    """Compare cold and proven-warm JAMMA LOCO runs independently with GEMMA."""
    from loguru import logger

    from jamma.pipeline import PipelineRunner
    from jamma.pipeline_config import PipelineConfig

    manifest = load_loco_manifest()
    cases = [
        case for case in manifest["cases"] if case_ids is None or case["id"] in case_ids
    ]
    if not cases or (
        case_ids is not None and {c["id"] for c in cases} != set(case_ids)
    ):
        raise ValueError("case_ids must name at least one declared LOCO case")
    destination.mkdir(parents=True, exist_ok=False)
    bundle = {
        "schema_version": 1,
        "status": "INCONCLUSIVE",
        "environment": environment(),
        "manifest": manifest,
        "invocation": sys.argv,
        "cases": [],
        "untested": manifest["untested"],
    }
    statuses = []
    for case in cases:
        source, provenance = require_loco_reference(case)
        copy_reference(source, destination / f"{case['id']}-reference", provenance)
        cache = destination / f"{case['id']}-cache"
        runs = []
        for route, write_eigen in (("cold", True), ("warm", False)):
            out = destination / f"{case['id']}-{route}"
            config = PipelineConfig(
                bfile=source / "tiny",
                lmm_mode=case["mode"],
                maf=0,
                miss=1,
                backend="numpy-streaming",
                output_dir=out,
                output_prefix="study",
                loco=True,
                write_eigen=write_eigen,
                eigen_dir=cache,
                check_memory=False,
                show_progress=False,
                no_telemetry=True,
            )
            messages = []
            sink = logger.add(
                lambda message, captured=messages: captured.append(str(message)),
                format="{message}",
            )
            try:
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
            expected_ids = json.loads((source / "model.json").read_text())["snp_ids"]
            reused = route == "cold" or any(
                "Found complete LOCO eigen cache" in message for message in messages
            )
            if actual_ids != expected_ids or not reused:
                comparison["status"] = "NOT VERIFIED"
                comparison["failure_ids"].append("loco:ordered-case-ids-or-cache-reuse")
            statuses.append(comparison["status"])
            runs.append(
                {
                    "route": route,
                    "cache_reused": route == "warm" and reused,
                    "pipeline_config": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in asdict(config).items()
                    },
                    "comparison": comparison,
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
