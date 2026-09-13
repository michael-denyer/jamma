#!/usr/bin/env python3
"""Disk-to-disk LOCO comparison on mouse_hs1940.

GEMMA computes each chromosome's excluded kinship, then tests only that
chromosome's SNPs. JAMMA computes kinship internally and tests each SNP once.
Both start from PLINK files and finish with association files. All child
process startup, input loading, kinship computation and output I/O are timed.

Usage: uv run python scripts/bench_loco.py --runs 3 --json loco.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from _bench_common import (
    MOUSE_COVAR_4,
    MOUSE_PREFIX,
    add_gemma_args,
    find_gemma,
    fmt_seconds,
    print_hardware_header,
    read_associations,
    run_commands,
    speedup,
    verify_associations,
)


def prepare_inputs(directory: Path) -> dict[str, Path]:
    """Prepare complementary kinship and association SNP lists, untimed."""
    rows = [
        line.split()
        for line in MOUSE_PREFIX.with_suffix(".bim").read_text().splitlines()
    ]
    chromosomes = sorted({row[0] for row in rows}, key=int)
    lists = {}
    for chrom in chromosomes:
        path = directory / f"chr{chrom}.snps"
        path.write_text("".join(f"{row[1]}\n" for row in rows if row[0] == chrom))
        path.with_suffix(".kinship.snps").write_text(
            "".join(f"{row[1]}\n" for row in rows if row[0] != chrom)
        )
        lists[chrom] = path
    return lists


def gemma_commands(
    executable: Path, directory: Path, snp_lists: dict[str, Path], covariates: bool
) -> list[list[str]]:
    """Compute the correct LOCO matrix before each chromosome's association."""
    commands = []
    for chrom, snps in snp_lists.items():
        prefix = f"chr{chrom}"
        common = [
            str(executable),
            "-bfile",
            str(MOUSE_PREFIX),
            "-outdir",
            str(directory),
            "-o",
            prefix,
        ]
        if covariates:
            common += ["-c", str(MOUSE_COVAR_4)]
        # GEMMA 0.98.5 PlinkKin ignores the -loco/-ksnps sets. Restrict
        # the input SNPs explicitly, then use the complementary association set.
        commands.append(
            [*common, "-gk", "1", "-snps", str(snps.with_suffix(".kinship.snps"))]
        )
        commands.append(
            [
                *common,
                "-lmm",
                "1",
                "-k",
                str(directory / f"{prefix}.cXX.txt"),
                "-snps",
                str(snps),
            ]
        )
    return commands


def run_benchmarks(
    gemma: Path | None, accelerate: Path | None, runs: int, covariates: bool
) -> list[dict]:
    variants = [
        ("GEMMA (OpenBLAS)", gemma),
        ("GEMMA (Accelerate)", accelerate),
        ("JAMMA NumPy+C", None),
    ]
    variants = [
        (name, path) for name, path in variants if path or name.startswith("JAMMA")
    ]
    records: list[dict] = []
    reference = None
    env = dict(os.environ)
    env.pop("JAMMA_FORCE_NUMPY_FALLBACK", None)
    for repetition in range(runs):
        ordered = (
            variants[repetition % len(variants) :]
            + variants[: repetition % len(variants)]
        )
        for name, executable in ordered:
            print(f"LOCO: {name}, run {repetition + 1}/{runs}", flush=True)
            with tempfile.TemporaryDirectory() as tmpdir:
                directory = Path(tmpdir)
                snp_lists = prepare_inputs(directory)
                if executable is not None:
                    commands = gemma_commands(
                        executable, directory, snp_lists, covariates
                    )
                    outputs = [
                        directory / f"chr{chrom}.assoc.txt" for chrom in snp_lists
                    ]
                else:
                    commands = [
                        [
                            sys.executable,
                            "-m",
                            "jamma",
                            "-bfile",
                            str(MOUSE_PREFIX),
                            "-lmm",
                            "1",
                            "-loco",
                            "-outdir",
                            str(directory),
                            "-o",
                            "bench",
                            "--no-telemetry",
                        ]
                    ]
                    if covariates:
                        commands[0] += ["-c", str(MOUSE_COVAR_4)]
                    outputs = [directory / "bench.assoc.txt"]
                elapsed = run_commands(commands, env)
                associations: dict[str, dict[str, str]] = {}
                for output in outputs:
                    rows = read_associations(output)
                    if associations.keys() & rows.keys():
                        raise ValueError("LOCO tested a SNP more than once")
                    if executable is not None:
                        chrom = output.name.split(".")[0].removeprefix("chr")
                        if any(row["chr"] != chrom for row in rows.values()):
                            raise ValueError(f"Wrong chromosome in {output}")
                    associations.update(rows)
                if reference is None:
                    reference = associations
                else:
                    verify_associations(reference, associations)
                records.append(
                    {
                        "backend": name,
                        "repetition": repetition + 1,
                        "seconds": elapsed,
                        "snps": len(associations),
                        "commands": commands,
                    }
                )
                print(f"  {fmt_seconds(elapsed)}, {len(associations)} SNPs", flush=True)
    best = {
        name: min(r["seconds"] for r in records if r["backend"] == name)
        for name, _ in variants
    }
    gemma_times = [
        seconds for name, seconds in best.items() if name.startswith("GEMMA")
    ]
    reference_time = min(gemma_times) if gemma_times else None
    print("| Backend | LOCO Wald | vs fastest GEMMA |")
    print("|---------|-----------|------------------|")
    for name, seconds in best.items():
        print(
            f"| {name} | {fmt_seconds(seconds)} | {speedup(reference_time, seconds)} |"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gemma_args(parser)
    parser.add_argument(
        "--covariates", action="store_true", help="Use four covariates in both tools"
    )
    parser.add_argument("--json", type=Path, help="Save raw timings and commands")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.covariates and not MOUSE_COVAR_4.exists():
        parser.error("Covariate fixture is missing")
    print_hardware_header(args.runs)
    records = run_benchmarks(
        find_gemma(args.gemma_path, "gemma"),
        find_gemma(args.gemma_accelerate_path, "gemma-accelerate"),
        args.runs,
        args.covariates,
    )
    if args.json:
        args.json.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
