#!/usr/bin/env python3
"""LOCO benchmark: GEMMA vs JAMMA on mouse_hs1940.

Runs LOCO association (-loco) across all chromosomes and compares
GEMMA (sequential per-chromosome) vs JAMMA (all chromosomes in one call).

Usage:
    uv run python scripts/bench_loco.py
    uv run python scripts/bench_loco.py --gemma-path /path/to/gemma
    uv run python scripts/bench_loco.py --runs 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _bench_common import (
    MOUSE_DIR,
    MOUSE_KINSHIP,
    MOUSE_PREFIX,
    add_gemma_args,
    best_of,
    find_gemma,
    fmt_seconds,
    load_covariates_4,
    load_fam_phenotypes,
    print_hardware_header,
    speedup,
)


@dataclass(frozen=True)
class LocoTiming:
    """Best-of-N LOCO timing for one backend.

    A backend that failed has no timing at all, so callers get None in
    place of the whole object rather than a ``LocoTiming`` with an empty
    total.

    Attributes:
        total: Fastest whole-run wall-clock seconds.
        per_chr: First run's seconds per chromosome, keyed by the
            chromosome label from the ``.bim``. None for backends that do
            not run chromosomes separately.
    """

    total: float
    per_chr: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _generate_annotation_file(output_path: Path) -> None:
    """Generate BIMBAM annotation file from .bim for GEMMA -loco."""
    bim = np.loadtxt(MOUSE_PREFIX.with_suffix(".bim"), dtype=str, usecols=(0, 1, 3))
    # BIMBAM format: rs, position, chr
    with open(output_path, "w") as f:
        for row in bim:
            chr_num, rs, pos = row
            f.write(f"{rs}, {pos}, {chr_num}\n")


# ---------------------------------------------------------------------------
# GEMMA LOCO benchmark
# ---------------------------------------------------------------------------
def bench_gemma_loco(
    gemma_path: Path, chromosomes: list[str], runs: int
) -> LocoTiming | None:
    """Benchmark GEMMA LOCO, running -loco for each chromosome sequentially.

    Args:
        gemma_path: GEMMA binary to run.
        chromosomes: Chromosome labels to leave out, one run each.
        runs: Repetitions of the whole chromosome sweep.

    Returns:
        The best-of-N timing, or None when any chromosome's run failed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        annot_path = tmpdir_path / "mouse_hs1940.annot"
        _generate_annotation_file(annot_path)

        best_total = float("inf")
        per_chr_times: dict[str, float] = {}

        for run_idx in range(runs):
            run_total = 0.0
            for chrom in chromosomes:
                cmd = [
                    str(gemma_path),
                    "-bfile",
                    str(MOUSE_PREFIX),
                    "-k",
                    str(MOUSE_KINSHIP),
                    "-a",
                    str(annot_path),
                    "-loco",
                    chrom,
                    "-lmm",
                    "1",
                    "-o",
                    f"bench_chr{chrom}",
                    "-outdir",
                    tmpdir,
                ]
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                elapsed = time.perf_counter() - t0

                if proc.returncode != 0:
                    err = proc.stderr or proc.stdout
                    print(
                        f"  GEMMA LOCO chr{chrom} failed: {err[:200]}",
                        file=sys.stderr,
                    )
                    return None

                run_total += elapsed
                if run_idx == 0:
                    per_chr_times[chrom] = elapsed

            if run_total < best_total:
                best_total = run_total

        return LocoTiming(total=best_total, per_chr=per_chr_times)


# ---------------------------------------------------------------------------
# JAMMA LOCO benchmark
# ---------------------------------------------------------------------------
def bench_jamma_loco(
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    runs: int,
) -> float:
    """Benchmark JAMMA LOCO, running all chromosomes in one call.

    Args:
        phenotypes: Per-sample phenotypes.
        covariates: Covariate matrix, or None.
        runs: Repetitions.

    Returns:
        Wall-clock seconds for the fastest repetition.
    """
    from jamma.lmm.loco import run_lmm_loco
    from jamma.lmm.schema import LmmConfig

    def one_run():
        with tempfile.TemporaryDirectory() as tmpdir:
            run_lmm_loco(
                bed_path=MOUSE_PREFIX,
                phenotypes=phenotypes,
                covariates=covariates,
                config=LmmConfig(lmm_mode=1, check_memory=False, show_progress=True),
                output_path=Path(tmpdir) / "loco_results.assoc.txt",
            )

    return best_of(one_run, runs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_gemma_args(parser)
    parser.add_argument(
        "--covariates",
        action="store_true",
        default=False,
        help="Include 4-covariate benchmark row",
    )
    args = parser.parse_args()

    gemma_path = find_gemma(args.gemma_path, "gemma")
    gemma_accel_path = find_gemma(args.gemma_accelerate_path, "gemma-accelerate")

    for label, path in [("GEMMA", gemma_path), ("GEMMA+Accelerate", gemma_accel_path)]:
        if path is not None and not path.exists():
            print(f"ERROR: {label} binary not found at {path}", file=sys.stderr)
            sys.exit(1)

    if not MOUSE_PREFIX.with_suffix(".bed").exists():
        print(f"ERROR: mouse_hs1940 data not found at {MOUSE_DIR}", file=sys.stderr)
        sys.exit(1)

    print_hardware_header(args.runs)

    bim_chr = np.loadtxt(MOUSE_PREFIX.with_suffix(".bim"), dtype=str, usecols=0)
    chromosomes = sorted(set(bim_chr), key=lambda c: int(c) if c.isdigit() else 99)
    print(f"Dataset: {len(bim_chr)} SNPs across {len(chromosomes)} chromosomes")
    print(f"Chromosomes: {', '.join(chromosomes)}")
    print()

    phenotypes = load_fam_phenotypes(MOUSE_PREFIX.with_suffix(".fam"))
    covariates_4 = load_covariates_4() if args.covariates else None

    configs: list[tuple[str, np.ndarray | None]] = [
        ("Wald", None),
    ]
    if covariates_4 is not None:
        configs.append(("Wald+4cov", covariates_4))

    gemma_variants: list[tuple[str, str, Path]] = []
    if gemma_accel_path:
        gemma_variants.append(
            ("GEMMA+Accelerate", "gemma_accelerate", gemma_accel_path)
        )
    if gemma_path:
        gemma_variants.append(("GEMMA+OpenBLAS", "gemma_openblas", gemma_path))
    if not gemma_variants:
        print("No GEMMA binaries found, skipping GEMMA benchmarks")
        print()

    for config_label, covars in configs:
        print(f"=== LOCO {config_label} ===")
        print()

        results: dict[str, LocoTiming | None] = {}

        for gemma_label, gemma_key, gpath in gemma_variants:
            print(f"Benchmarking {gemma_label} LOCO {config_label}...", flush=True)
            gemma_timing = bench_gemma_loco(gpath, chromosomes, args.runs)
            results[gemma_key] = gemma_timing
            if gemma_timing is not None:
                print(f"  {gemma_label} total: {fmt_seconds(gemma_timing.total)}")
                per_chr = gemma_timing.per_chr or {}
                if per_chr:
                    slowest = max(per_chr, key=lambda c: per_chr[c])
                    fastest = min(per_chr, key=lambda c: per_chr[c])
                    print(
                        f"  Per-chr range: {fmt_seconds(per_chr[fastest])}"
                        f" (chr{fastest})"
                        f" – {fmt_seconds(per_chr[slowest])} (chr{slowest})"
                    )
            else:
                print(f"  {gemma_label} LOCO failed")
            print()

        backends_to_run = ["numpy"]

        jamma_totals: dict[str, float] = {}
        for backend in backends_to_run:
            label = "NumPy+C"
            print(f"Benchmarking JAMMA LOCO ({label}) {config_label}...", flush=True)
            jamma_total = bench_jamma_loco(phenotypes, covars, args.runs)
            jamma_totals[backend] = jamma_total
            print(f"  JAMMA ({label}): {fmt_seconds(jamma_total)}")
            print()

        # GEMMA -loco tests ALL SNPs per chromosome (redundant), while JAMMA
        # tests only each chromosome's own SNPs.
        n_total_snps = len(bim_chr)
        n_chr = len(chromosomes)
        gemma_tests = n_chr * n_total_snps
        jamma_tests = n_total_snps

        print(f"Note: GEMMA -loco tests ALL {n_total_snps} SNPs per chromosome")
        print(f"  ({n_chr} × {n_total_snps} = {gemma_tests:,} total SNP-tests)")
        print(f"  JAMMA tests each SNP once ({jamma_tests:,} total SNP-tests)")
        print()

        gemma_times = [
            timing.total
            for _, gk, _ in gemma_variants
            if (timing := results[gk]) is not None
        ]
        gemma_ref = min(gemma_times) if gemma_times else None

        print(f"| Backend | LOCO {config_label} | vs fastest GEMMA |")
        print(
            "|---------|"
            + "-" * (len(f" LOCO {config_label} ") + 2)
            + "|------------------|"
        )

        for gemma_label, gemma_key, _ in gemma_variants:
            timing = results[gemma_key]
            if timing is not None:
                gt = timing.total
                print(
                    f"| {gemma_label} | {fmt_seconds(gt)} | {speedup(gemma_ref, gt)} |"
                )

        for backend in backends_to_run:
            jt = jamma_totals[backend]
            print(f"| JAMMA NumPy+C | {fmt_seconds(jt)} | {speedup(gemma_ref, jt)} |")

        print()


if __name__ == "__main__":
    main()
