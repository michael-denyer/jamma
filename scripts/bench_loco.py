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
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOUSE_DIR = _REPO_ROOT / "tests" / "fixtures" / "mouse_hs1940"
_MOUSE_PREFIX = _MOUSE_DIR / "mouse_hs1940"
_MOUSE_KINSHIP = _MOUSE_DIR / "mouse_hs1940_kinship.cXX.txt"
_MOUSE_COVAR_4 = _MOUSE_DIR / "covariates_4.txt"
_DEFAULT_GEMMA = Path.home() / ".local" / "bin" / "gemma"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _speedup(ref: float | None, fast: float) -> str:
    if ref is None:
        return "—"
    return f"{ref / fast:.1f}x"


def _load_phenotypes() -> np.ndarray:
    from jamma.core.constants import PHENOTYPE_MISSING

    fam_data = np.loadtxt(_MOUSE_PREFIX.with_suffix(".fam"), usecols=5, dtype=str)
    missing = np.isin(fam_data, [str(int(PHENOTYPE_MISSING)), "NA"])
    phenotypes = np.where(missing, "0", fam_data).astype(np.float64)
    phenotypes[missing] = np.nan
    return phenotypes


def _load_covariates_4() -> np.ndarray | None:
    if _MOUSE_COVAR_4.exists():
        return np.loadtxt(_MOUSE_COVAR_4)
    return None


def _generate_annotation_file(output_path: Path) -> None:
    """Generate BIMBAM annotation file from .bim for GEMMA -loco."""
    bim = np.loadtxt(_MOUSE_PREFIX.with_suffix(".bim"), dtype=str, usecols=(0, 1, 3))
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
) -> dict[str, float | None]:
    """Benchmark GEMMA LOCO: runs -loco for each chromosome sequentially."""
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
                    str(_MOUSE_PREFIX),
                    "-k",
                    str(_MOUSE_KINSHIP),
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
                    return {"loco_total": None}

                run_total += elapsed
                if run_idx == 0:
                    per_chr_times[chrom] = elapsed

            if run_total < best_total:
                best_total = run_total

        return {
            "loco_total": best_total,
            "per_chr": per_chr_times,
        }


# ---------------------------------------------------------------------------
# JAMMA LOCO benchmark
# ---------------------------------------------------------------------------
def bench_jamma_loco(
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    backend: str,
    runs: int,
) -> dict[str, float]:
    """Benchmark JAMMA LOCO (all chromosomes in one call)."""
    from jamma.lmm.loco import run_lmm_loco

    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_lmm_loco(
                bed_path=_MOUSE_PREFIX,
                phenotypes=phenotypes,
                covariates=covariates,
                lmm_mode=1,
                output_path=Path(tmpdir) / "loco_results.assoc.txt",
                check_memory=False,
                show_progress=True,
                backend=backend,
            )
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)

    return {"loco_total": best}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemma-path",
        type=Path,
        default=None,
        help=f"Path to GEMMA binary (default: auto-detect at {_DEFAULT_GEMMA})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs, report best (default: 1)",
    )
    parser.add_argument(
        "--covariates",
        action="store_true",
        default=False,
        help="Include 4-covariate benchmark row",
    )
    args = parser.parse_args()

    # Resolve GEMMA path
    gemma_path = args.gemma_path
    if gemma_path is None:
        if _DEFAULT_GEMMA.exists():
            gemma_path = _DEFAULT_GEMMA
        else:
            found = shutil.which("gemma")
            if found:
                gemma_path = Path(found)

    # Validate data exists
    if not _MOUSE_PREFIX.with_suffix(".bed").exists():
        print(f"ERROR: mouse_hs1940 data not found at {_MOUSE_DIR}", file=sys.stderr)
        sys.exit(1)

    # Print hardware context
    from jamma.core.hardware import get_hardware_context

    ctx = get_hardware_context()
    phys, log = ctx["cpu_count_physical"], ctx["cpu_count_logical"]
    print(f"CPU: {ctx['cpu_model']} ({phys}P/{log}L)")
    print(f"BLAS: {ctx['blas_backend']} ({ctx['blas_threads']} threads)")
    print(f"NumPy: {ctx['numpy_version']}")
    print(f"Platform: {ctx['platform']}")
    print(f"Runs: {args.runs} (best of)")
    print()

    # Get chromosome list
    bim_chr = np.loadtxt(_MOUSE_PREFIX.with_suffix(".bim"), dtype=str, usecols=0)
    chromosomes = sorted(set(bim_chr), key=lambda c: int(c) if c.isdigit() else 99)
    print(f"Dataset: {len(bim_chr)} SNPs across {len(chromosomes)} chromosomes")
    print(f"Chromosomes: {', '.join(chromosomes)}")
    print()

    phenotypes = _load_phenotypes()
    covariates_4 = _load_covariates_4() if args.covariates else None

    # Build benchmark configurations
    configs: list[tuple[str, np.ndarray | None]] = [
        ("Wald", None),
    ]
    if covariates_4 is not None:
        configs.append(("Wald+4cov", covariates_4))

    for config_label, covars in configs:
        print(f"=== LOCO {config_label} ===")
        print()

        results: dict[str, dict] = {}

        # GEMMA
        if gemma_path:
            print(f"Benchmarking GEMMA LOCO {config_label}...", flush=True)
            gemma_results = bench_gemma_loco(gemma_path, chromosomes, args.runs)
            results["gemma"] = gemma_results
            gt = gemma_results.get("loco_total")
            if gt is not None:
                print(f"  GEMMA total: {_fmt(gt)}")
                per_chr = gemma_results.get("per_chr", {})
                if per_chr:
                    slowest = max(per_chr, key=per_chr.get)
                    fastest = min(per_chr, key=per_chr.get)
                    print(
                        f"  Per-chr range: {_fmt(per_chr[fastest])} (chr{fastest})"
                        f" – {_fmt(per_chr[slowest])} (chr{slowest})"
                    )
            else:
                print("  GEMMA LOCO failed")
            print()
        else:
            print("GEMMA not found, skipping")
            results["gemma"] = {"loco_total": None}
            print()

        # JAMMA NumPy+C backend
        backends_to_run = ["numpy"]

        for backend in backends_to_run:
            label = "NumPy+C"
            print(f"Benchmarking JAMMA LOCO ({label}) {config_label}...", flush=True)
            jamma_results = bench_jamma_loco(phenotypes, covars, backend, args.runs)
            results[f"jamma_{backend}"] = jamma_results
            jt = jamma_results["loco_total"]
            print(f"  JAMMA ({label}): {_fmt(jt)}")
            print()

        # Summary table
        gemma_t = results["gemma"].get("loco_total")

        # Note: GEMMA -loco tests ALL SNPs per chromosome (redundant),
        # while JAMMA tests only each chromosome's own SNPs.
        # GEMMA total tests: n_chr * n_snps_total
        # JAMMA total tests: n_snps_total (each SNP tested once)
        n_total_snps = len(bim_chr)
        n_chr = len(chromosomes)
        gemma_tests = n_chr * n_total_snps
        jamma_tests = n_total_snps

        print(f"Note: GEMMA -loco tests ALL {n_total_snps} SNPs per chromosome")
        print(f"  ({n_chr} × {n_total_snps} = {gemma_tests:,} total SNP-tests)")
        print(f"  JAMMA tests each SNP once ({jamma_tests:,} total SNP-tests)")
        print()

        print(f"| Backend | LOCO {config_label} | vs GEMMA |")
        print(
            "|---------|" + "-" * (len(f" LOCO {config_label} ") + 2) + "|----------|"
        )

        if gemma_t is not None:
            print(f"| GEMMA 0.98.5 | {_fmt(gemma_t)} | 1.0x |")

        for backend in backends_to_run:
            key = f"jamma_{backend}"
            jt = results[key]["loco_total"]
            print(f"| JAMMA NumPy+C | {_fmt(jt)} | {_speedup(gemma_t, jt)} |")

        print()


if __name__ == "__main__":
    main()
