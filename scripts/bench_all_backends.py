#!/usr/bin/env python3
"""End-to-end backend comparison benchmark on mouse_hs1940.

Runs kinship (-gk 1), LMM Wald (-lmm 1), and LMM All (-lmm 4) across all
available backends and prints a formatted table matching the README.

Usage:
    uv run python scripts/bench_all_backends.py
    uv run python scripts/bench_all_backends.py --gemma-path /path/to/gemma
    uv run python scripts/bench_all_backends.py --runs 3

Backends run sequentially to avoid cross-contamination of timings.
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
_DEFAULT_GEMMA = Path.home() / ".local" / "bin" / "gemma"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _speedup(gemma: float | None, jamma: float) -> str:
    if gemma is None:
        return "—"
    return f"**{gemma / jamma:.1f}x**"


def _load_mouse_data():
    """Load mouse_hs1940 PLINK data and phenotypes."""
    from jamma.io import load_plink_binary

    plink = load_plink_binary(_MOUSE_PREFIX)

    # Load phenotypes from .fam (column 6)
    from jamma.core.constants import PHENOTYPE_MISSING

    fam_data = np.loadtxt(_MOUSE_PREFIX.with_suffix(".fam"), usecols=5, dtype=str)
    missing = np.isin(fam_data, [str(int(PHENOTYPE_MISSING)), "NA"])
    phenotypes = np.where(missing, "0", fam_data).astype(np.float64)
    phenotypes[missing] = np.nan

    return plink, phenotypes


def _build_snp_info(plink):
    """Build snp_info list from PLINK data."""
    return [
        {
            "chr": str(plink.chromosome[i]),
            "rs": plink.sid[i],
            "pos": int(plink.bp_position[i]),
            "a1": plink.allele_1[i],
            "a0": plink.allele_2[i],
        }
        for i in range(plink.n_snps)
    ]


# ---------------------------------------------------------------------------
# GEMMA benchmark
# ---------------------------------------------------------------------------
def bench_gemma(gemma_path: Path, runs: int) -> dict[str, float | None]:
    """Benchmark GEMMA binary on mouse_hs1940."""
    results: dict[str, float | None] = {}

    for op, args in [
        ("kinship", ["-gk", "1"]),
        ("lmm_wald", ["-lmm", "1", "-k", str(_MOUSE_KINSHIP)]),
        ("lmm_all", ["-lmm", "4", "-k", str(_MOUSE_KINSHIP)]),
    ]:
        best = float("inf")
        for _ in range(runs):
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    str(gemma_path),
                    "-bfile",
                    str(_MOUSE_PREFIX),
                    *args,
                    "-o",
                    "bench",
                    "-outdir",
                    tmpdir,
                ]
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                elapsed = time.perf_counter() - t0

                if proc.returncode != 0:
                    print(f"  GEMMA {op} failed: {proc.stderr[:200]}", file=sys.stderr)
                    results[op] = None
                    break
                best = min(best, elapsed)
        else:
            results[op] = best

    return results


# ---------------------------------------------------------------------------
# JAMMA NumPy+C benchmark
# ---------------------------------------------------------------------------
def bench_numpy(plink, phenotypes, kinship, snp_info, runs: int) -> dict[str, float]:
    """Benchmark NumPy+C backend."""
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    results: dict[str, float] = {}

    for op, mode in [("lmm_wald", 1), ("lmm_all", 4)]:
        best = float("inf")
        for _ in range(runs):
            t0 = time.perf_counter()
            run_lmm_association_numpy(
                genotypes=plink.genotypes,
                phenotypes=phenotypes,
                kinship=kinship.copy(),
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
                lmm_mode=mode,
            )
            elapsed = time.perf_counter() - t0
            best = min(best, elapsed)
        results[op] = best

    return results


# ---------------------------------------------------------------------------
# JAMMA JAX batch benchmark
# ---------------------------------------------------------------------------
def bench_jax_batch(
    plink, phenotypes, kinship, snp_info, runs: int
) -> dict[str, float]:
    """Benchmark JAX batch backend."""
    from jamma.lmm.runner_jax import run_lmm_association_jax

    results: dict[str, float] = {}

    # Warmup run (JIT compilation)
    run_lmm_association_jax(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship.copy(),
        snp_info=snp_info,
        show_progress=False,
        check_memory=False,
        lmm_mode=1,
    )

    for op, mode in [("lmm_wald", 1), ("lmm_all", 4)]:
        best = float("inf")
        for _ in range(runs):
            t0 = time.perf_counter()
            run_lmm_association_jax(
                genotypes=plink.genotypes,
                phenotypes=phenotypes,
                kinship=kinship.copy(),
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
                lmm_mode=mode,
            )
            elapsed = time.perf_counter() - t0
            best = min(best, elapsed)
        results[op] = best

    return results


# ---------------------------------------------------------------------------
# JAMMA JAX streaming benchmark
# ---------------------------------------------------------------------------
def bench_jax_streaming(phenotypes, kinship, runs: int) -> dict[str, float]:
    """Benchmark JAX streaming backend."""
    from jamma.lmm.runner_streaming import run_lmm_association_streaming

    results: dict[str, float] = {}

    # Warmup run (JIT compilation)
    run_lmm_association_streaming(
        bed_path=_MOUSE_PREFIX,
        phenotypes=phenotypes,
        kinship=kinship.copy(),
        show_progress=False,
        check_memory=False,
        lmm_mode=1,
    )

    for op, mode in [("lmm_wald", 1), ("lmm_all", 4)]:
        best = float("inf")
        for _ in range(runs):
            t0 = time.perf_counter()
            run_lmm_association_streaming(
                bed_path=_MOUSE_PREFIX,
                phenotypes=phenotypes,
                kinship=kinship.copy(),
                show_progress=False,
                check_memory=False,
                lmm_mode=mode,
            )
            elapsed = time.perf_counter() - t0
            best = min(best, elapsed)
        results[op] = best

    return results


# ---------------------------------------------------------------------------
# Kinship benchmark
# ---------------------------------------------------------------------------
def bench_kinship(plink, runs: int) -> dict[str, float]:
    """Benchmark kinship computation (uses JAX via compute_centered_kinship)."""
    from jamma.kinship import compute_centered_kinship

    # Warmup
    compute_centered_kinship(plink.genotypes, check_memory=False)

    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        compute_centered_kinship(plink.genotypes, check_memory=False)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)

    return {"kinship": best}


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
        help="Number of runs per operation, report best (default: 1)",
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

    # Configure JAX
    from jamma.core.jax_config import ensure_jax_configured

    ensure_jax_configured()

    # Print hardware context
    from jamma.core.hardware import get_hardware_context

    ctx = get_hardware_context()
    phys, log = ctx["cpu_count_physical"], ctx["cpu_count_logical"]
    print(f"CPU: {ctx['cpu_model']} ({phys}P/{log}L)")
    print(f"BLAS: {ctx['blas_backend']} ({ctx['blas_threads']} threads)")
    jv, jb, jd = ctx["jax_version"], ctx["jax_backend"], ctx["jax_device_count"]
    print(f"JAX: {jv} ({jb}, {jd} devices)")
    print(f"NumPy: {ctx['numpy_version']}")
    print(f"Platform: {ctx['platform']}")
    print(f"Runs: {args.runs} (best of)")
    print()

    # Load data once
    print("Loading mouse_hs1940 data...", flush=True)
    plink, phenotypes = _load_mouse_data()
    print(f"  {plink.n_samples} samples, {plink.n_snps} SNPs")

    # Load pre-computed kinship for LMM runs
    from jamma.kinship.io import read_kinship_matrix

    kinship = read_kinship_matrix(_MOUSE_KINSHIP)
    snp_info = _build_snp_info(plink)
    print()

    # Collect results: {backend: {op: seconds}}
    timings: dict[str, dict[str, float | None]] = {}

    # --- GEMMA ---
    if gemma_path:
        print(f"Benchmarking GEMMA ({gemma_path})...", flush=True)
        timings["gemma"] = bench_gemma(gemma_path, args.runs)
    else:
        print("GEMMA not found, skipping (use --gemma-path to specify)")
        timings["gemma"] = {"kinship": None, "lmm_wald": None, "lmm_all": None}

    # --- Kinship (JAMMA) ---
    print("Benchmarking kinship (JAMMA)...", flush=True)
    kinship_times = bench_kinship(plink, args.runs)

    # --- NumPy+C ---
    print("Benchmarking NumPy+C...", flush=True)
    numpy_times = bench_numpy(plink, phenotypes, kinship, snp_info, args.runs)
    numpy_times["kinship"] = kinship_times["kinship"]
    timings["numpy"] = numpy_times

    # --- JAX batch ---
    print("Benchmarking JAX batch...", flush=True)
    jax_times = bench_jax_batch(plink, phenotypes, kinship, snp_info, args.runs)
    jax_times["kinship"] = kinship_times["kinship"]
    timings["jax_batch"] = jax_times

    # --- JAX streaming ---
    print("Benchmarking JAX streaming...", flush=True)
    streaming_times = bench_jax_streaming(phenotypes, kinship, args.runs)
    streaming_times["kinship"] = None  # streaming doesn't do kinship
    timings["jax_streaming"] = streaming_times

    print()

    # --- Print results table ---
    gemma = timings["gemma"]
    npy = timings["numpy"]
    jax_b = timings["jax_batch"]
    jax_s = timings["jax_streaming"]

    def _cell(t: float | None) -> str:
        return _fmt(t) if t is not None else "—"

    def _vs(t: float | None, op: str) -> str:
        g = gemma.get(op)
        if g is None or t is None:
            return "—"
        return f"{g / t:.1f}x"

    # Find the fastest JAMMA backend per operation for the "vs GEMMA" column
    def _best_jamma(op: str) -> float | None:
        candidates = [
            npy.get(op),
            jax_b.get(op),
            jax_s.get(op),
        ]
        valid = [c for c in candidates if c is not None]
        return min(valid) if valid else None

    rows = [
        ("Kinship (`-gk 1`)", "kinship"),
        ("LMM Wald (`-lmm 1`)", "lmm_wald"),
        ("LMM All (`-lmm 4`)", "lmm_all"),
    ]

    # Header
    hdr = (
        "| Operation | GEMMA 0.98.5 | JAMMA NumPy+C"
        " | JAMMA JAX (batch) | JAMMA JAX (streaming)"
        " | vs GEMMA |"
    )
    sep = (
        "|-----------|-------------|--------------|"
        "-------------------|----------------------|"
        "----------|"
    )
    print(hdr)
    print(sep)

    for label, op in rows:
        best = _best_jamma(op)
        vs = _vs(best, op) if best else "—"
        print(
            f"| {label} | {_cell(gemma.get(op))} | {_cell(npy.get(op))} "
            f"| {_cell(jax_b.get(op))} | {_cell(jax_s.get(op))} | {vs} |"
        )

    print()


if __name__ == "__main__":
    main()
