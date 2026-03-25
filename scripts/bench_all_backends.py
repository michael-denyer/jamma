#!/usr/bin/env python3
"""End-to-end backend comparison benchmark on mouse_hs1940.

Runs kinship (-gk 1), LMM Wald (-lmm 1), and LMM All (-lmm 4) across
NumPy backends (batch, streaming, pure-Python) and GEMMA, then prints
a formatted table matching the README.

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
_MOUSE_COVAR_4 = _MOUSE_DIR / "covariates_4.txt"
_DEFAULT_GEMMA = Path.home() / ".local" / "bin" / "gemma"
_DEFAULT_GEMMA_ACCEL = Path.home() / ".local" / "bin" / "gemma-accelerate"


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


def _load_covariates_4() -> np.ndarray | None:
    """Load 4-column covariate file if it exists."""
    if _MOUSE_COVAR_4.exists():
        return np.loadtxt(_MOUSE_COVAR_4)
    return None


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

    ops = [
        ("kinship", ["-gk", "1"]),
        ("lmm_wald", ["-lmm", "1", "-k", str(_MOUSE_KINSHIP)]),
        ("lmm_all", ["-lmm", "4", "-k", str(_MOUSE_KINSHIP)]),
    ]
    if _MOUSE_COVAR_4.exists():
        ops.append(
            (
                "lmm_wald_c4",
                ["-lmm", "1", "-k", str(_MOUSE_KINSHIP), "-c", str(_MOUSE_COVAR_4)],
            )
        )

    for op, args in ops:
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
def _bench_numpy_inner(
    plink, phenotypes, kinship, snp_info, covariates_4, runs: int, *, disable_c: bool
) -> dict[str, float]:
    """Benchmark NumPy backend with or without C acceleration."""
    import jamma.lmm.compute_numpy as cn
    import jamma.lmm.runner_numpy as rn
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    results: dict[str, float] = {}

    # Optionally disable C extension for pure-Python comparison.
    # Must patch both compute_numpy (where _compute_wald_numpy checks flags)
    # AND runner_numpy (which imports copies of the flags at module level).
    cn_saved = (cn._C_ACCEL_AVAILABLE, cn._C_SPLIT_AVAILABLE, cn._C_GENERAL_AVAILABLE)
    rn_saved = (rn._C_ACCEL_AVAILABLE, rn._C_SPLIT_AVAILABLE, rn._C_GENERAL_AVAILABLE)
    if disable_c:
        cn._C_ACCEL_AVAILABLE = False
        cn._C_SPLIT_AVAILABLE = False
        cn._C_GENERAL_AVAILABLE = False
        rn._C_ACCEL_AVAILABLE = False
        rn._C_SPLIT_AVAILABLE = False
        rn._C_GENERAL_AVAILABLE = False

    try:
        ops: list[tuple[str, int, np.ndarray | None]] = [
            ("lmm_wald", 1, None),
            ("lmm_all", 4, None),
        ]
        if covariates_4 is not None:
            ops.append(("lmm_wald_c4", 1, covariates_4))

        for op, mode, covars in ops:
            best = float("inf")
            for _ in range(runs):
                t0 = time.perf_counter()
                run_lmm_association_numpy(
                    genotypes=plink.genotypes,
                    phenotypes=phenotypes,
                    kinship=kinship.copy(),
                    snp_info=snp_info,
                    covariates=covars,
                    show_progress=False,
                    check_memory=False,
                    lmm_mode=mode,
                )
                elapsed = time.perf_counter() - t0
                best = min(best, elapsed)
            results[op] = best
    finally:
        cn._C_ACCEL_AVAILABLE, cn._C_SPLIT_AVAILABLE, cn._C_GENERAL_AVAILABLE = cn_saved
        rn._C_ACCEL_AVAILABLE, rn._C_SPLIT_AVAILABLE, rn._C_GENERAL_AVAILABLE = rn_saved

    return results


def bench_numpy(
    plink, phenotypes, kinship, snp_info, covariates_4, runs: int
) -> dict[str, float]:
    """Benchmark NumPy+C backend."""
    return _bench_numpy_inner(
        plink, phenotypes, kinship, snp_info, covariates_4, runs, disable_c=False
    )


def bench_numpy_pure(
    plink, phenotypes, kinship, snp_info, covariates_4, runs: int
) -> dict[str, float]:
    """Benchmark pure NumPy backend (C extension disabled)."""
    return _bench_numpy_inner(
        plink, phenotypes, kinship, snp_info, covariates_4, runs, disable_c=True
    )


# ---------------------------------------------------------------------------
# JAMMA NumPy streaming benchmark
# ---------------------------------------------------------------------------
def bench_numpy_streaming(
    phenotypes, kinship, covariates_4, runs: int
) -> dict[str, float]:
    """Benchmark NumPy streaming backend (disk I/O + C extension)."""
    from jamma.lmm.runner_numpy_streaming import run_lmm_association_numpy_streaming

    results: dict[str, float] = {}

    ops: list[tuple[str, int, np.ndarray | None]] = [
        ("lmm_wald", 1, None),
        ("lmm_all", 4, None),
    ]
    if covariates_4 is not None:
        ops.append(("lmm_wald_c4", 1, covariates_4))

    for op, mode, covars in ops:
        best = float("inf")
        for _ in range(runs):
            t0 = time.perf_counter()
            run_lmm_association_numpy_streaming(
                bed_path=_MOUSE_PREFIX,
                phenotypes=phenotypes,
                kinship=kinship.copy(),
                covariates=covars,
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
    """Benchmark kinship computation (NumPy/BLAS via compute_centered_kinship)."""
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
        "--gemma-accelerate-path",
        type=Path,
        default=None,
        help=f"Path to Accelerate GEMMA (default: {_DEFAULT_GEMMA_ACCEL})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per operation, report best (default: 1)",
    )
    args = parser.parse_args()

    # Resolve GEMMA paths
    gemma_path = args.gemma_path
    if gemma_path is None:
        if _DEFAULT_GEMMA.exists():
            gemma_path = _DEFAULT_GEMMA
        else:
            found = shutil.which("gemma")
            if found:
                gemma_path = Path(found)

    gemma_accel_path = args.gemma_accelerate_path
    if gemma_accel_path is None and _DEFAULT_GEMMA_ACCEL.exists():
        gemma_accel_path = _DEFAULT_GEMMA_ACCEL

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

    # Load data once
    print("Loading mouse_hs1940 data...", flush=True)
    plink, phenotypes = _load_mouse_data()
    print(f"  {plink.n_samples} samples, {plink.n_snps} SNPs")

    # Load pre-computed kinship for LMM runs
    from jamma.kinship.io import read_kinship_matrix

    kinship = read_kinship_matrix(_MOUSE_KINSHIP)
    snp_info = _build_snp_info(plink)
    covariates_4 = _load_covariates_4()
    if covariates_4 is not None:
        print(
            f"  Covariates: {covariates_4.shape[1]} columns from {_MOUSE_COVAR_4.name}"
        )
    print()

    # Collect results: {backend: {op: seconds}}
    timings: dict[str, dict[str, float | None]] = {}

    # --- GEMMA (OpenBLAS) ---
    if gemma_path:
        print(f"Benchmarking GEMMA OpenBLAS ({gemma_path})...", flush=True)
        timings["gemma"] = bench_gemma(gemma_path, args.runs)
    else:
        print("GEMMA not found, skipping (use --gemma-path to specify)")
        timings["gemma"] = {"kinship": None, "lmm_wald": None, "lmm_all": None}

    # --- GEMMA (Accelerate) ---
    if gemma_accel_path:
        print(f"Benchmarking GEMMA Accelerate ({gemma_accel_path})...", flush=True)
        timings["gemma_accel"] = bench_gemma(gemma_accel_path, args.runs)
    else:
        print("GEMMA Accelerate not found, skipping (use --gemma-accelerate-path)")
        timings["gemma_accel"] = {"kinship": None, "lmm_wald": None, "lmm_all": None}

    # --- Kinship (JAMMA) ---
    print("Benchmarking kinship (JAMMA)...", flush=True)
    kinship_times = bench_kinship(plink, args.runs)

    # --- Pure NumPy (no C) ---
    print("Benchmarking NumPy (pure Python, no C)...", flush=True)
    numpy_pure_times = bench_numpy_pure(
        plink, phenotypes, kinship, snp_info, covariates_4, args.runs
    )
    timings["numpy_pure"] = numpy_pure_times

    # --- NumPy+C ---
    print("Benchmarking NumPy+C...", flush=True)
    numpy_times = bench_numpy(
        plink, phenotypes, kinship, snp_info, covariates_4, args.runs
    )
    timings["numpy"] = numpy_times

    # --- NumPy streaming ---
    print("Benchmarking NumPy streaming...", flush=True)
    numpy_streaming_times = bench_numpy_streaming(
        phenotypes, kinship, covariates_4, args.runs
    )
    numpy_streaming_times["kinship"] = None  # streaming doesn't do kinship
    timings["numpy_streaming"] = numpy_streaming_times

    print()

    # --- Print results table ---
    gemma = timings["gemma"]
    gemma_a = timings["gemma_accel"]
    npy_pure = timings["numpy_pure"]
    npy = timings["numpy"]
    npy_s = timings["numpy_streaming"]

    def _cell(t: float | None) -> str:
        return _fmt(t) if t is not None else "—"

    def _vs(t: float | None, op: str, *, ref: dict = None) -> str:
        src = ref if ref is not None else gemma
        g = src.get(op)
        if g is None or t is None:
            return "—"
        return f"{g / t:.1f}x"

    def _c_speedup(op: str) -> str:
        """C extension speedup vs pure NumPy."""
        pure = npy_pure.get(op)
        c = npy.get(op)
        if pure is None or c is None:
            return "—"
        return f"{pure / c:.1f}x"

    # Find the fastest JAMMA backend per operation for the "vs GEMMA" column
    def _best_jamma(op: str) -> float | None:
        candidates = [
            npy.get(op),
            npy_s.get(op),
        ]
        valid = [c for c in candidates if c is not None]
        return min(valid) if valid else None

    rows = [
        ("Kinship (`-gk 1`)", "kinship"),
        ("LMM Wald (`-lmm 1`)", "lmm_wald"),
        ("LMM All (`-lmm 4`)", "lmm_all"),
    ]
    if covariates_4 is not None:
        rows.append(("LMM Wald+4cov (`-lmm 1 -c`)", "lmm_wald_c4"))

    # Kinship is always NumPy/BLAS (no C extension).
    # Inject into both NumPy dicts so it appears in those columns.
    npy_pure["kinship"] = kinship_times["kinship"]
    npy["kinship"] = kinship_times["kinship"]

    # Header
    hdr = (
        "| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy"
        " | JAMMA NumPy+C"
        " | JAMMA NumPy+C (stream)"
        " | C speedup | vs GEMMA (OB) | vs GEMMA (Accel) |"
    )
    sep = (
        "|-----------|-----------------|-------------------|-------------|--------------|"
        "------------------------|"
        "-----------|---------------|------------------|"
    )
    print(hdr)
    print(sep)

    for label, op in rows:
        best = _best_jamma(op)
        vs_ob = _vs(best, op) if best else "—"
        vs_ac = _vs(best, op, ref=gemma_a) if best else "—"
        print(
            f"| {label} | {_cell(gemma.get(op))}"
            f" | {_cell(gemma_a.get(op))}"
            f" | {_cell(npy_pure.get(op))} | {_cell(npy.get(op))}"
            f" | {_cell(npy_s.get(op))}"
            f" | {_c_speedup(op)} | {vs_ob} | {vs_ac} |"
        )

    print()


if __name__ == "__main__":
    main()
