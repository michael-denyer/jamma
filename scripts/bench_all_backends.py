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
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _bench_common import (
    MOUSE_COVAR_4,
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
)

OpTimings = dict[str, float | None]


@dataclass(frozen=True)
class Timing:
    """Best-of-N seconds per operation for every benchmarked backend.

    Each field maps an operation key (``kinship``, ``lmm_wald``,
    ``lmm_all``, ``lmm_wald_c4``) to its fastest observed time, or None
    when that backend did not run the operation.
    """

    gemma: OpTimings
    gemma_accel: OpTimings
    numpy_pure: OpTimings
    numpy: OpTimings
    numpy_streaming: OpTimings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_mouse_data():
    """Load mouse_hs1940 PLINK data and phenotypes."""
    from jamma.io import load_plink_binary

    plink = load_plink_binary(MOUSE_PREFIX)
    phenotypes = load_fam_phenotypes(MOUSE_PREFIX.with_suffix(".fam"))
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
def bench_gemma(gemma_path: Path, runs: int) -> OpTimings:
    """Benchmark GEMMA binary on mouse_hs1940."""
    results: OpTimings = {}

    ops = [
        ("kinship", ["-gk", "1"]),
        ("lmm_wald", ["-lmm", "1", "-k", str(MOUSE_KINSHIP)]),
        ("lmm_all", ["-lmm", "4", "-k", str(MOUSE_KINSHIP)]),
    ]
    if MOUSE_COVAR_4.exists():
        ops.append(
            (
                "lmm_wald_c4",
                ["-lmm", "1", "-k", str(MOUSE_KINSHIP), "-c", str(MOUSE_COVAR_4)],
            )
        )

    for op, args in ops:
        best = float("inf")
        for _ in range(runs):
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    str(gemma_path),
                    "-bfile",
                    str(MOUSE_PREFIX),
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
) -> OpTimings:
    """Benchmark NumPy backend with or without C acceleration."""
    import jamma.lmm.compute_numpy as cn
    from jamma.lmm.runner_numpy import run_lmm_association_numpy
    from jamma.lmm.schema import LmmConfig, LmmMode

    results: OpTimings = {}

    # compute_numpy is the single source of truth: chunk_runner_numpy reads
    # it live when it selects the dispatch path, so dropping the extension
    # here forces the NumPy fallback everywhere.
    cn_saved = cn._accel
    if disable_c:
        cn._accel = None

    try:
        ops: list[tuple[str, LmmMode, np.ndarray | None]] = [
            ("lmm_wald", 1, None),
            ("lmm_all", 4, None),
        ]
        if covariates_4 is not None:
            ops.append(("lmm_wald_c4", 1, covariates_4))

        for op, mode, covars in ops:

            def one_run(mode: LmmMode = mode, covars=covars):
                return run_lmm_association_numpy(
                    genotypes=plink.genotypes,
                    phenotypes=phenotypes,
                    kinship=kinship.copy(),
                    snp_info=snp_info,
                    covariates=covars,
                    config=LmmConfig(
                        show_progress=False, check_memory=False, lmm_mode=mode
                    ),
                )

            results[op] = best_of(one_run, runs)
    finally:
        cn._accel = cn_saved

    return results


def bench_numpy(
    plink, phenotypes, kinship, snp_info, covariates_4, runs: int
) -> OpTimings:
    """Benchmark NumPy+C backend."""
    return _bench_numpy_inner(
        plink, phenotypes, kinship, snp_info, covariates_4, runs, disable_c=False
    )


def bench_numpy_pure(
    plink, phenotypes, kinship, snp_info, covariates_4, runs: int
) -> OpTimings:
    """Benchmark pure NumPy backend (C extension disabled)."""
    return _bench_numpy_inner(
        plink, phenotypes, kinship, snp_info, covariates_4, runs, disable_c=True
    )


# ---------------------------------------------------------------------------
# JAMMA NumPy streaming benchmark
# ---------------------------------------------------------------------------
def bench_numpy_streaming(phenotypes, kinship, covariates_4, runs: int) -> OpTimings:
    """Benchmark NumPy streaming backend (disk I/O + C extension)."""
    from jamma.lmm.runner_numpy_streaming import run_lmm_association_numpy_streaming
    from jamma.lmm.schema import LmmConfig, LmmMode

    results: OpTimings = {}

    ops: list[tuple[str, LmmMode, np.ndarray | None]] = [
        ("lmm_wald", 1, None),
        ("lmm_all", 4, None),
    ]
    if covariates_4 is not None:
        ops.append(("lmm_wald_c4", 1, covariates_4))

    for op, mode, covars in ops:

        def one_run(mode: LmmMode = mode, covars=covars):
            return run_lmm_association_numpy_streaming(
                bed_path=MOUSE_PREFIX,
                phenotypes=phenotypes,
                kinship=kinship.copy(),
                covariates=covars,
                config=LmmConfig(
                    show_progress=False, check_memory=False, lmm_mode=mode
                ),
            )

        results[op] = best_of(one_run, runs)

    return results


# ---------------------------------------------------------------------------
# Kinship benchmark
# ---------------------------------------------------------------------------
def bench_kinship(plink, runs: int) -> OpTimings:
    """Benchmark kinship computation (NumPy/BLAS via compute_centered_kinship)."""
    from tests.reference.kinship import compute_centered_kinship

    # Warmup
    compute_centered_kinship(plink.genotypes, check_memory=False)

    best = best_of(
        lambda: compute_centered_kinship(plink.genotypes, check_memory=False), runs
    )
    return {"kinship": best}


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def load_inputs():
    """Load PLINK data, phenotypes, kinship, SNP metadata, and covariates.

    Returns:
        Tuple of ``(plink, phenotypes, kinship, snp_info, covariates_4)``.
    """
    from jamma.kinship.io import read_kinship_matrix

    print("Loading mouse_hs1940 data...", flush=True)
    plink, phenotypes = _load_mouse_data()
    print(f"  {plink.n_samples} samples, {plink.n_snps} SNPs")

    kinship = read_kinship_matrix(MOUSE_KINSHIP)
    snp_info = _build_snp_info(plink)
    covariates_4 = load_covariates_4()
    if covariates_4 is not None:
        print(
            f"  Covariates: {covariates_4.shape[1]} columns from {MOUSE_COVAR_4.name}"
        )
    print()

    return plink, phenotypes, kinship, snp_info, covariates_4


def run_benchmarks(
    gemma_path: Path | None,
    gemma_accel_path: Path | None,
    plink,
    phenotypes,
    kinship,
    snp_info,
    covariates_4,
    runs: int,
) -> Timing:
    """Run every backend sequentially and collect their best-of-N timings.

    Kinship is pure NumPy and BLAS, so its one timing is injected into both
    NumPy columns and left absent from the streaming column.

    Args:
        gemma_path: OpenBLAS GEMMA binary, or None to skip it.
        gemma_accel_path: Accelerate GEMMA binary, or None to skip it.
        plink: Loaded PLINK data.
        phenotypes: Per-sample phenotypes.
        kinship: Pre-computed kinship matrix for the LMM runs.
        snp_info: Per-SNP metadata for the batch runner.
        covariates_4: Covariate matrix, or None when the file is absent.
        runs: Repetitions per operation.

    Returns:
        A ``Timing`` holding every backend's per-operation seconds.
    """
    if gemma_path:
        print(f"Benchmarking GEMMA OpenBLAS ({gemma_path})...", flush=True)
        gemma = bench_gemma(gemma_path, runs)
    else:
        print("GEMMA not found, skipping (use --gemma-path to specify)")
        gemma = {"kinship": None, "lmm_wald": None, "lmm_all": None}

    if gemma_accel_path:
        print(f"Benchmarking GEMMA Accelerate ({gemma_accel_path})...", flush=True)
        gemma_accel = bench_gemma(gemma_accel_path, runs)
    else:
        print("GEMMA Accelerate not found, skipping (use --gemma-accelerate-path)")
        gemma_accel = {"kinship": None, "lmm_wald": None, "lmm_all": None}

    print("Benchmarking kinship (JAMMA)...", flush=True)
    kinship_times = bench_kinship(plink, runs)

    print("Benchmarking NumPy (pure Python, no C)...", flush=True)
    numpy_pure = bench_numpy_pure(
        plink, phenotypes, kinship, snp_info, covariates_4, runs
    )

    print("Benchmarking NumPy+C...", flush=True)
    numpy = bench_numpy(plink, phenotypes, kinship, snp_info, covariates_4, runs)

    print("Benchmarking NumPy streaming...", flush=True)
    numpy_streaming = bench_numpy_streaming(phenotypes, kinship, covariates_4, runs)
    numpy_streaming["kinship"] = None

    numpy_pure["kinship"] = kinship_times["kinship"]
    numpy["kinship"] = kinship_times["kinship"]

    print()

    return Timing(
        gemma=gemma,
        gemma_accel=gemma_accel,
        numpy_pure=numpy_pure,
        numpy=numpy,
        numpy_streaming=numpy_streaming,
    )


def print_results_table(timing: Timing, covariates_4) -> None:
    """Print the markdown comparison table.

    The ``vs GEMMA`` columns compare the fastest JAMMA backend for each
    operation against that GEMMA variant.

    Args:
        timing: Collected per-backend timings.
        covariates_4: Covariate matrix, or None; its presence adds a row.
    """

    def cell(t: float | None) -> str:
        return fmt_seconds(t) if t is not None else "—"

    def vs(t: float | None, op: str, ref: OpTimings) -> str:
        g = ref.get(op)
        if g is None or t is None:
            return "—"
        return f"{g / t:.1f}x"

    def c_speedup(op: str) -> str:
        pure = timing.numpy_pure.get(op)
        c = timing.numpy.get(op)
        if pure is None or c is None:
            return "—"
        return f"{pure / c:.1f}x"

    def best_jamma(op: str) -> float | None:
        valid = [
            c
            for c in (timing.numpy.get(op), timing.numpy_streaming.get(op))
            if c is not None
        ]
        return min(valid) if valid else None

    rows = [
        ("Kinship (`-gk 1`)", "kinship"),
        ("LMM Wald (`-lmm 1`)", "lmm_wald"),
        ("LMM All (`-lmm 4`)", "lmm_all"),
    ]
    if covariates_4 is not None:
        rows.append(("LMM Wald+4cov (`-lmm 1 -c`)", "lmm_wald_c4"))

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
        best = best_jamma(op)
        vs_ob = vs(best, op, timing.gemma) if best else "—"
        vs_ac = vs(best, op, timing.gemma_accel) if best else "—"
        print(
            f"| {label} | {cell(timing.gemma.get(op))}"
            f" | {cell(timing.gemma_accel.get(op))}"
            f" | {cell(timing.numpy_pure.get(op))} | {cell(timing.numpy.get(op))}"
            f" | {cell(timing.numpy_streaming.get(op))}"
            f" | {c_speedup(op)} | {vs_ob} | {vs_ac} |"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_gemma_args(parser)
    args = parser.parse_args()

    gemma_path = find_gemma(args.gemma_path, "gemma")
    gemma_accel_path = find_gemma(args.gemma_accelerate_path, "gemma-accelerate")

    if not MOUSE_PREFIX.with_suffix(".bed").exists():
        print(f"ERROR: mouse_hs1940 data not found at {MOUSE_DIR}", file=sys.stderr)
        sys.exit(1)

    print_hardware_header(args.runs)

    plink, phenotypes, kinship, snp_info, covariates_4 = load_inputs()

    timing = run_benchmarks(
        gemma_path,
        gemma_accel_path,
        plink,
        phenotypes,
        kinship,
        snp_info,
        covariates_4,
        args.runs,
    )

    print_results_table(timing, covariates_4)


if __name__ == "__main__":
    main()
