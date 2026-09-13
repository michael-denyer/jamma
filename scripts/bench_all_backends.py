#!/usr/bin/env python3
"""End-to-end backend comparison benchmark on mouse_hs1940.

Runs kinship (-gk 1), LMM Wald (-lmm 1), and LMM All (-lmm 4) across
fresh JAMMA processes (batch, streaming, pure-Python) and GEMMA, then prints
a formatted table matching the README.

Usage:
    uv run python scripts/bench_all_backends.py
    uv run python scripts/bench_all_backends.py --gemma-path /path/to/gemma
    uv run python scripts/bench_all_backends.py --runs 3

Backends run sequentially to avoid cross-contamination of timings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _bench_common import (
    MOUSE_COVAR_4,
    MOUSE_DIR,
    MOUSE_KINSHIP,
    MOUSE_PREFIX,
    add_gemma_args,
    find_gemma,
    fmt_seconds,
    print_hardware_header,
    read_associations,
    run_commands,
    verify_associations,
)

OpTimings = dict[str, float | None]


@dataclass(frozen=True)
class Timing:
    """Best-of-N seconds per operation for every benchmarked backend.

    Each field maps an operation key (``kinship``, ``lmm_wald``,
    ``lmm_all``, ``lmm_wald_c4``, ``gwas_wald``) to its fastest observed time, or None
    when that backend did not run the operation.
    """

    gemma: OpTimings
    gemma_accel: OpTimings
    numpy_pure: OpTimings
    numpy: OpTimings
    numpy_streaming: OpTimings


def operation_args(op: str) -> list[str]:
    """Identical statistical options and disk inputs for both programs."""
    if op == "kinship":
        return ["-gk", "1"]
    args = ["-lmm", "4" if op == "lmm_all" else "1"]
    if op != "gwas_wald":
        args += ["-k", str(MOUSE_KINSHIP)]
    if op == "lmm_wald_c4":
        args += ["-c", str(MOUSE_COVAR_4)]
    return args


def commands_for(
    executable: list[str], op: str, outdir: Path, backend: str | None
) -> list[list[str]]:
    """Build complete workflows, retaining JAMMA's in-memory kinship benefit."""
    common = ["-bfile", str(MOUSE_PREFIX), "-o", "bench", "-outdir", str(outdir)]
    if backend is not None and op == "gwas_wald":
        return [
            [
                sys.executable,
                "-c",
                "import sys; from jamma import gwas; "
                "gwas(sys.argv[1], output_dir=sys.argv[2], output_prefix='bench', "
                "backend=sys.argv[3], no_telemetry=True)",
                str(MOUSE_PREFIX),
                str(outdir),
                backend,
            ]
        ]
    options = operation_args(op)
    if backend is not None:
        options += ["--backend", backend, "--no-telemetry"]
        if op == "kinship":
            options += ["--legacy-text"]
    if op == "gwas_wald":
        return [
            executable + common + ["-gk", "1"],
            executable + common + options + ["-k", str(outdir / "bench.cXX.txt")],
        ]
    return [executable + common + options]


def run_benchmarks(
    gemma_path: Path | None, gemma_accel_path: Path | None, runs: int
) -> tuple[Timing, list[dict]]:
    """Time disk-to-disk workflows, rotating backend order between rounds.

    Every invocation starts a new process. Setup and result validation are
    outside the timer; imports, loading, computation and writing are inside.
    """
    variants: list[tuple[str, list[str], str | None, bool]] = []
    for name, path in (("gemma", gemma_path), ("gemma_accel", gemma_accel_path)):
        if path is not None:
            variants.append((name, [str(path)], None, False))
    variants.extend(
        [
            ("numpy_pure", [sys.executable, "-m", "jamma"], "numpy", True),
            ("numpy", [sys.executable, "-m", "jamma"], "numpy", False),
            (
                "numpy_streaming",
                [sys.executable, "-m", "jamma"],
                "numpy-streaming",
                False,
            ),
        ]
    )
    timings: dict[str, OpTimings] = {name: {} for name in Timing.__annotations__}
    records: list[dict] = []
    ops = ["kinship", "lmm_wald", "lmm_all", "gwas_wald"]
    if MOUSE_COVAR_4.exists():
        ops.append("lmm_wald_c4")
    for op in ops:
        reference = None
        reference_matrix = None
        for repetition in range(runs):
            ordered = (
                variants[repetition % len(variants) :]
                + variants[: repetition % len(variants)]
            )
            for name, executable, backend, pure in ordered:
                if op == "kinship" and name == "numpy_streaming":
                    continue
                print(f"{op}: {name}, run {repetition + 1}/{runs}", flush=True)
                env = dict(os.environ)
                env.pop("JAMMA_FORCE_NUMPY_FALLBACK", None)
                if pure:
                    env["JAMMA_FORCE_NUMPY_FALLBACK"] = "1"
                with tempfile.TemporaryDirectory() as tmpdir:
                    outdir = Path(tmpdir)
                    commands = commands_for(executable, op, outdir, backend)
                    elapsed = run_commands(commands, env)
                    if (
                        op == "gwas_wald"
                        and backend is not None
                        and list(outdir.glob("*.cXX.*"))
                    ):
                        raise ValueError("Full GWAS unexpectedly saved kinship")
                    if op == "kinship":
                        matrix = np.loadtxt(outdir / "bench.cXX.txt")
                        if reference_matrix is None:
                            reference_matrix = matrix
                        else:
                            np.testing.assert_allclose(
                                matrix, reference_matrix, rtol=1e-8, atol=1e-14
                            )
                    else:
                        associations = read_associations(outdir / "bench.assoc.txt")
                        if reference is None:
                            reference = associations
                        else:
                            verify_associations(reference, associations)
                    records.append(
                        {
                            "operation": op,
                            "backend": name,
                            "repetition": repetition + 1,
                            "seconds": elapsed,
                            "commands": commands,
                        }
                    )
                    previous = timings[name].get(op)
                    timings[name][op] = (
                        min(previous, elapsed) if previous is not None else elapsed
                    )
    return Timing(**timings), records


def print_results_table(timing: Timing, covariates_4: bool) -> None:
    """Print the markdown comparison table.

    The ``vs GEMMA`` columns compare the fastest JAMMA backend for each
    operation against that GEMMA variant.

    Args:
        timing: Collected per-backend timings.
        covariates_4: Whether to include the four-covariate row.
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
        ("Full GWAS Wald (compute kinship + association)", "gwas_wald"),
    ]
    if covariates_4:
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
    parser.add_argument("--json", type=Path, help="Save raw timings and commands")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    gemma_path = find_gemma(args.gemma_path, "gemma")
    gemma_accel_path = find_gemma(args.gemma_accelerate_path, "gemma-accelerate")

    if not MOUSE_PREFIX.with_suffix(".bed").exists():
        print(f"ERROR: mouse_hs1940 data not found at {MOUSE_DIR}", file=sys.stderr)
        sys.exit(1)

    print_hardware_header(args.runs)

    timing, records = run_benchmarks(gemma_path, gemma_accel_path, args.runs)
    print_results_table(timing, MOUSE_COVAR_4.exists())
    if args.json:
        args.json.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
