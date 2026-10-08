#!/usr/bin/env python3
"""Time ``jlinalg.eigh`` across sample counts and fit the wall-time power law.

Produces the coefficient and exponent of a per-backend eigendecomposition
estimate in ``jamma.estimates``: wall seconds = coeff * (n / 1000) ** alpha,
fitted by ordinary least squares on log(median wall) against log(n / 1000).

Each timed call is the one ``timed_progress`` wraps in production, on a
kinship-shaped matrix K = G G^T / p with p = ``--snp-ratio`` * n. Rounds visit
every size in turn, so drift during the run spreads across sizes.

Usage:
    uv run python scripts/bench_eigendecomp_scaling.py
    uv run python scripts/bench_eigendecomp_scaling.py --samples 2000 4000 --rounds 5

Run it alone on an idle machine. Other load contaminates the timings.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from jamma import jlinalg


def _kinship(n: int, snp_ratio: int, rng: np.random.Generator) -> np.ndarray:
    """Build K = G G^T / p for p = snp_ratio * n, one n-column block at a time."""
    K = np.zeros((n, n))
    for _ in range(snp_ratio):
        block = rng.standard_normal((n, n))
        K += block @ block.T
    K /= snp_ratio * n
    return K


def _time_eigh(n: int, snp_ratio: int, seed: int) -> dict[str, float | int | str]:
    """Run one solve and return its wall time, CPU time and driver."""
    K = _kinship(n, snp_ratio, np.random.default_rng(seed))
    trace = float(np.trace(K))

    cpu_start = time.process_time()
    start = time.perf_counter()
    eigenvalues, _, status = jlinalg.eigh(K)
    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu_start

    if not math.isclose(float(eigenvalues.sum()), trace, rel_tol=1e-8):
        raise RuntimeError(
            f"n={n}: eigenvalue sum {eigenvalues.sum()!r} != trace {trace!r}"
        )
    return {"n": n, "wall_s": wall, "cpu_s": cpu, "driver": status.driver_used}


def _fit_power_law(medians: dict[int, float]) -> tuple[float, float]:
    """Return (coeff, alpha) of the log-linear least-squares fit."""
    log_n = np.log([n / 1000 for n in medians])
    log_wall = np.log(list(medians.values()))
    alpha, log_coeff = np.polyfit(log_n, log_wall, 1)
    return float(np.exp(log_coeff)), float(alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[4_000, 6_000, 8_000, 10_000, 14_000, 20_000],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--snp-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Write runs and fit as JSON.")
    args = parser.parse_args()

    print(f"backend: {jlinalg.blas_backend}")
    runs: list[dict[str, float | int | str]] = []
    for round_index in range(args.rounds):
        for n in args.samples:
            run = _time_eigh(n, args.snp_ratio, args.seed + round_index)
            runs.append(run)
            print(
                f"round {round_index + 1} n={n}: wall={run['wall_s']:.2f}s "
                f"cpu={run['cpu_s']:.2f}s driver={run['driver']}",
                flush=True,
            )

    medians: dict[int, float] = {}
    print("\n      n  median_s     min_s     max_s  median/n_k^3")
    for n in args.samples:
        walls = [float(run["wall_s"]) for run in runs if run["n"] == n]
        medians[n] = statistics.median(walls)
        print(
            f"{n:7d} {medians[n]:9.2f} {min(walls):9.2f} {max(walls):9.2f} "
            f"{medians[n] / (n / 1000) ** 3:13.4f}"
        )

    fit: dict[str, float] = {}
    if len(medians) > 1:
        coeff, alpha = _fit_power_law(medians)
        worst = max(
            abs(coeff * (n / 1000) ** alpha / median - 1)
            for n, median in medians.items()
        )
        fit = {"coeff": coeff, "alpha": alpha, "max_rel_error": worst}
        print(
            f"\nfit: wall_s = {coeff:.6f} * n_k^{alpha:.4f} "
            f"(max error {worst:.1%} against the medians)"
        )

    if args.output is not None:
        payload = {"backend": str(jlinalg.blas_backend), "runs": runs, "fit": fit}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
