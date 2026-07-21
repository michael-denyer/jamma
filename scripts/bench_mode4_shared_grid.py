#!/usr/bin/env python3
"""Benchmark the split/SoA n_cvt=1 mode-4 kernel on deterministic inputs."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import numpy as np

from jamma.lmm.compute_numpy import (
    compute_mode4_split_c_ws,
    create_lmm_workspace_mode4,
)


def build_inputs(
    n_samples: int, n_snps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build well-conditioned split Uab arrays without a full AoS copy."""
    rng = np.random.default_rng(20260721)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    w = np.abs(rng.standard_normal(n_samples)) + 1.0
    y = rng.standard_normal(n_samples)
    x = rng.standard_normal((n_snps, n_samples))

    uab_inv = np.ascontiguousarray(np.stack((w * w, w * y, y * y)))
    uab_var = np.empty((n_snps, 3, n_samples), dtype=np.float64)
    uab_var[:, 0, :] = x * w
    uab_var[:, 1, :] = x * x
    uab_var[:, 2, :] = x * y
    return eigenvalues, uab_inv, uab_var


def benchmark(
    n_samples: int, n_snps: int, n_threads: int, runs: int
) -> tuple[float, float, list[float], str]:
    eigenvalues, uab_inv, uab_var = build_inputs(n_samples, n_snps)
    hi_eval_null = 1.0 / (eigenvalues + 1.0)
    workspace = create_lmm_workspace_mode4(
        eigenvalues,
        uab_inv,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        n_threads,
        hi_eval_null,
        0.0,
    )

    compute_mode4_split_c_ws(workspace, uab_var[: min(32, n_snps)], n_threads)
    timings = []
    output_digest = ""
    for _ in range(runs):
        start = time.perf_counter()
        result = compute_mode4_split_c_ws(workspace, uab_var, n_threads)
        timings.append(time.perf_counter() - start)
        digest = hashlib.sha256()
        for key in sorted(result):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(result[key]).tobytes())
        output_digest = digest.hexdigest()
    return min(timings), statistics.median(timings), timings, output_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_410)
    parser.add_argument("--snps", type=int, default=10_768)
    parser.add_argument("--threads", type=int, default=18)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    best, median, timings, output_digest = benchmark(
        args.samples, args.snps, args.threads, args.runs
    )
    raw_timings = ",".join(f"{timing:.6f}" for timing in timings)
    print(
        f"samples={args.samples} snps={args.snps} threads={args.threads} "
        f"best_seconds={best:.6f} median_seconds={median:.6f} "
        f"output_sha256={output_digest} timings_seconds={raw_timings}"
    )


if __name__ == "__main__":
    main()
