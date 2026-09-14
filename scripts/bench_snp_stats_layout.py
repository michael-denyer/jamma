"""Measure native SNP statistics on C/F inputs, including boundary copies.

Run before and after a native rebuild, sequentially with other benchmarks.
The C/F/F/C order balances warm caches; allocations exclude native scratch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from jamma import jlinalg


def measure(n_samples: int, n_snps: int, threads: int) -> list[dict]:
    rng = np.random.default_rng(194)
    records = []
    old_threads = jlinalg.set_n_threads(threads)
    try:
        for dtype in (np.float32, np.float64):
            values = rng.integers(0, 3, (n_samples, n_snps)).astype(dtype)
            values[::17, ::7] = np.nan
            arrays = {"C": values, "F": np.asfortranarray(values)}
            outputs = {
                order: [np.empty(n_snps, dtype=d) for d in (float, np.intp, float)]
                for order in arrays
            }
            timings: dict[str, list[float]] = {"C": [], "F": []}
            for order in "CFCF":
                jlinalg.compute_snp_stats_chunk(arrays[order], *outputs[order])
            for _ in range(5):
                for order in "CFFC":
                    start = time.perf_counter()
                    jlinalg.compute_snp_stats_chunk(arrays[order], *outputs[order])
                    timings[order].append(time.perf_counter() - start)
            for c, f in zip(outputs["C"], outputs["F"], strict=True):
                np.testing.assert_array_equal(c, f)
            for order in arrays:
                tracemalloc.start()
                try:
                    jlinalg.compute_snp_stats_chunk(arrays[order], *outputs[order])
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                records.append(
                    {
                        "shape": [n_samples, n_snps],
                        "dtype": np.dtype(dtype).name,
                        "threads": threads,
                        "order": order,
                        "median_s": statistics.median(timings[order]),
                        "traced_peak_bytes": peak,
                    }
                )
    finally:
        jlinalg.set_n_threads(old_threads)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1940)
    parser.add_argument("--snps", type=int, default=10_000)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()
    if min(args.samples, args.snps, *args.threads) < 1:
        parser.error("dimensions and thread counts must be positive")
    if not jlinalg.HAS_C_EXTENSION:
        parser.error("jlinalg C extension required")
    records = []
    for threads in args.threads:
        records.extend(measure(args.samples, args.snps, threads))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
