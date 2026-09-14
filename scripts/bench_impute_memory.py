"""Measure association imputation time and scratch on real chunk layouts."""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc

import numpy as np

from jamma.lmm.impute import impute_missing_inplace


def main() -> None:
    rng = np.random.default_rng(204)
    records = []
    for order in ("C", "F"):
        for rate in (0.0, 0.01, 0.2):
            original = np.array(
                rng.integers(0, 3, (4000, 2000)), dtype=float, order=order
            )
            original[rng.random(original.shape) < rate] = np.nan
            means = np.nanmean(original, axis=0)
            expected = np.where(np.isnan(original), means, original)
            timings = []
            for _ in range(7):
                values = original.copy(order=order)
                start = time.perf_counter()
                impute_missing_inplace(values, means)
                timings.append(time.perf_counter() - start)
            np.testing.assert_array_equal(values, expected)
            values = original.copy(order=order)
            tracemalloc.start()
            try:
                impute_missing_inplace(values, means)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            records.append(
                {
                    "shape": list(original.shape),
                    "order": order,
                    "missing_rate": rate,
                    "median_s": statistics.median(timings),
                    "traced_peak_bytes": peak,
                }
            )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
