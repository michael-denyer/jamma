"""Measure time and traced peak memory of the association and kinship kernels.

Subcommands:

- ``impute``: association chunk imputation on C and F layouts at three
  missing rates.
- ``snp-stats``: native SNP statistics on C and F inputs, including boundary
  copies. Needs the jlinalg C extension. The C/F/F/C order balances warm caches.
- ``kinship``: the streaming kinship preprocessing peak against its memory
  quote, over the first phenotype's analysed samples.

Run sequentially with other benchmarks. Tracemalloc sees NumPy allocations
but not native BLAS or C scratch; the kinship quote includes that backend's
declaration. Output is JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from _bench_common import MOUSE_PREFIX, traced_peak
from loguru import logger

from jamma import jlinalg
from jamma.genotype.dataset import GenotypeDataset
from jamma.io.plink import parse_fam_phenotype_column
from jamma.kinship import compute_kinship_streaming
from jamma.kinship.memory import estimate_kinship_memory
from jamma.lmm.impute import impute_missing_inplace


def measure_impute() -> list[dict]:
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
            _, _, peak = traced_peak(impute_missing_inplace, values, means)
            records.append(
                {
                    "shape": list(original.shape),
                    "order": order,
                    "missing_rate": rate,
                    "median_s": statistics.median(timings),
                    "traced_peak_bytes": peak,
                }
            )
    return records


def measure_snp_stats(n_samples: int, n_snps: int, threads: int) -> list[dict]:
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
                _, _, peak = traced_peak(
                    jlinalg.compute_snp_stats_chunk, arrays[order], *outputs[order]
                )
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


def measure_kinship(bfile: Path, chunk_size: int) -> dict[str, str | int | float]:
    logger.remove()
    dataset = GenotypeDataset.open_plink(bfile)
    fam = np.loadtxt(Path(f"{bfile}.fam"), dtype=str, ndmin=2)
    phenotype = parse_fam_phenotype_column(fam, 1)
    indices = np.flatnonzero(np.isfinite(phenotype))
    selected = None if len(indices) == dataset.n_samples else indices
    quote = estimate_kinship_memory(
        n_input_samples=dataset.n_samples,
        n_output_samples=len(indices),
        n_snps=dataset.n_variants,
        chunk_size=chunk_size,
    )
    matrix, elapsed, peak = traced_peak(
        lambda: compute_kinship_streaming(
            dataset,
            chunk_size=chunk_size,
            maf_threshold=0.01,
            miss_threshold=0.05,
            check_memory=False,
            show_progress=False,
            valid_indices=selected,
            filter_sample_indices=selected,
        )
    )
    return {
        "backend": jlinalg.blas_backend,
        "n_input_samples": dataset.n_samples,
        "n_analyzed_samples": matrix.shape[0],
        "n_snps": dataset.n_variants,
        "chunk_size": chunk_size,
        "elapsed_s": elapsed,
        "traced_peak_bytes": peak,
        "quoted_bytes": quote * 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="kernel", required=True)
    sub.add_parser("impute")
    snp = sub.add_parser("snp-stats")
    snp.add_argument("--samples", type=int, default=1940)
    snp.add_argument("--snps", type=int, default=10_000)
    snp.add_argument("--threads", type=int, nargs="+", default=[1, 3])
    kin = sub.add_parser("kinship")
    kin.add_argument("--bfile", type=Path, default=MOUSE_PREFIX)
    kin.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()

    if args.kernel == "impute":
        result: object = measure_impute()
    elif args.kernel == "snp-stats":
        if min(args.samples, args.snps, *args.threads) < 1:
            parser.error("dimensions and thread counts must be positive")
        if not jlinalg.HAS_C_EXTENSION:
            parser.error("jlinalg C extension required")
        result = [
            record
            for threads in args.threads
            for record in measure_snp_stats(args.samples, args.snps, threads)
        ]
    else:
        if args.chunk_size < 1:
            parser.error("--chunk-size must be positive")
        result = measure_kinship(args.bfile, args.chunk_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
