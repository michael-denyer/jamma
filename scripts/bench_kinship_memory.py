"""Measure the streaming kinship preprocessing peak against its memory quote.

Run sequentially with other benchmarks. Tracemalloc includes NumPy allocations
but excludes native BLAS scratch; the quote includes that backend's declaration.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
from loguru import logger

from jamma import jlinalg
from jamma.core.eigen_plan import array_gb, square_matrix_gb
from jamma.core.memory import kinship_cost
from jamma.io.plink import get_plink_metadata, parse_fam_phenotype_column
from jamma.kinship import compute_kinship_streaming


def measure(bfile: Path, chunk_size: int) -> dict[str, str | int | float]:
    """Compute centered kinship over the first phenotype's analyzed samples."""
    meta = get_plink_metadata(bfile)
    fam = np.loadtxt(Path(f"{bfile}.fam"), dtype=str, ndmin=2)
    phenotype = parse_fam_phenotype_column(fam, 1)
    indices = np.flatnonzero(np.isfinite(phenotype))
    selected = None if len(indices) == meta.n_samples else indices
    quote = kinship_cost(
        square_matrix_gb(len(indices)),
        array_gb(meta.n_samples, min(chunk_size, meta.n_snps)),
        jlinalg.dsyrk_scratch_bytes(len(indices)) / 1e9,
    )

    tracemalloc.start()
    start = time.perf_counter()
    try:
        matrix = compute_kinship_streaming(
            bfile,
            chunk_size=chunk_size,
            maf_threshold=0.01,
            miss_threshold=0.05,
            check_memory=False,
            show_progress=False,
            valid_indices=selected,
            filter_sample_indices=selected,
        )
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return {
        "backend": jlinalg.blas_backend,
        "n_input_samples": meta.n_samples,
        "n_analyzed_samples": matrix.shape[0],
        "n_snps": meta.n_snps,
        "chunk_size": chunk_size,
        "elapsed_s": elapsed,
        "traced_peak_bytes": peak,
        "quoted_bytes": quote * 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bfile",
        type=Path,
        default=Path("tests/fixtures/mouse_hs1940/mouse_hs1940"),
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    logger.remove()
    print(json.dumps(measure(args.bfile, args.chunk_size), indent=2))


if __name__ == "__main__":
    main()
