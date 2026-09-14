"""Interleaved native/process writer and fresh CLI benchmarks, with byte checks.

Run alone on an idle machine. Timings include atomic file publication, with
filesystem caching enabled and no fsync. The process comparison disables only
native formatter availability, exercising the retained generic writer.

    python scripts/bench_matrix_text.py --cases mouse square --cli --json result.json
    python scripts/bench_matrix_text.py --cases wide --repetitions 3 --json wide.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
from loguru import logger

from jamma.io import matrix_writer
from jamma.io._parallel_text import default_worker_count

_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def process_writer() -> Iterator[None]:
    """Select the fallback without changing numerical work or format semantics."""
    with patch.object(matrix_writer, "native_formatter", return_value=None):
        yield


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def summaries(timings: dict[str, list[float]]) -> dict:
    return {
        name: {"seconds": values, "median_seconds": statistics.median(values)}
        for name, values in timings.items()
    }


def benchmark_matrix(
    matrix: np.ndarray, directory: Path, repeats: int, workers: int
) -> dict:
    reference = directory / "reference.txt"
    np.savetxt(reference, matrix, fmt="%.10g", delimiter="\t")
    expected = digest(reference)
    output_bytes = reference.stat().st_size
    reference.unlink()
    timings: dict[str, list[float]] = {"process": [], "native": []}
    output = directory / "matrix.txt"
    for repeat in range(repeats):
        order = list(timings) if repeat % 2 == 0 else list(reversed(timings))
        for method in order:
            start = time.perf_counter()
            if method == "process":
                with process_writer():
                    matrix_writer.write_matrix_parallel(
                        matrix, output, n_workers=workers
                    )
            else:
                matrix_writer.write_matrix_parallel(matrix, output, n_workers=workers)
            elapsed = time.perf_counter() - start
            assert digest(output) == expected, (
                f"{method} output differs from np.savetxt"
            )
            timings[method].append(elapsed)
            output.unlink()
            print(
                f"{matrix.shape} {method} repeat {repeat + 1}: {elapsed:.6f}s",
                flush=True,
            )
    return {
        "shape": matrix.shape,
        "output_bytes": output_bytes,
        "sha256": expected,
        "methods": summaries(timings),
    }


def benchmark_cli(directory: Path, repeats: int) -> dict:
    prefix = _ROOT / "tests/fixtures/mouse_hs1940/mouse_hs1940"
    args = [
        "-bfile",
        str(prefix),
        "-gk",
        "1",
        "--legacy-text",
        "--no-telemetry",
        "-o",
        "matrix",
    ]
    timings: dict[str, list[float]] = {"process": [], "native": []}
    expected = None
    for repeat in range(repeats):
        order = list(timings) if repeat % 2 == 0 else list(reversed(timings))
        for method in order:
            output = directory / method
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--cli-child",
                method,
                *args,
                "-outdir",
                str(output),
            ]
            start = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            actual = digest(output / "matrix.cXX.txt")
            if expected is None:
                expected = actual
            assert actual == expected, f"{method} CLI output differs"
            timings[method].append(elapsed)
            print(f"CLI {method} repeat {repeat + 1}: {elapsed:.6f}s", flush=True)
    return {"sha256": expected, "methods": summaries(timings)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=["mouse", "square", "wide"],
        default=["mouse", "square"],
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.workers < 1:
        parser.error("repetitions and workers must be positive")
    logger.remove()
    if matrix_writer.native_formatter() is None:
        raise RuntimeError(
            "native formatter unavailable; compile it before benchmarking"
        )
    results = {
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "physical_cores": psutil.cpu_count(logical=False),
            "workers": args.workers,
            "load_at_start": os.getloadavg(),
            "note": "filesystem cache enabled; no fsync",
        },
        "cases": {},
    }
    rng = np.random.default_rng(376)
    with tempfile.TemporaryDirectory(prefix="jamma-matrix-bench-") as temporary:
        directory = Path(temporary)
        for name in args.cases:
            if name == "mouse":
                matrix = np.loadtxt(
                    _ROOT / "tests/fixtures/mouse_hs1940/mouse_hs1940_kinship.cXX.txt"
                )
            elif name == "square":
                matrix = rng.normal(0.0, 0.05, (5000, 5000))
                np.fill_diagonal(matrix, 0.5)
            else:
                # Enough rows to occupy 18 process workers despite their
                # minimum 100-row task size; avoid an underutilized baseline.
                matrix = rng.normal(0.0, 0.05, (2000, 100_000))
            results["cases"][name] = benchmark_matrix(
                matrix, directory, args.repetitions, args.workers
            )
            del matrix
        if args.cli:
            results["cli"] = benchmark_cli(directory, args.repetitions)
    args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--cli-child":
        from jamma.cli import main as cli_main

        method = sys.argv[2]
        sys.argv = ["jamma", *sys.argv[3:]]
        if method == "process":
            with process_writer():
                cli_main()
        else:
            cli_main()
    else:
        main()
