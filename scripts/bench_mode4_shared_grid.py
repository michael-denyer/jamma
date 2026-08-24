#!/usr/bin/env python3
"""Interleaved A/B benchmark for the fused n_cvt=1 mode-4 kernel.

Each revision runs in a persistent, warmed subprocess. Measurements alternate
in balanced ABBA/BAAB blocks so machine drift is not aliased with revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def build_inputs(
    n_samples: int, n_snps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build well-conditioned inputs for the fused kernel.

    The fused kernel forms Uab from the rotated vectors itself, so it is handed
    those plus the invariant block rather than a prebuilt varying SoA.
    """
    rng = np.random.default_rng(20260721)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    w = np.abs(rng.standard_normal(n_samples)) + 1.0
    y = rng.standard_normal(n_samples)
    utg_t = np.ascontiguousarray(rng.standard_normal((n_snps, n_samples)))

    uab_inv = np.ascontiguousarray(np.stack((w * w, w * y, y * y)))
    return eigenvalues, uab_inv, w, y, utg_t


def _digest_result(result: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(result):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(result[key]).tobytes())
    return digest.hexdigest()


def _worker(source_root: Path, n_samples: int, n_snps: int, n_threads: int) -> None:
    """Serve warmed benchmark measurements over a line-oriented protocol."""
    sys.path.insert(0, str(source_root / "src"))
    from jamma.lmm.compute_numpy import (
        compute_mode4_fused_c_ws,
        create_lmm_workspace_mode4_fused,
    )

    eigenvalues, uab_inv, w, Uty, utg_t = build_inputs(n_samples, n_snps)
    hi_eval_null = 1.0 / (eigenvalues + 1.0)
    workspace = create_lmm_workspace_mode4_fused(
        eigenvalues,
        uab_inv,
        w,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        n_threads,
        hi_eval_null=hi_eval_null,
        logl_H0=0.0,
    )

    # Exercise the full working set once before any timed command. A small
    # warmup leaves first-touch and OpenMP effects in the first measurement.
    compute_mode4_fused_c_ws(workspace, utg_t, n_threads)
    print("ready", flush=True)
    for command in sys.stdin:
        if command.strip() == "stop":
            return
        if command.strip() != "run":
            raise ValueError(f"unknown worker command: {command.strip()}")
        start = time.perf_counter()
        result = compute_mode4_fused_c_ws(workspace, utg_t, n_threads)
        elapsed = time.perf_counter() - start
        print(
            json.dumps({"seconds": elapsed, "output_sha256": _digest_result(result)}),
            flush=True,
        )


class WorkerProcess:
    """Persistent benchmark worker for one source tree."""

    def __init__(
        self,
        source_root: Path,
        n_samples: int,
        n_snps: int,
        n_threads: int,
    ) -> None:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source-root",
            str(source_root),
            "--samples",
            str(n_samples),
            "--snps",
            str(n_snps),
            "--threads",
            str(n_threads),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        if self.process.stdout is None or self.process.stdin is None:
            raise RuntimeError("benchmark worker pipes were not created")
        self.stdout = self.process.stdout
        self.stdin = self.process.stdin
        ready = self.stdout.readline().strip()
        if ready != "ready":
            self.close()
            raise RuntimeError(
                f"benchmark worker for {source_root} failed to start: {ready!r}"
            )

    def measure(self) -> tuple[float, str]:
        self.stdin.write("run\n")
        self.stdin.flush()
        payload = json.loads(self.stdout.readline())
        return float(payload["seconds"]), str(payload["output_sha256"])

    def close(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            self.process.stdin.write("stop\n")
            self.process.stdin.flush()
        self.process.wait(timeout=30)


def balanced_schedule(blocks: int) -> list[list[str]]:
    """Alternate ABBA and BAAB blocks to balance position across the session."""
    if blocks < 1:
        raise ValueError("blocks must be >= 1")
    return [
        ["A", "B", "B", "A"] if block % 2 == 0 else ["B", "A", "A", "B"]
        for block in range(blocks)
    ]


def _percent_change(after: float, before: float) -> float:
    return 100.0 * (after / before - 1.0)


def _git_revision(source_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def compare(
    a_root: Path,
    b_root: Path,
    n_samples: int,
    n_snps: int,
    n_threads: int,
    blocks: int,
) -> dict[str, object]:
    """Run the balanced comparison and return drift-visible summary data."""
    workers = {
        "A": WorkerProcess(a_root, n_samples, n_snps, n_threads),
        "B": WorkerProcess(b_root, n_samples, n_snps, n_threads),
    }
    timings: dict[str, list[float]] = {"A": [], "B": []}
    digests: dict[str, set[str]] = {"A": set(), "B": set()}
    block_deltas: list[float] = []
    all_blocks: list[list[float]] = []
    try:
        for order in balanced_schedule(blocks):
            block_timings: dict[str, list[float]] = {"A": [], "B": []}
            chronological: list[float] = []
            for label in order:
                elapsed, digest = workers[label].measure()
                timings[label].append(elapsed)
                block_timings[label].append(elapsed)
                chronological.append(elapsed)
                digests[label].add(digest)
            block_deltas.append(
                _percent_change(
                    statistics.median(block_timings["B"]),
                    statistics.median(block_timings["A"]),
                )
            )
            all_blocks.append(chronological)
    finally:
        workers["A"].close()
        workers["B"].close()

    if len(digests["A"]) != 1 or digests["A"] != digests["B"]:
        raise RuntimeError(f"output digests differ: A={digests['A']} B={digests['B']}")

    first_block = statistics.median(all_blocks[0])
    last_block = statistics.median(all_blocks[-1])
    median_a = statistics.median(timings["A"])
    median_b = statistics.median(timings["B"])
    paired_block_median = statistics.median(block_deltas)
    conclusion = (
        "no_stable_winner"
        if min(block_deltas) <= 0.0 <= max(block_deltas)
        else "consistent_direction_requires_replication"
    )
    return {
        "a_root": str(a_root),
        "b_root": str(b_root),
        "a_revision": _git_revision(a_root),
        "b_revision": _git_revision(b_root),
        "samples": n_samples,
        "snps": n_snps,
        "threads": n_threads,
        "blocks": blocks,
        "measurements_per_revision": len(timings["A"]),
        "median_a_seconds": median_a,
        "median_b_seconds": median_b,
        "b_vs_a_percent": _percent_change(median_b, median_a),
        "paired_block_median_percent": paired_block_median,
        "block_delta_min_percent": min(block_deltas),
        "block_delta_max_percent": max(block_deltas),
        "session_drift_percent": _percent_change(last_block, first_block),
        "conclusion": conclusion,
        "output_sha256": next(iter(digests["A"])),
        "a_timings_seconds": timings["A"],
        "b_timings_seconds": timings["B"],
        "block_deltas_percent": block_deltas,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-root", type=Path)
    parser.add_argument("--b-root", type=Path)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1_410)
    parser.add_argument("--snps", type=int, default=10_768)
    parser.add_argument("--threads", type=int, default=18)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker:
        if args.source_root is None:
            raise SystemExit("--worker requires --source-root")
        _worker(args.source_root.resolve(), args.samples, args.snps, args.threads)
        return

    schedule = balanced_schedule(args.blocks)
    if args.dry_run:
        for index, order in enumerate(schedule, start=1):
            print(f"block={index} order={','.join(order)}")
        return

    if args.a_root is None or args.b_root is None:
        raise SystemExit("comparison requires --a-root and --b-root")
    result = compare(
        args.a_root.resolve(),
        args.b_root.resolve(),
        args.samples,
        args.snps,
        args.threads,
        args.blocks,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
