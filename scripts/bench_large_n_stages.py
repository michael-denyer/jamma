#!/usr/bin/env python3
"""Interleaved A/B timing for the large-N numerical stages.

Each revision runs in a separate, persistent worker. For every stage the
workers alternate in ABBA/BAAB blocks, which keeps time-on-machine from being
mistaken for a code change. The workers hash numerical results and stop when
the revisions disagree.

Examples:
    uv run python scripts/bench_large_n_stages.py \
        --a-root /path/to/base --b-root /path/to/candidate
    uv run python scripts/bench_large_n_stages.py \
        --a-root . --b-root . --samples 256 --snps 128 --blocks 1

The default is intentionally substantial but not a 100k-sample claim. Pick a
sample count that fits the target machine and measure at least two block sizes
before treating a result as a performance conclusion.
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

_STAGES = ("kinship", "eigen", "rotation", "mode4")


def _digest(value: object) -> str:
    """Return a shape- and dtype-sensitive digest for numerical results."""
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"array\0")
            digest.update(array.dtype.str.encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                digest.update(key.encode())
                update(item[key])
        elif isinstance(item, tuple):
            digest.update(b"tuple\0")
            for child in item:
                update(child)
        else:
            raise TypeError(
                f"cannot hash benchmark result of type {type(item).__name__}"
            )

    update(value)
    return digest.hexdigest()


def _build_inputs(
    n_samples: int, n_snps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic, well-conditioned inputs for all timed stages."""
    rng = np.random.default_rng(20260801)
    genotypes = np.ascontiguousarray(rng.standard_normal((n_samples, n_snps)))
    genotypes -= genotypes.mean(axis=0)

    w = np.abs(rng.standard_normal(n_samples)) + 1.0
    y = rng.standard_normal(n_samples)
    uab_invariant = np.ascontiguousarray(np.stack((w * w, w * y, y * y)))
    uab_varying = np.empty((n_snps, 3, n_samples), dtype=np.float64)
    snp_by_sample = genotypes.T
    uab_varying[:, 0, :] = snp_by_sample * w
    uab_varying[:, 1, :] = snp_by_sample * snp_by_sample
    uab_varying[:, 2, :] = snp_by_sample * y
    return genotypes, w, y, uab_invariant, uab_varying


class _StageRunner:
    """Own stage inputs and reusable output buffers inside one worker."""

    def __init__(self, n_samples: int, n_snps: int, n_threads: int) -> None:
        from jamma import jlinalg
        from jamma.lmm.compute_numpy import (
            compute_mode4_split_c_ws,
            create_lmm_workspace_mode4,
        )

        self._jlinalg = jlinalg
        self._compute_mode4 = compute_mode4_split_c_ws
        self._n_threads = n_threads
        (
            self._genotypes,
            _w,
            _y,
            uab_invariant,
            self._uab_varying,
        ) = _build_inputs(n_samples, n_snps)

        self._kinship_out = np.empty((n_samples, n_samples), dtype=np.float64)
        self._jlinalg.dsyrk(self._genotypes, out=self._kinship_out, beta=0.0)
        self._kinship_seed = self._kinship_out.copy()
        self._kinship_seed /= n_snps
        self._kinship_seed.flat[:: n_samples + 1] += 1e-6

        eigen_work = self._kinship_seed.copy()
        self._eigenvalues, self._eigenvectors = self._jlinalg.eigh(
            eigen_work, inplace=True
        )
        self._eigen_work = np.empty_like(self._kinship_seed)
        self._rotation_out = np.empty((n_snps, n_samples), dtype=np.float64)
        hi_eval_null = 1.0 / (self._eigenvalues + 1.0)
        self._mode4_workspace = create_lmm_workspace_mode4(
            self._eigenvalues,
            uab_invariant,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            n_threads,
            hi_eval_null,
            0.0,
        )

    def run(self, stage: str) -> object:
        if stage == "kinship":
            return self._run_kinship()
        if stage == "eigen":
            return self._run_eigen()
        if stage == "rotation":
            return self._run_rotation()
        if stage == "mode4":
            return self._compute_mode4(
                self._mode4_workspace, self._uab_varying, self._n_threads
            )
        raise ValueError(f"unknown benchmark stage: {stage}")

    def _run_kinship(self) -> np.ndarray:
        self._jlinalg.dsyrk(self._genotypes, out=self._kinship_out, beta=0.0)
        return self._kinship_out

    def _run_eigen(self) -> tuple[np.ndarray, np.ndarray]:
        np.copyto(self._eigen_work, self._kinship_seed)
        return self._jlinalg.eigh(self._eigen_work, inplace=True)

    def _run_rotation(self) -> np.ndarray:
        return self._jlinalg.dgemm(
            self._genotypes,
            self._eigenvectors,
            transa="T",
            out=self._rotation_out,
        )


def _worker(source_root: Path, n_samples: int, n_snps: int, n_threads: int) -> None:
    """Serve warmed stage timings over a line-oriented protocol."""
    sys.path.insert(0, str(source_root / "src"))
    runner = _StageRunner(n_samples, n_snps, n_threads)

    for stage in _STAGES:
        runner.run(stage)
    print("ready", flush=True)

    for command in sys.stdin:
        stage = command.strip()
        if stage == "stop":
            return
        if stage not in _STAGES:
            raise ValueError(f"unknown worker command: {stage}")
        started = time.perf_counter()
        result = runner.run(stage)
        elapsed = time.perf_counter() - started
        print(json.dumps({"seconds": elapsed, "sha256": _digest(result)}), flush=True)


class _WorkerProcess:
    """One persistent worker, isolated to one source tree."""

    def __init__(
        self, source_root: Path, n_samples: int, n_snps: int, n_threads: int
    ) -> None:
        self._source_root = source_root
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
            stderr=subprocess.PIPE,
            text=True,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("benchmark worker pipes were not created")
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        if self.stdout.readline().strip() != "ready":
            error = self._read_error()
            self.close()
            raise RuntimeError(
                f"benchmark worker {source_root} failed to start: {error}"
            )

    def measure(self, stage: str) -> tuple[float, str]:
        self.stdin.write(f"{stage}\n")
        self.stdin.flush()
        line = self.stdout.readline()
        if not line:
            raise RuntimeError(
                f"benchmark worker {self._source_root} exited while timing {stage}: "
                f"{self._read_error()}"
            )
        payload = json.loads(line)
        return float(payload["seconds"]), str(payload["sha256"])

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.stdin.write("stop\n")
                self.stdin.flush()
            except BrokenPipeError:
                pass
        self.process.wait(timeout=30)

    def _read_error(self) -> str:
        if self.process.stderr is None:
            return "no stderr available"
        return self.process.stderr.read().strip() or "no worker error output"


def _balanced_schedule(blocks: int) -> list[list[str]]:
    if blocks < 1:
        raise ValueError("blocks must be >= 1")
    return [
        ["A", "B", "B", "A"] if index % 2 == 0 else ["B", "A", "A", "B"]
        for index in range(blocks)
    ]


def _percent_change(after: float, before: float) -> float:
    return 100.0 * (after / before - 1.0)


def _revision(source_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _summarize_stage(
    stage: str, workers: dict[str, _WorkerProcess], schedule: list[list[str]]
) -> dict[str, object]:
    timings: dict[str, list[float]] = {"A": [], "B": []}
    digests: dict[str, set[str]] = {"A": set(), "B": set()}
    deltas: list[float] = []
    block_medians: list[float] = []

    for order in schedule:
        by_revision: dict[str, list[float]] = {"A": [], "B": []}
        chronological: list[float] = []
        for revision in order:
            seconds, digest = workers[revision].measure(stage)
            timings[revision].append(seconds)
            by_revision[revision].append(seconds)
            chronological.append(seconds)
            digests[revision].add(digest)
        deltas.append(
            _percent_change(
                statistics.median(by_revision["B"]),
                statistics.median(by_revision["A"]),
            )
        )
        block_medians.append(statistics.median(chronological))

    if len(digests["A"]) != 1 or digests["A"] != digests["B"]:
        raise RuntimeError(
            f"{stage} output mismatch: A={digests['A']} B={digests['B']}"
        )

    median_a = statistics.median(timings["A"])
    median_b = statistics.median(timings["B"])
    return {
        "median_a_seconds": median_a,
        "median_b_seconds": median_b,
        "b_vs_a_percent": _percent_change(median_b, median_a),
        "paired_block_median_percent": statistics.median(deltas),
        "block_delta_min_percent": min(deltas),
        "block_delta_max_percent": max(deltas),
        "session_drift_percent": _percent_change(block_medians[-1], block_medians[0]),
        "conclusion": (
            "no_stable_winner"
            if min(deltas) <= 0.0 <= max(deltas)
            else "consistent_direction_requires_replication"
        ),
        "output_sha256": next(iter(digests["A"])),
        "a_timings_seconds": timings["A"],
        "b_timings_seconds": timings["B"],
        "block_deltas_percent": deltas,
    }


def compare(
    a_root: Path,
    b_root: Path,
    n_samples: int,
    n_snps: int,
    n_threads: int,
    blocks: int,
    stages: tuple[str, ...],
) -> dict[str, object]:
    """Compare every requested stage with balanced, correctness-checked A/B runs."""
    workers: dict[str, _WorkerProcess] = {}
    schedule = _balanced_schedule(blocks)
    try:
        workers["A"] = _WorkerProcess(a_root, n_samples, n_snps, n_threads)
        workers["B"] = _WorkerProcess(b_root, n_samples, n_snps, n_threads)
        stage_results = {
            stage: _summarize_stage(stage, workers, schedule) for stage in stages
        }
    finally:
        for worker in workers.values():
            worker.close()

    return {
        "a_root": str(a_root),
        "b_root": str(b_root),
        "a_revision": _revision(a_root),
        "b_revision": _revision(b_root),
        "samples": n_samples,
        "snps": n_snps,
        "threads": n_threads,
        "blocks": blocks,
        "measurements_per_revision_per_stage": 2 * blocks,
        "stages": stage_results,
    }


def _parse_stages(value: str) -> tuple[str, ...]:
    stages = tuple(stage.strip() for stage in value.split(",") if stage.strip())
    invalid = sorted(set(stages) - set(_STAGES))
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown stages: {', '.join(invalid)}")
    if not stages:
        raise argparse.ArgumentTypeError("at least one stage is required")
    if len(set(stages)) != len(stages):
        raise argparse.ArgumentTypeError("stages must not contain duplicates")
    return stages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-root", type=Path)
    parser.add_argument("--b-root", type=Path)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--snps", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--stages", type=_parse_stages, default=_STAGES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.samples < 2:
        raise SystemExit("--samples must be >= 2")
    if args.snps < 1:
        raise SystemExit("--snps must be >= 1")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.worker:
        if args.source_root is None:
            raise SystemExit("--worker requires --source-root")
        _worker(args.source_root.resolve(), args.samples, args.snps, args.threads)
        return

    schedule = _balanced_schedule(args.blocks)
    if args.dry_run:
        for stage in args.stages:
            for index, order in enumerate(schedule, start=1):
                print(f"stage={stage} block={index} order={','.join(order)}")
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
        args.stages,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
