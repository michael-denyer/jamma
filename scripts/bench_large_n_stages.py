#!/usr/bin/env python3
"""Interleaved A/B timing for the large-N numerical stages.

Each revision runs in a separate worker. For every stage, each ABBA/BAAB block
gets a fresh pair of workers and reverses their creation order. This controls
both time-on-machine drift and allocation or process-layout bias. Each stage
owns representative prebuilt inputs, so unrelated setup work never
contaminates its timing. The workers hash numerical results and stop when the
revisions disagree.

The kinship, eigen, and rotation stages time one jlinalg call each. The
association stage times ``run_lmm_association_numpy`` end to end on a
prebuilt eigenbasis, so it covers the chunk plan, the rotation GEMM, the C
kernel, and whatever overlap the pipeline achieves between them. It uses
JAMMA's own thread policy; ``--threads`` does not reach it.

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
from collections.abc import Callable
from pathlib import Path

import numpy as np

_STAGES = ("kinship", "eigen", "rotation", "association")


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


def _build_inputs(n_samples: int, n_snps: int) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic {0, 1, 2} dosages and phenotypes.

    Dosages rather than Gaussian values, because the association stage runs
    the real runner, whose MAF and missingness filters would drop every
    Gaussian column and time an empty run.
    """
    rng = np.random.default_rng(20260801)
    maf = rng.uniform(0.1, 0.5, n_snps)
    genotypes = rng.binomial(2, maf, size=(n_samples, n_snps)).astype(np.float64)
    genotypes = np.ascontiguousarray(genotypes)
    phenotypes = rng.standard_normal(n_samples)
    return genotypes, phenotypes


def _householder_basis(n_samples: int) -> np.ndarray:
    """A dense orthogonal matrix built in O(n^2), standing in for eigenvectors.

    The association stage needs an orthonormal basis so the rotation GEMM and
    the kernel see realistic dense inputs, not a timed eigendecomposition. A
    Householder reflector I - 2 v v^T is exactly orthogonal and costs one
    outer product to build, where a real eigendecomposition would dominate
    worker start-up at large N.
    """
    v = np.random.default_rng(20260804).standard_normal(n_samples)
    v /= np.linalg.norm(v)
    basis = np.eye(n_samples)
    # Row blocks keep the outer-product temporary at block x n rather than
    # n x n, so building the basis never doubles the worker's footprint.
    block = 1024
    for start in range(0, n_samples, block):
        rows = v[start : start + block]
        basis[start : start + block] -= 2.0 * np.outer(rows, v)
    return basis


class _StageRunner:
    """Own stage inputs and reusable output buffers inside one worker."""

    def __init__(
        self,
        n_samples: int,
        n_snps: int,
        n_threads: int,
        stages: tuple[str, ...],
    ) -> None:
        from jamma import jlinalg

        self._jlinalg = jlinalg
        self._n_threads = n_threads
        self._genotypes, self._phenotypes = _build_inputs(n_samples, n_snps)

        self._kinship_out: np.ndarray | None = None
        self._kinship_seed: np.ndarray | None = None
        self._eigen_work: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None
        self._rotation_out: np.ndarray | None = None
        self._association: Callable[[], dict[str, np.ndarray]] | None = None

        if "kinship" in stages or "eigen" in stages:
            self._kinship_out = np.empty((n_samples, n_samples), dtype=np.float64)
            self._jlinalg.dsyrk(self._genotypes, out=self._kinship_out, beta=0.0)

        if "eigen" in stages:
            assert self._kinship_out is not None
            self._kinship_seed = self._kinship_out.copy()
            self._kinship_seed /= n_snps
            self._kinship_seed.flat[:: n_samples + 1] += 1e-6
            self._eigen_work = np.empty_like(self._kinship_seed)

        if "rotation" in stages:
            # Rotation needs a dense eigenvector-like matrix, not a timed
            # eigendecomposition. Generating it directly isolates DGEMM.
            rotation_rng = np.random.default_rng(20260802)
            self._eigenvectors = np.ascontiguousarray(
                rotation_rng.standard_normal((n_samples, n_samples))
            )
            self._rotation_out = np.empty((n_snps, n_samples), dtype=np.float64)

        if "association" in stages:
            self._association = self._bind_association(n_samples, n_snps)

    def run(self, stage: str) -> object:
        if stage == "kinship":
            return self._run_kinship()
        if stage == "eigen":
            return self._run_eigen()
        if stage == "rotation":
            return self._run_rotation()
        if stage == "association":
            assert self._association is not None
            return self._association()
        raise ValueError(f"unknown benchmark stage: {stage}")

    def _bind_association(
        self, n_samples: int, n_snps: int
    ) -> Callable[[], dict[str, np.ndarray]]:
        from jamma.lmm.runner_numpy import run_lmm_association_numpy
        from jamma.lmm.schema import LmmConfig

        eigenvalues = np.sort(
            np.random.default_rng(20260803).uniform(0.1, 2.0, n_samples)
        )
        eigenvectors = _householder_basis(n_samples)
        snp_info = [
            {"chr": "1", "rs": f"rs{i}", "pos": i + 1, "a1": "A", "a0": "T"}
            for i in range(n_snps)
        ]
        config = LmmConfig(lmm_mode=1, show_progress=False, check_memory=False)
        fields = ("beta", "se", "p_wald", "logl_H1", "l_remle")

        def run() -> dict[str, np.ndarray]:
            result = run_lmm_association_numpy(
                genotypes=self._genotypes,
                phenotypes=self._phenotypes,
                kinship=None,
                snp_info=snp_info,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                config=config,
            )
            rows = result.associations
            if len(rows) != n_snps:
                raise RuntimeError(
                    f"association stage tested {len(rows)} of {n_snps} SNPs; "
                    "the inputs must pass every runner filter for the timing "
                    "to mean anything"
                )
            return {
                field: np.array([getattr(row, field) for row in rows], dtype=np.float64)
                for field in fields
            }

        return run

    def _run_kinship(self) -> np.ndarray:
        assert self._kinship_out is not None
        self._jlinalg.dsyrk(self._genotypes, out=self._kinship_out, beta=0.0)
        return self._kinship_out

    def _run_eigen(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._kinship_seed is not None
        assert self._eigen_work is not None
        np.copyto(self._eigen_work, self._kinship_seed)
        eigenvalues, eigenvectors, _status = self._jlinalg.eigh(
            self._eigen_work, inplace=True
        )
        return eigenvalues, eigenvectors

    def _run_rotation(self) -> np.ndarray:
        assert self._eigenvectors is not None
        assert self._rotation_out is not None
        return self._jlinalg.dgemm(
            self._genotypes,
            self._eigenvectors,
            transa="T",
            out=self._rotation_out,
        )


def _worker(
    source_root: Path,
    n_samples: int,
    n_snps: int,
    n_threads: int,
    stages: tuple[str, ...],
) -> None:
    """Serve warmed stage timings over a line-oriented protocol."""
    sys.path.insert(0, str(source_root / "src"))
    runner = _StageRunner(n_samples, n_snps, n_threads, stages)

    for stage in stages:
        runner.run(stage)
    print("ready", flush=True)

    for command in sys.stdin:
        stage = command.strip()
        if stage == "stop":
            return
        if stage not in stages:
            raise ValueError(f"unknown worker command: {stage}")
        started = time.perf_counter()
        result = runner.run(stage)
        elapsed = time.perf_counter() - started
        print(json.dumps({"seconds": elapsed, "sha256": _digest(result)}), flush=True)


class _WorkerProcess:
    """One persistent worker, isolated to one source tree."""

    def __init__(
        self,
        source_root: Path,
        n_samples: int,
        n_snps: int,
        n_threads: int,
        stages: tuple[str, ...],
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
            "--stages",
            ",".join(stages),
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


def _worker_start_order(block_index: int) -> tuple[str, str]:
    """Reverse worker creation order with each timing block."""
    return ("A", "B") if block_index % 2 == 0 else ("B", "A")


def _percent_change(after: float, before: float) -> float:
    return 100.0 * (after / before - 1.0)


def _revision(source_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _summarize_stage(
    stage: str,
    a_root: Path,
    b_root: Path,
    n_samples: int,
    n_snps: int,
    n_threads: int,
    schedule: list[list[str]],
    tolerate_output_mismatch: bool = False,
) -> dict[str, object]:
    timings: dict[str, list[float]] = {"A": [], "B": []}
    digests: dict[str, set[str]] = {"A": set(), "B": set()}
    deltas: list[float] = []
    block_medians: list[float] = []

    roots = {"A": a_root, "B": b_root}
    for block_index, order in enumerate(schedule):
        worker_start_order = _worker_start_order(block_index)
        workers: dict[str, _WorkerProcess] = {}
        by_revision: dict[str, list[float]] = {"A": [], "B": []}
        chronological: list[float] = []
        try:
            for revision in worker_start_order:
                workers[revision] = _WorkerProcess(
                    roots[revision],
                    n_samples,
                    n_snps,
                    n_threads,
                    (stage,),
                )
            for revision in order:
                seconds, digest = workers[revision].measure(stage)
                timings[revision].append(seconds)
                by_revision[revision].append(seconds)
                chronological.append(seconds)
                digests[revision].add(digest)
        finally:
            for worker in workers.values():
                worker.close()
        deltas.append(
            _percent_change(
                statistics.median(by_revision["B"]),
                statistics.median(by_revision["A"]),
            )
        )
        block_medians.append(statistics.median(chronological))

    if len(digests["A"]) != 1 or len(digests["B"]) != 1:
        raise RuntimeError(
            f"{stage} output is not deterministic within a revision: "
            f"A={digests['A']} B={digests['B']}"
        )
    outputs_match = digests["A"] == digests["B"]
    if not outputs_match and not tolerate_output_mismatch:
        raise RuntimeError(
            f"{stage} output mismatch: A={digests['A']} B={digests['B']}. "
            "Pass --tolerate-output-mismatch to time revisions that differ "
            "in the last bits on purpose."
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
        "outputs_match": outputs_match,
        "a_output_sha256": next(iter(digests["A"])),
        "b_output_sha256": next(iter(digests["B"])),
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
    tolerate_output_mismatch: bool = False,
) -> dict[str, object]:
    """Compare every requested stage with balanced, correctness-checked A/B runs."""
    schedule = _balanced_schedule(blocks)
    stage_results = {
        stage: _summarize_stage(
            stage,
            a_root,
            b_root,
            n_samples,
            n_snps,
            n_threads,
            schedule,
            tolerate_output_mismatch=tolerate_output_mismatch,
        )
        for stage in stages
    }

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
    parser.add_argument(
        "--tolerate-output-mismatch",
        action="store_true",
        help="Report rather than abort when A and B outputs differ, for "
        "revisions that change the last bits on purpose. Each revision must "
        "still be deterministic with itself.",
    )
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
        _worker(
            args.source_root.resolve(),
            args.samples,
            args.snps,
            args.threads,
            args.stages,
        )
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
        tolerate_output_mismatch=args.tolerate_output_mismatch,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
