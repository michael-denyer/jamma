"""Shared fixtures, process timing and output verification for GWAS benchmarks.

Imported directly by scripts launched with ``python scripts/bench_*.py``.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MOUSE_DIR = REPO_ROOT / "tests" / "fixtures" / "mouse_hs1940"
MOUSE_PREFIX = MOUSE_DIR / "mouse_hs1940"
MOUSE_KINSHIP = MOUSE_DIR / "mouse_hs1940_kinship.cXX.txt"
MOUSE_COVAR_4 = MOUSE_DIR / "covariates_4.txt"
DEFAULT_GEMMA = Path.home() / ".local" / "bin" / "gemma"
DEFAULT_GEMMA_ACCELERATE = Path.home() / ".local" / "bin" / "gemma-accelerate"


def fmt_seconds(seconds: float) -> str:
    """Format a duration as milliseconds, seconds, or minutes and seconds.

    Args:
        seconds: Duration to format.

    Returns:
        Human-readable duration, for example ``"430ms"``, ``"7.1s"``, or
        ``"2m14s"``.
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def speedup(ref: float | None, fast: float) -> str:
    """Format a speedup ratio against a reference time.

    Args:
        ref: Reference duration in seconds, or None when unavailable.
        fast: Duration to compare against the reference.

    Returns:
        The ratio as ``"23.6x"``, or an em dash when ``ref`` is None.
    """
    if ref is None:
        return "—"
    return f"{ref / fast:.1f}x"


def add_gemma_args(parser: argparse.ArgumentParser) -> None:
    """Add the GEMMA path and repetition arguments to ``parser``.

    Args:
        parser: Parser to extend with ``--gemma-path``,
            ``--gemma-accelerate-path``, and ``--runs``.
    """
    parser.add_argument(
        "--gemma-path",
        type=Path,
        default=None,
        help=f"Path to GEMMA binary (default: auto-detect at {DEFAULT_GEMMA})",
    )
    parser.add_argument(
        "--gemma-accelerate-path",
        type=Path,
        default=None,
        help=(
            "Path to GEMMA+Accelerate binary"
            f" (default: auto-detect at {DEFAULT_GEMMA_ACCELERATE})"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs, report best (default: 3)",
    )


def find_gemma(cli_arg: Path | None, name: str) -> Path | None:
    """Resolve a GEMMA binary from the CLI argument, home, or PATH.

    Args:
        cli_arg: Explicit ``--gemma-path`` style value, returned as given.
        name: Binary name, for example ``"gemma"`` or ``"gemma-accelerate"``.

    Returns:
        The resolved path, or None when no binary was found.
    """
    if cli_arg is not None:
        return cli_arg
    default = Path.home() / ".local" / "bin" / name
    if default.exists():
        return default
    found = shutil.which(name)
    return Path(found) if found else None


def print_hardware_header(runs: int) -> None:
    """Print the CPU, BLAS, NumPy, platform, and repetition-count header.

    Args:
        runs: Repetition count to report.
    """
    from jamma import jlinalg
    from jamma.core.hardware import get_hardware_context
    from jamma.lmm import accel

    if accel._accel is None:
        raise RuntimeError("Build the JAMMA C extension before benchmarking NumPy+C")
    ctx = get_hardware_context()
    phys, log = ctx["cpu_count_physical"], ctx["cpu_count_logical"]
    print(f"CPU: {ctx['cpu_model']} ({phys}P/{log}L)")
    print(f"BLAS: {jlinalg.blas_backend} ({ctx['blas_threads']} threads)")
    print(f"NumPy: {ctx['numpy_version']}")
    print(f"Platform: {ctx['platform']}")
    print(f"Runs: {runs} (best of)")
    print()


def run_commands(commands: list[list[str]], env: dict[str, str]) -> float:
    """Time complete child processes, failing rather than reporting partial work."""
    import subprocess

    started = time.perf_counter()
    for command in commands:
        proc = subprocess.run(command, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Benchmark command failed ({proc.returncode}): {command!r}\n"
                f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
            )
    return time.perf_counter() - started


def read_associations(path: Path) -> dict[str, dict[str, str]]:
    """Read actual output rows; missing, empty or duplicate results fail the run."""
    import csv

    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    results = {row["rs"]: row for row in rows}
    if not results or len(results) != len(rows):
        raise ValueError(f"Empty or duplicate association results: {path}")
    return results


def verify_associations(
    reference: dict[str, dict[str, str]], actual: dict[str, dict[str, str]]
) -> None:
    """Require the same tested SNPs, alleles and numerically agreeing results."""
    if reference.keys() != actual.keys():
        raise ValueError(
            f"Tested SNP sets differ: {len(reference)} vs {len(actual)}; "
            f"missing={len(reference.keys() - actual.keys())}, "
            f"extra={len(actual.keys() - reference.keys())}"
        )
    for column in ("allele1", "allele0"):
        if any(reference[snp][column] != actual[snp][column] for snp in reference):
            raise ValueError(f"Association {column} differs")
    from jamma.validation.tolerances import ToleranceConfig

    tolerances = ToleranceConfig()
    snps = list(reference)
    first = reference[snps[0]]
    for column, rtol in (
        ("beta", tolerances.beta_rtol),
        ("se", tolerances.se_rtol),
        ("p_wald", tolerances.pvalue_rtol),
        ("p_lrt", tolerances.p_lrt_rtol),
        ("p_score", tolerances.pvalue_rtol),
    ):
        if column not in first:
            continue
        expected = np.array([float(reference[snp][column]) for snp in snps])
        observed = np.array([float(actual[snp][column]) for snp in snps])
        atol: float | np.ndarray = 1e-14
        if column == "beta":
            # A null effect is round-off on a value far below its standard
            # error, so beta carries an absolute floor scaled by that error.
            se = np.array([float(reference[snp]["se"]) for snp in snps])
            atol = atol + tolerances.beta_se_floor * np.abs(se)
        close = np.isclose(observed, expected, rtol=rtol, atol=atol, equal_nan=True)
        if not close.all():
            worst = int(np.argmax(np.where(close, 0.0, np.abs(observed - expected))))
            raise AssertionError(
                f"{column} differs at {int((~close).sum())} SNPs; worst {snps[worst]}: "
                f"actual={observed[worst]:.6e} reference={expected[worst]:.6e}"
            )
