"""Shared plumbing for the mouse_hs1940 benchmark scripts.

``bench_all_backends.py`` and ``bench_loco.py`` both time JAMMA against a
GEMMA binary on the same fixture, so both had carried their own copy of the
fixture paths, the duration formatter, the ``.fam`` phenotype loader, the
``--gemma-path``/``--runs`` arguments, the GEMMA auto-detection, the
hardware header, and the best-of-N timing loop. The last one had been
inlined six times between them.

The other two benchmark scripts share nothing with these and do not import
this module. ``bench_jlinalg.py`` keeps its own ``_best_time`` because it
warms up before timing, which ``best_of`` deliberately does not.

Imported bare rather than by path: a script invoked as ``python
scripts/bench_x.py`` gets ``scripts/`` as ``sys.path[0]``, which is what
makes ``import _bench_common`` resolve. This is the same arrangement
``_lint_common`` uses for the ``check_*`` lints.
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np

_T = TypeVar("_T")

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


def best_of(fn: Callable[[], _T], runs: int) -> float:
    """Time ``fn`` over ``runs`` repetitions and return the fastest.

    No warmup: the caller decides whether a cold first iteration is part of
    what it wants to measure.

    Args:
        fn: Zero-argument callable to time. Its return value is discarded.
        runs: Number of repetitions.

    Returns:
        Wall-clock seconds for the fastest repetition.
    """
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def load_fam_phenotypes(fam_path: Path) -> np.ndarray:
    """Read phenotypes from column 6 of a PLINK ``.fam`` file.

    GEMMA's missing-phenotype sentinel and the literal ``"NA"`` both become
    NaN.

    Args:
        fam_path: Path to the ``.fam`` file.

    Returns:
        One float64 phenotype per sample, NaN where missing.
    """
    from jamma.core.constants import PHENOTYPE_MISSING

    fam_data = np.loadtxt(fam_path, usecols=5, dtype=str)
    missing = np.isin(fam_data, [str(int(PHENOTYPE_MISSING)), "NA"])
    phenotypes = np.where(missing, "0", fam_data).astype(np.float64)
    phenotypes[missing] = np.nan
    return phenotypes


def load_covariates_4() -> np.ndarray | None:
    """Load the 4-column mouse_hs1940 covariate file if it is present.

    Returns:
        The covariate matrix, or None when the file does not exist.
    """
    if MOUSE_COVAR_4.exists():
        return np.loadtxt(MOUSE_COVAR_4)
    return None


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
        default=1,
        help="Number of runs, report best (default: 1)",
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
    from jamma.core.hardware import get_hardware_context

    ctx = get_hardware_context()
    phys, log = ctx["cpu_count_physical"], ctx["cpu_count_logical"]
    print(f"CPU: {ctx['cpu_model']} ({phys}P/{log}L)")
    print(f"BLAS: {ctx['blas_backend']} ({ctx['blas_threads']} threads)")
    print(f"NumPy: {ctx['numpy_version']}")
    print(f"Platform: {ctx['platform']}")
    print(f"Runs: {runs} (best of)")
    print()
