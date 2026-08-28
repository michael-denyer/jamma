#!/usr/bin/env python3
"""Check whether compiled C extensions are stale relative to their sources.

JAMMA ships two C extensions built from `src/jamma/**/*.c` and `*.h`:

  - src/jamma/lmm/_lmm_accel.<EXT_SUFFIX>  <- LMM_ACCEL_SOURCES + _lmm_*.h
  - src/jamma/jlinalg/_jlinalg.<EXT_SUFFIX> <- src/jamma/jlinalg/src/*.c + include/*.h

The _lmm_accel source list is read from ``LMM_ACCEL_SOURCES`` in
``src/jamma/_build_support/compile_and_link.py``, the tuple both build entry
points compile, so a kernel file added there is checked here without a
second list. The module is loaded by path rather than imported through
``jamma``: importing the package loads ``_lmm_accel`` and may trigger the
ABI-mismatch auto-rebuild of the very .so this script is about to inspect.
The headers keep a glob because no build constant lists them (the compiler
finds them by include). The jlinalg globs are deliberately wider than any
one constant: its source set is BASELINE_SOURCES minus pymodule.c plus
LAPACK_SOURCES plus the test sources, and every one of them lives under
``src/jamma/jlinalg/src``.

Editable installs (``uv sync``) build these once. Python source edits are
picked up immediately by the editable install, but C source edits are
NOT — they require a manual rebuild:

    uv run python -m jamma.lmm._compile_accel
    uv run python -m jamma.jlinalg._compile_jlinalg

Without a rebuild, ``pytest`` silently tests the stale ``.so``. This script
walks the known C sources, compares their mtimes against the compiled
``.so``, and reports whether any extension is stale.

Exit codes:
  0 — all fresh OR no ``.so`` present (nothing to compare against)
  1 — at least one extension is stale, or could not be checked at all

Use as:
  - conftest.py: `check_all()` returns structured results for a warning
  - pre-push hook: run the script and fail on exit 1

Rationale: a pre-push fail prevents pushing commits whose tests passed
against a stale .so. A warning in conftest alerts developers mid-session
so they can rebuild before noticing weird test output.
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtensionSpec:
    """One C extension, its .so target, and the sources that feed it."""

    label: str
    so_path: Path
    source_globs: tuple[tuple[Path, str], ...]  # (base_dir, glob_pattern)
    rebuild_command: str


def _project_root() -> Path:
    """Locate the project root by walking up from this script."""
    return Path(__file__).resolve().parent.parent


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _lmm_accel_sources(root: Path) -> tuple[str, ...]:
    """Read LMM_ACCEL_SOURCES from compile_and_link.py without importing jamma."""
    path = root / "src/jamma/_build_support/compile_and_link.py"
    spec = importlib.util.spec_from_file_location("_freshness_compile_and_link", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the file defines dataclasses, and the decorator
    # resolves annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return tuple(module.LMM_ACCEL_SOURCES)


def _discover_extensions() -> list[ExtensionSpec]:
    """Enumerate the C extensions JAMMA builds."""
    root = _project_root()
    ext = _ext_suffix()
    lmm_dir = root / "src/jamma/lmm"
    return [
        ExtensionSpec(
            label="_lmm_accel",
            so_path=root / f"src/jamma/lmm/_lmm_accel{ext}",
            source_globs=(
                *((lmm_dir, name) for name in _lmm_accel_sources(root)),
                (lmm_dir, "_lmm_*.h"),
            ),
            rebuild_command="uv run python -m jamma.lmm._compile_accel",
        ),
        ExtensionSpec(
            label="_jlinalg",
            so_path=root / f"src/jamma/jlinalg/_jlinalg{ext}",
            source_globs=(
                (root / "src/jamma/jlinalg/src", "*.c"),
                (root / "src/jamma/jlinalg/include", "*.h"),
            ),
            rebuild_command="uv run python -m jamma.jlinalg._compile_jlinalg",
        ),
    ]


@dataclass(frozen=True)
class FreshnessResult:
    """Outcome of comparing an extension's .so against its sources."""

    spec: ExtensionSpec
    so_exists: bool
    newest_source: Path | None
    newest_source_mtime: float
    so_mtime: float
    is_stale: bool
    error: str | None = None


def _check_extension(spec: ExtensionSpec) -> FreshnessResult:
    """Compare a single extension's .so mtime against its newest source.

    Any ``OSError`` (unreadable ``.so``, missing source raced by glob,
    permission issues on locked-down hosts) lands in ``error`` rather than
    raising, so pytest_configure never aborts on a transient FS failure.
    ``error`` is set and ``is_stale`` stays False, which keeps the conftest
    advisory quiet while letting ``main`` fail the pre-push gate. Reporting
    "up to date" for an extension nobody could look at is the one answer
    this must not give.
    """
    try:
        if not spec.so_path.exists():
            return FreshnessResult(
                spec=spec,
                so_exists=False,
                newest_source=None,
                newest_source_mtime=0.0,
                so_mtime=0.0,
                is_stale=False,
            )

        so_mtime = spec.so_path.stat().st_mtime

        newest_source: Path | None = None
        newest_mtime = 0.0
        for base_dir, pattern in spec.source_globs:
            for src in base_dir.glob(pattern):
                m = src.stat().st_mtime
                if m > newest_mtime:
                    newest_mtime = m
                    newest_source = src
    except OSError as exc:
        return FreshnessResult(
            spec=spec,
            so_exists=False,
            newest_source=None,
            newest_source_mtime=0.0,
            so_mtime=0.0,
            is_stale=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    is_stale = newest_source is not None and newest_mtime > so_mtime
    return FreshnessResult(
        spec=spec,
        so_exists=True,
        newest_source=newest_source,
        newest_source_mtime=newest_mtime,
        so_mtime=so_mtime,
        is_stale=is_stale,
    )


def check_all() -> list[FreshnessResult]:
    """Check every discovered extension. Pure function — no I/O side effects."""
    return [_check_extension(s) for s in _discover_extensions()]


def _format_result(r: FreshnessResult) -> str:
    """Human-readable drift report for one extension."""
    if r.error is not None:
        return f"  {r.spec.label}: could not be checked ({r.error})"
    if not r.so_exists:
        return f"  {r.spec.label}: not built (no .so at {r.spec.so_path}) — skipped"
    if not r.is_stale:
        return f"  {r.spec.label}: up to date"
    assert r.newest_source is not None
    src_rel = r.newest_source.relative_to(_project_root())
    delta_s = r.newest_source_mtime - r.so_mtime
    return (
        f"  {r.spec.label}: STALE — {src_rel} is {delta_s:.0f}s newer than "
        f"the compiled .so. Rebuild with:\n      {r.spec.rebuild_command}"
    )


def main() -> int:
    results = check_all()
    stale = [r for r in results if r.is_stale]
    unchecked = [r for r in results if r.error is not None]
    print("C extension freshness check:")
    for r in results:
        print(_format_result(r))
    if unchecked:
        print(
            f"\n{len(unchecked)} extension(s) could not be checked. A "
            f"freshness gate that cannot read the sources cannot vouch for "
            f"the .so, so this is a failure, not a pass.",
            file=sys.stderr,
        )
    if stale:
        print(
            f"\n{len(stale)} extension(s) stale. Tests may silently run against "
            f"old compiled code. Rebuild before continuing.",
            file=sys.stderr,
        )
        return 1
    return 1 if unchecked else 0


if __name__ == "__main__":
    sys.exit(main())
