#!/usr/bin/env python3
"""Check whether compiled C extensions are stale relative to their sources.

JAMMA ships two C extensions built from `src/jamma/**/*.c` and `*.h`:

  - src/jamma/lmm/_lmm_accel.<EXT_SUFFIX>  <- src/jamma/lmm/_lmm_accel.c
  - src/jamma/jlinalg/_jlinalg.<EXT_SUFFIX> <- src/jamma/jlinalg/src/*.c + include/*.h

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
  1 — at least one extension is stale

Use as:
  - conftest.py: `check_all()` returns structured results for a warning
  - pre-push hook: run the script and fail on exit 1

Rationale: a pre-push fail prevents pushing commits whose tests passed
against a stale .so. A warning in conftest alerts developers mid-session
so they can rebuild before noticing weird test output.
"""

from __future__ import annotations

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


def _discover_extensions() -> list[ExtensionSpec]:
    """Enumerate the C extensions JAMMA builds."""
    root = _project_root()
    ext = _ext_suffix()
    return [
        ExtensionSpec(
            label="_lmm_accel",
            so_path=root / f"src/jamma/lmm/_lmm_accel{ext}",
            source_globs=((root / "src/jamma/lmm", "_lmm_accel.c"),),
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


def _check_extension(spec: ExtensionSpec) -> FreshnessResult:
    """Compare a single extension's .so mtime against its newest source.

    Any ``OSError`` (unreadable ``.so``, missing source raced by glob,
    permission issues on locked-down hosts) degrades to ``is_stale=False``
    so callers — including pytest_configure — never abort on transient FS
    failures. The pre-push hook re-runs in a clean environment.
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
    except OSError:
        return FreshnessResult(
            spec=spec,
            so_exists=False,
            newest_source=None,
            newest_source_mtime=0.0,
            so_mtime=0.0,
            is_stale=False,
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
    print("C extension freshness check:")
    for r in results:
        print(_format_result(r))
    if stale:
        print(
            f"\n{len(stale)} extension(s) stale. Tests may silently run against "
            f"old compiled code. Rebuild before continuing.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
