"""Post-install compiler for the _jlinalg C extension.

Run this after ``pip install jamma`` to compile the jlinalg C extension in-place:

    python -m jamma.jlinalg._compile_jlinalg

Or from a Databricks/Jupyter notebook cell:

    from jamma.jlinalg._compile_jlinalg import compile_extension
    compile_extension()

Requires: gcc (or cc), Python development headers, numpy >= 2.0.
OpenMP support is optional — falls back to single-threaded if unavailable.

The jlinalg extension compiles per-file to enable per-source-group compiler flags
(e.g. strict IEEE 754 for LAPACK sources vs standard optimization for baseline).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# jamma._build_support ships inside the installed package, so the same
# import path works in both modes:
#   1. Dev-mode: ``python -m jamma.jlinalg._compile_jlinalg`` from a source
#      checkout.
#   2. Wheel install: runtime ABI-mismatch recompile via
#      ``jamma.core.recompile.auto_recompile_c_extension`` calls
#      ``compile_extension()`` from this module.
from jamma._build_support.compile_and_link import JLINALG_SPEC
from jamma._build_support.compile_and_link import compile_extension as _compile


def compile_extension(
    verbose: bool = False,
    on_retry: Callable[[str], None] | None = None,
) -> bool:
    """Compile jlinalg C sources into a shared library in the installed package.

    Thin shim over ``jamma._build_support.compile_and_link.compile_extension``
    bound to ``JLINALG_SPEC``. See that function for the build behavior.

    Dev-mode entry point. Called by:
      - ``python -m jamma.jlinalg._compile_jlinalg`` from a source checkout
      - ``jamma.core.recompile.auto_recompile_c_extension`` on ABI mismatch

    Args:
        verbose: Print per-command compile details. When False (default),
            only errors and a one-line summary are printed.
        on_retry: Optional callback invoked with a single string argument
            when the build retries without OpenMP.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    return _compile(
        JLINALG_SPEC,
        Path(__file__).parents[1],  # the installed jamma/ package directory
        verbose=verbose,
        on_retry=on_retry,
        out=sys.stderr,
    )


def _load_proof(
    import_code: str = (
        "from jamma.jlinalg._jlinalg import HAS_OPENMP, jlinalg_isa; "
        "print(f'jlinalg compiled OK (ISA={jlinalg_isa}, "
        "OpenMP={HAS_OPENMP})')"
    ),
) -> bool:
    """Prove the freshly compiled extension imports, in a fresh subprocess.

    A successful compile+link does not guarantee a usable module — bad
    RPATH, missing runtime library, ABI mismatch with the host numpy, or a
    missing C symbol can let the link pass but the import fail. Runs in a
    subprocess rather than this interpreter so the proof never re-executes
    ``jamma.jlinalg``'s own import machinery (the #181 self-deadlock cause).

    Skipped when JAMMA_SANITIZE is set: importing an ASan-instrumented .so
    requires LD_PRELOAD=libasan.so, which the sanitizer workflow only exports
    for the pytest step, not this compile step — the subprocess would abort
    with "ASan runtime does not come first in initial library list" (exit
    134). The pytest step exercises the .so under the correct LD_PRELOAD.

    Args:
        import_code: The statement run in the subprocess. Overridable so a
            test can point the proof at a module that cannot import, without
            touching a real .so.
    """
    if os.environ.get("JAMMA_SANITIZE", "").strip() not in ("", "0"):
        print(
            "jlinalg compiled (skipping post-link import proof — JAMMA_SANITIZE set)",
            file=sys.stderr,
        )
        return True

    proc = subprocess.run(
        [sys.executable, "-c", import_code],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(
            "ERROR: jlinalg compiled but failed to import in a fresh "
            f"interpreter (exit {proc.returncode}):",
            file=sys.stderr,
        )
        print(proc.stderr, file=sys.stderr)
        return False
    print(proc.stdout.strip(), file=sys.stderr)
    return True


if __name__ == "__main__":
    success = compile_extension(verbose=True) and _load_proof()
    sys.exit(0 if success else 1)
