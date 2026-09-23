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

import sys
from pathlib import Path

from jamma._build_support.compile_and_link import JLINALG_SPEC, BuildReport
from jamma._build_support.compile_and_link import compile_extension as _compile
from jamma._build_support.load_proof import load_proof as _load_proof_for


def compile_extension(verbose: bool = False) -> bool:
    """Compile jlinalg C sources into a shared library in the installed package.

    Thin shim over ``jamma._build_support.compile_and_link.compile_extension``
    bound to ``JLINALG_SPEC``. See that function for the build behavior.

    Dev-mode entry point for ``python -m jamma.jlinalg._compile_jlinalg``.

    Args:
        verbose: Print per-command compile details and the success summary.
            When False (default), only errors and retry notices are printed.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    return _compile(
        JLINALG_SPEC,
        Path(__file__).parents[1],  # the installed jamma/ package directory
        BuildReport.to_stream(sys.stderr, verbose=verbose),
    )


def _load_proof(import_code: str | None = None) -> bool:
    """Prove the freshly compiled ``_jlinalg`` imports, in a subprocess.

    Bound to ``JLINALG_SPEC``; see
    ``jamma._build_support.load_proof.load_proof`` for the behavior and why
    the probe runs out of process.
    """
    return _load_proof_for(JLINALG_SPEC, import_code)


if __name__ == "__main__":
    success = compile_extension(verbose=True) and _load_proof()
    sys.exit(0 if success else 1)
