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
from jamma._build_support.load_proof import load_proof as _load_proof_for


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
