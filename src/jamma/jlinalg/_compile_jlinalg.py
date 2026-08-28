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
from jamma._build_support.compile_and_link import JLINALG_SPEC, run_build
from jamma._build_support.find_compiler import find_c_compiler
from jamma._build_support.openmp_detect import detect_openmp_flags


def compile_extension(
    verbose: bool = False,
    on_retry: Callable[[str], None] | None = None,
) -> bool:
    """Compile jlinalg C sources into a shared library in the installed package.

    Performs per-file compile-then-link via
    ``jamma._build_support.compile_and_link`` to enable different compiler
    flags per source group (strict IEEE 754 for LAPACK, standard
    optimization for baseline sources).

    Dev-mode entry point. Called by:
      - ``python -m jamma.jlinalg._compile_jlinalg`` from a source checkout
      - ``jamma.core.recompile.auto_recompile_c_extension`` on ABI mismatch

    Args:
        verbose: Print per-command compile details to stderr. When False
            (default), only errors and a one-line summary are printed.
        on_retry: Optional callback invoked with a single string argument
            when the build retries without OpenMP. When None, retry notices
            are routed to ``_print`` and surface on stderr. The runtime
            recompile shim uses this hook to forward downgrade notices to
            loguru so users whose ABI-mismatch recompile silently loses
            parallelism still see a warning.

    Returns:
        True if compilation succeeded, False otherwise.
    """

    def _print(*args: object) -> None:
        """Always print (errors, results)."""
        print(*args, file=sys.stderr, flush=True)

    def _detail(*args: object) -> None:
        """Print only when verbose."""
        if verbose:
            print(*args, file=sys.stderr, flush=True)

    def _retry(msg: str) -> None:
        if on_retry is not None:
            on_retry(msg)
        else:
            _print(msg)

    outcome = run_build(
        JLINALG_SPEC,
        Path(__file__).parents[1],  # the installed jamma/ package directory
        dev_mode=True,
        find_c_compiler=find_c_compiler,
        detect_openmp_flags=detect_openmp_flags,
        on_retry=_retry,
        verbose_print=_detail,
        error_print=_print,
    )
    if outcome.skipped:
        return False
    if not outcome.ok:
        error = outcome.result.error if outcome.result else "unknown"
        _print(f"ERROR: jlinalg compilation failed: {error}")
        return False

    _detail(f"Compiled: {outcome.output_path}")

    # Skip the post-link import probe when JAMMA_SANITIZE is set.
    # Importing an ASan-instrumented .so requires LD_PRELOAD=libasan.so; the
    # sanitizer workflow only exports LD_PRELOAD for the pytest step, not the
    # compile step, so the probe would abort with
    # "ASan runtime does not come first in initial library list" (exit 134).
    # The pytest step exercises the .so under the correct LD_PRELOAD anyway.
    if os.environ.get("JAMMA_SANITIZE", "").strip():
        _print("jlinalg compiled (skipping import probe — JAMMA_SANITIZE set)")
        return True

    # Verify import — evict both _jlinalg and the parent jamma.jlinalg package
    # so the freshly compiled extension is loaded instead of the cached fallback.
    try:
        mods_to_remove = [k for k in sys.modules if k.startswith("jamma.jlinalg")]
        for k in mods_to_remove:
            del sys.modules[k]

        from jamma.jlinalg._jlinalg import HAS_OPENMP, jlinalg_isa

        omp_status = "OpenMP" if HAS_OPENMP else "single-threaded"
        _print(f"jlinalg compiled OK (ISA={jlinalg_isa}, {omp_status})")
        return True
    except ImportError as e:
        _print(f"ERROR: compiled but import failed (ImportError): {e}")
        _print("  This usually means ABI mismatch or missing shared libraries.")
        return False
    except OSError as e:
        _print(f"ERROR: compiled but import failed (OSError): {e}")
        _print("  Check that all shared library dependencies are available.")
        return False
    except Exception as e:  # noqa: BLE001 — last-resort diagnostic: ImportError and OSError are handled above; anything else must still be surfaced with a traceback rather than propagated out of a compile helper
        import traceback

        _print(f"ERROR: compiled but import failed ({type(e).__name__}): {e}")
        _print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = compile_extension(verbose=True)
    sys.exit(0 if success else 1)
