"""Post-install compiler for the _lmm_accel C extension.

Run this after `pip install jamma` to compile the C extension in-place:

    python -m jamma.lmm._compile_accel

Or from a Databricks/Jupyter notebook cell:

    from jamma.lmm._compile_accel import compile_extension
    compile_extension()

Requires: gcc (or cc), Python development headers, numpy >= 2.0.
OpenMP support is optional — falls back to single-threaded if unavailable.

Env vars:
    JAMMA_SENTINEL_UB: when truthy (anything other than "" or "0"),
        injects ``-DJAMMA_SENTINEL_UB`` so the gated ``jamma_sentinel_oob``
        heap-OOB function in ``_lmm_accel.c`` is compiled in. Used exclusively
        by the sanitizer workflow's sentinel-meta-test job to prove ASAN
        actually catches a deliberate bug. The macro is resolved by
        ``resolve_build_spec`` from ``LMM_ACCEL_SPEC.reads_sentinel_env``;
        wheel builds NEVER set it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

# jamma._build_support ships inside the installed package, so the same
# import path works in both modes:
#   1. Dev-mode: ``python -m jamma.lmm._compile_accel`` from a source checkout.
#   2. Wheel install: runtime ABI-mismatch recompile via
#      ``jamma.core.recompile.auto_recompile_c_extension`` calls
#      ``compile_extension()`` from this module.
from jamma._build_support.compile_and_link import LMM_ACCEL_SPEC
from jamma._build_support.compile_and_link import compile_extension as _compile
from jamma._build_support.load_proof import load_proof as _load_proof_for


def compile_extension(
    verbose: bool = False,
    diagnose: bool = False,
    on_retry: Callable[[str], None] | None = None,
) -> bool:
    """Compile the _lmm_accel sources into a shared library in the package.

    Thin shim over ``jamma._build_support.compile_and_link.compile_extension``
    bound to ``LMM_ACCEL_SPEC``. See that function for the build behavior.

    Called by:
      - ``python -m jamma.lmm._compile_accel`` from a source checkout
      - ``jamma.core.recompile.auto_recompile_c_extension`` on ABI mismatch

    Args:
        verbose: Print per-command compile details. When False (default),
            only errors and a one-line summary are printed.
        diagnose: Emit compiler vectorization reports (clang ``-Rpass``,
            gcc ``-fopt-info-vec-all``). Use to verify AVX-512 codegen on
            target hardware.
        on_retry: Optional callback invoked with a single string argument
            when the build retries without OpenMP.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    return _compile(
        LMM_ACCEL_SPEC,
        Path(__file__).parents[1],  # the installed jamma/ package directory
        verbose=verbose,
        diagnose=diagnose,
        on_retry=on_retry,
        out=sys.stdout,
    )


def _load_proof(import_code: str | None = None) -> bool:
    """Prove the freshly compiled ``_lmm_accel`` imports, in a subprocess.

    Bound to ``LMM_ACCEL_SPEC``; see
    ``jamma._build_support.load_proof.load_proof`` for the behavior and why
    the probe runs out of process.
    """
    return _load_proof_for(LMM_ACCEL_SPEC, import_code)


if __name__ == "__main__":
    success = compile_extension(verbose=True) and _load_proof()
    sys.exit(0 if success else 1)
