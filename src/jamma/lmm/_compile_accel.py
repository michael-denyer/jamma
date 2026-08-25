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

import os
import sys
from collections.abc import Callable
from pathlib import Path

# jamma._build_support ships inside the installed package, so the same
# import path works in both modes:
#   1. Dev-mode: ``python -m jamma.lmm._compile_accel`` from a source checkout.
#   2. Wheel install: runtime ABI-mismatch recompile via
#      ``jamma.core.recompile.auto_recompile_c_extension`` calls
#      ``compile_extension()`` from this module.
from jamma._build_support.compile_and_link import LMM_ACCEL_SPEC, run_build
from jamma._build_support.find_compiler import find_c_compiler
from jamma._build_support.openmp_detect import detect_openmp_flags


def compile_extension(
    verbose: bool = False,
    diagnose: bool = False,
    on_retry: Callable[[str], None] | None = None,
) -> bool:
    """Compile _lmm_accel.c into a shared library in the installed package.

    Drives ``run_build`` with ``LMM_ACCEL_SPEC`` (dev mode), then verifies the
    freshly built extension actually imports. The dev-only ``-march=native``
    flag and the ``JAMMA_SENTINEL_UB`` macro are supplied by
    ``resolve_build_spec`` from the spec, so they never reach the portable
    wheel-build path in ``hatch_build.py``.

    Called by:
      - ``python -m jamma.lmm._compile_accel`` from a source checkout
      - ``jamma.core.recompile.auto_recompile_c_extension`` on ABI mismatch

    Args:
        verbose: Print per-command compile details to stdout. When False
            (default), only errors and a one-line summary are printed.
        diagnose: Emit compiler vectorization reports (clang ``-Rpass``,
            gcc ``-fopt-info-vec-all``). Use to verify AVX-512 codegen on
            target hardware.
        on_retry: Optional callback invoked with a single string argument
            when the build retries without OpenMP. When None, retry notices
            are routed to ``_print`` and surface on stdout. The runtime
            recompile shim uses this hook to forward downgrade notices to
            loguru so users whose ABI-mismatch recompile silently loses
            parallelism still see a warning.

    Returns:
        True if compilation succeeded, False otherwise.
    """

    def _print(*args: object) -> None:
        """Always print (errors, results)."""
        print(*args, flush=True)

    def _detail(*args: object) -> None:
        """Print only when verbose."""
        if verbose:
            print(*args, flush=True)

    def _retry(msg: str) -> None:
        if on_retry is not None:
            on_retry(msg)
        else:
            _print(msg)

    outcome = run_build(
        LMM_ACCEL_SPEC,
        Path(__file__).parents[1],  # the installed jamma/ package directory
        dev_mode=True,
        find_c_compiler=find_c_compiler,
        detect_openmp_flags=detect_openmp_flags,
        diagnose=diagnose,
        on_retry=_retry,
        verbose_print=_detail,
        error_print=_print,
    )
    if outcome.skipped:
        return False
    if not outcome.ok:
        error = outcome.result.error if outcome.result else "unknown"
        _print(f"ERROR: _lmm_accel compilation failed: {error}")
        return False

    # Evict cached import so a subsequent `from jamma.lmm._lmm_accel import ...`
    # picks up the freshly compiled extension instead of the stale module.
    for k in [k for k in sys.modules if k.startswith("jamma.lmm._lmm_accel")]:
        del sys.modules[k]

    # Skip the post-link import probe when JAMMA_SANITIZE is set.
    # Importing an ASan-instrumented .so requires LD_PRELOAD=libasan.so; the
    # sanitizer workflow only exports LD_PRELOAD for the pytest step, not the
    # compile step, so the probe would abort with
    # "ASan runtime does not come first in initial library list" (exit 134).
    # The pytest step exercises the .so under the correct LD_PRELOAD anyway.
    if os.environ.get("JAMMA_SANITIZE", "").strip():
        _detail("skipping post-link import probe — JAMMA_SANITIZE is set")
        return True

    # Verify the compiled extension actually imports. A successful compile+link
    # does not guarantee a usable module — bad RPATH, missing runtime library,
    # ABI mismatch with the host numpy, or a missing C symbol can let the link
    # pass but the `import` fail. Mirrors the check in `_compile_jlinalg`.
    try:
        from jamma.lmm._lmm_accel import (
            compute_lmm_chunk_fused_c as _probe,  # noqa: F401
        )
    except ImportError as e:
        _print(f"ERROR: compiled but import failed (ImportError): {e}")
        _print("  This usually means ABI mismatch or missing shared libraries.")
        return False
    except OSError as e:
        _print(f"ERROR: compiled but import failed (OSError): {e}")
        _print("  Check that all shared library dependencies are available.")
        return False
    except Exception as e:  # noqa: BLE001 — last-resort diagnostic: ImportError and OSError handled above; anything else must surface with a traceback rather than propagate out of a compile helper
        import traceback

        _print(f"ERROR: compiled but import failed ({type(e).__name__}): {e}")
        _print(traceback.format_exc())
        return False

    omp_status = (
        "OpenMP"
        if outcome.result and outcome.result.used_openmp
        else ("single-threaded")
    )
    _print(f"_lmm_accel extension compiled: {outcome.output_path} ({omp_status})")
    return True


if __name__ == "__main__":
    success = compile_extension(verbose=True)
    sys.exit(0 if success else 1)
