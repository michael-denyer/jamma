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
        injects ``-DJAMMA_SENTINEL_UB`` into ``extra_cflags`` so the
        gated ``jamma_sentinel_oob`` heap-OOB function in
        ``_lmm_accel.c`` is compiled in. Used exclusively by the
        sanitizer workflow's sentinel-meta-test job to prove ASAN
        actually catches a deliberate bug. Orthogonal to
        ``JAMMA_SANITIZE`` — either, both, or neither may be set.
        Wheel builds NEVER set it (hatch_build.py is a separate path
        that does not read this env var).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable
from pathlib import Path

# jamma._build_support ships inside the installed package, so the same
# import path works in both modes:
#   1. Dev-mode: ``python -m jamma.lmm._compile_accel`` from a source
#      checkout.
#   2. Wheel install: runtime ABI-mismatch recompile via
#      ``jamma.core.recompile.auto_recompile_c_extension`` calls
#      ``compile_extension()`` from this module.
from jamma._build_support.compile_and_link import (
    LINK_FLAGS_BY_PLATFORM,
    LMM_ACCEL_SOURCES,
    apply_sanitizer_overrides,
    compile_jlinalg,
)
from jamma._build_support.find_compiler import find_c_compiler
from jamma._build_support.openmp_detect import detect_openmp_flags
from jamma.core.constants import env_flag

# Dev-mode + sanitizer-workflow sentinel macro. Toggled by the
# JAMMA_SENTINEL_UB env var; when set, _lmm_accel.c's gated heap-OOB function
# `jamma_sentinel_oob` is exposed. The sanitizer workflow's
# asan-sentinel-meta-test job sets this to verify that ASAN actually catches
# a deliberate bug. Wheel builds NEVER set it (the macro never reaches
# hatch_build.py — that path has its own extra_cflags assembly).
#
# The literal lives here (not in jamma._build_support.compile_and_link)
# because it's a -D preprocessor macro, not a compile-optimization flag.
# The compile-flag-literal lint regex covers -O/-f/-W/-m/-s/-p prefixes,
# not -D, so this assignment is naturally lint-clean. The named constant
# makes the literal greppable.
_SENTINEL_UB_DEFINE = "-DJAMMA_SENTINEL_UB"


def compile_extension(
    verbose: bool = False,
    diagnose: bool = False,
    on_retry: Callable[[str], None] | None = None,
) -> bool:
    """Compile _lmm_accel.c into a shared library in the installed package.

    Routes through ``jamma._build_support.compile_and_link.compile_jlinalg``
    with ``LMM_ACCEL_SOURCES`` and no LAPACK sources.
    Dev-mode-only flags (``-march=native``, vectorization-report flags) are
    supplied via ``extra_cflags`` so they do not leak into the portable
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
            when the build retries without OpenMP (compile- or link-phase
            downgrade). When None, retry notices are routed to ``_print``
            and surface on stdout. The runtime recompile shim uses this
            hook to forward downgrade notices to loguru so users whose
            ABI-mismatch recompile silently loses parallelism still see
            a warning.

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

    # Locate sources
    lmm_dir = Path(__file__).parent
    sources = [lmm_dir / name for name in LMM_ACCEL_SOURCES]
    missing = [s for s in sources if not s.exists()]
    if missing:
        names = ", ".join(str(s) for s in missing)
        _print(f"ERROR: {names} not found — package may be incomplete")
        return False

    # Numpy
    try:
        import numpy as np
    except ImportError:
        _print("ERROR: numpy not installed")
        return False

    np_major = int(np.__version__.split(".")[0])
    if np_major < 2:
        _print(
            f"ERROR: numpy {np.__version__} is 1.x — "
            "C extension requires numpy >= 2.0 headers"
        )
        return False

    _detail(f"numpy {np.__version__} OK")

    # Compiler
    cc_result = find_c_compiler()
    if not cc_result:
        _print(
            "ERROR: No C compiler found on PATH (tried $CC, sysconfig, cc, clang, gcc)"
        )
        _print("  Install: apt-get install -y gcc  (Linux)")
        _print("  Install: xcode-select --install  (macOS)")
        return False
    cc_cmd, cc_extra = cc_result
    _detail(f"Compiler: {shutil.which(cc_cmd)}")

    # Python headers
    python_inc = sysconfig.get_config_var("INCLUDEPY") or ""
    python_h = Path(python_inc) / "Python.h" if python_inc else None
    if not python_h or not python_h.exists():
        _print(f"ERROR: Python.h not found at {python_inc}")
        _print("  Install: apt-get install -y python3-dev")
        return False
    _detail(f"Python.h: {python_h}")

    # Windows/MSVC is not supported — JAMMA targets Linux/macOS only
    if platform.system() == "Windows":
        _print("ERROR: Windows is not supported for C extension compilation")
        return False

    # Output path
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = lmm_dir / f"_lmm_accel{ext_suffix}"

    numpy_inc = np.get_include()
    _detail(f"NumPy include: {numpy_inc}")

    # OpenMP detection — may override cc_cmd to use clang when libiomp5 is
    # found, since GCC's GOMP compatibility shim has assertion failures after
    # MKL LAPACK operations.
    omp_compile, omp_link, cc_cmd = detect_openmp_flags(
        cc_cmd, platform.system(), _detail, _warn=_print
    )

    # Dev-mode extras: -march=native unconditionally, diagnose flags optional.
    # These are LOCAL to _compile_accel.py and MUST NOT be moved to
    # BASE_CFLAGS in jamma._build_support — hatch_build.py (portable wheel
    # path) must not bake -march=native into the wheel. Dev builds target
    # the local CPU; wheels target the lowest common denominator.
    # allow-compile-flag-literal: dev-only, see rationale above
    extra_cflags: list[str] = ["-march=native"]

    diag_flags: list[str] = []
    if diagnose:
        # Detect compiler to pick the right vectorization-report flag.
        # GCC uses -fopt-info-vec-all; Clang uses -Rpass flags.
        try:
            probe = subprocess.run(
                [cc_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            compiler_id = probe.stdout.lower() if probe.returncode == 0 else ""
        except (subprocess.TimeoutExpired, OSError):
            # Wedged compiler or broken symlink; fall back to GCC-style flags.
            compiler_id = ""
        if "clang" in compiler_id:
            diag_flags = [
                "-Rpass=loop-vectorize",
                "-Rpass-missed=loop-vectorize",
                "-Rpass-analysis=loop-vectorize",
            ]
        else:
            diag_flags = ["-fopt-info-vec-all"]
    extra_cflags.extend(diag_flags)

    # Opt-in sentinel macro for the sanitizer-workflow self-test.
    # See _SENTINEL_UB_DEFINE comment at module top. Truthy convention mirrors
    # JAMMA_FORCE_NUMPY_FALLBACK: "" and "0" are off, anything else
    # is on. Orthogonal to JAMMA_SANITIZE — either, both, or neither can
    # be set.
    if env_flag("JAMMA_SENTINEL_UB"):
        extra_cflags.append(_SENTINEL_UB_DEFINE)
        _detail(f"sentinel: appended {_SENTINEL_UB_DEFINE} (JAMMA_SENTINEL_UB env set)")

    # Platform-specific link flags. Helper appends omp_link + extra_link_flags.
    ldflags = list(LINK_FLAGS_BY_PLATFORM.get(platform.system(), ()))

    # Route through apply_sanitizer_overrides so JAMMA_SANITIZE
    # env var (set by .github/workflows/sanitizers.yml) injects sanitizer
    # flags into BOTH extra_cflags AND extra_lapack_cflags. The helper is a
    # no-op when JAMMA_SANITIZE is unset, so this call is safe in every
    # invocation. macOS note: -fsanitize=address requires Xcode clang +
    # findable libasan; the workflow targets ubuntu-latest, so the macOS
    # dev rebuild may fail under JAMMA_SANITIZE — that's expected.
    extra_cflags, extra_link_flags_for_call, extra_lapack_cflags = (
        apply_sanitizer_overrides(extra_cflags, ["-lm"])
    )

    # Compile + link via the shared helper. ``_lmm_accel.c`` is a baseline
    # source (NOT LAPACK) — pass empty lapack_sources=[]. ``-lm`` was
    # already routed through apply_sanitizer_overrides above so the
    # sanitizer link flags can land alongside it.
    tmp_dir = Path(tempfile.mkdtemp(prefix="lmm_accel_compile_"))
    try:
        result = compile_jlinalg(
            sources=sources,
            lapack_sources=[],
            include_dirs=[python_inc, numpy_inc],
            cc_cmd=cc_cmd,
            cc_extra=cc_extra,
            omp_compile=omp_compile,
            omp_link=omp_link,
            ldflags=ldflags,
            output=out,
            tmp_dir=tmp_dir,
            extra_cflags=extra_cflags,
            extra_link_flags=extra_link_flags_for_call,
            extra_lapack_cflags=extra_lapack_cflags,
            on_retry=_retry,
            verbose_print=_detail,
            error_print=_print,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not result.success:
        _print(f"ERROR: _lmm_accel compilation failed: {result.error}")
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
    # The pytest step exercises the .so under the correct LD_PRELOAD anyway,
    # so the probe adds no coverage on a sanitizer build.
    if os.environ.get("JAMMA_SANITIZE", "").strip():
        _detail("skipping post-link import probe — JAMMA_SANITIZE is set")
        return True

    # Verify the compiled extension actually imports. A successful
    # compile+link does not guarantee a usable module — bad RPATH,
    # missing runtime library, ABI mismatch with the host numpy, or a
    # missing C symbol can let the link pass but the `import` fail.
    # Without this check, `python -m jamma.lmm._compile_accel` exits 0
    # and `auto_recompile_c_extension` reports success even though the
    # subsequent `from jamma.lmm._lmm_accel import ...` still raises.
    # Mirrors the verification in `jamma.jlinalg._compile_jlinalg`.
    try:
        from jamma.lmm._lmm_accel import compute_lmm_batch_c as _probe  # noqa: F401
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

    omp_status = "OpenMP" if result.used_openmp else "single-threaded"
    _print(f"_lmm_accel extension compiled: {out} ({omp_status})")
    return True


if __name__ == "__main__":
    success = compile_extension(verbose=True)
    sys.exit(0 if success else 1)
