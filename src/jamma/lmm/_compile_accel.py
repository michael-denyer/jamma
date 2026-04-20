"""Post-install compiler for the _lmm_accel C extension.

Run this after `pip install jamma` to compile the C extension in-place:

    python -m jamma.lmm._compile_accel

Or from a Databricks/Jupyter notebook cell:

    from jamma.lmm._compile_accel import compile_extension
    compile_extension()

Requires: gcc (or cc), Python development headers, numpy >= 2.0.
OpenMP support is optional — falls back to single-threaded if unavailable.
"""

from __future__ import annotations

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
    compile_jlinalg,
)
from jamma._build_support.find_compiler import find_c_compiler
from jamma._build_support.openmp_detect import detect_openmp_flags


def compile_extension(
    verbose: bool = False,
    diagnose: bool = False,
    on_retry: Callable[[str], None] | None = None,
) -> bool:
    """Compile _lmm_accel.c into a shared library in the installed package.

    Routes through ``jamma._build_support.compile_and_link.compile_jlinalg``
    with a single source (``_lmm_accel.c``) and no LAPACK sources.
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

    # Locate source
    lmm_dir = Path(__file__).parent
    src = lmm_dir / "_lmm_accel.c"
    if not src.exists():
        _print(f"ERROR: {src} not found — package may be incomplete")
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
    # Deliberate divergence, not duplication.
    extra_cflags: list[str] = ["-march=native"]

    diag_flags: list[str] = []
    if diagnose:
        # Detect compiler to pick the right vectorization-report flag.
        # GCC uses -fopt-info-vec-all; Clang uses -Rpass flags.
        probe = subprocess.run([cc_cmd, "--version"], capture_output=True, text=True)
        compiler_id = probe.stdout.lower() if probe.returncode == 0 else ""
        if "clang" in compiler_id:
            diag_flags = [
                "-Rpass=loop-vectorize",
                "-Rpass-missed=loop-vectorize",
                "-Rpass-analysis=loop-vectorize",
            ]
        else:
            diag_flags = ["-fopt-info-vec-all"]
    extra_cflags.extend(diag_flags)

    # Platform-specific link flags. Helper appends omp_link + extra_link_flags.
    ldflags = list(LINK_FLAGS_BY_PLATFORM.get(platform.system(), ()))

    # Compile + link via the shared helper. ``_lmm_accel.c`` is a baseline
    # source (NOT LAPACK) — pass empty lapack_sources=[]. ``-lm`` goes
    # through extra_link_flags because it's universal but not encoded in
    # the per-platform LINK_FLAGS_BY_PLATFORM table.
    tmp_dir = Path(tempfile.mkdtemp(prefix="lmm_accel_compile_"))
    try:
        result = compile_jlinalg(
            sources=[src],
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
            extra_link_flags=["-lm"],
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

    omp_status = "OpenMP" if result.used_openmp else "single-threaded"
    _print(f"_lmm_accel extension compiled: {out} ({omp_status})")
    return True


if __name__ == "__main__":
    success = compile_extension(verbose=True)
    sys.exit(0 if success else 1)
