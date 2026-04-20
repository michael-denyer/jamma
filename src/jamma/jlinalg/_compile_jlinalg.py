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

import platform
import shutil
import sys
import sysconfig
import tempfile
from collections.abc import Callable
from pathlib import Path

# jamma._build_support ships inside the installed package, so the same
# import path works in both modes:
#   1. Dev-mode: ``python -m jamma.jlinalg._compile_jlinalg`` from a source
#      checkout.
#   2. Wheel install: runtime ABI-mismatch recompile via
#      ``jamma.core.recompile.auto_recompile_c_extension`` calls
#      ``compile_extension()`` from this module.
from jamma._build_support.compile_and_link import (
    BASELINE_SOURCES,
    LAPACK_SOURCES,
    LINK_FLAGS_BY_PLATFORM,
    compile_jlinalg,
)
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

    # Locate jlinalg source directory relative to this file
    jlinalg_dir = Path(__file__).parent
    jlinalg_src_dir = jlinalg_dir / "src"
    jlinalg_inc_dir = jlinalg_dir / "include"

    if not jlinalg_src_dir.is_dir():
        _print(f"ERROR: jlinalg source directory not found: {jlinalg_src_dir}")
        _print("  Package may be incomplete — reinstall from source.")
        return False

    # Source files split into two groups:
    # - baseline: portable C compiled with standard flags
    # - lapack: eigh.c compiled with strict IEEE 754 flags (-O2 -fno-fast-math)
    baseline = [jlinalg_src_dir / name for name in BASELINE_SOURCES]
    lapack = [jlinalg_src_dir / name for name in LAPACK_SOURCES]
    source_files = baseline + lapack

    missing = [str(s) for s in source_files if not s.exists()]
    if missing:
        _print(f"ERROR: jlinalg source files missing: {missing}")
        return False

    # NumPy
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
        _print("  Install: apt-get install -y python3-dev  (Linux)")
        return False
    _detail(f"Python.h: {python_h}")

    # Windows is not supported
    if platform.system() == "Windows":
        _print("ERROR: Windows is not supported for C extension compilation")
        return False

    # Output path (next to __init__.py in the installed package)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = jlinalg_dir / f"_jlinalg{ext_suffix}"
    numpy_inc = np.get_include()
    _detail(f"NumPy include: {numpy_inc}")
    _detail(f"Output: {out}")

    # OpenMP detection — may override cc_cmd to use clang when libiomp5 is
    # found, since GCC's GOMP compatibility shim has assertion failures after
    # MKL LAPACK operations.
    omp_compile, omp_link, cc_cmd = detect_openmp_flags(
        cc_cmd, platform.system(), _detail, _warn=_print
    )

    # Platform-specific link flags. Helper appends omp_link + extra_link_flags.
    ldflags = list(LINK_FLAGS_BY_PLATFORM.get(platform.system(), ()))

    # Compile + link via the shared helper. ``-lm`` goes through
    # extra_link_flags because it's universal but not encoded in the
    # per-platform LINK_FLAGS_BY_PLATFORM table.
    tmp_dir = Path(tempfile.mkdtemp(prefix="jlinalg_compile_"))
    try:
        result = compile_jlinalg(
            sources=source_files,
            lapack_sources=lapack,
            include_dirs=[python_inc, numpy_inc, str(jlinalg_inc_dir)],
            cc_cmd=cc_cmd,
            cc_extra=cc_extra,
            omp_compile=omp_compile,
            omp_link=omp_link,
            ldflags=ldflags,
            output=out,
            tmp_dir=tmp_dir,
            extra_link_flags=["-lm"],
            on_retry=_retry,
            verbose_print=_detail,
            error_print=_print,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not result.success:
        _print(f"ERROR: jlinalg compilation failed: {result.error}")
        return False

    _detail(f"Compiled: {out}")

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
    except Exception as e:
        import traceback

        _print(f"ERROR: compiled but import failed ({type(e).__name__}): {e}")
        _print(traceback.format_exc())
        return False


def compile_test_harness(verbose: bool = True) -> Path:
    """Compile the Unity C test harness into a standalone executable.

    Routes through ``jamma._build_support.compile_and_link.compile_jlinalg``
    with ``link_shared=False`` to produce a native executable (not a .so).
    Compiles test_boundaries.c + unity.c + all jlinalg .c source files
    (minus pymodule.c, which is Python-module glue and not needed for the
    standalone binary). Per-source flag split (LAPACK vs baseline) is
    identical to ``compile_extension``.

    Args:
        verbose: Print progress and diagnostics to stderr.

    Returns:
        Path to the compiled test binary.

    Raises:
        RuntimeError: If compilation fails.
    """

    def _print(*args: object) -> None:
        if verbose:
            print(*args, file=sys.stderr, flush=True)

    def _warn(*args: object) -> None:
        """Always visible — used for OMP downgrade + similar warnings."""
        print(*args, file=sys.stderr, flush=True)

    jlinalg_dir = Path(__file__).parent
    jlinalg_src_dir = jlinalg_dir / "src"
    jlinalg_inc_dir = jlinalg_dir / "include"
    tests_dir = jlinalg_dir / "tests"
    unity_dir = tests_dir / "unity"

    # Verify Unity framework is vendored
    for f in ("unity.h", "unity.c", "unity_internals.h"):
        if not (unity_dir / f).exists():
            raise RuntimeError(f"Unity framework file missing: {unity_dir / f}")

    # Source files: same as main extension minus pymodule.c (no Python API)
    # plus test_boundaries.c and unity.c. BASELINE_SOURCES includes
    # pymodule.c, so filter it out here; the test harness links as a
    # standalone executable and pymodule.c only exports PyInit symbols.
    baseline_names = tuple(n for n in BASELINE_SOURCES if n != "pymodule.c")
    baseline_prod_sources = [jlinalg_src_dir / name for name in baseline_names]
    lapack_sources = [jlinalg_src_dir / name for name in LAPACK_SOURCES]

    # Test-specific sources (excluded from BASELINE_SOURCES — they live
    # in jlinalg/tests/, not in jlinalg/src/).
    test_sources = [
        tests_dir / "test_boundaries.c",
        unity_dir / "unity.c",
    ]

    all_sources = baseline_prod_sources + lapack_sources + test_sources

    # NumPy headers (for npy_intp)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy not installed -- required for test harness") from exc

    numpy_inc = np.get_include()
    python_inc = sysconfig.get_config_var("INCLUDEPY") or ""

    # Compiler
    cc_result = find_c_compiler()
    if not cc_result:
        raise RuntimeError(
            "No C compiler found on PATH (tried $CC, sysconfig, cc, clang, gcc). "
            "Install: apt-get install -y gcc (Linux) or xcode-select --install (macOS)"
        )
    cc_cmd, cc_extra = cc_result

    # Link against libpython (blas_dispatch.c uses Python C API for numpy discovery).
    python_libdir = sysconfig.get_config_var("LIBDIR") or ""
    python_version = sysconfig.get_config_var("VERSION") or ""

    ldflags: list[str] = ["-lm"]
    if python_libdir:
        ldflags.extend([f"-L{python_libdir}", f"-lpython{python_version}"])
        ldflags.append(f"-Wl,-rpath,{python_libdir}")
    if platform.system() == "Linux":
        ldflags.append("-ldl")
        ldflags.append("-lpthread")  # snp_stats.c uses pthreads directly

    # OpenMP detection (same as compile_extension — split compile/link).
    # _warn is always-visible so the GCC-libiomp5 downgrade and GNU-libgomp
    # fallback warnings surface even when verbose=False.
    omp_compile, omp_link, cc_cmd = detect_openmp_flags(
        cc_cmd, platform.system(), _print, _warn=_warn
    )

    # Per-source extra includes: test_boundaries.c needs -I<tests_dir> to
    # reach unity.h and other test headers. unity.c lives under unity_dir
    # and already finds its own headers relative to the source path.
    extra_source_includes: dict[str, list[str]] = {
        "test_boundaries.c": [str(tests_dir)],
    }

    # -DUNITY_INCLUDE_DOUBLE tells Unity to enable double-precision
    # assertions. It's a no-op on non-Unity sources (platform.c, etc.),
    # so passing it globally via extra_cflags is safe. (extra_cflags is
    # not applied to LAPACK sources by resolve_cflags_for, which is fine
    # — eigh.c doesn't use Unity.)
    extra_cflags = ["-DUNITY_INCLUDE_DOUBLE"]

    # Output path — executable under tests_dir.
    out = tests_dir / "test_boundaries"
    if platform.system() == "Windows":
        out = out.with_suffix(".exe")

    tmp_dir = Path(tempfile.mkdtemp(prefix="jlinalg_test_"))
    try:
        result = compile_jlinalg(
            sources=all_sources,
            lapack_sources=lapack_sources,
            include_dirs=[python_inc, numpy_inc, str(jlinalg_inc_dir)],
            cc_cmd=cc_cmd,
            cc_extra=cc_extra,
            omp_compile=omp_compile,
            omp_link=omp_link,
            ldflags=ldflags,
            output=out,
            tmp_dir=tmp_dir,
            extra_cflags=extra_cflags,
            extra_source_includes=extra_source_includes,
            link_shared=False,
            on_retry=lambda msg: _warn(msg),
            verbose_print=_print,
            error_print=_warn,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not result.success:
        raise RuntimeError(f"jlinalg test harness compilation failed: {result.error}")

    _print(f"Test harness compiled: {out}")
    return out


if __name__ == "__main__":
    success = compile_extension(verbose=True)
    sys.exit(0 if success else 1)
