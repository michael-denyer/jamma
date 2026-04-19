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
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

# _compile_jlinalg.py runs in two modes:
#   1. Dev-mode: `python -m jamma.jlinalg._compile_jlinalg` from a source
#      checkout. build_support/ sits at repo root (sibling to src/).
#   2. Wheel install: jamma is installed but build_support/ is NOT shipped
#      with the wheel. On ABI-mismatch recompile at import time,
#      jamma.core.recompile.auto_recompile_c_extension calls this module's
#      compile_extension(). That code path should emit a clear error, NOT
#      a vague ImportError from the top-of-module `from build_support...`.
#
# Bootstrap: try the source-checkout path; if build_support/ is absent
# (wheel install), leave the helper refs as None and guard the public
# entry points at call time with a RuntimeError that explains the
# situation. This preserves the existing ABI-mismatch graceful fallback
# behavior (see 123-PATTERNS.md lines 320-334).
_repo_root = Path(__file__).resolve().parents[3]
_build_support_available = (_repo_root / "build_support").is_dir()

if _build_support_available:
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from build_support.compile_and_link import (  # noqa: E402
        BASELINE_SOURCES,
        LAPACK_SOURCES,
        LINK_FLAGS_BY_PLATFORM,
        compile_jlinalg,
    )
    from build_support.find_compiler import find_c_compiler  # noqa: E402
    from build_support.openmp_detect import detect_openmp_flags  # noqa: E402
else:
    # Wheel-install path: build_support/ not present. Do NOT raise at
    # import time — jamma.jlinalg.__init__ imports from this module to
    # access other helpers and must succeed. Instead, leave the helper
    # refs as None and guard the public functions with a clear
    # RuntimeError.
    BASELINE_SOURCES = None  # type: ignore[assignment]
    LAPACK_SOURCES = None  # type: ignore[assignment]
    LINK_FLAGS_BY_PLATFORM = None  # type: ignore[assignment]
    compile_jlinalg = None  # type: ignore[assignment]
    find_c_compiler = None  # type: ignore[assignment]
    detect_openmp_flags = None  # type: ignore[assignment]


def compile_extension(verbose: bool = False) -> bool:
    """Compile jlinalg C sources into a shared library in the installed package.

    Performs per-file compile-then-link via ``build_support.compile_and_link``
    to enable different compiler flags per source group (strict IEEE 754 for
    LAPACK, standard optimization for baseline sources).

    Dev-mode entry point. Called by:
      - ``python -m jamma.jlinalg._compile_jlinalg`` from a source checkout
      - ``jamma.core.recompile.auto_recompile_c_extension`` on ABI mismatch

    Args:
        verbose: Print per-command compile details to stderr. When False
            (default), only errors and a one-line summary are printed.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    if compile_jlinalg is None:
        raise RuntimeError(
            "compile_extension() called from a wheel-install environment "
            "(build_support/ not found at repo root). This function is only "
            "available in dev-mode or sdist installs. For wheel installs, "
            "ABI mismatches trigger jamma.core.recompile.auto_recompile_c_extension() "
            "instead — it uses its own minimal compiler discovery and does not "
            "require build_support/."
        )

    def _print(*args: object) -> None:
        """Always print (errors, results)."""
        print(*args, file=sys.stderr, flush=True)

    def _detail(*args: object) -> None:
        """Print only when verbose."""
        if verbose:
            print(*args, file=sys.stderr, flush=True)

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
            on_retry=lambda msg: _print(msg),
            verbose_print=_detail,
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
    """Compile the Unity C test harness into a standalone binary.

    Compiles test_boundaries.c + unity.c + all jlinalg .c source files into
    a single test executable (not a shared library).  Uses the same compiler
    flags and per-file SIMD/LAPACK flag split as the main extension.

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
    # plus test_boundaries.c and unity.c
    baseline_sources = [
        jlinalg_src_dir / "platform.c",
        jlinalg_src_dir / "blas_dispatch.c",
        jlinalg_src_dir / "snp_stats.c",
    ]
    lapack_sources = [
        jlinalg_src_dir / "eigh.c",
    ]

    # Test-specific sources
    test_sources = [
        tests_dir / "test_boundaries.c",
        unity_dir / "unity.c",
    ]

    all_sources = baseline_sources + lapack_sources + test_sources

    # NumPy headers (for npy_intp)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy not installed -- required for test harness") from exc

    numpy_inc = np.get_include()
    python_inc = sysconfig.get_config_var("INCLUDEPY") or ""

    # Compiler
    from jamma.core._compile_utils import find_c_compiler

    result = find_c_compiler()
    if not result:
        raise RuntimeError(
            "No C compiler found on PATH (tried $CC, sysconfig, cc, clang, gcc). "
            "Install: apt-get install -y gcc (Linux) or xcode-select --install (macOS)"
        )
    cc_cmd, cc_extra = result

    # Link against libpython (blas_dispatch.c uses Python C API for numpy discovery)
    python_libdir = sysconfig.get_config_var("LIBDIR") or ""
    python_version = sysconfig.get_config_var("VERSION") or ""

    # OpenMP detection (same as compile_extension — split compile/link)
    ldflags: list[str] = ["-lm"]
    if python_libdir:
        ldflags.extend([f"-L{python_libdir}", f"-lpython{python_version}"])
        ldflags.append(f"-Wl,-rpath,{python_libdir}")
    if platform.system() == "Linux":
        ldflags.append("-ldl")
        ldflags.append("-lpthread")  # snp_stats.c uses pthreads directly
    omp_compile, omp_link, cc_cmd = detect_openmp_flags(
        cc_cmd, platform.system(), _print
    )

    # Base compile flags
    base_cflags = [
        "-O3",
        "-ftree-vectorize",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-funroll-loops",
        "-fno-finite-math-only",
        "-std=c11",
        f"-I{python_inc}",
        f"-I{numpy_inc}",
        f"-I{jlinalg_inc_dir}",
    ]

    lapack_cflags = [
        "-O2",
        "-fno-fast-math",
        "-fno-finite-math-only",
        "-std=c11",
        f"-I{python_inc}",
        f"-I{numpy_inc}",
        f"-I{jlinalg_inc_dir}",
    ]

    lapack_source_set = {str(s) for s in lapack_sources}

    # Compile each source to .o.  Mirrors compile_extension's OpenMP retry:
    # first attempt with OpenMP, retry single-threaded if compilation fails.
    def _compile_all(
        current_omp_compile: list[str],
    ) -> list[Path] | None:
        """Compile all sources, return obj file list or None on failure."""
        td = Path(tempfile.mkdtemp(prefix="jlinalg_test_"))
        objs: list[Path] = []
        try:
            for src in all_sources:
                if not src.exists():
                    raise RuntimeError(f"Required source file not found: {src}")

                obj_file = td / f"{src.stem}.o"
                src_str = str(src)

                cflags = lapack_cflags if src_str in lapack_source_set else base_cflags

                # Test sources and Unity need double precision + include path
                extra_inc: list[str] = []
                if "test_boundaries.c" in src.name:
                    extra_inc = [f"-I{tests_dir}", "-DUNITY_INCLUDE_DOUBLE"]
                elif "unity.c" in src.name:
                    extra_inc = ["-DUNITY_INCLUDE_DOUBLE"]

                cmd = [
                    cc_cmd,
                    *cc_extra,
                    *cflags,
                    *current_omp_compile,
                    *extra_inc,
                    "-c",
                    str(src),
                    "-o",
                    str(obj_file),
                ]
                _print(f"compile: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    _print(f"Compile failed for {src.name}:")
                    _print(result.stderr)
                    shutil.rmtree(td, ignore_errors=True)
                    return None
                objs.append(obj_file)
        except Exception:
            shutil.rmtree(td, ignore_errors=True)
            raise
        return objs

    # First attempt: with OpenMP
    use_omp = bool(omp_compile)
    current_omp_link = omp_link
    obj_files = _compile_all(omp_compile)

    if obj_files is None and use_omp:
        _print(
            "OpenMP compilation failed, retrying without OpenMP (single-threaded)..."
        )
        obj_files = _compile_all([])
        current_omp_link = []  # no OMP runtime to link

    if obj_files is None:
        raise RuntimeError("jlinalg test harness compilation failed")

    try:
        # Link into executable
        out = tests_dir / "test_boundaries"
        if platform.system() == "Windows":
            out = out.with_suffix(".exe")

        cmd_link = [
            cc_cmd,
            *cc_extra,
            *[str(o) for o in obj_files],
            "-o",
            str(out),
            *current_omp_link,
            *ldflags,
        ]
        _print(f"link: {' '.join(cmd_link)}")
        result_link = subprocess.run(cmd_link, capture_output=True, text=True)
        if result_link.returncode != 0:
            raise RuntimeError(f"Link failed:\n{result_link.stderr}")

        _print(f"Test harness compiled: {out}")
        return out

    finally:
        # Clean up obj files (they live in a temp dir from the last _compile_all call)
        if obj_files:
            td = obj_files[0].parent
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    success = compile_extension(verbose=True)
    sys.exit(0 if success else 1)
