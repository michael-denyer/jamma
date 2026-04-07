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
from pathlib import Path

from jamma.core.openmp_detect import detect_openmp_flags


def compile_extension(verbose: bool = False, diagnose: bool = False) -> bool:
    """Compile _lmm_accel.c into a shared library in the installed package.

    Args:
        verbose: Print per-command compile details to stdout. When False
            (default), only errors and a one-line summary are printed.
        diagnose: Print GCC vectorization report showing which loops were
            SIMD-vectorized. Use to verify AVX-512 codegen on target hardware.

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
    from jamma.core._compile_utils import find_c_compiler

    result = find_c_compiler()
    if not result:
        _print("ERROR: No C compiler found on PATH (tried cc, clang, gcc)")
        _print("  Install: apt-get install -y gcc  (Linux)")
        _print("  Install: xcode-select --install  (macOS)")
        return False
    cc_cmd, cc_extra = result
    _detail(f"Compiler: {shutil.which(cc_cmd)}")

    # Python headers
    python_inc = sysconfig.get_config_var("INCLUDEPY") or ""
    python_h = Path(python_inc) / "Python.h" if python_inc else None
    if not python_h or not python_h.exists():
        _print(f"ERROR: Python.h not found at {python_inc}")
        _print("  Install: apt-get install -y python3-dev")
        return False
    _detail(f"Python.h: {python_h}")

    # Output path
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = lmm_dir / f"_lmm_accel{ext_suffix}"

    numpy_inc = np.get_include()
    _detail(f"NumPy include: {numpy_inc}")

    # Windows/MSVC is not supported — JAMMA targets Linux/macOS only
    if platform.system() == "Windows":
        _print("ERROR: Windows is not supported for C extension compilation")
        return False

    # Platform flags (GCC/Clang) — split into compile-only and link-only to
    # avoid loading both libgomp (-fopenmp on GCC linker) and libiomp5 (MKL).
    ldflags: list[str] = []
    omp_compile, omp_link, cc_cmd = detect_openmp_flags(
        cc_cmd, platform.system(), _detail, _warn=_print
    )
    if platform.system() == "Darwin":
        ldflags = ["-undefined", "dynamic_lookup"]

    # -march=native is safe here: compiles on the user's own machine.
    # hatch_build.py omits this flag for portable wheel builds.
    diag_flags: list[str] = []
    if diagnose:
        # Detect compiler to use correct vectorization report flags.
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

    # Two-step compile+link to prevent dual OpenMP runtime.
    # GCC's -fopenmp implicitly adds -lgomp at link time. When libiomp5
    # (Intel OpenMP, bundled with MKL numpy) is also linked, both runtimes
    # initialize and abort: "OMP: Error #13: Assertion failure at
    # kmp_runtime.cpp". Splitting into compile (.o) then link (.so) lets
    # us pass -fopenmp only to the compiler and link only libiomp5.
    obj = out.with_suffix(".o")

    compile_cmd = [
        cc_cmd,
        *cc_extra,
        "-O3",
        "-ftree-vectorize",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-march=native",
        "-fPIC",
        "-std=c11",
        f"-I{python_inc}",
        f"-I{numpy_inc}",
        *omp_compile,
        *diag_flags,
        "-c",
        str(src),
        "-o",
        str(obj),
    ]

    _detail(f"Compiling: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if diagnose and (result.stderr or result.stdout):
        _print("\n=== Vectorization Report ===")
        diag_text = result.stderr + result.stdout
        for line in diag_text.splitlines():
            ll = line.lower()
            if any(kw in ll for kw in ("vectoriz", "simd", "remark")):
                _print(f"  {line}")
        _print("=== End Report ===\n")

    compiled_with_omp = bool(omp_compile)
    if result.returncode != 0 and omp_compile:
        _print(f"OpenMP compilation failed: {result.stderr.strip()}")
        _print("Retrying without OpenMP (single-threaded)...")
        compile_cmd_no_omp = [x for x in compile_cmd if x not in omp_compile]
        result = subprocess.run(compile_cmd_no_omp, capture_output=True, text=True)
        compiled_with_omp = False
        omp_link = []  # Don't link OpenMP if compile failed

    if result.returncode != 0:
        _print(f"ERROR: compilation failed:\n{result.stderr}")
        # Clean up .o if it exists
        obj.unlink(missing_ok=True)
        return False

    # Link step: .o -> .so (no -fopenmp here — only libiomp5 link flags)
    link_cmd = [
        cc_cmd,
        "-shared",
        str(obj),
        "-o",
        str(out),
        *omp_link,
        "-lm",
        *ldflags,
    ]

    _detail(f"Linking: {' '.join(link_cmd)}")
    result = subprocess.run(link_cmd, capture_output=True, text=True)
    obj.unlink(missing_ok=True)  # Clean up .o regardless

    if result.returncode != 0:
        _print(f"ERROR: link failed:\n{result.stderr}")
        return False

    _detail(f"Compiled: {out}")

    # Verify import
    try:
        # Force re-import by removing cached failure
        mods_to_remove = [
            k for k in sys.modules if k.startswith("jamma.lmm._lmm_accel")
        ]
        for k in mods_to_remove:
            del sys.modules[k]

        from jamma.lmm._lmm_accel import compute_lmm_batch_c  # noqa: F401

        omp_status = "OpenMP" if compiled_with_omp else "single-threaded"
        _print(f"_lmm_accel compiled OK ({omp_status})")
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


if __name__ == "__main__":
    success = compile_extension(verbose=True)
    sys.exit(0 if success else 1)
