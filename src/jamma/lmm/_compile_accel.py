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


def compile_extension(verbose: bool = True, diagnose: bool = False) -> bool:
    """Compile _lmm_accel.c into a shared library in the installed package.

    Args:
        verbose: Print progress and diagnostics to stdout.
        diagnose: Print GCC vectorization report showing which loops were
            SIMD-vectorized. Use to verify AVX-512 codegen on target hardware.

    Returns:
        True if compilation succeeded, False otherwise.
    """

    def _print(*args: object) -> None:
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

    _print(f"numpy {np.__version__} OK")

    # Compiler
    cc_name = sysconfig.get_config_var("CC") or "cc"
    cc_cmd = cc_name.split()[0]
    cc_extra = cc_name.split()[1:]

    cc_path = shutil.which(cc_cmd)
    if not cc_path:
        _print(f"ERROR: C compiler '{cc_cmd}' not found on PATH")
        _print("  Install: apt-get install -y gcc")
        return False
    _print(f"Compiler: {cc_path}")

    # Python headers
    python_inc = sysconfig.get_config_var("INCLUDEPY") or ""
    python_h = Path(python_inc) / "Python.h" if python_inc else None
    if not python_h or not python_h.exists():
        _print(f"ERROR: Python.h not found at {python_inc}")
        _print("  Install: apt-get install -y python3-dev")
        return False
    _print(f"Python.h: {python_h}")

    # Output path
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = lmm_dir / f"_lmm_accel{ext_suffix}"

    numpy_inc = np.get_include()
    _print(f"NumPy include: {numpy_inc}")

    # Windows/MSVC is not supported — JAMMA targets Linux/macOS only
    if platform.system() == "Windows":
        _print("ERROR: Windows is not supported for C extension compilation")
        return False

    # Platform flags (GCC/Clang)
    omp_flags: list[str] = []
    ldflags: list[str] = []
    if platform.system() == "Darwin":
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            omp_flags = [
                f"-I{prefix}/include",
                f"-L{prefix}/lib",
                "-Xpreprocessor",
                "-fopenmp",
                "-lomp",
            ]
        except (FileNotFoundError, subprocess.CalledProcessError):
            _print(
                "OpenMP not available (libomp not found via Homebrew). "
                "Extension will be single-threaded. "
                "Install for parallelism: brew install libomp"
            )
        ldflags = ["-undefined", "dynamic_lookup"]
    else:
        omp_flags = ["-fopenmp"]

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

    cmd = [
        cc_cmd,
        *cc_extra,
        "-O3",
        "-ftree-vectorize",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-march=native",
        "-fPIC",
        "-shared",
        "-std=c99",
        f"-I{python_inc}",
        f"-I{numpy_inc}",
        *omp_flags,
        *diag_flags,
        str(src),
        "-o",
        str(out),
        "-lm",
        *ldflags,
    ]

    _print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if diagnose and (result.stderr or result.stdout):
        _print("\n=== Vectorization Report ===")
        diag_text = result.stderr + result.stdout
        for line in diag_text.splitlines():
            ll = line.lower()
            if any(kw in ll for kw in ("vectoriz", "simd", "remark")):
                _print(f"  {line}")
        _print("=== End Report ===\n")

    if result.returncode != 0 and omp_flags:
        _print(f"OpenMP compilation failed: {result.stderr.strip()}")
        _print("Retrying without OpenMP (single-threaded)...")
        cmd_no_omp = [x for x in cmd if x not in omp_flags]
        result = subprocess.run(cmd_no_omp, capture_output=True, text=True)

    if result.returncode != 0:
        _print(f"ERROR: compilation failed:\n{result.stderr}")
        return False

    _print(f"Compiled: {out}")

    # Verify import
    try:
        # Force re-import by removing cached failure
        mods_to_remove = [
            k for k in sys.modules if k.startswith("jamma.lmm._lmm_accel")
        ]
        for k in mods_to_remove:
            del sys.modules[k]

        from jamma.lmm._lmm_accel import compute_lmm_batch_c  # noqa: F401

        _print("Import OK — C extension is active")
        return True
    except ImportError as e:
        _print(f"ERROR: compiled but import failed (ImportError): {e}")
        _print("  This usually means ABI mismatch or missing shared libraries.")
        return False
    except OSError as e:
        _print(f"ERROR: compiled but import failed (OSError): {e}")
        _print("  Check that all shared library dependencies are available.")
        return False


if __name__ == "__main__":
    success = compile_extension()
    sys.exit(0 if success else 1)
