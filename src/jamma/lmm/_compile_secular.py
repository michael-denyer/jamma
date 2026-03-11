"""Post-install compiler for the _secular_accel C extension.

Run this after `pip install jamma` to compile the C extension in-place:

    python -m jamma.lmm._compile_secular

Or from a Databricks/Jupyter notebook cell:

    from jamma.lmm._compile_secular import compile_extension
    compile_extension()

Requires: gcc (or cc), Python development headers, numpy >= 2.0.
No OpenMP needed — DLAED4 calls are sequential; each solves a single eigenvalue.

No LAPACK link flags needed — the extension discovers LAPACK at runtime via
dlopen (resolves dlaed4_64_ or dlaed4_ from numpy's bundled BLAS/LAPACK).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def compile_extension(verbose: bool = True) -> bool:
    """Compile _secular_accel.c into a shared library in the installed package.

    Args:
        verbose: Print progress and diagnostics to stdout.

    Returns:
        True if compilation succeeded, False otherwise.
    """

    def _print(*args: object) -> None:
        if verbose:
            print(*args, flush=True)

    # Locate source
    lmm_dir = Path(__file__).parent
    src = lmm_dir / "_secular_accel.c"
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
    out = lmm_dir / f"_secular_accel{ext_suffix}"

    numpy_inc = np.get_include()
    _print(f"NumPy include: {numpy_inc}")

    # Windows/MSVC is not supported — JAMMA targets Linux/macOS only
    if platform.system() == "Windows":
        _print("ERROR: Windows is not supported for C extension compilation")
        return False

    # Platform flags
    ldflags: list[str] = []
    if platform.system() == "Darwin":
        ldflags = ["-undefined", "dynamic_lookup"]

    # dlopen needs -ldl on Linux (macOS has dlopen in libSystem)
    dl_flags = ["-ldl"] if platform.system() == "Linux" else []

    # -march=native is safe here: compiles on the user's own machine.
    # Not used in wheel builds (compiled post-install, not by hatch_build.py).
    # No LAPACK link flags — DLAED4 is resolved at runtime via dlopen.
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
        str(src),
        "-o",
        str(out),
        "-lm",
        *dl_flags,
        *ldflags,
    ]

    _print(f"Compiling: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except (FileNotFoundError, PermissionError) as e:
        _print(f"ERROR: failed to execute compiler: {e}")
        return False

    if result.returncode != 0:
        _print(f"ERROR: compilation failed:\n{result.stderr}")
        return False

    _print(f"Compiled: {out}")

    # Verify import
    try:
        # Force re-import by removing cached failure
        mods_to_remove = [
            k for k in sys.modules if k.startswith("jamma.lmm._secular_accel")
        ]
        for k in mods_to_remove:
            del sys.modules[k]

        from jamma.lmm._secular_accel import (  # noqa: F401
            rank1_eigenvalue_update,
            rank1_eigenvalues_and_norms,
        )

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
