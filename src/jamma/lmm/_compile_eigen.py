"""Post-install compiler for the _eigen_accel C extension.

Run this after `pip install jamma` to compile the C extension in-place:

    python -m jamma.lmm._compile_eigen

Or from a Databricks/Jupyter notebook cell:

    from jamma.lmm._compile_eigen import compile_extension
    compile_extension()

Requires: gcc (or cc), Python development headers, numpy >= 2.0.
No OpenMP needed — DSYEVR is a single LAPACK call; MKL/OpenBLAS handles
threading internally.

ILP64 detection: if numpy is built with ILP64 MKL, the extension is compiled
with -DJAMMA_ILP64 to use the dsyevr_64_ symbol instead of dsyevr_.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def _find_numpy_lapack_flags() -> list[str]:
    """Return compiler flags to link against numpy's bundled LAPACK (Linux).

    On Linux, pip-installed numpy bundles OpenBLAS in numpy.libs/ or
    numpy/_core/.libs/. Extensions calling LAPACK symbols need to link
    against this library or the symbols won't resolve at import time.

    Note: Keep in sync with hatch_build.py:_find_numpy_lapack_flags().

    Returns:
        List of compiler flags, empty if not needed (macOS/Windows) or
        if numpy's LAPACK library cannot be found.
    """
    if platform.system() != "Linux":
        return []

    import numpy as np

    np_dir = Path(np.__file__).parent
    candidates = [
        np_dir / ".libs",
        np_dir.parent / "numpy.libs",
    ]
    for d in candidates:
        if not d.is_dir():
            continue
        for pat in ["libscipy_openblas*.so", "libopenblas*.so"]:
            matches = sorted(d.glob(pat))
            if matches:
                lib_file = matches[0].name
                lib_name = lib_file.split(".so")[0]
                if lib_name.startswith("lib"):
                    lib_name = lib_name[3:]
                return [
                    f"-L{d}",
                    f"-l{lib_name}",
                    f"-Wl,-rpath,{d}",
                ]
    return []


def _detect_ilp64() -> bool:
    """Check if numpy is built with ILP64 BLAS (64-bit integers).

    Detects ILP64 from two sources:
    1. np.show_config() BLAS name containing "ilp64" (custom MKL builds)
    2. Bundled library name containing "openblas64" (PyPI numpy ships
       libscipy_openblas64_ which uses 64-bit integers with _64_ suffixed symbols)

    Note: Keep in sync with hatch_build.py:CustomBuildHook._detect_ilp64().

    Returns:
        True if numpy uses ILP64 BLAS, False otherwise.
    """
    try:
        import numpy as np

        config = np.show_config(mode="dicts")
        blas_info = config.get("Build Dependencies", {}).get("blas", {})
        name = blas_info.get("name", "")
        if "ilp64" in name.lower():
            return True
    except (TypeError, AttributeError) as e:
        print(
            f"WARNING: ILP64 detection failed ({type(e).__name__}: {e}). "
            "Defaulting to LP64 (dsyevr_). If numpy is ILP64, "
            "pass -DJAMMA_ILP64 manually.",
            flush=True,
        )

    # Check bundled library name: PyPI numpy uses libscipy_openblas64_ which
    # has ILP64 interface (all LAPACK symbols suffixed with _64_).
    if _bundled_lapack_is_ilp64():
        return True

    return False


def _bundled_lapack_is_ilp64() -> bool:
    """Check if numpy's bundled LAPACK library uses ILP64 (64-bit integer) symbols.

    PyPI numpy bundles libscipy_openblas64_ which uses _64_ suffixed symbols
    (e.g. dsyevr_64_ instead of dsyevr_). The "64" in the library name indicates
    64-bit integer interface.

    Returns:
        True if the bundled library name contains "openblas64", False otherwise.
    """
    if platform.system() != "Linux":
        return False

    try:
        import numpy as np
    except ImportError:
        return False

    np_dir = Path(np.__file__).parent
    candidates = [
        np_dir / ".libs",
        np_dir.parent / "numpy.libs",
    ]
    for d in candidates:
        if not d.is_dir():
            continue
        for lib in d.iterdir():
            if "openblas64" in lib.name:
                return True
    return False


def compile_extension(verbose: bool = True) -> bool:
    """Compile _eigen_accel.c into a shared library in the installed package.

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
    src = lmm_dir / "_eigen_accel.c"
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
    out = lmm_dir / f"_eigen_accel{ext_suffix}"

    numpy_inc = np.get_include()
    _print(f"NumPy include: {numpy_inc}")

    # Windows/MSVC is not supported — JAMMA targets Linux/macOS only
    if platform.system() == "Windows":
        _print("ERROR: Windows is not supported for C extension compilation")
        return False

    # ILP64 detection — determines which DSYEVR symbol to use
    ilp64 = _detect_ilp64()
    if ilp64:
        _print("ILP64 numpy detected — compiling with -DJAMMA_ILP64 (dsyevr_64_)")
    else:
        _print("LP64 numpy detected — compiling with dsyevr_ (standard symbol)")

    # Platform flags
    ldflags: list[str] = []
    if platform.system() == "Darwin":
        ldflags = ["-undefined", "dynamic_lookup"]

    # ILP64 define
    ilp64_flags: list[str] = []
    if ilp64:
        ilp64_flags = ["-DJAMMA_ILP64"]

    # LAPACK linkage: on Linux, numpy bundles OpenBLAS in numpy.libs/.
    # Without explicit linkage, dsyevr_ is an undefined symbol at import.
    # macOS uses -undefined dynamic_lookup (Accelerate resolves at runtime).
    lapack_flags = _find_numpy_lapack_flags()
    if lapack_flags:
        _print(f"LAPACK link flags: {' '.join(lapack_flags)}")

    # -march=native is safe here: compiles on the user's own machine.
    # hatch_build.py omits this flag for portable wheel builds.
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
        *ilp64_flags,
        str(src),
        "-o",
        str(out),
        "-lm",
        *lapack_flags,
        *ldflags,
    ]

    _print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        _print(f"ERROR: compilation failed:\n{result.stderr}")
        return False

    _print(f"Compiled: {out}")

    # Verify import
    try:
        # Force re-import by removing cached failure
        mods_to_remove = [
            k for k in sys.modules if k.startswith("jamma.lmm._eigen_accel")
        ]
        for k in mods_to_remove:
            del sys.modules[k]

        from jamma.lmm._eigen_accel import eigh_dsyevr  # noqa: F401

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
