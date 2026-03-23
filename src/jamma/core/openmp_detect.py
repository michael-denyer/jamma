"""OpenMP detection for C extension compilation.

Shared by _compile_jlinalg.py, _compile_accel.py, and (by copy) hatch_build.py.
hatch_build.py cannot import from jamma.* at wheel-build time, so it maintains
its own copy — keep the two in sync.

The core problem: MKL-backed numpy bundles Intel OpenMP (libiomp5). If we also
link against GNU OpenMP (libgomp via GCC's ``-fopenmp`` at link time), both
runtimes initialize in the same process and Intel's runtime aborts with
``OMP: Error #13: Assertion failure at kmp_runtime.cpp``.

Solution: split OpenMP flags into compile-only (``-fopenmp``) and link-only
(libiomp5 by full path, no ``-fopenmp`` to the linker). This prevents GCC
from implicitly adding ``-lgomp``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path


def detect_openmp_flags(
    cc_cmd: str,
    system: str,
    _print: Callable[..., None] = print,
) -> tuple[list[str], list[str]]:
    """Detect OpenMP compile and link flags for the current platform.

    Args:
        cc_cmd: C compiler command name (e.g. "gcc", "cc").
        system: Platform name from ``platform.system()`` ("Linux" or "Darwin").
        _print: Print function for verbose output.

    Returns:
        ``(compile_flags, link_flags)`` for OpenMP, or ``([], [])`` if
        unavailable.
    """
    if os.environ.get("JAMMA_NO_OPENMP", "").strip() not in ("", "0"):
        _print(
            "OpenMP disabled (JAMMA_NO_OPENMP set). "
            "C extensions will be single-threaded."
        )
        return ([], [])
    if system == "Darwin":
        return _detect_darwin_openmp_flags(_print)
    return _detect_linux_openmp_flags(cc_cmd, _print)


def _detect_darwin_openmp_flags(
    _print: Callable[..., None],
) -> tuple[list[str], list[str]]:
    """Detect OpenMP via Homebrew libomp on macOS."""
    try:
        prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        lib_dir = Path(prefix) / "lib"
        if lib_dir.is_dir():
            _print(f"OpenMP: Homebrew libomp at {prefix}")
            return (
                [f"-I{prefix}/include", "-Xpreprocessor", "-fopenmp"],
                [f"-L{prefix}/lib", "-lomp"],
            )
        _print(
            f"OpenMP: Homebrew prefix {prefix} exists but lib/ not found. "
            "Extension will be single-threaded. "
            "Install for parallelism: brew install libomp"
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        _print(
            "OpenMP not available (libomp not found via Homebrew). "
            "Extension will be single-threaded. "
            "Install for parallelism: brew install libomp"
        )
    return ([], [])


def _detect_linux_openmp_flags(
    cc_cmd: str, _print: Callable[..., None]
) -> tuple[list[str], list[str]]:
    """Detect the best OpenMP flags for Linux.

    Prefers Intel OpenMP (libiomp5) when available to avoid the libgomp/libiomp5
    dual-runtime conflict on systems with MKL-backed numpy. MKL uses libiomp5
    internally; linking against libgomp creates two OpenMP runtimes in the
    same process, which can cause thread oversubscription or hangs.

    On GCC, ``-fopenmp`` at link time implicitly adds ``-lgomp``.  When we
    link libiomp5 by full path, passing ``-fopenmp`` to the linker would load
    *both* libgomp and libiomp5 into the same process — triggering an Intel
    OMP assertion failure (``kmp_runtime.cpp``).  We therefore split the
    return into compile-only and link-only flags.

    Detection order:
    1. libiomp5 (Intel OpenMP) — found via numpy's MKL libs or system paths
    2. libgomp (GNU OpenMP) — standard fallback via -fopenmp
    """
    # Check if numpy bundles MKL (and thus libiomp5)
    try:
        import numpy as np

        np_dir = Path(np.__file__).parent
        search_dirs = [
            np_dir / ".libs",
            np_dir.parent / "numpy.libs",
            np_dir / "_core" / ".libs",
        ]
        for d in search_dirs:
            if not d.is_dir():
                continue
            for lib in d.iterdir():
                if "libiomp5" in lib.name and ".so" in lib.name:
                    _print(f"Intel OpenMP found: {lib}")
                    # Link by full path — numpy bundles versioned names like
                    # libiomp5-2f035e84.so with no unversioned symlink, so
                    # -liomp5 fails at link time.  Do NOT pass -fopenmp to
                    # the linker: GCC would add -lgomp, creating a dual
                    # OpenMP runtime that aborts at init.
                    return (
                        ["-fopenmp"],
                        [str(lib), f"-Wl,-rpath,{d}"],
                    )
    except ImportError:
        pass

    # Check system-wide libiomp5 (e.g. from intel-openmp package)
    result = subprocess.run(
        [cc_cmd, "-liomp5", "-x", "c", "-", "-o", "/dev/null"],
        input="int main(){return 0;}\n",
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        _print("Intel OpenMP (system libiomp5) detected")
        return (["-fopenmp"], ["-liomp5"])

    # Fallback to GNU OpenMP (safe to use -fopenmp for both compile and link)
    return (["-fopenmp"], ["-fopenmp"])
