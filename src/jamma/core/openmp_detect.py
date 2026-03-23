"""OpenMP detection for C extension compilation.

Shared by _compile_jlinalg.py, _compile_accel.py, and (by copy) hatch_build.py.
hatch_build.py cannot import from jamma.* at wheel-build time, so it maintains
its own copy — keep the two in sync.

The core problem: MKL-backed numpy bundles Intel OpenMP (libiomp5).  Two
failure modes exist when compiling C extensions with OpenMP on such systems:

1. **Dual runtime** (Error #15): linking ``-fopenmp`` on GCC implicitly adds
   ``-lgomp``, creating two OpenMP runtimes.  Suppressed by
   ``KMP_DUPLICATE_LIB_OK=TRUE`` but unreliable.

2. **GOMP shim assertion** (Error #13): even when linked against libiomp5
   only, GCC's ``-fopenmp`` emits ``GOMP_*`` ABI calls.  libiomp5 provides
   a compatibility shim for these, but it has assertion failures
   (``kmp_runtime.cpp``) after heavy MKL LAPACK usage (e.g. DSYEVR).

Solution: prefer **clang** when libiomp5 is present.  Clang with
``-fopenmp`` natively generates ``kmp_*`` calls that libiomp5 handles
correctly — no GOMP shim involved.  Falls back to GCC with a warning.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path


def detect_openmp_flags(
    cc_cmd: str,
    system: str,
    _print: Callable[..., None] = print,
) -> tuple[list[str], list[str], str]:
    """Detect OpenMP compile and link flags for the current platform.

    Args:
        cc_cmd: C compiler command name (e.g. "gcc", "cc").
        system: Platform name from ``platform.system()`` ("Linux" or "Darwin").
        _print: Print function for verbose output.

    Returns:
        ``(compile_flags, link_flags, cc_override)`` for OpenMP, or
        ``([], [], cc_cmd)`` if unavailable.  ``cc_override`` may differ
        from *cc_cmd* when the detector switches to clang for libiomp5
        compatibility (see ``_detect_linux_openmp_flags``).
    """
    if os.environ.get("JAMMA_NO_OPENMP", "").strip() not in ("", "0"):
        _print(
            "OpenMP disabled (JAMMA_NO_OPENMP set). "
            "C extensions will be single-threaded."
        )
        return ([], [], cc_cmd)
    if system == "Darwin":
        cflags, lflags = _detect_darwin_openmp_flags(_print)
        return (cflags, lflags, cc_cmd)
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
) -> tuple[list[str], list[str], str]:
    """Detect the best OpenMP flags for Linux.

    Prefers Intel OpenMP (libiomp5) when available to avoid the libgomp/libiomp5
    dual-runtime conflict on systems with MKL-backed numpy. MKL uses libiomp5
    internally; linking against libgomp creates two OpenMP runtimes in the
    same process, which can cause thread oversubscription or hangs.

    When libiomp5 is found, we prefer clang over GCC as the compiler.  GCC's
    ``-fopenmp`` emits ``GOMP_*`` ABI calls even when linked against libiomp5;
    libiomp5 provides a GOMP compatibility shim, but it has known assertion
    failures (``kmp_runtime.cpp``) after heavy MKL LAPACK usage (e.g.
    DSYEVR).  Clang with ``-fopenmp`` natively generates ``kmp_*`` calls
    that libiomp5 handles correctly.

    Detection order:
    1. libiomp5 via numpy's bundled MKL libs → prefer clang, fallback to GCC
    2. libiomp5 via system paths → prefer clang, fallback to GCC
    3. libgomp (GNU OpenMP) → standard fallback via -fopenmp
    """
    libiomp5_path = _find_libiomp5(_print)
    if libiomp5_path is not None:
        return _openmp_flags_for_libiomp5(cc_cmd, libiomp5_path, _print)

    # Fallback to GNU OpenMP (safe to use -fopenmp for both compile and link)
    return (["-fopenmp"], ["-fopenmp"], cc_cmd)


def _find_libiomp5(
    _print: Callable[..., None],
) -> Path | None:
    """Locate libiomp5.so — first in numpy's bundled libs, then system-wide."""
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
                    return lib
    except ImportError:
        pass

    # Check well-known system paths for libiomp5
    for search_dir in [Path("/usr/lib"), Path("/usr/lib64"), Path("/usr/local/lib")]:
        if not search_dir.is_dir():
            continue
        for lib in search_dir.iterdir():
            if "libiomp5" in lib.name and ".so" in lib.name:
                _print(f"Intel OpenMP found (system): {lib}")
                return lib

    return None


def _openmp_flags_for_libiomp5(
    cc_cmd: str,
    libiomp5_path: Path,
    _print: Callable[..., None],
) -> tuple[list[str], list[str], str]:
    """Build OpenMP flags for linking against a specific libiomp5.

    Prefers clang when available — clang with ``-fopenmp`` natively generates
    ``kmp_*`` calls compatible with libiomp5.  GCC generates ``GOMP_*`` calls
    that go through libiomp5's compatibility shim, which has known assertion
    failures (``OMP: Error #13 at kmp_runtime.cpp``) after heavy MKL LAPACK
    usage.
    """
    lib_dir = libiomp5_path.parent
    link_flags = [str(libiomp5_path), f"-Wl,-rpath,{lib_dir}"]

    # Prefer clang for libiomp5 compatibility — avoids GOMP shim bugs
    clang_path = shutil.which("clang")
    if clang_path is not None:
        # Verify clang can compile with OpenMP against this libiomp5
        result = subprocess.run(
            [
                clang_path,
                "-fopenmp",
                "-x",
                "c",
                "-",
                "-o",
                "/dev/null",
                str(libiomp5_path),
                f"-Wl,-rpath,{lib_dir}",
            ],
            input="#include <omp.h>\nint main(){return omp_get_max_threads();}\n",
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            _print(
                f"Using clang ({clang_path}) for libiomp5 compatibility "
                f"(avoids GCC GOMP shim assertion failures)"
            )
            return (["-fopenmp"], link_flags, clang_path)
        _print(f"clang found but OpenMP test failed: {result.stderr.strip()}")

    # Fallback to GCC — compile with -fopenmp (generates GOMP_* calls),
    # link against libiomp5 by full path.  This relies on libiomp5's GOMP
    # compatibility shim, which may trigger assertion failures after MKL
    # LAPACK operations on some systems.
    _print(
        f"WARNING: Using {cc_cmd} with libiomp5 — GCC's GOMP compatibility "
        f"shim may cause assertion failures after MKL LAPACK calls.  "
        f"Install clang to avoid this: apt-get install clang"
    )
    return (["-fopenmp"], link_flags, cc_cmd)
