"""Post-install compiler for the _jblas C extension.

Run this after ``pip install jamma`` to compile the jblas C extension in-place:

    python -m jamma.jblas._compile_jblas

Or from a Databricks/Jupyter notebook cell:

    from jamma.jblas._compile_jblas import compile_extension
    compile_extension()

Requires: gcc (or cc), Python development headers, numpy >= 2.0.
OpenMP support is optional — falls back to single-threaded if unavailable.

The jblas extension compiles per-file to enable per-source-group compiler flags
in future phases (e.g. LAPACK files needing -ldl).  Currently all sources use
the same flags; the x86_64/aarch64 ISA split is handled by #if guards in C.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def _detect_linux_openmp_flags(cc_cmd: str, _print: object = print) -> list[str]:
    """Detect the best OpenMP flags for Linux.

    Prefers Intel OpenMP (libiomp5) when available to avoid the libgomp/libiomp5
    dual-runtime conflict on systems with MKL-backed numpy. MKL uses libiomp5
    internally; linking _jblas against libgomp creates two OpenMP runtimes in the
    same process, which can cause thread oversubscription or hangs.

    Detection order:
    1. libiomp5 (Intel OpenMP) — found via numpy's MKL libs or system paths
    2. libgomp (GNU OpenMP) — standard fallback via -fopenmp

    Args:
        cc_cmd: C compiler command name.
        _print: Print function (for verbose output).

    Returns:
        Compiler flags for OpenMP, or empty list if unavailable.
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
                    return [
                        f"-L{d}",
                        "-liomp5",
                        f"-Wl,-rpath,{d}",
                        "-fopenmp",
                    ]
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
        return ["-liomp5", "-fopenmp"]

    # Fallback to GNU OpenMP
    return ["-fopenmp"]


def compile_extension(verbose: bool = True) -> bool:
    """Compile jblas C sources into a shared library in the installed package.

    Performs per-file compile-then-link to enable different compiler flags
    per source group. Currently all sources use the same base flags with
    an x86_64/aarch64 ISA split for -mavx2/-mfma.

    Args:
        verbose: Print progress and diagnostics to stderr.

    Returns:
        True if compilation succeeded, False otherwise.
    """

    def _print(*args: object) -> None:
        if verbose:
            print(*args, file=sys.stderr, flush=True)

    # Locate jblas source directory relative to this file
    jblas_dir = Path(__file__).parent
    jblas_src_dir = jblas_dir / "src"
    jblas_inc_dir = jblas_dir / "include"

    if not jblas_src_dir.is_dir():
        _print(f"ERROR: jblas source directory not found: {jblas_src_dir}")
        _print("  Package may be incomplete — reinstall from source.")
        return False

    # Source files split into baseline (portable) and SIMD (AVX2 intrinsics).
    # Only SIMD sources get -mavx2/-mfma to avoid SIGILL on older x86_64 CPUs.
    baseline_sources = [
        jblas_src_dir / "platform.c",
        jblas_src_dir / "dnrm2.c",
        jblas_src_dir / "dgemv.c",
        jblas_src_dir / "pymodule.c",
    ]
    simd_sources = [
        jblas_src_dir / "ddot.c",
        jblas_src_dir / "daxpy.c",
        jblas_src_dir / "dscal.c",
    ]
    source_files = baseline_sources + simd_sources

    missing = [str(s) for s in source_files if not s.exists()]
    if missing:
        _print(f"ERROR: jblas source files missing: {missing}")
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

    _print(f"numpy {np.__version__} OK")

    # Compiler
    cc_name = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    cc_cmd = cc_name.split()[0]
    cc_extra = cc_name.split()[1:]

    cc_path = shutil.which(cc_cmd)
    if not cc_path:
        _print(f"ERROR: C compiler '{cc_cmd}' not found on PATH")
        _print("  Install: apt-get install -y gcc  (Linux)")
        _print("  Install: xcode-select --install  (macOS)")
        return False
    _print(f"Compiler: {cc_path}")

    # Python headers
    python_inc = sysconfig.get_config_var("INCLUDEPY") or ""
    python_h = Path(python_inc) / "Python.h" if python_inc else None
    if not python_h or not python_h.exists():
        _print(f"ERROR: Python.h not found at {python_inc}")
        _print("  Install: apt-get install -y python3-dev  (Linux)")
        return False
    _print(f"Python.h: {python_h}")

    # Output path (next to __init__.py in the installed package)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = jblas_dir / f"_jblas{ext_suffix}"
    numpy_inc = np.get_include()
    _print(f"NumPy include: {numpy_inc}")
    _print(f"Output: {out}")

    # Windows is not supported
    if platform.system() == "Windows":
        _print("ERROR: Windows is not supported for C extension compilation")
        return False

    # ISA-specific flags — applied only to SIMD sources (ddot.c, daxpy.c, dscal.c).
    # Baseline sources (platform.c, pymodule.c, etc.) are compiled without SIMD
    # flags so the extension runs on any x86_64 CPU; runtime CPUID dispatch in
    # platform.c selects the appropriate kernels.
    machine = platform.machine()
    simd_flags: list[str] = []
    if machine in ("x86_64", "AMD64"):
        simd_flags = ["-mavx2", "-mfma"]
        _print(f"ISA: x86_64 — SIMD sources get {simd_flags}")
    elif machine in ("aarch64", "arm64"):
        _print("ISA: aarch64/arm64 — no extra SIMD flags (generics only)")
    else:
        _print(f"ISA: {machine} — no extra SIMD flags")

    # OpenMP detection
    omp_flags: list[str] = []
    ldflags: list[str] = []
    if platform.system() == "Darwin":
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            lib_dir = Path(prefix) / "lib"
            if lib_dir.is_dir():
                omp_flags = [
                    f"-I{prefix}/include",
                    f"-L{prefix}/lib",
                    "-Xpreprocessor",
                    "-fopenmp",
                    "-lomp",
                ]
                _print(f"OpenMP: Homebrew libomp at {prefix}")
            else:
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
        ldflags = ["-undefined", "dynamic_lookup"]
    else:
        omp_flags = _detect_linux_openmp_flags(cc_cmd, _print)

    # Base compile flags (shared by all source files — no SIMD flags here)
    base_cflags = [
        "-O3",
        "-ftree-vectorize",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-funroll-loops",
        "-fno-finite-math-only",  # ensure isnan() works correctly
        "-fPIC",
        "-std=c11",
        f"-I{python_inc}",
        f"-I{numpy_inc}",
        f"-I{jblas_inc_dir}",
    ]

    simd_source_set = set(str(s) for s in simd_sources)

    # OpenMP compile-time flags (exclude link-only flags like -lomp/-liomp5)
    omp_compile: list[str] = []
    for flag in omp_flags:
        if flag.startswith("-I") or flag.startswith("-L") or flag.startswith("-Wl"):
            omp_compile.append(flag)
        elif flag in ("-Xpreprocessor", "-fopenmp"):
            omp_compile.append(flag)

    def _compile_sources(
        extra_compile: list[str], suffix: str
    ) -> tuple[list[Path], Path] | None:
        """Compile all source files with the given extra flags.

        Returns list of object files and temp dir path, or None on failure.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="jblas_compile_"))
        obj_files: list[Path] = []
        try:
            for src in source_files:
                obj_file = tmp_dir / f"{src.stem}{suffix}.o"
                extra_simd = simd_flags if str(src) in simd_source_set else []
                cmd = [
                    cc_cmd,
                    *cc_extra,
                    *base_cflags,
                    *extra_simd,
                    *extra_compile,
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
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return None
                obj_files.append(obj_file)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return obj_files, tmp_dir

    # First attempt: with OpenMP
    use_omp = bool(omp_flags)
    current_omp = omp_flags
    compile_result = _compile_sources(omp_compile, "")
    tmp_dir = None

    if compile_result is None and use_omp:
        _print(
            "OpenMP compilation failed, retrying without OpenMP (single-threaded)..."
        )
        compile_result = _compile_sources([], "_noomp")
        current_omp = []  # no OMP runtime to link

    if compile_result is None:
        _print("ERROR: jblas compilation failed")
        return False

    obj_files, tmp_dir = compile_result

    try:
        # Link all object files into the shared library
        cmd_link = [
            cc_cmd,
            *cc_extra,
            "-shared",
            "-fPIC",
            *[str(o) for o in obj_files],
            "-o",
            str(out),
            "-lm",
            *current_omp,
            *ldflags,
        ]
        _print(f"link: {' '.join(cmd_link)}")
        result_link = subprocess.run(cmd_link, capture_output=True, text=True)
        if result_link.returncode != 0:
            _print(f"ERROR: link failed:\n{result_link.stderr}")
            return False
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _print(f"Compiled: {out}")

    # Verify import — evict both _jblas and the parent jamma.jblas package
    # so the freshly compiled extension is loaded instead of the cached fallback.
    try:
        mods_to_remove = [k for k in sys.modules if k.startswith("jamma.jblas")]
        for k in mods_to_remove:
            del sys.modules[k]

        from jamma.jblas._jblas import HAS_OPENMP, jblas_isa  # noqa: F401

        _print(f"Import OK — jblas_isa={jblas_isa!r}, HAS_OPENMP={HAS_OPENMP}")
        _print("C extension is active")
        _print("Reloaded jamma.jblas — HAS_C_EXTENSION is now True in this process.")
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
        _print(f"ERROR: compiled but import failed ({type(e).__name__}): {e}")
        return False


if __name__ == "__main__":
    success = compile_extension()
    sys.exit(0 if success else 1)
