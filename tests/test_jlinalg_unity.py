"""VALID-01: C-level boundary tests via Unity test framework.

Compiles and runs the C test harness (test_boundaries.c) which validates
dgemm, dsyrk, and eigh at blocking boundary sizes (MR/NR/MC/NC/KC +/- 1)
directly from C, without Python overhead.

The harness is compiled by _compile_jlinalg.compile_test_harness() using the
same compiler flags as the main extension, and invoked via subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from jamma.jlinalg import HAS_C_EXTENSION, blas_has_dsyevd, blas_has_dsyevr

pytestmark = pytest.mark.tier0


def _python_env() -> dict[str, str]:
    """Build environment dict with PYTHONHOME/PYTHONPATH for the test binary.

    The test binary embeds Py_Initialize() to satisfy blas_dispatch.c's
    Python C API calls.  It needs PYTHONHOME pointing to the Python
    installation's base prefix, and PYTHONPATH including site-packages
    and the project src directory.
    """
    env = os.environ.copy()
    env["PYTHONHOME"] = sys.base_prefix
    # Include site-packages for numpy, and src/ for jamma
    import sysconfig

    site_packages = sysconfig.get_path("purelib")
    src_dir = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = f"{site_packages}:{src_dir}"
    return env


@pytest.fixture(scope="module")
def c_test_binary():
    """Compile and return path to C test binary."""
    if not HAS_C_EXTENSION:
        pytest.skip("jlinalg C extension not compiled")
    if not blas_has_dsyevd and not blas_has_dsyevr:
        pytest.skip("No vendor LAPACK — C eigh tests require DSYEVD or DSYEVR")
    from jamma.jlinalg._compile_jlinalg import compile_test_harness

    try:
        binary = compile_test_harness()
    except RuntimeError as e:
        if "not found" in str(e):
            pytest.skip(f"C source files archived (v5.0 simplification): {e}")
        raise
    if not binary.exists():
        pytest.fail(f"Failed to compile C test harness at {binary}")
    return binary


def test_c_boundary_tests(c_test_binary):
    """Run Unity C boundary tests for dgemm, dsyrk, eigh."""
    result = subprocess.run(
        [str(c_test_binary)],
        capture_output=True,
        text=True,
        timeout=120,
        env=_python_env(),
    )
    # Print output for visibility in pytest -v
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        pytest.fail(
            f"C boundary tests failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
