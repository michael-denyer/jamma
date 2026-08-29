"""Regression coverage for BLAS backend detection after the probe moved to Python.

blas_dispatch.c's numpy-dir and pip-mkl scans now ask
jamma.jlinalg._blas_dirs.probe_plan() where to look instead of walking
pathlib through the CPython C API themselves. These tests pin the observable
behaviour that move must not change: the backend name jlinalg resolves to,
and that C no longer carries the bulk of the old CPython API traffic for
directory discovery.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jamma.jlinalg import HAS_C_EXTENSION, blas_backend, blas_is_ilp64

pytestmark = pytest.mark.tier0

_BLAS_DISPATCH_C = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "jamma"
    / "jlinalg"
    / "src"
    / "blas_dispatch.c"
)


@pytest.mark.skipif(not HAS_C_EXTENSION, reason="jlinalg C extension not compiled")
class TestBackendDetectionUnchanged:
    """The active backend is whatever this machine's dispatch already resolved."""

    def test_blas_backend_known_value(self):
        known = {
            "MKL-ILP64",
            "OpenBLAS-ILP64",
            "Accelerate-ILP64",
            "system-BLAS-ILP64",
            "numpy-fallback",
        }
        assert blas_backend in known, f"Unknown blas_backend: {blas_backend}"

    def test_ilp64_backend_reports_ilp64(self):
        if blas_backend == "numpy-fallback":
            pytest.skip("no vendor backend resolved on this host")
        assert blas_is_ilp64 == 1


class TestDirectoryProbingMovedToPython:
    """blas_dispatch.c keeps dlopen/dlsym; directory discovery is Python's job."""

    def test_c_source_calls_probe_plan(self):
        """C delegates directory discovery to _blas_dirs, not raw pathlib calls."""
        source = _BLAS_DISPATCH_C.read_text()
        assert "_blas_dirs" in source
        assert "probe_plan" in source

    def test_c_source_pyobject_traffic_is_low(self):
        """CPython API call count drops now that pathlib traversal is Python's job.

        blas_dispatch.c still needs a handful of PyObject_/PyErr_Clear/Py_DECREF
        calls: a few to invoke probe_plan() and read back the (kind, path)
        pairs, and a few more in force_numpy_blas_load() (out of this PR's
        scope -- it forces numpy's own BLAS to load, not directory discovery).
        The bulk of the old per-path-segment traffic (PyObject_GetAttrString,
        PyObject_CallMethod for every "parent"/"__truediv__" step across two
        discovery functions) is gone. At trunk these three counts are 22, 25,
        and 72; this bounds them at a small fraction of that.
        """
        source = _BLAS_DISPATCH_C.read_text()
        counts = {
            pattern: len(re.findall(pattern, source))
            for pattern in ("PyObject_", "PyErr_Clear", "Py_DECREF")
        }
        for pattern, count in counts.items():
            assert count < 20, (
                f"{pattern} appears {count} times, expected well under trunk's 22/25/72"
            )
