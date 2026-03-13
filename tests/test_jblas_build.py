"""Build and import verification tests for jamma.jblas.

Covers BUILD-01 through BUILD-06 requirements:
- Import succeeds (BUILD-03)
- ISA detection returns valid string (BUILD-02)
- HAS_C_EXTENSION and HAS_OPENMP are bools (BUILD-03)
- Fallback functions have correct signatures (BUILD-05)
- OpenMP parallel ddot gives correct result (BUILD-06) — skipped when no C extension
"""

import numpy as np
import pytest

import jamma.jblas as jblas
from jamma.jblas import (
    HAS_C_EXTENSION,
    HAS_OPENMP,
    daxpy,
    ddot,
    dgemv,
    dnrm2,
    dscal,
    jblas_isa,
)

_VALID_ISA_STRINGS = {"AVX2", "NEON", "generic", "numpy-fallback"}


@pytest.mark.tier0
def test_import():
    """jblas_isa is a string and the module is importable."""
    assert isinstance(jblas_isa, str), f"Expected str, got {type(jblas_isa)}"


@pytest.mark.tier0
def test_isa_detection():
    """jblas_isa is one of the known ISA strings."""
    assert jblas_isa in _VALID_ISA_STRINGS, (
        f"Unknown ISA string: {jblas_isa!r}. Expected one of {_VALID_ISA_STRINGS}"
    )


@pytest.mark.tier0
def test_has_c_extension_type():
    """HAS_C_EXTENSION is a bool."""
    assert isinstance(HAS_C_EXTENSION, bool), (
        f"Expected bool, got {type(HAS_C_EXTENSION)}"
    )


@pytest.mark.tier0
def test_has_openmp_type():
    """HAS_OPENMP is a bool."""
    assert isinstance(HAS_OPENMP, bool), f"Expected bool, got {type(HAS_OPENMP)}"


@pytest.mark.tier0
def test_fallback_signatures():
    """All 5 functions accept documented args and return expected types."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8)
    y = rng.standard_normal(8)
    A = rng.standard_normal((4, 8))

    # ddot: (ndarray, ndarray) -> float
    result_ddot = ddot(x, y)
    assert isinstance(result_ddot, float), (
        f"ddot returned {type(result_ddot)}, expected float"
    )

    # dnrm2: (ndarray) -> float
    result_dnrm2 = dnrm2(x)
    assert isinstance(result_dnrm2, float), (
        f"dnrm2 returned {type(result_dnrm2)}, expected float"
    )

    # daxpy: (float, ndarray, ndarray) -> None
    y_copy = y.copy()
    result_daxpy = daxpy(2.0, x, y_copy)
    assert result_daxpy is None, f"daxpy returned {result_daxpy!r}, expected None"

    # dscal: (float, ndarray) -> None
    x_copy = x.copy()
    result_dscal = dscal(3.0, x_copy)
    assert result_dscal is None, f"dscal returned {result_dscal!r}, expected None"

    # dgemv: (ndarray, ndarray) -> ndarray
    x_short = rng.standard_normal(8)
    result_dgemv = dgemv(A, x_short)
    assert isinstance(result_dgemv, np.ndarray), (
        f"dgemv returned {type(result_dgemv)}, expected ndarray"
    )
    assert result_dgemv.shape == (4,), (
        f"dgemv shape {result_dgemv.shape}, expected (4,)"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not HAS_C_EXTENSION, reason="C extension not compiled")
def test_openmp_ddot():
    """OpenMP parallel ddot produces correct result (data race detection).

    This test runs a large ddot to exercise parallel reduction paths.
    If parallelism causes a data race, the result will differ from np.dot.
    """
    rng = np.random.default_rng(12345)
    n = 100_000
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    result = ddot(x, y)
    expected = np.dot(x, y)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-12,
        err_msg=f"OpenMP ddot data race: got {result}, expected {expected}",
    )


@pytest.mark.tier0
def test_all_exports_present():
    """All documented exports are present in jamma.jblas.__all__."""
    expected = {
        "ddot",
        "dnrm2",
        "daxpy",
        "dscal",
        "dgemv",
        "dgemm",
        "jblas_isa",
        "HAS_C_EXTENSION",
        "HAS_OPENMP",
    }
    missing = expected - set(jblas.__all__)
    assert not missing, f"Missing exports: {missing}"
