"""Build and import verification tests for jamma.jlinalg.

- Import succeeds and module constants have correct types
- ISA detection returns a valid string
- All fallback functions have correct signatures
- OpenMP parallel ddot gives correct result (skipped when no C extension)
"""

import numpy as np
import pytest

import jamma.jlinalg as jlinalg
from jamma.jlinalg import (
    HAS_C_EXTENSION,
    HAS_OPENMP,
    daxpy,
    ddot,
    dgemv,
    dnrm2,
    dscal,
    jlinalg_isa,
)

_VALID_ISA_STRINGS = {"AVX2", "NEON", "generic", "numpy-fallback"}


@pytest.mark.tier0
def test_import():
    """jlinalg_isa is a string and the module is importable."""
    assert isinstance(jlinalg_isa, str), f"Expected str, got {type(jlinalg_isa)}"


@pytest.mark.tier0
def test_isa_detection():
    """jlinalg_isa is one of the known ISA strings."""
    assert jlinalg_isa in _VALID_ISA_STRINGS, (
        f"Unknown ISA string: {jlinalg_isa!r}. Expected one of {_VALID_ISA_STRINGS}"
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
    """All documented exports are present in jamma.jlinalg.__all__."""
    expected = {
        "ABI_VERSION",
        "ddot",
        "dnrm2",
        "daxpy",
        "dscal",
        "dgemv",
        "dgemm",
        "dsyrk",
        "dsyr2k",
        "eigh",
        "get_n_threads",
        "set_n_threads",
        "blas_backend",
        "blas_has_dsyrk",
        "blas_has_dsyevd",
        "blas_has_lapacke_dsyevd",
        "blas_is_ilp64",
        "jlinalg_isa",
        "HAS_C_EXTENSION",
        "HAS_OPENMP",
        "JLINALG_MR",
        "JLINALG_NR",
        "JLINALG_KC",
        "JLINALG_MC",
        "JLINALG_NC",
    }
    missing = expected - set(jlinalg.__all__)
    assert not missing, f"Missing exports: {missing}"


@pytest.mark.tier0
def test_abi_version():
    """ABI_VERSION is 9 (bumped for DSYEVR vendor dispatch)."""
    from jamma.jlinalg import ABI_VERSION

    assert ABI_VERSION == 9, f"Expected ABI_VERSION=9, got {ABI_VERSION}"


@pytest.mark.tier0
def test_dgemm_exported():
    """dgemm is callable and C extension exports it when available."""
    from jamma.jlinalg import HAS_C_EXTENSION, dgemm

    assert callable(dgemm)
    if HAS_C_EXTENSION:
        from jamma.jlinalg import _jlinalg  # type: ignore[import]

        assert callable(getattr(_jlinalg, "dgemm", None)), (
            "C extension loaded but does not export 'dgemm'."
        )
