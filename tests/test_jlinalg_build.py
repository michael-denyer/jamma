"""Build and import verification tests for jamma.jlinalg.

- Import succeeds and module constants have correct types
- ISA detection returns a valid string
- All documented exports are present
"""

import pytest

import jamma.jlinalg as jlinalg
from jamma.jlinalg import (
    HAS_C_EXTENSION,
    HAS_OPENMP,
    jlinalg_isa,
)

pytestmark = pytest.mark.tier0

_VALID_ISA_STRINGS = {"AVX2", "NEON", "generic", "numpy-fallback"}


def test_import():
    """jlinalg_isa is a string and the module is importable."""
    assert isinstance(jlinalg_isa, str), f"Expected str, got {type(jlinalg_isa)}"


def test_isa_detection():
    """jlinalg_isa is one of the known ISA strings."""
    assert jlinalg_isa in _VALID_ISA_STRINGS, (
        f"Unknown ISA string: {jlinalg_isa!r}. Expected one of {_VALID_ISA_STRINGS}"
    )


def test_has_c_extension_type():
    """HAS_C_EXTENSION is a bool."""
    assert isinstance(HAS_C_EXTENSION, bool), (
        f"Expected bool, got {type(HAS_C_EXTENSION)}"
    )


def test_has_openmp_type():
    """HAS_OPENMP is a bool."""
    assert isinstance(HAS_OPENMP, bool), f"Expected bool, got {type(HAS_OPENMP)}"


def test_all_exports_present():
    """All documented exports are present in jamma.jlinalg.__all__."""
    expected = {
        "ABI_VERSION",
        "dgemm",
        "dsyrk",
        "eigh",
        "get_n_threads",
        "set_n_threads",
        "blas_backend",
        "blas_has_dgemm",
        "blas_has_dsyrk",
        "blas_has_dsyevd",
        "blas_has_dsyevr",
        "blas_has_lapacke_dsyevd",
        "blas_is_ilp64",
        "jlinalg_isa",
        "HAS_C_EXTENSION",
        "HAS_OPENMP",
        "compute_snp_stats_chunk",
    }
    missing = expected - set(jlinalg.__all__)
    assert not missing, f"Missing exports: {missing}"


def test_abi_version():
    """ABI_VERSION is 19 after plumbing eigh's driver parameter through."""
    from jamma.jlinalg import ABI_VERSION

    assert ABI_VERSION == 19, f"Expected ABI_VERSION=19, got {ABI_VERSION}"


def test_dgemm_exported():
    """dgemm is callable and C extension exports it when available."""
    from jamma.jlinalg import HAS_C_EXTENSION, dgemm

    assert callable(dgemm)
    if HAS_C_EXTENSION:
        from jamma.jlinalg import _jlinalg  # type: ignore[import]

        assert callable(getattr(_jlinalg, "dgemm", None)), (
            "C extension loaded but does not export 'dgemm'."
        )
