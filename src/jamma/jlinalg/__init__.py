"""Vendor BLAS/LAPACK dispatch with NumPy fallbacks.

This package facade owns backend discovery and reload semantics. Numerical
contracts and fallback implementations live in operation-specific modules.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from typing import Literal

import numpy as np

from jamma._build_support.compile_and_link import JLINALG_SPEC
from jamma.core.constants import Env, env_flag
from jamma.core.recompile import _load_c_module
from jamma.jlinalg import _dgemm as _dgemm_operation
from jamma.jlinalg import _dsyrk as _dsyrk_operation
from jamma.jlinalg import _eigh as _eigh_operation
from jamma.jlinalg._eigh import EighStatus
from jamma.jlinalg._snp_stats import compute_snp_stats_chunk as _numpy_snp_stats

_EXPECTED_JLINALG_ABI = 19
_so_exists = importlib.util.find_spec("jamma.jlinalg._jlinalg") is not None
_force_numpy = Env.current().force_numpy_fallback

# Private aliases retained for backend-specific tests and diagnostics.
_dgemm_numpy_impl = _dgemm_operation.numpy_impl
_dgemm_numpy = _dgemm_operation.numpy
_dsyrk_numpy_impl = _dsyrk_operation.numpy_impl
_dsyrk_numpy = _dsyrk_operation.numpy
_eigh_numpy = _eigh_operation.numpy

_dgemm_backend = _dgemm_numpy_impl
_dsyrk_backend = _dsyrk_numpy_impl
_eigh_backend = _eigh_numpy

HAS_C_EXTENSION = False
_module = _load_c_module(JLINALG_SPEC, _EXPECTED_JLINALG_ABI)

if _module is not None:
    ABI_VERSION = _module.ABI_VERSION
    HAS_OPENMP = _module.HAS_OPENMP
    blas_backend = _module.blas_backend
    blas_has_dgemm = _module.blas_has_dgemm
    blas_has_dsyevd = _module.blas_has_dsyevd
    blas_has_dsyevr = _module.blas_has_dsyevr
    blas_has_dsyrk = _module.blas_has_dsyrk
    blas_has_lapacke_dsyevd = _module.blas_has_lapacke_dsyevd
    blas_is_ilp64 = _module.blas_is_ilp64
    compute_snp_stats_chunk = _module.compute_snp_stats_chunk
    get_n_threads = _module.get_n_threads
    jlinalg_isa = _module.jlinalg_isa
    set_n_threads = _module.set_n_threads

    HAS_C_EXTENSION = True
    _dgemm_backend = _module.dgemm if blas_has_dgemm else _dgemm_operation.numpy_impl
    _dsyrk_backend = _module.dsyrk if blas_has_dsyrk else _dsyrk_operation.numpy_impl
    if blas_has_dsyevd or blas_has_dsyevr:
        _eigh_backend = _eigh_operation.native_wrapper(_module.eigh)

else:
    ABI_VERSION = 0
    HAS_OPENMP = False
    jlinalg_isa = "numpy-fallback"
    blas_backend = "numpy-fallback-forced" if _force_numpy else "numpy-fallback"
    blas_is_ilp64 = 0
    blas_has_dgemm = 0
    blas_has_dsyrk = 0
    blas_has_dsyevd = 0
    blas_has_dsyevr = 0
    blas_has_lapacke_dsyevd = 0
    compute_snp_stats_chunk = _numpy_snp_stats

    if not _force_numpy:
        message = (
            "jlinalg C extension found but failed to load; this usually indicates "
            "an ABI mismatch or missing shared library. Falling back to NumPy "
            "(slower). Reinstall jamma or run "
            "'python -m jamma.jlinalg._compile_jlinalg'."
            if _so_exists
            else "jlinalg C extension not compiled; using NumPy fallback (slower). "
            "Run 'python -m jamma.jlinalg._compile_jlinalg' to compile."
        )
        warnings.warn(message, stacklevel=2)

    _fallback_thread_state = [os.cpu_count() or 1]

    def get_n_threads() -> int:
        """Return the fallback thread count."""
        return _fallback_thread_state[0]

    def set_n_threads(n: int) -> int:
        """Set and return the previous fallback thread count."""
        if n < 1:
            raise ValueError("set_n_threads: n must be >= 1")
        previous = _fallback_thread_state[0]
        _fallback_thread_state[0] = min(n, os.cpu_count() or 1)
        return previous


def dgemm(
    A: np.ndarray,
    B: np.ndarray,
    transa: str = "N",
    transb: str = "N",
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``op(A) @ op(B)`` through vendor BLAS or NumPy."""
    return _dgemm_operation.run(_dgemm_backend, A, B, transa, transb, out)


def dsyrk(
    X: np.ndarray, *, out: np.ndarray | None = None, beta: float = 0.0
) -> np.ndarray:
    """Compute ``X @ X.T + beta*out`` through vendor BLAS or NumPy."""
    return _dsyrk_operation.run(_dsyrk_backend, X, out=out, beta=beta)


def dsyrk_scratch_bytes(n: int) -> int:
    """Return scratch memory required beyond the n-by-n result."""
    return _dsyrk_operation.scratch_bytes(n, _dsyrk_backend)


def eigh(
    K: np.ndarray,
    inplace: bool = False,
    *,
    driver: Literal["auto", "dsyevd", "dsyevr"] = "auto",
) -> tuple[np.ndarray, np.ndarray, EighStatus]:
    """Eigendecompose a symmetric matrix through vendor LAPACK or NumPy."""
    return _eigh_operation.run(
        _eigh_backend,
        K,
        inplace,
        driver,
        force_numpy=env_flag("JLINALG_NO_VENDOR_LAPACK"),
    )


__all__ = [
    "ABI_VERSION",
    "HAS_C_EXTENSION",
    "HAS_OPENMP",
    "EighStatus",
    "blas_backend",
    "blas_has_dgemm",
    "blas_has_dsyevd",
    "blas_has_dsyevr",
    "blas_has_dsyrk",
    "blas_has_lapacke_dsyevd",
    "blas_is_ilp64",
    "compute_snp_stats_chunk",
    "dgemm",
    "dsyrk",
    "dsyrk_scratch_bytes",
    "eigh",
    "get_n_threads",
    "jlinalg_isa",
    "set_n_threads",
]
