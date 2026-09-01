"""DGEMM contract validation and NumPy implementation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DgemmBackend = Callable[
    [np.ndarray, np.ndarray, str, str, np.ndarray | None], np.ndarray
]


def validate(
    a: np.ndarray,
    b: np.ndarray,
    transa: str,
    transb: str,
    out: np.ndarray | None,
) -> None:
    """Validate the public DGEMM contract before backend dispatch."""
    if a.ndim != 2:
        raise ValueError(f"dgemm: A must be a 2-D array, got {a.ndim}-D")
    if b.ndim != 2:
        raise ValueError(f"dgemm: B must be a 2-D array, got {b.ndim}-D")
    if not isinstance(transa, str):
        raise TypeError(f"dgemm: transa must be a string, got {type(transa).__name__}")
    if not isinstance(transb, str):
        raise TypeError(f"dgemm: transb must be a string, got {type(transb).__name__}")
    if transa.upper() not in ("N", "T"):
        raise ValueError(f"dgemm: transa must be 'N' or 'T', got '{transa}'")
    if transb.upper() not in ("N", "T"):
        raise ValueError(f"dgemm: transb must be 'N' or 'T', got '{transb}'")

    m = a.shape[1] if transa.upper() == "T" else a.shape[0]
    k_a = a.shape[0] if transa.upper() == "T" else a.shape[1]
    k_b = b.shape[1] if transb.upper() == "T" else b.shape[0]
    n = b.shape[0] if transb.upper() == "T" else b.shape[1]
    if k_a != k_b:
        raise ValueError(f"dgemm: op(A) columns ({k_a}) must match op(B) rows ({k_b})")
    if out is None:
        return

    expected = (m, n)
    if out.ndim != 2 or out.shape != expected:
        raise ValueError(
            f"dgemm: out shape {out.shape} doesn't match result shape {expected}"
        )
    if out.dtype != np.float64:
        raise ValueError(f"dgemm: out must be float64, got {out.dtype}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("dgemm: out must be C-contiguous")
    if not out.flags["ALIGNED"]:
        raise ValueError("dgemm: out must be aligned")
    if not out.flags["WRITEABLE"]:
        raise ValueError("dgemm: out must be writeable")


def numpy_impl(
    a: np.ndarray,
    b: np.ndarray,
    transa: str = "N",
    transb: str = "N",
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``op(a) @ op(b)`` after the caller has validated inputs."""
    left = a.T if transa.upper() == "T" else a
    right = b.T if transb.upper() == "T" else b
    if out is not None:
        np.matmul(
            left.astype(np.float64, copy=False),
            right.astype(np.float64, copy=False),
            out=out,
        )
        return out
    return np.asarray(
        np.matmul(
            left.astype(np.float64, copy=False),
            right.astype(np.float64, copy=False),
        ),
        dtype=np.float64,
    )


def numpy(
    a: np.ndarray,
    b: np.ndarray,
    transa: str = "N",
    transb: str = "N",
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Validated NumPy implementation exposed for backend-specific tests."""
    validate(a, b, transa, transb, out)
    return numpy_impl(a, b, transa, transb, out)


def run(
    backend: DgemmBackend,
    a: np.ndarray,
    b: np.ndarray,
    transa: str,
    transb: str,
    out: np.ndarray | None,
) -> np.ndarray:
    """Validate once and dispatch to the selected implementation."""
    validate(a, b, transa, transb, out)
    return backend(a, b, transa, transb, out)
