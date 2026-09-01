"""DSYRK contract, NumPy implementation, and scratch accounting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DsyrkBackend = Callable[..., np.ndarray]

_BLOCK_BYTES = 64 << 20
_BLOCK_FRACTION = 8
_MIRROR_BLOCK = 512
_MIRROR_SCRATCH_BYTES = 8 * (
    3 * (_MIRROR_BLOCK * (_MIRROR_BLOCK - 1) // 2) + _MIRROR_BLOCK * _MIRROR_BLOCK
)


def validate(x: np.ndarray, out: np.ndarray | None, beta: float) -> None:
    """Validate the public DSYRK contract before backend dispatch."""
    if x.ndim != 2:
        raise ValueError(f"dsyrk: X must be a 2-D array, got {x.ndim}-D")
    if out is None:
        if beta != 0.0:
            raise ValueError("dsyrk: beta requires out")
        return
    if not isinstance(out, np.ndarray):
        raise TypeError("dsyrk: out must be a numpy array")
    if out.dtype != np.float64:
        raise ValueError(f"dsyrk: out must be float64, got {out.dtype}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("dsyrk: out must be C-contiguous")
    if not out.flags["ALIGNED"]:
        raise ValueError("dsyrk: out must be aligned")
    if not out.flags["WRITEABLE"]:
        raise ValueError("dsyrk: out must be writeable")
    if out.ndim != 2:
        raise ValueError(f"dsyrk: out must be 2-D, got {out.ndim}-D")
    expected = (x.shape[0], x.shape[0])
    if out.shape != expected:
        raise ValueError(
            f"dsyrk: out shape {out.shape} doesn't match result shape {expected}"
        )


def _row_block(n: int) -> int:
    by_bytes = _BLOCK_BYTES // max(1, n * 8)
    by_fraction = max(1, n // _BLOCK_FRACTION)
    return max(1, min(n, by_fraction, by_bytes))


def scratch_bytes(n: int, backend: DsyrkBackend) -> int:
    """Return the fallback scratch bound; the native backend allocates none."""
    if backend is not numpy_impl:
        return 0
    return max(_row_block(n) * n * 8, _MIRROR_SCRATCH_BYTES)


def _mirror_lower_to_upper(result: np.ndarray) -> None:
    """Copy the strictly lower triangle onto the upper, tile by tile."""
    n = result.shape[0]
    block = min(_MIRROR_BLOCK, n) or 1
    full_il = np.tril_indices(block, -1)
    for i in range(0, n, block):
        j = min(i + block, n)
        for column in range(0, i, block):
            result[column : column + block, i:j] = result[
                i:j, column : column + block
            ].T
        diagonal = result[i:j, i:j]
        edge = j - i
        il = full_il if edge == block else np.tril_indices(edge, -1)
        diagonal.T[il] = diagonal[il]


def numpy_impl(
    x: np.ndarray, *, out: np.ndarray | None = None, beta: float = 0.0
) -> np.ndarray:
    """Compute ``x @ x.T + beta*out`` after input validation."""
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    n = x64.shape[0]
    if out is None:
        result = np.dot(x64, x64.T)
    else:
        result = out
        if beta == 0.0:
            np.dot(x64, x64.T, out=result)
        else:
            if beta != 1.0:
                result *= beta
            block = _row_block(n)
            for i in range(0, n, block):
                j = min(i + block, n)
                result[i:j, :j] += x64[i:j] @ x64[:j].T
    _mirror_lower_to_upper(result)
    return result


def numpy(
    x: np.ndarray, *, out: np.ndarray | None = None, beta: float = 0.0
) -> np.ndarray:
    """Validated NumPy implementation exposed for backend-specific tests."""
    validate(x, out, beta)
    return numpy_impl(x, out=out, beta=beta)


def run(
    backend: DsyrkBackend,
    x: np.ndarray,
    *,
    out: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    """Validate once and dispatch to the selected implementation."""
    validate(x, out, beta)
    return backend(x, out=out, beta=beta)
