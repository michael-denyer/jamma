"""DSYRK contract, NumPy implementation, and scratch accounting."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DsyrkBackend(Protocol):
    """Compute step behind ``dsyrk``, native or NumPy, after validation."""

    def __call__(
        self, x: np.ndarray, *, out: np.ndarray | None = None, beta: float = 0.0
    ) -> np.ndarray: ...


_BLOCK_BYTES = 64 << 20  # Absolute cap on the fallback's scratch buffer
_BLOCK_FRACTION = 8  # Scratch is also held to 1/8 of the output
_MIRROR_BLOCK = 512  # Mirror tile edge; its scratch is O(edge^2), not O(n)

# What one mirror tile holds: tril_indices' two int64 index arrays plus the
# gathered float64 values (each edge*(edge-1)/2 long), and one edge-by-edge
# transposed tile. Tiling both mirror axes keeps this independent of n, so
# scratch_bytes can bound the fallback with a constant rather than a term
# that grows with the matrix.
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
    """Rows per accumulation block.

    Held to a fraction of ``n`` as well as an absolute byte cap: the scratch is
    block-by-n, so a byte cap alone leaves it comparable to the whole output at
    modest ``n``.
    """
    by_bytes = _BLOCK_BYTES // max(1, n * 8)
    by_fraction = max(1, n // _BLOCK_FRACTION)
    return max(1, min(n, by_fraction, by_bytes))


def scratch_bytes(n: int, backend: DsyrkBackend) -> int:
    """Upper bound on what one ``dsyrk`` call holds beyond its n-by-n output.

    Zero on the native backend, which accumulates in place. The NumPy fallback
    holds one block-by-n float64 product during accumulation and a fixed set of
    tiles during the mirror, whichever is larger. A memory pre-flight budgeting
    only the accumulator under-counts a fallback run, so ``jamma.core.memory``
    adds this to the kinship phase peak.
    """
    if backend is not numpy_impl:
        return 0
    return max(_row_block(n) * n * 8, _MIRROR_SCRATCH_BYTES)


def _mirror_lower_to_upper(result: np.ndarray) -> None:
    """Copy the strictly lower triangle onto the upper, tile by tile.

    ``np.tril_indices_from`` allocates two n^2/2 index arrays plus an n^2/2
    value gather, together more than the matrix being symmetrised. Tiling both
    axes caps every temporary at the tile edge, so the scratch is constant in
    ``n`` and ``_MIRROR_SCRATCH_BYTES`` can state it exactly.
    """
    n = result.shape[0]
    block = min(_MIRROR_BLOCK, n) or 1
    full_il = np.tril_indices(block, -1)
    for i in range(0, n, block):
        j = min(i + block, n)
        for column in range(0, i, block):
            # Rows i:j, columns column:column+block lie entirely below the
            # diagonal. Every such tile is full width: i is a multiple of
            # block, so the partial tile can only ever be the diagonal one
            # handled below.
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
    """Unchecked NumPy implementation of ``C = x @ x.T + beta*C``.

    The accumulating path walks row blocks and touches only the lower triangle,
    which the closing mirror copies up. ``result += x @ x.T`` would materialise
    a second full n-by-n product, so the fallback's peak was several times the
    output it writes, memory the kinship pre-flight does not budget for.
    ``scratch_bytes`` reports what one block costs so it can.
    """
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
