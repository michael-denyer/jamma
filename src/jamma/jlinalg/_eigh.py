"""Symmetric eigendecomposition dispatch contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np

Driver = Literal["auto", "dsyevd", "dsyevr"]


class EighStatus(NamedTuple):
    """Diagnostic outcome of one ``eigh`` call."""

    driver_used: Literal["dsyevd", "dsyevr", "none"]


EighBackend = Callable[
    [np.ndarray, bool, Driver], tuple[np.ndarray, np.ndarray, EighStatus]
]

_DRIVER_USED_NAMES: dict[int, Literal["dsyevd", "dsyevr", "none"]] = {
    0: "none",
    1: "dsyevd",
    2: "dsyevr",
}


def _check_square(k: np.ndarray) -> None:
    if k.ndim != 2:
        raise ValueError(f"eigh: K must be a 2-D array, got {k.ndim}-D")
    if k.shape[0] != k.shape[1]:
        raise ValueError(f"eigh: K must be square, got shape {k.shape}")


def numpy(
    k: np.ndarray,
    inplace: bool = False,
    driver: Driver = "auto",
) -> tuple[np.ndarray, np.ndarray, EighStatus]:
    """Eigendecompose with NumPy while matching the native consume contract.

    The single fallback shared by the C-present-but-no-vendor-LAPACK path and
    the no-C-extension path. Matches the vendor eigh contract: K is consumed
    (overwritten as scratch) whether or not ``inplace`` is set.

    Args:
        k: Symmetric matrix, shape (N, N). Consumed on exit.
        inplace: If True, return the eigenvectors in k's buffer. Requires a
            C-contiguous writeable float64 array.
        driver: Accepted for signature parity with the vendor backend. NumPy
            has no DSYEVD/DSYEVR choice, so this has no effect; the returned
            status always reports ``driver_used="none"``.

    Returns:
        Tuple of (eigenvalues ascending, eigenvectors, status).

    Raises:
        ValueError: If k is not 2-D square, or ``inplace`` is set on an array
            that is not C-contiguous writeable float64.
    """
    del driver
    _check_square(k)
    if inplace:
        if k.dtype != np.float64:
            raise ValueError(f"eigh: inplace=True requires float64, got {k.dtype}")
        if not k.flags["C_CONTIGUOUS"]:
            raise ValueError("eigh: inplace=True requires a C-contiguous array")
        if not k.flags["WRITEABLE"]:
            raise ValueError("eigh: inplace=True requires a writeable array")
    k64 = np.asarray(k, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(k64)
    status = EighStatus(driver_used="none")
    if inplace:
        k[:] = eigenvectors
        return eigenvalues, k, status
    if k.dtype == np.float64 and k.flags["WRITEABLE"]:
        # Vendor eigh consumes K as scratch; zero it so the fallback matches
        # that contract and no caller relies on K surviving the call.
        k[:] = 0.0
    return eigenvalues, eigenvectors, status


def native_wrapper(native_eigh: Callable[..., object]) -> EighBackend:
    """Adapt the extension's integer driver status to ``EighStatus``."""

    def call(
        k: np.ndarray, inplace: bool = False, driver: Driver = "auto"
    ) -> tuple[np.ndarray, np.ndarray, EighStatus]:
        result = native_eigh(k, inplace=inplace, driver=driver)
        eigenvalues, eigenvectors, driver_used = result  # type: ignore[misc]
        return (
            eigenvalues,
            eigenvectors,
            EighStatus(driver_used=_DRIVER_USED_NAMES[driver_used]),
        )

    return call


def run(
    backend: EighBackend,
    k: np.ndarray,
    inplace: bool,
    driver: Driver,
    *,
    force_numpy: bool,
) -> tuple[np.ndarray, np.ndarray, EighStatus]:
    """Validate the driver and dispatch to the selected implementation."""
    if driver not in ("auto", "dsyevd", "dsyevr"):
        raise ValueError(
            f"eigh: driver must be 'auto', 'dsyevd', or 'dsyevr', got {driver!r}"
        )
    if force_numpy:
        return numpy(k, inplace=inplace, driver=driver)
    return backend(k, inplace, driver)
