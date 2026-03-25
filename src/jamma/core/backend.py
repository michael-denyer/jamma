"""Backend detection and information for JAMMA.

JAMMA uses a single compute backend:
- "numpy": Pure-NumPy pipeline with vendor BLAS/LAPACK dispatch.

Backend selection always returns "numpy".
"""

from __future__ import annotations

import os
from typing import Literal

from loguru import logger

BackendRequest = Literal["auto", "numpy", "numpy-streaming"]
BackendResolved = Literal["numpy"]


def detect_backend(requested: BackendRequest = "auto") -> BackendResolved:
    """Detect or validate the compute backend.

    Always returns "numpy". The JAMMA_BACKEND environment variable is
    checked for backward compatibility but only "auto", "numpy", and
    "numpy-streaming" are accepted.

    Args:
        requested: Requested backend — "auto" or "numpy".

    Returns:
        "numpy".

    Raises:
        ValueError: If `requested` (or JAMMA_BACKEND) is not a valid value.

    Example:
        >>> detect_backend("auto")
        'numpy'
        >>> detect_backend("numpy")
        'numpy'
    """
    env_override = os.environ.get("JAMMA_BACKEND")
    effective = env_override if env_override is not None else requested

    # Compound requests resolve to a base backend.
    _compound_map: dict[str, str] = {"numpy-streaming": "numpy"}
    if effective in _compound_map:
        effective = _compound_map[effective]

    valid = ("auto", "numpy")
    if effective not in valid:
        source = (
            " (from JAMMA_BACKEND environment variable)"
            if env_override is not None
            else ""
        )
        raise ValueError(
            f"Unknown backend {effective!r}{source}. "
            f"Must be one of {valid} (or 'numpy-streaming'). "
            "JAX backend was removed in v5.0 — use 'numpy' or 'auto'."
        )

    return "numpy"


def log_backend_selection(
    active: BackendResolved,
    requested: BackendRequest,
    env_override: str | None = None,
) -> None:
    """Log the selected backend at INFO level.

    Args:
        active: The backend that was selected ("numpy").
        requested: The originally requested backend
            ("auto", "numpy", or "numpy-streaming").
        env_override: Value of JAMMA_BACKEND if set, None otherwise.
            Caller passes this so the log reflects what actually drove selection.

    Example:
        >>> log_backend_selection("numpy", "auto")
        # logs: "Backend: numpy (auto-selected)"
        >>> log_backend_selection("numpy", "numpy")
        # logs: "Backend: numpy (explicitly requested)"
    """
    if env_override is not None:
        logger.info(f"Backend: {active} (from JAMMA_BACKEND={env_override})")
    elif requested != "auto":
        logger.info(f"Backend: {active} (explicitly requested)")
    else:
        logger.info(f"Backend: {active} (auto-selected)")


_BLAS_DISPLAY: dict[str, str] = {
    "mkl": "MKL",
    "openblas": "OpenBLAS",
    "accelerate": "Accelerate",
}


def format_pipeline_banner(
    runner: str,
    blas: str,
    eigen_driver: str,
    c_ext: bool,
    threads: int,
) -> str:
    """Build a single-line pipeline startup banner.

    Consolidates runner, BLAS backend, eigen driver, C extension status,
    and thread count into one authoritative log line.

    Args:
        runner: Runner name (e.g. "numpy-batch", "numpy-streaming").
        blas: BLAS backend identifier (e.g. "mkl", "openblas",
            "accelerate").
        eigen_driver: Eigen driver name (e.g. "DSYEVD", "DSYEVR").
        c_ext: Whether the C extension is usable.
        threads: BLAS/OpenMP thread count.

    Returns:
        Formatted banner string.

    Example:
        >>> format_pipeline_banner("numpy-batch", "mkl", "DSYEVD", True, 48)
        'Pipeline: numpy-batch | MKL | DSYEVD | C-ext (48 threads)'
    """
    blas_display = _BLAS_DISPLAY.get(blas, blas.title())
    c_ext_str = "C-ext" if c_ext else "no C-ext"
    return (
        f"Pipeline: {runner} | {blas_display} | {eigen_driver}"
        f" | {c_ext_str} ({threads} threads)"
    )


def get_backend_info() -> dict[str, str | bool]:
    """Get information about the compute backend.

    Returns:
        Dictionary with backend info:
        - selected: Backend name ("numpy")
    """
    return {
        "selected": detect_backend(),
    }
