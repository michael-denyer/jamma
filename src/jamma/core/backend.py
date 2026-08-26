"""Backend detection and information for JAMMA.

JAMMA uses a single compute backend:
- "numpy": Pure-NumPy pipeline with vendor BLAS/LAPACK dispatch.

Backend selection always returns "numpy".
"""

from __future__ import annotations

from loguru import logger


def log_backend_selection(
    active: str,
    requested: str,
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
    jlinalg_backend: str | None = None,
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
        jlinalg_backend: jlinalg's ``blas_backend`` (e.g. "MKL-ILP64",
            "numpy-fallback"). Omitted from the banner when None, since
            jlinalg can report "numpy-fallback" even with its C extension
            loaded (``JLINALG_NO_VENDOR_DGEMM``), a state the ``c_ext`` flag
            alone cannot show.

    Returns:
        Formatted banner string.

    Example:
        >>> format_pipeline_banner("numpy-batch", "mkl", "DSYEVD", True, 48)
        'Pipeline: numpy-batch | MKL | DSYEVD | C-ext (48 threads)'
        >>> format_pipeline_banner(
        ...     "numpy-batch", "mkl", "DSYEVD", True, 48, jlinalg_backend="MKL-ILP64"
        ... )
        'Pipeline: numpy-batch | MKL | DSYEVD | C-ext (48 threads) | jlinalg: MKL-ILP64'
    """
    blas_display = _BLAS_DISPLAY.get(blas, blas.title())
    c_ext_str = "C-ext" if c_ext else "no C-ext"
    banner = (
        f"Pipeline: {runner} | {blas_display} | {eigen_driver}"
        f" | {c_ext_str} ({threads} threads)"
    )
    if jlinalg_backend is not None:
        banner += f" | jlinalg: {jlinalg_backend}"
    return banner


def get_backend_info() -> dict[str, str | bool]:
    """Get information about the compute backend."""
    return {"selected": "numpy"}
