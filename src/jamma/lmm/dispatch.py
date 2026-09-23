"""LMM dispatch-path selection."""

from __future__ import annotations

from enum import Enum
from typing import assert_never

from loguru import logger

from jamma.core.constants import n_index


class DispatchPath(Enum):
    """The one authoritative C-kernel dispatch decision for an LMM run.

    Derived once by ``select_dispatch_path`` from ``accel``
    and consulted per chunk. Exactly one member is active, so the
    contradictory flag combinations a multi-boolean form admits are
    unrepresentable and need no runtime guard. Every C path resolves the mode it runs
    from ``lmm_mode`` at workspace creation, in ``chunk_kernel.py``.
    """

    NUMPY_FALLBACK = "numpy_fallback"
    FUSED = "fused"  # C workspace, fused Uab, any n_cvt and lmm_mode

    @property
    def is_native(self) -> bool:
        """True for the C workspace path, which pipelines, owns a C workspace,
        and consumes raw ``utg_t``.
        """
        return self is DispatchPath.FUSED

    def varying_rows(self, n_cvt: int) -> int:
        """Rows of ``n_samples`` float64 one SNP materialises beyond ``utg_t``."""
        match self:
            case DispatchPath.FUSED:
                return 0
            case DispatchPath.NUMPY_FALLBACK:
                return n_index(n_cvt)
            case _:
                assert_never(self)

    def iab_cells(self, n_cvt: int) -> int:
        """Per-SNP Iab float64 cells held alongside the varying rows."""
        match self:
            case DispatchPath.FUSED:
                return 0
            case DispatchPath.NUMPY_FALLBACK:
                return (n_cvt + 2) * n_index(n_cvt)
            case _:
                assert_never(self)

    def invariant_rows(self, n_cvt: int) -> int:
        """Rows of ``n_samples`` the run holds once for the invariant Uab columns."""
        match self:
            case DispatchPath.FUSED:
                return n_index(n_cvt) - (n_cvt + 2)
            case DispatchPath.NUMPY_FALLBACK:
                return 0
            case _:
                assert_never(self)


def select_dispatch_path(*, accel: bool) -> DispatchPath:
    """Derive the single active kernel path for this run.

    Args:
        accel: Whether the C extension is loaded. One bit, because the
            ABI-equality gate admits all of ``methods[]`` or none of it.

    Returns:
        The single active ``DispatchPath`` for this run.
    """
    path = DispatchPath.FUSED if accel else DispatchPath.NUMPY_FALLBACK
    message = _PATH_LOG_MESSAGES.get(path)
    if message is not None:
        logger.debug(message)
    return path


_PATH_LOG_MESSAGES = {
    DispatchPath.FUSED: "Fused Uab path active: utg_t passed directly to C workspace",
}
