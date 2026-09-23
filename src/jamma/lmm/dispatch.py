"""LMM dispatch-path selection.

Pure derivation of which C kernel path the NumPy LMM runner should take,
based on n_cvt, lmm_mode, and which optional C extension symbols are
present at import time.
"""

from __future__ import annotations

from enum import Enum
from typing import assert_never

from loguru import logger

from jamma.core.constants import n_index
from jamma.lmm.schema import LmmMode, LmmTest, get_spec


class DispatchPath(Enum):
    """The one authoritative C-kernel dispatch decision for an LMM run.

    Derived once by ``select_dispatch_path`` from ``(n_cvt, lmm_mode, accel)``
    and consulted per chunk. Exactly one member is active, so the
    contradictory flag combinations a multi-boolean form admits are
    unrepresentable and need no runtime guard. Every C path resolves the mode it runs
    from ``lmm_mode`` at workspace creation, in ``chunk_kernel.py``.
    """

    NUMPY_FALLBACK = "numpy_fallback"  # not split: pure-NumPy full-Uab path
    NUMPY_WALD = "numpy_wald"  # intercept-only Wald, split products
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
            case DispatchPath.NUMPY_WALD:
                return 3
            case DispatchPath.NUMPY_FALLBACK:
                return n_index(n_cvt)
            case _:
                assert_never(self)

    def iab_cells(self, n_cvt: int) -> int:
        """Per-SNP Iab float64 cells held alongside the varying rows."""
        match self:
            case DispatchPath.FUSED | DispatchPath.NUMPY_WALD:
                return 0
            case DispatchPath.NUMPY_FALLBACK:
                return (n_cvt + 2) * n_index(n_cvt)
            case _:
                assert_never(self)

    def invariant_rows(self, n_cvt: int) -> int:
        """Rows of ``n_samples`` the run holds once for the invariant Uab columns."""
        match self:
            case DispatchPath.FUSED | DispatchPath.NUMPY_WALD:
                return n_index(n_cvt) - (n_cvt + 2)
            case DispatchPath.NUMPY_FALLBACK:
                return 0
            case _:
                assert_never(self)


def select_dispatch_path(
    n_cvt: int,
    lmm_mode: LmmMode,
    *,
    accel: bool,
) -> DispatchPath:
    """Derive the single active C kernel path for this run.

    Resolved directly: each branch returns the path it selects rather than
    setting a flag for a later ladder to re-interpret. Reading top to bottom
    gives the whole decision, and the priorities (fused beats the split mode-4
    kernel; a workspace Score/LRT variant beats its stateless twin) are the
    order of the returns.

    Args:
        n_cvt: Number of covariates (intercept counts as 1).
        lmm_mode: 1=Wald, 2=LRT, 3=Score, 4=All.
        accel: Whether the C extension is loaded. One bit, because the
            ABI-equality gate admits all of ``methods[]`` or none of it.

    Returns:
        The single active ``DispatchPath`` for this run.
    """
    path = _resolve_dispatch_path(n_cvt, lmm_mode, accel)
    message = _PATH_LOG_MESSAGES.get(path)
    if message is not None:
        logger.debug(f"{message} (n_cvt={n_cvt}, mode={lmm_mode})")
    return path


def _resolve_dispatch_path(n_cvt: int, lmm_mode: LmmMode, accel: bool) -> DispatchPath:
    """Map ``(n_cvt, lmm_mode, accel)`` to a path. Pure, no logging."""
    wald_only = get_spec(lmm_mode).tests == LmmTest.WALD
    if not accel:
        return (
            DispatchPath.NUMPY_WALD
            if n_cvt == 1 and wald_only
            else DispatchPath.NUMPY_FALLBACK
        )

    return DispatchPath.FUSED


_PATH_LOG_MESSAGES = {
    DispatchPath.FUSED: "Fused Uab path active: utg_t passed directly to C workspace",
}
