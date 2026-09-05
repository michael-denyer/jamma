"""LMM dispatch-path selection.

Pure derivation of which C kernel path the NumPy LMM runner should take,
based on n_cvt, lmm_mode, and which optional C extension symbols are
present at import time.
"""

from __future__ import annotations

from enum import Enum

from loguru import logger

from jamma.lmm import accel
from jamma.lmm.schema import LmmMode


class DispatchPath(Enum):
    """The one authoritative C-kernel dispatch decision for an LMM run.

    Derived once by ``select_dispatch_path`` from ``(n_cvt, lmm_mode, accel)``
    and consulted per chunk. Exactly one member is active, so the
    contradictory flag combinations a multi-boolean form admits are
    unrepresentable and need no runtime guard. Every C path resolves the mode it runs
    from ``lmm_mode`` at workspace creation, in ``chunk_kernel.py``.
    """

    NUMPY_FALLBACK = "numpy_fallback"  # not split: pure-NumPy full-Uab path
    FUSED = "fused"  # n_cvt==1 fused Uab, any lmm_mode
    FUSED_GENERAL = "fused_general"  # n_cvt>=2 fused Uab, any lmm_mode

    @property
    def use_split(self) -> bool:
        """False only for the NumPy fallback, which takes the full-Uab path."""
        return self is not DispatchPath.NUMPY_FALLBACK

    @property
    def needs_null_w(self) -> bool:
        """True when the run needs the null-model ``w = UtW[:, 0]`` vector.

        The fused Wald/mode-4 workspace packs it in at construction; the fused
        Score/LRT kernels take it per call. Both consumers read the same vector,
        so the chunk runner materialises it once for either.
        """
        return self is DispatchPath.FUSED

    @property
    def feeds_raw_utg(self) -> bool:
        """True when chunk preparation hands raw ``utg_t`` straight to the kernel.

        Every C path (the fused family and the workspace Score/LRT variants)
        consumes ``utg_t`` directly. Only the negation, the NumPy fallback,
        builds a full Uab batch instead.
        """
        return self is not DispatchPath.NUMPY_FALLBACK


def select_dispatch_path(
    n_cvt: int,
    lmm_mode: LmmMode,
    *,
    accel: bool,
    log_choices: bool = True,
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
        log_choices: If True, emit debug logs describing the chosen path. Off
            in unit tests to keep output clean.

    Returns:
        The single active ``DispatchPath`` for this run.
    """
    path = _resolve_dispatch_path(n_cvt, lmm_mode, accel)
    if log_choices:
        _log_dispatch_choice(path, n_cvt, lmm_mode)
    return path


def select_current(
    n_cvt: int,
    lmm_mode: LmmMode,
    *,
    log_choices: bool = True,
) -> DispatchPath:
    """Select the dispatch path for the currently loaded extension.

    ``accel.available()`` is read at call time, not import time, so a test
    that drops the extension (directly, or through the ``no_c_kernels``
    fixture) drives the fallback for real rather than describing it.
    """
    return select_dispatch_path(
        n_cvt, lmm_mode, accel=accel.available(), log_choices=log_choices
    )


def _resolve_dispatch_path(n_cvt: int, lmm_mode: LmmMode, accel: bool) -> DispatchPath:
    """Map ``(n_cvt, lmm_mode, accel)`` to a path. Pure, no logging."""
    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    if not accel:
        return DispatchPath.NUMPY_FALLBACK

    # n_cvt > MAX_C_N_CVT resolves to a C path here and is rejected by the
    # kernel itself, which raises "n_cvt must be 1..100". There used to be a
    # Python guard further down in compute_numpy that fell back instead, but it
    # sat inside _compute_wald_numpy, which the runner reaches only when the
    # extension is absent, so it never ran.

    if n_cvt >= 2:
        return DispatchPath.FUSED_GENERAL

    return DispatchPath.FUSED


_PATH_LOG_MESSAGES = {
    DispatchPath.FUSED: (
        "Fused Uab path active: utg_t passed directly to C workspace "
        "(eliminates uab_varying_soa buffer)"
    ),
    DispatchPath.FUSED_GENERAL: (
        "Fused general Uab path active: utg_t passed directly to C workspace"
    ),
}


def _log_dispatch_choice(path: DispatchPath, n_cvt: int, lmm_mode: LmmMode) -> None:
    """Debug-log the chosen path. Pure side-effect."""
    message = _PATH_LOG_MESSAGES.get(path)
    if message is not None:
        logger.debug(f"{message} (n_cvt={n_cvt}, mode={lmm_mode})")

    if lmm_mode != 4:
        return

    if path is DispatchPath.FUSED_GENERAL:
        logger.debug("Mode-4 dispatch: fused general Uab kernel (single pass)")
    elif path is DispatchPath.FUSED:
        logger.debug("Mode-4 dispatch: fused Uab kernel (single pass)")
