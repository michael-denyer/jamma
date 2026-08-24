"""LMM dispatch-path selection.

Pure derivation of which C kernel path the NumPy LMM runner should take,
based on n_cvt, lmm_mode, and which optional C extension symbols are
present at import time.
"""

from __future__ import annotations

from enum import Enum

from loguru import logger

from jamma.lmm.schema import LmmMode


class DispatchPath(Enum):
    """The one authoritative C-kernel dispatch decision for an LMM run.

    Derived once by ``select_dispatch_path`` from ``(n_cvt, lmm_mode, accel)``
    and consulted per chunk. Exactly one member is active, so the
    contradictory flag combinations a multi-boolean form admits are
    unrepresentable and need no runtime guard. Wald-vs-mode-4 for the FUSED and
    SOA_SPLIT families is resolved from ``lmm_mode`` at the call site;
    """

    NUMPY_FALLBACK = "numpy_fallback"  # not split: pure-NumPy full-Uab path
    FUSED = "fused"  # n_cvt==1 fused Uab (Wald/mode-4 by lmm_mode)
    FUSED_GENERAL = "fused_general"  # n_cvt>=2 fused Uab (Wald/mode-4 by lmm_mode)
    FUSED_SCORE_WS = "fused_score_ws"  # mode 3 workspace-based
    FUSED_LRT_WS = "fused_lrt_ws"  # mode 2 workspace-based
    SOA_SPLIT = "soa_split"  # SoA split; Wald+compose / score / lrt by lmm_mode

    @property
    def use_split(self) -> bool:
        """False only for the NumPy fallback, which takes the full-Uab path."""
        return self is not DispatchPath.NUMPY_FALLBACK

    @property
    def use_fused_general(self) -> bool:
        """True for the n_cvt>=2 fused path, which sizes chunks differently."""
        return self is DispatchPath.FUSED_GENERAL

    @property
    def uses_fused_score_or_lrt(self) -> bool:
        """True when a fused Score/LRT path (mode 2/3) is active.

        These paths take the null-model ``w`` vector as a separate argument
        rather than packing it into a Wald workspace, so the chunk runner
        materializes ``w = UtW[:, 0]`` only for this family.
        """
        return self in _SCORE_LRT_FAMILY

    @property
    def needs_null_w(self) -> bool:
        """True when the run needs the null-model ``w = UtW[:, 0]`` vector.

        The fused Wald/mode-4 workspace packs it in at construction; the fused
        Score/LRT kernels take it per call. Both consumers read the same vector,
        so the chunk runner materialises it once for either.
        """
        return self is DispatchPath.FUSED or self in _SCORE_LRT_FAMILY

    @property
    def feeds_raw_utg(self) -> bool:
        """True when chunk preparation hands raw ``utg_t`` straight to the kernel.

        The whole fused family (Wald/mode-4 fused plus the workspace Score/LRT
        variants) consumes ``utg_t`` directly, so no ``uab_varying_soa`` buffer
        is built. The negation selects the SoA-split path, which does build it.
        """
        return self in _FUSED_FAMILY or self in _SCORE_LRT_FAMILY


_FUSED_FAMILY = frozenset({DispatchPath.FUSED, DispatchPath.FUSED_GENERAL})
_SCORE_LRT_FAMILY = frozenset({DispatchPath.FUSED_SCORE_WS, DispatchPath.FUSED_LRT_WS})


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
        # The general fused workspace is wired for Wald and mode 4 only; modes
        # 2 and 3 have no general fused kernel and use the split path.
        if lmm_mode in (1, 4):
            return DispatchPath.FUSED_GENERAL
        return DispatchPath.SOA_SPLIT

    if lmm_mode in (1, 4):
        return DispatchPath.FUSED
    return DispatchPath.FUSED_SCORE_WS if lmm_mode == 3 else DispatchPath.FUSED_LRT_WS


_PATH_LOG_MESSAGES = {
    DispatchPath.FUSED: (
        "Fused Uab path active: utg_t passed directly to C workspace "
        "(eliminates uab_varying_soa buffer)"
    ),
    DispatchPath.FUSED_GENERAL: (
        "Fused general Uab path active: utg_t passed directly to C workspace"
    ),
    DispatchPath.FUSED_SCORE_WS: (
        "Fused Score workspace path active: workspace created once, "
        "utg_t passed per-chunk (eliminates per-chunk malloc)"
    ),
    DispatchPath.FUSED_LRT_WS: (
        "Fused LRT workspace path active: workspace created once, "
        "utg_t passed per-chunk (eliminates per-chunk malloc/grid precompute)"
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
    elif path is DispatchPath.SOA_SPLIT:
        logger.debug("Mode-4 dispatch: compose fallback (no general fused kernel)")
