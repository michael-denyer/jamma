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

    Derived once by ``select_dispatch_path`` from ``(n_cvt, lmm_mode, C-kernel
    availability)`` and consulted per chunk. Exactly one member is active, so the
    contradictory flag combinations the old 8-boolean form admitted are
    unrepresentable and need no runtime guard. Wald-vs-mode-4 for the FUSED and
    SOA_SPLIT families is resolved from ``lmm_mode`` at the call site;
    ``SOA_SPLIT_MODE4`` is the one mode-4 kernel not recoverable from ``lmm_mode``
    (a distinct split entrypoint and workspace), so it is its own member. The
    properties reproduce the old ``LmmDispatch`` fields/properties by the same
    names, so consumer read sites are unchanged.
    """

    NUMPY_FALLBACK = "numpy_fallback"  # not split: pure-NumPy full-Uab path
    FUSED = "fused"  # n_cvt==1 fused Uab (Wald/mode-4 by lmm_mode)
    FUSED_GENERAL = "fused_general"  # n_cvt>=2 fused Uab (Wald/mode-4 by lmm_mode)
    FUSED_SCORE = "fused_score"  # mode 3 stateless
    FUSED_SCORE_WS = "fused_score_ws"  # mode 3 workspace-based
    FUSED_LRT = "fused_lrt"  # mode 2 stateless
    FUSED_LRT_WS = "fused_lrt_ws"  # mode 2 workspace-based
    SOA_SPLIT = "soa_split"  # SoA split; Wald+compose / score / lrt by lmm_mode
    SOA_SPLIT_MODE4 = "soa_split_mode4"  # SoA split single-pass mode-4 kernel

    @property
    def use_split(self) -> bool:
        """False only for the NumPy fallback (was the ``use_split`` flag)."""
        return self is not DispatchPath.NUMPY_FALLBACK

    @property
    def use_fused(self) -> bool:
        """True for the fused-Uab Wald family (was the ``use_fused`` flag)."""
        return self in _FUSED_FAMILY

    @property
    def use_fused_general(self) -> bool:
        """True for the n_cvt>=2 fused path (was the ``use_fused_general`` flag)."""
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
    def feeds_raw_utg(self) -> bool:
        """True when chunk preparation hands raw ``utg_t`` straight to the kernel.

        The whole fused family (Wald/mode-4 fused plus the fused Score/LRT
        variants) consumes ``utg_t`` directly, so no ``uab_varying_soa`` buffer
        is built. The negation selects the SoA-split path, which does build it.
        """
        return self in _FUSED_FAMILY or self in _SCORE_LRT_FAMILY


_FUSED_FAMILY = frozenset({DispatchPath.FUSED, DispatchPath.FUSED_GENERAL})
_SCORE_LRT_FAMILY = frozenset(
    {
        DispatchPath.FUSED_SCORE,
        DispatchPath.FUSED_SCORE_WS,
        DispatchPath.FUSED_LRT,
        DispatchPath.FUSED_LRT_WS,
    }
)


def select_dispatch_path(
    n_cvt: int,
    lmm_mode: LmmMode,
    *,
    c_split_available: bool,
    c_general_available: bool,
    c_fused_available: bool,
    c_fused_general_available: bool,
    c_mode4_available: bool,
    c_mode4_fused_available: bool,
    c_mode4_fused_general_available: bool,
    c_score_fused_available: bool,
    c_score_fused_ws_available: bool,
    c_lrt_fused_available: bool,
    c_lrt_fused_ws_available: bool,
    log_choices: bool = True,
) -> DispatchPath:
    """Derive the C kernel dispatch flags for this run.

    The branching matrix:

    * ``use_split``: SoA-native C split dispatch. Available when n_cvt=1
      with the basic split kernel, or n_cvt>1 with the general split
      kernel.
    * ``use_fused_mode4``: single-pass Wald/Score/LRT from SoA data, no
      Uab reconstruction. n_cvt=1 + mode-4 + mode-4 kernel.
    * ``use_fused``: skip uab_varying_soa entirely; pass utg_t to the
      C workspace which computes wx/xx/xy on the fly. n_cvt=1 (ABI v8)
      or n_cvt>=2 (ABI v9 general). Modes 2/3 don't use workspace, so
      they don't get this path here — see the dedicated score/lrt
      flags below.
    * ``use_fused_general``: implies use_fused with n_cvt>=2.
    * ``use_fused_score_ws`` / ``use_fused_lrt_ws``: workspace-based
      Score/LRT (n_cvt=1 only). Created once, reused across chunks —
      eliminates per-chunk malloc + grid precompute.
    * ``use_fused_score`` / ``use_fused_lrt``: stateless fallback when
      the WS variant isn't available. Only one of (ws, stateless) per
      mode is True at a time.

    Args:
        n_cvt: Number of covariates (intercept counts as 1).
        lmm_mode: 1=Wald, 2=LRT, 3=Score, 4=All.
        c_*_available: Per-feature C kernel availability flags pulled
            from jamma.lmm.compute_numpy at import time.
        log_choices: If True, emit debug logs describing which paths
            were chosen. Off in unit tests to keep output clean.

    Returns:
        The single active ``DispatchPath`` for this run.
    """
    use_split = (c_split_available and n_cvt == 1) or (
        c_general_available and n_cvt > 1
    )

    use_fused_mode4 = use_split and lmm_mode == 4 and n_cvt == 1 and c_mode4_available

    use_fused = use_split and (
        (
            n_cvt == 1
            and c_fused_available
            and (lmm_mode == 1 or (lmm_mode == 4 and c_mode4_fused_available))
        )
        or (n_cvt >= 2 and c_fused_general_available and lmm_mode == 1)
        or (n_cvt >= 2 and c_mode4_fused_general_available and lmm_mode == 4)
    )
    use_fused_general = use_fused and n_cvt >= 2

    use_fused_score_ws = (
        use_split and n_cvt == 1 and lmm_mode == 3 and c_score_fused_ws_available
    )
    use_fused_lrt_ws = (
        use_split and n_cvt == 1 and lmm_mode == 2 and c_lrt_fused_ws_available
    )
    use_fused_score = (
        use_split
        and n_cvt == 1
        and lmm_mode == 3
        and c_score_fused_available
        and not use_fused_score_ws
    )
    use_fused_lrt = (
        use_split
        and n_cvt == 1
        and lmm_mode == 2
        and c_lrt_fused_available
        and not use_fused_lrt_ws
    )

    if log_choices:
        _log_dispatch_choices(
            n_cvt,
            lmm_mode,
            use_fused=use_fused,
            use_fused_general=use_fused_general,
            use_fused_mode4=use_fused_mode4,
            use_fused_score=use_fused_score,
            use_fused_score_ws=use_fused_score_ws,
            use_fused_lrt=use_fused_lrt,
            use_fused_lrt_ws=use_fused_lrt_ws,
            c_mode4_available=c_mode4_available,
        )

    # Resolve to the single active path in the consumer priority order (the
    # _dispatch_compute ladder in chunk_dispatch.py): fused-general, fused,
    # score-WS, lrt-WS, score, lrt, then SoA-split. use_fused wins over
    # use_fused_mode4 (matching that ladder), so the mode-4 split path is
    # reached only when use_fused is False.
    if not use_split:
        return DispatchPath.NUMPY_FALLBACK
    if use_fused:
        return DispatchPath.FUSED_GENERAL if use_fused_general else DispatchPath.FUSED
    if use_fused_score_ws:
        return DispatchPath.FUSED_SCORE_WS
    if use_fused_lrt_ws:
        return DispatchPath.FUSED_LRT_WS
    if use_fused_score:
        return DispatchPath.FUSED_SCORE
    if use_fused_lrt:
        return DispatchPath.FUSED_LRT
    if use_fused_mode4:
        return DispatchPath.SOA_SPLIT_MODE4
    return DispatchPath.SOA_SPLIT


def _log_dispatch_choices(
    n_cvt: int,
    lmm_mode: LmmMode,
    *,
    use_fused: bool,
    use_fused_general: bool,
    use_fused_mode4: bool,
    use_fused_score: bool,
    use_fused_score_ws: bool,
    use_fused_lrt: bool,
    use_fused_lrt_ws: bool,
    c_mode4_available: bool,
) -> None:
    """Debug-log which dispatch paths the runner picked. Pure side-effect."""
    if use_fused and not use_fused_general:
        logger.debug(
            "Fused Uab path active: utg_t passed directly to C workspace "
            f"(mode={lmm_mode}, eliminates uab_varying_soa buffer)"
        )
    elif use_fused_general:
        logger.debug(
            "Fused general Uab path active: utg_t passed directly to C workspace "
            f"(n_cvt={n_cvt}, n_var={n_cvt + 2})"
        )

    if lmm_mode == 4:
        if use_fused:
            variant = "fused general" if use_fused_general else "fused"
            logger.debug(
                f"Mode-4 dispatch: {variant} Uab kernel (Wald/Score/LRT single pass)"
            )
        elif use_fused_mode4:
            # Reached when mode-4 C kernel is available but its fused variant
            # is not (e.g. partial extension rebuild). Independent from the
            # use_fused branch above so the log faithfully reports the path.
            logger.debug("Mode-4 dispatch: fused kernel (Wald/Score/LRT single pass)")
        else:
            reason = (
                "fused general kernel unavailable"
                if n_cvt > 1
                else "fused kernel unavailable"
                if not c_mode4_available
                else "C split extension unavailable"
            )
            logger.debug(f"Mode-4 dispatch: compose fallback ({reason})")

    if use_fused_score_ws:
        logger.debug(
            "Fused Score workspace path active: workspace created once, "
            "utg_t passed per-chunk (eliminates per-chunk malloc)"
        )
    elif use_fused_score:
        logger.debug(
            "Fused Score path active: utg_t passed directly to C "
            "(eliminates uab_varying_soa buffer for mode 3)"
        )
    if use_fused_lrt_ws:
        logger.debug(
            "Fused LRT workspace path active: workspace created once, "
            "utg_t passed per-chunk (eliminates per-chunk malloc/grid precompute)"
        )
    elif use_fused_lrt:
        logger.debug(
            "Fused LRT path active: utg_t passed directly to C "
            "(eliminates uab_varying_soa buffer for mode 2)"
        )
