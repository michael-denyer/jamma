"""LMM dispatch-path selection.

Pure derivation of which C kernel path the NumPy LMM runner should take,
based on n_cvt, lmm_mode, and which optional C extension symbols are
present at import time.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from jamma.lmm.schema import LmmMode


@dataclass(frozen=True)
class LmmDispatch:
    """Per-run C kernel dispatch flags.

    Each ``use_*`` flag answers "should this chunk loop take the C kernel
    fast path for this feature?" — derived once before the chunk loop and
    consulted per chunk. Flags are not independent: the docstrings on
    ``select_dispatch_path`` document the precedence rules.
    """

    use_split: bool
    use_fused: bool
    use_fused_general: bool
    use_fused_mode4: bool
    use_fused_score: bool
    use_fused_score_ws: bool
    use_fused_lrt: bool
    use_fused_lrt_ws: bool

    def __post_init__(self) -> None:
        """Enforce the flag implications ``select_dispatch_path`` guarantees.

        The sanctioned producer never emits an impossible combination, but a
        hand-constructed instance could; the per-chunk dispatch then trusts these
        flags, so reject illegal states at construction rather than dispatch
        incorrectly.
        """
        if self.use_fused and not self.use_split:
            raise ValueError("use_fused requires use_split")
        if self.use_fused_general and not self.use_fused:
            raise ValueError("use_fused_general requires use_fused")
        if self.use_fused_mode4 and not self.use_split:
            raise ValueError("use_fused_mode4 requires use_split")
        if (self.use_fused_score or self.use_fused_score_ws) and not self.use_split:
            raise ValueError("fused Score paths require use_split")
        if (self.use_fused_lrt or self.use_fused_lrt_ws) and not self.use_split:
            raise ValueError("fused LRT paths require use_split")
        if self.use_fused_score and self.use_fused_score_ws:
            raise ValueError("use_fused_score excludes use_fused_score_ws")
        if self.use_fused_lrt and self.use_fused_lrt_ws:
            raise ValueError("use_fused_lrt excludes use_fused_lrt_ws")

    @property
    def uses_fused_score_or_lrt(self) -> bool:
        """True when a fused Score/LRT path (mode 2/3) is active.

        These paths take the null-model ``w`` vector as a separate argument
        rather than packing it into a Wald workspace, so the chunk runner
        materializes ``w = UtW[:, 0]`` only for this family.
        """
        return (
            self.use_fused_score
            or self.use_fused_score_ws
            or self.use_fused_lrt
            or self.use_fused_lrt_ws
        )

    @property
    def feeds_raw_utg(self) -> bool:
        """True when chunk preparation hands raw ``utg_t`` straight to the kernel.

        The whole fused family (Wald/mode-4 fused plus the fused Score/LRT
        variants) consumes ``utg_t`` directly, so no ``uab_varying_soa`` buffer
        is built. The negation selects the SoA-split path, which does build it.
        """
        return self.use_fused or self.uses_fused_score_or_lrt


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
) -> LmmDispatch:
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
        Populated LmmDispatch with all 8 ``use_*`` flags.
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

    return LmmDispatch(
        use_split=use_split,
        use_fused=use_fused,
        use_fused_general=use_fused_general,
        use_fused_mode4=use_fused_mode4,
        use_fused_score=use_fused_score,
        use_fused_score_ws=use_fused_score_ws,
        use_fused_lrt=use_fused_lrt,
        use_fused_lrt_ws=use_fused_lrt_ws,
    )


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
