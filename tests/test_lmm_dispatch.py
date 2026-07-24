"""Unit tests for jamma.lmm.dispatch.select_dispatch_path.

Pure-function tests over the dispatch matrix. Each named test pins one
human-meaningful cell (WS preempts stateless, fused-general needs n_cvt>=2, the
split mode-4 fallback). ``test_cross_product_invariants`` then sweeps the full
input space and asserts the structural gating every ``DispatchPath`` must obey,
so a regression in the priority resolution surfaces here, not as a slow
end-to-end runner failure.
"""

from __future__ import annotations

from itertools import product

import pytest

from jamma.lmm.dispatch import DispatchPath, KernelCaps, select_dispatch_path

_ALL_AVAILABLE = dict.fromkeys(KernelCaps._fields, True)
_NONE_AVAILABLE = dict.fromkeys(_ALL_AVAILABLE, False)
_FLAG_NAMES = tuple(_ALL_AVAILABLE)

_FEEDS_RAW_UTG = {
    DispatchPath.FUSED,
    DispatchPath.FUSED_GENERAL,
    DispatchPath.FUSED_SCORE,
    DispatchPath.FUSED_SCORE_WS,
    DispatchPath.FUSED_LRT,
    DispatchPath.FUSED_LRT_WS,
}


def _select(n_cvt: int, lmm_mode: int, **overrides) -> DispatchPath:
    caps = KernelCaps(**{**_ALL_AVAILABLE, **overrides})
    return select_dispatch_path(n_cvt, lmm_mode, caps, log_choices=False)


@pytest.mark.tier0
def test_no_c_kernels_is_numpy_fallback():
    assert _select(1, 1, **_NONE_AVAILABLE) is DispatchPath.NUMPY_FALLBACK


@pytest.mark.tier0
def test_split_requires_basic_kernel_for_n_cvt_1():
    assert not _select(1, 1, split=False, general=True).use_split
    assert _select(1, 1, split=True, general=False).use_split


@pytest.mark.tier0
def test_split_requires_general_kernel_for_n_cvt_2_plus():
    assert not _select(3, 1, split=True, general=False).use_split
    assert _select(3, 1, split=False, general=True).use_split


@pytest.mark.tier0
def test_fused_general_implies_n_cvt_ge_2():
    assert _select(1, 1) is DispatchPath.FUSED
    assert _select(3, 1) is DispatchPath.FUSED_GENERAL


@pytest.mark.tier0
def test_ws_path_preempts_stateless_for_score_and_lrt():
    """When the WS variant is available, the stateless variant must not win."""
    assert _select(1, 3) is DispatchPath.FUSED_SCORE_WS
    assert _select(1, 2) is DispatchPath.FUSED_LRT_WS


@pytest.mark.tier0
def test_stateless_score_lrt_used_only_when_ws_unavailable():
    assert _select(1, 3, score_fused_ws=False) is DispatchPath.FUSED_SCORE
    assert _select(1, 2, lrt_fused_ws=False) is DispatchPath.FUSED_LRT


@pytest.mark.tier0
def test_score_lrt_fused_only_for_n_cvt_1():
    """The fused Score/LRT paths are n_cvt=1 fast paths; higher n_cvt falls to split."""
    assert _select(2, 3) is DispatchPath.SOA_SPLIT
    assert _select(2, 2) is DispatchPath.SOA_SPLIT


@pytest.mark.tier0
def test_soa_split_mode4_only_when_fused_unavailable():
    """The split mode-4 single-pass path (n_cvt=1, mode 4, mode-4 kernel) is
    reached only when the fused path is unavailable; otherwise fused wins."""
    assert _select(1, 4) is DispatchPath.FUSED
    assert _select(1, 4, fused=False) is DispatchPath.SOA_SPLIT_MODE4
    assert _select(1, 4, fused=False, mode4=False) is DispatchPath.SOA_SPLIT
    assert (
        _select(2, 4, fused=False, mode4_fused_general=False) is DispatchPath.SOA_SPLIT
    )


@pytest.mark.tier0
def test_fused_for_mode_2_or_3_at_high_n_cvt_is_disabled():
    """The general Uab fused path is wired for modes 1 and 4 only. Modes 2/3
    should not take a fused path at n_cvt>=2 since they don't use the workspace.
    """
    fused_family = (DispatchPath.FUSED, DispatchPath.FUSED_GENERAL)
    assert _select(3, 2) not in fused_family
    assert _select(3, 3) not in fused_family
    assert _select(3, 1) is DispatchPath.FUSED_GENERAL
    assert _select(3, 4) is DispatchPath.FUSED_GENERAL


@pytest.mark.tier0
def test_cross_product_invariants():
    """Sweep every (n_cvt, lmm_mode, availability) and assert the structural
    gating each DispatchPath must obey. These invariants are stated from the
    domain, not read off the selector's internal booleans, so a wrong priority
    resolution or a wrongly gated member fails here.
    """
    for n_cvt, lmm_mode, bits in product(
        (1, 2, 5), (1, 2, 3, 4), product((False, True), repeat=len(_FLAG_NAMES))
    ):
        flags = dict(zip(_FLAG_NAMES, bits, strict=True))
        path = select_dispatch_path(
            n_cvt, lmm_mode, KernelCaps(**flags), log_choices=False
        )

        # Property consistency.
        assert path.use_split == (path is not DispatchPath.NUMPY_FALLBACK)
        assert path.use_fused_general == (path is DispatchPath.FUSED_GENERAL)
        assert path.feeds_raw_utg == (path in _FEEDS_RAW_UTG)
        assert path.uses_fused_score_or_lrt == (
            path
            in (
                DispatchPath.FUSED_SCORE,
                DispatchPath.FUSED_SCORE_WS,
                DispatchPath.FUSED_LRT,
                DispatchPath.FUSED_LRT_WS,
            )
        )

        # Mode gating: fused Uab and split mode-4 are Wald/All; score is mode 3;
        # lrt is mode 2.
        if path in (DispatchPath.FUSED, DispatchPath.FUSED_GENERAL):
            assert lmm_mode in (1, 4)
        if path is DispatchPath.SOA_SPLIT_MODE4:
            assert lmm_mode == 4
            assert n_cvt == 1
        if path in (DispatchPath.FUSED_SCORE, DispatchPath.FUSED_SCORE_WS):
            assert lmm_mode == 3
            assert n_cvt == 1
        if path in (DispatchPath.FUSED_LRT, DispatchPath.FUSED_LRT_WS):
            assert lmm_mode == 2
            assert n_cvt == 1

        # n_cvt gating: general fused is multi-covariate; bare fused is n_cvt==1.
        if path is DispatchPath.FUSED_GENERAL:
            assert n_cvt >= 2
        if path is DispatchPath.FUSED:
            assert n_cvt == 1

        # WS preempts stateless: a stateless fused Score/LRT is chosen only when
        # its workspace kernel is unavailable.
        if path is DispatchPath.FUSED_SCORE:
            assert not flags["score_fused_ws"]
        if path is DispatchPath.FUSED_LRT:
            assert not flags["lrt_fused_ws"]

        # Any non-fallback path needs a split kernel present.
        if path is not DispatchPath.NUMPY_FALLBACK:
            assert flags["split"] or flags["general"]
