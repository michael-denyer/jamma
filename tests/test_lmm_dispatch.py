"""Unit tests for jamma.lmm.dispatch.select_dispatch_path.

Pure-function tests covering the dispatch matrix that used to live
inline in run_lmm_association_numpy. Each test pins one cell of the
matrix so a regression in dispatch logic surfaces here, not as a
slow end-to-end runner failure.
"""

from __future__ import annotations

import pytest

from jamma.lmm.dispatch import LmmDispatch, select_dispatch_path

_ALL_AVAILABLE = {
    "c_split_available": True,
    "c_general_available": True,
    "c_fused_available": True,
    "c_fused_general_available": True,
    "c_mode4_available": True,
    "c_mode4_fused_available": True,
    "c_mode4_fused_general_available": True,
    "c_score_fused_available": True,
    "c_score_fused_ws_available": True,
    "c_lrt_fused_available": True,
    "c_lrt_fused_ws_available": True,
}
_NONE_AVAILABLE = dict.fromkeys(_ALL_AVAILABLE, False)


def _select(n_cvt: int, lmm_mode: int, **overrides) -> LmmDispatch:
    flags = {**_ALL_AVAILABLE, **overrides}
    return select_dispatch_path(n_cvt, lmm_mode, log_choices=False, **flags)


@pytest.mark.tier0
def test_no_c_kernels_disables_everything():
    d = _select(1, 1, **_NONE_AVAILABLE)
    assert not d.use_split
    assert not d.use_fused
    assert not d.use_fused_general
    assert not d.use_fused_mode4
    assert not d.use_fused_score
    assert not d.use_fused_score_ws
    assert not d.use_fused_lrt
    assert not d.use_fused_lrt_ws


@pytest.mark.tier0
def test_split_requires_basic_kernel_for_n_cvt_1():
    assert (
        _select(1, 1, c_split_available=False, c_general_available=True).use_split
        is False
    )
    assert (
        _select(1, 1, c_split_available=True, c_general_available=False).use_split
        is True
    )


@pytest.mark.tier0
def test_split_requires_general_kernel_for_n_cvt_2_plus():
    assert (
        _select(3, 1, c_split_available=True, c_general_available=False).use_split
        is False
    )
    assert (
        _select(3, 1, c_split_available=False, c_general_available=True).use_split
        is True
    )


@pytest.mark.tier0
def test_fused_general_implies_n_cvt_ge_2():
    d_solo = _select(1, 1)
    assert d_solo.use_fused is True
    assert d_solo.use_fused_general is False

    d_multi = _select(3, 1)
    assert d_multi.use_fused is True
    assert d_multi.use_fused_general is True


@pytest.mark.tier0
def test_ws_path_preempts_stateless_for_score_and_lrt():
    """When the WS variant is available, the stateless variant must be False."""
    d = _select(1, 3)
    assert d.use_fused_score_ws is True
    assert d.use_fused_score is False, "stateless must defer to WS"

    d = _select(1, 2)
    assert d.use_fused_lrt_ws is True
    assert d.use_fused_lrt is False, "stateless must defer to WS"


@pytest.mark.tier0
def test_stateless_score_lrt_used_only_when_ws_unavailable():
    d = _select(1, 3, c_score_fused_ws_available=False)
    assert d.use_fused_score_ws is False
    assert d.use_fused_score is True

    d = _select(1, 2, c_lrt_fused_ws_available=False)
    assert d.use_fused_lrt_ws is False
    assert d.use_fused_lrt is True


@pytest.mark.tier0
def test_score_lrt_fused_only_for_n_cvt_1():
    """The fused Score/LRT paths are n_cvt=1 fast paths only."""
    d = _select(2, 3)
    assert d.use_fused_score is False
    assert d.use_fused_score_ws is False

    d = _select(2, 2)
    assert d.use_fused_lrt is False
    assert d.use_fused_lrt_ws is False


@pytest.mark.tier0
def test_mode4_fused_n_cvt_1_only():
    """use_fused_mode4 is gated on n_cvt=1 + mode 4 + the mode4 kernel."""
    d = _select(1, 4)
    assert d.use_fused_mode4 is True

    d = _select(2, 4)
    assert d.use_fused_mode4 is False

    d = _select(1, 4, c_mode4_available=False)
    assert d.use_fused_mode4 is False


@pytest.mark.tier0
def test_fused_for_mode_2_or_3_at_high_n_cvt_is_disabled():
    """The general Uab fused path is wired for modes 1 and 4 only.
    Modes 2/3 should NOT activate use_fused at n_cvt>=2 since they
    don't use the workspace.
    """
    assert _select(3, 2).use_fused is False
    assert _select(3, 3).use_fused is False
    assert _select(3, 1).use_fused is True
    assert _select(3, 4).use_fused is True
