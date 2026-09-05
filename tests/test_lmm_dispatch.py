"""Unit tests for jamma.lmm.dispatch.select_dispatch_path.

The selector used to take an eleven-field capability snapshot, so its input
space was large enough that only a property sweep could cover it. The
ABI-equality gate admits all of ``methods[]`` or none of it, so the capability
is one bit and the whole space is small enough to write down. This file states
the mapping as a table and checks it exhaustively, which pins the actual
decision rather than the invariants it happens to satisfy.
"""

from __future__ import annotations

from itertools import product

import pytest

from jamma.lmm.dispatch import DispatchPath, select_dispatch_path
from jamma.lmm.schema import LmmMode

pytestmark = pytest.mark.tier0

_MODES: tuple[LmmMode, ...] = (1, 2, 3, 4)
_NCVT_1 = (1,)
_NCVT_MANY = (2, 3, 5, 100, 101)

# The complete mapping when the extension is loaded, by (n_cvt==1?, lmm_mode).
_EXPECTED = {
    (True, 1): DispatchPath.FUSED,
    (True, 4): DispatchPath.FUSED,
    (True, 3): DispatchPath.FUSED,
    (True, 2): DispatchPath.FUSED,
    (False, 1): DispatchPath.FUSED_GENERAL,
    (False, 4): DispatchPath.FUSED_GENERAL,
    (False, 3): DispatchPath.FUSED_GENERAL,
    (False, 2): DispatchPath.FUSED_GENERAL,
}

_FEEDS_RAW_UTG = {
    DispatchPath.FUSED,
    DispatchPath.FUSED_GENERAL,
}


def _select(n_cvt: int, lmm_mode: LmmMode, *, accel: bool = True) -> DispatchPath:
    return select_dispatch_path(n_cvt, lmm_mode, accel=accel, log_choices=False)


def test_no_extension_is_always_the_numpy_fallback():
    for n_cvt, mode in product(_NCVT_1 + _NCVT_MANY, _MODES):
        assert _select(n_cvt, mode, accel=False) is DispatchPath.NUMPY_FALLBACK


def test_every_input_maps_to_the_documented_path():
    for n_cvt, mode in product(_NCVT_1 + _NCVT_MANY, _MODES):
        expected = _EXPECTED[(n_cvt == 1, mode)]
        assert _select(n_cvt, mode) is expected, (
            f"n_cvt={n_cvt} mode={mode}: expected {expected.name}, "
            f"got {_select(n_cvt, mode).name}"
        )


def test_every_path_is_reachable():
    """A member no input can select is dead weight, and this is what catches it."""
    reached = {
        _select(n_cvt, mode, accel=accel)
        for n_cvt, mode, accel in product(_NCVT_1 + _NCVT_MANY, _MODES, (True, False))
    }
    assert reached == set(DispatchPath), (
        f"unreachable members: {sorted(m.name for m in set(DispatchPath) - reached)}"
    )


def test_path_properties_agree_with_membership():
    """The derived properties must not drift from the members they describe."""
    for n_cvt, mode, accel in product(_NCVT_1 + _NCVT_MANY, _MODES, (True, False)):
        path = _select(n_cvt, mode, accel=accel)
        assert path.use_split == (path is not DispatchPath.NUMPY_FALLBACK)
        assert path.feeds_raw_utg == (path in _FEEDS_RAW_UTG)


def test_mode_and_ncvt_gating():
    """Each C path is wired for particular modes and covariate counts."""
    for n_cvt, mode in product(_NCVT_1 + _NCVT_MANY, _MODES):
        path = _select(n_cvt, mode)
        if path is DispatchPath.FUSED:
            assert n_cvt == 1
        if path is DispatchPath.FUSED_GENERAL:
            assert n_cvt >= 2


@pytest.mark.parametrize("bad_mode", [0, 5, -1, 99])
def test_invalid_mode_raises(bad_mode):
    with pytest.raises(ValueError, match="lmm_mode must be"):
        _select(1, bad_mode)
