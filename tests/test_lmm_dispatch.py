"""Unit tests for jamma.lmm.dispatch.select_dispatch_path.

The ABI-equality gate admits all of ``methods[]`` or none of it, so the
capability is one bit and the mapping has two rows. This file writes both down
and checks every member is reachable.
"""

from __future__ import annotations

import pytest

from jamma.lmm.dispatch import DispatchPath, select_dispatch_path

pytestmark = pytest.mark.tier0

_PIPELINED_PATHS = {DispatchPath.FUSED}


def test_the_extension_bit_decides_the_path():
    assert select_dispatch_path(accel=True) is DispatchPath.FUSED
    assert select_dispatch_path(accel=False) is DispatchPath.NUMPY_FALLBACK


def test_every_path_is_reachable():
    """A member no input can select is dead weight, and this is what catches it."""
    reached = {select_dispatch_path(accel=accel) for accel in (True, False)}
    assert reached == set(DispatchPath), (
        f"unreachable members: {sorted(m.name for m in set(DispatchPath) - reached)}"
    )


def test_path_properties_agree_with_membership():
    """The derived properties must not drift from the members they describe."""
    for accel in (True, False):
        path = select_dispatch_path(accel=accel)
        assert path.is_native == (path in _PIPELINED_PATHS)
