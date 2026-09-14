"""Beta comparison carries an absolute floor scaled by the standard error."""

from __future__ import annotations

import pytest

from jamma.validation.compare import compare_assoc_results
from jamma.validation.tolerances import ToleranceConfig
from tests.assoc_test_helpers import make_assoc as _make_assoc

pytestmark = pytest.mark.tier0


def test_null_effect_beta_is_compared_against_its_standard_error():
    """rs13475789 from LOCO chromosome 1: 1.8% relative on a beta 1e-4 of its SE."""
    actual = [_make_assoc(rs="rs13475789", beta=4.444851e-6, se=3.515983e-2)]
    expected = [_make_assoc(rs="rs13475789", beta=4.366448e-6, se=3.515983e-2)]

    assert compare_assoc_results(actual, expected).passed is True
    without_floor = compare_assoc_results(
        actual, expected, ToleranceConfig(beta_se_floor=0.0)
    )
    assert without_floor.passed is False
    assert without_floor.beta.passed is False


def test_beta_floor_scales_with_the_standard_error():
    """One absolute beta difference passes at a large SE, fails at a small one."""
    loose = compare_assoc_results(
        [_make_assoc(rs="rs1", beta=1e-3, se=100.0)],
        [_make_assoc(rs="rs1", beta=0.0, se=100.0)],
    )
    tight = compare_assoc_results(
        [_make_assoc(rs="rs1", beta=1e-3, se=0.01)],
        [_make_assoc(rs="rs1", beta=0.0, se=0.01)],
    )
    assert loose.beta.passed is True
    assert tight.beta.passed is False
