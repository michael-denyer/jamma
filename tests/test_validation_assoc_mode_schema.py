"""Association-comparison coverage derived from the LMM mode schema."""

from __future__ import annotations

import pytest

from jamma.lmm.schema import MODE_SPECS
from jamma.validation.compare import compare_assoc_results
from tests.assoc_test_helpers import make_assoc

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize(
    ("mode", "sample"),
    [
        (1, make_assoc()),
        (
            2,
            make_assoc(
                beta=float("nan"),
                se=float("nan"),
                p_wald=None,
                logl_H1=None,
                l_remle=None,
                p_lrt=0.02,
                l_mle=0.8,
            ),
        ),
        (
            3,
            make_assoc(p_wald=None, logl_H1=None, l_remle=None, p_score=0.05),
        ),
        (
            4,
            make_assoc(p_lrt=0.02, p_score=0.05, l_mle=0.8),
        ),
    ],
)
def test_compared_columns_match_mode_spec(mode, sample):
    """The set of columns compare_assoc_results actively compares for a mode
    equals MODE_SPECS[mode].stat_columns, so the comparator cannot drift from
    the schema's declared column set for that mode.

    beta/se are excluded from this check: they are always-present output slots,
    not entries in MODE_SPECS[mode].stat_columns, and LRT compares them too
    (both sides all-NaN by construction) rather than skipping them outright.
    """
    comparison = compare_assoc_results([sample], [sample])
    expected = frozenset(
        column.field_name for column in MODE_SPECS[mode].stat_columns
    ) - {"beta", "se"}
    actual = frozenset(
        field
        for field in ("p_wald", "logl_H1", "l_remle", "p_score", "p_lrt", "l_mle")
        if (result := getattr(comparison, field)) is not None
        and not (
            result.passed
            and result.worst_location is None
            and "skipped" in result.message
        )
    )
    assert actual == expected
