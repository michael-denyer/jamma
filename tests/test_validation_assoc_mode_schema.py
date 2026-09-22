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
                logl_H1=-42.0,
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
    """compare_assoc_results compares exactly af plus MODE_SPECS[mode]'s columns,
    so the comparator cannot drift from the schema's declared column set."""
    comparison = compare_assoc_results([sample], [sample])
    expected = {"af"} | {c.field_name for c in MODE_SPECS[mode].stat_columns}
    actual = set(comparison.columns)
    assert actual == expected
