"""Comparison distinguishes schema absence from numeric values."""

from dataclasses import replace

import pytest

from jamma.validation.compare import compare_assoc_results
from tests.assoc_test_helpers import make_assoc

pytestmark = pytest.mark.tier0


def test_zero_likelihood_is_compared():
    expected = make_assoc(logl_H1=0.0)
    actual = replace(expected, logl_H1=100.0)
    result = compare_assoc_results([actual], [expected])
    assert not result.passed
    assert not result.logl_H1.passed


def test_inconsistent_rows_after_initial_sample_are_rejected():
    rows = [make_assoc(rs=f"rs{i}") for i in range(6)]
    rows[-1] = replace(rows[-1], p_wald=None, p_score=0.01)
    with pytest.raises(ValueError, match="schema"):
        compare_assoc_results(rows, rows)


@pytest.mark.parametrize("test_type", ["wald", "lrt", "score", "all"])
def test_empty_parsed_table_preserves_schema(tmp_path, test_type):
    from jamma.lmm.schema import HEADERS
    from jamma.validation.compare import load_gemma_assoc

    path = tmp_path / "empty.assoc.txt"
    path.write_text(HEADERS[test_type] + "\n")
    rows = load_gemma_assoc(path)
    assert rows.columns == tuple(HEADERS[test_type].split("\t"))
    result = compare_assoc_results(rows, rows)
    assert result.passed
    assert result.n_snps == 0


def test_absent_likelihood_does_not_match_present_nan():
    expected = make_assoc(logl_H1=float("nan"))
    actual = replace(expected, logl_H1=None)
    assert not compare_assoc_results([actual], [expected]).passed
