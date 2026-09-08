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
def test_empty_file_compares_as_zero_snps(tmp_path, test_type):
    from jamma.lmm.schema import HEADERS
    from jamma.validation.compare import load_gemma_assoc

    path = tmp_path / "empty.assoc.txt"
    path.write_text(HEADERS[test_type] + "\n")
    rows = load_gemma_assoc(path)
    assert len(rows) == 0
    result = compare_assoc_results(rows, rows)
    assert result.passed
    assert result.n_snps == 0


def test_mode_mismatch_between_actual_and_reference_is_rejected():
    reference = make_assoc()
    actual = replace(reference, p_lrt=0.02, p_score=0.03, l_mle=0.6)
    with pytest.raises(ValueError, match="actual mode 4 but reference mode 1"):
        compare_assoc_results([actual], [reference])


def test_absent_likelihood_does_not_match_present_nan():
    expected = make_assoc(logl_H1=float("nan"))
    actual = replace(expected, logl_H1=None)
    assert not compare_assoc_results([actual], [expected]).passed


@pytest.mark.parametrize("actual_type", ["wald", "lrt", "score", "all"])
@pytest.mark.parametrize("expected_type", ["wald", "lrt", "score", "all"])
def test_empty_files_keep_their_modes(tmp_path, actual_type, expected_type):
    from jamma.lmm.schema import HEADERS
    from jamma.validation.compare import load_gemma_assoc

    actual_path, expected_path = tmp_path / "actual.txt", tmp_path / "expected.txt"
    actual_path.write_text(HEADERS[actual_type] + "\n")
    expected_path.write_text(HEADERS[expected_type] + "\n")
    actual, expected = load_gemma_assoc(actual_path), load_gemma_assoc(expected_path)
    if actual_type == expected_type:
        assert compare_assoc_results(actual, expected).passed
    else:
        with pytest.raises(ValueError, match="Association schemas differ"):
            compare_assoc_results(actual, expected)


@pytest.mark.parametrize(
    "row",
    [
        make_assoc(),
        make_assoc(p_wald=None, p_lrt=0.02, l_mle=0.8),
        make_assoc(p_wald=None, p_score=0.05),
        make_assoc(p_lrt=0.02, p_score=0.05, l_mle=0.8),
    ],
)
def test_empty_in_memory_rows_report_count_mismatch(row):
    result = compare_assoc_results([], [row])
    assert not result.passed
    assert "SNP count mismatch" in result.beta.message


def test_empty_slice_preserves_the_parsed_mode(tmp_path):
    from jamma.lmm.schema import HEADERS
    from jamma.validation.compare import load_gemma_assoc

    path = tmp_path / "score.txt"
    path.write_text(HEADERS["score"] + "\n")
    rows = load_gemma_assoc(path)
    assert rows[:0].mode == 3


def test_dataset_rejects_rows_with_a_different_mode():
    from jamma.validation import AssocDataset

    with pytest.raises(ValueError, match="Rows carry mode 1, expected mode 3"):
        AssocDataset(3, (make_assoc(),))


@pytest.fixture
def loaded_wald_pair(tmp_path):
    from jamma.validation import load_gemma_assoc
    from tests.fakes.assoc_files import WALD_FULL_COLS, write_assoc

    path = tmp_path / "wald.txt"
    write_assoc(
        path,
        WALD_FULL_COLS,
        [
            ["1", f"rs{i}", i, 0, "A", "G", 0.2, 0.1, 0.2, -10, 0.8, 0.05]
            for i in range(2)
        ],
    )
    return load_gemma_assoc(path), load_gemma_assoc(path)


@pytest.mark.parametrize("sides", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("row_indices", [(1,), (0, 1)])
def test_comparison_rejects_mutated_loaded_schema(loaded_wald_pair, sides, row_indices):
    for side in sides:
        for index in row_indices:
            row = loaded_wald_pair[side][index]
            row.p_wald = None
            row.l_remle = None
            row.logl_H1 = None
            row.p_score = 0.01 if side == 0 else 0.9

    with pytest.raises(ValueError, match=r"schema|Rows carry mode"):
        compare_assoc_results(*loaded_wald_pair)


def test_comparison_still_compares_mutated_numeric_values(loaded_wald_pair):
    actual, expected = loaded_wald_pair
    actual[0].p_wald = 0.9
    result = compare_assoc_results(actual, expected)
    assert not result.passed
    assert not result.p_wald.passed
