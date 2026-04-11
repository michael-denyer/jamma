"""Tests for the unified output schema.

Verifies that derived dispatch tables match their original hardcoded values
exactly, and that write_arrays_batch produces byte-identical output to
write_batch via AssocResult.
"""

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.schema import (
    ACCUM_KEYS,
    FORMAT_COLUMNS,
    HEADERS,
    MODE_SPECS,
    RESULT_FIELDS,
    TEST_TYPE_MAP,
    get_spec,
)

# ── Schema correctness: derived tables match old hardcoded values ────


class TestDerivedTables:
    """Assert derived views match the original dispatch tables exactly."""

    def test_test_type_map(self) -> None:
        assert TEST_TYPE_MAP == {1: "wald", 2: "lrt", 3: "score", 4: "all"}

    def test_accum_keys_mode1(self) -> None:
        assert ACCUM_KEYS[1] == ("betas", "ses", "logls", "lambdas", "pwalds")

    def test_accum_keys_mode2(self) -> None:
        assert ACCUM_KEYS[2] == ("lambdas_mle", "p_lrts")

    def test_accum_keys_mode3(self) -> None:
        assert ACCUM_KEYS[3] == ("betas", "ses", "p_scores")

    def test_accum_keys_mode4(self) -> None:
        assert ACCUM_KEYS[4] == (
            "betas",
            "ses",
            "logls",
            "lambdas",
            "lambdas_mle",
            "pwalds",
            "p_lrts",
            "p_scores",
        )

    def test_result_fields(self) -> None:
        expected = {
            1: {
                "betas": "beta",
                "ses": "se",
                "logls": "logl_H1",
                "lambdas": "l_remle",
                "pwalds": "p_wald",
            },
            2: {"lambdas_mle": "l_mle", "p_lrts": "p_lrt"},
            3: {"betas": "beta", "ses": "se", "p_scores": "p_score"},
            4: {
                "betas": "beta",
                "ses": "se",
                "logls": "logl_H1",
                "lambdas": "l_remle",
                "lambdas_mle": "l_mle",
                "pwalds": "p_wald",
                "p_lrts": "p_lrt",
                "p_scores": "p_score",
            },
        }
        assert expected == RESULT_FIELDS

    def test_format_columns(self) -> None:
        expected = {
            "wald": ["beta", "se", "logl_H1", "l_remle", "p_wald"],
            "score": ["beta", "se", "p_score"],
            "lrt": ["l_mle", "p_lrt"],
            "all": [
                "beta",
                "se",
                "logl_H1",
                "l_remle",
                "l_mle",
                "p_wald",
                "p_lrt",
                "p_score",
            ],
        }
        assert expected == FORMAT_COLUMNS

    def test_headers_contain_prefix(self) -> None:
        prefix = "chr\trs\tps\tn_miss\tallele1\tallele0\taf"
        for header in HEADERS.values():
            assert header.startswith(prefix)

    def test_headers_match_format_columns(self) -> None:
        for tt, cols in FORMAT_COLUMNS.items():
            header_cols = HEADERS[tt].split("\t")[7:]  # skip 7-col prefix
            assert header_cols == cols, f"Header mismatch for {tt}"

    def test_get_spec_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown lmm_mode=99"):
            get_spec(99)

    def test_get_spec_valid_modes(self) -> None:
        for mode in (1, 2, 3, 4):
            spec = get_spec(mode)
            assert spec.test_type == TEST_TYPE_MAP[mode]

    def test_all_modes_have_specs(self) -> None:
        assert set(MODE_SPECS.keys()) == {1, 2, 3, 4}

    def test_lrt_has_no_beta_se_columns(self) -> None:
        lrt_keys = {c.field_name for c in MODE_SPECS[2].stat_columns}
        assert "beta" not in lrt_keys
        assert "se" not in lrt_keys
        for mode in (1, 3, 4):
            keys = {c.field_name for c in MODE_SPECS[mode].stat_columns}
            assert "beta" in keys
            assert "se" in keys

    def test_stat_column_fmt_validation(self) -> None:
        from jamma.lmm.schema import StatColumn

        with pytest.raises(ValueError, match="Invalid format spec"):
            StatColumn("x", "x", "x", fmt="{bad}")

    def test_mode_spec_empty_columns_rejected(self) -> None:
        from jamma.lmm.schema import ModeSpec

        with pytest.raises(ValueError, match="stat_columns must not be empty"):
            ModeSpec("empty", ())

    def test_stat_column_empty_array_key_rejected(self) -> None:
        from jamma.lmm.schema import StatColumn

        with pytest.raises(ValueError, match="array_key must be a non-empty"):
            StatColumn("", "x", "x")

    def test_stat_column_empty_field_name_rejected(self) -> None:
        from jamma.lmm.schema import StatColumn

        with pytest.raises(ValueError, match="field_name must be a non-empty"):
            StatColumn("x", "", "x")

    def test_stat_column_empty_header_rejected(self) -> None:
        from jamma.lmm.schema import StatColumn

        with pytest.raises(ValueError, match=r"StatColumn.header must be a non-empty"):
            StatColumn("x", "x", "")

    def test_mode_spec_duplicate_array_key_rejected(self) -> None:
        from jamma.lmm.schema import ModeSpec, StatColumn

        col_a = StatColumn("same_key", "name_a", "hdr_a")
        col_b = StatColumn("same_key", "name_b", "hdr_b")
        with pytest.raises(ValueError, match="Duplicate array_key"):
            ModeSpec("test", (col_a, col_b))

    def test_mode_spec_duplicate_field_name_rejected(self) -> None:
        from jamma.lmm.schema import ModeSpec, StatColumn

        col_a = StatColumn("key_a", "same_name", "hdr_a")
        col_b = StatColumn("key_b", "same_name", "hdr_b")
        with pytest.raises(ValueError, match="Duplicate field_name"):
            ModeSpec("test", (col_a, col_b))

    def test_mode_spec_duplicate_header_rejected(self) -> None:
        from jamma.lmm.schema import ModeSpec, StatColumn

        col_a = StatColumn("key_a", "name_a", "same_hdr")
        col_b = StatColumn("key_b", "name_b", "same_hdr")
        with pytest.raises(ValueError, match="Duplicate header"):
            ModeSpec("test", (col_a, col_b))

    def test_stat_column_non_string_fmt_rejected(self) -> None:
        from jamma.lmm.schema import StatColumn

        with pytest.raises(ValueError, match="fmt must be a non-empty"):
            StatColumn("x", "x", "x", fmt=123)

    def test_mode_specs_is_immutable(self) -> None:
        with pytest.raises(TypeError):
            MODE_SPECS[99] = "should fail"  # type: ignore[index]


# ── Byte-identical output: write_arrays_batch vs write_batch ─────────


def _make_snp_info(n: int) -> list[dict]:
    return [
        {
            "chr": str(i % 22 + 1),
            "rs": f"rs{1000 + i}",
            "pos": 100 * i,
            "a1": "A",
            "a0": "G",
        }
        for i in range(n)
    ]


def _make_arrays(mode: int, n: int, rng: np.random.Generator) -> dict:
    """Create stat arrays matching RESULT_FIELDS for the given mode."""
    return {key: rng.random(n) for key in RESULT_FIELDS[mode]}


@pytest.mark.tier0
@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_write_arrays_batch_matches_write_batch(mode: int, tmp_path: Path) -> None:
    """write_arrays_batch produces byte-identical output to write_batch."""
    from jamma.lmm.io import IncrementalAssocWriter
    from jamma.lmm.results import _build_results

    n = 5
    rng = np.random.default_rng(42)
    snp_info = _make_snp_info(n)
    afs = rng.random(n)
    miss_counts = rng.integers(0, 3, size=n)
    arrays = _make_arrays(mode, n, rng)
    snp_indices = np.arange(n)
    test_type = TEST_TYPE_MAP[mode]

    # Path A: write_batch via AssocResult (using production _build_results)
    path_a = tmp_path / "via_assoc.txt"
    results = _build_results(mode, snp_indices, afs, miss_counts, snp_info, arrays)
    with IncrementalAssocWriter(path_a, test_type=test_type) as w:
        w.write_batch(results)

    # Path B: write_arrays_batch (no AssocResult)
    path_b = tmp_path / "via_arrays.txt"
    with IncrementalAssocWriter(path_b, test_type=test_type) as w:
        w.write_arrays_batch(mode, snp_indices, snp_info, afs, miss_counts, arrays)

    assert path_a.read_text() == path_b.read_text(), (
        f"Mode {mode}: write_arrays_batch output differs from write_batch"
    )


@pytest.mark.tier0
def test_write_arrays_batch_with_pre_sliced_subset(tmp_path: Path) -> None:
    """write_arrays_batch works with pre-sliced subset arrays."""
    from jamma.lmm.io import IncrementalAssocWriter

    n_total = 10
    rng = np.random.default_rng(99)
    snp_info = _make_snp_info(n_total)
    afs = rng.random(n_total)
    miss_counts = rng.integers(0, 3, size=n_total)
    arrays_full = _make_arrays(1, n_total, rng)

    # Select a subset and pre-slice everything
    subset = np.array([2, 5, 7])
    arrays_subset = {k: v[subset] for k, v in arrays_full.items()}

    path = tmp_path / "subset.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        w.write_arrays_batch(
            1,
            subset,
            snp_info,
            afs[subset],
            miss_counts[subset],
            arrays_subset,
        )

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 4  # header + 3 data rows


@pytest.mark.tier0
def test_write_arrays_batch_empty(tmp_path: Path) -> None:
    """write_arrays_batch with empty snp_indices writes nothing."""
    from jamma.lmm.io import IncrementalAssocWriter

    path = tmp_path / "empty.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        w.write_arrays_batch(
            1, np.array([], dtype=int), [], np.array([]), np.array([], dtype=int), {}
        )
    assert w.count == 0
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1  # header only


@pytest.mark.tier0
def test_write_arrays_batch_nan_formatting(tmp_path: Path) -> None:
    """NaN values format identically via both paths."""
    from jamma.lmm.io import IncrementalAssocWriter
    from jamma.lmm.results import _build_results

    mode = 1
    n = 2
    snp_info = _make_snp_info(n)
    afs = np.array([0.25, 0.5])
    miss_counts = np.array([0, 1])
    arrays = {k: np.array([float("nan")] * n) for k in RESULT_FIELDS[mode]}
    snp_indices = np.arange(n)

    path_a = tmp_path / "nan_assoc.txt"
    results = _build_results(mode, snp_indices, afs, miss_counts, snp_info, arrays)
    with IncrementalAssocWriter(path_a, test_type="wald") as w:
        w.write_batch(results)

    path_b = tmp_path / "nan_arrays.txt"
    with IncrementalAssocWriter(path_b, test_type="wald") as w:
        w.write_arrays_batch(mode, snp_indices, snp_info, afs, miss_counts, arrays)

    assert path_a.read_text() == path_b.read_text()


# ── write_arrays_batch error handling ────────────────────────────────


@pytest.mark.tier0
def test_write_arrays_batch_raises_if_not_opened(tmp_path: Path) -> None:
    """write_arrays_batch raises RuntimeError when writer is not opened."""
    from jamma.lmm.io import IncrementalAssocWriter

    writer = IncrementalAssocWriter(tmp_path / "dummy.txt")
    with pytest.raises(RuntimeError, match="not opened"):
        writer.write_arrays_batch(
            1,
            np.array([0]),
            _make_snp_info(1),
            np.array([0.5]),
            np.array([0]),
            _make_arrays(1, 1, np.random.default_rng(0)),
        )


@pytest.mark.tier0
def test_write_arrays_batch_mode_mismatch_raises(tmp_path: Path) -> None:
    """write_arrays_batch raises ValueError on mode/test_type mismatch."""
    from jamma.lmm.io import IncrementalAssocWriter

    path = tmp_path / "mismatch.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        with pytest.raises(ValueError, match="does not match"):
            w.write_arrays_batch(
                3,  # score mode, but writer is wald
                np.array([0]),
                _make_snp_info(1),
                np.array([0.5]),
                np.array([0]),
                _make_arrays(3, 1, np.random.default_rng(0)),
            )


@pytest.mark.tier0
def test_write_arrays_batch_missing_array_key_raises(tmp_path: Path) -> None:
    """write_arrays_batch raises ValueError when arrays dict is incomplete."""
    from jamma.lmm.io import IncrementalAssocWriter

    path = tmp_path / "missing.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        with pytest.raises(ValueError, match="missing arrays"):
            w.write_arrays_batch(
                1,
                np.array([0]),
                _make_snp_info(1),
                np.array([0.5]),
                np.array([0]),
                {"betas": np.array([1.0])},  # missing other keys
            )


@pytest.mark.tier0
def test_write_arrays_batch_length_mismatch_raises(tmp_path: Path) -> None:
    """write_arrays_batch raises ValueError when array lengths don't match."""
    from jamma.lmm.io import IncrementalAssocWriter

    path = tmp_path / "length.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        with pytest.raises(ValueError, match="afs has length 2, expected 1"):
            w.write_arrays_batch(
                1,
                np.array([0]),
                _make_snp_info(1),
                np.array([0.5, 0.6]),  # length 2, but snp_indices has length 1
                np.array([0]),
                _make_arrays(1, 1, np.random.default_rng(0)),
            )


@pytest.mark.tier0
def test_write_arrays_batch_stat_array_length_mismatch_raises(tmp_path: Path) -> None:
    """write_arrays_batch raises ValueError when a stat array has wrong length."""
    from jamma.lmm.io import IncrementalAssocWriter

    rng = np.random.default_rng(42)
    n = 3
    snp_info = _make_snp_info(n)
    afs = rng.random(n)
    miss_counts = rng.integers(0, 3, size=n)
    arrays = _make_arrays(1, n, rng)
    # Corrupt one stat array to have wrong length
    arrays["betas"] = np.array([1.0, 2.0])  # length 2, expected 3

    path = tmp_path / "bad_stat.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        with pytest.raises(ValueError, match="stat array 'betas' has length 2"):
            w.write_arrays_batch(1, np.arange(n), snp_info, afs, miss_counts, arrays)


@pytest.mark.tier0
def test_write_arrays_batch_missing_snp_key_raises(tmp_path: Path) -> None:
    """write_arrays_batch raises KeyError when snp_info missing required keys."""
    from jamma.lmm.io import IncrementalAssocWriter

    rng = np.random.default_rng(42)
    n = 2
    # snp_info dicts missing "pos" key
    snp_info = [{"chr": "1", "rs": "rs100", "a1": "A", "a0": "G"} for _ in range(n)]
    afs = rng.random(n)
    miss_counts = rng.integers(0, 3, size=n)
    arrays = _make_arrays(1, n, rng)

    path = tmp_path / "bad_snp.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        with pytest.raises(KeyError, match="missing required keys"):
            w.write_arrays_batch(1, np.arange(n), snp_info, afs, miss_counts, arrays)


@pytest.mark.tier0
def test_write_arrays_batch_multi_batch_count(tmp_path: Path) -> None:
    """write_arrays_batch accumulates count correctly across multiple calls."""
    from jamma.lmm.io import IncrementalAssocWriter

    rng = np.random.default_rng(77)
    snp_info = _make_snp_info(6)

    path = tmp_path / "multi.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        for start in (0, 3):
            batch = np.arange(start, start + 3)
            w.write_arrays_batch(
                1,
                batch,
                snp_info,
                rng.random(3),
                rng.integers(0, 3, size=3),
                _make_arrays(1, 3, rng),
            )
        assert w.count == 6

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 7  # header + 6 data rows


# ── write_arrays_batch with _LazySnpMeta ─────────────────────────────


@pytest.mark.tier0
def test_write_arrays_batch_with_lazy_snp_meta(tmp_path: Path) -> None:
    """write_arrays_batch works with LazySnpMeta (the production type)."""
    from jamma.lmm.io import IncrementalAssocWriter
    from jamma.lmm.schema import LazySnpMeta as _LazySnpMeta

    meta = {
        "chromosome": np.array(["1", "2", "3"]),
        "sid": np.array(["rs100", "rs200", "rs300"]),
        "bp_position": np.array([1000, 2000, 3000]),
        "allele_1": np.array(["A", "T", "C"]),
        "allele_2": np.array(["G", "C", "A"]),
    }
    snp_info = _LazySnpMeta(meta)

    rng = np.random.default_rng(55)
    n = 3
    path = tmp_path / "lazy.txt"
    with IncrementalAssocWriter(path, test_type="wald") as w:
        w.write_arrays_batch(
            1,
            np.arange(n),
            snp_info,
            rng.random(n),
            rng.integers(0, 3, size=n),
            _make_arrays(1, n, rng),
        )

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 4  # header + 3 data rows
    # Verify SNP metadata came through correctly
    first_data = lines[1].split("\t")
    assert first_data[0] == "1"  # chr
    assert first_data[1] == "rs100"  # rs
    assert first_data[2] == "1000"  # pos
