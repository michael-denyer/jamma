"""Tests for the unified output schema.

Pins each mode's columns in MODE_SPECS, and checks that
write_arrays_batch produces the expected TSV rows.
"""

from pathlib import Path

import numpy as np
import pytest

from jamma.io.plink import PlinkMetadata, get_plink_metadata
from jamma.lmm.assoc_output import IncrementalAssocWriter, build_results
from jamma.lmm.schema import (
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    DEFAULT_MAF,
    DEFAULT_MISS,
    DEFAULT_N_GRID,
    DEFAULT_N_REFINE,
    MIN_N_REFINE,
    MODE_SPECS,
    SnpMeta,
    get_spec,
)

pytestmark = pytest.mark.tier0

# ── Default LMM knobs: pin the single-source-of-truth constants ────


class TestDefaultKnobs:
    """Pin the DEFAULT_* LMM knobs so a default cannot silently drift.

    These constants are the single source of truth threaded through LmmConfig,
    PipelineConfig, the CLI/gwas() maf/miss/l_min/l_max options, and every runner
    dispatch entry point. A one-character edit here would change GWAS behaviour
    everywhere with no other test failing — this test locks the values.
    """

    def test_gemma_cli_defaults(self) -> None:
        # maf/miss/l_min/l_max match GEMMA v0.98.5 CLI defaults.
        assert DEFAULT_MAF == 0.01
        assert DEFAULT_MISS == 0.05
        assert DEFAULT_L_MIN == 1e-5
        assert DEFAULT_L_MAX == 1e5

    def test_golden_section_knobs(self) -> None:
        # JAMMA's lambda-optimizer knobs — no GEMMA equivalent (GEMMA uses Brent).
        assert DEFAULT_N_GRID == 50
        # The default is the minimum: every run already ran 20 refinements,
        # since LmmConfig raises a lower value to MIN_N_REFINE.
        assert DEFAULT_N_REFINE == MIN_N_REFINE == 20


# ── Schema correctness: each mode's columns match GEMMA's ─────────────


class TestModeSpecs:
    """Pin each mode's name, kernel array keys, and output column names."""

    @pytest.mark.parametrize(
        ("mode", "test_type", "columns"),
        [
            (
                1,
                "wald",
                [
                    ("betas", "beta"),
                    ("ses", "se"),
                    ("logls", "logl_H1"),
                    ("lambdas", "l_remle"),
                    ("pwalds", "p_wald"),
                ],
            ),
            (
                2,
                "lrt",
                [("logls", "logl_H1"), ("lambdas_mle", "l_mle"), ("p_lrts", "p_lrt")],
            ),
            (3, "score", [("betas", "beta"), ("ses", "se"), ("p_scores", "p_score")]),
            (
                4,
                "all",
                [
                    ("betas", "beta"),
                    ("ses", "se"),
                    ("logls", "logl_H1"),
                    ("lambdas", "l_remle"),
                    ("lambdas_mle", "l_mle"),
                    ("pwalds", "p_wald"),
                    ("p_lrts", "p_lrt"),
                    ("p_scores", "p_score"),
                ],
            ),
        ],
    )
    def test_columns(self, mode: int, test_type: str, columns: list) -> None:
        spec = get_spec(mode)
        assert spec.test_type == test_type
        assert [(c.array_key, c.field_name) for c in spec.stat_columns] == columns

    def test_header_is_gemma_prefix_then_stat_columns(self) -> None:
        assert MODE_SPECS[1].header == (
            "chr\trs\tps\tn_miss\tallele1\tallele0\taf"
            "\tbeta\tse\tlogl_H1\tl_remle\tp_wald"
        )

    def test_get_spec_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="lmm_mode must be"):
            get_spec(99)

    def test_all_modes_have_specs(self) -> None:
        assert set(MODE_SPECS.keys()) == {1, 2, 3, 4}

    def test_mode_specs_is_immutable(self) -> None:
        with pytest.raises(TypeError):
            MODE_SPECS[99] = "should fail"  # type: ignore[index]


# ── Both sinks render the same row ────────────────────────────────────


def _make_snp_info(n: int) -> SnpMeta:
    return SnpMeta.from_dicts(
        [
            {
                "chr": str(i % 22 + 1),
                "rs": f"rs{1000 + i}",
                "pos": 100 * i,
                "a1": "A",
                "a0": "G",
            }
            for i in range(n)
        ]
    )


def _make_arrays(mode: int, n: int, rng: np.random.Generator) -> dict:
    """Create stat arrays keyed by the mode's array keys."""
    return {c.array_key: rng.random(n) for c in get_spec(mode).stat_columns}


def _render(results: list, mode: int) -> str:
    """Render AssocResult records as .assoc.txt text, the test's own oracle."""
    spec = get_spec(mode)
    lines = [spec.header]
    for r in results:
        meta = (
            f"{r.chr}\t{r.rs}\t{r.ps}\t{r.n_miss}\t{r.allele1}\t{r.allele0}\t{r.af:.3f}"
        )
        stats = "\t".join(f"{getattr(r, c.field_name):.6e}" for c in spec.stat_columns)
        lines.append(f"{meta}\t{stats}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_writer_matches_built_results(mode: int, tmp_path: Path) -> None:
    """The disk writer and the in-memory AssocResult sink carry the same row."""
    n = 5
    rng = np.random.default_rng(42)
    snp_info = _make_snp_info(n)
    afs = rng.random(n)
    miss_counts = rng.integers(0, 3, size=n)
    arrays = _make_arrays(mode, n, rng)
    snp_indices = np.arange(n)
    spec = get_spec(mode)

    results = build_results(spec, snp_indices, afs, miss_counts, snp_info, arrays)

    path = tmp_path / "via_arrays.txt"
    with IncrementalAssocWriter(path, spec) as w:
        w.write_arrays_batch(snp_indices, snp_info, afs, miss_counts, arrays)

    assert path.read_text() == _render(results, mode)


def test_write_arrays_batch_with_pre_sliced_subset(tmp_path: Path) -> None:
    """write_arrays_batch works with pre-sliced subset arrays."""

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
    with IncrementalAssocWriter(path, MODE_SPECS[1]) as w:
        w.write_arrays_batch(
            subset,
            snp_info,
            afs[subset],
            miss_counts[subset],
            arrays_subset,
        )

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 4  # header + 3 data rows


def test_write_arrays_batch_empty(tmp_path: Path) -> None:
    """write_arrays_batch with empty snp_indices writes nothing."""

    path = tmp_path / "empty.txt"
    with IncrementalAssocWriter(path, MODE_SPECS[1]) as w:
        w.write_arrays_batch(
            np.array([], dtype=int),
            SnpMeta.from_dicts([]),
            np.array([]),
            np.array([], dtype=int),
            {},
        )
    assert w.count == 0
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1  # header only


def test_write_arrays_batch_nan_formatting(tmp_path: Path) -> None:
    """NaN statistics render as GEMMA's ``nan``."""
    n = 2
    snp_info = _make_snp_info(n)
    arrays = {c.array_key: np.full(n, np.nan) for c in MODE_SPECS[1].stat_columns}

    path = tmp_path / "nan_arrays.txt"
    with IncrementalAssocWriter(path, MODE_SPECS[1]) as w:
        w.write_arrays_batch(
            np.arange(n), snp_info, np.array([0.25, 0.5]), np.array([0, 1]), arrays
        )

    assert path.read_text().splitlines()[1:] == [
        "1\trs1000\t0\t0\tA\tG\t0.250\tnan\tnan\tnan\tnan\tnan",
        "2\trs1001\t100\t1\tA\tG\t0.500\tnan\tnan\tnan\tnan\tnan",
    ]


# ── write_arrays_batch error handling ────────────────────────────────


def test_write_arrays_batch_raises_if_not_opened(tmp_path: Path) -> None:
    """write_arrays_batch raises RuntimeError when writer is not opened."""

    writer = IncrementalAssocWriter(tmp_path / "dummy.txt", MODE_SPECS[1])
    with pytest.raises(RuntimeError, match="not opened"):
        writer.write_arrays_batch(
            np.array([0]),
            _make_snp_info(1),
            np.array([0.5]),
            np.array([0]),
            _make_arrays(1, 1, np.random.default_rng(0)),
        )


def test_write_arrays_batch_missing_array_key_raises(tmp_path: Path) -> None:
    """write_arrays_batch raises ValueError when arrays dict is incomplete."""

    path = tmp_path / "missing.txt"
    with IncrementalAssocWriter(path, MODE_SPECS[1]) as w:
        with pytest.raises(ValueError, match="missing arrays"):
            w.write_arrays_batch(
                np.array([0]),
                _make_snp_info(1),
                np.array([0.5]),
                np.array([0]),
                {"betas": np.array([1.0])},  # missing other keys
            )


def test_write_arrays_batch_length_mismatch_raises(tmp_path: Path) -> None:
    """write_arrays_batch raises ValueError when array lengths don't match."""

    path = tmp_path / "length.txt"
    with IncrementalAssocWriter(path, MODE_SPECS[1]) as w:
        with pytest.raises(ValueError, match="afs has length 2, expected 1"):
            w.write_arrays_batch(
                np.array([0]),
                _make_snp_info(1),
                np.array([0.5, 0.6]),  # length 2, but snp_indices has length 1
                np.array([0]),
                _make_arrays(1, 1, np.random.default_rng(0)),
            )


def test_write_arrays_batch_stat_array_length_mismatch_raises(tmp_path: Path) -> None:
    """write_arrays_batch raises ValueError when a stat array has wrong length."""

    rng = np.random.default_rng(42)
    n = 3
    snp_info = _make_snp_info(n)
    afs = rng.random(n)
    miss_counts = rng.integers(0, 3, size=n)
    arrays = _make_arrays(1, n, rng)
    # Corrupt one stat array to have wrong length
    arrays["betas"] = np.array([1.0, 2.0])  # length 2, expected 3

    path = tmp_path / "bad_stat.txt"
    with IncrementalAssocWriter(path, MODE_SPECS[1]) as w:
        with pytest.raises(ValueError, match="stat array 'betas' has length 2"):
            w.write_arrays_batch(np.arange(n), snp_info, afs, miss_counts, arrays)


def test_snp_meta_from_dicts_missing_key_raises() -> None:
    """SnpMeta.from_dicts rejects a dict missing a canonical key at the boundary."""
    with pytest.raises(KeyError, match="pos"):
        SnpMeta.from_dicts([{"chr": "1", "rs": "rs100", "a1": "A", "a0": "G"}])


def test_snp_meta_from_plink_meta_sliced_matches_from_dicts() -> None:
    """from_plink_meta sliced by an index array equals the old dict path.

    The pipeline used to build a list of per-SNP dicts for the indices it
    kept, then parse that list back into SnpMeta via from_dicts. Fancy
    indexing on from_plink_meta's arrays must produce identical columns.
    """
    from tests.fixture_paths import SYNTHETIC

    meta = get_plink_metadata(SYNTHETIC.bfile)
    indices = np.array([0, 2, 5, meta.n_snps - 1])

    sliced = SnpMeta.from_plink_meta(meta, indices)

    old_path = SnpMeta.from_dicts(
        [
            {
                "chr": str(meta.chromosome[i]),
                "rs": meta.sid[i],
                "pos": int(meta.bp_position[i]),
                "a1": meta.allele_1[i],
                "a0": meta.allele_2[i],
            }
            for i in indices
        ]
    )

    np.testing.assert_array_equal(sliced.chr, old_path.chr)
    np.testing.assert_array_equal(sliced.rs, old_path.rs)
    np.testing.assert_array_equal(sliced.pos, old_path.pos)
    np.testing.assert_array_equal(sliced.a1, old_path.a1)
    np.testing.assert_array_equal(sliced.a0, old_path.a0)


def test_snp_meta_from_plink_meta_no_indices_keeps_every_snp() -> None:
    """from_plink_meta with indices=None returns every SNP, unfiltered."""
    from tests.fixture_paths import SYNTHETIC

    meta = get_plink_metadata(SYNTHETIC.bfile)

    full = SnpMeta.from_plink_meta(meta)

    assert len(full) == meta.n_snps


def test_write_arrays_batch_multi_batch_count(tmp_path: Path) -> None:
    """write_arrays_batch accumulates count correctly across multiple calls."""

    rng = np.random.default_rng(77)
    snp_info = _make_snp_info(6)

    path = tmp_path / "multi.txt"
    with IncrementalAssocWriter(path, MODE_SPECS[1]) as w:
        for start in (0, 3):
            batch = np.arange(start, start + 3)
            w.write_arrays_batch(
                batch,
                snp_info,
                rng.random(3),
                rng.integers(0, 3, size=3),
                _make_arrays(1, 3, rng),
            )
        assert w.count == 6

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 7  # header + 6 data rows


# ── write_arrays_batch with SnpMeta from PLINK metadata ──────────────


def test_write_arrays_batch_with_plink_meta(tmp_path: Path) -> None:
    """write_arrays_batch works with SnpMeta built from PLINK metadata."""

    meta = PlinkMetadata(
        n_samples=4,
        n_snps=3,
        iid=np.array([["F1", "I1"], ["F2", "I2"], ["F3", "I3"], ["F4", "I4"]]),
        sid=np.array(["rs100", "rs200", "rs300"]),
        chromosome=np.array(["1", "2", "3"]),
        bp_position=np.array([1000, 2000, 3000]),
        allele_1=np.array(["A", "T", "C"]),
        allele_2=np.array(["G", "C", "A"]),
    )
    snp_info = SnpMeta.from_plink_meta(meta)

    rng = np.random.default_rng(55)
    n = 3
    path = tmp_path / "lazy.txt"
    with IncrementalAssocWriter(path, MODE_SPECS[1]) as w:
        w.write_arrays_batch(
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


class TestEnvFlag:
    """Every JAMMA toggle shares one truthiness rule.

    The rule was re-spelled at six call sites across five environment
    variables, so the accepted values could drift apart. These pin the
    documented behaviour: presence-based, with only unset/""/"0" meaning off.
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, False),
            ("", False),
            ("0", False),
            ("  0  ", False),
            ("1", True),
            ("yes", True),
            # Deliberate: presence-based, so these read as ON. Documented in
            # docs/CONFIGURATION.md and in env_flag's own docstring.
            ("false", True),
            ("off", True),
            ("no", True),
        ],
    )
    def test_truthiness(self, monkeypatch, value, expected):
        from jamma.core.constants import env_flag

        name = "JAMMA_TEST_FLAG"
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

        assert env_flag(name) is expected

    def test_build_support_copy_matches(self, monkeypatch):
        """openmp_detect cannot import env_flag; its inline rule must agree."""
        from jamma._build_support.openmp_detect import openmp_disabled_by_env
        from jamma.core.constants import env_flag

        for value in ("", "0", "1", "false", "off"):
            monkeypatch.setenv("JAMMA_NO_OPENMP", value)
            assert openmp_disabled_by_env() == env_flag("JAMMA_NO_OPENMP"), value
