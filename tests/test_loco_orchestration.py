"""LOCO kinship orchestration: buffer aliasing, biological chromosome
ordering, fallback-path cleanup.

Companion to test_loco_numpy.py (which covers the NumPy LOCO computation
paths) and test_loco_eigen_cache.py (which covers cache I/O).
"""

import numpy as np
import pytest

from jamma.kinship.loco import (
    LocoKinshipStream,
    _yield_full_kinship_fallback,
    _yield_loco_matrices,
)
from jamma.utils import chr_sort_key

pytestmark = pytest.mark.tier0


class TestFallbackKinshipAliasing:
    """Verify _yield_full_kinship_fallback yields independent copies."""

    def test_yielded_matrices_are_independent(self):
        """Each yielded matrix must be a separate buffer, not aliased."""
        n = 10
        S_full = np.random.default_rng(42).standard_normal((n, n))
        S_full = S_full @ S_full.T  # symmetric
        chrs = ["3", "7"]

        results = list(_yield_full_kinship_fallback(S_full, chrs, n_filtered=100))

        assert len(results) == 2
        _, K0 = results[0]
        _, K1 = results[1]
        # Must be different buffer objects
        assert K0.ctypes.data != K1.ctypes.data
        # But numerically equal (both are K_full)
        np.testing.assert_array_equal(K0, K1)

    def test_mutation_does_not_propagate(self):
        """Mutating one yielded matrix must not affect the other."""
        n = 5
        S_full = np.eye(n, dtype=np.float64)
        chrs = ["1", "2"]

        results = list(_yield_full_kinship_fallback(S_full, chrs, n_filtered=1))
        _, K0 = results[0]
        _, K1 = results[1]

        original = K0.copy()
        K0[:] = 999.0  # mutate first
        np.testing.assert_array_equal(K1, original)

    def test_empty_chrs_yields_nothing(self):
        """No chromosomes = no output."""
        S_full = np.eye(3)
        assert list(_yield_full_kinship_fallback(S_full, [], n_filtered=10)) == []

    def test_raises_on_zero_n_filtered(self):
        """n_filtered=0 raises ValueError (division by zero guard)."""
        S_full = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError, match="n_filtered is 0"):
            list(_yield_full_kinship_fallback(S_full, ["1"], n_filtered=0))

    def test_yields_correctly_normalized_kinship(self):
        """Yielded matrices equal S_full / n_filtered."""
        n = 5
        rng = np.random.default_rng(42)
        S_full = rng.standard_normal((n, n))
        S_full = S_full @ S_full.T
        expected = S_full / 100
        results = list(
            _yield_full_kinship_fallback(S_full.copy(), ["1", "2"], n_filtered=100)
        )
        for _, K in results:
            np.testing.assert_allclose(K, expected, rtol=1e-14)


class TestChromosomeSortKey:
    """Verify biological chromosome ordering."""

    def test_numeric_order(self):
        """Numeric chromosomes sort by integer value, not lexicographically."""
        chrs = ["1", "10", "11", "2", "20", "3", "9", "22"]
        result = sorted(chrs, key=chr_sort_key)
        assert result == ["1", "2", "3", "9", "10", "11", "20", "22"]

    def test_special_chromosomes_after_numeric(self):
        """X, Y, XY, MT sort after numeric chromosomes."""
        chrs = ["X", "1", "MT", "22", "Y"]
        result = sorted(chrs, key=chr_sort_key)
        assert result == ["1", "22", "X", "Y", "MT"]

    def test_case_insensitive_specials(self):
        """Special chromosome names are case-insensitive."""
        chrs = ["x", "y", "mt", "1"]
        result = sorted(chrs, key=chr_sort_key)
        assert result == ["1", "x", "y", "mt"]

    def test_unknown_chromosomes_sort_last(self):
        """Unknown chromosome names sort after all known ones, alphabetically."""
        chrs = ["1", "X", "scaffold_17", "Un"]
        result = sorted(chrs, key=chr_sort_key)
        assert result == ["1", "X", "Un", "scaffold_17"]

    def test_m_alias_for_mt(self):
        """'M' is an alias for 'MT' (both are mitochondrial)."""
        chrs = ["1", "M", "MT"]
        result = sorted(chrs, key=chr_sort_key)
        # M and MT have same sort position; stable sort preserves input order
        assert result == ["1", "M", "MT"]

    def test_full_human_karyotype(self):
        """All human chromosomes in correct biological order."""
        chrs = [str(i) for i in range(1, 23)] + ["X", "Y", "XY", "MT"]
        shuffled = chrs.copy()
        np.random.default_rng(0).shuffle(shuffled)
        assert sorted(shuffled, key=chr_sort_key) == chrs


class TestYieldLocoMatricesOrdering:
    """Verify _yield_loco_matrices produces biological order."""

    def test_biological_order(self):
        """Chromosomes yielded in biological order, not lexicographic."""
        n = 4
        S_full = np.eye(n, dtype=np.float64) * 100
        chr_names = ["1", "10", "2", "X"]
        S_chr = {name: np.eye(n, dtype=np.float64) for name in chr_names}
        n_chr_filtered = dict.fromkeys(chr_names, 10)
        K_loco_buf = np.empty((n, n), dtype=np.float64)

        yielded_order = [
            name
            for name, _ in _yield_loco_matrices(
                S_full, S_chr, n_chr_filtered, n_filtered=40, K_loco_buf=K_loco_buf
            )
        ]
        assert yielded_order == ["1", "2", "10", "X"]


def _loco_fixtures(n=10):
    """Shared LOCO test fixtures: S_full, S_chr, n_chr_filtered, K_loco_buf."""
    chr_names = ["1", "2", "3"]
    S_full = np.eye(n, dtype=np.float64) * 3.0
    S_chr = {
        name: np.eye(n, dtype=np.float64) * (i + 1) for i, name in enumerate(chr_names)
    }
    n_chr_filtered = dict.fromkeys(chr_names, 10)
    K_loco_buf = np.empty((n, n), dtype=np.float64)
    return S_full, S_chr, n_chr_filtered, K_loco_buf


class TestLocoKinshipStreamMaterialize:
    """Verify LocoKinshipStream.materialize() copies each K_loco, breaking aliasing."""

    def test_materialize_yields_independent_arrays(self):
        """materialize() must produce a distinct K_loco per chromosome."""
        S_full, S_chr, n_chr_filtered, K_loco_buf = _loco_fixtures()

        stream = LocoKinshipStream(
            _matrices=_yield_loco_matrices(
                S_full, S_chr, n_chr_filtered, n_filtered=30, K_loco_buf=K_loco_buf
            )
        )
        results = stream.materialize()

        assert not np.allclose(results["1"], results["2"]), (
            "Chromosomes 1 and 2 should have different K_loco matrices"
        )
        assert not np.allclose(results["1"], results["3"]), (
            "Chromosomes 1 and 3 should have different K_loco matrices"
        )
        assert not np.allclose(results["2"], results["3"]), (
            "Chromosomes 2 and 3 should have different K_loco matrices"
        )


class TestYieldLocoMatricesBufferReuse:
    """Verify the shared-buffer yield is correct when each matrix is consumed first."""

    def test_sequential_consumption_produces_correct_values(self):
        """Each yielded matrix is correct when copied before advancing."""
        S_full, _, n_chr_filtered, K_loco_buf = _loco_fixtures()

        # Reference: copy each matrix as it is yielded (consume-before-advance).
        reference = {
            chr_name: K.copy()
            for chr_name, K in _yield_loco_matrices(
                S_full,
                _loco_fixtures()[1],  # fresh S_chr (consumed by iterator)
                n_chr_filtered,
                n_filtered=30,
                K_loco_buf=K_loco_buf.copy(),
            )
        }

        sequential_results = {}
        for chr_name, K_loco in _yield_loco_matrices(
            S_full,
            _loco_fixtures()[1],  # fresh S_chr
            n_chr_filtered,
            n_filtered=30,
            K_loco_buf=K_loco_buf,
        ):
            sequential_results[chr_name] = K_loco.copy()

        for chr_name in reference:
            np.testing.assert_array_equal(
                sequential_results[chr_name],
                reference[chr_name],
                err_msg=f"chr {chr_name} mismatch under buffer reuse",
            )

    def test_dict_materialization_aliases_all_to_last(self):
        """dict() on the raw yields aliases every entry to the final buffer.

        This is the hazard LocoKinshipStream.materialize() exists to avoid; the
        raw generator is documented as consume-once for exactly this reason.
        """
        S_full, S_chr, n_chr_filtered, K_loco_buf = _loco_fixtures()

        results = dict(
            _yield_loco_matrices(
                S_full,
                S_chr,
                n_chr_filtered,
                n_filtered=30,
                K_loco_buf=K_loco_buf,
            )
        )
        assert results["1"].base is results["3"].base or results["1"] is results["3"]
        assert np.array_equal(results["1"], results["3"])


class TestFallbackOrderingBiological:
    """Verify _yield_full_kinship_fallback produces biological order."""

    def test_biological_order(self):
        """Fallback chromosomes yielded in biological order."""
        n = 3
        S_full = np.eye(n, dtype=np.float64)
        chrs = ["10", "2", "1", "X"]

        results = list(_yield_full_kinship_fallback(S_full, chrs, n_filtered=10))
        yielded_order = [name for name, _ in results]
        assert yielded_order == ["1", "2", "10", "X"]
