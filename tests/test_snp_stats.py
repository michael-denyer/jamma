"""Tests for jlinalg compute_snp_stats_chunk C kernel and fallback."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from jamma.jlinalg import HAS_C_EXTENSION, compute_snp_stats_chunk

pytestmark = pytest.mark.tier0


def _python_reference(data, compute_hwe=False):
    """Pure-Python reference implementation for validation.

    Always computes in float64 to match the C kernel's double-precision
    accumulation (C kernel always accumulates in double regardless of input dtype).
    """
    import warnings

    data64 = data.astype(np.float64)
    mc = np.sum(np.isnan(data64), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(data64, axis=0)
        v = np.nanvar(data64, axis=0)
    m = np.nan_to_num(m, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    result = {"means": m, "miss_counts": mc, "vars": v}
    if compute_hwe:
        valid = ~np.isnan(data)
        result["n_aa"] = np.sum((data == 0) & valid, axis=0)
        result["n_ab"] = np.sum((data == 1) & valid, axis=0)
        result["n_bb"] = np.sum((data == 2) & valid, axis=0)
    return result


class TestSnpStatsC:
    """Test C kernel parity with Python reference."""

    @pytest.fixture
    def genotypes_f32(self):
        """Known genotype chunk: 5 samples x 4 SNPs with NaN."""
        return np.array(
            [
                [0.0, 1.0, 2.0, np.nan],
                [1.0, 1.0, 0.0, np.nan],
                [2.0, 0.0, 1.0, np.nan],
                [0.0, 2.0, np.nan, np.nan],
                [1.0, 1.0, 1.0, np.nan],
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def genotypes_f64(self, genotypes_f32):
        return genotypes_f32.astype(np.float64)

    def _call(self, data, compute_hwe=False):
        n_snps = data.shape[1]
        means = np.zeros(n_snps, dtype=np.float64)
        miss = np.zeros(n_snps, dtype=np.intp)
        vari = np.zeros(n_snps, dtype=np.float64)
        naa = nab = nbb = None
        if compute_hwe:
            naa = np.zeros(n_snps, dtype=np.int64)
            nab = np.zeros(n_snps, dtype=np.int64)
            nbb = np.zeros(n_snps, dtype=np.int64)
        compute_snp_stats_chunk(data, means, miss, vari, naa, nab, nbb)
        return means, miss, vari, naa, nab, nbb

    def test_float32_parity(self, genotypes_f32):
        ref = _python_reference(genotypes_f32)
        means, miss, vari, _, _, _ = self._call(genotypes_f32)
        assert_allclose(means, ref["means"], rtol=1e-14)
        np.testing.assert_array_equal(miss, ref["miss_counts"])
        assert_allclose(vari, ref["vars"], rtol=1e-14)

    def test_float64_parity(self, genotypes_f64):
        ref = _python_reference(genotypes_f64)
        means, miss, vari, _, _, _ = self._call(genotypes_f64)
        assert_allclose(means, ref["means"], rtol=1e-14)
        np.testing.assert_array_equal(miss, ref["miss_counts"])
        assert_allclose(vari, ref["vars"], rtol=1e-14)

    def test_dtype_parity(self, genotypes_f32, genotypes_f64):
        m32, mc32, v32, _, _, _ = self._call(genotypes_f32)
        m64, mc64, v64, _, _, _ = self._call(genotypes_f64)
        assert_allclose(m32, m64, rtol=1e-14)
        np.testing.assert_array_equal(mc32, mc64)
        assert_allclose(v32, v64, rtol=1e-14)

    def test_all_nan(self):
        data = np.full((10, 3), np.nan, dtype=np.float32)
        means, miss, vari, _, _, _ = self._call(data)
        assert_allclose(means, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(miss, [10, 10, 10])
        assert_allclose(vari, [0.0, 0.0, 0.0])

    def test_hwe_counts(self, genotypes_f32):
        ref = _python_reference(genotypes_f32, compute_hwe=True)
        _, _, _, naa, nab, nbb = self._call(genotypes_f32, compute_hwe=True)
        np.testing.assert_array_equal(naa, ref["n_aa"])
        np.testing.assert_array_equal(nab, ref["n_ab"])
        np.testing.assert_array_equal(nbb, ref["n_bb"])
        # HWE counts sum to n_valid
        n_valid = genotypes_f32.shape[0] - np.sum(np.isnan(genotypes_f32), axis=0)
        np.testing.assert_array_equal(naa + nab + nbb, n_valid)

    def test_rectangular_chunk(self):
        """Non-square chunk catches stride bugs."""
        rng = np.random.default_rng(42)
        data = rng.choice(
            [0.0, 1.0, 2.0, np.nan], size=(7, 13), p=[0.3, 0.4, 0.2, 0.1]
        ).astype(np.float32)
        ref = _python_reference(data, compute_hwe=True)
        means, miss, vari, naa, nab, nbb = self._call(data, compute_hwe=True)
        assert_allclose(means, ref["means"], rtol=1e-14)
        np.testing.assert_array_equal(miss, ref["miss_counts"])
        assert_allclose(vari, ref["vars"], rtol=1e-14)
        np.testing.assert_array_equal(naa, ref["n_aa"])
        np.testing.assert_array_equal(nab, ref["n_ab"])
        np.testing.assert_array_equal(nbb, ref["n_bb"])

    def test_no_hwe(self, genotypes_f32):
        """Passing None for HWE arrays works without error."""
        means, miss, vari, naa, nab, nbb = self._call(genotypes_f32, compute_hwe=False)
        assert naa is None
        assert nab is None
        assert nbb is None
        assert means.shape == (4,)

    def test_c_extension_loaded(self):
        """Verify C extension is available (not fallback)."""
        assert HAS_C_EXTENSION, "C extension not loaded; tests need compiled _jlinalg"

    def test_single_sample(self):
        """Edge case: single sample per SNP."""
        data = np.array([[1.0, 0.0, 2.0]], dtype=np.float32)
        means, miss, vari, _, _, _ = self._call(data)
        assert_allclose(means, [1.0, 0.0, 2.0])
        assert_allclose(vari, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(miss, [0, 0, 0])

    def test_fortran_order_input(self):
        """Non-contiguous (Fortran-order) input handled by C wrapper."""
        data_c = np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        data_f = np.asfortranarray(data_c)
        assert not data_f.flags["C_CONTIGUOUS"]
        means_c, miss_c, var_c, _, _, _ = self._call(data_c)
        means_f, miss_f, var_f, _, _, _ = self._call(data_f)
        assert_allclose(means_c, means_f)
        np.testing.assert_array_equal(miss_c, miss_f)
        assert_allclose(var_c, var_f)

    def test_single_snp(self):
        """Edge case: single SNP column."""
        data = np.array([[0.0], [1.0], [2.0], [np.nan]], dtype=np.float32)
        means, miss, vari, _, _, _ = self._call(data)
        assert_allclose(means, [1.0])
        assert miss[0] == 1
