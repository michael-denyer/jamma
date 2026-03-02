"""NumPy LOCO tests that run without JAX.

Kept in a separate file from test_loco.py because test_loco.py has
``pytest.importorskip("jax")`` at module level, which skips the entire
module when JAX is not installed. Tests here exercise the NumPy backend
only and must not import JAX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.loco import run_lmm_loco
from tests.conftest import load_phenotypes_from_fam

# Fixture with 3 chromosomes — required for LOCO (needs >1 chromosome to leave one out)
_LOCO_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gemma_loco"
_LOCO_BFILE = _LOCO_FIXTURE_ROOT / "test"


@pytest.mark.tier0
class TestComputeLocoKinshipNumpy:
    """Tests for KIN-01: compute_loco_kinship works without JAX."""

    def test_compute_loco_kinship_no_jax_import(self):
        """compute_loco_kinship can be imported and called without JAX."""
        from jamma.kinship import compute_loco_kinship

        rng = np.random.default_rng(42)
        n_samples, n_snps = 20, 30
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
        chrs = np.array(["1"] * 10 + ["2"] * 10 + ["3"] * 10)

        results = list(
            compute_loco_kinship(genotypes, chrs, batch_size=15, check_memory=False)
        )

        assert len(results) == 3
        for _chr_name, K_loco in results:
            assert K_loco.shape == (n_samples, n_samples)
            np.testing.assert_allclose(K_loco, K_loco.T, atol=1e-14)

    def test_loco_subtraction_identity(self):
        """K_loco_c = (S_full - S_c) / (p - p_c) identity holds."""
        from jamma.kinship import compute_loco_kinship
        from jamma.kinship.missing import impute_and_center

        rng = np.random.default_rng(99)
        n_samples, n_snps = 15, 20
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
        chrs = np.array(["1"] * 8 + ["2"] * 12)

        # Compute reference: full kinship from centered genotypes
        X = genotypes.copy().astype(np.float64)
        X_centered = impute_and_center(X)
        S_full = X_centered @ X_centered.T

        results = dict(
            compute_loco_kinship(
                genotypes.copy(), chrs, batch_size=100, check_memory=False
            )
        )

        for chr_name in ["1", "2"]:
            chr_mask = chrs == chr_name
            X_chr = impute_and_center(genotypes[:, chr_mask].copy().astype(np.float64))
            S_chr = X_chr @ X_chr.T
            p_loco = int(np.sum(~chr_mask))
            K_loco_expected = (S_full - S_chr) / p_loco
            np.testing.assert_allclose(
                results[chr_name], K_loco_expected, rtol=1e-12, atol=1e-14
            )

    def test_loco_kinship_with_nan_genotypes(self):
        """LOCO kinship handles NaN (missing) genotypes correctly."""
        from jamma.kinship import compute_loco_kinship

        rng = np.random.default_rng(77)
        n_samples, n_snps = 15, 20
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
        # Inject missing values
        genotypes[0, 3] = np.nan
        genotypes[5, 10] = np.nan
        genotypes[12, 0] = np.nan
        chrs = np.array(["1"] * 10 + ["2"] * 10)

        results = list(
            compute_loco_kinship(
                genotypes.copy(), chrs, batch_size=100, check_memory=False
            )
        )

        assert len(results) == 2
        for _chr_name, K_loco in results:
            assert K_loco.shape == (n_samples, n_samples)
            assert K_loco.dtype == np.float64
            np.testing.assert_allclose(K_loco, K_loco.T, atol=1e-14)
            assert np.all(np.isfinite(K_loco))


@pytest.mark.tier1
def test_loco_numpy_show_progress_true():
    """NumPy LOCO with show_progress=True completes without error.

    Exercises the tqdm progress bars and logger.info calls in
    _compute_loco_kinship_streaming_numpy and run_lmm_loco.
    Not marked @requires_jax — runs in NumPy-only CI.
    """
    if not _LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    results, n_tested = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        lmm_mode=1,
        show_progress=True,
        check_memory=False,
        backend="numpy",
    )

    assert n_tested > 0, "Expected at least one SNP to be tested"
    assert len(results) > 0, "Expected at least one association result"
