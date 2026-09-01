"""Self-tests for tests/builders.py and the fixture dataset table."""

from __future__ import annotations

import numpy as np
import pytest

from jamma.io import parse_fam_phenotype_column, read_fam_phenotypes
from jamma.lmm.pab import compute_Uab
from tests.builders import rotated_lmm_inputs, write_fam
from tests.fixture_paths import LOCO, MOUSE, SYNTHETIC, FixtureDataset, KinshipDataset

pytestmark = pytest.mark.tier0


class TestRotatedLmmInputs:
    def test_reproduces_the_inline_recipe_bit_for_bit(self):
        rng = np.random.default_rng(123)
        n_samples, n_snps = 80, 30
        eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
        UtW = np.ones((n_samples, 1))
        Uty = rng.standard_normal(n_samples)
        UtG = rng.standard_normal((n_samples, n_snps))

        d = rotated_lmm_inputs(n_samples, n_snps, seed=123)

        assert np.array_equal(d.eigenvalues, eigenvalues)
        assert np.array_equal(d.UtW, UtW)
        assert np.array_equal(d.Uty, Uty)
        assert np.array_equal(d.UtG, UtG)

    def test_covariates_are_random_for_ncvt_above_one(self):
        d = rotated_lmm_inputs(40, 5, n_cvt=3, seed=7)
        assert d.UtW.shape == (40, 3)
        assert d.n_cvt == 3
        assert not np.all(d.UtW == 1.0)

    def test_shapes_and_ordering(self):
        d = rotated_lmm_inputs(50, 10, seed=1, eig_range=(0.5, 3.0))
        assert (d.n_samples, d.n_snps) == (50, 10)
        assert np.all(np.diff(d.eigenvalues) >= 0)
        assert d.eigenvalues.min() >= 0.5
        assert d.eigenvalues.max() <= 3.0

    def test_uab_batch_matches_compute_uab_per_snp(self):
        d = rotated_lmm_inputs(30, 4, n_cvt=2, seed=9)
        uab = d.uab_batch()
        assert uab.shape == (4, 30, 10)
        for i in range(4):
            np.testing.assert_array_equal(
                uab[i], compute_Uab(d.UtW, d.Uty, d.UtG[:, i])
            )


class TestWriteFam:
    def test_round_trips_through_read_fam_phenotypes(self, tmp_path):
        path = write_fam(
            tmp_path / "t.fam", [1.0, 2.5, "NA", "-9"], [4.0, 5.0, 6.0, 7.0]
        )

        np.testing.assert_array_equal(
            read_fam_phenotypes(path, 1), [1.0, 2.5, np.nan, np.nan]
        )
        np.testing.assert_array_equal(
            read_fam_phenotypes(path, 2), [4.0, 5.0, 6.0, 7.0]
        )

    def test_missing_at_blanks_every_column(self, tmp_path):
        path = write_fam(
            tmp_path / "t.fam", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], missing_at={1}
        )

        lines = path.read_text().splitlines()
        assert lines[1] == "FAM001\tIND001\t0\t0\t0\tNA\tNA"
        assert lines[2] == "FAM002\tIND002\t0\t0\t0\t3.0\t6.0"


class TestReadFamPhenotypes:
    def test_column_beyond_file_is_a_value_error(self, tmp_path):
        path = write_fam(tmp_path / "t.fam", [1.0, 2.0])
        with pytest.raises(ValueError, match="phenotype column 2 exceeds"):
            read_fam_phenotypes(path, 2)

    def test_parse_does_not_mutate_the_table(self):
        fam = np.array([["F", "I", "0", "0", "0", "NA"]], dtype=str)
        parse_fam_phenotype_column(fam, 1)
        assert fam[0, 5] == "NA"

    def test_synthetic_fixture_first_column(self):
        pheno = read_fam_phenotypes(SYNTHETIC.fam)
        assert pheno.shape[0] > 0
        assert np.isfinite(pheno).any()


class TestFixtureDatasets:
    @pytest.mark.parametrize(
        "dataset", [SYNTHETIC, MOUSE, LOCO], ids=["synthetic", "mouse", "loco"]
    )
    def test_every_named_path_exists(self, dataset: FixtureDataset):
        paths = [dataset.bed, dataset.bim, dataset.fam, *dataset.assoc.values()]
        if isinstance(dataset, KinshipDataset):
            paths += [dataset.kinship, dataset.covariates]
        missing = [p for p in paths if not p.is_file()]
        assert not missing, missing

    def test_ref_names_the_dataset_on_a_missing_run(self):
        with pytest.raises(KeyError, match="gemma_loco has no recorded 'wald'"):
            LOCO.ref("wald")
