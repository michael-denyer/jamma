"""Self-tests for tests/builders.py and the fixture dataset table."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from jamma.io import parse_fam_phenotype_column, read_fam_phenotypes
from jamma.lmm.pab import compute_Uab
from tests.builders import (
    covariate_lmm_inputs,
    gram_uab_batch,
    make_runner_synthetic_data,
    rotated_lmm_inputs,
    write_fam,
)
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


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()[:32]


# Digests of the ad-hoc constructions these builders replaced, taken at
# d867e2d3: _build_synthetic_covariate_data and the lmm_accel split_wald_data
# and synthetic_wald_data fixtures. Seeded tests read these arrays, so a
# builder that drifts by one bit fails here before it moves a tolerance test.
_INPUT_PINS = [
    pytest.param(
        lambda: covariate_lmm_inputs(n_cvt=2, seed=42),
        {
            "eigenvalues": "0c0028094c2dcc691a2f9c5f0e2d8c7e",
            "UtW": "a687661e66ab5c62f8e46fb8794669df",
            "Uty": "9fd56376eeb7fd4c49f550322f9c4e37",
            "UtG": "d71f0d9e42973293c6186d07990cc991",
        },
        id="cov2",
    ),
    pytest.param(
        lambda: covariate_lmm_inputs(n_cvt=4, seed=99),
        {
            "eigenvalues": "34420fd6e0263ab1165560fd5f03f73c",
            "UtW": "ebdc2f076b5c9de76a5a7e6100ae4c3a",
            "Uty": "eed43517f40182f008fb1b673a493ff1",
            "UtG": "443b7705d3323765f4f1674feabeb328",
        },
        id="cov4",
    ),
    pytest.param(
        lambda: rotated_lmm_inputs(200, 50, eig_range=(0.1, 2.0), intercept=False),
        {
            "eigenvalues": "2eda09241ce0452255950f629a038e92",
            "UtW": "61c54a0602bcbd562d0f70024708275f",
            "Uty": "8921c9425356085ede38930444928283",
            "UtG": "552133c65030a80c9f7f26e4d6e72255",
        },
        id="split",
    ),
]


class TestBuilderBytePins:
    @pytest.mark.parametrize(("build", "digests"), _INPUT_PINS)
    def test_lmm_inputs_match_the_replaced_recipe(self, build, digests):
        inputs = build()
        assert {k: _sha256(getattr(inputs, k)) for k in digests} == digests

    def test_gram_uab_batch_matches_the_replaced_fixture(self):
        eigenvalues, uab_batch = gram_uab_batch()
        assert _sha256(eigenvalues) == "2eda09241ce0452255950f629a038e92"
        assert _sha256(uab_batch) == "b9a0bf4c0c0b01b7c2f88fcebc22c493"

    def test_covariate_uab_batch_matches_the_replaced_recipe(self):
        uab_batch = covariate_lmm_inputs(n_cvt=2, seed=42).uab_batch()
        assert _sha256(uab_batch) == "71f7328fc08c5860f9641e0a686352b0"

    def test_runner_data_matches_the_moved_builder(self):
        genotypes, phenotypes, _, snp_info = make_runner_synthetic_data()
        assert _sha256(genotypes) == "6b96f49bb7b24df50fe909a81c860a1f"
        assert _sha256(phenotypes) == "7a847cdbb62cae6b94f65c56ae2f0ea0"
        assert len(snp_info) == 50


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
