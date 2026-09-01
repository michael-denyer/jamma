"""Fused Score/LRT dispatch tests for the streaming LMM runner."""

from __future__ import annotations

import numpy as np
import pytest

from jamma.io import load_plink_binary, read_fam_phenotypes
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.runner_numpy_streaming import run_lmm_association_numpy_streaming
from jamma.lmm.schema import LmmConfig
from tests.conftest import requires_c
from tests.fixture_paths import SYNTHETIC
from tests.lmm_accel._helpers import assert_fused_matches_reference


@pytest.fixture
def synthetic_eigen():
    plink = load_plink_binary(SYNTHETIC.bfile)
    kinship = read_kinship_matrix(SYNTHETIC.kinship)
    phenotypes = read_fam_phenotypes(SYNTHETIC.fam)
    valid_mask = ~np.isnan(phenotypes)
    eigenvalues, eigenvectors = np.linalg.eigh(kinship[np.ix_(valid_mask, valid_mask)])
    return plink, kinship, phenotypes, eigenvalues, eigenvectors


def _run(eigenvalues, eigenvectors, phenotypes, lmm_mode):
    return run_lmm_association_numpy_streaming(
        bed_path=SYNTHETIC.bfile,
        phenotypes=phenotypes,
        kinship=None,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors.copy(),
        config=LmmConfig(lmm_mode=lmm_mode, show_progress=False, check_memory=False),
        chunk_size=200,
    )


@pytest.mark.tier0
@requires_c
class TestStreamingFusedScoreDispatch:
    def test_streaming_fused_score_matches_split(self, synthetic_eigen):
        _plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen
        assert_fused_matches_reference(
            lambda: _run(eigenvalues, eigenvectors, phenotypes, lmm_mode=3),
            fields={"p_score": 1e-8},
            label=" Score WS (streaming)",
        )


@pytest.mark.tier0
@requires_c
class TestStreamingFusedLrtDispatch:
    def test_streaming_fused_lrt_matches_split(self, synthetic_eigen):
        _plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen
        assert_fused_matches_reference(
            lambda: _run(eigenvalues, eigenvectors, phenotypes, lmm_mode=2),
            fields={"p_lrt": 5e-5, "l_mle": 5e-5},
            label=" LRT WS (streaming)",
        )
