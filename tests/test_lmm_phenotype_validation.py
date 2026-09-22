"""Every public LMM association entry rejects a non-finite phenotype.

The batch, streaming and LOCO entries all pass through ``compute_valid_mask``,
so these three tests pin one rule at each of its public callers.
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.io import read_fam_phenotypes
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm import (
    run_lmm_association_numpy,
    run_lmm_association_numpy_streaming,
)
from jamma.lmm.loco import run_lmm_loco
from jamma.lmm.schema import LmmConfig
from tests.conftest import make_runner_synthetic_data, require_fixture
from tests.fixture_paths import LOCO, SYNTHETIC

_QUIET = LmmConfig(check_memory=False, show_progress=False)


@pytest.mark.tier0
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_batch_rejects_non_finite_phenotype(value: float) -> None:
    genotypes, phenotypes, kinship, snp_info = make_runner_synthetic_data()
    phenotypes[7] = value
    with pytest.raises(ValueError, match="only finite values"):
        run_lmm_association_numpy(
            genotypes, phenotypes, kinship, snp_info, config=_QUIET
        )


@pytest.mark.tier1
def test_streaming_rejects_non_finite_phenotype() -> None:
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam, SYNTHETIC.kinship)
    phenotypes = read_fam_phenotypes(SYNTHETIC.fam)
    phenotypes[3] = np.inf
    kinship = read_kinship_matrix(SYNTHETIC.kinship)
    with pytest.raises(ValueError, match="only finite values"):
        run_lmm_association_numpy_streaming(
            SYNTHETIC.bfile, phenotypes, kinship, config=_QUIET
        )


@pytest.mark.tier1
def test_loco_rejects_non_finite_phenotype() -> None:
    require_fixture(LOCO.bed, LOCO.bim, LOCO.fam)
    phenotypes = read_fam_phenotypes(LOCO.fam)
    phenotypes[3] = np.inf
    with pytest.raises(ValueError, match="only finite values"):
        run_lmm_loco(LOCO.bfile, phenotypes, config=_QUIET)
