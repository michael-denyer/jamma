"""Eigen inputs are demand-paged and released between acquisitions."""

from pathlib import Path

import numpy as np
import pytest

from jamma.core.eigen_plan import plan_eigen_driver
from jamma.io import read_fam_phenotypes
from jamma.lmm import loco_eigen
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files
from jamma.lmm.loco import LocoConfig, run_lmm_loco
from jamma.lmm.schema import LmmConfig
from tests.conftest import require_fixture
from tests.fakes.eigen_lifetime import (
    LifetimeCheckedEigenReader,
    LifetimeCheckedJlinalg,
)
from tests.fakes.jlinalg import use_fake_jlinalg
from tests.fixture_paths import LOCO

pytestmark = pytest.mark.tier0


def _run_loco(eigen_dir: Path | None = None, *, write_eigen: bool = False):
    """Run the real consumer, including its references between chromosomes."""
    require_fixture(LOCO.bed, LOCO.bim, LOCO.fam)
    return run_lmm_loco(
        LOCO.bfile,
        read_fam_phenotypes(LOCO.fam),
        config=LmmConfig(check_memory=False, show_progress=False),
        loco=LocoConfig(eigen_dir=eigen_dir, write_eigen=write_eigen),
    )


def test_direct_binary_eigen_inputs_are_read_only_mappings(tmp_path):
    d_path, u_path = tmp_path / "d.npy", tmp_path / "u.npy"
    np.save(d_path, np.arange(1.0, 5.0))
    np.save(u_path, np.eye(4))
    eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
    for array in (eigenvalues, eigenvectors):
        assert isinstance(array, np.memmap)
        assert not array.flags.writeable
    np.testing.assert_array_equal(eigenvectors, np.eye(4))


def test_computed_loco_releases_previous_matrix_before_decomposition(monkeypatch):
    # The documented jlinalg boundary performs real NumPy decomposition with
    # an independent U. The observer checks before allocating the next U.
    observer = LifetimeCheckedJlinalg()
    use_fake_jlinalg(monkeypatch, observer)
    result = _run_loco()
    assert result.n_tested > 0
    assert observer.previous is not None
    assert observer.previous() is None


def test_cached_loco_releases_previous_matrix_before_read(tmp_path, monkeypatch):
    fresh = _run_loco(tmp_path, write_eigen=True)
    reader = LifetimeCheckedEigenReader()
    monkeypatch.setattr(loco_eigen, "read_eigen_files", reader)
    cached = _run_loco(tmp_path)
    assert cached.n_tested == fresh.n_tested > 0
    assert reader.previous is not None
    assert reader.previous() is None


def test_reserved_dsyevr_plan_prices_the_decomposition_a_dsyevd_plan_cannot():
    """The plan the caller hands in is the one the run reserves against.

    At n=1000 the input and the eigenvectors cost 0.008 GB each and the driver
    decides the rest: DSYEVR's MRRR workspace is 26N+10N words, for a
    0.016288 GB peak, against DSYEVD's 2N^2+6N words and a 0.032088032 GB
    peak. Only the first fits a 0.02 GB ceiling, and the decomposition it
    returns reconstructs the input.
    """
    n = 1000
    kinship = np.full((n, n), 1.0 / n) + np.eye(n)
    dsyevr = plan_eigen_driver(
        n,
        256.0,
        has_dsyevd=True,
        has_dsyevr=True,
        no_vendor=False,
        inplace_eligible=True,
        budget_gb=0.02,
    )
    assert dsyevr.driver == "DSYEVR"
    assert dsyevr.required_gb == pytest.approx(0.016288)

    eigenvalues, eigenvectors = eigendecompose_kinship(
        kinship.copy(), eigen_plan=dsyevr, mem_budget=0.02
    )
    np.testing.assert_allclose(eigenvalues, [1.0] * (n - 1) + [2.0], rtol=1e-9)
    np.testing.assert_allclose(
        (eigenvectors * eigenvalues) @ eigenvectors.T, kinship, atol=1e-9
    )

    dsyevd = plan_eigen_driver(
        n,
        256.0,
        has_dsyevd=True,
        has_dsyevr=False,
        no_vendor=False,
        inplace_eligible=False,
    )
    assert dsyevd.driver == "DSYEVD"
    assert dsyevd.required_gb == pytest.approx(0.032088032)
    with pytest.raises(MemoryError, match=r"exceeds budget \(0\.02GB\)"):
        eigendecompose_kinship(kinship.copy(), eigen_plan=dsyevd, mem_budget=0.02)
