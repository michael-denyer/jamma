"""Eigen inputs are demand-paged and released between acquisitions."""

import weakref
from pathlib import Path

import numpy as np
import pytest

from jamma.core.eigen_plan import plan_eigen_driver
from jamma.io.plink import get_plink_metadata, partitions_from_metadata
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files
from jamma.lmm.loco_config import LocoConfig
from jamma.lmm.loco_eigen import eigen_pairs_for
from jamma.utils import chr_sort_key
from tests.conftest import require_fixture
from tests.fixture_paths import LOCO

pytestmark = pytest.mark.tier0


def _loco_eigen_pairs(eigen_dir: Path | None, *, write_eigen: bool = False):
    """The LOCO fixture's eigenpair generator, one chromosome per ``next``.

    ``eigen_dir`` with ``write_eigen`` false and a valid manifest reads the
    cache; anything else streams kinship and eigendecomposes.

    The plan is out-of-place on purpose. In-place DSYEVD returns the streamer's
    reused kinship buffer as U, so every chromosome yields the same object and
    no lifetime is observable; out-of-place is the case where one chromosome's
    eigenvectors can outlive their turn.
    """
    require_fixture(LOCO.bfile.with_suffix(".bed"), LOCO.bfile.with_suffix(".fam"))
    meta = get_plink_metadata(LOCO.bfile)
    partitions = partitions_from_metadata(meta)
    return eigen_pairs_for(
        LOCO.bfile,
        sorted(partitions, key=chr_sort_key),
        loco=LocoConfig(eigen_dir=eigen_dir, write_eigen=write_eigen),
        maf_threshold=0.0,
        miss_threshold=1.0,
        valid_mask=np.ones(meta.n_samples, dtype=bool),
        partitions=partitions,
        check_memory=False,
        show_progress=False,
        eigen_plan=plan_eigen_driver(
            meta.n_samples,
            256.0,
            has_dsyevd=True,
            has_dsyevr=True,
            no_vendor=False,
            inplace_eligible=False,
        ),
    ).pairs


def test_direct_binary_eigen_inputs_are_read_only_mappings(tmp_path):
    d_path, u_path = tmp_path / "d.npy", tmp_path / "u.npy"
    np.save(d_path, np.arange(1.0, 5.0))
    np.save(u_path, np.eye(4))
    eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
    for array in (eigenvalues, eigenvectors):
        assert isinstance(array, np.memmap)
        assert not array.flags.writeable
    np.testing.assert_array_equal(eigenvectors, np.eye(4))


def test_computed_eigenpairs_drop_each_chromosome_before_yielding_the_next():
    pairs = _loco_eigen_pairs(None)
    try:
        first_chr, _, first_u = next(pairs)
        released = weakref.ref(first_u)
        del first_u
        second_chr, _, _ = next(pairs)
    finally:
        pairs.close()
    assert second_chr != first_chr
    assert released() is None


def test_cached_eigenpairs_drop_each_chromosome_before_yielding_the_next(tmp_path):
    written = _loco_eigen_pairs(tmp_path, write_eigen=True)
    try:
        chr_names = [name for name, _, _ in written]
    finally:
        written.close()

    pairs = _loco_eigen_pairs(tmp_path)
    try:
        first_chr, _, first_u = next(pairs)
        released = weakref.ref(first_u)
        del first_u
        second_chr, _, _ = next(pairs)
    finally:
        pairs.close()
    assert [first_chr, second_chr] == chr_names[:2]
    assert released() is None


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
