"""Eigen inputs are demand-paged and released between acquisitions."""

import numpy as np
import pytest

from jamma.lmm.eigen_io import read_eigen_files

pytestmark = pytest.mark.tier0


def test_direct_binary_eigen_inputs_are_read_only_mappings(tmp_path):
    d_path, u_path = tmp_path / "d.npy", tmp_path / "u.npy"
    np.save(d_path, np.arange(1.0, 5.0))
    np.save(u_path, np.eye(4))
    eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
    for array in (eigenvalues, eigenvectors):
        assert isinstance(array, np.memmap)
        assert not array.flags.writeable
    np.testing.assert_array_equal(eigenvectors, np.eye(4))


def test_cached_loco_releases_previous_matrix_before_next_read(tmp_path):
    import sys
    import weakref

    from jamma.io import read_fam_phenotypes
    from jamma.lmm.loco import LocoConfig, run_lmm_loco
    from jamma.lmm.schema import LmmConfig
    from tests.conftest import require_fixture
    from tests.fixture_paths import LOCO

    require_fixture(LOCO.bfile.with_suffix(".bed"), LOCO.bfile.with_suffix(".fam"))
    phenotypes = read_fam_phenotypes(LOCO.bfile.with_suffix(".fam"))
    config = LmmConfig(check_memory=False, show_progress=False)
    fresh = run_lmm_loco(
        LOCO.bfile,
        phenotypes,
        config=config,
        loco=LocoConfig(eigen_dir=tmp_path, write_eigen=True),
    )
    previous = None
    alive_at_acquisition = []

    def observe(frame, event, result):
        nonlocal previous
        if frame.f_code is read_eigen_files.__code__:
            if event == "call" and previous is not None:
                alive_at_acquisition.append(previous() is not None)
            elif event == "return" and result is not None:
                previous = weakref.ref(result[1])

    old_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        cached = run_lmm_loco(
            LOCO.bfile, phenotypes, config=config, loco=LocoConfig(eigen_dir=tmp_path)
        )
    finally:
        sys.setprofile(old_profile)
    assert len(alive_at_acquisition) >= 1
    assert not any(alive_at_acquisition)
    assert cached.n_tested == fresh.n_tested > 0


def test_computed_loco_executes_the_reserved_eigen_driver(tmp_path, monkeypatch):
    import sys
    import weakref

    from jamma.io import read_fam_phenotypes
    from jamma.lmm.eigen import eigendecompose_kinship
    from jamma.lmm.loco import run_lmm_loco
    from jamma.lmm.schema import LmmConfig
    from tests.conftest import require_fixture
    from tests.fixture_paths import LOCO

    require_fixture(LOCO.bfile.with_suffix(".bed"), LOCO.bfile.with_suffix(".fam"))
    # Out-of-place eigenvectors have a lifetime independent of the reusable K.
    monkeypatch.setenv("JLINALG_NO_VENDOR_LAPACK", "1")
    plans = []
    previous = None
    alive_at_acquisition = []

    def observe(frame, event, result):
        nonlocal previous
        if frame.f_code is eigendecompose_kinship.__code__:
            if event == "call":
                plans.append(frame.f_locals.get("eigen_plan"))
                if previous is not None:
                    alive_at_acquisition.append(previous() is not None)
            elif event == "return" and result is not None:
                previous = weakref.ref(result[1])

    old_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        run_lmm_loco(
            LOCO.bfile,
            read_fam_phenotypes(LOCO.bfile.with_suffix(".fam")),
            config=LmmConfig(show_progress=False),
            output_path=tmp_path / "result.assoc.txt",
        )
    finally:
        sys.setprofile(old_profile)
    assert len(plans) > 1
    assert all(plan is not None for plan in plans)
    assert alive_at_acquisition
    assert not any(alive_at_acquisition)
