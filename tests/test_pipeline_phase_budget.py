"""Pipeline budgets include phases before association starts."""

import pytest

from jamma.core import memory
from jamma.lmm.association_plan import plan_association
from jamma.pipeline_config import PipelineConfig
from tests.conftest import preflight
from tests.fixture_paths import SYNTHETIC

pytestmark = pytest.mark.tier0


def test_batch_preflight_rejects_unaffordable_eigen_phase(monkeypatch):
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 256.0)
    config = PipelineConfig(bfile=SYNTHETIC.bfile, mem_budget=1.2)
    plan = plan_association(10_000, 100, requested="numpy", mem_budget=1.2)
    assert plan.price(eigen=None).total_peak_gb < 1.2
    with pytest.raises(MemoryError, match="exceeds"):
        preflight(config, plan)


def test_eigen_driver_selection_respects_user_ceiling():
    from jamma.core.eigen_plan import plan_eigen_driver

    plan = plan_eigen_driver(
        10_000,
        256.0,
        has_dsyevd=True,
        has_dsyevr=True,
        no_vendor=False,
        inplace_eligible=True,
        budget_gb=2.0,
    )
    assert plan.driver == "DSYEVR"
    # K and U at 0.8 GB each, plus DSYEVR's 26N+10N-word MRRR workspace.
    assert plan.required_gb == pytest.approx(1.60288)


def test_standalone_eigen_rejects_budget_before_decomposition(monkeypatch):
    import numpy as np

    from jamma.lmm.eigen import eigendecompose_kinship

    monkeypatch.setattr(memory, "available_ram_gb", lambda: 256.0)
    kinship = np.eye(100)
    with pytest.raises(MemoryError, match="exceeds"):
        eigendecompose_kinship(kinship, mem_budget=0.0001)
    np.testing.assert_array_equal(kinship, np.eye(100))


def test_saved_full_sample_kinship_is_in_batch_preflight(monkeypatch):
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 256.0)
    config = PipelineConfig(bfile=SYNTHETIC.bfile, mem_budget=1.2, save_kinship=True)
    plan = plan_association(
        100,
        100,
        n_input_samples=10_000,
        requested="numpy",
        mem_budget=1.2,
    )
    with pytest.raises(MemoryError, match="exceeds"):
        preflight(config, plan)


def test_loco_batch_selection_fits_user_budget():
    from jamma.kinship.loco import loco_retained_set, plan_loco_passes

    plan = plan_loco_passes(
        loco_retained_set(10_000, 10_000, 10_000),
        2.5,
        22,
        256.0,
        budget_gb=8.0,
        max_batch_chrs=None,
    )
    # S_full and K_loco_buf at 0.8 GB each, a 0.8 GB chunk buffer and the
    # 2.5 GB consumer are fixed at 4.9 GB, so the 8 GB ceiling buys three
    # more 0.8 GB S_chr accumulators and stops there.
    assert plan.batch_size == 3
    assert plan.single_pass is False
    assert plan.required_gb == pytest.approx(7.3)


def test_impossible_loco_budget_fails_before_genotype_statistics(tmp_path):
    """A header-only .bed makes the two orderings raise different exceptions.

    The metadata files are intact, so the run reaches its memory gate; the
    genotype bytes are absent, so the first statistics read cannot succeed.
    An impossible budget therefore has to surface as MemoryError, and the
    same call without a budget surfaces the truncated read.
    """
    import shutil

    from jamma.kinship import compute_loco_kinship_streaming
    from tests.conftest import require_fixture
    from tests.fixture_paths import LOCO

    require_fixture(LOCO.bfile.with_suffix(".bed"), LOCO.bfile.with_suffix(".fam"))
    bfile = tmp_path / "loco"
    for suffix in (".bim", ".fam"):
        shutil.copy(LOCO.bfile.with_suffix(suffix), bfile.with_suffix(suffix))
    bfile.with_suffix(".bed").write_bytes(b"\x6c\x1b\x01")

    with pytest.raises(MemoryError, match="exceeds budget"):
        compute_loco_kinship_streaming(
            bfile, mem_budget=1e-8, show_progress=False, consumer_gb=0.0
        )
    with pytest.raises(ValueError, match="Ill-formed BED file"):
        compute_loco_kinship_streaming(bfile, show_progress=False, consumer_gb=0.0)


def test_precomputed_eigen_streaming_does_not_reserve_decomposition(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(memory, "available_ram_gb", lambda: 256.0)
    config = PipelineConfig(
        bfile=SYNTHETIC.bfile,
        mem_budget=2.0,
        eigenvalue_file=Path("provided.eigenD.npy"),
        eigenvector_file=Path("provided.eigenU.npy"),
    )
    plan = plan_association(10_000, 100, requested="numpy-streaming", mem_budget=2.0)
    assert plan.price(eigen=None).total_peak_gb <= 2.0
    assert preflight(config, plan) is None


def test_loco_rechecks_capacity_after_genotype_statistics(monkeypatch):
    from jamma.kinship import compute_loco_kinship_streaming
    from tests.conftest import require_fixture
    from tests.fixture_paths import LOCO

    require_fixture(LOCO.bed, LOCO.bim, LOCO.fam)
    readings = iter((256.0, 0.0))
    monkeypatch.setattr(memory, "available_ram_gb", lambda: next(readings))
    with pytest.raises(MemoryError, match="LOCO kinship"):
        stream = compute_loco_kinship_streaming(
            LOCO.bfile, consumer_gb=0.0, show_progress=False
        )
        next(iter(stream))
