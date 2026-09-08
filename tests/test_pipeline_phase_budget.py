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
    assert plan.required_gb <= 2.0


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
    from jamma.kinship.loco import plan_loco_passes

    plan = plan_loco_passes(
        10_000,
        10_000,
        22,
        10_000,
        256.0,
        max_batch_chrs=None,
        budget_gb=8.0,
        consumer_peak_gb=2.5,
    )
    assert 1 <= plan.batch_size < 22
    assert plan.required_gb <= 8.0


def test_impossible_loco_budget_fails_before_genotype_statistics():
    import sys

    from jamma.io.plink import stream_genotype_chunks
    from jamma.kinship import compute_loco_kinship_streaming
    from tests.conftest import require_fixture
    from tests.fixture_paths import LOCO

    require_fixture(LOCO.bfile.with_suffix(".bed"))
    reads = []

    def observe(frame, event, result):
        if event == "call" and frame.f_code is stream_genotype_chunks.__code__:
            reads.append(True)

    old_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        with pytest.raises(MemoryError, match="exceeds"):
            compute_loco_kinship_streaming(
                LOCO.bfile, mem_budget=1e-8, show_progress=False
            )
    finally:
        sys.setprofile(old_profile)
    assert not reads


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
