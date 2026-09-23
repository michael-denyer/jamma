"""Pipeline budgets include phases before association starts."""

import shutil
from pathlib import Path

import pytest

from jamma.core import memory
from jamma.lmm.association_plan import plan_association
from jamma.lmm.schema import LmmConfig
from jamma.pipeline_config import PipelineConfig
from tests.conftest import preflight
from tests.fixture_paths import SYNTHETIC, FixtureDataset

pytestmark = pytest.mark.tier0


def _header_only_bed(tmp_path: Path, dataset: FixtureDataset) -> Path:
    bfile = tmp_path / "header_only"
    for suffix in (".bim", ".fam"):
        shutil.copy(dataset.bfile.with_suffix(suffix), bfile.with_suffix(suffix))
    bfile.with_suffix(".bed").write_bytes(b"\x6c\x1b\x01")
    return bfile


def test_batch_preflight_rejects_unaffordable_eigen_phase(monkeypatch):
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 256.0)
    config = PipelineConfig(bfile=SYNTHETIC.bfile, mem_budget=1.2)
    plan = plan_association(
        10_000, 100, config=LmmConfig(mem_budget=1.2), backend="numpy"
    )
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
        forced_numpy=False,
        inplace_blocker=None,
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
    # The full saved K alone needs 0.8 GB. Only 100 SNPs exist, so the
    # preprocessing quote must not assume a full 10,000-column block.
    config = PipelineConfig(bfile=SYNTHETIC.bfile, mem_budget=0.5, save_kinship=True)
    plan = plan_association(
        100,
        100,
        config=LmmConfig(mem_budget=0.5),
        backend="numpy",
        n_input_samples=10_000,
    )
    preflight(PipelineConfig(bfile=SYNTHETIC.bfile, mem_budget=0.5), plan)
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
    from jamma.kinship import compute_loco_kinship_streaming
    from tests.conftest import require_fixture
    from tests.fixture_paths import LOCO

    require_fixture(LOCO.bfile.with_suffix(".bed"), LOCO.bfile.with_suffix(".fam"))
    bfile = _header_only_bed(tmp_path, LOCO)

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
    plan = plan_association(
        10_000, 100, config=LmmConfig(mem_budget=2.0), backend="numpy-streaming"
    )
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


def test_impossible_kinship_budget_fails_before_genotype_read(tmp_path):
    from jamma.kinship import compute_kinship_streaming
    from tests.conftest import require_fixture

    require_fixture(SYNTHETIC.bim, SYNTHETIC.fam)
    bfile = _header_only_bed(tmp_path, SYNTHETIC)

    with pytest.raises(MemoryError, match="exceeds budget"):
        compute_kinship_streaming(bfile, mem_budget=1e-8, show_progress=False)
    with pytest.raises(ValueError, match="Ill-formed BED file"):
        compute_kinship_streaming(bfile, show_progress=False)


def test_gk_budget_gates_kinship_accumulation(tmp_path):
    from jamma.pipeline_kinship import compute_kinship
    from tests.conftest import require_fixture

    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)
    config = PipelineConfig(
        bfile=SYNTHETIC.bfile,
        output_dir=tmp_path,
        mem_budget=1e-8,
        show_progress=False,
    )
    with pytest.raises(MemoryError, match=r"kinship accumulation.*exceeds budget"):
        compute_kinship(config, mode=1)
    assert not (tmp_path / "result.cXX.npy").exists()


def test_gk_eigen_budget_gates_eigendecomposition(tmp_path):
    import numpy as np
    from bed_reader import to_bed

    from jamma.core.eigen_plan import dsyevr_peak_gb
    from jamma.core.memory import estimate_kinship_memory
    from jamma.pipeline_kinship import compute_kinship

    n_samples, n_snps = 2000, 10
    values = np.random.default_rng(1).integers(0, 3, (n_samples, n_snps))
    bfile = tmp_path / "small"
    to_bed(bfile.with_suffix(".bed"), values.astype(float))
    kinship_gb = estimate_kinship_memory(
        n_input_samples=n_samples,
        n_output_samples=n_samples,
        n_snps=n_snps,
        chunk_size=10_000,
    )
    budget_gb = (kinship_gb + dsyevr_peak_gb(n_samples)) / 2
    assert kinship_gb < budget_gb < dsyevr_peak_gb(n_samples)
    config = PipelineConfig(
        bfile=bfile,
        output_dir=tmp_path,
        write_eigen=True,
        mem_budget=budget_gb,
        show_progress=False,
    )
    with pytest.raises(MemoryError, match=r"eigendecomposition.*exceeds budget"):
        compute_kinship(config, mode=1)
    assert (tmp_path / "result.cXX.npy").exists()
    assert not list(tmp_path.glob("*.eigenD.npy"))
