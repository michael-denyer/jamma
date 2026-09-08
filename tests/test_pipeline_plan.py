"""The private pipeline plan starts after the public validation boundary."""

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.association_plan import KinshipShape, plan_association
from jamma.pipeline_config import PipelineConfig
from jamma.pipeline_plan import (
    ComputedKinship,
    KinshipToEigen,
    LocoAnalysisPlan,
    ProvidedEigen,
    ProvidedKinship,
    StandardAnalysisPlan,
    resolve_analysis_plan,
)

pytestmark = pytest.mark.tier0


def _execution():  # type: ignore[no-untyped-def]
    return plan_association(10, 20, requested="numpy", n_cvt=1, lmm_mode=1)


def test_computed_kinship_carries_resolved_snp_choices(tmp_path: Path) -> None:
    ksnps = np.array([1, 3, 5])
    snps = np.array([2, 4])

    plan = resolve_analysis_plan(
        PipelineConfig(bfile=tmp_path / "study"),
        execution=_execution(),
        snps_indices=snps,
        ksnps_indices=ksnps,
    )

    assert isinstance(plan, StandardAnalysisPlan)
    assert isinstance(plan.eigen_source, KinshipToEigen)
    assert isinstance(plan.eigen_source.source, ComputedKinship)
    assert plan.eigen_source.source.ksnps_indices is ksnps
    assert plan.snps_indices is snps
    assert not plan.lmm.check_memory


def test_provided_kinship_is_a_distinct_source(tmp_path: Path) -> None:
    kinship = tmp_path / "kinship.npy"
    plan = resolve_analysis_plan(
        PipelineConfig(bfile=tmp_path / "study", kinship_file=kinship),
        execution=_execution(),
        snps_indices=None,
        ksnps_indices=np.array([1]),
    )

    assert isinstance(plan, StandardAnalysisPlan)
    assert isinstance(plan.eigen_source, KinshipToEigen)
    assert plan.eigen_source.source == ProvidedKinship(kinship)


def test_eigen_source_keeps_ignored_kinship_for_warning(tmp_path: Path) -> None:
    kinship = tmp_path / "kinship.npy"
    plan = resolve_analysis_plan(
        PipelineConfig(
            bfile=tmp_path / "study",
            kinship_file=kinship,
            eigenvalue_file=tmp_path / "eigenD.npy",
            eigenvector_file=tmp_path / "eigenU.npy",
        ),
        execution=_execution(),
        snps_indices=None,
        ksnps_indices=None,
    )

    assert isinstance(plan, StandardAnalysisPlan)
    assert plan.eigen_source == ProvidedEigen(
        tmp_path / "eigenD.npy", tmp_path / "eigenU.npy", kinship
    )


def test_loco_plan_owns_lmm_and_loco_configuration(tmp_path: Path) -> None:
    plan = resolve_analysis_plan(
        PipelineConfig(
            bfile=tmp_path / "study",
            loco=True,
            check_memory=True,
            save_kinship=True,
            output_dir=tmp_path,
        ),
        execution=_execution(),
        snps_indices=np.array([2]),
        ksnps_indices=np.array([3]),
    )

    assert isinstance(plan, LocoAnalysisPlan)
    assert plan.lmm.check_memory
    assert plan.loco.kinship_output_dir == tmp_path


def _subset_execution():  # type: ignore[no-untyped-def]
    return plan_association(10, 20, requested="numpy", n_input_samples=12)


def test_provided_eigen_materialises_no_kinship(tmp_path: Path) -> None:
    plan = resolve_analysis_plan(
        PipelineConfig(
            bfile=tmp_path / "study",
            eigenvalue_file=tmp_path / "eigenD.npy",
            eigenvector_file=tmp_path / "eigenU.npy",
        ),
        execution=_subset_execution(),
        snps_indices=None,
        ksnps_indices=None,
    )

    assert plan.execution.kinship is None
    with pytest.raises(ValueError, match="materialises no kinship"):
        _ = plan.execution.resolved_kinship


def test_provided_kinship_is_read_at_full_size(tmp_path: Path) -> None:
    plan = resolve_analysis_plan(
        PipelineConfig(bfile=tmp_path / "study", kinship_file=tmp_path / "k.npy"),
        execution=_subset_execution(),
        snps_indices=None,
        ksnps_indices=None,
    )

    assert plan.execution.kinship == KinshipShape(n_samples=12, loaded=True)


def test_computed_kinship_accumulates_over_analysed_samples(tmp_path: Path) -> None:
    plan = resolve_analysis_plan(
        PipelineConfig(bfile=tmp_path / "study"),
        execution=_subset_execution(),
        snps_indices=None,
        ksnps_indices=None,
    )

    assert plan.execution.kinship == KinshipShape(n_samples=10, loaded=False)


def test_saved_kinship_is_computed_at_full_size(tmp_path: Path) -> None:
    plan = resolve_analysis_plan(
        PipelineConfig(
            bfile=tmp_path / "study", save_kinship=True, output_dir=tmp_path
        ),
        execution=_subset_execution(),
        snps_indices=None,
        ksnps_indices=None,
    )

    assert plan.execution.kinship == KinshipShape(n_samples=12, loaded=False)


@pytest.mark.parametrize(("save", "expected"), [(False, 10), (True, 12)])
def test_loco_kinship_follows_the_same_rule(
    tmp_path: Path, save: bool, expected: int
) -> None:
    plan = resolve_analysis_plan(
        PipelineConfig(
            bfile=tmp_path / "study", loco=True, save_kinship=save, output_dir=tmp_path
        ),
        execution=_subset_execution(),
        snps_indices=None,
        ksnps_indices=None,
    )

    assert plan.execution.kinship == KinshipShape(n_samples=expected, loaded=False)
