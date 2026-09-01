"""The private pipeline plan starts after the public validation boundary."""

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.association_plan import plan_association
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
