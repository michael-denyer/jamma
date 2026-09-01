"""Validated internal analysis choices for ``PipelineRunner``.

The public ``PipelineConfig`` stays flat for CLI and Python compatibility.
After its existing ordered validation succeeds, this module converts those
fields into the variants the runner can safely execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jamma.lmm.association_plan import ExecutableAssociationPlan
from jamma.lmm.loco_config import LocoConfig
from jamma.lmm.schema import LmmConfig
from jamma.pipeline_config import PipelineConfig


@dataclass(frozen=True, slots=True)
class ProvidedEigen:
    eigenvalue_file: Path
    eigenvector_file: Path
    ignored_kinship_file: Path | None


@dataclass(frozen=True, slots=True)
class ProvidedKinship:
    path: Path


@dataclass(frozen=True, slots=True)
class ComputedKinship:
    ksnps_indices: np.ndarray | None


KinshipSource = ProvidedKinship | ComputedKinship


@dataclass(frozen=True, slots=True)
class KinshipToEigen:
    source: KinshipSource
    write_eigen: bool


EigenSource = ProvidedEigen | KinshipToEigen


@dataclass(frozen=True, slots=True)
class StandardAnalysisPlan:
    execution: ExecutableAssociationPlan
    lmm: LmmConfig
    eigen_source: EigenSource
    snps_indices: np.ndarray | None


@dataclass(frozen=True, slots=True)
class LocoAnalysisPlan:
    execution: ExecutableAssociationPlan
    lmm: LmmConfig
    loco: LocoConfig


AnalysisPlan = StandardAnalysisPlan | LocoAnalysisPlan


def resolve_analysis_plan(
    config: PipelineConfig,
    *,
    execution: ExecutableAssociationPlan,
    snps_indices: np.ndarray | None,
    ksnps_indices: np.ndarray | None,
) -> AnalysisPlan:
    """Convert an already validated flat config into one executable variant."""
    if config.loco:
        return LocoAnalysisPlan(
            execution=execution,
            lmm=config.lmm_config(check_memory=config.check_memory),
            loco=LocoConfig(
                kinship_output_dir=config.output_dir if config.save_kinship else None,
                prefix=config.output_prefix,
                snps_indices=snps_indices,
                ksnps_indices=ksnps_indices,
                write_eigen=config.write_eigen,
                eigen_dir=config.eigen_dir,
                legacy_text=config.legacy_text,
            ),
        )

    if config.eigenvalue_file is not None:
        if config.eigenvector_file is None:
            raise RuntimeError(
                "resolve_analysis_plan requires validate_inputs() to pair eigen files"
            )
        eigen_source: EigenSource = ProvidedEigen(
            config.eigenvalue_file,
            config.eigenvector_file,
            config.kinship_file,
        )
    else:
        if config.eigenvector_file is not None:
            raise RuntimeError(
                "resolve_analysis_plan requires validate_inputs() to pair eigen files"
            )
        kinship_source: KinshipSource = (
            ProvidedKinship(config.kinship_file)
            if config.kinship_file is not None
            else ComputedKinship(ksnps_indices)
        )
        eigen_source = KinshipToEigen(kinship_source, config.write_eigen)

    return StandardAnalysisPlan(
        execution=execution,
        lmm=config.lmm_config(),
        eigen_source=eigen_source,
        snps_indices=snps_indices,
    )
