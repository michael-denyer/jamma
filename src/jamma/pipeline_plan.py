"""Validated internal analysis choices for ``PipelineRunner``.

The public ``PipelineConfig`` stays flat for CLI and Python compatibility.
``PipelineConfig.source()`` parses its kinship and eigen fields, and this
module attaches the index and shape data the runner executes with.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from jamma.lmm.association_plan import ExecutableAssociationPlan, KinshipShape
from jamma.lmm.loco_config import LocoConfig
from jamma.lmm.schema import LmmConfig
from jamma.pipeline_config import (
    GenotypeKinship,
    LocoKinship,
    PipelineConfig,
    ProvidedEigen,
    ProvidedKinship,
)


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
    """Convert the flat config into one executable variant.

    The kinship shape is resolved here, once, from the same source variants,
    so the memory quote and the runtime load read one rule.
    """
    source = config.source()
    if isinstance(source, LocoKinship):
        kinship = KinshipShape.resolve(
            execution.n_samples,
            execution.n_input_samples,
            loaded=False,
            saved=config.save_kinship,
        )
        return LocoAnalysisPlan(
            execution=replace(execution, kinship=kinship),
            lmm=config.lmm_config(check_memory=config.check_memory),
            loco=LocoConfig(
                kinship_output_dir=config.output_dir if config.save_kinship else None,
                prefix=config.output_prefix,
                snps_indices=snps_indices,
                ksnps_indices=ksnps_indices,
                write_eigen=config.write_eigen,
                eigen_dir=source.eigen_dir,
                legacy_text=config.legacy_text,
                info_threshold=config.info_threshold,
            ),
        )

    if isinstance(source, ProvidedEigen):
        eigen_source: EigenSource = source
        kinship = None
    else:
        kinship_source: KinshipSource = (
            ComputedKinship(ksnps_indices)
            if isinstance(source, GenotypeKinship)
            else source
        )
        eigen_source = KinshipToEigen(kinship_source, config.write_eigen)
        kinship = KinshipShape.resolve(
            execution.n_samples,
            execution.n_input_samples,
            loaded=isinstance(kinship_source, ProvidedKinship),
            saved=config.save_kinship,
        )

    return StandardAnalysisPlan(
        execution=replace(execution, kinship=kinship),
        lmm=config.lmm_config(),
        eigen_source=eigen_source,
        snps_indices=snps_indices,
    )
