"""Validated internal analysis choices for ``PipelineRunner``.

The public ``PipelineConfig`` stays flat for CLI and Python compatibility.
After its existing ordered validation succeeds, this module converts those
fields into the variants the runner can safely execute.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from jamma.lmm.association_plan import ExecutableAssociationPlan, KinshipShape
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


def resolve_kinship_source(
    kinship_file: Path | None, ksnps_indices: np.ndarray | None
) -> KinshipSource:
    """Derive the kinship source from config fields, in exactly one place."""
    if kinship_file is not None:
        return ProvidedKinship(kinship_file)
    return ComputedKinship(ksnps_indices)


def resolve_analysis_plan(
    config: PipelineConfig,
    *,
    execution: ExecutableAssociationPlan,
    snps_indices: np.ndarray | None,
    ksnps_indices: np.ndarray | None,
) -> AnalysisPlan:
    """Convert an already validated flat config into one executable variant.

    The kinship shape is resolved here, once, from the same source variants,
    so the memory quote and the runtime load read one rule.
    """
    if config.loco:
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
        kinship = None
    else:
        if config.eigenvector_file is not None:
            raise RuntimeError(
                "resolve_analysis_plan requires validate_inputs() to pair eigen files"
            )
        source = resolve_kinship_source(config.kinship_file, ksnps_indices)
        eigen_source = KinshipToEigen(source, config.write_eigen)
        kinship = KinshipShape.resolve(
            execution.n_samples,
            execution.n_input_samples,
            loaded=isinstance(source, ProvidedKinship),
            saved=config.save_kinship,
        )

    return StandardAnalysisPlan(
        execution=replace(execution, kinship=kinship),
        lmm=config.lmm_config(),
        eigen_source=eigen_source,
        snps_indices=snps_indices,
    )
