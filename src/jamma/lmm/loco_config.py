"""The LOCO-only half of run_lmm_loco's configuration.

Its own module for the same reason ``pipeline_config`` is: this is data, and
``loco`` is behaviour. Eigen member names belong to
:class:`~jamma.lmm.eigen_io.EigenGeneration`, not to this config.

``jamma.lmm.loco`` re-exports both names, so ``from jamma.lmm.loco import
LocoConfig`` keeps working — that is the path ``jamma.pipeline`` and
``jamma.lmm.__init__`` use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jamma.genotype.dataset import GenotypeDataset
from jamma.lmm.association_plan import ExecutableAssociationPlan
from jamma.lmm.eigen_plan import EigenDriverPlan
from jamma.lmm.prepare_common import AnalysedPhenotype
from jamma.lmm.schema import LmmConfig


@dataclass(frozen=True)
class LocoConfig:
    """LOCO-specific options for :func:`run_lmm_loco`.

    The nine numerical knobs shared with every other runner live in
    ``LmmConfig``; these are the ones only LOCO has — where kinship and eigen
    artefacts are written, which SNPs take part, and the streaming chunk width.

    Frozen to match ``LmmConfig``. The ndarray fields are frozen by reference
    only, as usual for a dataclass: callers must not mutate an array after
    handing it over.

    Attributes:
        kinship_output_dir: Directory for K_loco files. Set it to write each
            chromosome's K_loco to disk; None writes none.
        prefix: Filename prefix shared by K_loco, eigen and manifest files.
        snps_indices: Global indices of SNPs to test. None tests all.
        ksnps_indices: Global indices of SNPs used to build kinship. None
            uses all.
        col_chunk_size: Widest SNP chunk for association. Kinship and SNP
            statistics use their own chunk width.
        write_eigen: Write per-chromosome eigenvalues and eigenvectors.
        eigen_dir: Directory for eigen files. Required when write_eigen is set.
        legacy_text: Write kinship and eigen files as GEMMA text rather than
            .npy.
        info_threshold: Minimum imputation INFO over the analysed samples,
            for kinship and association SNPs alike. 0.0 disables the filter.
    """

    kinship_output_dir: Path | None = None
    prefix: str = "result"
    snps_indices: np.ndarray | None = None
    ksnps_indices: np.ndarray | None = None
    col_chunk_size: int = 5_000
    write_eigen: bool = False
    eigen_dir: Path | None = None
    legacy_text: bool = False
    info_threshold: float = 0.0

    def __post_init__(self) -> None:
        # Checked here rather than partway through the run: the caller learns
        # at construction, before any chromosome has been eigendecomposed.
        if self.write_eigen and self.eigen_dir is None:
            raise ValueError(
                "write_eigen=True requires eigen_dir to be set. "
                "Pass eigen_dir=<directory> alongside write_eigen=True."
            )
        if self.col_chunk_size <= 0:
            raise ValueError(
                f"col_chunk_size must be positive, got {self.col_chunk_size}"
            )

    @property
    def artifact_suffix(self) -> str:
        """Extension for kinship and eigen artifacts: .txt for GEMMA, else .npy."""
        return ".txt" if self.legacy_text else ".npy"

    def kinship_path(self, chr_name: str) -> Path:
        """Path for one chromosome's LOCO kinship matrix.

        Raises:
            ValueError: If ``kinship_output_dir`` is None: nothing asked for
                the kinship to be saved, so there is no directory to name it under.
        """
        if self.kinship_output_dir is None:
            raise ValueError(
                "kinship_path() requires kinship_output_dir, which is None"
            )
        name = f"{self.prefix}.loco.cXX.chr{chr_name}{self.artifact_suffix}"
        return self.kinship_output_dir / name


DEFAULT_LOCO_CONFIG = LocoConfig()
"""The all-defaults LOCO config, shared as run_lmm_loco's default argument.

LocoConfig is frozen, so one instance is safe to share.
"""


@dataclass(frozen=True, slots=True)
class LocoRun:
    """One LOCO run, resolved once and read by every stage below it.

    Attributes:
        dataset: The genotypes, opened once; its rows are the ones
            ``samples.valid_mask`` indexes.
        samples: The phenotype and covariates over the analysed samples; its
            ``valid_mask`` indexes the dataset rows.
        config: Numerical settings shared with every other runner.
        loco: LOCO-only settings.
        execution: The association plan, with its kinship shape resolved.
        eigen_plan: The driver every chromosome's eigendecomposition runs.

    Raises:
        ValueError: If ``execution`` resolves no kinship shape, or plans chunks
            wider than ``loco.col_chunk_size``.
    """

    dataset: GenotypeDataset
    samples: AnalysedPhenotype
    config: LmmConfig
    loco: LocoConfig
    execution: ExecutableAssociationPlan
    eigen_plan: EigenDriverPlan

    def __post_init__(self) -> None:
        if self.execution.kinship is None:
            raise ValueError("LOCO needs a plan with its kinship shape resolved")
        chunk_size = self.execution.conservative_chunks.chunk_size
        if chunk_size > self.loco.col_chunk_size:
            raise ValueError(
                f"execution plans {chunk_size}-SNP chunks but "
                f"loco.col_chunk_size is {self.loco.col_chunk_size}"
            )

    @property
    def analysed_rows(self) -> np.ndarray:
        """Dataset row indices of the analysed samples."""
        return np.flatnonzero(self.samples.valid_mask)
