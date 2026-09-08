"""The analysed-sample basis shared by ``-lmm`` and ``-gk``.

GEMMA's ``ProcessCvtPhen`` builds one ``indicator_idv`` from the selected
phenotype columns and the covariate file, and every later stage measures over
it. ``load_analysed_samples`` is that step: it reads the ``.fam`` once, loads
and validates the covariates once, and returns the rows both programs analyse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from jamma.io.covariate import encode_categorical_covariates, read_covariate_file
from jamma.io.plink import parse_fam_phenotype_column
from jamma.lmm.genotype_source import SampleBasis
from jamma.lmm.prepare_common import compute_valid_mask, with_intercept
from jamma.pipeline_config import PipelineConfig

__all__ = ["AnalysedSamples", "load_analysed_samples", "load_covariates"]


@dataclass(frozen=True, slots=True)
class AnalysedSamples:
    """GEMMA's indicator_idv: the rows every selected phenotype and covariate has."""

    phenotypes: dict[int, np.ndarray]
    covariates: np.ndarray | None
    valid_mask: np.ndarray
    basis: SampleBasis

    @property
    def filter_indices(self) -> np.ndarray | None:
        """Analysed positions, or None when every sample qualifies.

        The kinship streams take an all-samples fast path on None.
        """
        return None if self.basis.is_all_samples else self.basis.positions

    @property
    def n_covariates(self) -> int:
        """Effective covariate count, 1 for the intercept-only model."""
        return 1 if self.covariates is None else self.covariates.shape[1]


def load_covariates(config: PipelineConfig, n_samples: int) -> np.ndarray | None:
    """Load and validate the covariate file.

    Args:
        config: Pipeline configuration. Reads ``covariate_file`` and
            ``cat_columns``.
        n_samples: Number of samples for row-count validation.

    Returns:
        Covariate array of shape (n_samples, n_covariates), or None if no
        covariate file was specified. The intercept column is appended later,
        by ``with_intercept`` once the analysed-sample mask exists.

    Raises:
        ValueError: If covariate row count does not match n_samples.
    """
    if config.covariate_file is None:
        return None

    logger.info(f"Loading covariates from {config.covariate_file}")
    covariates, _ = read_covariate_file(config.covariate_file)

    if covariates.shape[0] != n_samples:
        raise ValueError(
            f"Covariate file has {covariates.shape[0]} rows "
            f"but PLINK data has {n_samples} samples. "
            f"Covariate rows must match sample count exactly."
        )

    logger.info(f"Loaded {covariates.shape[1]} covariates")

    if config.cat_columns is not None:
        covariates = encode_categorical_covariates(covariates, config.cat_columns)
        logger.info(
            f"Categorical encoding applied to columns {config.cat_columns}: "
            f"expanded to {covariates.shape[1]} covariate columns"
        )

    return covariates


def load_analysed_samples(config: PipelineConfig, n_samples: int) -> AnalysedSamples:
    """Read the phenotype columns and covariates, and intersect their masks.

    Reads the ``.fam`` once, parses each configured phenotype column, and
    intersects the per-column valid masks so every stage measures over the
    sample set common to all of them. The covariates come back carrying an
    intercept when no column is constant over that intersection, so
    ``covariates.shape[1]`` is the n_cvt every later stage uses.

    Args:
        config: Pipeline configuration. Reads ``bfile``, ``phenotype_columns``,
            ``covariate_file``, and ``cat_columns``.
        n_samples: Sample count from the PLINK metadata, used to validate the
            covariate row count.

    Returns:
        The phenotype columns, the loaded covariates, and the analysed basis.

    Raises:
        ValueError: If the covariate row count is wrong, if the ``.fam`` file
            cannot be read, or if no sample is valid across all phenotype
            columns (per-column counts appear in the message for diagnosis).
    """
    covariates = load_covariates(config, n_samples)

    fam_path = f"{config.bfile}.fam"
    try:
        fam_data = np.loadtxt(fam_path, dtype=str, ndmin=2)
    except (ValueError, OSError) as e:
        raise ValueError(f"Failed to read .fam file {fam_path}: {e}") from e

    phenotypes: dict[int, np.ndarray] = {}
    masks: list[np.ndarray] = []
    for col in config.phenotype_columns:
        logger.info(f"Using phenotype column {col} (file column {col + 5})")
        pheno = parse_fam_phenotype_column(fam_data, col)
        phenotypes[col] = pheno
        masks.append(compute_valid_mask(pheno, covariates))

    valid_mask = np.all(masks, axis=0)
    n_valid = int(np.sum(valid_mask))
    per_column_counts = [int(m.sum()) for m in masks]

    if n_valid == 0:
        raise ValueError(
            f"No samples have valid values across all {len(masks)} phenotype "
            f"columns. Per-column valid counts: "
            f"{dict(zip(config.phenotype_columns, per_column_counts, strict=True))}"
        )
    if n_valid < min(per_column_counts):
        logger.warning(
            f"Sample mask intersection reduced valid samples: "
            f"per-phenotype counts {per_column_counts}, intersection {n_valid}"
        )

    return AnalysedSamples(
        phenotypes=phenotypes,
        covariates=with_intercept(covariates, valid_mask),
        valid_mask=valid_mask,
        basis=SampleBasis.from_mask(valid_mask),
    )
