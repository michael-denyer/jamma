"""Kinship matrix computation.

This module provides GEMMA-compatible kinship matrix computation with
missing data handling.

Key functions:
- compute_centered_kinship: Compute K = X_c @ X_c.T / p (GEMMA -gk 1)
- compute_standardized_kinship: Compute K = Z @ Z.T / p (GEMMA -gk 2)
- compute_loco_kinship_streaming: Compute LOCO kinship from disk (streaming)
- impute_and_center: Impute missing values to SNP mean and center
- impute_center_and_standardize: Impute, center, and standardize per SNP
- write_kinship_matrix: Write kinship matrix in GEMMA format
"""

from jamma.kinship.compute import (
    SnpStatsCache,
    compute_centered_kinship,
    compute_kinship_streaming,
    compute_loco_kinship_streaming,
    compute_standardized_kinship,
)
from jamma.kinship.io import (
    read_kinship_matrix,
    write_kinship_matrix,
    write_loco_kinship_matrices,
)
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize

__all__ = [
    "SnpStatsCache",
    "compute_centered_kinship",
    "compute_kinship_streaming",
    "compute_loco_kinship_streaming",
    "compute_standardized_kinship",
    "impute_and_center",
    "impute_center_and_standardize",
    "read_kinship_matrix",
    "write_kinship_matrix",
    "write_loco_kinship_matrices",
]
