"""Kinship matrix computation.

This module provides JAX-accelerated kinship matrix computation with
GEMMA-compatible missing data handling. The kinship matrix (also known
as the genetic relatedness matrix or GRM) is fundamental to linear
mixed model association analysis.

Key functions:
- compute_centered_kinship: Compute K = X_c @ X_c.T / p (GEMMA -gk 1)
- compute_standardized_kinship: Compute K = Z @ Z.T / p (GEMMA -gk 2)
- compute_loco_kinship: Compute LOCO kinship via subtraction (in-memory)
- compute_loco_kinship_streaming: Compute LOCO kinship from disk (streaming)
- impute_and_center: Impute missing values to SNP mean and center
- impute_center_and_standardize: Impute, center, and standardize per SNP
- write_kinship_matrix: Write kinship matrix in GEMMA format
"""

from jamma.io.plink import get_chromosome_partitions
from jamma.kinship.compute import (
    compute_centered_kinship,
    compute_kinship_streaming,
    compute_loco_kinship,
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
    "compute_centered_kinship",
    "compute_kinship_streaming",
    "compute_loco_kinship",
    "compute_loco_kinship_streaming",
    "compute_standardized_kinship",
    "get_chromosome_partitions",
    "impute_and_center",
    "impute_center_and_standardize",
    "read_kinship_matrix",
    "write_kinship_matrix",
    "write_loco_kinship_matrices",
]
