"""Kinship matrix computation.

This module provides GEMMA-compatible kinship matrix computation with
missing data handling.

Key functions:
- compute_kinship_streaming: -gk 1 from disk (streaming)
- compute_standardized_kinship_streaming: -gk 2 from disk (streaming)
- compute_loco_kinship_streaming: Compute LOCO kinship from disk (streaming)
- impute_and_center: Impute missing values to SNP mean and center
- impute_center_and_standardize: Impute, center, and standardize per SNP
- write_kinship_matrix: Write kinship matrix in GEMMA format

The in-memory oracle (compute_centered_kinship, compute_standardized_kinship)
has no production caller; it lives at tests/reference/kinship.py.
"""

from jamma.kinship.compute import (
    LocoKinshipStream,
    SnpStatsCache,
    compute_kinship_streaming,
    compute_loco_kinship_streaming,
    compute_standardized_kinship_streaming,
    validate_valid_indices,
)
from jamma.kinship.io import (
    read_kinship_matrix,
    write_kinship_matrix,
    write_loco_kinship_matrices,
)
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize

__all__ = [
    "LocoKinshipStream",
    "SnpStatsCache",
    "compute_kinship_streaming",
    "compute_loco_kinship_streaming",
    "compute_standardized_kinship_streaming",
    "impute_and_center",
    "impute_center_and_standardize",
    "read_kinship_matrix",
    "validate_valid_indices",
    "write_kinship_matrix",
    "write_loco_kinship_matrices",
]
