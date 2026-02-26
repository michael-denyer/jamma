"""Kinship matrix computation.

This module provides GEMMA-compatible kinship matrix computation with
missing data handling. Standard kinship (centered, standardized) uses
NumPy BLAS; LOCO kinship (leave-one-chromosome-out) requires JAX.

Key functions:
- compute_centered_kinship: Compute K = X_c @ X_c.T / p (GEMMA -gk 1)
- compute_standardized_kinship: Compute K = Z @ Z.T / p (GEMMA -gk 2)
- compute_loco_kinship: Compute LOCO kinship via subtraction (in-memory, JAX)
- compute_loco_kinship_streaming: Compute LOCO kinship from disk (streaming, JAX)
- impute_and_center: Impute missing values to SNP mean and center
- impute_center_and_standardize: Impute, center, and standardize per SNP
- write_kinship_matrix: Write kinship matrix in GEMMA format
"""

from jamma.core.backend import has_jax
from jamma.io.plink import get_chromosome_partitions
from jamma.kinship.compute import (
    compute_centered_kinship,
    compute_kinship_streaming,
    compute_standardized_kinship,
)
from jamma.kinship.io import (
    read_kinship_matrix,
    write_kinship_matrix,
    write_loco_kinship_matrices,
)
from jamma.kinship.missing import impute_and_center, impute_center_and_standardize

# LOCO kinship functions require JAX — probe first, import unconditionally.
_HAS_JAX = has_jax()

if _HAS_JAX:
    from jamma.kinship.compute import (
        compute_loco_kinship,
        compute_loco_kinship_streaming,
    )

__all__ = [
    "compute_centered_kinship",
    "compute_kinship_streaming",
    "compute_standardized_kinship",
    "get_chromosome_partitions",
    "impute_and_center",
    "impute_center_and_standardize",
    "read_kinship_matrix",
    "write_kinship_matrix",
    "write_loco_kinship_matrices",
]
if _HAS_JAX:
    __all__ += [
        "compute_loco_kinship",
        "compute_loco_kinship_streaming",
    ]
