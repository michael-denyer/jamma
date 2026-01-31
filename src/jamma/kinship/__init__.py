"""Kinship matrix computation.

This module provides JAX-accelerated kinship matrix computation with
GEMMA-compatible missing data handling. The kinship matrix (also known
as the genetic relatedness matrix or GRM) is fundamental to linear
mixed model association analysis.

Key functions:
- compute_centered_kinship: Compute K = X_c @ X_c.T / p (GEMMA -gk 1)
- impute_and_center: Impute missing values to SNP mean and center
"""

from jamma.kinship.compute import compute_centered_kinship
from jamma.kinship.missing import impute_and_center

__all__ = ["compute_centered_kinship", "impute_and_center"]
