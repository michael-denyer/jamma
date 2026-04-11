"""Validation modules for JAMMA.

This package contains utilities for validating JAMMA output:
- tolerances: Configurable tolerance thresholds for different value types
- compare: Numerical comparison utilities with tolerance configuration
"""

from jamma.validation.compare import (
    AssocComparisonResult,
    ComparisonResult,
    compare_arrays,
    compare_assoc_results,
    compare_kinship_matrices,
    load_gemma_assoc,
    load_gemma_kinship,
)
from jamma.validation.tolerances import ToleranceConfig

__all__ = [
    "AssocComparisonResult",
    "ComparisonResult",
    "ToleranceConfig",
    "compare_arrays",
    "compare_assoc_results",
    "compare_kinship_matrices",
    "load_gemma_assoc",
    "load_gemma_kinship",
]
