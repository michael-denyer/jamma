"""Validation modules for GEMMA-Next.

This package contains utilities for validating GEMMA-Next output:
- tolerances: Configurable tolerance thresholds for different value types
- compare: Numerical comparison utilities with tolerance configuration
"""

from jamma.validation.compare import (
    ComparisonResult,
    compare_arrays,
    compare_assoc_results,
    compare_kinship_matrices,
    load_gemma_assoc,
    load_gemma_kinship,
)
from jamma.validation.tolerances import ToleranceConfig

__all__ = [
    "ToleranceConfig",
    "ComparisonResult",
    "compare_arrays",
    "compare_kinship_matrices",
    "compare_assoc_results",
    "load_gemma_kinship",
    "load_gemma_assoc",
]
