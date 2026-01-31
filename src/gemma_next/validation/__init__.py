"""Validation modules for GEMMA-Next.

This package contains utilities for validating GEMMA-Next output:
- tolerances: Configurable tolerance thresholds for different value types
- compare: Numerical comparison utilities with tolerance configuration
"""

from gemma_next.validation.tolerances import ToleranceConfig

__all__ = ["ToleranceConfig"]
