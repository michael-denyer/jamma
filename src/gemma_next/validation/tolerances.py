"""Tolerance configuration for numerical comparisons in GEMMA-Next validation.

This module provides configurable tolerance thresholds for comparing GEMMA-Next
output to reference GEMMA output. Different value types require different
tolerances due to how they are computed:

- **Kinship matrices**: Direct matrix operations, tightest tolerance (1e-8)
- **Effect sizes (beta)**: Linear algebra solutions, tight tolerance (1e-6)
- **Standard errors**: Square root operations, tight tolerance (1e-6)
- **P-values**: CDF computations differ between GSL (GEMMA) and SciPy (GEMMA-Next),
  especially for very small p-values (<1e-10), so looser tolerance (1e-5)
"""

from dataclasses import dataclass


@dataclass
class ToleranceConfig:
    """Configuration for numerical comparison tolerances.

    Different types of statistical values require different tolerance levels
    due to numerical precision characteristics of the underlying computations.

    Attributes:
        beta_rtol: Relative tolerance for effect sizes (beta coefficients).
            Effect sizes are computed via linear algebra, typically well-conditioned.
        se_rtol: Relative tolerance for standard errors.
            Standard errors involve square roots but are generally stable.
        pvalue_rtol: Relative tolerance for p-values.
            P-values require looser tolerance because GEMMA uses GSL's CDF
            implementation while GEMMA-Next uses SciPy. Very small p-values
            (< 1e-10) may diverge more significantly.
        kinship_rtol: Relative tolerance for kinship matrix elements.
            Kinship computation is a direct matrix operation with well-defined
            numerical properties, allowing tightest tolerance.
        atol: Absolute tolerance for values near zero.
            Applies across all comparisons to handle values close to machine
            epsilon where relative tolerance becomes meaningless.

    Example:
        >>> config = ToleranceConfig()
        >>> config.kinship_rtol
        1e-08
        >>> config = ToleranceConfig.strict()
        >>> config.kinship_rtol
        1e-10
    """

    beta_rtol: float = 1e-6
    se_rtol: float = 1e-6
    pvalue_rtol: float = 1e-5
    kinship_rtol: float = 1e-8
    atol: float = 1e-12

    @classmethod
    def strict(cls) -> "ToleranceConfig":
        """Create a strict tolerance configuration.

        Use for debugging or when exact numerical match is required.
        Note: May fail on p-value comparisons due to CDF implementation differences.

        Returns:
            ToleranceConfig with tighter tolerances (10x stricter than default).
        """
        return cls(
            beta_rtol=1e-7,
            se_rtol=1e-7,
            pvalue_rtol=1e-6,
            kinship_rtol=1e-10,
            atol=1e-14,
        )

    @classmethod
    def relaxed(cls) -> "ToleranceConfig":
        """Create a relaxed tolerance configuration.

        Use for debugging when investigating discrepancies, or when comparing
        across different platforms/compilers where floating-point behavior
        may differ slightly.

        Returns:
            ToleranceConfig with looser tolerances (100x looser than default).
        """
        return cls(
            beta_rtol=1e-4,
            se_rtol=1e-4,
            pvalue_rtol=1e-3,
            kinship_rtol=1e-6,
            atol=1e-10,
        )
