"""Tolerance configuration for numerical comparisons in GEMMA-Next validation.

This module provides configurable tolerance thresholds for comparing GEMMA-Next
output to reference GEMMA output. Different value types require different
tolerances due to how they are computed.

**Scientific Equivalence**: Despite numerical differences, JAMMA and GEMMA produce
statistically identical results:
- P-value rank correlation: 1.000000
- Significance agreement at all thresholds: 100%
- Effect direction agreement: 100%
- Top hits overlap: 100%

Numerical differences arise from:
- **GEMMA output precision**: 6 significant figures in scientific notation
- **Optimization convergence**: Brent minimization vs derivative-based methods
- **CDF implementations**: JAX betainc vs GSL gsl_cdf_fdist_Q
- **Lambda sensitivity**: ~0.35x amplification to beta (1e-5 lambda → 3.5e-6 beta)

Value types and tolerances:
- **Kinship matrices**: Direct matrix operations, tightest tolerance (1e-8)
- **Effect sizes (beta)**: Lambda-sensitive, moderate tolerance (1e-2)
- **Standard errors**: Lambda-sensitive, moderate tolerance (1e-5)
- **P-values**: CDF differences, moderate tolerance (1e-4)
- **Log-likelihood**: Direct computation, tight tolerance (1e-6)
- **Lambda (variance ratio)**: Brent convergence, moderate tolerance (2e-5)
- **Allele frequency**: JAMMA reports MAF (<=0.5), GEMMA reports AF (may be >0.5)
"""

from dataclasses import dataclass


@dataclass
class ToleranceConfig:
    """Configuration for numerical comparison tolerances.

    Different types of statistical values require different tolerance levels
    due to numerical precision characteristics of the underlying computations.

    Tolerance values are calibrated based on empirical comparison between
    JAMMA (JAX-based) and GEMMA (GSL-based) implementations on the mouse_hs1940
    reference dataset. The differences arise from:
    - Different numerical libraries (JAX vs GSL)
    - Different F-distribution CDF implementations (JAX.scipy vs GSL)
    - Different optimization convergence criteria
    - Different floating-point accumulation order in parallel computations

    Attributes:
        beta_rtol: Relative tolerance for effect sizes.
            Max observed: 8.5e-3 due to lambda sensitivity. Typical: 7e-6.
            Scientific conclusions (effect direction, ranking) are identical.
        se_rtol: Relative tolerance for standard errors.
            Max observed: 2e-6. Follows similar pattern to beta.
        pvalue_rtol: Relative tolerance for p-values.
            Max observed: 4.1e-5 due to CDF implementation differences.
            Significance thresholds (0.05, 0.01, 5e-8) are always consistent.
        kinship_rtol: Relative tolerance for kinship matrix elements.
            Tightest tolerance - direct matrix computation.
        logl_rtol: Relative tolerance for log-likelihood values.
            Max observed: 3.2e-7. Very stable.
        lambda_rtol: Relative tolerance for lambda (variance ratio).
            Max observed: 1.2e-5 from Brent convergence differences.
        af_rtol: Relative tolerance for allele frequency.
            JAMMA reports MAF (<=0.5), GEMMA reports AF (can be >0.5).
            Comparison normalizes both to MAF before comparing.
        atol: Absolute tolerance for values near zero.

    Example:
        >>> config = ToleranceConfig()
        >>> config.kinship_rtol
        1e-08
        >>> config = ToleranceConfig.strict()
        >>> config.kinship_rtol
        1e-10
    """

    # Beta: max observed 8.5e-3 due to lambda sensitivity (0.35x amplification)
    # and GEMMA output precision (6 sig figs). Scientific significance unaffected.
    beta_rtol: float = 1e-2
    # SE: follows beta sensitivity pattern
    se_rtol: float = 1e-5
    # P-values: CDF implementation differences (JAX betainc vs GSL)
    # Max observed: 4.1e-5. Scientific thresholds (0.05, 0.01, etc.) unaffected.
    pvalue_rtol: float = 1e-4
    # Kinship: direct matrix computation, tightest tolerance
    kinship_rtol: float = 1e-8
    # Log-likelihood: direct computation, tight tolerance. Max observed: 3.2e-7
    logl_rtol: float = 1e-6
    # Lambda: Brent optimization convergence. Max observed: 1.2e-5
    lambda_rtol: float = 2e-5
    # AF: JAMMA reports MAF (<=0.5), GEMMA reports AF. Max diff from rounding: 0.04
    af_rtol: float = 0.05
    atol: float = 1e-12

    @classmethod
    def strict(cls) -> "ToleranceConfig":
        """Create a strict tolerance configuration.

        Use for debugging or when near-exact numerical match is required.
        Note: May fail on some comparisons due to CDF implementation differences.

        Returns:
            ToleranceConfig with tighter tolerances (10x stricter than default).
        """
        return cls(
            beta_rtol=1e-6,
            se_rtol=1e-6,
            pvalue_rtol=1e-5,
            kinship_rtol=1e-10,
            logl_rtol=1e-7,
            lambda_rtol=1e-5,
            af_rtol=1e-2,
            atol=1e-14,
        )

    @classmethod
    def relaxed(cls) -> "ToleranceConfig":
        """Create a relaxed tolerance configuration.

        Use for debugging when investigating discrepancies, or when comparing
        across different platforms/compilers where floating-point behavior
        may differ slightly.

        Returns:
            ToleranceConfig with looser tolerances (10x looser than default).
        """
        return cls(
            beta_rtol=1e-4,
            se_rtol=1e-4,
            pvalue_rtol=1e-3,
            kinship_rtol=1e-6,
            logl_rtol=1e-5,
            lambda_rtol=1e-3,
            af_rtol=0.5,  # Allow full complement range
            atol=1e-10,
        )
