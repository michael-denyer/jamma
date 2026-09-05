"""Field-specific policies for comparing JAMMA output with GEMMA references.

These thresholds are comparison gates, not a guarantee for untested inputs or
platforms. The mathematical validation bundles record the cases and numerical
backends actually exercised.

GEMMA writes statistics with six digits after the decimal in scientific notation
(seven significant digits). JAMMA uses a grid and vectorized golden-section
search, with safeguarded score refinement for interior REML optima; GEMMA uses
GSL scalar optimization. The existing lambda policy checks boundary classes
separately from ordinary interior relative error.

``logl_H1`` is REML in mode 1 and MLE in modes 2 and 4. The earlier mode 4
likelihood mismatch was an output-semantics defect, not evidence for widening
``logl_rtol``. AF comparisons in the legacy comparator fold to MAF; the independent
validation driver additionally checks the declared allele-orientation contract.
"""

import math
from dataclasses import dataclass, field

from jamma.lmm.schema import DEFAULT_L_MAX, DEFAULT_L_MIN

DEFAULT_LAMBDA_RTOL = 2e-5


@dataclass(frozen=True)
class LambdaBoundaryPolicy:
    """Optimizer bounds used to classify lambda comparison results."""

    lower: float = DEFAULT_L_MIN
    upper: float = DEFAULT_L_MAX
    rtol: float = DEFAULT_LAMBDA_RTOL

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("lambda boundary policy bounds must be finite")
        if not math.isfinite(self.rtol):
            raise ValueError("lambda boundary classification rtol must be finite")
        if not 0 < self.lower < self.upper:
            raise ValueError(
                "lambda boundary policy requires 0 < lower < upper, "
                f"got lower={self.lower}, upper={self.upper}"
            )
        if not 0 <= self.rtol < 1:
            raise ValueError(
                "lambda boundary classification rtol must be in [0, 1), "
                f"got {self.rtol}"
            )
        if self.lower * (1 + self.rtol) >= self.upper * (1 - self.rtol):
            raise ValueError("lambda boundary classification bands must not overlap")


@dataclass
class ToleranceConfig:
    """Configuration for numerical comparison tolerances.

    Uses numpy's allclose semantics: |a - b| <= atol + rtol * |b|

    When comparing values near zero (e.g., p-value = 1e-15), relative
    tolerance alone is misleading -- "100% relative error" on a biologically
    irrelevant value. The atol floor (default 1e-12) ensures values smaller
    than atol are considered equal regardless of relative difference.

    Example: atol=1e-12, rtol=1e-4
    - Comparing 0.05 vs 0.0500001: rtol dominates (relative check)
    - Comparing 1e-15 vs 2e-15: atol dominates (both effectively zero)

    Tolerance values are calibrated based on empirical comparison between
    JAMMA and GEMMA (GSL-based) implementations on the mouse_hs1940
    reference dataset. The differences arise from:
    - Different numerical libraries (Cephes vs GSL)
    - Different F-distribution CDF implementations (Cephes betainc vs GSL)
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
            For REML logl (null model): max observed 3.2e-7, very stable.
            For per-SNP MLE logl_H1: max observed ~1.35e-3 on mouse_hs1940
            due to golden section vs Brent optimizer divergence on weak-signal
            SNPs with flat optimization landscapes. Override with wider tolerance
            in dataset-specific ToleranceConfig when comparing logl_H1.
        lambda_rtol: Relative tolerance for lambda (variance ratio).
            Max observed: 1.2e-5 from Brent convergence differences.
        lambda_boundary: Bounds used by both optimizers and the relative margin
            used to classify lower- and upper-bound hits.
        af_rtol: Relative tolerance for allele frequency.
            JAMMA reports MAF (<=0.5), GEMMA reports AF (can be >0.5).
            Comparison normalizes both to MAF before comparing.
        atol: Absolute tolerance floor for near-zero comparisons.
            Values smaller than atol are considered equal regardless of
            relative difference. Used by np.allclose: |a-b| <= atol + rtol*|b|

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
    # P-values (Wald/Score): CDF implementation differences (Cephes betainc vs GSL)
    # Max observed: 4.1e-5. Scientific thresholds (0.05, 0.01, etc.) unaffected.
    pvalue_rtol: float = 1e-4
    # LRT p-values: wider than Wald/Score due to chi-squared distribution
    # magnifying small log-likelihood differences, especially with covariates.
    p_lrt_rtol: float = 5e-3
    # Kinship: direct matrix computation, tightest tolerance
    kinship_rtol: float = 1e-8
    # Log-likelihood: REML logl max observed 3.2e-7. Per-SNP MLE logl_H1 can be
    # wider (~1.35e-3 on mouse_hs1940) due to optimizer divergence on weak-signal
    # SNPs. Override in dataset-specific configs when comparing logl_H1.
    logl_rtol: float = 1e-6
    # Lambda: Brent optimization convergence. Max observed: 1.2e-5
    lambda_rtol: float = DEFAULT_LAMBDA_RTOL
    # Optimizer bounds and classification tolerance for boundary exemptions.
    lambda_boundary: LambdaBoundaryPolicy = field(default_factory=LambdaBoundaryPolicy)
    # AF: JAMMA reports MAF (<=0.5), GEMMA reports AF. Max diff from rounding: 0.04
    af_rtol: float = 0.05
    # Absolute tolerance floor for near-zero comparisons.
    # Values smaller than atol are considered equal regardless of relative difference.
    # Used by np.allclose: |a-b| <= atol + rtol*|b|
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
            beta_rtol=0.1,  # 10x looser than default (1e-2)
            se_rtol=1e-4,  # 10x looser than default (1e-5)
            pvalue_rtol=1e-3,  # 10x looser than default (1e-4)
            kinship_rtol=1e-6,  # 100x looser than default (1e-8)
            logl_rtol=1e-5,  # 10x looser than default (1e-6)
            lambda_rtol=2e-4,  # 10x looser than default (2e-5)
            af_rtol=0.5,  # Allow full complement range
            atol=1e-10,  # 100x looser than default (1e-12)
        )
