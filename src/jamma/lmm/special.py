"""Pure-stdlib special functions for JAMMA p-value computation.

Implements regularized incomplete beta (betainc) via Cephes Lentz continued
fraction and chi-squared survival (chi2_sf) via erfc. Both use only the Python
`math` standard library — no numpy, no scipy, no third-party imports.

Algorithm source: codeplea/incbeta (https://codeplea.com/incomplete-beta-function-c)
  and the Cephes mathematical library by Stephen L. Moshier.
  Ported to Python with tighter convergence threshold (_CF_STOP=1e-14 vs
  original 1e-8) to achieve < 4e-11 rtol vs scipy across JAMMA's full
  parameter range (a=df/2 for df 10-100000, b=0.5).

License: Zlib (same as codeplea/incbeta source). See LICENSE for details.

Accuracy (verified against scipy):
  - betainc: max rtol < 4e-11 across a=df/2 (df 10-100000), b=0.5, F in [0.01, 100]
  - chi2_sf: max rtol < 9e-15 across x in [0.001, 500]
"""

import math

# Lentz continued fraction constants
_CF_TINY = 1.0e-30  # underflow guard: replaces values < TINY with TINY
_CF_STOP = 1.0e-14  # convergence threshold (tighter than codeplea 1e-8 for accuracy)
_CF_MAX_ITER = 200  # sufficient for JAMMA's parameter range; converges in < 200


def _betainc_cf(a: float, b: float, x: float) -> float:
    """Lentz continued fraction core for regularized incomplete beta.

    Evaluates I_x(a, b) using the continued fraction representation.
    Caller guarantees x < (a+1)/(a+b+2), ensuring CF convergence.

    Port of codeplea/incbeta (zlib license) with STOP tightened to 1e-14.

    Args:
        a: First shape parameter (> 0).
        b: Second shape parameter (> 0).
        x: Upper limit, in (0, 1) and below the symmetry threshold.

    Returns:
        I_x(a, b) value.

    Raises:
        ArithmeticError: If continued fraction does not converge in _CF_MAX_ITER.
    """
    lbeta_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta_ab) / a

    # Lentz's method: f, c, d are the current CF convergents
    f = 1.0
    c = 1.0
    d = 0.0

    for i in range(_CF_MAX_ITER + 1):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            # Even step: numerator from Cephes formula
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            # Odd step: numerator from Cephes formula
            numerator = -((a + m) * (a + b + m) * x) / (
                (a + 2.0 * m) * (a + 2.0 * m + 1.0)
            )

        d = 1.0 + numerator * d
        if math.fabs(d) < _CF_TINY:
            d = _CF_TINY
        d = 1.0 / d

        c = 1.0 + numerator / c
        if math.fabs(c) < _CF_TINY:
            c = _CF_TINY

        cd = c * d
        f *= cd

        if math.fabs(1.0 - cd) < _CF_STOP:
            return front * (f - 1.0)

    raise ArithmeticError(
        f"betainc CF did not converge after {_CF_MAX_ITER} iterations "
        f"for a={a}, b={b}, x={x}"
    )


def betainc(a: float, b: float, z: float, complement_z: float | None = None) -> float:
    """Regularized incomplete beta function I_z(a, b).

    Computes I_z(a, b) = B(a, b, z) / B(a, b) where B(a, b, z) is the
    incomplete beta function and B(a, b) is the complete beta function.

    Uses the Cephes Lentz continued fraction (codeplea/incbeta port) with
    symmetry relation I_z(a,b) = 1 - I_{1-z}(b,a) to ensure CF convergence
    across all z in [0, 1].

    Args:
        a: First shape parameter (> 0).
        b: Second shape parameter (> 0).
        z: Upper limit of integration, in [0, 1].
        complement_z: If provided, used as 1 - z in the symmetry branch to
            avoid float64 cancellation when z is very close to 1. Must equal
            1 - z; caller is responsible for accuracy. Has no effect when the
            direct CF branch is taken (z <= threshold).

    Returns:
        I_z(a, b) in [0, 1].

    Raises:
        ValueError: If z is not in [0, 1].
    """
    if not (0.0 <= z <= 1.0):
        raise ValueError(f"z must be in [0, 1], got {z}")
    if z == 0.0:
        return 0.0
    if z == 1.0:
        return 1.0

    # Symmetry threshold: use I_{1-z}(b,a) = 1 - I_z(a,b) when z > threshold
    # for better CF convergence (ensures x < (a+1)/(a+b+2) for the CF call)
    threshold = (a + 1.0) / (a + b + 2.0)
    if z > threshold:
        # Use caller-provided complement to avoid 1.0 - z precision loss
        cz = complement_z if complement_z is not None else 1.0 - z
        return 1.0 - _betainc_cf(b, a, cz)

    return _betainc_cf(a, b, z)


def chi2_sf(x: float, df: int = 1) -> float:
    """Chi-squared survival function P(X > x) for df=1.

    Uses the identity chi2(1) = N(0,1)^2, which gives:
        P(chi2(1) > x) = erfc(sqrt(x/2))

    This is exact for df=1 (single degree of freedom). Max rtol vs scipy:
    8.9e-15 across x in [0.001, 500].

    Args:
        x: Test statistic value.
        df: Degrees of freedom. Must be 1; other values raise ValueError.

    Returns:
        P(chi2(df) > x) in [0, 1].

    Raises:
        ValueError: If df != 1.
    """
    if df != 1:
        raise ValueError(f"chi2_sf only supports df=1, got df={df}")
    if x <= 0.0:
        return 1.0
    if not math.isfinite(x):
        return 0.0
    return math.erfc(math.sqrt(x / 2.0))
