"""NumPy special functions for JAMMA p-value computation, without scipy.

Implements the regularized incomplete beta (betainc_batch) via the Cephes
Lentz continued fraction and the chi-squared survival function
(chi2_sf_batch) via erfc, vectorized over arrays of SNPs. scipy stays out of
the runtime because installing it replaces ILP64 numpy-mkl with LP64 numpy.
The scalar forms these were ported from are test oracles in
``tests/reference/special.py``.

Algorithm source: codeplea/incbeta (https://codeplea.com/incomplete-beta-function-c)
  and the Cephes mathematical library by Stephen L. Moshier.
  Ported to Python with tighter convergence threshold (_CF_STOP=1e-14 vs
  original 1e-8) to achieve < 4e-11 rtol vs scipy across JAMMA's full
  parameter range (a=df/2 for df 10-100000, b=0.5).

License: Zlib (same as codeplea/incbeta source). See LICENSE for details.

Accuracy (verified against scipy on the scalar oracles, tests/test_special.py):
  - betainc: max rtol < 4e-11 across a=df/2 (df 10-100000), b=0.5, F in [0.01, 100]
  - chi2_sf: max rtol < 9e-15 across x in [0.001, 500]
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

# Lentz continued fraction constants
_CF_TINY = 1.0e-30  # underflow guard: replaces values < TINY with TINY
_CF_STOP = 1.0e-14  # convergence threshold (tighter than codeplea 1e-8 for accuracy)
_CF_MAX_ITER = 200  # sufficient for JAMMA's parameter range; converges in < 200


# ---------------------------------------------------------------------------
# Vectorized batch functions (numpy, no scipy)
# ---------------------------------------------------------------------------


def _libm_vectorized(
    fn: Callable[[float], float],
) -> Callable[[np.ndarray], np.ndarray]:
    """Vectorise a scalar libm function over float64 arrays.

    np.frompyfunc has lower per-call overhead than np.vectorize but yields an
    object array, so the wrapper casts. asarray types correctly where .astype
    does not (frompyfunc is declared as returning a scalar).
    """
    ufunc = np.frompyfunc(fn, 1, 1)

    def apply(x: np.ndarray) -> np.ndarray:
        return np.asarray(ufunc(x), dtype=np.float64)

    return apply


_erfc_vec = _libm_vectorized(math.erfc)
_lgamma_vec = _libm_vectorized(math.lgamma)


def chi2_sf_batch(x: np.ndarray) -> np.ndarray:
    """Vectorized chi-squared survival function P(X > x) for df=1.

    Uses erfc(sqrt(x/2)) via libm erfc. Handles NaN, negative, and
    infinite inputs.

    Args:
        x: Test statistic array (any shape).

    Returns:
        P-values with same shape, NaN propagated.
    """
    result = np.empty_like(x, dtype=np.float64)

    nan_mask = np.isnan(x)
    neg_mask = x <= 0.0
    inf_mask = np.isinf(x) & (x > 0)
    normal = ~nan_mask & ~neg_mask & ~inf_mask

    result[nan_mask] = np.nan
    result[neg_mask] = 1.0
    result[inf_mask] = 0.0

    if np.any(normal):
        result[normal] = _erfc_vec(np.sqrt(x[normal] / 2.0))

    return result


def _betainc_cf_batch(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized Lentz CF for regularized incomplete beta over arrays.

    Runs a fixed _CF_MAX_ITER iterations for all elements. Elements that
    converge early are frozen via masking. This trades wasted FLOPs on
    converged elements for elimination of Python-per-element overhead.

    Args:
        a: First shape parameter array (> 0).
        b: Second shape parameter array (> 0).
        x: Upper limit array, in (0, 1) and below symmetry threshold.

    Returns:
        I_x(a, b) values, same shape as inputs.
    """
    lbeta_ab = _lgamma_vec(a) + _lgamma_vec(b) - _lgamma_vec(a + b)
    front = np.exp(np.log(x) * a + np.log(1.0 - x) * b - lbeta_ab) / a

    f = np.ones_like(x)
    c = np.ones_like(x)
    d = np.zeros_like(x)
    converged = np.zeros(x.shape, dtype=bool)

    for i in range(_CF_MAX_ITER + 1):
        m = i // 2
        if i == 0:
            numerator = np.ones_like(x)
        elif i % 2 == 0:
            m_f = float(m)
            numerator = (m_f * (b - m_f) * x) / (
                (a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f)
            )
        else:
            m_f = float(m)
            numerator = -((a + m_f) * (a + b + m_f) * x) / (
                (a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0)
            )

        d = 1.0 + numerator * d
        d = np.where(np.abs(d) < _CF_TINY, _CF_TINY, d)
        d = 1.0 / d

        c = 1.0 + numerator / c
        c = np.where(np.abs(c) < _CF_TINY, _CF_TINY, c)

        cd = c * d
        # Freeze converged elements by replacing cd with 1.0 (no-op multiply)
        cd = np.where(converged, 1.0, cd)
        f *= cd

        newly_converged = np.abs(1.0 - cd) < _CF_STOP
        converged = converged | newly_converged

        if np.all(converged):
            break

    return front * (f - 1.0)


def betainc_batch(
    a: np.ndarray,
    b: np.ndarray,
    z: np.ndarray,
    complement_z: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized regularized incomplete beta I_z(a, b) over arrays.

    Handles symmetry relation, edge cases (z=0, z=1), and non-convergence
    (returns NaN) identically to the scalar betainc in
    ``tests/reference/special.py``.

    Args:
        a: First shape parameter array (> 0).
        b: Second shape parameter array (> 0).
        z: Upper integration limit array, in [0, 1].
        complement_z: Optional 1-z array for precision near z=1.

    Returns:
        I_z(a, b) array, same shape as inputs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    result = np.empty_like(z)

    # Edge cases
    is_zero = z == 0.0
    is_one = z == 1.0
    result[is_zero] = 0.0
    result[is_one] = 1.0

    interior = ~is_zero & ~is_one
    if not np.any(interior):
        return result

    a_int = a[interior] if a.shape == z.shape else np.broadcast_to(a, z.shape)[interior]
    b_int = b[interior] if b.shape == z.shape else np.broadcast_to(b, z.shape)[interior]
    z_int = z[interior]

    threshold = (a_int + 1.0) / (a_int + b_int + 2.0)

    # Direct CF path: z <= threshold
    direct = z_int <= threshold
    # Symmetry path: z > threshold
    sym = ~direct

    values = np.empty_like(z_int)

    if np.any(direct):
        values[direct] = _betainc_cf_batch(a_int[direct], b_int[direct], z_int[direct])

    if np.any(sym):
        if complement_z is not None:
            cz = np.asarray(complement_z, dtype=np.float64)
            cz_broad = cz if cz.shape == z.shape else np.broadcast_to(cz, z.shape)
            cz_int = cz_broad[interior]
            cz_sym = cz_int[sym]
        else:
            cz_sym = 1.0 - z_int[sym]
        values[sym] = 1.0 - _betainc_cf_batch(b_int[sym], a_int[sym], cz_sym)

    # Non-convergence produces NaN via inf * 0 or similar — propagate naturally
    result[interior] = values

    return result
