"""Lambda optimization via custom Brent's method.

Implements bounded scalar minimization for REML log-likelihood optimization.
Uses pure Python/JAX implementation - no scipy dependency.

Reference: Brent, R.P. (1973) "Algorithms for Minimization without Derivatives"
"""

import warnings
from typing import Callable

import numpy as np

from jamma.lmm.likelihood import reml_log_likelihood


def brent_minimize(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-5,
    maxiter: int = 500,
) -> tuple[float, float]:
    """Minimize a scalar function using Brent's method.

    Combines golden section search with parabolic interpolation for
    efficient bounded minimization without derivatives.

    Args:
        func: Scalar function to minimize
        a: Lower bound of search interval
        b: Upper bound of search interval
        tol: Convergence tolerance (default: 1e-5, matches GEMMA)
        maxiter: Maximum iterations (default: 500)

    Returns:
        Tuple of (x_min, f_min) where x_min is the minimizer and f_min is the minimum value
    """
    # Golden ratio constant
    golden = 0.5 * (3.0 - np.sqrt(5.0))

    # Ensure a < b
    if a > b:
        a, b = b, a

    # Initialize: x is current best, w is second best, v is previous w
    x = w = v = a + golden * (b - a)
    fx = fw = fv = func(x)

    # d and e track step sizes
    d = 0.0
    e = 0.0

    for _ in range(maxiter):
        midpoint = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1

        # Check for convergence
        if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
            return x, fx

        # Attempt parabolic interpolation
        if abs(e) > tol1:
            # Fit parabola through x, w, v
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)

            if q > 0:
                p = -p
            else:
                q = -q

            r = e
            e = d

            # Check if parabolic step is acceptable
            if abs(p) < abs(0.5 * q * r) and p > q * (a - x) and p < q * (b - x):
                # Take parabolic step
                d = p / q
                u = x + d

                # Don't evaluate too close to bounds
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if x < midpoint else -tol1
            else:
                # Fall back to golden section
                e = (b if x < midpoint else a) - x
                d = golden * e
        else:
            # Golden section step
            e = (b if x < midpoint else a) - x
            d = golden * e

        # Ensure step is at least tol1
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d > 0 else -tol1)

        fu = func(u)

        # Update interval and best points
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    # Max iterations reached
    warnings.warn(
        f"Brent optimization did not converge in {maxiter} iterations",
        RuntimeWarning,
        stacklevel=2,
    )
    return x, fx


def optimize_lambda(
    reml_func: Callable[[float], float],
    l_min: float = 1e-5,
    l_max: float = 1e5,
    tol: float = 1e-5,
) -> tuple[float, float]:
    """Optimize lambda (variance ratio) using custom Brent's method.

    Finds the lambda that minimizes the negative REML log-likelihood
    (equivalently, maximizes the REML log-likelihood).

    Args:
        reml_func: Function taking lambda and returning negative log-likelihood
        l_min: Lower bound for lambda search (default: 1e-5, GEMMA default)
        l_max: Upper bound for lambda search (default: 1e5, GEMMA default)
        tol: Convergence tolerance (default: 1e-5, GEMMA's xatol)

    Returns:
        Tuple of (optimal_lambda, log_likelihood) where log_likelihood is positive
    """
    # Find minimum of negative log-likelihood
    lambda_opt, neg_logl = brent_minimize(reml_func, l_min, l_max, tol=tol)

    # Check for boundary optimum (within 1% of bounds)
    if lambda_opt <= l_min * 1.01:
        warnings.warn(
            f"Lambda optimization converged at lower bound ({lambda_opt:.2e} ~ {l_min:.2e}). "
            "True optimum may be below search range.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif lambda_opt >= l_max * 0.99:
        warnings.warn(
            f"Lambda optimization converged at upper bound ({lambda_opt:.2e} ~ {l_max:.2e}). "
            "True optimum may be above search range.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Return positive log-likelihood
    return lambda_opt, -neg_logl


def optimize_lambda_for_snp(
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float, float]:
    """Convenience wrapper for lambda optimization.

    Creates the REML function closure and optimizes lambda for a given SNP.

    Args:
        eigenvalues: Kinship matrix eigenvalues (n_samples,)
        Uab: Matrix products from compute_Uab
        n_cvt: Number of covariates
        l_min: Lower bound for lambda search
        l_max: Upper bound for lambda search

    Returns:
        Tuple of (lambda_opt, logl_H1)
    """

    def reml_func(l: float) -> float:
        return reml_log_likelihood(l, eigenvalues, Uab, n_cvt)

    return optimize_lambda(reml_func, l_min=l_min, l_max=l_max)
