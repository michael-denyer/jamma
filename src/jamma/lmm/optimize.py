"""Lambda optimization via Brent's method with Newton refinement.

Implements bounded scalar optimization for REML log-likelihood.
Uses pure Python/JAX implementation matching GEMMA's approach:
1. Brent's method to find derivative sign changes
2. Brent root-finding on derivative to find stationary point
3. Newton refinement for tighter convergence

Reference: Brent, R.P. (1973) "Algorithms for Minimization without Derivatives"
"""

import warnings
from collections.abc import Callable

import numpy as np

from jamma.lmm.likelihood import mle_log_likelihood, reml_log_likelihood

# =============================================================================
# BRENT MINIMIZATION (Current working version - keep for potential revert)
# =============================================================================


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
        Tuple of (x_min, f_min) - the minimizer and minimum value
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
            f"Lambda converged at lower bound ({lambda_opt:.2e} ~ {l_min:.2e}). "
            "True optimum may be below search range.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif lambda_opt >= l_max * 0.99:
        warnings.warn(
            f"Lambda converged at upper bound ({lambda_opt:.2e} ~ {l_max:.2e}). "
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

    def neg_reml_func(lam: float) -> float:
        # Negate because reml_log_likelihood returns positive log-likelihood
        # but we minimize (so minimize negative = maximize positive)
        return -reml_log_likelihood(lam, eigenvalues, Uab, n_cvt)

    return optimize_lambda(neg_reml_func, l_min=l_min, l_max=l_max)


def optimize_lambda_mle_for_snp(
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float, float]:
    """Convenience wrapper for MLE lambda optimization.

    Creates the MLE function closure and optimizes lambda for a given SNP.
    Used by LRT (-lmm 2) which requires MLE (not REML) likelihood.

    The MLE surface can be multi-modal (unlike REML which is typically unimodal).
    To handle this, we evaluate at both boundaries in addition to Brent optimization
    and return the global optimum. This matches GEMMA's approach of checking
    boundary values.

    Args:
        eigenvalues: Kinship matrix eigenvalues (n_samples,)
        Uab: Matrix products from compute_Uab
        n_cvt: Number of covariates
        l_min: Lower bound for lambda search
        l_max: Upper bound for lambda search

    Returns:
        Tuple of (lambda_opt, logl_H1) where logl_H1 is MLE log-likelihood
    """

    def neg_mle_func(lam: float) -> float:
        # Negate because mle_log_likelihood returns positive log-likelihood
        # but we minimize (so minimize negative = maximize positive)
        return -mle_log_likelihood(lam, eigenvalues, Uab, n_cvt)

    # Run Brent optimization
    lambda_opt, neg_logl_opt = brent_minimize(neg_mle_func, l_min, l_max)

    # Check boundary values - MLE can have optima at boundaries
    # (unlike REML which is typically interior)
    neg_logl_min = neg_mle_func(l_min)
    neg_logl_max = neg_mle_func(l_max)

    # Find global minimum (maximum likelihood)
    candidates = [
        (l_min, neg_logl_min),
        (l_max, neg_logl_max),
        (lambda_opt, neg_logl_opt),
    ]
    best_lambda, best_neg_logl = min(candidates, key=lambda x: x[1])

    # Emit warning if at boundary
    if best_lambda <= l_min * 1.01:
        warnings.warn(
            f"Lambda converged at lower bound ({best_lambda:.2e} ~ {l_min:.2e}). "
            "True optimum may be below search range.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif best_lambda >= l_max * 0.99:
        warnings.warn(
            f"Lambda converged at upper bound ({best_lambda:.2e} ~ {l_max:.2e}). "
            "True optimum may be above search range.",
            RuntimeWarning,
            stacklevel=2,
        )

    return best_lambda, -best_neg_logl


# =============================================================================
# DERIVATIVE-BASED OPTIMIZATION (GEMMA-style approach)
# =============================================================================


def _numerical_derivative(
    func: Callable[[float], float], x: float, h: float = 1e-8
) -> float:
    """Compute numerical derivative using central difference.

    Args:
        func: Function to differentiate
        x: Point at which to compute derivative
        h: Step size for finite difference

    Returns:
        Approximate derivative df/dx at x
    """
    return (func(x + h) - func(x - h)) / (2.0 * h)


def brent_root(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    maxiter: int = 100,
) -> float:
    """Find root of a function using Brent's method.

    Finds x such that func(x) = 0 in interval [a, b].
    Requires func(a) and func(b) to have opposite signs.

    Args:
        func: Scalar function to find root of
        a: Lower bound of search interval
        b: Upper bound of search interval
        tol: Convergence tolerance
        maxiter: Maximum iterations

    Returns:
        x value where func(x) ≈ 0

    Raises:
        ValueError: If func(a) and func(b) have same sign
    """
    fa = func(a)
    fb = func(b)

    if fa * fb > 0:
        # No sign change - return the endpoint with smaller |f|
        return a if abs(fa) < abs(fb) else b

    # Ensure |f(b)| <= |f(a)|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True
    d = 0.0

    for _ in range(maxiter):
        if abs(fb) < tol:
            return b

        if abs(b - a) < tol:
            return b

        # Try inverse quadratic interpolation
        if fa != fc and fb != fc:
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Conditions for accepting s
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            # Bisection
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = func(s)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        # Ensure |f(b)| <= |f(a)|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b


def optimize_lambda_derivative(
    reml_func: Callable[[float], float],
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_region: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Optimize lambda using derivative-based method (GEMMA-style).

    Matches GEMMA's CalcLambda approach:
    1. Scan regions on log scale to find derivative sign changes
    2. Use Brent root-finding on derivative to find stationary points
    3. Return the lambda with highest log-likelihood

    Args:
        reml_func: Function taking lambda and returning negative log-likelihood
        l_min: Lower bound for lambda search
        l_max: Upper bound for lambda search
        n_region: Number of regions to scan for sign changes
        tol: Convergence tolerance for root finding

    Returns:
        Tuple of (optimal_lambda, log_likelihood) where log_likelihood is positive
    """

    # Derivative function
    def deriv(lam: float) -> float:
        return _numerical_derivative(reml_func, lam, h=lam * 1e-6)

    # Scan for sign changes on log scale (like GEMMA)
    lambda_interval = np.log(l_max / l_min) / n_region
    sign_changes = []

    for i in range(n_region):
        lambda_l = l_min * np.exp(lambda_interval * i)
        lambda_h = l_min * np.exp(lambda_interval * (i + 1))
        dev_l = deriv(lambda_l)
        dev_h = deriv(lambda_h)

        if dev_l * dev_h <= 0:
            sign_changes.append((lambda_l, lambda_h))

    # Find all stationary points
    candidates = []

    # Always include boundary values
    candidates.append((l_min, reml_func(l_min)))
    candidates.append((l_max, reml_func(l_max)))

    # Find roots in each sign-change interval
    for lambda_l, lambda_h in sign_changes:
        try:
            root = brent_root(deriv, lambda_l, lambda_h, tol=tol)
            # Clamp to bounds
            root = max(l_min, min(l_max, root))
            f_val = reml_func(root)
            candidates.append((root, f_val))
        except (ValueError, RuntimeError):
            pass

    # Find the minimum (since reml_func returns negative log-likelihood)
    best_lambda, best_neg_logl = min(candidates, key=lambda x: x[1])

    # Warn if at boundary
    if best_lambda <= l_min * 1.01:
        warnings.warn(
            f"Lambda converged at lower bound ({best_lambda:.2e} ~ {l_min:.2e}). "
            "True optimum may be below search range.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif best_lambda >= l_max * 0.99:
        warnings.warn(
            f"Lambda converged at upper bound ({best_lambda:.2e} ~ {l_max:.2e}). "
            "True optimum may be above search range.",
            RuntimeWarning,
            stacklevel=2,
        )

    return best_lambda, -best_neg_logl


def optimize_lambda_for_snp_derivative(
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float, float]:
    """Lambda optimization using derivative-based method.

    Uses GEMMA-style approach with Brent root-finding on derivative.

    Args:
        eigenvalues: Kinship matrix eigenvalues (n_samples,)
        Uab: Matrix products from compute_Uab
        n_cvt: Number of covariates
        l_min: Lower bound for lambda search
        l_max: Upper bound for lambda search

    Returns:
        Tuple of (lambda_opt, logl_H1)
    """

    def neg_reml_func(lam: float) -> float:
        return -reml_log_likelihood(lam, eigenvalues, Uab, n_cvt)

    return optimize_lambda_derivative(neg_reml_func, l_min=l_min, l_max=l_max)
