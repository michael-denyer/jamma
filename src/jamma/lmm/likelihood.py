"""REML log-likelihood computation using JAX.

Implements the restricted maximum likelihood (REML) function for
variance component estimation in LMM. Uses JAX for vectorized computation.
"""

from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import config

# Ensure 64-bit precision
config.update("jax_enable_x64", True)


def get_ab_index(a: int, b: int) -> int:
    """Compute index for symmetric matrix storage.

    GEMMA's GetabIndex formula for accessing Pab/Uab elements.
    Uses lower-triangular storage: index = a*(a+1)/2 + b for a >= b.

    Args:
        a: First index
        b: Second index

    Returns:
        Linear index into packed storage
    """
    if a >= b:
        return a * (a + 1) // 2 + b
    return b * (b + 1) // 2 + a


def compute_Uab(
    UtW: np.ndarray, Uty: np.ndarray, Utx: np.ndarray | None = None
) -> np.ndarray:
    """Compute Uab matrix products used in REML likelihood.

    Uab stores cumulative products needed for REML:
    - ab[0,0] = sum(UtW_0^2), ab[1,0] = sum(UtW_0*UtW_1), etc.
    - ab[n_cvt, n_cvt] = sum(Uty^2)
    - If Utx provided: ab[n_cvt+1, *] includes genotype products

    Args:
        UtW: Rotated covariates (n_samples, n_cvt)
        Uty: Rotated phenotype (n_samples,)
        Utx: Rotated genotype for current SNP (n_samples,) - optional

    Returns:
        Uab array with cumulative products
    """
    n = len(Uty)
    n_cvt = UtW.shape[1] if UtW.ndim > 1 else 1

    # Determine size: n_cvt + 1 (for y) + 1 (for x if provided)
    n_terms = n_cvt + 1 + (1 if Utx is not None else 0)
    n_elements = n_terms * (n_terms + 1) // 2

    # Use JAX for vectorized computation
    UtW_jax = jnp.array(UtW.reshape(n, -1) if UtW.ndim == 1 else UtW)
    Uty_jax = jnp.array(Uty)

    # Build Uab array
    Uab = np.zeros(n_elements, dtype=np.float64)

    # Covariate-covariate products
    for a in range(n_cvt):
        for b in range(a + 1):
            idx = get_ab_index(a, b)
            Uab[idx] = float(jnp.sum(UtW_jax[:, a] * UtW_jax[:, b]))

    # Covariate-phenotype products
    for a in range(n_cvt):
        idx = get_ab_index(n_cvt, a)
        Uab[idx] = float(jnp.sum(UtW_jax[:, a] * Uty_jax))

    # Phenotype-phenotype product
    idx = get_ab_index(n_cvt, n_cvt)
    Uab[idx] = float(jnp.sum(Uty_jax * Uty_jax))

    # If genotype provided, add genotype products
    if Utx is not None:
        Utx_jax = jnp.array(Utx)

        # Covariate-genotype products
        for a in range(n_cvt):
            idx = get_ab_index(n_cvt + 1, a)
            Uab[idx] = float(jnp.sum(UtW_jax[:, a] * Utx_jax))

        # Phenotype-genotype product
        idx = get_ab_index(n_cvt + 1, n_cvt)
        Uab[idx] = float(jnp.sum(Uty_jax * Utx_jax))

        # Genotype-genotype product
        idx = get_ab_index(n_cvt + 1, n_cvt + 1)
        Uab[idx] = float(jnp.sum(Utx_jax * Utx_jax))

    return Uab


def compute_pab(
    Hi_eval: np.ndarray, Uab: np.ndarray, ab_index_func: Callable[[int, int], int]
) -> np.ndarray:
    """Compute Pab products from Uab and H_inv weighting.

    Pab[i] = sum_j(Hi_eval[j] * Uab_element[j])

    This is the weighted version of Uab used in Wald test statistics.

    Args:
        Hi_eval: 1 / (lambda * eigenvalues + 1) vector (n_samples,)
        Uab: Matrix products from compute_Uab
        ab_index_func: Function to compute index (typically get_ab_index)

    Returns:
        Pab array with H_inv weighted products
    """
    # For now, return same-sized array with weighted sums
    # The actual Pab computation depends on the rotated data structure
    Pab = np.zeros_like(Uab)

    # Weight each Uab element by sum of Hi_eval
    # This is simplified; full implementation needs per-element weighting
    weight_sum = float(jnp.sum(jnp.array(Hi_eval)))
    Pab = Uab * weight_sum / len(Hi_eval)

    return Pab


def reml_log_likelihood(
    lambda_val: float, eigenvalues: np.ndarray, Uab: np.ndarray, n_cvt: int
) -> float:
    """Compute REML log-likelihood for given lambda (variance ratio).

    The REML log-likelihood for the null model (no SNP effect):
    logRL = -0.5 * [logdet(H) + logdet(WHW) + (n-c)*log(yPy)]

    where H = lambda*K + I, P is the projection matrix.

    Args:
        lambda_val: Variance component ratio (sigma_g^2 / sigma_e^2)
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products from compute_Uab
        n_cvt: Number of covariates

    Returns:
        Negative log-likelihood (for minimization)
    """
    eigenvalues_jax = jnp.array(eigenvalues)
    n = len(eigenvalues)

    # H_inv = 1 / (lambda * eigenvalues + 1)
    Hi_eval = 1.0 / (lambda_val * eigenvalues_jax + 1.0)

    # Log determinant of H: sum(log(lambda * d + 1))
    logdet_h = float(jnp.sum(jnp.log(lambda_val * eigenvalues_jax + 1.0)))

    # Get Pab elements for likelihood computation
    # P[y,y] index
    yy_idx = get_ab_index(n_cvt, n_cvt)
    Pyy = float(jnp.sum(Hi_eval)) * Uab[yy_idx] / n

    # Compute log-likelihood components
    # Simplified version - full implementation needs proper P matrix
    df = n - n_cvt
    if Pyy > 0:
        log_yPy = np.log(Pyy)
    else:
        log_yPy = 0.0

    # REML log-likelihood (negated for minimization)
    logRL = -0.5 * (logdet_h + df * log_yPy)

    return -logRL  # Return negative for minimization
