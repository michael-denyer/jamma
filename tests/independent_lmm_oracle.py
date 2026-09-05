"""Dense LMM oracle isolated from JAMMA's Pab and likelihood implementations."""

from __future__ import annotations

import numpy as np


def _design(UtW: np.ndarray, Utg: np.ndarray) -> np.ndarray:
    covariates = np.asarray(UtW, dtype=np.float64)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    genotype = np.asarray(Utg, dtype=np.float64)
    if genotype.ndim != 1 or genotype.shape[0] != covariates.shape[0]:
        raise ValueError("Utg must be one vector with the same rows as UtW")
    return np.column_stack((covariates, genotype))


def dense_lmm_log_likelihood(
    eigenvalues: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    Utg: np.ndarray,
    lambda_value: float,
    *,
    restricted: bool,
) -> float:
    """Evaluate the profiled transformed-basis LMM objective directly."""
    design = _design(UtW, Utg)
    y = np.asarray(Uty, dtype=np.float64)
    weights = 1.0 / (lambda_value * np.asarray(eigenvalues) + 1.0)
    weighted_design = weights[:, None] * design
    gram = design.T @ weighted_design
    rhs = design.T @ (weights * y)
    residual_ss = float(y @ (weights * y) - rhs @ np.linalg.solve(gram, rhs))
    n_samples, n_parameters = design.shape
    df = n_samples - n_parameters
    if residual_ss <= 0.0 or df <= 0:
        return float("nan")

    logdet_h = float(np.log(lambda_value * eigenvalues + 1.0).sum())
    if restricted:
        sign_weighted, logdet_weighted = np.linalg.slogdet(gram)
        sign_identity, logdet_identity = np.linalg.slogdet(design.T @ design)
        if sign_weighted <= 0.0 or sign_identity <= 0.0:
            return float("nan")
        constant = 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)
        return float(
            constant
            - 0.5 * logdet_h
            - 0.5 * (logdet_weighted - logdet_identity)
            - 0.5 * df * np.log(residual_ss)
        )

    constant = 0.5 * n_samples * (np.log(n_samples) - np.log(2.0 * np.pi) - 1.0)
    return float(constant - 0.5 * logdet_h - 0.5 * n_samples * np.log(residual_ss))


def dense_reml_score_log_lambda(
    eigenvalues: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    Utg: np.ndarray,
    lambda_value: float,
) -> float:
    """Evaluate the REML log-lambda score from dense projector algebra."""
    design = _design(UtW, Utg)
    y = np.asarray(Uty, dtype=np.float64)
    d = np.asarray(eigenvalues, dtype=np.float64)
    h = 1.0 / (1.0 + lambda_value * d)
    gram = design.T @ (h[:, None] * design)
    projected_y = h * y - (h[:, None] * design) @ np.linalg.solve(
        gram, design.T @ (h * y)
    )
    hi_k_hi_design = (d * h * h)[:, None] * design
    trace_pk = float(
        np.sum(d * h) - np.trace(np.linalg.solve(gram, design.T @ hi_k_hi_design))
    )
    ypy = float(y @ projected_y)
    ypkpy = float(projected_y @ (d * projected_y))
    df = len(y) - design.shape[1]
    return lambda_value * (-0.5 * trace_pk + 0.5 * df * ypkpy / ypy)


def dense_wald_at_lambda(
    eigenvalues: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    Utg: np.ndarray,
    lambda_value: float,
) -> tuple[float, float, float]:
    """Return beta, standard error, and F statistic from a dense GLS solve."""
    design = _design(UtW, Utg)
    y = np.asarray(Uty, dtype=np.float64)
    weights = 1.0 / (lambda_value * np.asarray(eigenvalues) + 1.0)
    gram = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(gram)
    beta = covariance @ (design.T @ (weights * y))
    residual = y - design @ beta
    df = design.shape[0] - design.shape[1]
    sigma2 = float(residual @ (weights * residual) / df)
    genotype_se = float(np.sqrt(sigma2 * covariance[-1, -1]))
    genotype_beta = float(beta[-1])
    return genotype_beta, genotype_se, (genotype_beta / genotype_se) ** 2
