"""Dense GLS from y ~ N(Z theta, sigma² (I + lambda K)).

No JAMMA imports, triangular-cell indexing, schema, or production optimizers.
REML uses orthonormal error-contrast normalization, hence +log|Z'Z|.
SciPy is a test-only scalar optimizer and distribution reference.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, f


def _fit(kinship, design, phenotype, lam):
    """Fit one dense GLS model without using transformed/Pab identities."""
    y = np.asarray(phenotype, dtype=float)
    z = np.asarray(design, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    n, p = z.shape
    if n <= p or np.linalg.matrix_rank(z) != p:
        raise ValueError(
            "positive residual degrees of freedom and full-rank design required"
        )
    h = np.eye(n) + lam * np.asarray(kinship)
    cholesky = np.linalg.cholesky(h)
    hiz = np.linalg.solve(h, z)
    information = z.T @ hiz
    coefficients = np.linalg.solve(information, hiz.T @ y)
    residual = y - z @ coefficients
    hir = np.linalg.solve(h, residual)
    rss = float(residual @ hir)
    if rss <= 0:
        raise ValueError("positive residual sum of squares required")
    logdet_h = float(2 * np.log(np.diag(cholesky)).sum())
    df = n - p
    mle = -0.5 * (n * (np.log(2 * np.pi * rss / n) + 1) + logdet_h)
    reml_terms = np.array(
        [
            df * (np.log(2 * np.pi * rss / df) + 1),
            logdet_h,
            np.linalg.slogdet(information)[1],
            -np.linalg.slogdet(z.T @ z)[1],
        ]
    )
    reml = -0.5 * reml_terms.sum()
    return {
        "coefficients": coefficients,
        "information": information,
        "rss": rss,
        "mle": float(mle),
        "reml": float(reml),
        "reml_term_scale": float(0.5 * np.abs(reml_terms).sum()),
    }


def evaluate(kinship, covariates, genotype, phenotype, lam):
    """Return named statistics at a fixed variance ratio, using dense solves."""
    z = np.column_stack((covariates, genotype))
    fit = _fit(kinship, z, phenotype, lam)
    n, p = z.shape
    df = n - p
    information = fit["information"]
    coefficients = fit["coefficients"]
    rss = fit["rss"]
    beta = float(coefficients[-1])
    se = float(np.sqrt(rss / df * np.linalg.inv(information)[-1, -1]))
    return {
        "beta": beta,
        "se": se,
        "p_wald": float(f.sf((beta / se) ** 2, 1, df)),
        "reml": fit["reml"],
        "mle": fit["mle"],
        "rss": rss,
        "reml_term_scale": fit["reml_term_scale"],
    }


def optimize_null(kinship, covariates, phenotype, *, bounds=(1e-5, 1e5)):
    """Independently optimize the covariate-only MLE on the constrained interval."""
    return _optimize_design(
        kinship, np.asarray(covariates), phenotype, objective="mle", bounds=bounds
    )


def _optimize_design(kinship, design, phenotype, *, objective, bounds):
    def value(log_lam):
        return _fit(kinship, design, phenotype, np.exp(log_lam))[objective]

    grid = np.linspace(np.log(bounds[0]), np.log(bounds[1]), 129)
    values = np.array([value(point) for point in grid])
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    for index in range(1, len(grid) - 1):
        if values[index] >= max(values[index - 1], values[index + 1]):
            fitted = minimize_scalar(
                lambda point: -value(point),
                bounds=(grid[index - 1], grid[index + 1]),
                method="bounded",
                options={"xatol": 1e-11},
            )
            candidates.append((-fitted.fun, fitted.x))
    _, log_lam = max(candidates)
    lam = float(np.exp(log_lam))
    return {"lambda": lam, "log_likelihood": float(value(log_lam))}


def all_test_statistics(
    kinship, covariates, genotype, phenotype, *, bounds=(1e-5, 1e5)
):
    """Return six independently evaluated fields from the all-tests contract."""
    reml = optimize(
        kinship, covariates, genotype, phenotype, objective="reml", bounds=bounds
    )
    mle = optimize(
        kinship, covariates, genotype, phenotype, objective="mle", bounds=bounds
    )
    null = optimize_null(kinship, covariates, phenotype, bounds=bounds)
    fixed_null = _fit(kinship, covariates, phenotype, null["lambda"])
    score_fit = evaluate(kinship, covariates, genotype, phenotype, null["lambda"])
    h = np.eye(len(phenotype)) + null["lambda"] * np.asarray(kinship)
    hi = np.linalg.solve(h, np.eye(len(phenotype)))
    w = np.asarray(covariates)
    if w.ndim == 1:
        w = w[:, None]
    projector = hi - hi @ w @ np.linalg.solve(w.T @ hi @ w, w.T @ hi)
    x = np.asarray(genotype)
    y = np.asarray(phenotype)
    p_xx = float(x @ projector @ x)
    p_xy = float(x @ projector @ y)
    p_yy = float(y @ projector @ y)
    score_f = len(y) * p_xy * p_xy / (p_yy * p_xx)
    lrt = max(0.0, 2.0 * (mle["logl_H1"] - fixed_null["mle"]))
    return {
        "beta": reml["beta"],
        "se": reml["se"],
        "score_beta": score_fit["beta"],
        "score_se": score_fit["se"],
        "logl_H1": mle["logl_H1"],
        "l_remle": reml["l_remle"],
        "l_mle": mle["l_mle"],
        "p_wald": reml["p_wald"],
        "p_lrt": float(chi2.sf(lrt, 1)),
        "p_score": float(f.sf(score_f, 1, len(y) - w.shape[1] - 1)),
        "reml_log_likelihood": reml["reml"],
        "null_mle_log_likelihood": fixed_null["mle"],
    }


def optimize(
    kinship, covariates, genotype, phenotype, *, objective="reml", bounds=(1e-5, 1e5)
):
    """Search every local grid peak with a test-only scalar solver, keep endpoints."""

    design = np.column_stack((covariates, genotype))
    optimized = _optimize_design(
        kinship, design, phenotype, objective=objective, bounds=bounds
    )
    lam = optimized["lambda"]
    result = evaluate(kinship, covariates, genotype, phenotype, lam)
    return {
        "l_remle" if objective == "reml" else "l_mle": lam,
        "logl_H1": result[objective],
        **result,
    }


def projection_products(kinship, vectors, lam):
    """Full named-vector products after removing each preceding design column.

    Each level is solved afresh. No recursive Schur updates or packed indexing.
    The final vector is phenotype, so the last level removes every design column.
    """
    v = np.asarray(vectors, dtype=float)
    hi = np.linalg.solve(np.eye(len(v)) + lam * kinship, np.eye(len(v)))
    levels = []
    for level in range(v.shape[1]):
        z = v[:, :level]
        projection = (
            hi if level == 0 else hi - hi @ z @ np.linalg.solve(z.T @ hi @ z, z.T @ hi)
        )
        levels.append(v.T @ projection @ v)
    return np.array(levels)
