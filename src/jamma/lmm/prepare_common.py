"""Pure-NumPy setup utilities shared by NumPy LMM runners.

Provides covariate matrix construction, eigendecomposition handling,
null model computation, and shared input validation. The NumPy runner
(runner_numpy.py) imports from this module.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
from loguru import logger

from jamma.core.constants import PHENOTYPE_MISSING
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import (
    compute_null_model_lambda,
    compute_null_model_mle,
    compute_Uab,
    finite_difference_dev2,
)
from jamma.utils.logging import log_rss_memory


def compute_valid_mask(
    phenotypes: np.ndarray, covariates: np.ndarray | None
) -> np.ndarray:
    """Compute boolean mask of samples with valid phenotype and covariate values.

    Args:
        phenotypes: Phenotype vector (n_samples,).
        covariates: Covariate matrix (n_samples, n_cvt) or None.

    Returns:
        Boolean mask array of shape (n_samples,) where True indicates
        a sample with valid phenotype and covariate values.
    """
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != PHENOTYPE_MISSING)
    if covariates is not None:
        valid_covariate = np.all(~np.isnan(covariates), axis=1)
        valid_mask = valid_mask & valid_covariate
    return valid_mask


@dataclass(frozen=True, slots=True)
class RunnerSetup:
    """Validated and filtered inputs for LMM runners.

    Returned by validate_runner_inputs() after applying the valid-sample
    mask, checking all invariants, and validating eigenpair dimensions.

    Attributes:
        phenotypes: Filtered phenotype vector (n_samples,).
        kinship: Filtered kinship matrix (n_samples, n_samples) or None.
        covariates: Filtered covariate matrix (n_samples, n_cvt) or None.
        eigenvalues: Pre-computed eigenvalues (n_samples,) or None.
        eigenvectors: Pre-computed eigenvectors (n_samples, n_samples) or None.
        valid_mask: Boolean mask used to filter samples (original length).
        n_samples: Number of valid samples after filtering.
    """

    phenotypes: np.ndarray
    kinship: np.ndarray | None
    covariates: np.ndarray | None
    eigenvalues: np.ndarray | None
    eigenvectors: np.ndarray | None
    valid_mask: np.ndarray
    n_samples: int


def validate_runner_inputs(
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    lmm_mode: int,
) -> RunnerSetup:
    """Validate LMM runner inputs and apply sample filtering.

    Performs the common validation sequence shared by both runners
    (numpy batch, numpy streaming): eigendecomposition pair check, lmm_mode guard,
    kinship/eigenvalue guard, valid-sample mask computation and application,
    empty-sample guard, and eigenpair dimension validation.

    Does NOT include: memory checks (differ between batch/streaming).

    Args:
        phenotypes: Phenotype vector (n_samples,), with NaN for missing.
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues/eigenvectors are provided.
        covariates: Covariate matrix (n_samples, n_cvt) or None.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.

    Returns:
        RunnerSetup with filtered arrays and validated n_samples.

    Raises:
        ValueError: If only one of eigenvalues/eigenvectors is provided,
            if lmm_mode is not in (1, 2, 3, 4), if neither kinship nor
            eigenvalues are provided, if no valid samples remain after
            filtering, or if eigenpair dimensions do not match n_samples.
    """
    # Validate eigendecomposition params — must provide both or neither
    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )

    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    if kinship is None and eigenvalues is None:
        raise ValueError(
            "Either kinship or pre-computed eigendecomposition (eigenvalues + "
            "eigenvectors) must be provided"
        )

    # Compute valid-sample mask from phenotype and covariate NaN
    valid_mask = compute_valid_mask(phenotypes, covariates)

    # Apply mask only when needed (avoid a copy if all samples are valid)
    if not np.all(valid_mask):
        phenotypes = phenotypes[valid_mask]
        if kinship is not None:
            kinship = kinship[np.ix_(valid_mask, valid_mask)]
        if covariates is not None:
            covariates = covariates[valid_mask, :]

    n_samples = phenotypes.shape[0]
    if n_samples == 0:
        raise ValueError(
            "No valid samples: all phenotypes are missing or -9"
            + (", or all have missing covariates" if covariates is not None else "")
        )

    # Validate precomputed eigenpair dimensions against (possibly filtered) n_samples
    if eigenvalues is not None and eigenvectors is not None:
        hint = (
            "Recompute eigenpairs on the filtered kinship, or pass kinship= "
            "and let JAMMA compute the eigendecomposition."
        )
        if eigenvalues.shape[0] != n_samples:
            raise ValueError(
                f"eigenvalues length ({eigenvalues.shape[0]}) does not match "
                f"n_samples ({n_samples}) after removing missing "
                f"phenotypes/covariates. {hint}"
            )
        if eigenvectors.shape != (n_samples, n_samples):
            raise ValueError(
                f"eigenvectors shape {eigenvectors.shape} does not match "
                f"({n_samples}, {n_samples}) after removing missing "
                f"phenotypes/covariates. {hint}"
            )

    return RunnerSetup(
        phenotypes=phenotypes,
        kinship=kinship,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        valid_mask=valid_mask,
        n_samples=n_samples,
    )


def _build_covariate_matrix(
    covariates: np.ndarray | None, n_samples: int
) -> tuple[np.ndarray, int]:
    """Construct covariate matrix W and return (W, n_cvt).

    If covariates is None, uses intercept-only model. Warns if provided
    covariates lack an intercept column.

    Args:
        covariates: Optional covariate matrix (n_samples, n_covariates).
        n_samples: Number of samples (for intercept construction).

    Returns:
        Tuple of (W, n_cvt) where W is the covariate matrix.
    """
    if covariates is None:
        W = np.ones((n_samples, 1))
    else:
        W = covariates.astype(np.float64)
        if not np.allclose(W[:, 0], 1.0):
            logger.warning(
                "Covariate matrix does not have intercept column "
                "(first column is not all 1s). "
                "Model will NOT include an intercept term."
            )
    n_cvt = W.shape[1]
    # df = n_samples - n_cvt - 1 must be positive for valid REML
    if n_samples <= n_cvt + 1:
        raise ValueError(
            f"Over-parameterized model: {n_samples} samples with {n_cvt} "
            f"covariates leaves df={n_samples - n_cvt - 1} "
            f"(need at least {n_cvt + 2} samples)"
        )
    # Rank-deficient covariates cause singular Pab → cryptic LAPACK errors
    rank = np.linalg.matrix_rank(W)
    if rank < n_cvt:
        raise ValueError(
            f"Covariate matrix is rank-deficient: rank={rank} but "
            f"n_cvt={n_cvt}. Check for linearly dependent columns."
        )
    return W, n_cvt


def _eigendecompose_or_reuse(
    kinship: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    show_progress: bool,
    label: str,
    *,
    check_memory: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigendecomposition, computing it if not provided.

    Args:
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues and eigenvectors are provided.
        eigenvalues: Pre-computed eigenvalues or None.
        eigenvectors: Pre-computed eigenvectors or None.
        show_progress: Whether to log memory usage.
        label: Label for memory logging (e.g. "lmm", "lmm_streaming").
        check_memory: If True (default), check available memory before
            eigendecomposition.

    Returns:
        Tuple of (eigenvalues, eigenvectors).
    """
    if eigenvalues is not None and eigenvectors is not None:
        if show_progress:
            logger.debug("Using pre-computed eigendecomposition")
        return eigenvalues, eigenvectors

    if kinship is None:
        raise ValueError(
            "Must provide either (eigenvalues, eigenvectors) or kinship matrix. "
            "All three are None."
        )

    if show_progress:
        log_rss_memory(label, "before_eigendecomp")
    eigenvalues_np, U = eigendecompose_kinship(kinship, check_memory=check_memory)
    # Release LAPACK DSYEVD workspace before LMM phase
    gc.collect()
    if show_progress:
        log_rss_memory(label, "after_eigendecomp")
    return eigenvalues_np, U


def _compute_null_model_common(
    lmm_mode: int,
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    show_progress: bool,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float | None, float | None, np.ndarray | None]:
    """Compute null model MLE for Score, LRT, and All-tests modes.

    Pure-NumPy version of the null model computation. Returns Hi_eval_null
    as a plain NumPy array.

    GEMMA computes both REML and MLE null lambdas in CalcLambda, but uses
    MLE lambda for Hi_eval in the Score test:
    Hi_eval_null = 1 / (lambda_null_mle * eigenvalues + 1).

    Wald (mode 1) skips this entirely; LRT (mode 2) needs only logl_H0;
    Score/All (modes 3, 4) precompute Hi_eval at the null lambda.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        eigenvalues_np: Kinship eigenvalues as numpy array.
        UtW: Rotated covariates.
        Uty: Rotated phenotype.
        n_cvt: Number of covariates.
        show_progress: Whether to log results.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.

    Returns:
        Tuple of (logl_H0, lambda_null_mle, Hi_eval_null_np).
        All None for Wald (mode 1). For LRT (mode 2), Hi_eval_null_np
        is None. For Score/All (modes 3, 4), all three are populated.
    """
    if lmm_mode == 1:
        return None, None, None
    if lmm_mode not in (2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    lambda_null_mle, logl_H0 = compute_null_model_mle(
        eigenvalues_np, UtW, Uty, n_cvt, l_min=l_min, l_max=l_max
    )
    if show_progress:
        logger.info(
            f"Null model MLE: lambda={lambda_null_mle:.6f}, logl_H0={logl_H0:.6f}"
        )

    Hi_eval_null_np = None
    if lmm_mode in (3, 4):
        Hi_eval_null_np = 1.0 / (lambda_null_mle * eigenvalues_np + 1.0)
        if not np.all(np.isfinite(Hi_eval_null_np)):
            bad_idx = np.where(~np.isfinite(Hi_eval_null_np))[0]
            raise ValueError(
                f"Hi_eval_null has {len(bad_idx)} non-finite value(s) at indices "
                f"{bad_idx[:5].tolist()}. lambda_null_mle={lambda_null_mle:.6g}. "
                "Null model optimization may have failed."
            )
        if not np.all(Hi_eval_null_np > 0):
            bad_idx = np.where(~(Hi_eval_null_np > 0))[0]
            raise ValueError(
                f"Hi_eval_null has {len(bad_idx)} non-positive value(s) at indices "
                f"{bad_idx[:5].tolist()}. lambda_null_mle={lambda_null_mle:.6g}. "
                "Check kinship matrix for negative eigenvalues."
            )

    return logl_H0, lambda_null_mle, Hi_eval_null_np


def compute_and_log_pve(
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float, float | None]:
    """Compute PVE and se(PVE) from null model REML lambda.

    PVE = lambda * trace(K) / (lambda * trace(K) + n), where lambda = vg/ve
    is the REML estimate under the null model (no genotype effect) and
    trace(K) = sum(eigenvalues). This trace-adjusted formula matches GEMMA's
    CalcPve which accounts for kinship matrices whose trace != n.

    se(PVE) is computed via the delta method: se(lambda) from the REML
    second derivative at the optimum, then propagated through the PVE
    transformation using d(PVE)/d(lambda) = trace_G / (trace_G * lambda + 1)^2
    where trace_G = trace(K) / n.

    Called by all LMM runners after eigendecomp + rotation, regardless of
    lmm_mode. The REML null lambda optimization is cheap (single golden
    section search, ~20 iterations).

    Args:
        eigenvalues_np: Kinship eigenvalues as numpy array.
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        n_cvt: Number of covariates.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.

    Returns:
        Tuple of (pve, pve_se) where pve is the PVE estimate (float between
        0 and 1) and pve_se is the standard error of PVE via delta method
        (None if the likelihood surface is flat).
    """
    lambda_remle, _logl = compute_null_model_lambda(
        eigenvalues_np, UtW, Uty, n_cvt, l_min=l_min, l_max=l_max
    )
    trace_K = float(np.sum(eigenvalues_np))
    n = len(eigenvalues_np)
    pve = lambda_remle * trace_K / (lambda_remle * trace_K + n)
    logger.info(f"pve estimate in the null model = {pve:.6f}")

    # Compute se(pve) via delta method using REML second derivative.
    # The analytical reml_log_likelihood_dev2 omits d²(logdet_hiw)/dλ²,
    # which makes it incomplete for all n_cvt. Use finite differences of
    # reml_log_likelihood_null until the analytical port is completed.
    Uab = compute_Uab(UtW, Uty, Utx=None)
    dev2 = finite_difference_dev2(
        lambda_remle,
        eigenvalues_np,
        Uab,
        n_cvt,
        l_min=l_min,
        l_max=l_max,
    )

    pve_se: float | None = None
    if dev2 < 0:
        se_lambda = np.sqrt(-1.0 / dev2)
        trace_G = trace_K / n
        # d(PVE)/d(lambda) = trace_G / (trace_G * lambda + 1)^2
        denom = trace_G * lambda_remle + 1.0
        pve_se = float(trace_G / (denom * denom) * se_lambda)
        logger.info(f"se(pve) in the null model = {pve_se:.6g}")
    elif np.isnan(dev2):
        logger.error(
            f"REML second derivative is NaN at lambda={lambda_remle:.6e} — "
            f"degenerate projection (P_yy likely zero). se(pve) unavailable"
        )
    elif dev2 > 0:
        logger.error(
            f"REML second derivative is positive ({dev2:.6e}) at lambda="
            f"{lambda_remle:.6e} — optimum may not be a maximum. se(pve) unavailable"
        )
    else:
        logger.warning(
            f"REML second derivative is zero at lambda={lambda_remle:.6e} — "
            f"flat likelihood surface, se(pve) unavailable"
        )

    return pve, pve_se
