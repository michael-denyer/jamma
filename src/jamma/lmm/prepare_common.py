"""Pure-NumPy setup utilities shared by NumPy LMM runners.

Provides covariate matrix construction, eigendecomposition handling,
null model computation, and shared input validation. NumPy LMM runners
import from this module.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
from loguru import logger

from jamma.core.constants import PHENOTYPE_MISSING
from jamma.core.memory_snapshot import log_memory_snapshot
from jamma.core.threading import blas_threads, get_blas_thread_count
from jamma.lmm.eigen import center_kinship, eigendecompose_kinship
from jamma.lmm.likelihood import (
    compute_null_model_lambda,
    compute_null_model_mle,
    finite_difference_dev2,
)
from jamma.lmm.pab import compute_Uab
from jamma.lmm.schema import DEFAULT_L_MAX, DEFAULT_L_MIN, LmmConfig


@dataclass(frozen=True, slots=True)
class NullModel:
    """The null-model MLE, computed unconditionally for every LMM run.

    The MLE optimization costs 0.8 ms at n=2k and 28.8 ms at n=100k, so
    gating it by lmm_mode saves nothing. Every runner computes both fields
    regardless of which test the run reports.

    Attributes:
        logl_H0: Null-model MLE log-likelihood.
        hi_eval_null: 1/(lambda_null_mle * eigenvalues + 1), per sample.
    """

    logl_H0: float
    hi_eval_null: np.ndarray


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

    Raises:
        ValueError: If any phenotype is infinite. Unlike NaN and -9, inf is
            not a missing-value code, so it is rejected rather than masked.
    """
    if np.isinf(phenotypes).any():
        raise ValueError("prepared phenotypes must contain only finite values")
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != PHENOTYPE_MISSING)
    if covariates is not None:
        valid_covariate = np.all(~np.isnan(covariates), axis=1)
        valid_mask = valid_mask & valid_covariate
    return valid_mask


@dataclass(frozen=True, slots=True)
class KinshipMatrix:
    """A kinship matrix that still needs eigendecomposition.

    The array is consumed: centred in place, then overwritten by the
    eigendecomposition. Callers that need the matrix afterwards pass a copy.
    """

    value: np.ndarray

    def __post_init__(self) -> None:
        if not self.value.flags.writeable:
            raise ValueError(
                "kinship must be writeable: it is consumed in place (centred, "
                "then overwritten by the eigendecomposition). Pass kinship.copy() "
                "to keep the original matrix."
            )


@dataclass(frozen=True, slots=True)
class EigenPairs:
    """A complete pre-computed eigendecomposition."""

    values: np.ndarray
    vectors: np.ndarray


EigenInput = KinshipMatrix | EigenPairs


def parse_eigen_input(
    kinship: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
) -> EigenInput:
    """Normalize the legacy public eigen arguments into one complete value."""
    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )
    if eigenvalues is not None and eigenvectors is not None:
        return EigenPairs(eigenvalues, eigenvectors)
    if kinship is not None:
        return KinshipMatrix(kinship)
    raise ValueError(
        "Either kinship or pre-computed eigendecomposition (eigenvalues + "
        "eigenvectors) must be provided"
    )


@dataclass(frozen=True, slots=True)
class AnalysedPhenotype:
    """One phenotype and its covariates, restricted to the analysed samples.

    Attributes:
        phenotypes: Phenotype values of the analysed samples only.
        covariates: Covariates of the analysed samples, carrying an intercept
            column, or None for the intercept-only model.
        valid_mask: Boolean analysed-sample mask over the caller's input rows.
    """

    phenotypes: np.ndarray
    covariates: np.ndarray | None
    valid_mask: np.ndarray

    @classmethod
    def from_inputs(
        cls, phenotypes: np.ndarray, covariates: np.ndarray | None
    ) -> AnalysedPhenotype:
        """Mask missing samples and add the intercept GEMMA's CheckCvt would.

        Raises:
            ValueError: If a phenotype is infinite or no valid sample remains.
        """
        valid_mask = compute_valid_mask(phenotypes, covariates)
        return cls.from_mask(
            phenotypes, with_intercept(covariates, valid_mask), valid_mask
        )

    @classmethod
    def from_mask(
        cls,
        phenotypes: np.ndarray,
        covariates: np.ndarray | None,
        valid_mask: np.ndarray,
    ) -> AnalysedPhenotype:
        """Apply an already-computed analysed-sample mask.

        Raises:
            ValueError: If the mask selects no sample.
        """
        if not valid_mask.any():
            raise ValueError(
                "No valid samples: all phenotypes are missing or -9"
                + (", or all have missing covariates" if covariates is not None else "")
            )
        if valid_mask.all():
            return cls(phenotypes, covariates, valid_mask)
        return cls(
            phenotypes[valid_mask],
            covariates[valid_mask, :] if covariates is not None else None,
            valid_mask,
        )

    @property
    def n_samples(self) -> int:
        """Number of analysed samples."""
        return self.phenotypes.shape[0]

    @property
    def n_cvt(self) -> int:
        """Covariate columns including the intercept."""
        return self.covariates.shape[1] if self.covariates is not None else 1


def restrict_eigen_input(eigen_input: EigenInput, valid_mask: np.ndarray) -> EigenInput:
    """Restrict a kinship to the analysed samples, or check eigenpairs fit them.

    Args:
        eigen_input: Kinship over every input sample, or eigenpairs that the
            caller computed over the analysed samples.
        valid_mask: Boolean analysed-sample mask over the input rows.

    Raises:
        ValueError: If eigenpair dimensions do not match the analysed sample
            count.
    """
    if isinstance(eigen_input, KinshipMatrix):
        if valid_mask.all():
            return eigen_input
        return KinshipMatrix(eigen_input.value[np.ix_(valid_mask, valid_mask)])

    n_samples = int(np.count_nonzero(valid_mask))
    hint = (
        "Recompute eigenpairs on the filtered kinship, or pass kinship= "
        "and let JAMMA compute the eigendecomposition."
    )
    if eigen_input.values.shape[0] != n_samples:
        raise ValueError(
            f"eigenvalues length ({eigen_input.values.shape[0]}) does not match "
            f"n_samples ({n_samples}) after removing missing "
            f"phenotypes/covariates. {hint}"
        )
    if eigen_input.vectors.shape != (n_samples, n_samples):
        raise ValueError(
            f"eigenvectors shape {eigen_input.vectors.shape} does not match "
            f"({n_samples}, {n_samples}) after removing missing "
            f"phenotypes/covariates. {hint}"
        )
    return eigen_input


def _covariates_include_intercept(covariates: np.ndarray) -> bool:
    """True if any covariate column is constant (GEMMA's intercept test)."""
    return bool(np.any(np.ptp(covariates, axis=0) == 0.0))


def with_intercept(
    covariates: np.ndarray | None, valid_mask: np.ndarray
) -> np.ndarray | None:
    """Append a column of 1s unless some column is constant over the analysed rows.

    GEMMA's CheckCvt tests constancy over indicator_idv, so rows the mask
    drops never vote and a NaN row cannot hide an existing intercept. A
    column of 1s is constant under any mask and adds no NaN, so the valid
    mask is unchanged by it and a second call returns its input.

    Args:
        covariates: Unmasked covariate matrix (n_input, n_covariates) or None.
        valid_mask: Boolean analysed-sample mask of length n_input.

    Returns:
        The input when it is None or already carries a constant column,
        otherwise the input with a ones column appended.
    """
    if covariates is None or _covariates_include_intercept(covariates[valid_mask]):
        return covariates
    logger.info("No intercept term found in the covariate file; adding a column of 1s.")
    return np.hstack([covariates, np.ones((covariates.shape[0], 1))])


def _build_covariate_matrix(
    covariates: np.ndarray | None, n_samples: int
) -> tuple[np.ndarray, int]:
    """Construct covariate matrix W and return (W, n_cvt).

    If covariates is None, uses intercept-only model. Supplied covariates
    must already carry an intercept column: with_intercept() appends one
    after masking, and the association plan is sized from that same array,
    so a matrix without one here means a caller skipped that step.

    Args:
        covariates: Optional masked covariate matrix (n_samples, n_covariates).
        n_samples: Number of samples (for intercept construction).

    Returns:
        Tuple of (W, n_cvt) where W is the covariate matrix.

    Raises:
        ValueError: If no covariate column is constant, the model is
            over-parameterized, or the covariates are rank-deficient.
    """
    if covariates is None:
        W = np.ones((n_samples, 1))
    else:
        W = covariates.astype(np.float64)
        if not _covariates_include_intercept(W):
            raise ValueError(
                "Covariate matrix has no intercept column; pass covariates "
                "through with_intercept() after computing the valid mask"
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


@dataclass(frozen=True, slots=True)
class RotatedBasis:
    """The eigenbasis and rotated covariates every phenotype in a group shares.

    Attributes:
        eigenvalues: Kinship eigenvalues, ascending.
        U: Kinship eigenvectors.
        W: Covariate matrix from ``_build_covariate_matrix``.
        UtW: Rotated covariates ``U.T @ W``.
    """

    eigenvalues: np.ndarray
    U: np.ndarray
    W: np.ndarray
    UtW: np.ndarray

    @property
    def n_samples(self) -> int:
        """Number of analysed samples the basis spans."""
        return self.U.shape[0]

    @property
    def n_cvt(self) -> int:
        """Number of covariate columns, including the intercept."""
        return self.W.shape[1]


def rotate_basis(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, W: np.ndarray
) -> RotatedBasis:
    """Rotate the phenotype-independent covariates into the eigenbasis once."""
    with blas_threads(get_blas_thread_count()):
        UtW = eigenvectors.T @ W
    return RotatedBasis(eigenvalues=eigenvalues, U=eigenvectors, W=W, UtW=UtW)


def _eigendecompose_or_reuse(
    eigen_input: EigenInput,
    show_progress: bool,
    label: str,
    *,
    check_memory: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigendecomposition, computing it if not provided.

    Args:
        eigen_input: Kinship matrix or complete pre-computed eigenpairs.
        show_progress: Whether to log memory usage.
        label: Label for memory logging (e.g. "lmm", "lmm_streaming").
        check_memory: If True (default), check available memory before
            eigendecomposition.

    Returns:
        Tuple of (eigenvalues, eigenvectors).
    """
    if isinstance(eigen_input, EigenPairs):
        if show_progress:
            logger.debug("Using pre-computed eigendecomposition")
        return eigen_input.values, eigen_input.vectors

    if show_progress:
        log_memory_snapshot(f"{label}:before_eigendecomp")
    # Centre the analysed kinship before eigendecomposition, as GEMMA's
    # CenterMatrix does and the pipeline does at pipeline.py. REML with an
    # intercept is invariant to this, but MLE, LRT, Score and PVE are not, so a
    # raw supplied kinship would otherwise give the wrong non-REML results.
    center_kinship(eigen_input.value)
    eigenvalues_np, U = eigendecompose_kinship(
        eigen_input.value, check_memory=check_memory
    )
    # Release LAPACK DSYEVD workspace before LMM phase
    gc.collect()
    if show_progress:
        log_memory_snapshot(f"{label}:after_eigendecomp")
    return eigenvalues_np, U


def _compute_null_model_common(
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    show_progress: bool,
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
) -> NullModel:
    """Compute the null model MLE, unconditionally, for every LMM run.

    Pure-NumPy version of the null model computation. The optimization costs
    0.8 ms at n=2k and 28.8 ms at n=100k, so a mode gate here saves nothing;
    every runner now gets both fields regardless of which test it reports.

    GEMMA computes both REML and MLE null lambdas in CalcLambda, but uses
    MLE lambda for Hi_eval in the Score test:
    Hi_eval_null = 1 / (lambda_null_mle * eigenvalues + 1).

    Args:
        eigenvalues_np: Kinship eigenvalues as numpy array.
        UtW: Rotated covariates.
        Uty: Rotated phenotype.
        n_cvt: Number of covariates.
        show_progress: Whether to log results.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.

    Returns:
        NullModel with logl_H0 and hi_eval_null populated.
    """
    lambda_null_mle, logl_H0 = compute_null_model_mle(
        eigenvalues_np, UtW, Uty, n_cvt, l_min=l_min, l_max=l_max
    )
    if show_progress:
        logger.info(
            f"Null model MLE: lambda={lambda_null_mle:.6f}, logl_H0={logl_H0:.6f}"
        )

    hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues_np + 1.0)
    if not np.all(np.isfinite(hi_eval_null)):
        bad_idx = np.where(~np.isfinite(hi_eval_null))[0]
        raise ValueError(
            f"Hi_eval_null has {len(bad_idx)} non-finite value(s) at indices "
            f"{bad_idx[:5].tolist()}. lambda_null_mle={lambda_null_mle:.6g}. "
            "Null model optimization may have failed."
        )
    if not np.all(hi_eval_null > 0):
        bad_idx = np.where(~(hi_eval_null > 0))[0]
        raise ValueError(
            f"Hi_eval_null has {len(bad_idx)} non-positive value(s) at indices "
            f"{bad_idx[:5].tolist()}. lambda_null_mle={lambda_null_mle:.6g}. "
            "Check kinship matrix for negative eigenvalues."
        )

    return NullModel(logl_H0=logl_H0, hi_eval_null=hi_eval_null)


@dataclass(frozen=True, slots=True)
class NullFit:
    """One phenotype's rotated values and null model over a shared basis.

    Attributes:
        Uty: Rotated phenotype (n_samples,).
        logl_H0: Null-model MLE log-likelihood, computed on every run.
        Hi_eval_null: Null-model Hi_eval, computed on every run.
        pve: Proportion of variance explained, from the null REML lambda.
            None when the caller skipped it (compute_pve=False).
        pve_se: Standard error of PVE, or None on a flat likelihood surface
            or when PVE was skipped.
    """

    Uty: np.ndarray
    logl_H0: float
    Hi_eval_null: np.ndarray
    pve: float | None
    pve_se: float | None


def fit_null(
    basis: RotatedBasis,
    phenotypes: np.ndarray,
    config: LmmConfig,
    *,
    compute_pve: bool,
) -> NullFit:
    """Rotate one phenotype, solve its null model, and estimate PVE.

    Args:
        basis: The group's shared eigenbasis and rotated covariates.
        phenotypes: Phenotype values of the analysed samples.
        config: Lambda bounds and the progress switch.
        compute_pve: Whether to run the extra null-REML golden section for
            PVE. LOCO passes False on every chromosome after the first.

    Raises:
        ValueError: If the phenotype length does not match the basis.
    """
    if phenotypes.shape != (basis.n_samples,):
        raise ValueError(
            "prepared phenotype length does not match genotype sample basis: "
            f"got {phenotypes.shape}, expected ({basis.n_samples},)"
        )
    l_min, l_max = config.l_min, config.l_max
    with blas_threads(get_blas_thread_count()):
        Uty = basis.U.T @ phenotypes

    null_model = _compute_null_model_common(
        basis.eigenvalues,
        basis.UtW,
        Uty,
        basis.n_cvt,
        config.show_progress,
        l_min=l_min,
        l_max=l_max,
    )
    pve: float | None = None
    pve_se: float | None = None
    if compute_pve:
        pve, pve_se = compute_and_log_pve(
            basis.eigenvalues, basis.UtW, Uty, basis.n_cvt, l_min, l_max
        )

    return NullFit(
        Uty=Uty,
        logl_H0=null_model.logl_H0,
        Hi_eval_null=null_model.hi_eval_null,
        pve=pve,
        pve_se=pve_se,
    )


def compute_and_log_pve(
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
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

    # Compute se(pve) via delta method using the REML second derivative.
    # finite_difference_dev2 is the production path, and the analytical
    # reml_log_likelihood_dev2 is not a stub. test_likelihood_derivatives.py
    # shows it computing analytically, agreeing with this oracle to rtol=1e-4
    # for n_cvt in 2..4, and reproducing GEMMA's se(pve) on mouse_hs1940 to
    # ~8.5e-5. Switching would move pve_se at the 1e-4 level, so it is a
    # numerics decision rather than a correctness fix.
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
