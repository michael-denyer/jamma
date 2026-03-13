"""Shared LMM chunk computation.

Encapsulates the mode-specific dispatch logic (modes 1-4) that is
shared across runner_jax.py, runner_streaming.py, and loco.py.

The caller is responsible for:
- Computing Uab_batch via batch_compute_uab (before calling this)
- Calling block_until_ready() on results (after calling this),
  or using block_chunk_result() for convenience
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from loguru import logger

from jamma.lmm.likelihood_jax import (
    batch_calc_score_stats,
    batch_calc_wald_stats,
    batch_compute_iab,
    calc_lrt_pvalue_jax,
    golden_section_optimize_lambda_mle,
)
from jamma.lmm.prepare import _grid_optimize_lambda_batched

# Which keys to sync on per mode (last-computed arrays for timing accuracy)
_SYNC_KEYS = {
    1: ("pwalds",),
    2: ("p_lrts",),
    3: ("p_scores",),
    4: ("p_scores", "p_lrts", "pwalds"),
}


def _compute_wald(
    n_cvt: int,
    eigenvalues: jax.Array,
    Uab_batch: jax.Array,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
) -> dict[str, jax.Array]:
    """Compute REML-optimized Wald test statistics."""
    Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
    lambdas, logls = _grid_optimize_lambda_batched(
        n_cvt,
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min,
        l_max,
        n_grid,
        n_refine,
    )
    betas, ses, pwalds = batch_calc_wald_stats(
        n_cvt, lambdas, eigenvalues, Uab_batch, n_samples
    )
    return {
        "lambdas": lambdas,
        "logls": logls,
        "betas": betas,
        "ses": ses,
        "pwalds": pwalds,
    }


def _compute_lrt(
    n_cvt: int,
    eigenvalues: jax.Array,
    Uab_batch: jax.Array,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
) -> dict[str, jax.Array]:
    """Compute MLE-optimized LRT statistics."""
    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=max(n_refine, 20),
    )
    p_lrts = jax.vmap(calc_lrt_pvalue_jax)(logls_mle, jnp.full_like(logls_mle, logl_H0))
    return {"lambdas_mle": lambdas_mle, "p_lrts": p_lrts}


def _compute_score(
    n_cvt: int,
    Hi_eval_null: jax.Array,
    Uab_batch: jax.Array,
    n_samples: int,
) -> dict[str, jax.Array]:
    """Compute Score test statistics (no optimization needed)."""
    if jnp.any(Hi_eval_null <= 0):
        raise ValueError(
            "Hi_eval_null contains non-positive values. "
            "Check kinship matrix conditioning."
        )
    betas, ses, p_scores = batch_calc_score_stats(
        n_cvt, Hi_eval_null, Uab_batch, n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _compute_lmm_chunk(
    lmm_mode: int,
    n_cvt: int,
    eigenvalues: jax.Array,
    Uab_batch: jax.Array,
    n_samples: int,
    *,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    Hi_eval_null: jax.Array | None = None,
    logl_H0: float | None = None,
) -> dict[str, jax.Array | None]:
    """Compute LMM statistics for a chunk of SNPs.

    Dispatches to mode-specific computation (Wald, LRT, Score, or All)
    without calling batch_compute_uab or block_until_ready -- those are
    the caller's responsibility.

    Args:
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues on device.
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations.
        Hi_eval_null: Pre-computed 1/(lambda_null*eval+1) for Score test.
        logl_H0: Null model MLE log-likelihood for LRT.

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds,
        lambdas_mle, p_lrts, p_scores. Keys not relevant to the
        mode are set to None.
    """
    if lmm_mode in (2, 4) and logl_H0 is None:
        raise ValueError("logl_H0 is required for LRT (mode 2) and All (mode 4)")
    if lmm_mode in (3, 4) and Hi_eval_null is None:
        raise ValueError("Hi_eval_null is required for Score (mode 3) and All (mode 4)")
    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    result: dict[str, jax.Array | None] = {
        "lambdas": None,
        "logls": None,
        "betas": None,
        "ses": None,
        "pwalds": None,
        "lambdas_mle": None,
        "p_lrts": None,
        "p_scores": None,
    }

    if lmm_mode == 1:
        result.update(
            _compute_wald(
                n_cvt,
                eigenvalues,
                Uab_batch,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
            )
        )

    elif lmm_mode == 2:
        result.update(
            _compute_lrt(
                n_cvt,
                eigenvalues,
                Uab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
                logl_H0,
            )
        )

    elif lmm_mode == 3:
        result.update(
            _compute_score(
                n_cvt,
                Hi_eval_null,
                Uab_batch,
                n_samples,
            )
        )

    elif lmm_mode == 4:
        # Compose all three tests; Score is cheapest so runs first
        score_result = _compute_score(
            n_cvt,
            Hi_eval_null,
            Uab_batch,
            n_samples,
        )
        # Score betas/ses omitted — Wald provides REML-optimized values below
        result["p_scores"] = score_result["p_scores"]
        result.update(
            _compute_lrt(
                n_cvt,
                eigenvalues,
                Uab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
                logl_H0,
            )
        )
        # Wald overwrites betas/ses from Score (REML-optimized values)
        result.update(
            _compute_wald(
                n_cvt,
                eigenvalues,
                Uab_batch,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
            )
        )

    return result


def block_chunk_result(result: dict[str, jax.Array | None], lmm_mode: int) -> None:
    """Block until async JAX computation completes for timing accuracy.

    Must be called after _compute_lmm_chunk when the caller needs
    accurate wall-clock timing (e.g., streaming and LOCO runners).

    Args:
        result: Dict returned by _compute_lmm_chunk.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
    """
    for key in _SYNC_KEYS.get(lmm_mode, ()):
        if result[key] is not None:
            result[key].block_until_ready()


def log_jax_error(
    e: Exception,
    *,
    chunk_label: str,
    chunk_snps: int,
    n_samples: int,
    n_cvt: int = 1,
) -> None:
    """Log a JAX computation error with context, detecting buffer overflow.

    String-based detection is fragile — update if JAX changes the
    wording in future versions.

    Args:
        e: The caught exception.
        chunk_label: Human-readable chunk identifier (e.g. "3/10").
        chunk_snps: Number of SNPs in the failing chunk.
        n_samples: Number of samples.
        n_cvt: Number of covariates (for buffer size calculation).
    """
    error_msg = str(e)
    if "exceeds the maximum representable value" in error_msg:
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2
        buffer_elements = n_samples * chunk_snps * n_index
        logger.error(
            f"JAX int32 buffer overflow during LMM computation.\n"
            f"  Chunk {chunk_label}: {chunk_snps:,} SNPs x "
            f"{n_samples:,} samples\n"
            f"  Buffer elements: {buffer_elements:,} (limit: ~2.1B)\n"
            f"  This should not happen with automatic chunking.\n"
            f"  Please report this issue with your dataset dimensions."
        )
    else:
        logger.error(
            f"JAX computation failed on chunk {chunk_label}:\n"
            f"  {type(e).__name__}: {error_msg}\n"
            f"  Chunk size: {chunk_snps:,} SNPs, Samples: {n_samples:,}"
        )


def exposed_rotation_time(
    rot_dur: float, t_rot_end: float, prev_compute_end: float | None
) -> float:
    """Compute exposed (non-overlapped) rotation time for a single chunk.

    When rotation and JAX compute overlap, only the portion of rotation
    that extends beyond the prior compute sync point is "exposed" — i.e.,
    time the pipeline spent waiting for rotation with no useful compute.

    Args:
        rot_dur: Total rotation wall-clock duration for this chunk.
        t_rot_end: perf_counter timestamp when rotation finished.
        prev_compute_end: perf_counter timestamp when prior JAX compute
            finished, or None if no compute has completed yet.

    Returns:
        Exposed rotation time in seconds.
    """
    if prev_compute_end is None:
        return rot_dur
    return min(rot_dur, max(0.0, t_rot_end - prev_compute_end))
