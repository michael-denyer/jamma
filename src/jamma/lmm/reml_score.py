"""Stable analytic REML scores and safeguarded refinement of interior optima.

Differentiate the weighted cross-products directly. Subtracting Pab and PPab
would lose precision when lambda is small. Compensated reductions retain the
score information that rounded log-likelihood comparisons can discard.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from jamma.lmm.pab import _NCVT1, build_index_table
from jamma.lmm.uab import _fill_pab_recursion

_SCORE_PROBE_DELTA = 1e-3


def _differentiate_pab_recursion(
    pab: np.ndarray, derivative: np.ndarray, n_cvt: int
) -> None:
    """Apply the exact Schur-complement chain rule to a Pab derivative."""
    table = build_index_table(n_cvt)
    for p in range(1, n_cvt + 2):
        for _a, _b, ab, aw, bw, ww in table.pab_recursion[p]:
            q = pab[..., p - 1, ww]
            safe = q != 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = np.where(safe, 1.0 / q, 0.0)
            derivative[..., p, ab] = (
                derivative[..., p - 1, ab]
                - (
                    derivative[..., p - 1, aw] * pab[..., p - 1, bw]
                    + pab[..., p - 1, aw] * derivative[..., p - 1, bw]
                )
                * inv
                + pab[..., p - 1, aw]
                * pab[..., p - 1, bw]
                * derivative[..., p - 1, ww]
                * inv
                * inv
            )


def _compensated_weighted_sum(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Kahan sum over samples while retaining vectorized SNP/column axes."""
    total = np.zeros(values.shape[:1] + values.shape[2:], dtype=np.float64)
    compensation = np.zeros_like(total)
    for sample in range(values.shape[1]):
        term = weights[:, sample, None] * values[:, sample, :] - compensation
        updated = total + term
        compensation = (updated - total) - term
        total = updated
    return total


def _batch_reml_score_log_lambda_numpy(
    n_cvt: int,
    log_lambdas: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate the analytic REML score with respect to log(lambda)."""
    lambdas = np.exp(log_lambdas)
    h = 1.0 / (1.0 + lambdas[:, None] * eigenvalues[None, :])
    dh = -lambdas[:, None] * eigenvalues[None, :] * h * h
    row0 = _compensated_weighted_sum(h, Uab_batch)
    drow0 = _compensated_weighted_sum(dh, Uab_batch)
    table = build_index_table(n_cvt)
    shape = (len(lambdas), n_cvt + 2, table.n_index)
    pab = np.zeros(shape, dtype=np.float64)
    dpab = np.zeros(shape, dtype=np.float64)
    pab[:, 0, :] = row0
    dpab[:, 0, :] = drow0
    _fill_pab_recursion(pab, table, n_cvt)
    _differentiate_pab_recursion(pab, dpab, n_cvt)

    trace_values = (lambdas[:, None] * eigenvalues[None, :] * h)[:, :, None]
    score = -0.5 * _compensated_weighted_sum(np.ones_like(h), trace_values)[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for row, col in table.logdet_diag_indices:
            score -= 0.5 * dpab[:, row, col] / pab[:, row, col]
        nc_total = n_cvt + 1
        pyy = pab[:, nc_total, table.idx_yy]
        score -= (
            0.5
            * (len(eigenvalues) - n_cvt - 1)
            * (dpab[:, nc_total, table.idx_yy] / pyy)
        )
    return score


def _refine_reml_optima(
    log_opt: np.ndarray,
    coarse_a: np.ndarray,
    coarse_b: np.ndarray,
    interior: np.ndarray,
    score_at: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Refine interior peaks until the estimated log-lambda error is small."""
    if not np.any(interior):
        return log_opt
    indices = np.flatnonzero(interior)
    result = log_opt.copy()
    for _ in range(3):
        x = result[indices]
        low = coarse_a[indices]
        high = coarse_b[indices]
        delta = np.minimum.reduce(
            (
                np.full_like(x, _SCORE_PROBE_DELTA),
                0.25 * (high - low),
                0.5 * (x - low),
                0.5 * (high - x),
            )
        )
        score = score_at(x, indices)
        score_minus = score_at(x - delta, indices)
        score_plus = score_at(x + delta, indices)
        curvature = (score_plus - score_minus) / (2.0 * delta)
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate = x - score / curvature
        eligible = (
            np.isfinite(candidate)
            & np.isfinite(curvature)
            & np.isfinite(delta)
            & (delta > 0.0)
            & (curvature < 0.0)
            & (candidate >= low)
            & (candidate <= high)
        )
        evaluated_candidate = np.where(eligible, candidate, x)
        corrected_score = score_at(evaluated_candidate, indices)
        safe = (
            eligible
            & np.isfinite(corrected_score)
            & (np.abs(corrected_score) < np.abs(score))
        )
        result[indices] = np.where(safe, candidate, x)
        # A small score alone is misleading when curvature is nearly zero.
        # Revisit only peaks whose estimated remaining relative lambda error
        # exceeds 1e-10; cap work at three safeguarded steps.
        with np.errstate(divide="ignore", invalid="ignore"):
            remaining = np.abs(corrected_score / curvature)
        indices = indices[safe & (remaining > 1e-10)]
        if indices.size == 0:
            break

    return result


def _batch_reml_score_log_lambda_split_ncvt1_numpy(
    log_lambdas: np.ndarray,
    eigenvalues: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
) -> np.ndarray:
    """Analytic log-lambda REML score without materialising combined Uab."""
    lambdas = np.exp(log_lambdas)
    h = 1.0 / (1.0 + lambdas[:, None] * eigenvalues[None, :])
    dh = -lambdas[:, None] * eigenvalues[None, :] * h * h
    row0 = np.empty((len(lambdas), 6), dtype=np.float64)
    drow0 = np.empty_like(row0)
    invariant = np.broadcast_to(
        uab_invariant_soa.T[None, :, :], (len(lambdas), len(eigenvalues), 3)
    )
    varying = np.transpose(uab_varying_soa, (0, 2, 1))
    row0[:, [_NCVT1.ww, _NCVT1.wy, _NCVT1.yy]] = _compensated_weighted_sum(h, invariant)
    drow0[:, [_NCVT1.ww, _NCVT1.wy, _NCVT1.yy]] = _compensated_weighted_sum(
        dh, invariant
    )
    row0[:, [_NCVT1.wx, _NCVT1.xx, _NCVT1.xy]] = _compensated_weighted_sum(h, varying)
    drow0[:, [_NCVT1.wx, _NCVT1.xx, _NCVT1.xy]] = _compensated_weighted_sum(dh, varying)
    pab = np.zeros((len(lambdas), 3, 6), dtype=np.float64)
    dpab = np.zeros_like(pab)
    pab[:, 0, :] = row0
    dpab[:, 0, :] = drow0
    table = build_index_table(1)
    _fill_pab_recursion(pab, table, 1)
    _differentiate_pab_recursion(pab, dpab, 1)
    trace_values = (lambdas[:, None] * eigenvalues[None, :] * h)[:, :, None]
    score = -0.5 * _compensated_weighted_sum(np.ones_like(h), trace_values)[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        score -= 0.5 * dpab[:, 0, _NCVT1.ww] / pab[:, 0, _NCVT1.ww]
        score -= 0.5 * dpab[:, 1, _NCVT1.xx] / pab[:, 1, _NCVT1.xx]
        score -= (
            0.5
            * (len(eigenvalues) - 2)
            * (dpab[:, 2, _NCVT1.yy] / pab[:, 2, _NCVT1.yy])
        )
    return score
