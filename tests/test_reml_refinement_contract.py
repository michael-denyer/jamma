"""Independent contracts for safeguarded REML score refinement."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from scipy.optimize import brentq

from jamma.lmm import accel
from jamma.lmm.pab import build_pab_table_for_c
from jamma.lmm.uab import compute_uab_invariant_soa
from tests.conftest import requires_c
from tests.independent_lmm_oracle import dense_reml_score_log_lambda

_L_MIN = 1e-5
_L_MAX = 1e5


@lru_cache(maxsize=1)
def _general_case_with_stationary_point():
    """Build a deterministic two-covariate case with a strict REML maximum."""
    rng = np.random.default_rng(20260905)
    n_samples = 40
    eigenvalues = np.exp(np.linspace(np.log(0.08), np.log(6.0), n_samples))
    UtW = np.column_stack((np.ones(n_samples), rng.standard_normal(n_samples)))
    Uty = rng.standard_normal(n_samples)
    log_grid = np.linspace(np.log(1e-3), np.log(1e3), 241)

    for _ in range(32):
        Utg = rng.standard_normal(n_samples)
        scores = np.array(
            [
                dense_reml_score_log_lambda(eigenvalues, UtW, Uty, Utg, np.exp(point))
                for point in log_grid
            ]
        )
        crossings = np.flatnonzero((scores[:-1] > 0.0) & (scores[1:] < 0.0))
        if crossings.size:
            index = int(crossings[0])
            root = brentq(
                lambda point, genotype=Utg: dense_reml_score_log_lambda(
                    eigenvalues, UtW, Uty, genotype, np.exp(point)
                ),
                log_grid[index],
                log_grid[index + 1],
                xtol=1e-13,
            )
            return eigenvalues, UtW, Uty, Utg, float(np.exp(root))

    raise AssertionError("deterministic fixture search found no strict REML maximum")


def _run_general_c_at_target(
    target_lambda: float,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Scale the spectrum so the independent stationary point is at target."""
    eigenvalues, UtW, Uty, Utg, original_root = _general_case_with_stationary_point()
    scaled_eigenvalues = np.ascontiguousarray(
        eigenvalues * (original_root / target_lambda)
    )
    invariant = compute_uab_invariant_soa(UtW, Uty, n_cvt=2)
    workspace = accel.require().create_workspace_general_c(
        scaled_eigenvalues,
        invariant,
        np.ascontiguousarray(UtW),
        np.ascontiguousarray(Uty),
        len(Uty),
        _L_MIN,
        _L_MAX,
        50,
        20,
        1,
        build_pab_table_for_c(2)._asdict(),
        lmm_mode=1,
    )
    result = accel.require().compute_lmm_chunk_fused_general_c(
        workspace, np.ascontiguousarray(Utg[None, :]), 1
    )
    oracle_args = (scaled_eigenvalues, UtW, Uty, Utg)
    return float(result["lambdas"][0]), oracle_args


@pytest.mark.tier0
@requires_c
def test_general_c_refinement_reaches_independent_stationary_root():
    actual, oracle_args = _run_general_c_at_target(0.7)
    expected_log = brentq(
        lambda point: dense_reml_score_log_lambda(*oracle_args, np.exp(point)),
        np.log(0.4),
        np.log(1.0),
        xtol=1e-13,
    )
    np.testing.assert_allclose(actual, np.exp(expected_log), rtol=5e-6, atol=0.0)


@pytest.mark.tier0
@requires_c
def test_general_c_refines_peak_close_to_lower_bound():
    target = 1.2 * _L_MIN
    actual, oracle_args = _run_general_c_at_target(target)
    assert dense_reml_score_log_lambda(*oracle_args, _L_MIN) > 0.0
    assert dense_reml_score_log_lambda(*oracle_args, 1.5 * _L_MIN) < 0.0
    expected_log = brentq(
        lambda point: dense_reml_score_log_lambda(*oracle_args, np.exp(point)),
        np.log(_L_MIN),
        np.log(1.5 * _L_MIN),
        xtol=1e-13,
    )
    np.testing.assert_allclose(actual, np.exp(expected_log), rtol=5e-6, atol=0.0)
    assert actual > _L_MIN


@pytest.mark.tier0
@requires_c
def test_general_c_preserves_monotone_lower_boundary():
    actual, oracle_args = _run_general_c_at_target(0.1 * _L_MIN)
    assert dense_reml_score_log_lambda(*oracle_args, _L_MIN) < 0.0
    assert dense_reml_score_log_lambda(*oracle_args, _L_MAX) < 0.0
    # Twenty golden iterations leave at most about 1.6e-5 relative midpoint
    # error in the one-grid-step boundary bracket.
    np.testing.assert_allclose(actual, _L_MIN, rtol=2e-5, atol=0.0)
