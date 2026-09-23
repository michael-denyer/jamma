"""Independent contracts for safeguarded MLE score refinement.

Every expected lambda is a root of the dense-solve MLE score, found without
production Pab, score, or optimizer code. GEMMA's output is not the target.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from scipy.optimize import brentq

from jamma.lmm import accel
from jamma.lmm.likelihood_numpy import golden_section_optimize_lambda_mle_numpy
from jamma.lmm.uab import batch_compute_uab_numpy, compute_uab_invariant_soa
from tests.math_validation.dense_oracle import mle_score_log_lambda
from tests.math_validation.weight_contract import _weighted_model
from tests.support import requires_c

_L_MIN = 1e-5
_L_MAX = 1e5
_BACKENDS = ["numpy", pytest.param("native", marks=requires_c)]


def _mle_lambda(backend, n_cvt, eigenvalues, UtW, Uty, Utg, n_refine):
    """Run one production MLE optimizer on a single SNP."""
    if backend == "numpy":
        uab = batch_compute_uab_numpy(n_cvt, UtW, Uty, Utg[None, :])
        lambdas, _ = golden_section_optimize_lambda_mle_numpy(
            n_cvt, eigenvalues, uab, n_iter=n_refine
        )
        return float(lambdas[0])
    workspace = accel.require().create_workspace_c(
        np.ascontiguousarray(eigenvalues),
        compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt),
        np.ascontiguousarray(UtW),
        np.ascontiguousarray(Uty),
        len(Uty),
        _L_MIN,
        _L_MAX,
        50,
        n_refine,
        1,
        n_cvt,
        lmm_mode=2,
        logl_H0=0.0,
    )
    result = accel.require().compute_lmm_chunk_c(
        workspace, np.ascontiguousarray(Utg[None, :]), 1
    )
    return float(result["lambdas_mle"][0])


def _mle_root(eigenvalues, UtW, Uty, Utg, low, high):
    return float(
        np.exp(
            brentq(
                lambda point: mle_score_log_lambda(
                    np.diag(eigenvalues), UtW, Utg, Uty, np.exp(point)
                ),
                np.log(low),
                np.log(high),
                xtol=1e-14,
            )
        )
    )


@lru_cache(maxsize=2)
def _synthetic_case(n_cvt):
    """A deterministic SNP with a strict interior MLE maximum."""
    rng = np.random.default_rng(20260923)
    n_samples = 40
    eigenvalues = np.exp(np.linspace(np.log(0.08), np.log(6.0), n_samples))
    UtW = np.column_stack(
        [np.ones(n_samples)] + [rng.standard_normal(n_samples)] * (n_cvt - 1)
    )
    Uty = rng.standard_normal(n_samples)
    kinship = np.diag(eigenvalues)
    log_grid = np.linspace(np.log(1e-3), np.log(1e3), 241)
    for _ in range(32):
        Utg = rng.standard_normal(n_samples)
        scores = np.array(
            [
                mle_score_log_lambda(kinship, UtW, Utg, Uty, np.exp(point))
                for point in log_grid
            ]
        )
        crossings = np.flatnonzero((scores[:-1] > 0.0) & (scores[1:] < 0.0))
        if crossings.size:
            index = int(crossings[0])
            root = _mle_root(
                eigenvalues,
                UtW,
                Uty,
                Utg,
                np.exp(log_grid[index]),
                np.exp(log_grid[index + 1]),
            )
            return eigenvalues, UtW, Uty, Utg, root
    raise AssertionError("deterministic fixture search found no strict MLE maximum")


@pytest.mark.tier0
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("n_cvt", [1, 2])
def test_mle_optimizer_matches_independent_root(backend, n_cvt):
    eigenvalues, UtW, Uty, Utg, root = _synthetic_case(n_cvt)
    actual = _mle_lambda(backend, n_cvt, eigenvalues, UtW, Uty, Utg, 20)
    np.testing.assert_allclose(actual, root, rtol=1e-8, atol=0.0)


@lru_cache(maxsize=1)
def _flat_weighted_peak():
    """boundary5 of the weighted mode-4 fixture, rotated by its own kinship.

    Its MLE curvature is about -6e-5 per (log lambda)^2, so golden section
    alone stalls where rounding decides the likelihood comparisons.
    """
    model = _weighted_model()
    eigenvalues, eigenvectors = np.linalg.eigh(model["kinship"])
    column = model["selected_snp_ids"].index("boundary5")
    UtW = eigenvectors.T @ model["covariates"]
    Uty = eigenvectors.T @ model["phenotype"]
    Utg = eigenvectors.T @ model["genotypes"][:, column]
    return eigenvalues, UtW, Uty, Utg


@pytest.mark.tier0
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("n_refine", [20, 30])
@pytest.mark.parametrize("seed", [None, 0, 1, 2, 3])
def test_flat_mle_peak_is_insensitive_to_last_bit_noise(backend, n_refine, seed):
    """One-ulp eigenvalue noise must not move lambda beyond the root's own shift."""
    eigenvalues, UtW, Uty, Utg = _flat_weighted_peak()
    if seed is not None:
        ulps = np.random.default_rng(seed).integers(-1, 2, size=eigenvalues.shape)
        eigenvalues = eigenvalues + ulps * np.spacing(eigenvalues)
    root = _mle_root(eigenvalues, UtW, Uty, Utg, 4e-3, 5.5e-3)
    actual = _mle_lambda(backend, 2, eigenvalues, UtW, Uty, Utg, n_refine)
    np.testing.assert_allclose(actual, root, rtol=1e-8, atol=0.0)
