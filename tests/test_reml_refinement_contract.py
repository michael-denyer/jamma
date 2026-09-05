"""Independent contracts for safeguarded REML score refinement."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from scipy.optimize import brentq

from jamma.lmm import accel
from jamma.lmm.likelihood_numpy import (
    golden_section_optimize_lambda_numpy,
    golden_section_optimize_lambda_split_ncvt1_numpy,
)
from jamma.lmm.pab import build_pab_table_for_c
from jamma.lmm.reml_score import (
    _batch_reml_score_log_lambda_numpy,
    _batch_reml_score_log_lambda_split_ncvt1_numpy,
    _refine_reml_optima,
)
from jamma.lmm.uab import (
    batch_compute_iab_numpy,
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_iab_invariant_scalars_ncvt1,
    compute_uab_invariant_soa,
)
from tests.conftest import require_fixture, requires_c
from tests.fixture_paths import SYNTHETIC
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


@pytest.fixture(scope="module")
def tiny_reml_peak():
    """The tiny interior peak exposed by saving a full-population kinship.

    Independently construct the same scientific input as the pipeline regression:
    retain the first 50 phenotypes and SNPs passing MAF 0.3 in those samples.
    The root uses dense projector algebra, without production Pab or score code.
    """
    from jamma.io.plink import load_plink_binary, read_fam_phenotypes

    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)
    genotypes = load_plink_binary(SYNTHETIC.bfile).genotypes.astype(np.float64)
    analysed = genotypes[:50]
    frequencies = np.nanmean(analysed, axis=0) / 2.0
    keep = (np.minimum(frequencies, 1.0 - frequencies) >= 0.3) & (
        np.nanvar(analysed, axis=0) > 0.0
    )
    assert np.count_nonzero(keep) == 265
    selected = genotypes[:, keep]
    means = np.nanmean(selected, axis=0)
    centred = (np.where(np.isnan(selected), means, selected) - means)[:50]
    centred -= centred.mean(axis=0)
    kinship = centred @ centred.T / centred.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(kinship)
    eigenvalues[np.abs(eigenvalues) < 1e-10] = 0.0
    UtW = np.ascontiguousarray(eigenvectors.T @ np.ones((50, 1)))
    Uty = np.ascontiguousarray(eigenvectors.T @ read_fam_phenotypes(SYNTHETIC.fam)[:50])
    # This is association row 120 after filtering, the smallest interior peak.
    genotype = analysed[:, keep][:, 120]
    genotype = np.where(np.isnan(genotype), np.nanmean(genotype), genotype)
    Utg = np.ascontiguousarray(eigenvectors.T @ genotype)
    oracle_args = eigenvalues, UtW, Uty, Utg
    expected_log = brentq(
        lambda point: dense_reml_score_log_lambda(*oracle_args, np.exp(point)),
        np.log(8e-5),
        np.log(1.4e-4),
        xtol=1e-13,
    )
    return oracle_args, expected_log


@pytest.mark.tier0
@pytest.mark.parametrize("backend", ["numpy", "split"])
def test_refinement_converges_after_resolvable_first_step_residual(
    tiny_reml_peak, backend
):
    """One Newton step leaves ~7e-6 relative error at this actual tiny peak.

    Fix the initial relative displacement so the regression is independent of
    the vendor BLAS's last bits changing the golden-section midpoint.
    """
    oracle_args, expected_log = tiny_reml_peak
    eigenvalues, UtW, Uty, Utg = oracle_args
    if backend == "numpy":
        uab = batch_compute_uab_numpy(1, UtW, Uty, Utg[None, :])

        def score_at(points, indices):
            return _batch_reml_score_log_lambda_numpy(
                1, points, eigenvalues, uab[indices]
            )

    else:
        varying = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, Utg[None, :])
        invariant = compute_uab_invariant_soa(UtW, Uty, n_cvt=1)

        def score_at(points, indices):
            return _batch_reml_score_log_lambda_split_ncvt1_numpy(
                points, eigenvalues, varying[indices], invariant
            )

    actual = _refine_reml_optima(
        np.array([expected_log + np.log1p(0.0022)]),
        np.array([np.log(1e-5)]),
        np.array([np.log(1e-3)]),
        np.array([True]),
        score_at,
    )
    np.testing.assert_allclose(
        np.exp(actual[0]), np.exp(expected_log), rtol=1e-8, atol=0.0
    )


@pytest.mark.tier0
@pytest.mark.parametrize(
    "backend", ["numpy", "split", pytest.param("native", marks=requires_c)]
)
def test_tiny_reml_peak_optimizer_matches_independent_root(tiny_reml_peak, backend):
    oracle_args, expected_log = tiny_reml_peak
    eigenvalues, UtW, Uty, Utg = oracle_args
    invariant = compute_uab_invariant_soa(UtW, Uty, n_cvt=1)
    if backend == "numpy":
        uab = batch_compute_uab_numpy(1, UtW, Uty, Utg[None, :])
        actual, _, _ = golden_section_optimize_lambda_numpy(
            1, eigenvalues, uab, batch_compute_iab_numpy(1, uab)
        )
    elif backend == "split":
        varying = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, Utg[None, :])
        actual, _, _ = golden_section_optimize_lambda_split_ncvt1_numpy(
            eigenvalues,
            varying,
            invariant,
            *compute_iab_invariant_scalars_ncvt1(invariant),
        )
    else:
        workspace = accel.require().create_workspace_ncvt1_c(
            eigenvalues,
            invariant,
            UtW[:, 0],
            Uty,
            len(Uty),
            _L_MIN,
            _L_MAX,
            50,
            20,
            lmm_mode=1,
        )
        actual = accel.require().compute_lmm_chunk_ncvt1_c(workspace, Utg[None, :], 1)[
            "lambdas"
        ]
    np.testing.assert_allclose(actual[0], np.exp(expected_log), rtol=1e-8, atol=0.0)
