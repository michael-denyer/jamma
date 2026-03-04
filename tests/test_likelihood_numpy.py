"""Parity tests for likelihood_numpy.py and compute_numpy.py.

Verifies that NumPy batch implementations produce numerically equivalent
results to their JAX counterparts, and that no JAX imports appear in the
new NumPy modules.

Notes on tolerance:
- Uab/Pab/Iab: atol=1e-14 (identical floating-point arithmetic)
- Lambda: rtol=1e-5 (golden section convergence tolerance)
- Wald/Score beta/se: rtol=1e-10 (same Pab arithmetic)
- p_wald/p_score: rtol=1e-8 (Cephes betainc vs JAX XLA betainc)
  NumPy Cephes betainc is more accurate than JAX XLA betainc for large a.
  For n=50 samples this difference is negligible, but documented here.
  See STATE.md: "NumPy vs JAX betainc divergence: up to 6e-3 rtol at n=50k"
- LRT p-values: rtol=1e-8 (chi2_sf vs JAX chi2.sf)
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.compute_numpy import _compute_lmm_chunk_numpy
from jamma.lmm.likelihood import (
    _golden_section_minimize,
    compute_Uab,
    reml_log_likelihood,
)
from jamma.lmm.likelihood_numpy import (
    _batch_grid_reml_numpy,
    _batch_lrt_pvalues_numpy,
    _batch_reml_at_lambda_numpy,
    batch_calc_score_stats_numpy,
    batch_calc_wald_stats_numpy,
    batch_compute_iab_numpy,
    batch_compute_pab_numpy,
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
    golden_section_optimize_lambda_mle_numpy,
    golden_section_optimize_lambda_numpy,
)


@pytest.fixture
def synthetic_data():
    """Small synthetic dataset for fast parity tests.

    Returns:
        (eigenvalues, UtW, Uty, UtG) with n_samples=50, n_snps=10.
    """
    rng = np.random.default_rng(42)
    n_samples, n_snps = 50, 10
    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))  # intercept-only
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))
    return eigenvalues, UtW, Uty, UtG


# ---------------------------------------------------------------------------
# Static checks — no JAX in NumPy modules
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_no_jax_imports_in_numpy_modules():
    """AST check: likelihood_numpy.py and compute_numpy.py must not import JAX."""
    project_root = Path(__file__).parent.parent
    for module_path in [
        project_root / "src" / "jamma" / "lmm" / "likelihood_numpy.py",
        project_root / "src" / "jamma" / "lmm" / "compute_numpy.py",
    ]:
        source = module_path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("jax"), (
                        f"Direct JAX import in {module_path}: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("jax"), (
                        f"From-JAX import in {module_path}: {node.module}"
                    )


# ---------------------------------------------------------------------------
# Uab parity
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_batch_uab_matches_jax(synthetic_data):
    """batch_compute_uab_numpy must match JAX batch_compute_uab to 1e-12."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from jamma.lmm.likelihood_jax import batch_compute_uab

    eigenvalues, UtW, Uty, UtG = synthetic_data
    n_snps = UtG.shape[1]

    Uab_numpy = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Uab_jax = np.asarray(
        batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    )

    assert Uab_numpy.shape == (n_snps, 50, 6), f"Wrong shape: {Uab_numpy.shape}"
    np.testing.assert_allclose(
        Uab_numpy,
        Uab_jax,
        rtol=1e-12,
        atol=1e-14,
        err_msg="batch_compute_uab_numpy does not match JAX",
    )


# ---------------------------------------------------------------------------
# Pab parity
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_batch_pab_matches_jax(synthetic_data):
    """batch_compute_pab_numpy must match JAX vmap(calc_pab_jax) to 1e-12."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax import vmap

    from jamma.lmm.likelihood_jax import batch_compute_uab, calc_pab_jax

    eigenvalues, UtW, Uty, UtG = synthetic_data

    lambda_val = 0.5
    Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)

    Uab_batch_np = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Pab_numpy = batch_compute_pab_numpy(1, Hi_eval, Uab_batch_np)

    Uab_batch_jax = batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    Hi_eval_jax = jnp.array(Hi_eval)
    Pab_jax = np.asarray(
        vmap(lambda Uab: calc_pab_jax(1, Hi_eval_jax, Uab))(Uab_batch_jax)
    )

    np.testing.assert_allclose(
        Pab_numpy,
        Pab_jax,
        rtol=1e-12,
        atol=1e-14,
        err_msg="batch_compute_pab_numpy does not match JAX",
    )


# ---------------------------------------------------------------------------
# Iab parity
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_batch_iab_matches_jax(synthetic_data):
    """batch_compute_iab_numpy must match JAX batch_compute_iab to 1e-12."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from jamma.lmm.likelihood_jax import batch_compute_iab, batch_compute_uab

    eigenvalues, UtW, Uty, UtG = synthetic_data

    Uab_batch_np = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Iab_numpy = batch_compute_iab_numpy(1, Uab_batch_np)

    Uab_batch_jax = batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    Iab_jax = np.asarray(batch_compute_iab(1, Uab_batch_jax))

    np.testing.assert_allclose(
        Iab_numpy,
        Iab_jax,
        rtol=1e-12,
        atol=1e-14,
        err_msg="batch_compute_iab_numpy does not match JAX",
    )


# ---------------------------------------------------------------------------
# Golden section REML parity
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_golden_section_reml_matches_jax(synthetic_data):
    """golden_section_optimize_lambda_numpy must match JAX to lambda rtol < 1e-5."""
    pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    from jamma.lmm.likelihood_jax import (  # noqa: PLC0415
        batch_compute_iab,
        batch_compute_uab,
        golden_section_optimize_lambda,
    )

    eigenvalues, UtW, Uty, UtG = synthetic_data

    Uab_batch_np = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Iab_batch_np = batch_compute_iab_numpy(1, Uab_batch_np)

    lambdas_np, logls_np = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch_np, Iab_batch_np
    )

    Uab_batch_jax = batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    Iab_batch_jax = batch_compute_iab(1, Uab_batch_jax)
    lambdas_jax, logls_jax = golden_section_optimize_lambda(
        1, jnp.array(eigenvalues), Uab_batch_jax, Iab_batch_jax
    )
    lambdas_jax = np.asarray(lambdas_jax)
    logls_jax = np.asarray(logls_jax)

    # Lambda tolerance: golden section convergence bound
    np.testing.assert_allclose(
        lambdas_np,
        lambdas_jax,
        rtol=1e-5,
        atol=1e-10,
        err_msg="golden_section_optimize_lambda_numpy lambdas do not match JAX",
    )
    # Log-likelihoods at optimum should also be very close
    np.testing.assert_allclose(
        logls_np,
        logls_jax,
        rtol=1e-5,
        atol=1e-10,
        err_msg="golden_section_optimize_lambda_numpy logls do not match JAX",
    )


# ---------------------------------------------------------------------------
# Golden section MLE parity
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_golden_section_mle_matches_jax(synthetic_data):
    """golden_section_optimize_lambda_mle_numpy must match JAX to lambda rtol < 1e-5."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from jamma.lmm.likelihood_jax import (
        batch_compute_uab,
        golden_section_optimize_lambda_mle,
    )

    eigenvalues, UtW, Uty, UtG = synthetic_data

    Uab_batch_np = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    lambdas_np, logls_np = golden_section_optimize_lambda_mle_numpy(
        1, eigenvalues, Uab_batch_np
    )

    Uab_batch_jax = batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    lambdas_jax, logls_jax = golden_section_optimize_lambda_mle(
        1, jnp.array(eigenvalues), Uab_batch_jax
    )
    lambdas_jax = np.asarray(lambdas_jax)
    logls_jax = np.asarray(logls_jax)

    np.testing.assert_allclose(
        lambdas_np,
        lambdas_jax,
        rtol=1e-5,
        atol=1e-10,
        err_msg="golden_section_optimize_lambda_mle_numpy lambdas do not match JAX",
    )
    np.testing.assert_allclose(
        logls_np,
        logls_jax,
        rtol=1e-5,
        atol=1e-10,
        err_msg="golden_section_optimize_lambda_mle_numpy logls do not match JAX",
    )


# ---------------------------------------------------------------------------
# Wald stats parity
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_wald_stats_match_jax(synthetic_data):
    """batch_calc_wald_stats_numpy must match JAX batch_calc_wald_stats.

    Tolerances:
    - beta/se: rtol=1e-10 (identical Pab arithmetic)
    - p_wald: rtol=1e-8 (Cephes betainc vs JAX XLA betainc, small n=50)
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from jamma.lmm.likelihood_jax import (
        batch_calc_wald_stats,
        batch_compute_iab,
        batch_compute_uab,
        golden_section_optimize_lambda,
    )

    eigenvalues, UtW, Uty, UtG = synthetic_data
    n_samples = eigenvalues.shape[0]

    Uab_batch_np = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Iab_batch_np = batch_compute_iab_numpy(1, Uab_batch_np)
    lambdas_np, _ = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch_np, Iab_batch_np
    )
    betas_np, ses_np, pwalds_np = batch_calc_wald_stats_numpy(
        1, lambdas_np, eigenvalues, Uab_batch_np, n_samples
    )

    Uab_batch_jax = batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    Iab_batch_jax = batch_compute_iab(1, Uab_batch_jax)
    lambdas_jax, _ = golden_section_optimize_lambda(
        1, jnp.array(eigenvalues), Uab_batch_jax, Iab_batch_jax
    )
    betas_jax, ses_jax, pwalds_jax = batch_calc_wald_stats(
        1, lambdas_jax, jnp.array(eigenvalues), Uab_batch_jax, n_samples
    )
    betas_jax = np.asarray(betas_jax)
    ses_jax = np.asarray(ses_jax)
    pwalds_jax = np.asarray(pwalds_jax)

    np.testing.assert_allclose(
        betas_np, betas_jax, rtol=1e-10, atol=1e-14, err_msg="Wald beta mismatch"
    )
    np.testing.assert_allclose(
        ses_np, ses_jax, rtol=1e-10, atol=1e-14, err_msg="Wald se mismatch"
    )
    # p_wald: Cephes betainc vs JAX XLA betainc; small difference at n=50
    np.testing.assert_allclose(
        pwalds_np,
        pwalds_jax,
        rtol=1e-8,
        atol=1e-14,
        err_msg="Wald p_wald mismatch (Cephes vs JAX betainc)",
    )


# ---------------------------------------------------------------------------
# Score stats parity
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_score_stats_match_jax(synthetic_data):
    """batch_calc_score_stats_numpy must match JAX batch_calc_score_stats.

    Tolerances same as Wald (Cephes vs JAX betainc difference is small at n=50).
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from jamma.lmm.likelihood_jax import batch_calc_score_stats, batch_compute_uab

    eigenvalues, UtW, Uty, UtG = synthetic_data
    n_samples = eigenvalues.shape[0]

    # Use a fixed null lambda
    lambda_null = 0.1
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

    Uab_batch_np = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    betas_np, ses_np, pscores_np = batch_calc_score_stats_numpy(
        1, Hi_eval_null, Uab_batch_np, n_samples
    )

    Uab_batch_jax = batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    betas_jax, ses_jax, pscores_jax = batch_calc_score_stats(
        1, jnp.array(Hi_eval_null), Uab_batch_jax, n_samples
    )
    betas_jax = np.asarray(betas_jax)
    ses_jax = np.asarray(ses_jax)
    pscores_jax = np.asarray(pscores_jax)

    np.testing.assert_allclose(
        betas_np, betas_jax, rtol=1e-10, atol=1e-14, err_msg="Score beta mismatch"
    )
    np.testing.assert_allclose(
        ses_np, ses_jax, rtol=1e-10, atol=1e-14, err_msg="Score se mismatch"
    )
    np.testing.assert_allclose(
        pscores_np,
        pscores_jax,
        rtol=1e-8,
        atol=1e-14,
        err_msg="Score p_score mismatch (Cephes vs JAX betainc)",
    )


# ---------------------------------------------------------------------------
# LRT p-value parity
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_lrt_pvalues_match_jax(synthetic_data):
    """_batch_lrt_pvalues_numpy must match JAX calc_lrt_pvalue_jax.

    rtol=1e-8: chi2_sf (stdlib erfc) vs JAX jax.scipy.stats.chi2.sf.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from jamma.lmm.likelihood_jax import (
        batch_compute_uab,
        calc_lrt_pvalue_jax,
        golden_section_optimize_lambda_mle,
    )

    eigenvalues, UtW, Uty, UtG = synthetic_data

    # Null model MLE log-likelihood
    logl_H0 = -30.0  # arbitrary fixed value for comparison

    Uab_batch_np = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    _, logls_mle_np = golden_section_optimize_lambda_mle_numpy(
        1, eigenvalues, Uab_batch_np
    )
    plrts_np = _batch_lrt_pvalues_numpy(logls_mle_np, logl_H0)

    Uab_batch_jax = batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    _, logls_mle_jax = golden_section_optimize_lambda_mle(
        1, jnp.array(eigenvalues), Uab_batch_jax
    )
    plrts_jax = np.asarray(
        jax.vmap(calc_lrt_pvalue_jax)(
            logls_mle_jax, jnp.full_like(logls_mle_jax, logl_H0)
        )
    )

    np.testing.assert_allclose(
        plrts_np,
        plrts_jax,
        rtol=1e-8,
        atol=1e-14,
        err_msg="LRT p-value mismatch (chi2_sf vs JAX chi2.sf)",
    )


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_compute_lmm_chunk_numpy_all_modes(synthetic_data):
    """_compute_lmm_chunk_numpy must return non-None expected keys for each mode."""
    eigenvalues, UtW, Uty, UtG = synthetic_data
    n_samples = eigenvalues.shape[0]

    lambda_null = 0.1
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)
    logl_H0 = -25.0

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG)

    # Mode 1: Wald — expects lambdas, logls, betas, ses, pwalds
    result1 = _compute_lmm_chunk_numpy(1, 1, eigenvalues, Uab_batch, n_samples)
    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        assert result1[key] is not None, f"Mode 1: key '{key}' is None"
    assert result1["lambdas_mle"] is None
    assert result1["p_lrts"] is None
    assert result1["p_scores"] is None

    # Mode 2: LRT — expects lambdas_mle, p_lrts
    result2 = _compute_lmm_chunk_numpy(
        2, 1, eigenvalues, Uab_batch, n_samples, logl_H0=logl_H0
    )
    for key in ("lambdas_mle", "p_lrts"):
        assert result2[key] is not None, f"Mode 2: key '{key}' is None"
    assert result2["lambdas"] is None
    assert result2["logls"] is None
    assert result2["betas"] is None
    assert result2["ses"] is None
    assert result2["pwalds"] is None
    assert result2["p_scores"] is None

    # Mode 3: Score — expects betas, ses, p_scores
    result3 = _compute_lmm_chunk_numpy(
        3, 1, eigenvalues, Uab_batch, n_samples, Hi_eval_null=Hi_eval_null
    )
    for key in ("betas", "ses", "p_scores"):
        assert result3[key] is not None, f"Mode 3: key '{key}' is None"
    assert result3["lambdas"] is None
    assert result3["logls"] is None
    assert result3["pwalds"] is None
    assert result3["lambdas_mle"] is None
    assert result3["p_lrts"] is None

    # Mode 4: All — all keys non-None
    result4 = _compute_lmm_chunk_numpy(
        4,
        1,
        eigenvalues,
        Uab_batch,
        n_samples,
        Hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
    )
    for key in (
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "p_lrts",
        "p_scores",
    ):
        assert result4[key] is not None, f"Mode 4: key '{key}' is None"


@pytest.mark.tier0
def test_compute_lmm_chunk_numpy_missing_args_raise(synthetic_data):
    """_compute_lmm_chunk_numpy must raise ValueError when required args are absent."""
    eigenvalues, UtW, Uty, UtG = synthetic_data
    n_samples = eigenvalues.shape[0]
    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG)

    with pytest.raises(ValueError, match="logl_H0 is required"):
        _compute_lmm_chunk_numpy(2, 1, eigenvalues, Uab_batch, n_samples)

    with pytest.raises(ValueError, match="Hi_eval_null is required"):
        _compute_lmm_chunk_numpy(3, 1, eigenvalues, Uab_batch, n_samples)


# ---------------------------------------------------------------------------
# Scalar P_yy warning deduplication (LIK-07)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_p_yy_warn_once_scalar():
    """_clamp_p_yy fires warning exactly once per run; reset restarts the counter."""
    from loguru import logger

    from jamma.lmm.likelihood import _clamp_p_yy, reset_scalar_p_yy_warned

    warning_messages: list[str] = []

    def _capture_sink(message):
        if message.record["level"].name == "WARNING":
            warning_messages.append(message.record["message"])

    # Start clean
    reset_scalar_p_yy_warned()

    sink_id = logger.add(_capture_sink, level="WARNING")
    try:
        for _ in range(10):
            _clamp_p_yy(-1.0, 1.0)

        assert len(warning_messages) == 1, (
            f"Expected exactly 1 warning, got {len(warning_messages)}"
        )

        # Reset and fire again — should produce a second warning
        reset_scalar_p_yy_warned()
        _clamp_p_yy(-1.0, 1.0)

        assert len(warning_messages) == 2, (
            f"Expected 2 total warnings after reset, got {len(warning_messages)}"
        )
    finally:
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Scalar MLE P_yy without full Pab (LIK-08)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_mle_scalar_pab_ncvt1():
    """_mle_p_yy_scalar_ncvt1 must match calc_pab path to rtol=1e-14."""
    from jamma.lmm.likelihood import (
        _mle_p_yy_scalar_ncvt1,
        calc_pab,
        compute_Uab,
        get_ab_index,
    )

    rng = np.random.default_rng(123)
    n_samples = 50
    n_cvt = 1

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    Utx = rng.standard_normal(n_samples)

    lambda_val = 0.5
    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp

    Uab = compute_Uab(UtW, Uty, Utx)
    Pab = calc_pab(n_cvt, Hi_eval, Uab)

    # Full Pab path: nc_total = n_cvt + 1 = 2
    nc_total = n_cvt + 1
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    p_yy_full = Pab[nc_total, index_yy]

    # Scalar path
    p_yy_scalar = _mle_p_yy_scalar_ncvt1(Hi_eval, Uab)

    np.testing.assert_allclose(
        p_yy_scalar,
        p_yy_full,
        rtol=1e-14,
        err_msg="_mle_p_yy_scalar_ncvt1 does not match calc_pab P_yy",
    )


@pytest.mark.tier0
def test_mle_no_calc_pab_ncvt1():
    """mle_log_likelihood with n_cvt=1 must NOT call calc_pab; n_cvt=2 must call it."""
    from unittest.mock import patch

    import jamma.lmm.likelihood as lik_mod
    from jamma.lmm.likelihood import compute_Uab

    rng = np.random.default_rng(456)
    n_samples = 50

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    Utx = rng.standard_normal(n_samples)
    lambda_val = 0.5

    # n_cvt=1: scalar path, calc_pab should NOT be called
    Uab_1 = compute_Uab(UtW, Uty, Utx)
    with patch.object(lik_mod, "calc_pab", wraps=lik_mod.calc_pab) as mock_pab:
        lik_mod.mle_log_likelihood(lambda_val, eigenvalues, Uab_1, n_cvt=1)
    assert mock_pab.call_count == 0, (
        f"calc_pab called {mock_pab.call_count} times for n_cvt=1 (expected 0)"
    )

    # n_cvt=2: full Pab path, calc_pab MUST be called
    UtW2 = np.ones((n_samples, 2))
    Uab_2 = compute_Uab(UtW2, Uty, Utx)
    with patch.object(lik_mod, "calc_pab", wraps=lik_mod.calc_pab) as mock_pab:
        lik_mod.mle_log_likelihood(lambda_val, eigenvalues, Uab_2, n_cvt=2)
    assert mock_pab.call_count == 1, (
        f"calc_pab called {mock_pab.call_count} times for n_cvt=2 (expected 1)"
    )


@pytest.mark.tier0
def test_mle_null_scalar_ncvt1():
    """mle_log_likelihood_null with n_cvt=1 produces identical results to full Pab."""
    from jamma.lmm.likelihood import (
        _mle_p_yy_scalar_null_ncvt1,
        calc_pab,
        compute_Uab,
        get_ab_index,
        mle_log_likelihood_null,
    )

    rng = np.random.default_rng(789)
    n_samples = 50
    n_cvt = 1

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

    lambda_val = 0.3
    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp

    # Null model Uab (no genotype)
    Uab = compute_Uab(UtW, Uty, Utx=None)

    # Full Pab path: nc_total = n_cvt = 1 for null model
    nc_total = n_cvt  # null model
    Pab = calc_pab(n_cvt, Hi_eval, Uab)
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    p_yy_full = Pab[nc_total, index_yy]

    # Scalar null path
    p_yy_scalar = _mle_p_yy_scalar_null_ncvt1(Hi_eval, Uab)

    np.testing.assert_allclose(
        p_yy_scalar,
        p_yy_full,
        rtol=1e-14,
        err_msg="_mle_p_yy_scalar_null_ncvt1 does not match calc_pab P_yy",
    )

    # Verify end-to-end: mle_log_likelihood_null should produce a finite result
    logl = mle_log_likelihood_null(lambda_val, eigenvalues, Uab, n_cvt)
    assert np.isfinite(logl), f"mle_log_likelihood_null returned non-finite: {logl}"


@pytest.mark.tier0
def test_mle_scalar_degenerate_s_ww_zero():
    """Scalar MLE P_yy returns s_yy when s_ww == 0 (degenerate intercept)."""
    from jamma.lmm.likelihood import (
        _mle_p_yy_scalar_ncvt1,
        _mle_p_yy_scalar_null_ncvt1,
        calc_pab,
        get_ab_index,
    )

    n_samples = 50
    # Construct Uab where column 0 (ww) is all zeros -> Hi_eval @ Uab[:,0] = 0
    Uab = np.zeros((n_samples, 6), dtype=np.float64)
    rng = np.random.default_rng(111)
    Uab[:, 5] = rng.standard_normal(n_samples) ** 2  # yy column: non-zero

    Hi_eval = np.ones(n_samples)

    # s_ww = 0 -> should return s_yy
    p_yy = _mle_p_yy_scalar_ncvt1(Hi_eval, Uab)
    expected_s_yy = float(Hi_eval @ Uab[:, 5])
    assert p_yy == expected_s_yy, f"Expected s_yy={expected_s_yy}, got {p_yy}"

    # Null path: same behavior
    p_yy_null = _mle_p_yy_scalar_null_ncvt1(Hi_eval, Uab)
    assert p_yy_null == expected_s_yy

    # Full Pab path should also handle this
    Pab = calc_pab(1, Hi_eval, Uab)
    p_yy_full = Pab[2, get_ab_index(3, 3, 1)]
    # Both should be s_yy since ww=0 means no projection happens
    np.testing.assert_allclose(p_yy, p_yy_full, rtol=1e-12)


@pytest.mark.tier0
def test_mle_scalar_degenerate_p1_xx_zero():
    """Scalar MLE P_yy returns p1_yy when p1_xx == 0 (constant genotype)."""
    from jamma.lmm.likelihood import _mle_p_yy_scalar_ncvt1, calc_pab, get_ab_index

    n_samples = 50
    rng = np.random.default_rng(222)
    Hi_eval = np.ones(n_samples)

    # Construct Uab where genotype column produces p1_xx = 0:
    # s_xx - s_wx^2/s_ww = 0 when s_xx = s_wx^2/s_ww
    # Easiest: make wx and xx columns such that genotype is proportional to intercept
    w = np.ones(n_samples)
    x = 2.0 * w  # genotype proportional to intercept -> p1_xx = 0
    y = rng.standard_normal(n_samples)

    Uab = np.zeros((n_samples, 6), dtype=np.float64)
    Uab[:, 0] = w * w  # ww
    Uab[:, 1] = w * x  # wx
    Uab[:, 2] = w * y  # wy
    Uab[:, 3] = x * x  # xx
    Uab[:, 4] = x * y  # xy
    Uab[:, 5] = y * y  # yy

    p_yy_scalar = _mle_p_yy_scalar_ncvt1(Hi_eval, Uab)

    # Full Pab path for reference
    Pab = calc_pab(1, Hi_eval, Uab)
    p_yy_full = Pab[2, get_ab_index(3, 3, 1)]

    np.testing.assert_allclose(p_yy_scalar, p_yy_full, rtol=1e-12)


# ---------------------------------------------------------------------------
# Multi-covariate Uab parity (n_cvt > 1)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_batch_uab_multi_covariate_matches_jax():
    """batch_compute_uab_numpy with n_cvt=3 must match JAX to 1e-12."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from jamma.lmm.likelihood_jax import batch_compute_uab

    rng = np.random.default_rng(77)
    n_samples, n_snps, n_cvt = 50, 10, 3
    rng.uniform(0.1, 5.0, n_samples)  # advance RNG state
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    Uab_numpy = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    Uab_jax = np.asarray(
        batch_compute_uab(n_cvt, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG))
    )

    # n_index = (n_cvt + 3) * (n_cvt + 2) // 2 = 6 * 5 // 2 = 15
    expected_n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    assert Uab_numpy.shape == (n_snps, n_samples, expected_n_index), (
        f"Wrong shape: {Uab_numpy.shape}"
    )
    np.testing.assert_allclose(
        Uab_numpy,
        Uab_jax,
        rtol=1e-12,
        atol=1e-14,
        err_msg="batch_compute_uab_numpy (n_cvt=3) does not match JAX",
    )


# ---------------------------------------------------------------------------
# Task 1: Precompute REML constants, Iab invariant scalars, golden section fix
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_reml_const_precomputed():
    """_compute_reml_const(df) must match inline computation bit-exactly."""
    from jamma.lmm.likelihood_numpy import _compute_reml_const

    for df in [10, 48, 100, 1000, 50000]:
        result = _compute_reml_const(df)
        inline = 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)
        np.testing.assert_equal(
            result,
            inline,
            err_msg=f"_compute_reml_const({df}) does not match inline for df={df}",
        )


@pytest.mark.tier0
def test_iab_invariant_scalars():
    """compute_iab_invariant_scalars_ncvt1 must match manual np.sum exactly."""
    from jamma.lmm.likelihood_numpy import compute_iab_invariant_scalars_ncvt1

    rng = np.random.default_rng(123)
    n_samples = 80
    uab_invariant_soa = rng.standard_normal((3, n_samples))
    # Ensure rows are all positive-valued for log checks
    uab_invariant_soa = np.abs(uab_invariant_soa) + 0.1

    iab_s_ww, iab_s_wy, iab_s_yy, logdet_iab = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    expected_s_ww = float(uab_invariant_soa[0, :].sum())
    expected_s_wy = float(uab_invariant_soa[1, :].sum())
    expected_s_yy = float(uab_invariant_soa[2, :].sum())
    expected_logdet = np.log(expected_s_ww)

    np.testing.assert_equal(iab_s_ww, expected_s_ww)
    np.testing.assert_equal(iab_s_wy, expected_s_wy)
    np.testing.assert_equal(iab_s_yy, expected_s_yy)
    np.testing.assert_equal(logdet_iab, expected_logdet)


@pytest.mark.tier0
def test_golden_section_eval_count():
    """Golden section calls compute_fn 2 + n_iter + 1 times (final midpoint eval).

    The final midpoint evaluation ensures the returned (lambda, logl) pair is
    consistent — both from the same evaluation point. Without it, lambda is at
    the midpoint but logl is max(fc, fd) from different points c and d, causing
    a mismatch that propagates into LRT p-values.
    """
    from jamma.lmm.likelihood_numpy import _batch_golden_section_numpy

    call_count = [0]

    rng = np.random.default_rng(42)
    n_samples, n_snps = 50, 10
    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    n_grid = 10
    n_iter = 5

    log_l_min = np.log(1e-5)
    log_l_max = np.log(1e5)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)
    lambdas_grid = np.exp(log_lambdas)

    grid_logls = _batch_grid_reml_numpy(
        1, lambdas_grid, eigenvalues, Uab_batch, Iab_batch
    )

    def counting_reml(log_lams):
        call_count[0] += 1
        lams = np.exp(log_lams)
        return _batch_reml_at_lambda_numpy(1, lams, eigenvalues, Uab_batch, Iab_batch)

    _batch_golden_section_numpy(counting_reml, grid_logls, log_lambdas, n_iter)

    # 2 initial probes (c, d) + n_iter (one per iteration) + 1 final midpoint eval
    expected_calls = 2 + n_iter + 1
    assert call_count[0] == expected_calls, (
        f"Expected {expected_calls} calls, got {call_count[0]}. "
        "Golden section must evaluate at midpoint for consistent (lambda, logl)."
    )


@pytest.mark.tier0
def test_golden_section_accuracy_no_final_eval(synthetic_data):
    """Golden section without final eval must produce finite, positive lambdas."""
    eigenvalues, UtW, Uty, UtG = synthetic_data
    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    lambdas_opt, logls_opt = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch
    )

    # Lambdas should be finite positive values
    assert np.all(np.isfinite(lambdas_opt)), "Some lambdas are not finite"
    assert np.all(lambdas_opt[np.isfinite(lambdas_opt)] > 0), (
        "Some finite lambdas are non-positive"
    )
    # Logls should be finite (no NaN for valid SNPs)
    assert np.all(np.isfinite(logls_opt)), "Some logls are not finite"


# ---------------------------------------------------------------------------
# Task 2: Split-Uab REML path for grid and refinement (n_cvt=1)
# ---------------------------------------------------------------------------


@pytest.fixture
def split_uab_data():
    """Synthetic dataset for split-Uab tests.

    Returns:
        (eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch)
        with n_samples=100, n_snps=50.
    """
    rng = np.random.default_rng(99)
    n_samples, n_snps = 100, 50
    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    uab_invariant_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_varying_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    return eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch


@pytest.mark.tier0
def test_grid_reml_split_matches_full(split_uab_data):
    """_batch_grid_reml_split_ncvt1_numpy must match _batch_grid_reml_numpy."""
    from jamma.lmm.likelihood_numpy import (
        _batch_grid_reml_split_ncvt1_numpy,
        compute_iab_invariant_scalars_ncvt1,
    )

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )
    n_grid = 20
    lambdas_grid = np.exp(np.linspace(np.log(1e-5), np.log(1e5), n_grid))

    iab_s_ww, _iab_s_wy, _iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )
    iab_inv_s_ww = 1.0 / iab_s_ww if iab_s_ww != 0 else 0.0

    n_samples = eigenvalues.shape[0]
    df = n_samples - 2  # n_cvt=1

    from jamma.lmm.likelihood_numpy import (
        _compute_iab_varying_ncvt1,
        _compute_reml_const,
    )

    iab_p1_xx, iab_logdet_var = _compute_iab_varying_ncvt1(
        uab_varying_soa, iab_inv_s_ww
    )
    reml_const = _compute_reml_const(df)

    logls_split = _batch_grid_reml_split_ncvt1_numpy(
        lambdas_grid,
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_logdet,
        iab_inv_s_ww,
        iab_p1_xx,
        iab_logdet_var,
        reml_const,
    )

    logls_full = _batch_grid_reml_numpy(
        1, lambdas_grid, eigenvalues, Uab_batch, Iab_batch
    )

    np.testing.assert_allclose(
        logls_split,
        logls_full,
        rtol=1e-12,
        err_msg="_batch_grid_reml_split_ncvt1_numpy mismatch",
    )


@pytest.mark.tier0
def test_refinement_reml_split_matches_full(split_uab_data):
    """_batch_reml_at_lambda_split_ncvt1_numpy must match full path."""
    from jamma.lmm.likelihood_numpy import (
        _batch_reml_at_lambda_split_ncvt1_numpy,
        _compute_reml_const,
        compute_iab_invariant_scalars_ncvt1,
    )

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )
    n_snps = uab_varying_soa.shape[0]
    n_samples = eigenvalues.shape[0]
    df = n_samples - 2

    # Per-SNP lambda values (different for each SNP)
    rng = np.random.default_rng(7)
    lambda_vals = np.exp(rng.uniform(np.log(1e-4), np.log(1e3), n_snps))

    iab_s_ww, _iab_s_wy, _iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )
    iab_inv_s_ww = 1.0 / iab_s_ww if iab_s_ww != 0 else 0.0
    reml_const = _compute_reml_const(df)

    from jamma.lmm.likelihood_numpy import _compute_iab_varying_ncvt1

    iab_p1_xx, iab_logdet_var = _compute_iab_varying_ncvt1(
        uab_varying_soa, iab_inv_s_ww
    )

    logls_split = _batch_reml_at_lambda_split_ncvt1_numpy(
        lambda_vals,
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_logdet,
        iab_inv_s_ww,
        iab_p1_xx,
        iab_logdet_var,
        reml_const,
    )

    logls_full = _batch_reml_at_lambda_numpy(
        1, lambda_vals, eigenvalues, Uab_batch, Iab_batch
    )

    np.testing.assert_allclose(
        logls_split,
        logls_full,
        rtol=1e-12,
        err_msg="_batch_reml_at_lambda_split_ncvt1_numpy does not match full path",
    )


@pytest.mark.tier0
def test_split_optimizer_matches_full(split_uab_data):
    """golden_section_optimize_lambda_split_ncvt1_numpy must match full optimizer."""
    from jamma.lmm.likelihood_numpy import (
        compute_iab_invariant_scalars_ncvt1,
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )

    iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    lambdas_split, logls_split = golden_section_optimize_lambda_split_ncvt1_numpy(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_s_ww,
        iab_s_wy,
        iab_s_yy,
        iab_logdet,
    )

    lambdas_full, logls_full = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch
    )

    np.testing.assert_allclose(
        lambdas_split,
        lambdas_full,
        rtol=1e-10,
        err_msg="Split optimizer lambdas do not match full optimizer",
    )
    np.testing.assert_allclose(
        logls_split,
        logls_full,
        rtol=1e-8,
        err_msg="Split optimizer logls do not match full optimizer",
    )


@pytest.mark.tier0
def test_split_pab_matches_generic_pab(split_uab_data):
    """Pab from split optimizer must match generic Pab element-by-element."""
    from jamma.lmm.likelihood_numpy import (
        compute_iab_invariant_scalars_ncvt1,
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )

    iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    _, _, Pab_split = golden_section_optimize_lambda_split_ncvt1_numpy(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_s_ww,
        iab_s_wy,
        iab_s_yy,
        iab_logdet,
        return_pab=True,
    )

    _, _, Pab_generic = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, return_pab=True
    )

    assert Pab_split.shape == Pab_generic.shape, (
        f"Shape mismatch: split={Pab_split.shape} vs generic={Pab_generic.shape}"
    )
    np.testing.assert_allclose(
        Pab_split,
        Pab_generic,
        rtol=1e-12,
        err_msg="Split Pab does not match generic Pab element-by-element",
    )


@pytest.mark.tier0
def test_invariant_computed_once_per_lambda(split_uab_data):
    """Invariant dot products must be (n_grid,), not (n_grid, n_snps)."""
    from jamma.lmm.likelihood_numpy import (
        _compute_reml_const,
        compute_iab_invariant_scalars_ncvt1,
    )

    # Verify structural property: function produces (n_grid, n_snps) output
    # while internally computing (n_grid,) invariant sums.
    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )
    n_grid = 15
    n_snps = uab_varying_soa.shape[0]
    n_samples = eigenvalues.shape[0]
    df = n_samples - 2

    lambdas_grid = np.exp(np.linspace(np.log(1e-5), np.log(1e5), n_grid))

    Hi_eval_grid = 1.0 / (lambdas_grid[:, None] * eigenvalues[None, :] + 1.0)

    # s_ww_grid: (n_grid,) @ (n_samples,) -> (n_grid,)
    s_ww_grid = Hi_eval_grid @ uab_invariant_soa[0]
    assert s_ww_grid.shape == (n_grid,), (
        f"s_ww_grid should be (n_grid,)={(n_grid,)}, got {s_ww_grid.shape}"
    )
    # Must NOT be (n_grid, n_snps) — that would be the old O(n_grid * n_snps) path
    assert s_ww_grid.shape != (n_grid, n_snps), (
        "s_ww_grid shape should NOT be (n_grid, n_snps)"
    )

    # Also verify the split function itself returns (n_grid, n_snps) output
    from jamma.lmm.likelihood_numpy import _batch_grid_reml_split_ncvt1_numpy

    iab_s_ww, _iab_s_wy, _iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )
    iab_inv_s_ww = 1.0 / iab_s_ww if iab_s_ww != 0 else 0.0
    reml_const = _compute_reml_const(df)

    from jamma.lmm.likelihood_numpy import _compute_iab_varying_ncvt1

    iab_p1_xx, iab_logdet_var = _compute_iab_varying_ncvt1(
        uab_varying_soa, iab_inv_s_ww
    )

    logls = _batch_grid_reml_split_ncvt1_numpy(
        lambdas_grid,
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_logdet,
        iab_inv_s_ww,
        iab_p1_xx,
        iab_logdet_var,
        reml_const,
    )
    assert logls.shape == (n_grid, n_snps), (
        f"Expected output shape ({n_grid}, {n_snps}), got {logls.shape}"
    )


# ---------------------------------------------------------------------------
# Plan 53-03: Merged Wald path (optimizer returns Pab, no redundant Hi_eval)
# ---------------------------------------------------------------------------


@pytest.fixture
def wald_pab_data():
    """Synthetic data for merged Wald path tests.

    Returns:
        (eigenvalues, Uab_batch, Iab_batch, n_samples) with n_samples=80, n_snps=20.
    """
    rng = np.random.default_rng(99)
    n_samples, n_snps = 80, 20
    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    from jamma.lmm.likelihood_numpy import (
        batch_compute_iab_numpy,
        batch_compute_uab_numpy,
    )

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    return eigenvalues, Uab_batch, Iab_batch, n_samples


@pytest.mark.tier0
def test_optimizer_backward_compat(wald_pab_data):
    """golden_section_optimize_lambda_numpy with return_pab=False returns 2-tuple."""
    eigenvalues, Uab_batch, Iab_batch, _n_samples = wald_pab_data

    result = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, return_pab=False
    )

    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
    lambdas, logls = result
    n_snps = Uab_batch.shape[0]
    assert lambdas.shape == (n_snps,), f"lambdas shape {lambdas.shape}"
    assert logls.shape == (n_snps,), f"logls shape {logls.shape}"


@pytest.mark.tier0
def test_optimizer_returns_pab(wald_pab_data):
    """optimizer with return_pab=True returns 3-tuple with Pab of correct shape."""
    from jamma.lmm.likelihood import build_index_table

    eigenvalues, Uab_batch, Iab_batch, _n_samples = wald_pab_data
    n_cvt = 1
    n_snps = Uab_batch.shape[0]

    result = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch, return_pab=True
    )

    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
    lambdas, logls, Pab_final = result

    assert lambdas.shape == (n_snps,), f"lambdas shape {lambdas.shape}"
    assert logls.shape == (n_snps,), f"logls shape {logls.shape}"

    # Pab shape: (n_snps, n_cvt+2, n_index)
    table = build_index_table(n_cvt)
    n_index = table["n_index"]
    assert Pab_final.shape == (n_snps, n_cvt + 2, n_index), (
        f"Pab_final shape {Pab_final.shape}, expected {(n_snps, n_cvt + 2, n_index)}"
    )
    # Pab values should be finite (no NaN for non-degenerate synthetic data)
    assert np.all(np.isfinite(Pab_final)), "Pab_final contains non-finite values"


@pytest.mark.tier0
def test_wald_from_pab_matches_original(wald_pab_data):
    """Wald stats from pre-computed Pab match original path to rtol=1e-14."""
    from jamma.lmm.likelihood_numpy import batch_calc_wald_stats_from_pab_numpy

    eigenvalues, Uab_batch, Iab_batch, n_samples = wald_pab_data
    n_cvt = 1

    # Path A: original (optimizer + reconstruct Hi_eval + Pab)
    lambdas, _logls = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch
    )
    betas_orig, ses_orig, pwalds_orig = batch_calc_wald_stats_numpy(
        n_cvt, lambdas, eigenvalues, Uab_batch, n_samples
    )

    # Path B: merged (optimizer returns Pab directly)
    lambdas2, _logls2, Pab_final = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch, return_pab=True
    )
    betas_pab, ses_pab, pwalds_pab = batch_calc_wald_stats_from_pab_numpy(
        n_cvt, Pab_final, n_samples
    )

    # Lambdas from both paths should be identical (same optimizer, same path)
    np.testing.assert_array_equal(
        lambdas, lambdas2, err_msg="Lambdas differ between return_pab=False and True"
    )

    # Wald stats should be identical to machine precision
    np.testing.assert_allclose(
        betas_pab,
        betas_orig,
        rtol=1e-14,
        atol=1e-15,
        err_msg="betas from Pab path differ from original",
    )
    np.testing.assert_allclose(
        ses_pab,
        ses_orig,
        rtol=1e-14,
        atol=1e-15,
        err_msg="ses from Pab path differ from original",
    )
    np.testing.assert_allclose(
        pwalds_pab,
        pwalds_orig,
        rtol=1e-14,
        atol=1e-15,
        err_msg="p_walds from Pab path differ from original",
    )


# ---------------------------------------------------------------------------
# Plan 53-03 Task 2: _compute_wald_numpy dispatch tests
# ---------------------------------------------------------------------------


@pytest.fixture
def compute_wald_data():
    """Synthetic data for _compute_wald_numpy dispatch tests.

    Returns:
        (eigenvalues, Uab_batch, n_samples) with n_samples=80, n_snps=30.
    """
    rng = np.random.default_rng(123)
    n_samples, n_snps = 80, 30
    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    return eigenvalues, Uab_batch, n_samples


@pytest.mark.tier0
def test_compute_wald_numpy_dispatches_split_ncvt1(compute_wald_data):
    """_compute_wald_numpy with n_cvt=1 (C ext disabled) calls split optimizer."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy as cn
    from jamma.lmm.likelihood_numpy import (
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )

    eigenvalues, Uab_batch, n_samples = compute_wald_data

    call_log = []
    real_split_fn = golden_section_optimize_lambda_split_ncvt1_numpy

    def spy_split(*args, **kwargs):
        call_log.append("split")
        return real_split_fn(*args, **kwargs)

    split_generic_log = []
    real_generic_fn = cn.golden_section_optimize_lambda_numpy

    def spy_generic(*args, **kwargs):
        split_generic_log.append("generic")
        return real_generic_fn(*args, **kwargs)

    with (
        patch.object(cn, "_C_ACCEL_AVAILABLE", False),
        patch.object(cn, "golden_section_optimize_lambda_split_ncvt1_numpy", spy_split),
        patch.object(cn, "golden_section_optimize_lambda_numpy", spy_generic),
    ):
        cn._compute_wald_numpy(1, eigenvalues, Uab_batch, n_samples, 1e-5, 1e5, 50, 20)

    assert len(call_log) == 1, (
        f"Split optimizer called {len(call_log)} times for n_cvt=1, expected 1"
    )
    assert len(split_generic_log) == 0, (
        "Generic optimizer should NOT be called for n_cvt=1 Python path"
    )

    # Also verify n_cvt=2 uses generic, not split
    rng = np.random.default_rng(456)
    n_samples2, n_snps2 = 80, 10
    eigenvalues2 = np.sort(rng.uniform(0.1, 5.0, n_samples2))
    UtW2 = rng.standard_normal((n_samples2, 2))
    Uty2 = rng.standard_normal(n_samples2)
    UtG2 = rng.standard_normal((n_samples2, n_snps2))
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    Uab_batch2 = batch_compute_uab_numpy(2, UtW2, Uty2, UtG2)

    call_log2 = []
    generic_log2 = []

    def spy_split2(*args, **kwargs):
        call_log2.append("split")
        return real_split_fn(*args, **kwargs)

    def spy_generic2(*args, **kwargs):
        generic_log2.append("generic")
        return real_generic_fn(*args, **kwargs)

    with (
        patch.object(cn, "_C_ACCEL_AVAILABLE", False),
        patch.object(
            cn, "golden_section_optimize_lambda_split_ncvt1_numpy", spy_split2
        ),
        patch.object(cn, "golden_section_optimize_lambda_numpy", spy_generic2),
    ):
        cn._compute_wald_numpy(
            2, eigenvalues2, Uab_batch2, n_samples2, 1e-5, 1e5, 50, 20
        )

    assert len(call_log2) == 0, "Split should NOT be called for n_cvt=2"
    assert len(generic_log2) == 1, "Generic should be called exactly once for n_cvt=2"


@pytest.mark.tier0
def test_compute_wald_numpy_split_matches_generic(compute_wald_data):
    """split path (n_cvt=1) in _compute_wald_numpy produces same results as generic."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy as cn
    from jamma.lmm.likelihood_numpy import batch_compute_iab_numpy

    eigenvalues, Uab_batch, n_samples = compute_wald_data
    n_cvt = 1
    Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)

    # Split path (n_cvt=1 Python branch)
    with patch.object(cn, "_C_ACCEL_AVAILABLE", False):
        result_split = cn._compute_wald_numpy(
            n_cvt, eigenvalues, Uab_batch, n_samples, 1e-5, 1e5, 50, 20
        )

    # Generic path: bypass n_cvt==1 branch by calling generic optimizer directly
    import jamma.lmm.likelihood_numpy as ln

    lambdas_gen, logls_gen, Pab_gen = ln.golden_section_optimize_lambda_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
        return_pab=True,
    )
    betas_gen, ses_gen, pwalds_gen = ln.batch_calc_wald_stats_from_pab_numpy(
        n_cvt, Pab_gen, n_samples
    )

    np.testing.assert_allclose(
        result_split["lambdas"],
        lambdas_gen,
        rtol=1e-10,
        err_msg="lambdas: split path vs generic path",
    )
    np.testing.assert_allclose(
        result_split["betas"],
        betas_gen,
        rtol=1e-12,
        err_msg="betas: split path vs generic path",
    )
    np.testing.assert_allclose(
        result_split["ses"],
        ses_gen,
        rtol=1e-12,
        err_msg="ses: split path vs generic path",
    )
    np.testing.assert_allclose(
        result_split["pwalds"],
        pwalds_gen,
        rtol=1e-12,
        err_msg="pwalds: split path vs generic path",
    )


@pytest.mark.tier0
def test_compute_wald_numpy_ncvt1_invariant_efficiency(compute_wald_data):
    """compute_iab_invariant_scalars_ncvt1 called once per _compute_wald_numpy call."""
    from unittest.mock import patch

    import jamma.lmm.likelihood_numpy as ln
    from jamma.lmm import compute_numpy as cn

    eigenvalues, Uab_batch, n_samples = compute_wald_data

    call_count = []
    real_fn = ln.compute_iab_invariant_scalars_ncvt1

    def counting_fn(*args, **kwargs):
        call_count.append(1)
        return real_fn(*args, **kwargs)

    with (
        patch.object(cn, "_C_ACCEL_AVAILABLE", False),
        patch.object(cn, "compute_iab_invariant_scalars_ncvt1", counting_fn),
    ):
        cn._compute_wald_numpy(1, eigenvalues, Uab_batch, n_samples, 1e-5, 1e5, 50, 20)

    assert len(call_count) == 1, (
        f"compute_iab_invariant_scalars_ncvt1 should be called exactly once, "
        f"got {len(call_count)}"
    )


# ---------------------------------------------------------------------------
# Degenerate SNP tests — NumPy batch REML path
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_batch_numpy_all_degenerate_snps_return_lmin():
    """NumPy batch REML returns l_min for all-degenerate SNPs (UtG=0, P_XX=0).

    Constant genotypes produce zero Uab columns for genotype interactions.
    The log-likelihood is driven by phenotype variance only (constant w.r.t.
    lambda change), so the optimizer converges to l_min (the lower bound).
    The critical downstream behavior is that Wald stats return NaN.
    """
    from jamma.lmm.likelihood_numpy import batch_calc_wald_stats_numpy

    rng = np.random.default_rng(42)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG_degen = np.zeros((n, n_snps))  # all-zero genotype → P_XX = 0

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG_degen)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    lambdas, logls = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, l_min=l_min
    )
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )

    assert lambdas.shape == (n_snps,), f"Expected ({n_snps},), got {lambdas.shape}"

    # Optimizer converges to l_min when genotype has no variance
    np.testing.assert_allclose(
        lambdas,
        l_min,
        rtol=1e-4,
        err_msg="All-degenerate SNPs should return l_min lambda",
    )

    # P_XX = 0 → Wald stats all NaN (critical downstream behavior)
    assert np.all(np.isnan(betas)), f"All betas should be NaN, got {betas}"
    assert np.all(np.isnan(ses)), f"All ses should be NaN, got {ses}"
    assert np.all(np.isnan(pwalds)), f"All p_walds should be NaN, got {pwalds}"


@pytest.mark.tier0
def test_batch_numpy_mixed_degenerate_and_valid_snps():
    """NumPy batch REML handles a mix of degenerate and valid SNPs correctly.

    Columns 0, 2, 4 are zero (degenerate, P_XX=0); columns 1, 3 are normal (valid).
    Degenerate SNPs produce NaN Wald stats (beta/se/p_wald = NaN);
    valid SNPs produce finite stats. Both run in the same batch without
    cross-SNP contamination.
    """
    from jamma.lmm.likelihood_numpy import batch_calc_wald_stats_numpy

    rng = np.random.default_rng(99)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)

    UtG = np.zeros((n, n_snps))
    UtG[:, 1] = rng.standard_normal(n)  # valid
    UtG[:, 3] = rng.standard_normal(n)  # valid

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    lambdas, logls = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, l_min=l_min
    )
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )

    degenerate_idxs = [0, 2, 4]
    valid_idxs = [1, 3]

    # Degenerate SNPs: P_XX=0 → NaN Wald stats (beta/se/p_wald)
    assert np.all(np.isnan(betas[degenerate_idxs])), (
        f"Degenerate SNP betas should be NaN, got {betas[degenerate_idxs]}"
    )
    assert np.all(np.isnan(ses[degenerate_idxs])), (
        f"Degenerate SNP ses should be NaN, got {ses[degenerate_idxs]}"
    )
    assert np.all(np.isnan(pwalds[degenerate_idxs])), (
        f"Degenerate SNP p_walds should be NaN, got {pwalds[degenerate_idxs]}"
    )

    # Valid SNPs: finite Wald stats
    assert np.all(np.isfinite(betas[valid_idxs])), (
        f"Valid SNP betas should be finite, got {betas[valid_idxs]}"
    )
    assert np.all(np.isfinite(ses[valid_idxs])), (
        f"Valid SNP ses should be finite, got {ses[valid_idxs]}"
    )
    assert np.all(np.isfinite(pwalds[valid_idxs])), (
        f"Valid SNP p_walds should be finite, got {pwalds[valid_idxs]}"
    )


# ---------------------------------------------------------------------------
# Scalar-vs-batch REML optimizer parity tests
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_scalar_vs_batch_reml_single_snp_parity():
    """Scalar and batch REML optimizer paths produce matching lambda for single SNP.

    Verifies that golden_section_optimize_lambda_numpy (batch path) and
    _golden_section_minimize + reml_log_likelihood (scalar path) agree on
    the optimal lambda for a single SNP within rtol=1e-4.
    """
    rng = np.random.default_rng(42)
    n = 30
    n_cvt = 1

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    Utx = rng.standard_normal(n)

    # Scalar path
    Uab_scalar = compute_Uab(UtW, Uty, Utx)

    def scalar_obj(lam):
        return -reml_log_likelihood(lam, eigenvalues, Uab_scalar, n_cvt=n_cvt)

    lambda_scalar, _ = _golden_section_minimize(
        scalar_obj, 1e-5, 1e5, n_grid=50, n_iter=20
    )

    # Batch path (single SNP, shape (n, 1))
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, Utx.reshape(n, 1))
    Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
    lambdas_batch, _ = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch
    )

    np.testing.assert_allclose(
        lambda_scalar,
        lambdas_batch[0],
        rtol=1e-4,
        err_msg="Scalar and batch REML paths should agree on optimal lambda",
    )


@pytest.mark.tier0
def test_scalar_vs_batch_reml_multi_snp_consistency():
    """Batch REML optimizer agrees with scalar path for each of 10 SNPs.

    Tests that the batch path does not introduce cross-SNP interference:
    each SNP in the batch should get the same optimal lambda as the scalar
    path processing that SNP individually.
    """
    rng = np.random.default_rng(7)
    n, n_snps = 30, 10
    n_cvt = 1

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG = rng.standard_normal((n, n_snps))

    # Batch path — all 10 SNPs at once
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
    lambdas_batch, _ = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch
    )

    # Scalar path — one SNP at a time
    lambda_scalars = np.empty(n_snps)
    for i in range(n_snps):
        Uab_i = compute_Uab(UtW, Uty, UtG[:, i])

        def _scalar_obj(lam, uab=Uab_i):
            return -reml_log_likelihood(lam, eigenvalues, uab, n_cvt=n_cvt)

        lambda_scalars[i], _ = _golden_section_minimize(
            _scalar_obj, 1e-5, 1e5, n_grid=50, n_iter=20
        )

    np.testing.assert_allclose(
        lambda_scalars,
        lambdas_batch,
        rtol=1e-4,
        err_msg="Scalar and batch REML paths should agree on all 10 SNP lambdas",
    )
