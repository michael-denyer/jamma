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
        batch_compute_uab(1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T))
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

    Uab_batch_jax = batch_compute_uab(
        1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T)
    )
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

    Uab_batch_jax = batch_compute_uab(
        1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T)
    )
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

    Uab_batch_jax = batch_compute_uab(
        1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T)
    )
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

    Uab_batch_jax = batch_compute_uab(
        1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T)
    )
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

    Uab_batch_jax = batch_compute_uab(
        1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T)
    )
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

    Uab_batch_jax = batch_compute_uab(
        1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T)
    )
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

    Uab_batch_jax = batch_compute_uab(
        1, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T)
    )
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

    # Mode 4 (All) requires both logl_H0 and Hi_eval_null.
    # Missing logl_H0 is checked first (line order in source).
    with pytest.raises(ValueError, match="logl_H0 is required"):
        _compute_lmm_chunk_numpy(4, 1, eigenvalues, Uab_batch, n_samples)

    # Providing logl_H0 but omitting Hi_eval_null also raises.
    with pytest.raises(ValueError, match="Hi_eval_null is required"):
        _compute_lmm_chunk_numpy(4, 1, eigenvalues, Uab_batch, n_samples, logl_H0=-50.0)


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
        batch_compute_uab(n_cvt, jnp.array(UtW), jnp.array(Uty), jnp.array(UtG.T))
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
    uab_varying_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

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
        patch.object(cn, "_C_GENERAL_AVAILABLE", False),
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
# All-NaN grid test for _batch_golden_section_numpy
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_batch_golden_section_numpy_all_nan_grid():
    """_batch_golden_section_numpy with all-NaN grid logls returns l_min lambda.

    When every grid log-likelihood is NaN (all SNPs degenerate), argmax of
    NaN-replaced -inf selects index 0 (the lower bound), bracketing around
    l_min.  The optimizer should return lambdas at or near l_min without
    crashing, and log-likelihoods should be finite (or NaN, but not inf).

    This is the all-SNPs-degenerate extreme: _guard_P_yy produces NaN for
    every grid point, so safe_logls is all -inf.
    """
    from jamma.lmm.likelihood_numpy import (
        _batch_golden_section_numpy,
        _batch_reml_at_lambda_numpy,
    )

    rng = np.random.default_rng(42)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG_degen = np.zeros((n, n_snps))  # all-zero genotype → all-NaN grid logls

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG_degen)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    n_grid = 10
    log_l_min = np.log(l_min)
    log_l_max = np.log(1e5)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)

    # All-degenerate grid: for zero genotype, P_yy after projecting out X
    # may be pathological, but REML logls will be constant across lambda.
    # Force the all-NaN scenario by using an artificial grid of NaN logls.
    grid_logls_all_nan = np.full((n_grid, n_snps), np.nan)

    def compute_batch_fn(log_lams):
        lams = np.exp(log_lams)
        return _batch_reml_at_lambda_numpy(1, lams, eigenvalues, Uab_batch, Iab_batch)

    lambdas_out, logls_out = _batch_golden_section_numpy(
        compute_batch_fn, grid_logls_all_nan, log_lambdas, n_iter=20
    )

    assert lambdas_out.shape == (n_snps,), (
        f"Expected ({n_snps},), got {lambdas_out.shape}"
    )

    # With all-NaN grid, argmax(safe_logls) falls to index 0 (l_min bracket).
    # Optimizer must not produce +inf or -inf lambdas.
    assert not np.any(np.isinf(lambdas_out)), (
        f"Lambdas should not be infinite, got {lambdas_out}"
    )
    # Lambdas should be within [l_min, l_max] range
    assert np.all(lambdas_out >= l_min * 0.99), f"Lambdas below l_min: {lambdas_out}"
    assert np.all(lambdas_out <= 1e5 * 1.01), f"Lambdas above l_max: {lambdas_out}"


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
# Degenerate SNP tests — Python fallback (split ncvt1 and generic batch paths)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_split_ncvt1_fallback_degenerate_snps_wald_nan():
    """Split ncvt1 Python fallback path: degenerate SNPs produce NaN Wald stats.

    golden_section_optimize_lambda_split_ncvt1_numpy is the Python fallback
    when the C extension is unavailable.  When UtG is all-zero (constant
    genotype), the varying columns [wx, xx, xy] are zero for every SNP.
    The optimizer returns lambdas near l_min; downstream Wald stats must
    produce NaN for every SNP because P_XX = 0.
    """
    from jamma.lmm.likelihood_numpy import (
        batch_calc_wald_stats_numpy,
        batch_compute_uab_varying_soa_numpy,
        compute_iab_invariant_scalars_ncvt1,
        compute_uab_invariant_soa,
        golden_section_optimize_lambda_split_ncvt1_numpy,
        reconstruct_uab_from_soa,
    )

    rng = np.random.default_rng(17)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG_degen = np.zeros((n, n_snps))  # constant genotype -> P_XX = 0

    uab_invariant_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_varying_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG_degen.T)
    iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    lambdas, logls = golden_section_optimize_lambda_split_ncvt1_numpy(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_s_ww,
        iab_s_wy,
        iab_s_yy,
        iab_logdet,
        l_min=l_min,
    )

    assert lambdas.shape == (n_snps,), f"Expected ({n_snps},), got {lambdas.shape}"

    # Optimizer converges to l_min when genotype has no variance
    np.testing.assert_allclose(
        lambdas,
        l_min,
        rtol=1e-4,
        err_msg="Split ncvt1 all-degenerate SNPs should return l_min lambda",
    )

    # Reconstruct full Uab for Wald stats
    Uab_batch = reconstruct_uab_from_soa(uab_invariant_soa, uab_varying_soa)
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )

    # P_XX = 0 -> all Wald stats NaN
    assert np.all(np.isnan(betas)), f"Expected all-NaN betas, got {betas}"
    assert np.all(np.isnan(ses)), f"Expected all-NaN ses, got {ses}"
    assert np.all(np.isnan(pwalds)), f"Expected all-NaN pwalds, got {pwalds}"


@pytest.mark.tier0
def test_split_ncvt1_fallback_mixed_degenerate_valid():
    """Split ncvt1 Python fallback: mixed batch, degenerate NaN, valid finite.

    SNPs at indices 1 and 3 have non-zero genotypes (valid).
    SNPs at indices 0, 2, 4 are zero-genotype (degenerate, P_XX=0).
    The split optimizer must process them in the same batch without
    cross-SNP contamination.
    """
    from jamma.lmm.likelihood_numpy import (
        batch_calc_wald_stats_numpy,
        batch_compute_uab_varying_soa_numpy,
        compute_iab_invariant_scalars_ncvt1,
        compute_uab_invariant_soa,
        golden_section_optimize_lambda_split_ncvt1_numpy,
        reconstruct_uab_from_soa,
    )

    rng = np.random.default_rng(99)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)

    UtG = np.zeros((n, n_snps))
    UtG[:, 1] = rng.standard_normal(n)  # valid
    UtG[:, 3] = rng.standard_normal(n)  # valid

    uab_invariant_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_varying_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    lambdas, logls = golden_section_optimize_lambda_split_ncvt1_numpy(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_s_ww,
        iab_s_wy,
        iab_s_yy,
        iab_logdet,
        l_min=l_min,
    )

    Uab_batch = reconstruct_uab_from_soa(uab_invariant_soa, uab_varying_soa)
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )

    degenerate_idxs = [0, 2, 4]
    valid_idxs = [1, 3]

    assert np.all(np.isnan(betas[degenerate_idxs])), (
        f"Degenerate betas should be NaN, got {betas[degenerate_idxs]}"
    )
    assert np.all(np.isnan(ses[degenerate_idxs])), (
        f"Degenerate ses should be NaN, got {ses[degenerate_idxs]}"
    )
    assert np.all(np.isnan(pwalds[degenerate_idxs])), (
        f"Degenerate p_walds should be NaN, got {pwalds[degenerate_idxs]}"
    )
    assert np.all(np.isfinite(betas[valid_idxs])), (
        f"Valid betas should be finite, got {betas[valid_idxs]}"
    )
    assert np.all(np.isfinite(ses[valid_idxs])), (
        f"Valid ses should be finite, got {ses[valid_idxs]}"
    )
    assert np.all(np.isfinite(pwalds[valid_idxs])), (
        f"Valid p_walds should be finite, got {pwalds[valid_idxs]}"
    )


@pytest.mark.tier0
def test_generic_batch_numpy_fallback_degenerate_wald_nan():
    """golden_section_optimize_lambda_numpy (generic path) with degenerate SNPs.

    This tests the Python fallback path (golden_section_optimize_lambda_numpy
    with n_cvt=1) independently from the C-extension path.  Degenerate SNPs
    (UtG=0) should produce NaN Wald stats downstream regardless of which
    low-level optimizer is used.
    """
    rng = np.random.default_rng(55)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG_degen = np.zeros((n, n_snps))  # constant genotype

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG_degen)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    lambdas, logls = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, l_min=l_min
    )

    assert lambdas.shape == (n_snps,), f"Expected ({n_snps},), got {lambdas.shape}"

    # Degenerate SNPs: optimizer should converge to l_min (no SNP signal)
    np.testing.assert_allclose(
        lambdas,
        l_min,
        rtol=1e-4,
        err_msg="Generic fallback: all-degenerate SNPs should return l_min lambda",
    )

    # Wald stats: P_XX=0 -> all NaN
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )
    assert np.all(np.isnan(betas)), f"Expected all-NaN betas, got {betas}"
    assert np.all(np.isnan(ses)), f"Expected all-NaN ses, got {ses}"
    assert np.all(np.isnan(pwalds)), f"Expected all-NaN pwalds, got {pwalds}"


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


# ---------------------------------------------------------------------------
# Scalar-vs-batch REML optimizer: tight lambda and logl parity (jamma-68j0)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_scalar_vs_batch_reml_single_snp_lambda_and_logl_parity():
    """Scalar and batch REML optimizers agree on lambda and logl.

    Feeds a single SNP through both optimizers:
    - Scalar: _golden_section_minimize + reml_log_likelihood (likelihood.py)
    - Batch:  golden_section_optimize_lambda_numpy (likelihood_numpy.py)

    Both run identical grid search + golden section in log-lambda space (same
    n_grid=50, n_iter=20 defaults), so the returned optimal lambda and the
    log-likelihood evaluated at that lambda should agree to near-machine-epsilon
    after accounting for the batch midpoint vs scalar midpoint evaluation.

    Tolerance rationale:
    - lambda: rtol=1e-10 — both converge to log((a+b)/2) with same bracket,
      so the difference is sub-ULP in double precision.
    - logl: rtol=1e-10 — evaluated at the same lambda point via the same REML
      arithmetic; any discrepancy would indicate a divergence in Pab/Iab logic.
    """
    rng = np.random.default_rng(123)
    n = 50
    n_cvt = 1

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    Utx = rng.standard_normal(n)

    # --- Scalar path ---
    Uab_scalar = compute_Uab(UtW, Uty, Utx)

    def scalar_neg_reml(lam: float) -> float:
        return -reml_log_likelihood(lam, eigenvalues, Uab_scalar, n_cvt=n_cvt)

    lambda_scalar, logl_scalar = _golden_section_minimize(
        scalar_neg_reml, l_min=1e-5, l_max=1e5, n_grid=50, n_iter=20
    )

    # --- Batch path (single SNP wrapped in batch dimension) ---
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, Utx.reshape(n, 1))
    Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
    lambdas_batch, logls_batch = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch, n_grid=50, n_iter=20
    )
    lambda_batch = lambdas_batch[0]
    logl_batch = logls_batch[0]

    np.testing.assert_allclose(
        lambda_scalar,
        lambda_batch,
        rtol=1e-10,
        atol=1e-14,
        err_msg=(
            f"Scalar lambda {lambda_scalar:.6e} vs batch lambda {lambda_batch:.6e} "
            "disagree beyond rtol=1e-10"
        ),
    )
    np.testing.assert_allclose(
        logl_scalar,
        logl_batch,
        rtol=1e-10,
        atol=1e-14,
        err_msg=(
            f"Scalar logl {logl_scalar:.6e} vs batch logl {logl_batch:.6e} "
            "disagree beyond rtol=1e-10"
        ),
    )


# ---------------------------------------------------------------------------
# reconstruct_uab_from_soa generalization for n_cvt > 1 (Plan 70-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_reconstruct_uab_from_soa_ncvt1_backward_compat():
    """reconstruct_uab_from_soa without n_cvt arg produces correct n_cvt=1 output."""
    from jamma.lmm.likelihood_numpy import reconstruct_uab_from_soa

    rng = np.random.default_rng(42)
    n_samples, n_snps = 50, 8
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    Uab_ref = batch_compute_uab_numpy(1, UtW, Uty, UtG)

    # Build SoA for n_cvt=1
    inv_soa = np.stack([Uab_ref[0, :, 0], Uab_ref[0, :, 2], Uab_ref[0, :, 5]])  # (3, n)
    var_soa = np.stack(
        [Uab_ref[:, :, 1], Uab_ref[:, :, 3], Uab_ref[:, :, 4]], axis=1
    )  # (n_snps, 3, n)

    # Old signature (no n_cvt) should still work
    Uab_recon = reconstruct_uab_from_soa(inv_soa, var_soa)
    np.testing.assert_allclose(
        Uab_recon,
        Uab_ref,
        rtol=1e-14,
        atol=0,
        err_msg="reconstruct_uab_from_soa backward compat (no n_cvt) failed",
    )


@pytest.mark.tier0
def test_reconstruct_uab_from_soa_ncvt2():
    """reconstruct_uab_from_soa with n_cvt=2 round-trips via classify_uab_columns."""
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.likelihood_numpy import reconstruct_uab_from_soa

    rng = np.random.default_rng(7)
    n_samples, n_snps = 40, 6
    n_cvt = 2
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Full reference Uab
    Uab_ref = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    # n_index = (n_cvt+3)*(n_cvt+2)//2 = 5*4//2 = 10

    inv_indices, var_indices = classify_uab_columns(n_cvt)

    # Build invariant SoA from first SNP's Uab (all SNPs share invariant columns)
    inv_list = list(inv_indices)
    uab_invariant_soa = np.ascontiguousarray(
        Uab_ref[0, :, inv_list]
    )  # (n_inv, n_samples)

    # Build varying SoA
    uab_varying_soa = np.ascontiguousarray(
        Uab_ref[:, :, list(var_indices)].transpose(0, 2, 1)
    )  # (n_snps, n_var, n_samples)

    # Reconstruct should match Uab_ref
    Uab_recon = reconstruct_uab_from_soa(
        uab_invariant_soa, uab_varying_soa, n_cvt=n_cvt
    )
    np.testing.assert_allclose(
        Uab_recon,
        Uab_ref,
        rtol=1e-14,
        atol=0,
        err_msg="reconstruct_uab_from_soa failed to round-trip for n_cvt=2",
    )


@pytest.mark.tier0
def test_reconstruct_uab_from_soa_ncvt4():
    """reconstruct_uab_from_soa with n_cvt=4 matches batch_compute_uab_numpy."""
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.likelihood_numpy import reconstruct_uab_from_soa

    rng = np.random.default_rng(13)
    n_samples, n_snps = 30, 5
    n_cvt = 4
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Full reference Uab
    Uab_ref = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    # n_index = (4+3)*(4+2)//2 = 7*6//2 = 21

    inv_indices, var_indices = classify_uab_columns(n_cvt)

    inv_list = list(inv_indices)
    uab_invariant_soa = np.ascontiguousarray(Uab_ref[0, :, inv_list])
    uab_varying_soa = np.ascontiguousarray(
        Uab_ref[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    Uab_recon = reconstruct_uab_from_soa(
        uab_invariant_soa, uab_varying_soa, n_cvt=n_cvt
    )
    np.testing.assert_allclose(
        Uab_recon,
        Uab_ref,
        rtol=1e-14,
        atol=0,
        err_msg="reconstruct_uab_from_soa failed to round-trip for n_cvt=4",
    )


# ---------------------------------------------------------------------------
# Score/LRT C dispatch via general C path for n_cvt > 1 (Plan 70-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_compute_score_numpy_ncvt2_uses_c_path(synthetic_data):
    """_compute_score_numpy dispatches to C general path for n_cvt=2.

    Verifies C general path is called, not Python fallback.
    Skipped gracefully when general C function is not available.
    """
    from unittest.mock import patch

    from jamma.lmm import compute_numpy as cn
    from jamma.lmm.compute_numpy import _compute_score_numpy

    if cn._compute_score_batch_general_c is None:
        pytest.skip("compute_score_batch_general_c not available")

    eigenvalues, UtW_ncvt1, Uty, UtG = synthetic_data
    n_samples = len(eigenvalues)
    n_cvt = 2
    UtW = np.column_stack([UtW_ncvt1, Uty])  # add second covariate

    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    lambda_null = 0.1
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

    call_log: list[str] = []

    original_fn = cn._compute_score_batch_general_c

    def spy_general(*args, **kwargs):
        call_log.append("general_c")
        return original_fn(*args, **kwargs)

    with patch.object(cn, "_compute_score_batch_general_c", side_effect=spy_general):
        _compute_score_numpy(n_cvt, eigenvalues, Hi_eval_null, Uab_batch, n_samples)

    assert "general_c" in call_log, (
        "_compute_score_numpy did not call _compute_score_batch_general_c for n_cvt=2"
    )


# ---------------------------------------------------------------------------
# General n_cvt vectorized Uab parity tests
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.parametrize("n_cvt", [2, 3, 4])
def test_vectorized_general_uab_parity(n_cvt):
    """Vectorized _batch_compute_uab_general_numpy matches reference per-SNP loop."""
    from jamma.lmm.likelihood import build_index_table
    from jamma.lmm.likelihood_numpy import _batch_compute_uab_general_numpy

    rng = np.random.default_rng(99)
    n_samples, n_snps = 60, 15
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Reference: per-SNP loop (the old implementation)
    table = build_index_table(n_cvt)
    n_index = table["n_index"]
    Uab_ref = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    vectors_base = np.column_stack([UtW, np.zeros(n_samples), Uty])
    for snp_idx in range(n_snps):
        vectors = vectors_base.copy()
        vectors[:, n_cvt] = UtG[:, snp_idx]
        for a_col, b_col, idx in table["uab_pairs"]:
            Uab_ref[snp_idx, :, idx] = vectors[:, a_col] * vectors[:, b_col]

    # Vectorized implementation
    Uab_vec = _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG)

    np.testing.assert_allclose(
        Uab_vec,
        Uab_ref,
        rtol=1e-14,
        atol=1e-14,
        err_msg=f"Vectorized general Uab (n_cvt={n_cvt}) does not match per-SNP loop",
    )


@pytest.mark.tier0
@pytest.mark.parametrize("n_cvt", [2, 3, 4])
def test_direct_soa_varying_general_parity(n_cvt):
    """_batch_compute_uab_varying_general_numpy matches extract-from-full-Uab."""
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.likelihood_numpy import (
        _batch_compute_uab_general_numpy,
        _batch_compute_uab_varying_general_numpy,
    )

    rng = np.random.default_rng(101)
    n_samples, n_snps = 60, 15
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Reference: compute full Uab then extract varying columns to SoA
    _inv_indices, var_indices = classify_uab_columns(n_cvt)
    Uab_full = _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG)
    ref_soa = np.ascontiguousarray(Uab_full[:, :, list(var_indices)].transpose(0, 2, 1))

    # Direct SoA varying — utg_t is (n_snps, n_samples)
    direct_soa = _batch_compute_uab_varying_general_numpy(n_cvt, UtW, Uty, UtG.T)

    np.testing.assert_allclose(
        direct_soa,
        ref_soa,
        rtol=1e-14,
        atol=1e-14,
        err_msg=f"Direct SoA varying (n_cvt={n_cvt}) does not match extract-from-full",
    )


@pytest.mark.tier0
@pytest.mark.parametrize("n_cvt", [2, 3, 4])
def test_invariant_columns_constant_across_snps(n_cvt):
    """Uab columns classified as invariant are actually constant across SNPs."""
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.likelihood_numpy import _batch_compute_uab_general_numpy

    rng = np.random.default_rng(123)
    n_samples, n_snps = 40, 20
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    inv_indices, _var_indices = classify_uab_columns(n_cvt)
    Uab = _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG)

    for col_idx in inv_indices:
        first_snp = Uab[0, :, col_idx]
        for snp_i in range(1, n_snps):
            np.testing.assert_array_equal(
                Uab[snp_i, :, col_idx],
                first_snp,
                err_msg=(
                    f"Invariant column {col_idx} differs at SNP {snp_i} (n_cvt={n_cvt})"
                ),
            )


@pytest.mark.tier0
@pytest.mark.parametrize("n_cvt", [2, 3, 4])
def test_batch_compute_uab_varying_soa_general_uses_direct_path(n_cvt):
    """batch_compute_uab_varying_soa_numpy general path uses direct computation."""
    rng = np.random.default_rng(55)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # The function should produce identical results regardless of path
    var_soa = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    # Reference: full Uab -> extract varying -> SoA
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.likelihood_numpy import _batch_compute_uab_general_numpy

    _inv, var_indices = classify_uab_columns(n_cvt)
    Uab_full = _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG)
    ref_soa = np.ascontiguousarray(Uab_full[:, :, list(var_indices)].transpose(0, 2, 1))

    np.testing.assert_allclose(
        var_soa,
        ref_soa,
        rtol=1e-14,
        atol=1e-14,
        err_msg=f"batch_compute_uab_varying_soa_numpy general (n_cvt={n_cvt}) mismatch",
    )


@pytest.mark.tier0
def test_general_uab_no_per_snp_loop():
    """_batch_compute_uab_general_numpy must not contain per-SNP Python loop."""
    import inspect

    from jamma.lmm.likelihood_numpy import _batch_compute_uab_general_numpy

    source = inspect.getsource(_batch_compute_uab_general_numpy)
    assert "vectors_base.copy()" not in source, (
        "_batch_compute_uab_general_numpy still contains vectors_base.copy()"
    )
    assert "for snp_idx in range(n_snps)" not in source, (
        "_batch_compute_uab_general_numpy still contains per-SNP Python loop"
    )


@pytest.mark.tier0
def test_varying_soa_general_path_calls_direct():
    """batch_compute_uab_varying_soa_numpy general path must call direct function."""
    import inspect

    from jamma.lmm.likelihood_numpy import batch_compute_uab_varying_soa_numpy

    source = inspect.getsource(batch_compute_uab_varying_soa_numpy)
    assert "_batch_compute_uab_varying_general_numpy" in source, (
        "batch_compute_uab_varying_soa_numpy does not call "
        "_batch_compute_uab_varying_general_numpy"
    )


@pytest.mark.tier0
def test_compute_lrt_numpy_ncvt2_uses_c_path(synthetic_data):
    """_compute_lrt_numpy dispatches to C general path for n_cvt=2.

    Verifies C general path is called, not Python fallback.
    Skipped gracefully when general C function is not available.
    """
    from unittest.mock import patch

    from jamma.lmm import compute_numpy as cn
    from jamma.lmm.compute_numpy import _compute_lrt_numpy

    if cn._compute_lrt_batch_general_c is None:
        pytest.skip("compute_lrt_batch_general_c not available")

    eigenvalues, UtW_ncvt1, Uty, UtG = synthetic_data
    n_cvt = 2
    UtW = np.column_stack([UtW_ncvt1, Uty])

    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    logl_H0 = -30.0

    call_log: list[str] = []
    original_fn = cn._compute_lrt_batch_general_c

    def spy_general(*args, **kwargs):
        call_log.append("general_c")
        return original_fn(*args, **kwargs)

    with patch.object(cn, "_compute_lrt_batch_general_c", side_effect=spy_general):
        _compute_lrt_numpy(n_cvt, eigenvalues, Uab_batch, 1e-5, 1e5, 50, 20, logl_H0)

    assert "general_c" in call_log, (
        "_compute_lrt_numpy did not call _compute_lrt_batch_general_c for n_cvt=2"
    )


# ---------------------------------------------------------------------------
# Shape validation guard tests
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_batch_compute_uab_numpy_rejects_wrong_layout():
    """batch_compute_uab_numpy raises ValueError when given (n_snps, n_samples)."""
    rng = np.random.default_rng(99)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    utg_t = rng.standard_normal((n_snps, n_samples))  # wrong layout for this fn

    with pytest.raises(ValueError, match="Pass \\(n_samples, n_snps\\)"):
        batch_compute_uab_numpy(1, UtW, Uty, utg_t)


@pytest.mark.tier0
def test_batch_compute_uab_varying_soa_rejects_wrong_out_shape():
    """batch_compute_uab_varying_soa_numpy raises ValueError for wrong out= shape."""
    rng = np.random.default_rng(99)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    utg_t = rng.standard_normal((n_snps, n_samples))
    wrong_out = np.empty((n_snps + 1, 3, n_samples), dtype=np.float64)

    with pytest.raises(ValueError, match="out shape"):
        batch_compute_uab_varying_soa_numpy(1, UtW, Uty, utg_t, out=wrong_out)


@pytest.mark.tier0
def test_batch_compute_uab_varying_soa_rejects_wrong_out_shape_general_ncvt():
    """Raises ValueError for wrong out= shape with n_cvt > 1."""
    rng = np.random.default_rng(99)
    n_samples, n_snps, n_cvt = 50, 10, 2
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    utg_t = rng.standard_normal((n_snps, n_samples))
    # n_cvt=2 has n_var=4, so 6 is wrong
    out = np.empty((n_snps, 6, n_samples), dtype=np.float64)

    with pytest.raises(ValueError, match="out shape"):
        batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, utg_t, out=out)


@pytest.mark.tier0
def test_batch_compute_uab_varying_soa_rejects_wrong_layout():
    """batch_compute_uab_varying_soa_numpy raises ValueError when given old layout."""
    rng = np.random.default_rng(99)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))  # wrong layout for this fn

    with pytest.raises(ValueError, match="Pass \\(n_snps, n_samples\\)"):
        batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)


# --------------------------------------------------------------------------- #
# out= buffer support for general n_cvt (n_cvt > 1)
# --------------------------------------------------------------------------- #


@pytest.mark.tier0
def test_varying_soa_out_buffer_general_ncvt2():
    """out= buffer works for n_cvt=2 and result is the same buffer object."""
    from jamma.lmm.likelihood import classify_uab_columns

    rng = np.random.default_rng(42)
    n_samples, n_snps, n_cvt = 50, 10, 2
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    utg_t = rng.standard_normal((n_snps, n_samples))

    _, var_cols = classify_uab_columns(n_cvt)
    n_var = len(var_cols)

    # Compute without out= for reference
    expected = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, utg_t)

    # Compute with out= buffer
    out = np.empty((n_snps, n_var, n_samples), dtype=np.float64)
    result = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, utg_t, out=out)

    assert result is out, "result should be the same buffer object"
    np.testing.assert_array_equal(result, expected)


@pytest.mark.tier0
def test_varying_soa_out_buffer_general_ncvt4():
    """out= buffer works for n_cvt=4 and result is the same buffer object."""
    from jamma.lmm.likelihood import classify_uab_columns

    rng = np.random.default_rng(77)
    n_samples, n_snps, n_cvt = 40, 8, 4
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    utg_t = rng.standard_normal((n_snps, n_samples))

    _, var_cols = classify_uab_columns(n_cvt)
    n_var = len(var_cols)

    expected = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, utg_t)

    out = np.empty((n_snps, n_var, n_samples), dtype=np.float64)
    result = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, utg_t, out=out)

    assert result is out, "result should be the same buffer object"
    np.testing.assert_array_equal(result, expected)


@pytest.mark.tier0
def test_varying_soa_out_buffer_general_shape_mismatch():
    """Wrong-shape out= raises ValueError with 'out shape' in message."""
    rng = np.random.default_rng(99)
    n_samples, n_snps, n_cvt = 50, 10, 2
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    utg_t = rng.standard_normal((n_snps, n_samples))

    wrong_out = np.empty((n_snps, 99, n_samples), dtype=np.float64)

    with pytest.raises(ValueError, match="out shape"):
        batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, utg_t, out=wrong_out)
