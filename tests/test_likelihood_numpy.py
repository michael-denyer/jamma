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
from jamma.lmm.likelihood_numpy import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    batch_calc_wald_stats_numpy,
    batch_compute_iab_numpy,
    batch_compute_pab_numpy,
    batch_compute_uab_numpy,
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
