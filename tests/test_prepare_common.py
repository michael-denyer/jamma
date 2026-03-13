"""Tests for prepare_common.py: import isolation and numerical parity.

Verifies that:
- prepare_common.py contains no JAX imports (static AST check)
- _compute_null_model_common produces numerically identical results to
  the JAX _compute_null_model wrapper
- _build_covariate_matrix behaves correctly for the intercept-only case
- _eigendecompose_or_reuse passes through pre-computed values unchanged
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    _eigendecompose_or_reuse,
)


@pytest.mark.tier0
def test_prepare_common_no_jax_imports():
    """Verify prepare_common.py does not import JAX at module level or anywhere."""
    source = (
        Path(__file__).parent.parent / "src" / "jamma" / "lmm" / "prepare_common.py"
    ).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("jax"), (
                    f"Direct JAX import in prepare_common.py: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert not node.module.startswith("jax"), (
                    f"From-JAX import in prepare_common.py: {node.module}"
                )


@pytest.mark.tier0
def test_compute_null_model_common_matches_prepare():
    """_compute_null_model_common must produce numerically identical results to JAX
    wrapper _compute_null_model."""
    jax = pytest.importorskip("jax")
    from jamma.lmm.prepare import _compute_null_model

    rng = np.random.default_rng(42)
    n_samples = 100

    # Eigenvalues: sorted positive values (realistic kinship spectrum)
    eigenvalues_np = np.sort(np.abs(rng.standard_normal(n_samples)) + 0.1)

    # Covariates: intercept-only (ones column)
    W = np.ones((n_samples, 1))
    n_cvt = 1

    # Rotated covariates and phenotype: use a random orthogonal rotation
    Q, _ = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))
    UtW = Q.T @ W
    Uty = Q.T @ rng.standard_normal(n_samples)

    # Test mode 4 (All): exercises the full null model path including Hi_eval_null
    rep_placement = jax.devices("cpu")[0]

    logl_H0_common, lam_common, Hi_eval_np = _compute_null_model_common(
        lmm_mode=4,
        eigenvalues_np=eigenvalues_np,
        UtW=UtW,
        Uty=Uty,
        n_cvt=n_cvt,
        show_progress=False,
    )

    logl_H0_jax, lam_jax, Hi_eval_jax = _compute_null_model(
        lmm_mode=4,
        eigenvalues_np=eigenvalues_np,
        UtW=UtW,
        Uty=Uty,
        n_cvt=n_cvt,
        rep_placement=rep_placement,
        show_progress=False,
    )

    # Scalar values must be identical (same computation path, no rounding differences)
    assert logl_H0_common == pytest.approx(logl_H0_jax, rel=0, abs=1e-14), (
        f"logl_H0 mismatch: common={logl_H0_common}, jax={logl_H0_jax}"
    )
    assert lam_common == pytest.approx(lam_jax, rel=0, abs=1e-14), (
        f"lambda_null_mle mismatch: common={lam_common}, jax={lam_jax}"
    )

    # Hi_eval_null arrays must match after converting JAX to numpy
    assert Hi_eval_np is not None, "Hi_eval_null_np should not be None for mode=4"
    assert Hi_eval_jax is not None, "Hi_eval_null_jax should not be None for mode=4"
    np.testing.assert_allclose(
        Hi_eval_np,
        np.asarray(Hi_eval_jax),
        rtol=0,
        atol=1e-14,
        err_msg="Hi_eval_null mismatch between prepare_common and prepare JAX wrapper",
    )


@pytest.mark.tier0
def test_compute_null_model_common_wald_returns_nones():
    """Wald mode (lmm_mode=1) should return (None, None, None) from common path."""
    rng = np.random.default_rng(0)
    n_samples = 50
    eigenvalues_np = np.abs(rng.standard_normal(n_samples)) + 0.1
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

    logl_H0, lam, Hi_eval = _compute_null_model_common(
        lmm_mode=1,
        eigenvalues_np=eigenvalues_np,
        UtW=UtW,
        Uty=Uty,
        n_cvt=1,
        show_progress=False,
    )
    assert logl_H0 is None
    assert lam is None
    assert Hi_eval is None


@pytest.mark.tier0
def test_compute_null_model_common_lrt_returns_no_hi_eval():
    """LRT mode (lmm_mode=2) returns logl_H0 and lambda but no Hi_eval."""
    rng = np.random.default_rng(1)
    n_samples = 50
    eigenvalues_np = np.sort(np.abs(rng.standard_normal(n_samples)) + 0.1)
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

    logl_H0, lam, Hi_eval = _compute_null_model_common(
        lmm_mode=2,
        eigenvalues_np=eigenvalues_np,
        UtW=UtW,
        Uty=Uty,
        n_cvt=1,
        show_progress=False,
    )
    assert logl_H0 is not None
    assert lam is not None
    assert Hi_eval is None


@pytest.mark.tier0
def test_build_covariate_matrix_from_common():
    """_build_covariate_matrix(None, 100) returns intercept-only W of shape (100, 1)."""
    W, n_cvt = _build_covariate_matrix(None, 100)

    assert W.shape == (100, 1), f"Expected (100, 1), got {W.shape}"
    assert n_cvt == 1
    np.testing.assert_array_equal(W, np.ones((100, 1)))


@pytest.mark.tier0
def test_eigendecompose_or_reuse_passthrough():
    """Pre-computed eigenvalues/eigenvectors are returned unchanged."""
    eigenvalues = np.array([1.0, 2.0, 3.0])
    eigenvectors = np.eye(3)

    result_evals, result_evecs = _eigendecompose_or_reuse(
        kinship=None,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        show_progress=False,
        label="test",
    )

    # Should return the exact same objects (no copy, no computation)
    assert result_evals is eigenvalues
    assert result_evecs is eigenvectors


@pytest.mark.tier0
def test_eigendecompose_or_reuse_raises_when_all_none():
    """Must raise ValueError when kinship=None and no pre-computed eigen provided."""
    with pytest.raises(ValueError, match="Must provide either"):
        _eigendecompose_or_reuse(
            kinship=None,
            eigenvalues=None,
            eigenvectors=None,
            show_progress=False,
            label="test",
        )


@pytest.mark.tier0
def test_build_covariate_matrix_with_user_covariates():
    """_build_covariate_matrix passes through user covariates as-is."""
    rng = np.random.default_rng(99)
    # Build covariates with intercept in first column (as expected by GEMMA convention)
    user_cov = np.column_stack([np.ones(50), rng.standard_normal((50, 2))])
    W, n_cvt = _build_covariate_matrix(user_cov, 50)

    assert W.shape == (50, 3), f"Expected (50, 3), got {W.shape}"
    assert n_cvt == 3
    np.testing.assert_array_equal(W, user_cov.astype(np.float64))


@pytest.mark.tier0
def test_build_covariate_matrix_over_parameterized():
    """Over-parameterized model (n_samples <= n_cvt + 1) must raise ValueError."""
    # 3 samples with 3 covariates → df = 3 - 3 - 1 = -1
    W = np.ones((3, 3))
    with pytest.raises(ValueError, match="Over-parameterized"):
        _build_covariate_matrix(W, 3)


@pytest.mark.tier0
def test_build_covariate_matrix_rank_deficient():
    """Rank-deficient covariate matrix must raise ValueError."""
    # Two identical columns → rank 1, n_cvt 2
    W = np.ones((50, 2))
    with pytest.raises(ValueError, match="rank-deficient"):
        _build_covariate_matrix(W, 50)


@pytest.mark.tier0
def test_compute_null_model_common_score_returns_hi_eval():
    """T6: Score mode (lmm_mode=3) returns logl_H0, lambda, and Hi_eval."""
    rng = np.random.default_rng(7)
    n_samples = 50
    eigenvalues_np = np.sort(np.abs(rng.standard_normal(n_samples)) + 0.1)
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

    logl_H0, lam, Hi_eval = _compute_null_model_common(
        lmm_mode=3,
        eigenvalues_np=eigenvalues_np,
        UtW=UtW,
        Uty=Uty,
        n_cvt=1,
        show_progress=False,
    )
    assert logl_H0 is not None, "logl_H0 should not be None for Score mode"
    assert lam is not None, "lambda should not be None for Score mode"
    assert Hi_eval is not None, "Hi_eval should not be None for Score mode"
    assert Hi_eval.shape == (n_samples,), (
        f"Expected ({n_samples},), got {Hi_eval.shape}"
    )
    assert np.all(Hi_eval > 0), "Hi_eval should be all positive"


@pytest.mark.tier0
def test_compute_null_model_common_rejects_negative_eigenvalues(monkeypatch):
    """_compute_null_model_common raises ValueError on non-positive Hi_eval_null.

    Injects eigenvalue=-10.0 and mocks optimizer to return lambda=1000,
    so lambda*eval+1 = 1000*(-10)+1 < 0.
    """
    import unittest.mock as mock

    from jamma.lmm import prepare_common

    rng = np.random.default_rng(99)
    n_samples = 50
    # One eigenvalue is strongly negative so lambda*eval+1 < 0
    eigenvalues_np = np.sort(np.abs(rng.standard_normal(n_samples)) + 0.1)
    eigenvalues_np[0] = -10.0  # negative eigenvalue (FP noise from degenerate kinship)

    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

    # Mock the MLE optimizer to return a large lambda (1000) so lambda*eval+1 < 0
    # for the negative eigenvalue: 1000*(-10)+1 = -9999
    monkeypatch.setattr(
        prepare_common,
        "compute_null_model_mle",
        mock.Mock(return_value=(1000.0, -50.0)),
    )

    with pytest.raises(ValueError, match="non-positive"):
        prepare_common._compute_null_model_common(
            lmm_mode=3,
            eigenvalues_np=eigenvalues_np,
            UtW=UtW,
            Uty=Uty,
            n_cvt=1,
            show_progress=False,
        )


@pytest.mark.tier0
def test_compute_null_model_common_rejects_nan_hi_eval_null(monkeypatch):
    """_compute_null_model_common raises ValueError on non-finite Hi_eval_null.

    Mocks optimizer to return NaN lambda, producing NaN Hi_eval_null.
    """
    import unittest.mock as mock

    from jamma.lmm import prepare_common

    rng = np.random.default_rng(100)
    n_samples = 50
    eigenvalues_np = np.sort(np.abs(rng.standard_normal(n_samples)) + 0.1)
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

    monkeypatch.setattr(
        prepare_common,
        "compute_null_model_mle",
        mock.Mock(return_value=(float("nan"), -50.0)),
    )

    with pytest.raises(ValueError, match="non-finite"):
        prepare_common._compute_null_model_common(
            lmm_mode=3,
            eigenvalues_np=eigenvalues_np,
            UtW=UtW,
            Uty=Uty,
            n_cvt=1,
            show_progress=False,
        )


@pytest.mark.tier0
def test_compute_null_model_common_accepts_near_zero_eigenvalues():
    """_compute_null_model_common does NOT raise for near-zero non-negative eigenvalues.

    With eigenvalue=1e-15 and any realistic lambda, lambda*eval+1 is positive.
    """
    rng = np.random.default_rng(42)
    n_samples = 50
    eigenvalues_np = np.sort(np.abs(rng.standard_normal(n_samples)) + 0.1)
    eigenvalues_np[0] = 1e-15  # near-zero but non-negative

    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

    # Should not raise — 1e-15 is positive so Hi_eval_null[0] > 0
    logl_H0, lam, Hi_eval = _compute_null_model_common(
        lmm_mode=3,
        eigenvalues_np=eigenvalues_np,
        UtW=UtW,
        Uty=Uty,
        n_cvt=1,
        show_progress=False,
    )
    assert Hi_eval is not None
    assert np.all(Hi_eval > 0), (
        "All Hi_eval_null should be positive for non-negative eigenvalues"
    )


@pytest.mark.tier0
def test_compute_score_numpy_rejects_negative_hi_eval_null(monkeypatch):
    """Python fallback Score path rejects non-positive Hi_eval_null."""
    import jamma.lmm.compute_numpy as compute_numpy

    rng = np.random.default_rng(101)
    n_samples, n_snps, n_cvt = 50, 5, 1

    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    Hi_eval_null = 1.0 / (0.5 * eigenvalues + 1.0)
    n_uab = (n_cvt + 2) * (n_cvt + 3) // 2
    Uab_batch = np.ones((n_snps, n_samples, n_uab), dtype=np.float64)

    hi_bad = Hi_eval_null.copy()
    hi_bad[2] = -0.5

    # Force Python fallback by hiding C extension
    monkeypatch.setattr(compute_numpy, "_compute_score_batch_c", None)
    monkeypatch.setattr(compute_numpy, "_compute_score_batch_general_c", None)

    with pytest.raises(ValueError, match="non-positive"):
        compute_numpy._compute_score_numpy(
            n_cvt, eigenvalues, hi_bad, Uab_batch, n_samples
        )


@pytest.mark.tier0
@pytest.mark.requires_jax
def test_compute_score_jax_rejects_negative_hi_eval_null():
    """JAX Score path rejects non-positive Hi_eval_null."""
    import jax.numpy as jnp

    from jamma.lmm.compute import _compute_score

    rng = np.random.default_rng(103)
    n_samples, n_snps, n_cvt = 50, 5, 1

    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    Hi_eval_null = 1.0 / (0.5 * eigenvalues + 1.0)
    n_uab = (n_cvt + 2) * (n_cvt + 3) // 2
    Uab_batch = jnp.ones((n_snps, n_samples, n_uab), dtype=jnp.float64)

    hi_bad = Hi_eval_null.copy()
    hi_bad[2] = -0.5

    with pytest.raises(ValueError, match="non-positive"):
        _compute_score(n_cvt, jnp.array(hi_bad), Uab_batch, n_samples)


@pytest.mark.tier0
def test_compute_score_numpy_rejects_nan_hi_eval_null(monkeypatch):
    """Python fallback Score path rejects NaN Hi_eval_null."""
    import jamma.lmm.compute_numpy as compute_numpy

    rng = np.random.default_rng(102)
    n_samples, n_snps, n_cvt = 50, 5, 1

    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    Hi_eval_null = 1.0 / (0.5 * eigenvalues + 1.0)
    n_uab = (n_cvt + 2) * (n_cvt + 3) // 2
    Uab_batch = np.ones((n_snps, n_samples, n_uab), dtype=np.float64)

    hi_bad = Hi_eval_null.copy()
    hi_bad[0] = np.nan

    # Force Python fallback by hiding C extension
    monkeypatch.setattr(compute_numpy, "_compute_score_batch_c", None)
    monkeypatch.setattr(compute_numpy, "_compute_score_batch_general_c", None)

    with pytest.raises(ValueError, match="non-finite"):
        compute_numpy._compute_score_numpy(
            n_cvt, eigenvalues, hi_bad, Uab_batch, n_samples
        )


@pytest.mark.tier0
def test_eigenvector_shape_mismatch_raises():
    """T7: Mismatched eigenvector dimensions raise ValueError in the runner."""
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(8)
    n_samples = 20
    n_snps = 10

    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    phenotypes = rng.standard_normal(n_samples)
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 100, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]

    eigenvalues = np.ones(n_samples)
    eigenvectors_wrong = np.eye(n_samples + 5)

    with pytest.raises(ValueError, match="eigenvectors shape"):
        run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors_wrong,
            check_memory=False,
            show_progress=False,
        )
