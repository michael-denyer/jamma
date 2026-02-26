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
    source = Path("src/jamma/lmm/prepare_common.py").read_text()
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
