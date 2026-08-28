"""Tests for prepare_common.py: numerical correctness and edge cases.

Verifies that:
- _compute_null_model_common produces numerically correct results
- _build_covariate_matrix behaves correctly for the intercept-only case
- _eigendecompose_or_reuse passes through pre-computed values unchanged
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    _eigendecompose_or_reuse,
)
from jamma.lmm.schema import LmmConfig


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
    from jamma.lmm import prepare_common

    rng = np.random.default_rng(99)
    n_samples = 50
    eigenvalues_np = np.sort(np.abs(rng.standard_normal(n_samples)) + 0.1)
    eigenvalues_np[0] = -10.0

    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

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
    from jamma.lmm import prepare_common

    rng = np.random.default_rng(100)
    n_samples = 50
    eigenvalues_np = np.sort(np.abs(rng.standard_normal(n_samples)) + 0.1)
    eigenvalues_np[0] = np.nan
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

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
    monkeypatch.setattr(
        compute_numpy, "_accel", None
    )  # allow-patch: dropping the extension forces the NumPy path

    with pytest.raises(ValueError, match="non-positive"):
        compute_numpy._compute_score_numpy(
            n_cvt, eigenvalues, hi_bad, Uab_batch, n_samples
        )


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
    monkeypatch.setattr(
        compute_numpy, "_accel", None
    )  # allow-patch: dropping the extension forces the NumPy path

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
            config=LmmConfig(check_memory=False, show_progress=False),
        )
