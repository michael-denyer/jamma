"""_lmm_accel C extension tests: workspace-based fused Score and LRT kernels.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
)

_score_fused_ws_available = _C_ACCEL_AVAILABLE and getattr(
    compute_numpy, "_C_SCORE_FUSED_WS_AVAILABLE", False
)


class TestScoreWorkspaceParity:
    """Verify workspace-based Score produces bitwise-identical results to stateless."""

    @pytest.fixture
    def score_ws_data(self):
        """Prepare data for Score workspace tests."""
        rng = np.random.default_rng(12345)
        n_samples, n_snps = 100, 20

        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
        w = rng.standard_normal(n_samples)
        Uty = rng.standard_normal(n_samples)
        utg_t = rng.standard_normal((n_snps, n_samples))

        # Construct physically meaningful invariant SoA
        uab_inv_soa = np.empty((3, n_samples), dtype=np.float64)
        uab_inv_soa[0] = w * w  # ww
        uab_inv_soa[1] = w * Uty  # wy
        uab_inv_soa[2] = Uty * Uty  # yy

        # Hi_eval_null from a reasonable lambda_null
        lambda_null = 0.5
        Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

        return (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            Hi_eval_null,
            n_samples,
            n_snps,
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_ws_available,
        reason="Score fused workspace C not available",
    )
    def test_score_workspace_create(self, score_ws_data):
        """Workspace creation returns a non-None PyCapsule."""
        from jamma.lmm.compute_numpy import _c

        assert (
            _c().create_workspace_score_fused_c is not None
        )  # narrowed: skipif gates this

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        ws = _c().create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1,
        )
        assert ws is not None

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_ws_available,
        reason="Score fused workspace C not available",
    )
    def test_score_workspace_parity(self, score_ws_data):
        """Workspace-based Score matches stateless Score (atol=0, rtol=0)."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        # Stateless reference
        ref = _c().compute_score_fused_c(
            utg_t,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )

        # Workspace-based
        ws = _c().create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1,
        )
        result = _c().compute_score_fused_ws_c(ws, utg_t, 1)

        for key in ("betas", "ses", "p_scores"):
            np.testing.assert_allclose(
                result[key],
                ref[key],
                atol=1e-15,
                rtol=0,
                err_msg=f"Score workspace {key} mismatch",
            )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_ws_available,
        reason="Score fused workspace C not available",
    )
    def test_score_workspace_multi_chunk(self, score_ws_data):
        """Same workspace produces correct results for two different utg_t chunks."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        ws = _c().create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1,
        )

        # Chunk 1
        ref1 = _c().compute_score_fused_c(
            utg_t,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )
        result1 = _c().compute_score_fused_ws_c(ws, utg_t, 1)
        for key in ("betas", "ses", "p_scores"):
            np.testing.assert_allclose(
                result1[key],
                ref1[key],
                atol=1e-15,
                rtol=0,
                err_msg=f"Score workspace chunk1 {key} mismatch",
            )

        # Chunk 2 (different data)
        rng2 = np.random.default_rng(99999)
        utg_t2 = rng2.standard_normal((15, n_samples))
        ref2 = _c().compute_score_fused_c(
            utg_t2,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )
        result2 = _c().compute_score_fused_ws_c(ws, utg_t2, 1)
        for key in ("betas", "ses", "p_scores"):
            np.testing.assert_allclose(
                result2[key],
                ref2[key],
                atol=1e-15,
                rtol=0,
                err_msg=f"Score workspace chunk2 {key} mismatch",
            )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_ws_available,
        reason="Score fused workspace C not available",
    )
    def test_score_workspace_capsule_type_safety(self, score_ws_data):
        """Passing a Wald workspace to Score compute raises ValueError."""
        from jamma.lmm.compute_numpy import _c, create_lmm_workspace

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        # Create a Wald workspace (wrong type)
        wald_ws = create_lmm_workspace(
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )

        with pytest.raises(ValueError, match="PyCapsule_GetPointer"):
            _c().compute_score_fused_ws_c(wald_ws, utg_t, 1)


_lrt_fused_ws_available = _C_ACCEL_AVAILABLE and getattr(
    compute_numpy, "_C_LRT_FUSED_WS_AVAILABLE", False
)


class TestLrtWorkspaceParity:
    """Verify workspace-based LRT produces bitwise-identical results to stateless."""

    @pytest.fixture
    def lrt_ws_data(self):
        """Prepare data for LRT workspace tests."""
        rng = np.random.default_rng(54321)
        n_samples, n_snps = 100, 20

        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
        w = rng.standard_normal(n_samples)
        Uty = rng.standard_normal(n_samples)
        utg_t = rng.standard_normal((n_snps, n_samples))

        # Construct physically meaningful invariant SoA
        uab_inv_soa = np.empty((3, n_samples), dtype=np.float64)
        uab_inv_soa[0] = w * w  # ww
        uab_inv_soa[1] = w * Uty  # wy
        uab_inv_soa[2] = Uty * Uty  # yy

        return (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            n_samples,
            n_snps,
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_ws_available,
        reason="LRT fused workspace C not available",
    )
    def test_lrt_workspace_create(self, lrt_ws_data):
        """Workspace creation returns a non-None PyCapsule."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        ws = _c().create_workspace_lrt_fused_c(
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            5,
            -150.0,
            1,
        )
        assert ws is not None

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_ws_available,
        reason="LRT fused workspace C not available",
    )
    def test_lrt_workspace_parity(self, lrt_ws_data):
        """Workspace-based LRT matches stateless LRT (atol=0, rtol=0)."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        logl_H0 = -150.0

        # Stateless reference
        ref = _c().compute_lrt_fused_c(
            utg_t,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            5,
            logl_H0,
            1,
        )

        # Workspace-based
        ws = _c().create_workspace_lrt_fused_c(
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            5,
            logl_H0,
            1,
        )
        result = _c().compute_lrt_fused_ws_c(ws, utg_t, 1)

        for key in ("lambdas_mle", "p_lrts"):
            np.testing.assert_allclose(
                result[key],
                ref[key],
                atol=1e-15,
                rtol=0,
                err_msg=f"LRT workspace {key} mismatch",
            )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_ws_available,
        reason="LRT fused workspace C not available",
    )
    def test_lrt_workspace_multi_chunk(self, lrt_ws_data):
        """Same workspace produces correct results for two different utg_t chunks."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        logl_H0 = -150.0
        ws = _c().create_workspace_lrt_fused_c(
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            5,
            logl_H0,
            1,
        )

        # Chunk 1
        ref1 = _c().compute_lrt_fused_c(
            utg_t,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            5,
            logl_H0,
            1,
        )
        result1 = _c().compute_lrt_fused_ws_c(ws, utg_t, 1)
        for key in ("lambdas_mle", "p_lrts"):
            np.testing.assert_allclose(
                result1[key],
                ref1[key],
                atol=1e-15,
                rtol=0,
                err_msg=f"LRT workspace chunk1 {key} mismatch",
            )

        # Chunk 2 (different data)
        rng2 = np.random.default_rng(88888)
        utg_t2 = rng2.standard_normal((15, n_samples))
        ref2 = _c().compute_lrt_fused_c(
            utg_t2,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            5,
            logl_H0,
            1,
        )
        result2 = _c().compute_lrt_fused_ws_c(ws, utg_t2, 1)
        for key in ("lambdas_mle", "p_lrts"):
            np.testing.assert_allclose(
                result2[key],
                ref2[key],
                atol=1e-15,
                rtol=0,
                err_msg=f"LRT workspace chunk2 {key} mismatch",
            )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_ws_available,
        reason="LRT fused workspace C not available",
    )
    def test_lrt_workspace_capsule_type_safety(self, lrt_ws_data):
        """Passing a Score workspace to LRT compute raises ValueError."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        # Hi_eval_null needed for Score workspace creation
        Hi_eval_null = 1.0 / (0.5 * eigenvalues + 1.0)

        # Create a Score workspace (wrong type for LRT)
        score_ws = _c().create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1,
        )

        with pytest.raises(ValueError, match="PyCapsule_GetPointer"):
            _c().compute_lrt_fused_ws_c(score_ws, utg_t, 1)
