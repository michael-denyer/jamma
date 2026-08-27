"""_lmm_accel C extension tests: workspace-based fused Score and LRT kernels.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.

These kernels are checked against the NumPy implementations of the same
statistics. The reference used to be the stateless C twin of each kernel, which
let the assertion be bitwise, but no dispatch path ever reached those twins and
they have been removed. NumPy is an independent implementation, so the
assertions carry a tolerance; the values come from the deviation measured on
this module's own fixtures, which peaks at 1.3e-13, with headroom for a
different compiler and CPU in CI.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.likelihood_numpy import golden_section_optimize_lambda_mle_numpy
from jamma.lmm.stats import _batch_lrt_pvalues_numpy, batch_calc_score_stats_numpy
from jamma.lmm.uab import batch_compute_uab_numpy
from tests.lmm_accel._helpers import _null_model_ncvt1

_C_RTOL = 1e-11
_C_ATOL = 1e-14

_score_fused_ws_available = compute_numpy._accel is not None


def _uab_from_fused_inputs(w, Uty, utg_t):
    """Rebuild the full Uab batch the NumPy kernels take from the fused SoA inputs."""
    return batch_compute_uab_numpy(1, w[:, None], Uty, utg_t.T)


def _numpy_score_reference(w, Uty, utg_t, Hi_eval_null, n_samples):
    """Score betas, SEs and p-values for the fused kernel's inputs, via NumPy."""
    betas, ses, p_scores = batch_calc_score_stats_numpy(
        1, Hi_eval_null, _uab_from_fused_inputs(w, Uty, utg_t), n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _numpy_lrt_reference(w, Uty, utg_t, eigenvalues, logl_H0, n_refine):
    """MLE lambdas and LRT p-values for the fused kernel's inputs, via NumPy."""
    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle_numpy(
        1,
        eigenvalues,
        _uab_from_fused_inputs(w, Uty, utg_t),
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=n_refine,
    )
    return {
        "lambdas_mle": lambdas_mle,
        "p_lrts": _batch_lrt_pvalues_numpy(logls_mle, logl_H0),
    }


def _assert_matches_numpy(result, reference, label):
    for key, ref in reference.items():
        np.testing.assert_allclose(
            result[key],
            ref,
            rtol=_C_RTOL,
            atol=_C_ATOL,
            err_msg=f"{label} {key} does not match the NumPy reference",
        )


class TestScoreWorkspaceParity:
    """Verify workspace-based Score matches the NumPy Score statistics."""

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
        """Workspace-based Score matches the NumPy Score statistics."""
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
        result = _c().compute_score_fused_ws_c(ws, utg_t, 1)

        _assert_matches_numpy(
            result,
            _numpy_score_reference(w, Uty, utg_t, Hi_eval_null, n_samples),
            "Score workspace",
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

        result1 = _c().compute_score_fused_ws_c(ws, utg_t, 1)
        _assert_matches_numpy(
            result1,
            _numpy_score_reference(w, Uty, utg_t, Hi_eval_null, n_samples),
            "Score workspace chunk1",
        )

        rng2 = np.random.default_rng(99999)
        utg_t2 = rng2.standard_normal((15, n_samples))
        result2 = _c().compute_score_fused_ws_c(ws, utg_t2, 1)
        _assert_matches_numpy(
            result2,
            _numpy_score_reference(w, Uty, utg_t2, Hi_eval_null, n_samples),
            "Score workspace chunk2",
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_ws_available,
        reason="Score fused workspace C not available",
    )
    def test_score_workspace_capsule_type_safety(self, score_ws_data):
        """Passing an LRT workspace to Score compute raises ValueError."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        lrt_ws = _c().create_workspace_lrt_fused_c(
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

        with pytest.raises(ValueError, match="PyCapsule_GetPointer"):
            _c().compute_score_fused_ws_c(lrt_ws, utg_t, 1)

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_ws_available,
        reason="Score fused workspace C not available",
    )
    def test_score_workspace_degenerate_snps(self, score_ws_data):
        """A constant genotype (P_xx <= 0) yields NaN beta/se/p_score for that SNP."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        utg_degen = utg_t.copy()
        utg_degen[0, :] = 0.0

        ws = _c().create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1,
        )
        result = _c().compute_score_fused_ws_c(ws, utg_degen, 1)

        assert np.isnan(result["betas"][0]), "degenerate SNP should have NaN beta"
        assert np.isnan(result["ses"][0]), "degenerate SNP should have NaN se"
        assert np.isnan(result["p_scores"][0]), "degenerate SNP should have NaN p_score"
        assert np.all(np.isfinite(result["betas"][1:])), (
            "non-degenerate SNPs should be finite"
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_ws_available,
        reason="Score fused workspace C not available",
    )
    def test_score_workspace_multithreaded(self, score_ws_data):
        """Score is bitwise deterministic across thread counts."""
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
        single = _c().compute_score_fused_ws_c(ws, utg_t, 1)
        multi = _c().compute_score_fused_ws_c(ws, utg_t, 2)

        for key in ("betas", "ses", "p_scores"):
            np.testing.assert_array_equal(
                single[key],
                multi[key],
                err_msg=f"Score {key}: 2-thread vs 1-thread mismatch",
            )


_lrt_fused_ws_available = compute_numpy._accel is not None


class TestLrtWorkspaceParity:
    """Verify workspace-based LRT matches the NumPy MLE lambda and LRT p-value."""

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
        """Workspace-based LRT matches the NumPy MLE lambdas and LRT p-values."""
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
        result = _c().compute_lrt_fused_ws_c(ws, utg_t, 1)

        _assert_matches_numpy(
            result,
            _numpy_lrt_reference(w, Uty, utg_t, eigenvalues, logl_H0, 5),
            "LRT workspace",
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

        result1 = _c().compute_lrt_fused_ws_c(ws, utg_t, 1)
        _assert_matches_numpy(
            result1,
            _numpy_lrt_reference(w, Uty, utg_t, eigenvalues, logl_H0, 5),
            "LRT workspace chunk1",
        )

        rng2 = np.random.default_rng(88888)
        utg_t2 = rng2.standard_normal((15, n_samples))
        result2 = _c().compute_lrt_fused_ws_c(ws, utg_t2, 1)
        _assert_matches_numpy(
            result2,
            _numpy_lrt_reference(w, Uty, utg_t2, eigenvalues, logl_H0, 5),
            "LRT workspace chunk2",
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_ws_available,
        reason="LRT fused workspace C not available",
    )
    def test_lrt_workspace_degenerate_snps(self, lrt_ws_data):
        """A constant genotype carries no signal, so its p_lrt sits at 1.

        lambda_mle is left unasserted: with the likelihood flat, its argmin is
        not determined.
        """
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        utg_degen = utg_t.copy()
        utg_degen[0, :] = 0.0

        # The real null log-likelihood, not the module's stand-in constant: a
        # p_lrt value is only interpretable against the model it is testing.
        _, logl_H0 = _null_model_ncvt1(eigenvalues, w, Uty)

        ws = _c().create_workspace_lrt_fused_c(
            w, Uty, eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, logl_H0, 1
        )
        result = _c().compute_lrt_fused_ws_c(ws, utg_degen, 1)

        assert result["p_lrts"][0] >= 0.99, (
            f"degenerate SNP p_lrt={result['p_lrts'][0]}, expected near 1"
        )
        assert np.all(np.isfinite(result["p_lrts"][1:])), (
            "non-degenerate p_lrts should be finite"
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_ws_available,
        reason="LRT fused workspace C not available",
    )
    def test_lrt_workspace_multithreaded(self, lrt_ws_data):
        """LRT is bitwise deterministic across thread counts."""
        from jamma.lmm.compute_numpy import _c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        ws = _c().create_workspace_lrt_fused_c(
            w, Uty, eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 5, -150.0, 1
        )
        single = _c().compute_lrt_fused_ws_c(ws, utg_t, 1)
        multi = _c().compute_lrt_fused_ws_c(ws, utg_t, 2)

        for key in ("lambdas_mle", "p_lrts"):
            np.testing.assert_array_equal(
                single[key],
                multi[key],
                err_msg=f"LRT {key}: 2-thread vs 1-thread mismatch",
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
