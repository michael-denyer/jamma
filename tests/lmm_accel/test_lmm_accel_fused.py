"""_lmm_accel C extension tests: fused Uab kernels, n_cvt=1 and general.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import compute_wald_split_c_ws, create_lmm_workspace
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)
from jamma.lmm.schema import LmmConfig


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
class TestHiEvalNullPositivity:
    """C extension rejects non-positive hi_eval_null values at all three sites."""

    def test_mode4_workspace_rejects_zero_hi_eval_null(self, score_lrt_data):
        """create_workspace_mode4_split_c raises ValueError on zero hi_eval_null."""
        from jamma.lmm._lmm_accel import create_workspace_mode4_split_c

        eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data
        uab_inv_soa = np.stack(
            [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
        )

        hi_bad = Hi_eval_null.copy()
        hi_bad[5] = 0.0  # inject zero

        with pytest.raises(ValueError, match="not positive"):
            create_workspace_mode4_split_c(
                eigenvalues,
                uab_inv_soa,
                n_samples,
                1e-5,
                1e5,
                50,
                20,
                1,
                hi_bad,
                logl_H0,
            )

    def test_mode4_workspace_rejects_negative_hi_eval_null(self, score_lrt_data):
        """create_workspace_mode4_split_c raises ValueError on negative hi_eval_null."""
        from jamma.lmm._lmm_accel import create_workspace_mode4_split_c

        eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data
        uab_inv_soa = np.stack(
            [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
        )

        hi_bad = Hi_eval_null.copy()
        hi_bad[10] = -0.5  # inject negative value

        with pytest.raises(ValueError, match="not positive"):
            create_workspace_mode4_split_c(
                eigenvalues,
                uab_inv_soa,
                n_samples,
                1e-5,
                1e5,
                50,
                20,
                1,
                hi_bad,
                logl_H0,
            )

    def test_score_batch_c_rejects_negative_hi_eval_null(self, score_lrt_data):
        """compute_score_batch_c raises ValueError when hi_eval_null is negative."""
        from jamma.lmm._lmm_accel import compute_score_batch_c

        eigenvalues, Uab_batch, n_samples, Hi_eval_null, _ = score_lrt_data

        hi_bad = Hi_eval_null.copy()
        hi_bad[3] = -1.0  # inject negative value

        with pytest.raises(ValueError, match="not positive"):
            compute_score_batch_c(
                eigenvalues,
                Uab_batch,
                hi_bad,
                n_samples,
                1,
            )

    def test_score_batch_c_rejects_zero_hi_eval_null(self, score_lrt_data):
        """compute_score_batch_c raises ValueError when hi_eval_null is zero."""
        from jamma.lmm._lmm_accel import compute_score_batch_c

        eigenvalues, Uab_batch, n_samples, Hi_eval_null, _ = score_lrt_data

        hi_bad = Hi_eval_null.copy()
        hi_bad[3] = 0.0  # inject zero

        with pytest.raises(ValueError, match="not positive"):
            compute_score_batch_c(
                eigenvalues,
                Uab_batch,
                hi_bad,
                n_samples,
                1,
            )

    def test_score_batch_general_c_rejects_negative_hi_eval_null(self):
        """compute_score_batch_general_c raises ValueError on negative hi_eval_null."""
        try:
            from jamma.lmm._lmm_accel import compute_score_batch_general_c
        except ImportError:
            pytest.skip("compute_score_batch_general_c not compiled yet")

        from jamma.lmm.likelihood import build_pab_table_for_c

        rng = np.random.default_rng(77)
        n_samples, n_snps, n_cvt = 80, 10, 2

        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
        lambda_null = 0.5
        Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

        # Build Uab_batch for n_cvt=2
        n_uab = (n_cvt + 2) * (n_cvt + 3) // 2
        Uab_batch = np.ones((n_snps, n_samples, n_uab), dtype=np.float64)
        pab_table_dict = build_pab_table_for_c(n_cvt)

        hi_bad = Hi_eval_null.copy()
        hi_bad[0] = -2.0  # inject negative value

        with pytest.raises(ValueError, match="not positive"):
            compute_score_batch_general_c(
                eigenvalues,
                Uab_batch,
                hi_bad,
                n_samples,
                n_cvt,
                pab_table_dict,
                1,
            )

    def test_score_batch_general_c_rejects_zero_hi_eval_null(self):
        """compute_score_batch_general_c raises ValueError on zero hi_eval_null."""
        try:
            from jamma.lmm._lmm_accel import compute_score_batch_general_c
        except ImportError:
            pytest.skip("compute_score_batch_general_c not compiled yet")

        from jamma.lmm.likelihood import build_pab_table_for_c

        rng = np.random.default_rng(78)
        n_samples, n_snps, n_cvt = 80, 10, 2

        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
        lambda_null = 0.5
        Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

        n_uab = (n_cvt + 2) * (n_cvt + 3) // 2
        Uab_batch = np.ones((n_snps, n_samples, n_uab), dtype=np.float64)
        pab_table_dict = build_pab_table_for_c(n_cvt)

        hi_bad = Hi_eval_null.copy()
        hi_bad[0] = 0.0  # inject zero

        with pytest.raises(ValueError, match="not positive"):
            compute_score_batch_general_c(
                eigenvalues,
                Uab_batch,
                hi_bad,
                n_samples,
                n_cvt,
                pab_table_dict,
                1,
            )


_fused_c_available = compute_numpy._accel is not None


@pytest.mark.tier0
@pytest.mark.skipif(not _fused_c_available, reason="Fused C extension not available")
class TestFusedParity:
    """Verify fused Uab path produces identical results to SoA path."""

    @pytest.fixture
    def fused_data(self, split_wald_data):
        """Prepare data for both SoA and fused paths."""
        eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
        w = UtW[:, 0].copy()
        utg_t = np.ascontiguousarray(UtG.T)
        uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
        uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
        return eigenvalues, w, Uty, utg_t, uab_inv_soa, uab_var_soa, n_samples

    def test_fused_workspace_creation(self, fused_data):
        """create_workspace_fused_c returns a PyCapsule."""
        from jamma.lmm.compute_numpy import create_lmm_workspace_fused

        eigenvalues, w, Uty, utg_t, uab_inv_soa, _, n_samples = fused_data
        ws = create_lmm_workspace_fused(
            eigenvalues,
            uab_inv_soa,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )
        assert ws is not None

    def test_wald_parity(self, fused_data):
        """Fused Wald produces bitwise-identical results to SoA Wald."""
        from jamma.lmm.compute_numpy import (
            compute_wald_fused_c_ws,
            create_lmm_workspace_fused,
        )

        eigenvalues, w, Uty, utg_t, uab_inv_soa, uab_var_soa, n_samples = fused_data

        # SoA path
        ws_soa = create_lmm_workspace(
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )
        soa_result = compute_wald_split_c_ws(ws_soa, uab_var_soa, 1)

        # Fused path
        ws_fused = create_lmm_workspace_fused(
            eigenvalues,
            uab_inv_soa,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )
        fused_result = compute_wald_fused_c_ws(ws_fused, utg_t, 1)

        for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
            np.testing.assert_array_equal(
                soa_result[key],
                fused_result[key],
                err_msg=f"Wald {key}: fused vs SoA mismatch (should be bitwise)",
            )

    def test_wald_parity_multithreaded(self, fused_data):
        """Fused Wald with multiple threads matches SoA path."""
        from jamma.lmm.compute_numpy import (
            compute_wald_fused_c_ws,
            create_lmm_workspace_fused,
        )

        eigenvalues, w, Uty, utg_t, uab_inv_soa, uab_var_soa, n_samples = fused_data

        ws_soa = create_lmm_workspace(
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )
        soa_result = compute_wald_split_c_ws(ws_soa, uab_var_soa, 1)

        ws_fused = create_lmm_workspace_fused(
            eigenvalues,
            uab_inv_soa,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )
        fused_result = compute_wald_fused_c_ws(ws_fused, utg_t, 4)

        for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
            np.testing.assert_array_equal(
                soa_result[key],
                fused_result[key],
                err_msg=f"Wald {key}: fused(4t) vs SoA mismatch",
            )

    def test_mode4_fused_workspace_creation(self, fused_data, score_lrt_data):
        """create_workspace_mode4_fused_c returns a PyCapsule."""
        from jamma.lmm.compute_numpy import create_lmm_workspace_mode4_fused

        eigenvalues, w, Uty, _, uab_inv_soa, _, n_samples = fused_data
        _, _, _, Hi_eval_null, logl_H0 = score_lrt_data

        ws = create_lmm_workspace_mode4_fused(
            eigenvalues,
            uab_inv_soa,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
            hi_eval_null=Hi_eval_null,
            logl_H0=logl_H0,
        )
        assert ws is not None

    def test_mode4_parity(self, fused_data, score_lrt_data):
        """Fused mode-4 produces bitwise-identical results to SoA mode-4."""
        from jamma.lmm import compute_numpy
        from jamma.lmm.compute_numpy import (
            compute_mode4_fused_c_ws,
            compute_mode4_split_c_ws,
            create_lmm_workspace_mode4,
            create_lmm_workspace_mode4_fused,
        )

        if compute_numpy._accel is None:
            pytest.skip("Mode-4 split C extension not available")

        eigenvalues, w, Uty, utg_t, uab_inv_soa, uab_var_soa, n_samples = fused_data
        _, _, _, Hi_eval_null, logl_H0 = score_lrt_data

        # SoA path
        ws_soa = create_lmm_workspace_mode4(
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
            Hi_eval_null,
            logl_H0,
        )
        soa_result = compute_mode4_split_c_ws(ws_soa, uab_var_soa, 1)

        # Fused path
        ws_fused = create_lmm_workspace_mode4_fused(
            eigenvalues,
            uab_inv_soa,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
            hi_eval_null=Hi_eval_null,
            logl_H0=logl_H0,
        )
        fused_result = compute_mode4_fused_c_ws(ws_fused, utg_t, 1)

        for key in (
            "lambdas",
            "logls",
            "betas",
            "ses",
            "pwalds",
            "p_scores",
            "lambdas_mle",
            "p_lrts",
        ):
            np.testing.assert_array_equal(
                soa_result[key],
                fused_result[key],
                err_msg=f"Mode-4 {key}: fused vs SoA mismatch (should be bitwise)",
            )

    def test_fused_wrong_utg_t_shape(self, fused_data):
        """Fused compute raises ValueError for wrong UtG_T shape."""
        from jamma.lmm.compute_numpy import (
            compute_wald_fused_c_ws,
            create_lmm_workspace_fused,
        )

        eigenvalues, w, Uty, utg_t, uab_inv_soa, _, n_samples = fused_data

        ws = create_lmm_workspace_fused(
            eigenvalues,
            uab_inv_soa,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )

        # 3D instead of 2D
        bad_utg = utg_t.reshape(utg_t.shape[0], 1, utg_t.shape[1])
        with pytest.raises(ValueError, match="utg_t"):
            compute_wald_fused_c_ws(ws, bad_utg, 1)

    def test_fused_workspace_refcount(self, fused_data):
        """w and Uty arrays not garbage collected while workspace alive."""
        import gc
        import sys

        from jamma.lmm.compute_numpy import create_lmm_workspace_fused

        eigenvalues, w, Uty, _, uab_inv_soa, _, n_samples = fused_data

        # Make copies that we can track
        w_tracked = w.copy()
        Uty_tracked = Uty.copy()
        initial_w_ref = sys.getrefcount(w_tracked)
        initial_Uty_ref = sys.getrefcount(Uty_tracked)

        ws = create_lmm_workspace_fused(
            eigenvalues,
            uab_inv_soa,
            w_tracked,
            Uty_tracked,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )

        # Workspace should hold a reference to w and Uty
        assert sys.getrefcount(w_tracked) > initial_w_ref
        assert sys.getrefcount(Uty_tracked) > initial_Uty_ref

        del ws
        gc.collect()

        # After workspace destruction, refcounts should be back to initial
        assert sys.getrefcount(w_tracked) == initial_w_ref
        assert sys.getrefcount(Uty_tracked) == initial_Uty_ref

    def test_fused_degenerate_snps(self, fused_data):
        """Fused Wald handles degenerate (constant) SNPs: NaN beta/se/pwald."""
        from jamma.lmm.compute_numpy import (
            compute_wald_fused_c_ws,
            create_lmm_workspace_fused,
        )

        eigenvalues, w, Uty, utg_t, uab_inv_soa, _, n_samples = fused_data

        # Make first SNP degenerate: constant genotype -> all zeros after rotation
        utg_t_degen = utg_t.copy()
        utg_t_degen[0, :] = 0.0

        ws = create_lmm_workspace_fused(
            eigenvalues,
            uab_inv_soa,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )
        cr = compute_wald_fused_c_ws(ws, utg_t_degen, 1)

        # Degenerate SNP: should produce NaN
        assert np.isnan(cr["betas"][0]), "degenerate SNP should have NaN beta"
        assert np.isnan(cr["ses"][0]), "degenerate SNP should have NaN se"
        assert np.isnan(cr["pwalds"][0]), "degenerate SNP should have NaN p_wald"

        # Non-degenerate SNPs should still be valid (compare against reference)
        ws_ref = create_lmm_workspace_fused(
            eigenvalues,
            uab_inv_soa,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )
        cr_ref = compute_wald_fused_c_ws(ws_ref, utg_t, 1)
        finite_mask = np.isfinite(cr_ref["betas"][1:])
        assert np.all(np.isfinite(cr["betas"][1:][finite_mask])), (
            "non-degenerate betas should be finite"
        )

    def test_fused_rejects_split_workspace(self, fused_data):
        """compute_wald_fused_c_ws rejects a non-fused (split) workspace."""
        from jamma.lmm.compute_numpy import compute_wald_fused_c_ws

        eigenvalues, _, _, utg_t, uab_inv_soa, _, n_samples = fused_data

        # Create a split (non-fused) workspace — w/Uty will be NULL
        ws_split = create_lmm_workspace(
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )
        with pytest.raises(ValueError, match=r"[Ff]used"):
            compute_wald_fused_c_ws(ws_split, utg_t, 1)


def _prepare_fused_general_data(data: dict) -> dict:
    """Prepare invariant SoA, varying SoA, UtG_T, and pab_c for fused general tests.

    Args:
        data: Dict from _build_synthetic_covariate_data.

    Returns:
        Dict with uab_inv_soa, uab_var_soa, utg_t, pab_c, and original data keys.
    """
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    n_cvt = data["n_cvt"]
    Uab_batch = data["Uab_batch"]
    UtG = data["UtG"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )
    utg_t = np.ascontiguousarray(UtG.T)
    pab_c = build_pab_table_for_c(n_cvt)

    return {
        **data,
        "uab_inv_soa": uab_inv_soa,
        "uab_var_soa": uab_var_soa,
        "utg_t": utg_t,
        "pab_c": pab_c,
    }


def _run_fused_general_wald_vs_nonfused(data: dict) -> None:
    """Compare fused general Wald against non-fused general Wald (bitwise).

    Args:
        data: Dict from _prepare_fused_general_data.
    """
    from jamma.lmm.compute_numpy import (
        compute_wald_fused_general_c_ws,
        compute_wald_general_c_ws,
        create_lmm_workspace_fused_general,
        create_lmm_workspace_general,
    )

    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    uab_inv_soa = data["uab_inv_soa"]
    uab_var_soa = data["uab_var_soa"]
    utg_t = data["utg_t"]
    pab_c = data["pab_c"]
    UtW = data["UtW"]
    Uty = data["Uty"]

    # Non-fused general path
    ws_nonfused = create_lmm_workspace_general(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        n_cvt,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    result_nonfused = compute_wald_general_c_ws(ws_nonfused, uab_var_soa, 1)

    # Fused general path
    pab_kwargs = {
        k: pab_c[k]
        for k in [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
            "idx_xx",
            "idx_xy",
            "idx_yy",
            "var_a_cols",
            "var_b_cols",
        ]
    }
    ws_fused = create_lmm_workspace_fused_general(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        n_cvt=n_cvt,
        **pab_kwargs,
    )
    result_fused = compute_wald_fused_general_c_ws(ws_fused, utg_t, 1)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_array_equal(
            result_nonfused[key],
            result_fused[key],
            err_msg=(
                f"Wald {key}: fused general vs non-fused general mismatch "
                f"(should be bitwise identical, n_cvt={n_cvt})"
            ),
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Fused general C not available",
)
def test_fused_general_ncvt2_wald(synthetic_covariate_data_ncvt2):
    """FGEN-04: Fused general Wald bitwise matches non-fused general for n_cvt=2."""
    _run_fused_general_wald_vs_nonfused(
        _prepare_fused_general_data(synthetic_covariate_data_ncvt2)
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Fused general C not available",
)
def test_fused_general_ncvt4_wald(synthetic_covariate_data_ncvt4):
    """FGEN-04: Fused general Wald bitwise matches non-fused general for n_cvt=4."""
    _run_fused_general_wald_vs_nonfused(
        _prepare_fused_general_data(synthetic_covariate_data_ncvt4)
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Mode-4 fused general C not available",
)
def test_fused_general_ncvt2_mode4(general_score_lrt_ncvt2):
    """FGEN-07: Fused general mode-4 Wald matches non-fused general Wald for n_cvt=2.

    Verifies the Wald component of mode-4 is bitwise identical to the non-fused
    general workspace. Score and LRT are exercised (no crash, plausible values)
    since their non-fused references use different code paths (batch C functions)
    that may produce different NaN patterns on synthetic data.
    """
    from jamma.lmm.compute_numpy import (
        compute_mode4_fused_general_c_ws,
        compute_wald_general_c_ws,
        create_lmm_workspace_general,
        create_lmm_workspace_mode4_fused_general,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    data = general_score_lrt_ncvt2
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    Uab_batch = data["Uab_batch"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]
    Hi_eval_null = data["Hi_eval_null"]
    logl_H0 = data["logl_H0"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )
    utg_t = np.ascontiguousarray(UtG.T)
    pab_c = build_pab_table_for_c(n_cvt)
    pab_kwargs = {
        k: pab_c[k]
        for k in [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
            "idx_xx",
            "idx_xy",
            "idx_yy",
            "var_a_cols",
            "var_b_cols",
        ]
    }

    # Non-fused reference: Wald from general workspace
    ws_nonfused = create_lmm_workspace_general(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        n_cvt,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    wald_nonfused = compute_wald_general_c_ws(ws_nonfused, uab_var_soa, 1)

    # Fused general mode-4 path
    ws_fused = create_lmm_workspace_mode4_fused_general(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        n_cvt=n_cvt,
        **pab_kwargs,
        hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
    )
    result_fused = compute_mode4_fused_general_c_ws(ws_fused, utg_t, 1)

    # Wald comparison (bitwise)
    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_array_equal(
            wald_nonfused[key],
            result_fused[key],
            err_msg=f"Mode-4 Wald {key}: fused general vs non-fused mismatch",
        )

    # Score/LRT: verify arrays present with correct shape and no crashes
    n_snps = UtG.shape[1]
    assert result_fused["p_scores"].shape == (n_snps,), "p_scores shape mismatch"
    assert result_fused["p_lrts"].shape == (n_snps,), "p_lrts shape mismatch"
    assert result_fused["lambdas_mle"].shape == (n_snps,), "lambdas_mle shape mismatch"

    # Score and LRT p-values should be in [0, 1] or NaN (degenerate SNPs)
    finite_scores = result_fused["p_scores"][np.isfinite(result_fused["p_scores"])]
    assert np.all((finite_scores >= 0) & (finite_scores <= 1)), (
        "Score p-values out of range [0, 1]"
    )
    finite_lrts = result_fused["p_lrts"][np.isfinite(result_fused["p_lrts"])]
    assert np.all((finite_lrts >= 0) & (finite_lrts <= 1)), (
        "LRT p-values out of range [0, 1]"
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Fused general C not available",
)
def test_fused_general_mode4_nan_lambda_regression(general_score_lrt_ncvt2):
    """FGEN-08: Regression test — fused general mode-4 produces finite lambda_mle.

    Previously, fused general mode-4 produced NaN lambda_mle due to missing
    mle_const in the workspace. This test verifies the fix: all non-degenerate
    SNPs must have finite lambda_mle values.
    """
    from jamma.lmm.compute_numpy import (
        compute_mode4_fused_general_c_ws,
        create_lmm_workspace_mode4_fused_general,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    data = general_score_lrt_ncvt2
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    Uab_batch = data["Uab_batch"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]
    Hi_eval_null = data["Hi_eval_null"]
    logl_H0 = data["logl_H0"]

    inv_indices, _ = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    utg_t = np.ascontiguousarray(UtG.T)
    pab_c = build_pab_table_for_c(n_cvt)
    pab_kwargs = {
        k: pab_c[k]
        for k in [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
            "idx_xx",
            "idx_xy",
            "idx_yy",
            "var_a_cols",
            "var_b_cols",
        ]
    }

    ws_fused = create_lmm_workspace_mode4_fused_general(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        n_cvt=n_cvt,
        **pab_kwargs,
        hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
    )
    result = compute_mode4_fused_general_c_ws(ws_fused, utg_t, 1)

    # All non-degenerate SNPs must have finite lambda_mle
    lambdas_mle = result["lambdas_mle"]
    # Degenerate SNPs (constant genotype) may produce NaN — check non-degenerate
    non_degen = np.isfinite(result["betas"])  # Wald beta finite => non-degenerate
    assert np.all(np.isfinite(lambdas_mle[non_degen])), (
        f"NaN lambda_mle found for {np.sum(~np.isfinite(lambdas_mle[non_degen]))} "
        f"non-degenerate SNPs (regression: mode-4 fused general NaN bug)"
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Fused general C not available",
)
def test_fused_general_mode4_lrt_parity_ncvt2(general_score_lrt_ncvt2):
    """FGEN-09: Fused general mode-4 LRT matches compose fallback.

    Compares fused general mode-4 lambdas_mle and p_lrts against the
    non-fused batch LRT C path (compute_lrt_batch_general_c).
    """
    from jamma.lmm._lmm_accel import compute_lrt_batch_general_c
    from jamma.lmm.compute_numpy import (
        compute_mode4_fused_general_c_ws,
        create_lmm_workspace_mode4_fused_general,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    data = general_score_lrt_ncvt2
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    Uab_batch = data["Uab_batch"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]
    Hi_eval_null = data["Hi_eval_null"]
    logl_H0 = data["logl_H0"]

    inv_indices, _ = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    utg_t = np.ascontiguousarray(UtG.T)
    pab_c = build_pab_table_for_c(n_cvt)
    pab_kwargs = {
        k: pab_c[k]
        for k in [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
            "idx_xx",
            "idx_xy",
            "idx_yy",
            "var_a_cols",
            "var_b_cols",
        ]
    }

    # Fused general mode-4 result
    ws_fused = create_lmm_workspace_mode4_fused_general(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        n_cvt=n_cvt,
        **pab_kwargs,
        hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
    )
    result_fused = compute_mode4_fused_general_c_ws(ws_fused, utg_t, 1)

    # Non-fused batch LRT reference
    result_lrt = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_c,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        1,
    )

    # lambdas_mle parity (golden section FP tolerance)
    np.testing.assert_allclose(
        result_fused["lambdas_mle"],
        result_lrt["lambdas_mle"],
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: fused general mode-4 vs batch LRT mismatch",
    )

    # p_lrts parity (CDF tolerance)
    np.testing.assert_allclose(
        result_fused["p_lrts"],
        result_lrt["p_lrts"],
        rtol=1e-4,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: fused general mode-4 vs batch LRT mismatch",
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Fused general C not available",
)
def test_fused_general_mode4_all_statistics_ncvt2(general_score_lrt_ncvt2):
    """FGEN-10: Fused general mode-4 all 8 output arrays match compose reference.

    Verifies lambdas, logls, betas, ses, pwalds (bitwise Wald parity),
    p_scores (Score CDF tolerance), lambdas_mle (golden section tolerance),
    and p_lrts (LRT CDF tolerance) against their respective non-fused references.
    """
    from jamma.lmm._lmm_accel import (
        compute_lrt_batch_general_c,
        compute_score_batch_general_c,
    )
    from jamma.lmm.compute_numpy import (
        compute_mode4_fused_general_c_ws,
        compute_wald_general_c_ws,
        create_lmm_workspace_general,
        create_lmm_workspace_mode4_fused_general,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    data = general_score_lrt_ncvt2
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    Uab_batch = data["Uab_batch"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]
    Hi_eval_null = data["Hi_eval_null"]
    logl_H0 = data["logl_H0"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )
    utg_t = np.ascontiguousarray(UtG.T)
    pab_c = build_pab_table_for_c(n_cvt)
    pab_kwargs = {
        k: pab_c[k]
        for k in [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
            "idx_xx",
            "idx_xy",
            "idx_yy",
            "var_a_cols",
            "var_b_cols",
        ]
    }

    # --- Fused general mode-4 ---
    ws_fused = create_lmm_workspace_mode4_fused_general(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        n_cvt=n_cvt,
        **pab_kwargs,
        hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
    )
    result_fused = compute_mode4_fused_general_c_ws(ws_fused, utg_t, 1)

    # --- Wald reference (non-fused general workspace) ---
    ws_wald = create_lmm_workspace_general(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        n_cvt,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    wald_ref = compute_wald_general_c_ws(ws_wald, uab_var_soa, 1)

    # Wald: bitwise parity (same workspace, same code path)
    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_array_equal(
            result_fused[key],
            wald_ref[key],
            err_msg=f"Mode-4 Wald {key}: fused general vs non-fused mismatch",
        )

    # --- Score reference (batch C) ---
    score_ref = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_c,
        1,
    )
    np.testing.assert_allclose(
        result_fused["p_scores"],
        score_ref["p_scores"],
        rtol=1e-4,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: fused general mode-4 vs batch Score mismatch",
    )

    # --- LRT reference (batch C) ---
    lrt_ref = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_c,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        1,
    )
    np.testing.assert_allclose(
        result_fused["lambdas_mle"],
        lrt_ref["lambdas_mle"],
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: fused general mode-4 vs batch LRT mismatch",
    )
    np.testing.assert_allclose(
        result_fused["p_lrts"],
        lrt_ref["p_lrts"],
        rtol=1e-4,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: fused general mode-4 vs batch LRT mismatch",
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Fused general C not available",
)
def test_fused_general_workspace_lifecycle(synthetic_covariate_data_ncvt2):
    """FGEN-04: Fused general workspace creates, computes, and destroys cleanly."""
    from jamma.lmm.compute_numpy import (
        compute_wald_fused_general_c_ws,
        create_lmm_workspace_fused_general,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = synthetic_covariate_data_ncvt2
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty, n_cvt)
    utg_t = np.ascontiguousarray(UtG.T)
    pab_c = build_pab_table_for_c(n_cvt)
    pab_kwargs = {
        k: pab_c[k]
        for k in [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
            "idx_xx",
            "idx_xy",
            "idx_yy",
            "var_a_cols",
            "var_b_cols",
        ]
    }

    ws = create_lmm_workspace_fused_general(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        n_cvt=n_cvt,
        **pab_kwargs,
    )
    assert ws is not None

    # Compute first half
    mid = UtG.shape[1] // 2
    r1 = compute_wald_fused_general_c_ws(ws, utg_t[:mid], 1)
    assert r1["lambdas"].shape == (mid,)

    # Reuse workspace for second half
    r2 = compute_wald_fused_general_c_ws(ws, utg_t[mid:], 1)
    assert r2["lambdas"].shape == (UtG.shape[1] - mid,)

    # Full batch
    r_full = compute_wald_fused_general_c_ws(ws, utg_t, 1)
    combined = np.concatenate([r1["lambdas"], r2["lambdas"]])
    np.testing.assert_allclose(
        combined,
        r_full["lambdas"],
        rtol=1e-12,
        atol=1e-14,
        err_msg="Chunked vs full fused general workspace mismatch",
    )

    # Destroy (PyCapsule GC)
    del ws


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Fused general C not available",
)
def test_fused_general_degenerate_snps(synthetic_covariate_data_ncvt2):
    """FGEN-04: Degenerate SNPs produce NaN in fused general (same as non-fused)."""
    from jamma.lmm.compute_numpy import (
        compute_wald_fused_general_c_ws,
        compute_wald_general_c_ws,
        create_lmm_workspace_fused_general,
        create_lmm_workspace_general,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    Uab_batch = data["Uab_batch"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    # Inject constant genotype columns (degenerate SNPs)
    UtG_degen = UtG.copy()
    UtG_degen[:, 0] = 0.0  # All zeros
    UtG_degen[:, 1] = 1.0  # All ones (constant)

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])

    # Recompute Uab for degenerate SNPs
    from jamma.lmm.likelihood import compute_Uab

    n_snps = UtG_degen.shape[1]
    n_index = Uab_batch.shape[2]
    Uab_degen = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    for i in range(n_snps):
        Uab_degen[i] = compute_Uab(UtW, Uty, UtG_degen[:, i])
    uab_var_soa_degen = np.ascontiguousarray(
        Uab_degen[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    # Non-fused reference
    ws_nonfused = create_lmm_workspace_general(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        n_cvt,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    result_nonfused = compute_wald_general_c_ws(ws_nonfused, uab_var_soa_degen, 1)

    # Fused general path
    pab_c = build_pab_table_for_c(n_cvt)
    pab_kwargs = {
        k: pab_c[k]
        for k in [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
            "idx_xx",
            "idx_xy",
            "idx_yy",
            "var_a_cols",
            "var_b_cols",
        ]
    }
    utg_t_degen = np.ascontiguousarray(UtG_degen.T)
    ws_fused = create_lmm_workspace_fused_general(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        n_cvt=n_cvt,
        **pab_kwargs,
    )
    result_fused = compute_wald_fused_general_c_ws(ws_fused, utg_t_degen, 1)

    # Degenerate SNPs (0, 1) should have NaN beta/se/pwald in both paths
    for key in ("betas", "ses", "pwalds"):
        np.testing.assert_array_equal(
            np.isnan(result_nonfused[key][:2]),
            np.isnan(result_fused[key][:2]),
            err_msg=f"Degenerate SNP NaN pattern mismatch for {key}",
        )

    # Non-degenerate SNPs should match bitwise
    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_array_equal(
            result_nonfused[key][2:],
            result_fused[key][2:],
            err_msg=f"Non-degenerate {key}: fused vs non-fused mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="C extension not available",
)
def test_fused_general_abi_version_9():
    """FGEN-06: ABI_VERSION is >= 9 for fused general kernel support."""
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION >= 9, f"Expected ABI_VERSION>=9, got {ABI_VERSION}"


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="Fused C not available")
def test_fused_ncvt1_regression(split_wald_data):
    """Regression: n_cvt=1 fused path works after general addition."""
    from jamma.lmm.compute_numpy import (
        compute_wald_fused_c_ws,
        create_lmm_workspace_fused,
    )

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    w = UtW[:, 0].copy()
    utg_t = np.ascontiguousarray(UtG.T)
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

    # SoA reference
    ws_soa = create_lmm_workspace(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    soa_result = compute_wald_split_c_ws(ws_soa, uab_var_soa, 1)

    # Fused n_cvt=1
    ws_fused = create_lmm_workspace_fused(
        eigenvalues,
        uab_inv_soa,
        w,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    fused_result = compute_wald_fused_c_ws(ws_fused, utg_t, 1)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_array_equal(
            soa_result[key],
            fused_result[key],
            err_msg=f"n_cvt=1 regression: {key} mismatch after fused general addition",
        )


@pytest.mark.tier1
@pytest.mark.skipif(
    compute_numpy._accel is None,
    reason="Fused general C not available",
)
def test_runner_fused_general_ncvt2_dispatch():
    """Runner integration: n_cvt=2 dispatches fused general path end-to-end.

    Exercises the full build_pab_table_for_c → create_workspace_fused_general →
    compute_wald_fused_general_c_ws pipeline through run_lmm_association_numpy.
    Compares fused general results (n_cvt=2 with the C extension) against the
    NumPy path, reached by dropping the extension.
    """
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(77)
    n_samples = 100
    n_snps = 80
    n_cvt = 2

    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.4, 0.4, 0.2])
    phenotypes = rng.standard_normal(n_samples)
    covariates = rng.standard_normal((n_samples, n_cvt))
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]
    U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]

    # Run with fused general enabled (default)
    result_fused = run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=None,
        snp_info=snp_info,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=U,
        config=LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=1,
            n_refine=20,
        ),
    )

    # Run with fused general disabled → falls back to non-fused general path.
    # Patch the source module (compute_numpy), which owns dispatch capability flags.
    from unittest.mock import patch

    with patch("jamma.lmm.compute_numpy._accel", None):
        result_nonfused = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=U,
            config=LmmConfig(
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                lmm_mode=1,
                n_refine=20,
            ),
        )

    assoc_fused = result_fused.associations
    assoc_nonfused = result_nonfused.associations

    assert len(assoc_fused) == len(assoc_nonfused), (
        f"Fused: {len(assoc_fused)}, Non-fused: {len(assoc_nonfused)}"
    )
    assert len(assoc_fused) > n_snps * 0.8, (
        f"Too many SNPs filtered: {len(assoc_fused)} of {n_snps}"
    )

    # Not bitwise: the reference run is now the NumPy path, not a second C
    # path. Dropping the fused general kernel used to leave the general split
    # kernel in place, and the two agreed to the last bit; no build exports one
    # without the other, so the honest reference is NumPy, which accumulates in
    # a different order.
    for a_f, a_nf in zip(assoc_fused, assoc_nonfused, strict=True):
        assert a_f.rs == a_nf.rs, f"SNP order mismatch: {a_f.rs} vs {a_nf.rs}"
        np.testing.assert_allclose(
            a_f.p_wald,
            a_nf.p_wald,
            rtol=1e-8,
            atol=1e-14,
            err_msg=f"p_wald mismatch for {a_f.rs}",
        )
        np.testing.assert_allclose(
            a_f.beta,
            a_nf.beta,
            rtol=1e-8,
            atol=1e-14,
            err_msg=f"beta mismatch for {a_f.rs}",
        )
