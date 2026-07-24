"""_lmm_accel C extension tests: single-pass mode-4 kernels.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _compute_lrt_batch_c,
    _compute_score_batch_c,
    compute_lmm_chunk_numpy,
)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_all_c_dispatch(score_lrt_data):
    """Mode 4 (All) returns all 8 keys non-None when C extension available."""
    eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data

    result = compute_lmm_chunk_numpy(
        lmm_mode=4,
        n_cvt=1,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        Hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
        n_threads=1,
    )

    expected_keys = [
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "p_lrts",
        "p_scores",
    ]
    for key in expected_keys:
        assert result[key] is not None, f"Mode 4 result['{key}'] should not be None"
        assert isinstance(result[key], np.ndarray), f"result['{key}'] should be ndarray"
        assert result[key].shape == (Uab_batch.shape[0],), (
            f"result['{key}'] shape mismatch: {result[key].shape}"
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_fused_workspace_api(score_lrt_data):
    """Fused mode-4 workspace creation and compute returns all 8 keys."""
    from jamma.lmm.compute_numpy import (
        _C_MODE4_AVAILABLE,
        compute_mode4_split_c_ws,
        create_lmm_workspace_mode4,
    )

    if not _C_MODE4_AVAILABLE:
        pytest.skip("Mode-4 fused C extension not available")

    eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data

    # Build SoA arrays
    uab_inv_soa = np.stack(
        [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
    )
    uab_var_soa = np.stack(
        [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
    )

    ws = create_lmm_workspace_mode4(
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
    assert ws is not None

    cr = compute_mode4_split_c_ws(ws, uab_var_soa, 1)

    expected_keys = [
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "p_lrts",
        "p_scores",
    ]
    for key in expected_keys:
        assert key in cr, f"Missing key '{key}' in fused mode-4 result"
        assert isinstance(cr[key], np.ndarray), f"result['{key}'] should be ndarray"
        assert cr[key].shape == (Uab_batch.shape[0],), (
            f"result['{key}'] shape {cr[key].shape} != ({Uab_batch.shape[0]},)"
        )


def _build_mode4_soa_and_fused(score_lrt_data):
    """Helper: build SoA arrays and compute both fused and compose results.

    Returns (fused_cr, compose_cr, eigenvalues, Uab_batch, n_samples,
             Hi_eval_null, logl_H0, uab_inv_soa, uab_var_soa).
    """
    from jamma.lmm.compute_numpy import (
        compute_mode4_split_c_ws,
        compute_wald_split_c_ws,
        create_lmm_workspace,
        create_lmm_workspace_mode4,
    )

    eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data

    # Build SoA arrays
    uab_inv_soa = np.stack(
        [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
    )
    uab_var_soa = np.stack(
        [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
    )

    # Fused path
    ws_mode4 = create_lmm_workspace_mode4(
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
    fused_cr = compute_mode4_split_c_ws(ws_mode4, uab_var_soa, 1)

    # Compose path: Wald workspace + SoA split Score/LRT
    from jamma.lmm.chunk_dispatch import _compose_mode4_from_split

    ws_wald = create_lmm_workspace(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    wald_cr = compute_wald_split_c_ws(ws_wald, uab_var_soa, 1)
    compose_cr = _compose_mode4_from_split(
        wald_cr,
        1,
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        n_samples,
        Hi_eval_null=Hi_eval_null,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        logl_H0=logl_H0,
        n_threads=1,
    )

    return (
        fused_cr,
        compose_cr,
        eigenvalues,
        Uab_batch,
        n_samples,
        Hi_eval_null,
        logl_H0,
        uab_inv_soa,
        uab_var_soa,
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_fused_split_parity(score_lrt_data):
    """Fused C kernel matches compose path (Wald+Score+LRT) within tolerance."""
    from jamma.lmm.compute_numpy import _C_MODE4_AVAILABLE

    if not _C_MODE4_AVAILABLE:
        pytest.skip("Mode-4 fused C extension not available")

    fused_cr, compose_cr, *_ = _build_mode4_soa_and_fused(score_lrt_data)

    # Wald outputs: betas, ses, lambdas, logls, pwalds
    for key in ("lambdas", "logls", "betas", "ses"):
        np.testing.assert_allclose(
            fused_cr[key],
            compose_cr[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: fused vs compose mismatch",
        )
    np.testing.assert_allclose(
        fused_cr["pwalds"],
        compose_cr["pwalds"],
        rtol=1e-6,
        atol=1e-14,
        equal_nan=True,
        err_msg="pwalds: fused vs compose mismatch",
    )

    # Score: p_scores
    np.testing.assert_allclose(
        fused_cr["p_scores"],
        compose_cr["p_scores"],
        rtol=1e-6,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: fused vs compose mismatch",
    )

    # LRT: lambdas_mle, p_lrts
    # lambdas_mle: fused uses SoA split accumulation while standalone uses
    # full Uab dot products — golden section on flat MLE landscapes can
    # converge to slightly different optima (~3e-5 relative).
    np.testing.assert_allclose(
        fused_cr["lambdas_mle"],
        compose_cr["lambdas_mle"],
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: fused vs compose mismatch",
    )
    np.testing.assert_allclose(
        fused_cr["p_lrts"],
        compose_cr["p_lrts"],
        rtol=1e-4,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: fused vs compose mismatch",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_shared_grid_preserves_distinct_reml_mle_brackets(score_lrt_data):
    """Shared reductions retain independent REML and MLE coarse brackets."""
    fused_cr, compose_cr, *_ = _build_mode4_soa_and_fused(score_lrt_data)
    grid_step = (np.log(1e5) - np.log(1e-5)) / 49
    log_separation = np.abs(
        np.log(fused_cr["lambdas"]) - np.log(fused_cr["lambdas_mle"])
    )

    assert np.any(log_separation > 2 * grid_step)
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
        np.testing.assert_allclose(
            fused_cr[key],
            compose_cr[key],
            rtol=1e-4,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: shared-grid vs independent compose mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_fused_score_matches_standalone(score_lrt_data):
    """Fused p_scores match standalone compute_score_batch_c on reconstructed Uab.

    Both paths use the same SoA invariant columns (from SNP 0), so the
    reconstructed Uab fed to standalone Score is consistent with the fused
    kernel's input. This tests that Score accumulation in the fused loop
    matches the standalone batch Score function.
    """
    from jamma.lmm.compute_numpy import _C_MODE4_AVAILABLE
    from jamma.lmm.likelihood_numpy import reconstruct_uab_from_soa

    if not _C_MODE4_AVAILABLE:
        pytest.skip("Mode-4 fused C extension not available")
    if _compute_score_batch_c is None:
        pytest.skip("Score C batch not available")

    (
        fused_cr,
        _,
        eigenvalues,
        _,
        n_samples,
        Hi_eval_null,
        _,
        uab_inv_soa,
        uab_var_soa,
    ) = _build_mode4_soa_and_fused(score_lrt_data)

    # Reconstruct full Uab from the same SoA data the fused kernel uses
    Uab_reconstructed = reconstruct_uab_from_soa(uab_inv_soa, uab_var_soa)

    # Standalone Score via reconstructed Uab
    standalone_score = _compute_score_batch_c(
        eigenvalues,
        Uab_reconstructed,
        Hi_eval_null,
        n_samples,
        1,
    )

    np.testing.assert_allclose(
        fused_cr["p_scores"],
        standalone_score["p_scores"],
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: fused vs standalone mismatch",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_fused_lrt_matches_standalone(score_lrt_data):
    """Fused p_lrts match standalone compute_lrt_batch_c on reconstructed Uab.

    Both paths use the same SoA invariant columns, so the reconstructed Uab
    is consistent with the fused kernel's input.
    """
    from jamma.lmm.compute_numpy import _C_MODE4_AVAILABLE
    from jamma.lmm.likelihood_numpy import reconstruct_uab_from_soa

    if not _C_MODE4_AVAILABLE:
        pytest.skip("Mode-4 fused C extension not available")
    if _compute_lrt_batch_c is None:
        pytest.skip("LRT C batch not available")

    (fused_cr, _, eigenvalues, _, n_samples, _, logl_H0, uab_inv_soa, uab_var_soa) = (
        _build_mode4_soa_and_fused(score_lrt_data)
    )

    # Reconstruct full Uab from the same SoA data
    Uab_reconstructed = reconstruct_uab_from_soa(uab_inv_soa, uab_var_soa)

    # Standalone LRT via reconstructed Uab
    standalone_lrt = _compute_lrt_batch_c(
        eigenvalues,
        Uab_reconstructed,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        1,
    )

    # lambdas_mle: fused uses SoA split accumulation, standalone uses full Uab
    # dot products — golden section on flat MLE landscapes can produce
    # ~3e-5 relative difference.
    np.testing.assert_allclose(
        fused_cr["lambdas_mle"],
        standalone_lrt["lambdas_mle"],
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: fused vs standalone mismatch",
    )
    np.testing.assert_allclose(
        fused_cr["p_lrts"],
        standalone_lrt["p_lrts"],
        rtol=1e-4,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: fused vs standalone mismatch",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_fused_degenerate_snps(score_lrt_data):
    """Fused mode-4 handles degenerate SNPs: NaN Wald/Score, p_lrt ~ 1.0."""
    from jamma.lmm.compute_numpy import (
        _C_MODE4_AVAILABLE,
        compute_mode4_split_c_ws,
        create_lmm_workspace_mode4,
    )

    if not _C_MODE4_AVAILABLE:
        pytest.skip("Mode-4 fused C extension not available")

    eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data

    # Build SoA arrays
    uab_inv_soa = np.stack(
        [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
    )
    uab_var_soa = np.stack(
        [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
    )

    # Make first SNP degenerate: constant genotype -> wx=0, xx=0, xy=0
    uab_var_soa_degen = uab_var_soa.copy()
    uab_var_soa_degen[0, :, :] = 0.0  # all three varying columns zeroed

    ws = create_lmm_workspace_mode4(
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
    cr = compute_mode4_split_c_ws(ws, uab_var_soa_degen, 1)

    # Degenerate SNP: Wald and Score outputs should be NaN
    assert np.isnan(cr["betas"][0]), "degenerate SNP should have NaN beta"
    assert np.isnan(cr["ses"][0]), "degenerate SNP should have NaN se"
    assert np.isnan(cr["pwalds"][0]), "degenerate SNP should have NaN p_wald"
    assert np.isnan(cr["p_scores"][0]), "degenerate SNP should have NaN p_score"

    # LRT: degenerate SNP has no signal, so p_lrt ~ 1.0
    assert cr["p_lrts"][0] >= 0.99, (
        f"degenerate SNP should have p_lrt ~ 1.0, got {cr['p_lrts'][0]}"
    )

    # Remaining SNPs: compare against un-zeroed run to exclude naturally-degenerate ones
    cr_ref = compute_mode4_split_c_ws(ws, uab_var_soa, 1)
    finite_betas = np.isfinite(cr_ref["betas"][1:])
    finite_lrts = np.isfinite(cr_ref["p_lrts"][1:])
    assert np.all(np.isfinite(cr["betas"][1:][finite_betas])), (
        "non-degenerate betas should be finite"
    )
    assert np.all(np.isfinite(cr["p_lrts"][1:][finite_lrts])), (
        "non-degenerate p_lrts should be finite"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_fused_rejects_wald_workspace(score_lrt_data):
    """Passing a Wald (mode=0) workspace to fused mode-4 compute raises ValueError."""
    from jamma.lmm.compute_numpy import (
        _C_MODE4_AVAILABLE,
        _C_SPLIT_AVAILABLE,
        compute_mode4_split_c_ws,
        create_lmm_workspace,
    )

    if not _C_MODE4_AVAILABLE or not _C_SPLIT_AVAILABLE:
        pytest.skip("Mode-4 fused or split C extension not available")

    eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data

    uab_inv_soa = np.stack(
        [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
    )
    uab_var_soa = np.stack(
        [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
    )

    # Create a standard Wald workspace (mode=0)
    wald_ws = create_lmm_workspace(
        eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1
    )

    with pytest.raises(ValueError, match="mode-4 workspace"):
        compute_mode4_split_c_ws(wald_ws, uab_var_soa, 1)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_mode4_fused_multithreaded_parity(score_lrt_data):
    """Multi-threaded fused mode-4 results must match single-threaded results.

    The fused mode-4 kernel dispatches Score/LRT/Wald in a single pass over
    SNPs. This test verifies that OpenMP parallelism does not introduce race
    conditions or thread-local state corruption in the fused kernel path.
    """
    from jamma.core.threading import get_physical_core_count
    from jamma.lmm.compute_numpy import (
        _C_MODE4_AVAILABLE,
        compute_mode4_split_c_ws,
        create_lmm_workspace_mode4,
    )

    if not _C_MODE4_AVAILABLE:
        pytest.skip("Mode-4 fused C extension not available")

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data

    # Build SoA arrays (same pattern as _build_mode4_soa_and_fused)
    uab_inv_soa = np.stack(
        [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
    )
    uab_var_soa = np.stack(
        [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
    )

    # Single-threaded computation
    ws_1t = create_lmm_workspace_mode4(
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
    cr_1t = compute_mode4_split_c_ws(ws_1t, uab_var_soa, 1)

    # Multi-threaded computation
    ws_mt = create_lmm_workspace_mode4(
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
    cr_mt = compute_mode4_split_c_ws(ws_mt, uab_var_soa, n_threads)

    # All 8 result keys must match at tight tolerance
    expected_keys = [
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "p_lrts",
        "p_scores",
    ]
    for key in expected_keys:
        assert key in cr_1t, f"Missing key '{key}' in single-threaded result"
        assert key in cr_mt, f"Missing key '{key}' in multi-threaded result"
        np.testing.assert_allclose(
            cr_mt[key],
            cr_1t[key],
            rtol=1e-12,
            err_msg=(
                f"Multi-threaded mode-4 '{key}' diverges from single-threaded "
                f"at n_threads={n_threads}"
            ),
        )
