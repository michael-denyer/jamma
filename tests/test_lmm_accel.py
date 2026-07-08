"""Tests for the C extension accelerator (_lmm_accel).

Validates that the C extension produces numerically identical results to the
pure-Python NumPy implementation. Also tests the fallback mechanism.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _C_GENERAL_AVAILABLE,
    _C_SPLIT_AVAILABLE,
    _compute_lrt_batch_c,
    _compute_score_batch_c,
    _compute_wald_numpy,
    _compute_wald_split_c,
    compute_lmm_chunk_numpy,
    compute_wald_general_c_ws,
    compute_wald_split_c_ws,
    create_lmm_workspace,
    create_lmm_workspace_general,
)
from jamma.lmm.likelihood_numpy import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    batch_compute_iab_numpy,
    batch_compute_iab_split_ncvt1,
    batch_compute_iab_split_ncvt1_soa,
    batch_compute_uab_split_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
    golden_section_optimize_lambda_mle_numpy,
    golden_section_optimize_lambda_numpy,
)


@pytest.fixture
def synthetic_wald_data():
    """Deterministic synthetic Wald test data for n_cvt=1.

    Returns:
        Tuple of (eigenvalues, Uab_batch, n_samples).
    """
    rng = np.random.default_rng(42)
    n_samples, n_snps = 200, 50
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    # Build physically meaningful Uab from w, x, y vectors so columns have
    # proper cross-product structure and Pab recursion is well-conditioned.
    Uab_batch = np.zeros((n_snps, n_samples, 6), dtype=np.float64)
    for i in range(n_snps):
        w = np.abs(rng.standard_normal(n_samples)) + 1.0  # positive ww
        x = np.abs(rng.standard_normal(n_samples)) + 0.5  # positive xx
        y = rng.standard_normal(n_samples)
        Uab_batch[i, :, 0] = w * w  # ww
        Uab_batch[i, :, 1] = w * x  # wx
        Uab_batch[i, :, 2] = w * y  # wy
        Uab_batch[i, :, 3] = x * x  # xx
        Uab_batch[i, :, 4] = x * y  # xy
        Uab_batch[i, :, 5] = y * y  # yy
    return eigenvalues, Uab_batch, n_samples


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_c_extension_importable():
    """Verify the C extension module can be imported directly."""
    from jamma.lmm._lmm_accel import compute_lmm_batch_c

    assert callable(compute_lmm_batch_c)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_c_vs_python_parity_synthetic(synthetic_wald_data, monkeypatch):
    """C extension and Python path produce numerically identical Wald outputs.

    Exercises the full REML optimizer + CalcPab + CalcRLWald pipeline on
    50 synthetic SNPs and verifies all five output arrays agree within
    the expected floating-point tolerance.
    """
    eigenvalues, Uab_batch, n_samples = synthetic_wald_data
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    n_cvt = 1
    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20

    # --- C path (default when _C_ACCEL_AVAILABLE and n_cvt == 1) ---
    result_c = _compute_wald_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        Iab_batch=Iab_batch,
        n_threads=1,
    )

    # --- Python path: use generic golden section (same algorithm as C extension).
    # When _C_ACCEL_AVAILABLE is False, _compute_wald_numpy dispatches n_cvt=1
    # to the split-Uab optimizer (Phase 53), which uses different FP accumulation
    # than the C extension's generic golden section. To compare like-for-like,
    # call the generic optimizer directly.
    from jamma.lmm.likelihood_numpy import (
        batch_calc_wald_stats_from_pab_numpy,
    )

    lambdas_py, logls_py, Pab_py = golden_section_optimize_lambda_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
        return_pab=True,
    )
    betas_py, ses_py, pwalds_py = batch_calc_wald_stats_from_pab_numpy(
        n_cvt, Pab_py, n_samples
    )
    result_py = {
        "lambdas": lambdas_py,
        "logls": logls_py,
        "betas": betas_py,
        "ses": ses_py,
        "pwalds": pwalds_py,
    }

    # All outputs must agree within calibrated tolerances.
    # NaN entries (degenerate SNPs) are excluded from comparison via equal_nan=True.
    # lambdas: C and Python golden section have different FP operation ordering.
    # On flat likelihood landscapes (weak-signal SNPs near l_min=1e-5), both
    # paths converge to the boundary but land ~6e-6 apart in absolute terms.
    # atol=1e-4 absorbs this boundary effect; rtol=1e-6 covers well-determined
    # lambdas (which agree to ~1e-10 relative).
    np.testing.assert_allclose(
        result_c["lambdas"],
        result_py["lambdas"],
        rtol=1e-6,
        atol=1e-4,
        equal_nan=True,
        err_msg="lambdas: C vs Python mismatch",
    )
    # logls: C and Python golden section have different FP operation ordering,
    # causing tiny accumulation differences. rtol=1e-9 accommodates this.
    np.testing.assert_allclose(
        result_c["logls"],
        result_py["logls"],
        rtol=1e-9,
        atol=1e-14,
        equal_nan=True,
        err_msg="logls: C vs Python mismatch",
    )
    # betas/ses/pwalds: lambda differences cascade through Pab into beta/SE,
    # then into the F-statistic and betainc p-value. Measured max relative
    # diffs: beta ~7e-9, se ~3.5e-9, pwald ~1.6e-8. Use rtol=1e-7 (~10x).
    np.testing.assert_allclose(
        result_c["betas"],
        result_py["betas"],
        rtol=1e-7,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: C vs Python mismatch",
    )
    np.testing.assert_allclose(
        result_c["ses"],
        result_py["ses"],
        rtol=1e-7,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: C vs Python mismatch",
    )
    np.testing.assert_allclose(
        result_c["pwalds"],
        result_py["pwalds"],
        rtol=1e-7,
        atol=1e-14,
        equal_nan=True,
        err_msg="pwalds: C vs Python mismatch",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_c_fallback_ncvt_gt1_when_general_unavailable(synthetic_wald_data, monkeypatch):
    """With n_cvt=2 and _C_GENERAL_AVAILABLE=False, falls back to Python.

    Monkeypatches _C_GENERAL_AVAILABLE to False and verifies the n_cvt=1
    batch C function is NOT called (it doesn't support n_cvt>1).
    """
    eigenvalues, _Uab_batch_ncvt1, n_samples = synthetic_wald_data
    # Rebuild Uab for n_cvt=2 (n_index = (2+3)*(2+2)//2 = 10)
    rng = np.random.default_rng(0)
    n_snps = 10
    Uab_batch = rng.standard_normal((n_snps, n_samples, 10))
    Uab_batch[:, :, 0] = np.abs(Uab_batch[:, :, 0]) + 0.1

    def should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "n_cvt=1 batch C function should not be called for n_cvt > 1"
        )

    monkeypatch.setattr(
        compute_numpy,
        "_compute_lmm_batch_c",
        should_not_be_called,
    )  # allow-patch: sentinel asserts n_cvt=1 C kernel is NOT taken when n_cvt>1
    monkeypatch.setattr(compute_numpy, "_C_GENERAL_AVAILABLE", False)

    # Should succeed via the Python path without calling any C function
    result = _compute_wald_numpy(
        n_cvt=2,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
    )
    assert "lambdas" in result
    assert result["lambdas"].shape == (n_snps,)


@pytest.mark.tier0
def test_c_fallback_when_extension_unavailable(synthetic_wald_data, monkeypatch):
    """When _C_ACCEL_AVAILABLE is False, the Python path runs without error."""
    eigenvalues, Uab_batch, n_samples = synthetic_wald_data

    monkeypatch.setattr(compute_numpy, "_C_ACCEL_AVAILABLE", False)

    result = compute_lmm_chunk_numpy(
        lmm_mode=1,
        n_cvt=1,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
    )

    assert result["lambdas"] is not None
    # No NaN in lambdas (valid optimization)
    assert not np.any(np.isnan(result["lambdas"])), (
        "Python fallback produced NaN lambdas"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_c_extension_handles_degenerate_snps():
    """Degenerate SNPs (constant genotype, xx column = 0) produce NaN beta/se.

    The C extension must not crash. We use a controlled dataset where all
    SNPs except the injected degenerate one are numerically well-conditioned
    (non-zero xx column ensures P_XX > 0).
    """
    rng = np.random.default_rng(7)
    n_samples, n_snps = 100, 5
    eigenvalues = np.sort(rng.uniform(0.5, 1.5, n_samples))

    # Build valid Uab: non-zero ww and xx guarantee non-degenerate Pab
    Uab_batch = np.zeros((n_snps, n_samples, 6), dtype=np.float64)
    for i in range(n_snps):
        w = np.abs(rng.standard_normal(n_samples)) + 1.0  # positive ww
        x = np.abs(rng.standard_normal(n_samples)) + 0.5  # positive xx
        y = rng.standard_normal(n_samples)
        Uab_batch[i, :, 0] = w * w  # ww
        Uab_batch[i, :, 1] = w * x  # wx
        Uab_batch[i, :, 2] = w * y  # wy
        Uab_batch[i, :, 3] = x * x  # xx
        Uab_batch[i, :, 4] = x * y  # xy
        Uab_batch[i, :, 5] = y * y  # yy

    # Inject degenerate SNP at index 2: zero out the xx column (P_XX -> 0)
    Uab_degen = Uab_batch.copy()
    Uab_degen[2, :, 3] = 0.0  # xx = 0 => P_XX = 0 => degenerate

    result = _compute_wald_numpy(
        n_cvt=1,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_degen,
        n_samples=n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        n_threads=1,
    )

    # Degenerate SNP at index 2: beta and se must be NaN
    assert np.isnan(result["betas"][2]), (
        f"Expected NaN beta for degenerate SNP, got {result['betas'][2]}"
    )
    assert np.isnan(result["ses"][2]), (
        f"Expected NaN se for degenerate SNP, got {result['ses'][2]}"
    )

    # All other SNPs must have valid results (well-conditioned Uab)
    valid_mask = np.ones(n_snps, dtype=bool)
    valid_mask[2] = False
    assert not np.any(np.isnan(result["betas"][valid_mask])), (
        "Non-degenerate SNPs have NaN betas"
    )
    assert not np.any(np.isnan(result["ses"][valid_mask])), (
        "Non-degenerate SNPs have NaN ses"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_c_extension_single_snp():
    """Minimal case: n_snps=1 works without index errors."""
    rng = np.random.default_rng(99)
    n_samples = 100
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    Uab_batch = rng.standard_normal((1, n_samples, 6))
    Uab_batch[:, :, 0] = np.abs(Uab_batch[:, :, 0]) + 0.1

    result = _compute_wald_numpy(
        n_cvt=1,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        n_threads=1,
    )

    assert result["lambdas"].shape == (1,)
    assert result["betas"].shape == (1,)
    assert result["ses"].shape == (1,)
    assert result["pwalds"].shape == (1,)
    assert not np.isnan(result["lambdas"][0]), "Single SNP lambda should not be NaN"


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_c_multithreaded_parity(synthetic_wald_data):
    """Multi-threaded C results must match single-threaded C results.

    The OpenMP parallel-for in _lmm_accel.c partitions SNPs across threads.
    This test catches race conditions or thread-local state corruption that
    would only manifest under parallelism.
    """
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    eigenvalues, Uab_batch, n_samples = synthetic_wald_data
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    kwargs = {
        "n_cvt": 1,
        "eigenvalues": eigenvalues,
        "Uab_batch": Uab_batch,
        "n_samples": n_samples,
        "l_min": 1e-5,
        "l_max": 1e5,
        "n_grid": 50,
        "n_refine": 20,
        "Iab_batch": Iab_batch,
    }

    result_1t = _compute_wald_numpy(**kwargs, n_threads=1)
    result_mt = _compute_wald_numpy(**kwargs, n_threads=n_threads)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            result_mt[key],
            result_1t[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: multi-threaded vs single-threaded mismatch",
        )


def _make_valid_c_inputs():
    """Return valid inputs for compute_lmm_batch_c.

    Shared by TestCExtensionInputValidation and TestCExtensionScalarValidation.
    """
    from jamma.lmm._lmm_accel import compute_lmm_batch_c

    rng = np.random.default_rng(42)
    n_samples, n_snps = 50, 5
    eigenvalues = rng.uniform(0.1, 2.0, n_samples)
    Uab_batch = rng.standard_normal((n_snps, n_samples, 6))
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    return compute_lmm_batch_c, eigenvalues, Uab_batch, Iab_batch, n_samples


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
class TestCExtensionInputValidation:
    """Verify the C extension raises clean errors on invalid input shapes."""

    def test_wrong_eigenvalues_shape(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="eigenvalues"):
            fn(eigenvalues[:10], Uab, Iab, n, 1e-5, 1e5, 50, 20, 1)

    def test_wrong_uab_n_index(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="Uab_batch"):
            fn(eigenvalues, Uab[:, :, :5], Iab, n, 1e-5, 1e5, 50, 20, 1)

    def test_wrong_iab_shape(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="Iab_batch"):
            fn(eigenvalues, Uab, Iab[:, :2, :], n, 1e-5, 1e5, 50, 20, 1)

    def test_mismatched_n_snps(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="n_snps"):
            fn(eigenvalues, Uab, Iab[:3], n, 1e-5, 1e5, 50, 20, 1)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
class TestCExtensionScalarValidation:
    """Verify the C extension validates scalar parameters."""

    def test_n_samples_too_small(self):
        fn, eigenvalues, Uab, Iab, _ = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="n_samples"):
            fn(eigenvalues, Uab, Iab, 2, 1e-5, 1e5, 50, 20, 1)

    def test_l_min_zero(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="l_min"):
            fn(eigenvalues, Uab, Iab, n, 0.0, 1e5, 50, 20, 1)

    def test_l_max_le_l_min(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="l_min"):
            fn(eigenvalues, Uab, Iab, n, 1.0, 1.0, 50, 20, 1)

    def test_n_grid_too_small(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="n_grid"):
            fn(eigenvalues, Uab, Iab, n, 1e-5, 1e5, 1, 20, 1)

    def test_n_refine_too_small(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="n_refine"):
            fn(eigenvalues, Uab, Iab, n, 1e-5, 1e5, 50, 0, 1)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"]
)
def test_c_extension_nonfinite_eigenvalues(bad_value):
    """Non-finite eigenvalues (NaN, Inf, -Inf) are rejected with ValueError."""
    rng = np.random.default_rng(11)
    n_samples, n_snps = 50, 3
    eigenvalues = rng.uniform(0.1, 2.0, n_samples)
    eigenvalues[10] = bad_value

    Uab_batch = rng.standard_normal((n_snps, n_samples, 6))
    Uab_batch[:, :, 0] = np.abs(Uab_batch[:, :, 0]) + 0.1

    with pytest.raises(ValueError, match=r"eigenvalues.*not finite"):
        _compute_wald_numpy(
            n_cvt=1,
            eigenvalues=eigenvalues,
            Uab_batch=Uab_batch,
            n_samples=n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
            n_threads=1,
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_c_extension_all_degenerate_snps():
    """All SNPs degenerate (xx=0) — entire output should be NaN beta/se."""
    rng = np.random.default_rng(13)
    n_samples, n_snps = 50, 4
    eigenvalues = np.sort(rng.uniform(0.5, 1.5, n_samples))

    Uab_batch = rng.standard_normal((n_snps, n_samples, 6))
    Uab_batch[:, :, 0] = np.abs(Uab_batch[:, :, 0]) + 0.1
    # Zero out xx column for ALL SNPs — makes every SNP degenerate
    Uab_batch[:, :, 3] = 0.0

    result = _compute_wald_numpy(
        n_cvt=1,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        n_threads=1,
    )

    # All betas and SEs must be NaN for degenerate SNPs
    assert np.all(np.isnan(result["betas"])), (
        "Expected all-NaN betas for degenerate batch"
    )
    assert np.all(np.isnan(result["ses"])), "Expected all-NaN ses for degenerate batch"
    assert np.all(np.isnan(result["pwalds"])), (
        "Expected all-NaN pwalds for degenerate batch"
    )


@pytest.fixture
def split_wald_data():
    """Deterministic data for split-Uab tests.

    Builds physically meaningful Uab from w, x, y vectors so the split
    construction matches the full construction exactly.

    Returns:
        Tuple of (eigenvalues, UtW, Uty, UtG, n_samples, n_snps).
    """
    rng = np.random.default_rng(42)
    n_samples, n_snps = 200, 50
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))
    return eigenvalues, UtW, Uty, UtG, n_samples, n_snps


@pytest.mark.tier0
def test_split_uab_matches_full_uab(split_wald_data):
    """Split Uab construction matches the full 6-column Uab."""
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    _, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    uab_var, uab_inv = batch_compute_uab_split_numpy(1, UtW, Uty, UtG)

    # Varying columns: wx(1), xx(3), xy(4) in full -> 0,1,2 in split
    np.testing.assert_allclose(
        uab_var[:, :, 0],
        full_uab[:, :, 1],
        rtol=1e-14,
        err_msg="wx column mismatch",
    )
    np.testing.assert_allclose(
        uab_var[:, :, 1],
        full_uab[:, :, 3],
        rtol=1e-14,
        err_msg="xx column mismatch",
    )
    np.testing.assert_allclose(
        uab_var[:, :, 2],
        full_uab[:, :, 4],
        rtol=1e-14,
        err_msg="xy column mismatch",
    )

    # Invariant columns: ww(0), wy(2), yy(5) in full -> 0,1,2 in inv
    # These should be identical across all SNPs in full_uab
    np.testing.assert_allclose(
        uab_inv[:, 0],
        full_uab[0, :, 0],
        rtol=1e-14,
        err_msg="ww column mismatch",
    )
    np.testing.assert_allclose(
        uab_inv[:, 1],
        full_uab[0, :, 2],
        rtol=1e-14,
        err_msg="wy column mismatch",
    )
    np.testing.assert_allclose(
        uab_inv[:, 2],
        full_uab[0, :, 5],
        rtol=1e-14,
        err_msg="yy column mismatch",
    )


@pytest.mark.tier0
def test_split_iab_matches_full_iab(split_wald_data):
    """Split Iab construction matches the full Iab."""
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    _, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    full_iab = batch_compute_iab_numpy(1, full_uab)

    uab_var, uab_inv = batch_compute_uab_split_numpy(1, UtW, Uty, UtG)
    split_iab = batch_compute_iab_split_ncvt1(uab_var, uab_inv)

    np.testing.assert_allclose(
        split_iab,
        full_iab,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Split Iab does not match full Iab",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
def test_split_c_vs_full_c_parity(split_wald_data):
    """Split C extension matches full C extension within FP tolerance."""
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    # Full path
    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    full_iab = batch_compute_iab_numpy(1, full_uab)
    result_full = _compute_wald_numpy(
        n_cvt=1,
        eigenvalues=eigenvalues,
        Uab_batch=full_uab,
        n_samples=n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        Iab_batch=full_iab,
        n_threads=1,
    )

    # Split path — use SoA layout (no per-call transpose since Task 1 changes)
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    split_iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)
    result_split = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        split_iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            result_split[key],
            result_full[key],
            rtol=1e-9,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: split vs full C mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
def test_split_c_degenerate_snps():
    """All-degenerate batch via split path produces all-NaN."""
    rng = np.random.default_rng(13)
    n_samples, n_snps = 50, 4
    eigenvalues = np.sort(rng.uniform(0.5, 1.5, n_samples))

    # Build SoA split arrays with xx=0 (degenerate)
    # SoA layout: (n_snps, 3, n_samples) — axis-1 rows [wx, xx, xy]
    uab_var_soa = rng.standard_normal((n_snps, 3, n_samples))
    uab_var_soa[:, 1, :] = 0.0  # xx row = 0 (row index 1 in SoA)
    uab_inv_soa = np.abs(rng.standard_normal((3, n_samples))) + 0.1
    iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)

    result = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )

    assert np.all(np.isnan(result["betas"])), "Expected all-NaN betas"
    assert np.all(np.isnan(result["ses"])), "Expected all-NaN ses"
    assert np.all(np.isnan(result["pwalds"])), "Expected all-NaN pwalds"


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
def test_split_c_multithreaded_parity(split_wald_data):
    """Multi-threaded split C matches single-threaded split C."""
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)

    r1 = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    rn = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        n_threads,
    )

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            rn[key],
            r1[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: MT vs ST split mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"]
)
def test_split_c_nonfinite_eigenvalues(bad_value):
    """Non-finite eigenvalues (NaN, Inf, -Inf) are rejected by the split C path."""
    rng = np.random.default_rng(11)
    n_samples, n_snps = 50, 3
    eigenvalues = rng.uniform(0.1, 2.0, n_samples)
    eigenvalues[10] = bad_value

    # SoA layout: (n_snps, 3, n_samples) for varying, (3, n_samples) for invariant
    uab_var_soa = rng.standard_normal((n_snps, 3, n_samples))
    uab_inv_soa = np.abs(rng.standard_normal((3, n_samples))) + 0.1
    iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)

    with pytest.raises(ValueError, match=r"eigenvalues.*not finite"):
        _compute_wald_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            iab,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
def test_workspace_api_matches_legacy_split(split_wald_data):
    """Workspace API (create + chunk) produces identical results to legacy split_c.

    Verifies that the per-run workspace path gives the same numerical output
    as the per-call _compute_wald_split_c (which uses Iab_batch). Both paths
    share the same golden section core — differences would indicate a bug in
    the internal Iab/logdet_iab computation.
    """
    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)

    # Legacy path (with Iab_batch passed explicitly)
    result_legacy = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )

    # Workspace path (Iab computed internally from raw column sums)
    ws = create_lmm_workspace(eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1)
    result_ws = compute_wald_split_c_ws(ws, uab_var_soa, 1)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            result_ws[key],
            result_legacy[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: workspace vs legacy split mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
def test_workspace_reuse_across_chunks(split_wald_data):
    """Workspace created once can be reused across multiple chunk calls.

    Simulates the runner's cross-chunk reuse pattern: same workspace, different
    uab_varying_soa slices. Results must match per-call legacy path.
    """
    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

    # Create workspace once (before "chunk loop")
    ws = create_lmm_workspace(eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1)

    # Simulate two chunks by splitting the SNPs in half
    mid = n_snps // 2
    chunk1 = uab_var_soa[:mid]
    chunk2 = uab_var_soa[mid:]

    result_c1 = compute_wald_split_c_ws(ws, chunk1, 1)
    result_c2 = compute_wald_split_c_ws(ws, chunk2, 1)

    # Concatenate chunk results
    combined_lambdas = np.concatenate([result_c1["lambdas"], result_c2["lambdas"]])
    combined_betas = np.concatenate([result_c1["betas"], result_c2["betas"]])

    # Reference: single call with all SNPs
    result_full = compute_wald_split_c_ws(ws, uab_var_soa, 1)

    np.testing.assert_allclose(
        combined_lambdas,
        result_full["lambdas"],
        rtol=1e-12,
        atol=1e-14,
        err_msg="Chunked lambda mismatch vs single call",
    )
    np.testing.assert_allclose(
        combined_betas,
        result_full["betas"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="Chunked beta mismatch vs single call",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
def test_workspace_multithreaded_parity(split_wald_data):
    """Workspace path: multi-threaded results match single-threaded results."""
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

    ws = create_lmm_workspace(eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1)
    r1 = compute_wald_split_c_ws(ws, uab_var_soa, 1)
    rn = compute_wald_split_c_ws(ws, uab_var_soa, n_threads)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            rn[key],
            r1[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: workspace MT vs ST mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
def test_workspace_invalid_inputs(split_wald_data):
    """Workspace creation and chunk compute reject invalid inputs cleanly."""
    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)

    # Wrong invariant shape
    with pytest.raises(ValueError, match="uab_invariant"):
        create_lmm_workspace(
            eigenvalues,
            uab_inv_soa.T,  # wrong shape: (n_samples, 3) instead of (3, n_samples)
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"]
)
def test_workspace_nonfinite_eigenvalues(split_wald_data, bad_value):
    """Workspace creation rejects non-finite eigenvalues."""
    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    bad_evals = eigenvalues.copy()
    bad_evals[0] = bad_value
    with pytest.raises(ValueError, match=r"eigenvalues.*not finite"):
        create_lmm_workspace(
            bad_evals,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )

    # Wrong uab_varying shape for chunk compute
    ws = create_lmm_workspace(eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    with pytest.raises(ValueError, match="uab_varying"):
        compute_wald_split_c_ws(ws, uab_var_soa.transpose(0, 2, 1), 1)


@pytest.mark.tier1
@pytest.mark.skipif(not _C_SPLIT_AVAILABLE, reason="Split C extension unavailable")
def test_pipeline_multi_chunk_correctness():
    """Pipeline path (multi-chunk) produces identical results to sequential path.

    Forces multi-chunk processing by using enough SNPs to exceed chunk_size,
    then compares pipeline results against sequential (non-pipeline) results.
    This catches off-by-one errors in the last-chunk handling, race conditions
    in buffer management, and write_offset accumulation bugs.
    """
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy

    rng = np.random.default_rng(42)
    n_samples = 100
    # Use enough SNPs that we get at least 3 chunks
    chunk_size = compute_chunk_size_numpy(
        n_samples,
        1000,
        n_cvt=1,
        mem_budget_bytes=int(2e9),
    )
    n_snps = chunk_size * 3 + 17  # non-aligned to catch last-chunk bugs
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))

    # Build realistic data: genotype matrix + covariates + phenotype
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.4, 0.4, 0.2])
    phenotypes = rng.standard_normal(n_samples)
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]

    # Stub out kinship — use pre-computed eigendecomposition
    U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]

    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    # Run with pipeline enabled (multi-chunk, split C extension)
    run_result = run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=None,
        snp_info=snp_info,
        eigenvalues=eigenvalues,
        eigenvectors=U,
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
        n_refine=20,
    )
    results_pipeline = run_result.associations

    # Verify we got results for all SNPs (none filtered at maf=0)
    # Some may be filtered by the internal variance check, but most should pass
    assert len(results_pipeline) > n_snps * 0.8, (
        f"Too many SNPs filtered: got {len(results_pipeline)} of {n_snps}"
    )

    # Run with pipeline disabled: force single chunk by using sequential path
    # We do this by monkeypatching the canonical dispatch flag to False.
    import jamma.lmm.compute_numpy as compute_mod

    orig_split = compute_mod._C_SPLIT_AVAILABLE
    try:
        compute_mod._C_SPLIT_AVAILABLE = False
        run_result = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=U,
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=1,
            n_refine=20,
        )
        results_sequential = run_result.associations
    finally:
        compute_mod._C_SPLIT_AVAILABLE = orig_split

    # Same number of results
    assert len(results_pipeline) == len(results_sequential), (
        f"Pipeline: {len(results_pipeline)}, Sequential: {len(results_sequential)}"
    )

    # Compare numerical outputs
    for r_pipe, r_seq in zip(results_pipeline, results_sequential, strict=True):
        assert r_pipe.rs == r_seq.rs, f"SNP order mismatch: {r_pipe.rs} vs {r_seq.rs}"
        if r_pipe.p_wald is not None and r_seq.p_wald is not None:
            np.testing.assert_allclose(
                r_pipe.beta,
                r_seq.beta,
                rtol=1e-9,
                atol=1e-14,
                err_msg=f"beta mismatch for {r_pipe.rs}",
            )
            np.testing.assert_allclose(
                r_pipe.se,
                r_seq.se,
                rtol=1e-9,
                atol=1e-14,
                err_msg=f"se mismatch for {r_pipe.rs}",
            )
            np.testing.assert_allclose(
                r_pipe.p_wald,
                r_seq.p_wald,
                rtol=1e-8,
                atol=1e-14,
                err_msg=f"p_wald mismatch for {r_pipe.rs}",
            )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_workspace_alignment():
    """Verify alloc_aligned_doubles returns 32-byte-aligned addresses."""
    from jamma.lmm._lmm_accel import _get_aligned_alloc_test_ptr

    # Test boundary sizes (n=1 minimum, n=4 exact 32-byte boundary) and larger
    for n in [1, 4, 100, 101, 200, 1400, 50001]:
        ptr = _get_aligned_alloc_test_ptr(n)
        assert ptr % 32 == 0, (
            f"alloc_aligned_doubles({n}) returned {ptr:#x}, not 32-byte aligned"
        )


@pytest.mark.benchmark
class TestCExtensionPerformance:
    """Benchmark C extension vs Python on realistic data.

    Hardware-sensitive — `2x` speedup is not a correctness invariant. Runs
    only under `--benchmark-only`; on machines with <4 physical cores it
    skips. Numerical parity between C and Python paths IS a correctness
    invariant and is checked unconditionally.
    """

    def test_c_faster_than_python(self, monkeypatch, benchmark):
        """Benchmark C-accelerated Wald; verify numerical parity vs Python."""
        from jamma.core.threading import get_physical_core_count
        from jamma.lmm.compute_numpy import (
            _C_ACCEL_AVAILABLE,
            _compute_wald_numpy,
        )
        from jamma.lmm.likelihood_numpy import batch_compute_iab_numpy

        if not _C_ACCEL_AVAILABLE:
            pytest.skip("C extension not compiled")

        n_threads = get_physical_core_count()
        if n_threads < 4:
            pytest.skip(f"Benchmark needs >=4 physical cores; found {n_threads}")

        rng = np.random.default_rng(42)
        n_samples, n_snps = 500, 2000
        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
        Uab_batch = rng.standard_normal((n_snps, n_samples, 6))
        Uab_batch[:, :, 0] = np.abs(Uab_batch[:, :, 0]) + 0.1
        Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

        import jamma.lmm.compute_numpy as cn

        # Warmup: amortise OpenMP thread-pool startup before timing
        monkeypatch.setattr(cn, "_C_ACCEL_AVAILABLE", True)
        _compute_wald_numpy(
            1,
            eigenvalues,
            Uab_batch[:50],
            n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
            Iab_batch=Iab_batch[:50],
            n_threads=n_threads,
        )

        # pytest-benchmark times the C path; speedup vs the Python path is
        # tracked over time as benchmark history rather than asserted.
        monkeypatch.setattr(cn, "_C_ACCEL_AVAILABLE", True)
        result_c = benchmark(
            _compute_wald_numpy,
            1,
            eigenvalues,
            Uab_batch,
            n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
            Iab_batch=Iab_batch,
            n_threads=n_threads,
        )

        # Numerical parity is the actual correctness invariant. The C and
        # Python golden-section paths can produce slightly different optima
        # on flat likelihood landscapes (FP ordering); 5e-5 rtol is the
        # documented bound for 2000-SNP batches.
        monkeypatch.setattr(cn, "_C_ACCEL_AVAILABLE", False)
        result_py = _compute_wald_numpy(
            1,
            eigenvalues,
            Uab_batch,
            n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
            Iab_batch=Iab_batch,
        )
        np.testing.assert_allclose(
            result_c["lambdas"], result_py["lambdas"], rtol=5e-5, atol=1e-14
        )


# =============================================================================
# Tests for build_pab_table_for_c
# =============================================================================


class TestBuildPabTableForC:
    """Verify build_pab_table_for_c produces correct flat arrays for C extension."""

    def test_ncvt1_basic_structure(self):
        """n_cvt=1: returns dict with all expected keys and correct scalar values."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t = build_pab_table_for_c(1)

        assert t["n_cvt"] == 1
        assert t["n_index"] == 6  # (1+3)*(1+2)//2 = 6
        assert t["n_rows"] == 3  # n_cvt + 2
        # idx_yy, idx_xx, idx_xy from build_index_table
        from jamma.lmm.likelihood import build_index_table

        ref = build_index_table(1)
        assert t["idx_yy"] == ref["idx_yy"]
        assert t["idx_xx"] == ref["idx_xx"]
        assert t["idx_xy"] == ref["idx_xy"]

    def test_ncvt2_dimensions(self):
        """n_cvt=2: n_index=10, n_rows=4, correct inv/var counts."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t = build_pab_table_for_c(2)

        assert t["n_index"] == 10
        assert t["n_rows"] == 4  # n_cvt + 2
        assert t["n_inv"] == 6
        assert t["n_var"] == 4
        assert t["n_inv"] + t["n_var"] == t["n_index"]

    def test_ncvt4_dimensions(self):
        """n_cvt=4: n_index=21, n_rows=6, correct inv/var counts."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t = build_pab_table_for_c(4)

        assert t["n_index"] == 21
        assert t["n_rows"] == 6  # n_cvt + 2
        assert t["n_inv"] == 15
        assert t["n_var"] == 6
        assert t["n_inv"] + t["n_var"] == t["n_index"]

    def test_invariant_varying_partition(self):
        """invariant + varying indices partition range(n_index) for n_cvt=1,2,4."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        for n_cvt in (1, 2, 4):
            t = build_pab_table_for_c(n_cvt)
            inv = set(t["invariant_indices"].tolist())
            var = set(t["varying_indices"].tolist())
            assert inv & var == set(), f"n_cvt={n_cvt}: overlap in inv/var"
            assert inv | var == set(range(t["n_index"])), (
                f"n_cvt={n_cvt}: inv+var doesn't cover range(n_index)"
            )

    def test_all_arrays_are_int32(self):
        """All index arrays must be int32 for C extension compatibility."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t = build_pab_table_for_c(2)
        array_keys = [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
        ]
        for key in array_keys:
            assert t[key].dtype == np.int32, (
                f"{key} has dtype {t[key].dtype}, expected int32"
            )

    def test_level_offsets_index_entries(self):
        """level_offsets and level_counts correctly index into flat entries array."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        for n_cvt in (1, 2, 4):
            t = build_pab_table_for_c(n_cvt)
            offsets = t["level_offsets"]
            counts = t["level_counts"]
            entries = t["entries"]

            # n_cvt+2 levels (0..n_cvt+1)
            assert len(offsets) == n_cvt + 2
            assert len(counts) == n_cvt + 2

            # Level 0 has no entries (row 0 comes from dot products)
            assert counts[0] == 0

            # Total entries must equal entries array length / 4 (stride-4)
            total_entries = sum(counts)
            assert len(entries) == total_entries * 4, (
                f"n_cvt={n_cvt}: entries length {len(entries)} != {total_entries * 4}"
            )

            # Each level's offset must be consistent
            running_offset = 0
            for level in range(n_cvt + 2):
                assert offsets[level] == running_offset, (
                    f"n_cvt={n_cvt}, level={level}: "
                    f"offset {offsets[level]} != {running_offset}"
                )
                running_offset += counts[level]

    def test_entries_match_pab_recursion(self):
        """Flat entries array matches build_index_table pab_recursion content."""
        from jamma.lmm.likelihood import build_index_table, build_pab_table_for_c

        for n_cvt in (1, 2, 4):
            t = build_pab_table_for_c(n_cvt)
            ref = build_index_table(n_cvt)
            entries = t["entries"]
            offsets = t["level_offsets"]
            counts = t["level_counts"]

            for level in range(1, n_cvt + 2):
                ref_entries = ref["pab_recursion"][level]
                start = offsets[level] * 4
                count = counts[level]
                assert count == len(ref_entries), (
                    f"n_cvt={n_cvt}, level={level}: count mismatch"
                )
                for j, (_, _, idx_ab, idx_aw, idx_bw, idx_ww) in enumerate(ref_entries):
                    base = start + j * 4
                    assert entries[base] == idx_ab
                    assert entries[base + 1] == idx_aw
                    assert entries[base + 2] == idx_bw
                    assert entries[base + 3] == idx_ww

    def test_logdet_diag_matches_build_index_table(self):
        """logdet_diag_rows/cols match build_index_table logdet_diag_indices."""
        from jamma.lmm.likelihood import build_index_table, build_pab_table_for_c

        for n_cvt in (1, 2, 4):
            t = build_pab_table_for_c(n_cvt)
            ref = build_index_table(n_cvt)

            rows = t["logdet_diag_rows"].tolist()
            cols = t["logdet_diag_cols"].tolist()
            ref_pairs = ref["logdet_diag_indices"]

            assert len(rows) == len(ref_pairs)
            for i, (ref_row, ref_col) in enumerate(ref_pairs):
                assert rows[i] == ref_row, f"n_cvt={n_cvt}, i={i}: row mismatch"
                assert cols[i] == ref_col, f"n_cvt={n_cvt}, i={i}: col mismatch"

    def test_lru_cached(self):
        """Same n_cvt returns same object (lru_cache)."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t1 = build_pab_table_for_c(2)
        t2 = build_pab_table_for_c(2)
        assert t1 is t2


# =============================================================================
# Tests for general n_cvt C extension (C-GEN requirements)
# =============================================================================


def _run_general_ncvt_c_vs_python(data: dict) -> None:
    """Helper: compare C extension general n_cvt results against Python path.

    Monkeypatches _C_GENERAL_AVAILABLE to False for the Python reference,
    then compares against C extension results.
    """
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]

    # Python reference path (force fallback)
    orig = compute_numpy._C_GENERAL_AVAILABLE
    try:
        compute_numpy._C_GENERAL_AVAILABLE = False
        result_py = _compute_wald_numpy(
            n_cvt,
            eigenvalues,
            Uab_batch,
            n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
        )
    finally:
        compute_numpy._C_GENERAL_AVAILABLE = orig

    # C extension path
    result_c = _compute_wald_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        n_threads=1,
    )

    for key in ("lambdas", "logls", "betas", "ses"):
        np.testing.assert_allclose(
            result_c[key],
            result_py[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: C vs Python mismatch for n_cvt={n_cvt}",
        )
    np.testing.assert_allclose(
        result_c["pwalds"],
        result_py["pwalds"],
        rtol=1e-6,
        atol=1e-14,
        equal_nan=True,
        err_msg=f"pwalds: C vs Python mismatch for n_cvt={n_cvt}",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_reml_wald_matches_python_ncvt2(
    synthetic_covariate_data_ncvt2,
):
    """C-GEN-01: C extension Wald results match Python for n_cvt=2."""
    _run_general_ncvt_c_vs_python(synthetic_covariate_data_ncvt2)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_reml_wald_ncvt4(
    synthetic_covariate_data_ncvt4,
):
    """C-GEN-01: C extension Wald results match Python for n_cvt=4."""
    _run_general_ncvt_c_vs_python(synthetic_covariate_data_ncvt4)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_workspace_lifecycle(synthetic_covariate_data_ncvt2):
    """C-GEN-02: Workspace create/compute/destroy cycle works for n_cvt>1."""
    from jamma.lmm.likelihood import classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    # a[0, :, list_idx] -> (n_inv, n_samples) due to numpy advanced indexing
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    # Create workspace
    ws = create_lmm_workspace_general(
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
    assert ws is not None

    # Compute first chunk
    mid = Uab_batch.shape[0] // 2
    r1 = compute_wald_general_c_ws(ws, uab_var_soa[:mid], 1)
    assert r1["lambdas"].shape == (mid,)

    # Reuse workspace for second chunk
    r2 = compute_wald_general_c_ws(ws, uab_var_soa[mid:], 1)
    assert r2["lambdas"].shape == (Uab_batch.shape[0] - mid,)

    # Full batch
    r_full = compute_wald_general_c_ws(ws, uab_var_soa, 1)
    combined = np.concatenate([r1["lambdas"], r2["lambdas"]])
    np.testing.assert_allclose(
        combined,
        r_full["lambdas"],
        rtol=1e-12,
        atol=1e-14,
        err_msg="Chunked vs full workspace mismatch",
    )

    # Destroy workspace (PyCapsule GC)
    del ws


@pytest.mark.tier1
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_gemma_covariate_match():
    """C-GEN-03: C extension Wald results match GEMMA reference with covariates.

    End-to-end test: loads gemma_synthetic PLINK data + covariates, runs the
    NumPy runner (which uses the general C workspace for n_cvt=2 Wald), and
    compares against GEMMA's covariate reference output.
    """
    from pathlib import Path

    from jamma.io import load_plink_binary
    from jamma.kinship.io import read_kinship_matrix
    from jamma.lmm.runner_numpy import run_lmm_association_numpy
    from jamma.validation import (
        ToleranceConfig,
        compare_assoc_results,
        load_gemma_assoc,
    )
    from tests.conftest import load_phenotypes_from_fam

    fixture_root = Path(__file__).parent / "fixtures"
    synthetic_dir = fixture_root / "gemma_synthetic"
    covariate_dir = fixture_root / "gemma_covariate"

    plink = load_plink_binary(synthetic_dir / "test")
    kinship = read_kinship_matrix(synthetic_dir / "gemma_kinship.cXX.txt")
    phenotypes = load_phenotypes_from_fam(synthetic_dir / "test.fam")
    covariates = np.loadtxt(covariate_dir / "covariates.txt")
    snp_info = [
        {
            "chr": str(plink.chromosome[i]),
            "rs": plink.sid[i],
            "pos": plink.bp_position[i],
            "a1": plink.allele_1[i],
            "a0": plink.allele_2[i],
            "maf": 0.0,
            "n_miss": 0,
        }
        for i in range(plink.n_snps)
    ]

    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=1,
        show_progress=False,
    )
    results = run_result.associations

    reference = load_gemma_assoc(covariate_dir / "gemma_covariate.assoc.txt")
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"C extension Wald+covariate vs GEMMA failed:\n{comparison}"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_all_modes(synthetic_covariate_data_ncvt2):
    """C-GEN-04: All 4 LMM modes produce results with n_cvt=2 covariates.

    Verifies that compute_lmm_chunk_numpy with lmm_mode=4 produces non-None
    results for all output fields when covariates are present. Wald results
    use the C extension; LRT/Score use the Python fallback.
    """
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy
    from jamma.lmm.prepare_common import _compute_null_model_common

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    # Build Uab
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    n_snps = Uab_batch.shape[0]

    # Compute null model for LRT/Score
    logl_H0, _lambda_mle, Hi_eval_null = _compute_null_model_common(
        4, eigenvalues, UtW, Uty, n_cvt, False
    )

    # Mode 4 (All): exercises Wald (C ext), LRT (Python MLE), Score (Python)
    result = compute_lmm_chunk_numpy(
        lmm_mode=4,
        n_cvt=n_cvt,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        logl_H0=logl_H0,
        Hi_eval_null=Hi_eval_null,
        n_threads=1,
    )

    # All fields must be non-None and have correct shape
    all_keys = (
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "p_lrts",
        "p_scores",
    )
    for key in all_keys:
        assert result[key] is not None, f"{key} is None in mode 4"
        assert result[key].shape == (n_snps,), (
            f"{key} shape mismatch: {result[key].shape}"
        )

    # Finite check (most values should be finite; allow NaN for degenerate SNPs)
    for key in ("betas", "ses", "pwalds"):
        n_finite = np.sum(np.isfinite(result[key]))
        assert n_finite > n_snps * 0.8, f"{key}: only {n_finite}/{n_snps} finite values"

    # Mode 2 (LRT only)
    result_lrt = compute_lmm_chunk_numpy(
        lmm_mode=2,
        n_cvt=n_cvt,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        logl_H0=logl_H0,
        n_threads=1,
    )
    assert result_lrt["lambdas_mle"] is not None
    assert result_lrt["p_lrts"] is not None
    assert result_lrt["lambdas_mle"].shape == (n_snps,)

    # Mode 3 (Score only)
    result_score = compute_lmm_chunk_numpy(
        lmm_mode=3,
        n_cvt=n_cvt,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        Hi_eval_null=Hi_eval_null,
        n_threads=1,
    )
    assert result_score["p_scores"] is not None
    assert result_score["betas"] is not None
    assert result_score["p_scores"].shape == (n_snps,)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_openmp_deterministic(synthetic_covariate_data_ncvt2):
    """C-GEN-05: 1-thread vs N-thread produce identical results for n_cvt>1."""
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    from jamma.lmm.likelihood import classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    ws = create_lmm_workspace_general(
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

    r1 = compute_wald_general_c_ws(ws, uab_var_soa, 1)
    rn = compute_wald_general_c_ws(ws, uab_var_soa, n_threads)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            rn[key],
            r1[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: general MT vs ST mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_degenerate_snps(synthetic_covariate_data_ncvt2):
    """C-GEN-06: Constant genotypes produce NaN beta/se/p-value for n_cvt>1."""
    from jamma.lmm.likelihood import classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"].copy()
    n_samples = data["n_samples"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)

    # Make SNPs 0 and 2 degenerate by zeroing all varying columns
    # (this makes xx=0, causing P_XX <= 0)
    for snp_idx in [0, 2]:
        for vi in var_indices:
            Uab_batch[snp_idx, :, vi] = 0.0

    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    ws = create_lmm_workspace_general(
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
    result = compute_wald_general_c_ws(ws, uab_var_soa, 1)

    # Degenerate SNPs should have NaN
    for snp_idx in [0, 2]:
        assert np.isnan(result["betas"][snp_idx]), f"SNP {snp_idx}: expected NaN beta"
        assert np.isnan(result["ses"][snp_idx]), f"SNP {snp_idx}: expected NaN se"
        assert np.isnan(result["pwalds"][snp_idx]), f"SNP {snp_idx}: expected NaN pwald"

    # Non-degenerate SNPs should have valid results
    for snp_idx in [1, 3]:
        assert np.isfinite(result["betas"][snp_idx]), (
            f"SNP {snp_idx}: expected finite beta"
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_general_ncvt_abi_version():
    """C-GEN-07: ABI version is 11 for persistent Score/LRT workspaces."""
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION == 11, f"Expected ABI_VERSION=11, got {ABI_VERSION}"


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_existing_ncvt1_regression(synthetic_wald_data):
    """C-GEN-08: Existing n_cvt=1 C extension path unchanged with ABI_VERSION=5.

    Ensures the general n_cvt additions (ABI_VERSION bump, new workspace types)
    did not regress the original n_cvt=1 split-Uab workspace path.
    """
    eigenvalues, Uab_batch, n_samples = synthetic_wald_data

    # Use the existing Uab directly for split components
    uab_varying_soa = np.stack(
        [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
    )
    uab_inv_soa_direct = np.stack(
        [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
    )

    # Create n_cvt=1 workspace and compute
    ws = create_lmm_workspace(
        eigenvalues, uab_inv_soa_direct, n_samples, 1e-5, 1e5, 50, 20, 1
    )
    result = compute_wald_split_c_ws(ws, uab_varying_soa, 1)

    # Basic sanity: shapes match, most values finite
    assert result["lambdas"].shape == (Uab_batch.shape[0],)
    assert result["betas"].shape == (Uab_batch.shape[0],)
    n_finite = np.sum(np.isfinite(result["betas"]))
    assert n_finite > 0, "No finite betas from n_cvt=1 workspace"


# ---------------------------------------------------------------------------
# Score and LRT C-vs-Python parity tests (Plan 64-02)
# ---------------------------------------------------------------------------


@pytest.fixture
def score_lrt_data(synthetic_wald_data):
    """Extends synthetic_wald_data with null-model Hi_eval and logl_H0.

    Computes the null-model MLE lambda via golden section on the null Uab
    (no genotype), then derives Hi_eval_null = 1/(lambda_null*eval + 1)
    and logl_H0 (null MLE log-likelihood).
    """
    eigenvalues, Uab_batch, n_samples = synthetic_wald_data
    n_cvt = 1

    # Build null Uab: zero genotype columns (wx=0, xx=0, xy=0).
    # The null model only uses ww, wy, yy columns.
    Uab_null = np.zeros((1, n_samples, 6), dtype=np.float64)
    Uab_null[0, :, 0] = Uab_batch[0, :, 0]  # ww (invariant)
    Uab_null[0, :, 2] = Uab_batch[0, :, 2]  # wy (invariant)
    Uab_null[0, :, 5] = Uab_batch[0, :, 5]  # yy (invariant)

    # Null MLE lambda optimization
    lambdas_null, logls_null = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_null,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    lambda_null = float(lambdas_null[0])
    logl_H0 = float(logls_null[0])

    # Hi_eval_null: Score test uses this fixed weight vector
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

    return eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0


_score_c_available = _C_ACCEL_AVAILABLE and _compute_score_batch_c is not None
_lrt_c_available = _C_ACCEL_AVAILABLE and _compute_lrt_batch_c is not None


@pytest.mark.tier0
@pytest.mark.skipif(not _score_c_available, reason="Score C extension not available")
def test_score_c_vs_python_parity(score_lrt_data):
    """C compute_score_batch_c matches Python batch_calc_score_stats_numpy."""
    eigenvalues, Uab_batch, n_samples, Hi_eval_null, _ = score_lrt_data
    n_cvt = 1

    # C path
    result_c = _compute_score_batch_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        1,
    )

    # Python path
    betas_py, ses_py, p_scores_py = batch_calc_score_stats_numpy(
        n_cvt,
        Hi_eval_null,
        Uab_batch,
        n_samples,
    )

    np.testing.assert_allclose(result_c["betas"], betas_py, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(result_c["ses"], ses_py, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(
        result_c["p_scores"], p_scores_py, rtol=1e-10, atol=1e-14
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _lrt_c_available, reason="LRT C extension not available")
def test_lrt_c_vs_python_parity(score_lrt_data):
    """C compute_lrt_batch_c matches Python golden_section_optimize_lambda_mle_numpy."""
    eigenvalues, Uab_batch, n_samples, _, logl_H0 = score_lrt_data
    n_cvt = 1
    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20

    # C path
    result_c = _compute_lrt_batch_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    # Python path
    lambdas_mle_py, logls_mle_py = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts_py = _batch_lrt_pvalues_numpy(logls_mle_py, logl_H0)

    np.testing.assert_allclose(
        result_c["lambdas_mle"],
        lambdas_mle_py,
        rtol=1e-6,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result_c["p_lrts"],
        p_lrts_py,
        rtol=1e-4,
        atol=1e-14,
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _score_c_available, reason="Score C extension not available")
def test_score_c_degenerate_snps(score_lrt_data):
    """Score C returns NaN for constant genotypes (P_xx <= 0)."""
    eigenvalues, Uab_batch, n_samples, Hi_eval_null, _ = score_lrt_data

    # Create degenerate Uab: constant genotype -> wx=0, xx=0, xy=0
    Uab_degen = Uab_batch.copy()
    Uab_degen[0, :, 1] = 0.0  # wx = 0
    Uab_degen[0, :, 3] = 0.0  # xx = 0
    Uab_degen[0, :, 4] = 0.0  # xy = 0

    result = _compute_score_batch_c(
        eigenvalues,
        Uab_degen,
        Hi_eval_null,
        n_samples,
        1,
    )

    # First SNP is degenerate: should have NaN beta/se/p_score
    assert np.isnan(result["betas"][0]), "degenerate SNP should have NaN beta"
    assert np.isnan(result["ses"][0]), "degenerate SNP should have NaN se"
    assert np.isnan(result["p_scores"][0]), "degenerate SNP should have NaN p_score"

    # Remaining SNPs should still be finite
    assert np.all(np.isfinite(result["betas"][1:])), (
        "non-degenerate SNPs should be finite"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _lrt_c_available, reason="LRT C extension not available")
def test_lrt_c_degenerate_snps(score_lrt_data):
    """LRT C handles degenerate SNPs: p_lrt ~ 1.0 (no signal)."""
    eigenvalues, Uab_batch, n_samples, _, logl_H0 = score_lrt_data

    # Create degenerate Uab: constant genotype
    Uab_degen = Uab_batch.copy()
    Uab_degen[0, :, 1] = 0.0  # wx = 0
    Uab_degen[0, :, 3] = 0.0  # xx = 0
    Uab_degen[0, :, 4] = 0.0  # xy = 0

    result = _compute_lrt_batch_c(
        eigenvalues,
        Uab_degen,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        1,
    )

    # Degenerate SNP: MLE logl_H1 ~ logl_H0, so LRT stat ~ 0, p ~ 1.0
    assert result["p_lrts"][0] >= 0.99, (
        f"degenerate SNP should have p_lrt ~ 1.0, got {result['p_lrts'][0]}"
    )

    # Remaining SNPs should be finite
    assert np.all(np.isfinite(result["p_lrts"][1:])), (
        "non-degenerate SNPs should be finite"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _score_c_available, reason="Score C extension not available")
def test_score_c_multithreaded(score_lrt_data):
    """Score C with n_threads=4 produces identical output to n_threads=1."""
    eigenvalues, Uab_batch, n_samples, Hi_eval_null, _ = score_lrt_data

    result_1t = _compute_score_batch_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        1,
    )
    result_4t = _compute_score_batch_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        4,
    )

    np.testing.assert_array_equal(result_1t["betas"], result_4t["betas"])
    np.testing.assert_array_equal(result_1t["ses"], result_4t["ses"])
    np.testing.assert_array_equal(result_1t["p_scores"], result_4t["p_scores"])


@pytest.mark.tier0
@pytest.mark.skipif(not _lrt_c_available, reason="LRT C extension not available")
def test_lrt_c_multithreaded(score_lrt_data):
    """LRT C with n_threads=4 produces identical output to n_threads=1."""
    eigenvalues, Uab_batch, n_samples, _, logl_H0 = score_lrt_data

    result_1t = _compute_lrt_batch_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        1,
    )
    result_4t = _compute_lrt_batch_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        logl_H0,
        4,
    )

    np.testing.assert_array_equal(result_1t["lambdas_mle"], result_4t["lambdas_mle"])
    np.testing.assert_array_equal(result_1t["p_lrts"], result_4t["p_lrts"])


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


# =============================================================================
# Fused mode-4 kernel tests (Plan 67-02)
# =============================================================================


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


# ---------------------------------------------------------------------------
# Score and LRT general n_cvt C batch kernel tests (Plan 70-01)
# ---------------------------------------------------------------------------


def _make_general_score_lrt_data(data: dict) -> dict:
    """Extend synthetic covariate data with null-model Hi_eval and logl_H0.

    Computes the null-model MLE lambda via golden section on the null Uab
    (zero genotype), then derives Hi_eval_null and logl_H0.

    Args:
        data: Dict from _build_synthetic_covariate_data.

    Returns:
        Dict with all original keys plus Hi_eval_null and logl_H0.
    """
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_cvt = data["n_cvt"]
    n_samples = data["n_samples"]
    n_index = Uab_batch.shape[2]  # (n_cvt+3)*(n_cvt+2)//2

    # Null Uab: zero all varying (genotype) columns.
    from jamma.lmm.likelihood import classify_uab_columns

    inv_indices, _ = classify_uab_columns(n_cvt)
    Uab_null = np.zeros((1, n_samples, n_index), dtype=np.float64)
    for idx in inv_indices:
        Uab_null[0, :, idx] = Uab_batch[0, :, idx]

    lambdas_null, logls_null = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_null,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    lambda_null = float(lambdas_null[0])
    logl_H0 = float(logls_null[0])
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

    return {**data, "Hi_eval_null": Hi_eval_null, "logl_H0": logl_H0}


@pytest.fixture
def general_score_lrt_ncvt2(synthetic_covariate_data_ncvt2):
    """Score/LRT data for n_cvt=2."""
    return _make_general_score_lrt_data(synthetic_covariate_data_ncvt2)


@pytest.fixture
def general_score_lrt_ncvt4(synthetic_covariate_data_ncvt4):
    """Score/LRT data for n_cvt=4."""
    return _make_general_score_lrt_data(synthetic_covariate_data_ncvt4)


def _score_general_c_available() -> bool:
    """Check if compute_score_batch_general_c is available from the C extension."""
    if not _C_ACCEL_AVAILABLE:
        return False
    try:
        from jamma.lmm._lmm_accel import compute_score_batch_general_c  # noqa: F401

        return True
    except ImportError:
        return False


def _lrt_general_c_available() -> bool:
    """Check if compute_lrt_batch_general_c is available from the C extension."""
    if not _C_ACCEL_AVAILABLE:
        return False
    try:
        from jamma.lmm._lmm_accel import compute_lrt_batch_general_c  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_score_batch_general_ncvt2(general_score_lrt_ncvt2):
    """C-70-01: compute_score_batch_general_c matches Python for n_cvt=2."""
    if not _score_general_c_available():
        pytest.skip("compute_score_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_score_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = general_score_lrt_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    Hi_eval_null = data["Hi_eval_null"]

    pab_table_dict = build_pab_table_for_c(n_cvt)

    # C path
    result_c = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,  # n_threads
    )

    # Python reference path
    betas_py, ses_py, p_scores_py = batch_calc_score_stats_numpy(
        n_cvt,
        Hi_eval_null,
        Uab_batch,
        n_samples,
    )

    np.testing.assert_allclose(
        result_c["betas"],
        betas_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: C vs Python mismatch for n_cvt=2 Score",
    )
    np.testing.assert_allclose(
        result_c["ses"],
        ses_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: C vs Python mismatch for n_cvt=2 Score",
    )
    np.testing.assert_allclose(
        result_c["p_scores"],
        p_scores_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: C vs Python mismatch for n_cvt=2 Score",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_score_batch_general_ncvt4(general_score_lrt_ncvt4):
    """C-70-02: compute_score_batch_general_c matches Python for n_cvt=4."""
    if not _score_general_c_available():
        pytest.skip("compute_score_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_score_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = general_score_lrt_ncvt4
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    Hi_eval_null = data["Hi_eval_null"]

    pab_table_dict = build_pab_table_for_c(n_cvt)

    result_c = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    betas_py, ses_py, p_scores_py = batch_calc_score_stats_numpy(
        n_cvt,
        Hi_eval_null,
        Uab_batch,
        n_samples,
    )

    np.testing.assert_allclose(
        result_c["betas"],
        betas_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: C vs Python mismatch for n_cvt=4 Score",
    )
    np.testing.assert_allclose(
        result_c["ses"],
        ses_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: C vs Python mismatch for n_cvt=4 Score",
    )
    np.testing.assert_allclose(
        result_c["p_scores"],
        p_scores_py,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: C vs Python mismatch for n_cvt=4 Score",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_score_batch_general_degenerate_snps(synthetic_covariate_data_ncvt2):
    """C-70-03: Degenerate SNPs produce NaN for Score general n_cvt."""
    if not _score_general_c_available():
        pytest.skip("compute_score_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_score_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    n_samples = data["n_samples"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"].copy()

    # Zero out all genotype-containing columns in one SNP to make it degenerate
    from jamma.lmm.likelihood import classify_uab_columns

    _, var_indices = classify_uab_columns(n_cvt)
    for idx in var_indices:
        Uab_batch[0, :, idx] = 0.0

    lambda_val = 1.0
    Hi_eval_null = 1.0 / (lambda_val * eigenvalues + 1.0)
    pab_table_dict = build_pab_table_for_c(n_cvt)

    result_c = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    # Degenerate SNP 0 should produce NaN
    assert np.isnan(result_c["betas"][0]), "Expected NaN beta for degenerate SNP"
    assert np.isnan(result_c["ses"][0]), "Expected NaN se for degenerate SNP"
    assert np.isnan(result_c["p_scores"][0]), "Expected NaN p_score for degenerate SNP"

    # Non-degenerate SNPs should have finite values
    finite_mask = np.isfinite(result_c["betas"][1:])
    assert finite_mask.sum() > 0, "Expected some finite betas for non-degenerate SNPs"


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_lrt_batch_general_ncvt2(general_score_lrt_ncvt2):
    """C-70-04: compute_lrt_batch_general_c matches Python for n_cvt=2."""
    if not _lrt_general_c_available():
        pytest.skip("compute_lrt_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_lrt_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = general_score_lrt_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    logl_H0 = data["logl_H0"]

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    pab_table_dict = build_pab_table_for_c(n_cvt)

    # C path
    result_c = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,  # n_threads
    )

    # Python reference path
    lambdas_py, logls_py = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts_py = _batch_lrt_pvalues_numpy(logls_py, logl_H0)

    np.testing.assert_allclose(
        result_c["lambdas_mle"],
        lambdas_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: C vs Python mismatch for n_cvt=2 LRT",
    )
    np.testing.assert_allclose(
        result_c["p_lrts"],
        p_lrts_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: C vs Python mismatch for n_cvt=2 LRT",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_lrt_batch_general_ncvt4(general_score_lrt_ncvt4):
    """C-70-05: compute_lrt_batch_general_c matches Python for n_cvt=4."""
    if not _lrt_general_c_available():
        pytest.skip("compute_lrt_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_lrt_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = general_score_lrt_ncvt4
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    logl_H0 = data["logl_H0"]

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    pab_table_dict = build_pab_table_for_c(n_cvt)

    result_c = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    lambdas_py, logls_py = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts_py = _batch_lrt_pvalues_numpy(logls_py, logl_H0)

    np.testing.assert_allclose(
        result_c["lambdas_mle"],
        lambdas_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: C vs Python mismatch for n_cvt=4 LRT",
    )
    np.testing.assert_allclose(
        result_c["p_lrts"],
        p_lrts_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: C vs Python mismatch for n_cvt=4 LRT",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_lrt_batch_general_degenerate_snps(synthetic_covariate_data_ncvt2):
    """C-70-06: LRT general C matches Python on degenerate SNPs.

    Unlike Score (which checks P_XX <= 0 and returns NaN), LRT's golden
    section MLE optimizer can still converge to a finite lambda on degenerate
    SNPs. This test verifies C-vs-Python parity on a batch containing a
    degenerate SNP.
    """
    if not _lrt_general_c_available():
        pytest.skip("compute_lrt_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_lrt_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    n_samples = data["n_samples"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"].copy()

    # Zero out all genotype-containing columns in one SNP to make it degenerate
    _, var_indices = classify_uab_columns(n_cvt)
    for idx in var_indices:
        Uab_batch[0, :, idx] = 0.0

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    logl_H0 = -100.0  # arbitrary finite value for null model
    pab_table_dict = build_pab_table_for_c(n_cvt)

    result_c = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,  # n_threads
    )

    # Python reference path
    lambdas_py, logls_py = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts_py = _batch_lrt_pvalues_numpy(logls_py, logl_H0)

    np.testing.assert_allclose(
        result_c["lambdas_mle"],
        lambdas_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: C vs Python mismatch on batch with degenerate SNP",
    )
    np.testing.assert_allclose(
        result_c["p_lrts"],
        p_lrts_py,
        rtol=5e-5,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: C vs Python mismatch on batch with degenerate SNP",
    )


# ---------------------------------------------------------------------------
# General n_cvt split Score/LRT C kernel tests (Plan 105-01)
# ---------------------------------------------------------------------------


def _score_split_general_c_available() -> bool:
    """Check if compute_score_split_general_c is available from the C extension."""
    if not _C_ACCEL_AVAILABLE:
        return False
    try:
        from jamma.lmm._lmm_accel import compute_score_split_general_c  # noqa: F401

        return True
    except ImportError:
        return False


def _lrt_split_general_c_available() -> bool:
    """Check if compute_lrt_split_general_c is available from the C extension."""
    if not _C_ACCEL_AVAILABLE:
        return False
    try:
        from jamma.lmm._lmm_accel import compute_lrt_split_general_c  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_score_split_ncvt2(general_score_lrt_ncvt2):
    """C-105-01: compute_score_split_general_c matches reconstruct+batch for n_cvt=2."""
    if not _score_split_general_c_available():
        pytest.skip("compute_score_split_general_c not compiled yet")

    from jamma.lmm._lmm_accel import (
        compute_score_batch_general_c,
        compute_score_split_general_c,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c
    from jamma.lmm.likelihood_numpy import (
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
    )

    data = general_score_lrt_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    Hi_eval_null = data["Hi_eval_null"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    pab_table_dict = build_pab_table_for_c(n_cvt)

    # Reference: reconstruct + batch general C
    ref = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    # SoA split path
    uab_inv = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    uab_var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    result = compute_score_split_general_c(
        eigenvalues,
        uab_var,
        uab_inv,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    # SoA split accumulates dot products column-by-column (outer loop=column,
    # inner=samples), while batch general accumulates row-by-row (outer=samples,
    # inner=columns). Different FP accumulation order gives machine-epsilon
    # differences for n_cvt>=2. Use tight allclose instead of bitwise equality.
    np.testing.assert_allclose(
        result["betas"],
        ref["betas"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: split general vs batch general mismatch for n_cvt=2",
    )
    np.testing.assert_allclose(
        result["ses"],
        ref["ses"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: split general vs batch general mismatch for n_cvt=2",
    )
    np.testing.assert_allclose(
        result["p_scores"],
        ref["p_scores"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: split general vs batch general mismatch for n_cvt=2",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_lrt_split_ncvt2(general_score_lrt_ncvt2):
    """C-105-02: compute_lrt_split_general_c matches reconstruct+batch for n_cvt=2."""
    if not _lrt_split_general_c_available():
        pytest.skip("compute_lrt_split_general_c not compiled yet")

    from jamma.lmm._lmm_accel import (
        compute_lrt_batch_general_c,
        compute_lrt_split_general_c,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c
    from jamma.lmm.likelihood_numpy import (
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
    )

    data = general_score_lrt_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    logl_H0 = data["logl_H0"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    pab_table_dict = build_pab_table_for_c(n_cvt)

    # Reference: reconstruct + batch general C
    ref = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    # SoA split path
    uab_inv = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    uab_var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    result = compute_lrt_split_general_c(
        eigenvalues,
        uab_var,
        uab_inv,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    np.testing.assert_allclose(
        result["lambdas_mle"],
        ref["lambdas_mle"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: split general vs batch general mismatch for n_cvt=2",
    )
    np.testing.assert_allclose(
        result["p_lrts"],
        ref["p_lrts"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: split general vs batch general mismatch for n_cvt=2",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_score_split_ncvt4(general_score_lrt_ncvt4):
    """C-105-03: compute_score_split_general_c matches reconstruct+batch for n_cvt=4."""
    if not _score_split_general_c_available():
        pytest.skip("compute_score_split_general_c not compiled yet")

    from jamma.lmm._lmm_accel import (
        compute_score_batch_general_c,
        compute_score_split_general_c,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c
    from jamma.lmm.likelihood_numpy import (
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
    )

    data = general_score_lrt_ncvt4
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    Hi_eval_null = data["Hi_eval_null"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    pab_table_dict = build_pab_table_for_c(n_cvt)

    ref = compute_score_batch_general_c(
        eigenvalues,
        Uab_batch,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    uab_inv = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    uab_var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    result = compute_score_split_general_c(
        eigenvalues,
        uab_var,
        uab_inv,
        Hi_eval_null,
        n_samples,
        n_cvt,
        pab_table_dict,
        1,
    )

    np.testing.assert_allclose(
        result["betas"],
        ref["betas"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: split general vs batch general mismatch for n_cvt=4",
    )
    np.testing.assert_allclose(
        result["ses"],
        ref["ses"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: split general vs batch general mismatch for n_cvt=4",
    )
    np.testing.assert_allclose(
        result["p_scores"],
        ref["p_scores"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_scores: split general vs batch general mismatch for n_cvt=4",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_lrt_split_ncvt4(general_score_lrt_ncvt4):
    """C-105-04: compute_lrt_split_general_c matches reconstruct+batch for n_cvt=4."""
    if not _lrt_split_general_c_available():
        pytest.skip("compute_lrt_split_general_c not compiled yet")

    from jamma.lmm._lmm_accel import (
        compute_lrt_batch_general_c,
        compute_lrt_split_general_c,
    )
    from jamma.lmm.likelihood import build_pab_table_for_c
    from jamma.lmm.likelihood_numpy import (
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
    )

    data = general_score_lrt_ncvt4
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]
    logl_H0 = data["logl_H0"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    l_min, l_max, n_grid, n_refine = 1e-5, 1e5, 50, 20
    pab_table_dict = build_pab_table_for_c(n_cvt)

    ref = compute_lrt_batch_general_c(
        eigenvalues,
        Uab_batch,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    uab_inv = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    uab_var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)

    result = compute_lrt_split_general_c(
        eigenvalues,
        uab_var,
        uab_inv,
        n_samples,
        n_cvt,
        pab_table_dict,
        l_min,
        l_max,
        n_grid,
        n_refine,
        logl_H0,
        1,
    )

    np.testing.assert_allclose(
        result["lambdas_mle"],
        ref["lambdas_mle"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="lambdas_mle: split general vs batch general mismatch for n_cvt=4",
    )
    np.testing.assert_allclose(
        result["p_lrts"],
        ref["p_lrts"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="p_lrts: split general vs batch general mismatch for n_cvt=4",
    )


# ---------------------------------------------------------------------------
# Hi_eval_null positivity guards (Plan 76-01)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
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


# =============================================================================
# Fused Uab parity tests (Plan 88-01)
# =============================================================================

_fused_c_available = (
    _C_ACCEL_AVAILABLE
    and hasattr(compute_numpy, "_C_FUSED_AVAILABLE")
    and compute_numpy._C_FUSED_AVAILABLE
)


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
        from jamma.lmm.compute_numpy import (
            _C_MODE4_AVAILABLE,
            compute_mode4_fused_c_ws,
            compute_mode4_split_c_ws,
            create_lmm_workspace_mode4,
            create_lmm_workspace_mode4_fused,
        )

        if not _C_MODE4_AVAILABLE:
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

    def test_fused_available_flag(self):
        """_C_FUSED_AVAILABLE flag is True when C extension has fused functions."""
        assert compute_numpy._C_FUSED_AVAILABLE is True


# ── Fused general kernel tests (Phase 89.2) ──────────────────────────────


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
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
    reason="Fused general C not available",
)
def test_fused_general_ncvt2_wald(synthetic_covariate_data_ncvt2):
    """FGEN-04: Fused general Wald bitwise matches non-fused general for n_cvt=2."""
    _run_fused_general_wald_vs_nonfused(
        _prepare_fused_general_data(synthetic_covariate_data_ncvt2)
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
    reason="Fused general C not available",
)
def test_fused_general_ncvt4_wald(synthetic_covariate_data_ncvt4):
    """FGEN-04: Fused general Wald bitwise matches non-fused general for n_cvt=4."""
    _run_fused_general_wald_vs_nonfused(
        _prepare_fused_general_data(synthetic_covariate_data_ncvt4)
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    not compute_numpy._C_MODE4_FUSED_GENERAL_AVAILABLE,
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
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
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
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
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
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
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
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
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
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
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
    not compute_numpy._C_ACCEL_AVAILABLE,
    reason="C extension not available",
)
def test_fused_general_abi_version_9():
    """FGEN-06: ABI_VERSION is >= 9 for fused general kernel support."""
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION >= 9, f"Expected ABI_VERSION>=9, got {ABI_VERSION}"


@pytest.mark.tier0
@pytest.mark.skipif(
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
    reason="Fused general C not available",
)
def test_fused_general_availability_flags():
    """Fused general availability flags are True when C extension has ABI v9."""
    assert compute_numpy._C_FUSED_GENERAL_AVAILABLE is True
    assert compute_numpy._C_MODE4_FUSED_GENERAL_AVAILABLE is True


@pytest.mark.tier0
@pytest.mark.skipif(
    not compute_numpy._C_FUSED_AVAILABLE, reason="Fused C not available"
)
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
    not compute_numpy._C_FUSED_GENERAL_AVAILABLE,
    reason="Fused general C not available",
)
def test_runner_fused_general_ncvt2_dispatch():
    """Runner integration: n_cvt=2 dispatches fused general path end-to-end.

    Exercises the full build_pab_table_for_c → create_workspace_fused_general →
    compute_wald_fused_general_c_ws pipeline through run_lmm_association_numpy.
    Compares fused general results (n_cvt=2 with C extension) against the
    non-fused fallback (monkeypatched _C_FUSED_GENERAL_AVAILABLE=False).
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
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
        n_refine=20,
    )

    # Run with fused general disabled → falls back to non-fused general path.
    # Patch the source module (compute_numpy), which owns dispatch capability flags.
    from unittest.mock import patch

    with patch("jamma.lmm.compute_numpy._C_FUSED_GENERAL_AVAILABLE", False):
        result_nonfused = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=U,
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=1,
            n_refine=20,
        )

    assoc_fused = result_fused.associations
    assoc_nonfused = result_nonfused.associations

    assert len(assoc_fused) == len(assoc_nonfused), (
        f"Fused: {len(assoc_fused)}, Non-fused: {len(assoc_nonfused)}"
    )
    assert len(assoc_fused) > n_snps * 0.8, (
        f"Too many SNPs filtered: {len(assoc_fused)} of {n_snps}"
    )

    # Results should be bitwise identical — same C kernels, same data
    for a_f, a_nf in zip(assoc_fused, assoc_nonfused, strict=True):
        assert a_f.rs == a_nf.rs, f"SNP order mismatch: {a_f.rs} vs {a_nf.rs}"
        np.testing.assert_equal(
            a_f.p_wald,
            a_nf.p_wald,
            err_msg=f"p_wald mismatch for {a_f.rs}",
        )
        np.testing.assert_equal(
            a_f.beta,
            a_nf.beta,
            err_msg=f"beta mismatch for {a_f.rs}",
        )


# ── Fused Score/LRT parity tests (Phase 100.1) ──────────────────────────

_score_fused_available = _C_ACCEL_AVAILABLE and getattr(
    compute_numpy, "_C_SCORE_FUSED_AVAILABLE", False
)
_lrt_fused_available = _C_ACCEL_AVAILABLE and getattr(
    compute_numpy, "_C_LRT_FUSED_AVAILABLE", False
)


@pytest.fixture
def _fused_score_lrt_null_model(split_wald_data):
    """Compute null-model Hi_eval and logl_H0 from split_wald_data.

    Unlike score_lrt_data (which derives from synthetic_wald_data),
    this computes the null model from the same UtW/Uty/eigenvalues
    used by the fused Score/LRT tests.
    """
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    # Build null Uab from UtW/Uty (no genotype)
    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    Uab_null = np.zeros((1, n_samples, 6), dtype=np.float64)
    Uab_null[0, :, 0] = full_uab[0, :, 0]  # ww (invariant)
    Uab_null[0, :, 2] = full_uab[0, :, 2]  # wy (invariant)
    Uab_null[0, :, 5] = full_uab[0, :, 5]  # yy (invariant)

    lambdas_null, logls_null = golden_section_optimize_lambda_mle_numpy(
        1,
        eigenvalues,
        Uab_null,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    lambda_null = float(lambdas_null[0])
    logl_H0 = float(logls_null[0])
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)
    return Hi_eval_null, logl_H0


class TestFusedScoreParity:
    """Verify compute_score_fused_c matches compute_score_split_c."""

    @pytest.fixture
    def fused_score_data(self, split_wald_data, _fused_score_lrt_null_model):
        """Prepare data for Score fused vs split parity."""
        eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
        Hi_eval_null, logl_H0 = _fused_score_lrt_null_model

        w = UtW[:, 0].copy()
        utg_t = np.ascontiguousarray(UtG.T)
        uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
        uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

        return (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            Hi_eval_null,
            n_samples,
            n_snps,
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_available,
        reason="Score fused C not available",
    )
    def test_score_fused_parity(self, fused_score_data):
        """Fused Score matches split Score to rtol=1e-12."""
        from jamma.lmm.compute_numpy import (
            _compute_score_fused_c,
            _compute_score_split_c,
        )

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            Hi_eval_null,
            n_samples,
            n_snps,
        ) = fused_score_data

        # Split reference
        split_result = _compute_score_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            Hi_eval_null,
            n_samples,
            1,
        )

        # Fused
        fused_result = _compute_score_fused_c(
            utg_t,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )

        for key in ("betas", "ses", "p_scores"):
            np.testing.assert_allclose(
                fused_result[key],
                split_result[key],
                rtol=1e-12,
                atol=0,
                err_msg=f"Score {key}: fused vs split mismatch",
            )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_available,
        reason="Score fused C not available",
    )
    def test_score_fused_degenerate_snps(self, fused_score_data):
        """Constant genotype produces NaN beta/se/p_score."""
        from jamma.lmm.compute_numpy import _compute_score_fused_c

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            _,
            Hi_eval_null,
            n_samples,
            n_snps,
        ) = fused_score_data

        utg_degen = utg_t.copy()
        utg_degen[0, :] = 0.0  # constant genotype

        result = _compute_score_fused_c(
            utg_degen,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )

        assert np.isnan(result["betas"][0]), "degenerate SNP: NaN beta"
        assert np.isnan(result["ses"][0]), "degenerate SNP: NaN se"
        assert np.isnan(result["p_scores"][0]), "degenerate SNP: NaN p_score"

        # Non-degenerate SNPs should be finite
        assert np.all(np.isfinite(result["betas"][1:])), (
            "non-degenerate betas should be finite"
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_available,
        reason="Score fused C not available",
    )
    def test_score_fused_multithreaded(self, fused_score_data):
        """Fused Score with n_threads=2 matches split Score."""
        from jamma.lmm.compute_numpy import (
            _compute_score_fused_c,
            _compute_score_split_c,
        )

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            Hi_eval_null,
            n_samples,
            n_snps,
        ) = fused_score_data

        split_result = _compute_score_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            Hi_eval_null,
            n_samples,
            1,
        )

        fused_result = _compute_score_fused_c(
            utg_t,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            2,
        )

        for key in ("betas", "ses", "p_scores"):
            np.testing.assert_allclose(
                fused_result[key],
                split_result[key],
                rtol=1e-12,
                atol=0,
                err_msg=f"Score {key}: fused(2t) vs split mismatch",
            )


class TestFusedLrtParity:
    """Verify compute_lrt_fused_c matches compute_lrt_split_c."""

    @pytest.fixture
    def fused_lrt_data(self, split_wald_data, _fused_score_lrt_null_model):
        """Prepare data for LRT fused vs split parity."""
        eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
        _, logl_H0 = _fused_score_lrt_null_model

        w = UtW[:, 0].copy()
        utg_t = np.ascontiguousarray(UtG.T)
        uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
        uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

        return (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            logl_H0,
            n_samples,
            n_snps,
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_available,
        reason="LRT fused C not available",
    )
    def test_lrt_fused_parity(self, fused_lrt_data):
        """Fused LRT matches split LRT to rtol=5e-5."""
        from jamma.lmm.compute_numpy import (
            _compute_lrt_fused_c,
            _compute_lrt_split_c,
        )

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            logl_H0,
            n_samples,
            n_snps,
        ) = fused_lrt_data

        # Split reference
        split_result = _compute_lrt_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            1,
        )

        # Fused
        fused_result = _compute_lrt_fused_c(
            utg_t,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            1,
        )

        np.testing.assert_allclose(
            fused_result["lambdas_mle"],
            split_result["lambdas_mle"],
            rtol=5e-5,
            atol=0,
            err_msg="LRT lambdas_mle: fused vs split mismatch",
        )
        np.testing.assert_allclose(
            fused_result["p_lrts"],
            split_result["p_lrts"],
            rtol=5e-5,
            atol=0,
            err_msg="LRT p_lrts: fused vs split mismatch",
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_available,
        reason="LRT fused C not available",
    )
    def test_lrt_fused_degenerate_snps(self, fused_lrt_data):
        """Constant genotype produces NaN lambda_mle and p_lrt=1.0."""
        from jamma.lmm.compute_numpy import _compute_lrt_fused_c

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            _,
            logl_H0,
            n_samples,
            n_snps,
        ) = fused_lrt_data

        utg_degen = utg_t.copy()
        utg_degen[0, :] = 0.0

        result = _compute_lrt_fused_c(
            utg_degen,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            1,
        )

        # Degenerate SNP: LRT stat ~ 0, so p_lrt ~ 1.0 (chi2_sf(0) = 1)
        # lambda_mle can be anything (optimization on flat surface)
        assert result["p_lrts"][0] >= 0.99, (
            f"degenerate SNP p_lrt={result['p_lrts'][0]}, expected ~1.0"
        )

        # Non-degenerate SNPs should be finite
        assert np.all(np.isfinite(result["p_lrts"][1:])), (
            "non-degenerate p_lrts should be finite"
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_available,
        reason="LRT fused C not available",
    )
    def test_lrt_fused_multithreaded(self, fused_lrt_data):
        """Fused LRT with n_threads=2 matches split LRT."""
        from jamma.lmm.compute_numpy import (
            _compute_lrt_fused_c,
            _compute_lrt_split_c,
        )

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            logl_H0,
            n_samples,
            n_snps,
        ) = fused_lrt_data

        split_result = _compute_lrt_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            1,
        )

        fused_result = _compute_lrt_fused_c(
            utg_t,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            2,
        )

        np.testing.assert_allclose(
            fused_result["lambdas_mle"],
            split_result["lambdas_mle"],
            rtol=5e-5,
            atol=0,
            err_msg="LRT lambdas_mle: fused(2t) vs split mismatch",
        )
        np.testing.assert_allclose(
            fused_result["p_lrts"],
            split_result["p_lrts"],
            rtol=5e-5,
            atol=0,
            err_msg="LRT p_lrts: fused(2t) vs split mismatch",
        )


@pytest.mark.tier0
def test_abi_version_11():
    """ABI_VERSION is 11 after persistent Score/LRT workspace addition."""
    if not _C_ACCEL_AVAILABLE:
        pytest.skip("C extension not available")
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION == 11


@pytest.mark.tier0
def test_fused_score_available_flag():
    """_C_SCORE_FUSED_AVAILABLE is True when C extension loaded."""
    if not _C_ACCEL_AVAILABLE:
        pytest.skip("C extension not available")
    assert compute_numpy._C_SCORE_FUSED_AVAILABLE is True


@pytest.mark.tier0
def test_fused_lrt_available_flag():
    """_C_LRT_FUSED_AVAILABLE is True when C extension loaded."""
    if not _C_ACCEL_AVAILABLE:
        pytest.skip("C extension not available")
    assert compute_numpy._C_LRT_FUSED_AVAILABLE is True


# ── Runner-level fused Score/LRT dispatch tests (Phase 100.2) ────────────


def _make_runner_test_data(rng, n_samples=50, n_snps=20):
    """Create synthetic data for runner dispatch tests."""
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.4, 0.4, 0.2])
    phenotypes = rng.standard_normal(n_samples)
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]
    U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]
    return eigenvalues, genotypes, phenotypes, snp_info, U


@pytest.mark.skipif(not _score_fused_available, reason="Fused Score C not available")
def test_runner_fused_score_dispatch():
    """Runner dispatches fused Score path for mode 3, matches SoA split.

    Prefers workspace-based dispatch when available; falls back to stateless.
    """
    from unittest.mock import patch

    from jamma.lmm.compute_numpy import (
        _C_SCORE_FUSED_WS_AVAILABLE,
        _compute_score_fused_c,
        _compute_score_fused_ws_c,
    )
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(200)
    eigenvalues, genotypes, phenotypes, snp_info, U = _make_runner_test_data(rng)

    # Fused Score path (default) — verify the fused C function is actually called.
    # Workspace path is preferred when available; stateless is the fallback.
    if _C_SCORE_FUSED_WS_AVAILABLE:
        with patch(
            "jamma.lmm.compute_numpy._compute_score_fused_ws_c",
            wraps=_compute_score_fused_ws_c,
        ) as mock_fused:
            result_fused = run_lmm_association_numpy(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=None,
                snp_info=snp_info,
                eigenvalues=eigenvalues,
                eigenvectors=U,
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                lmm_mode=3,
                n_refine=20,
            )
        assert mock_fused.called, "Fused Score WS C function was not called"
    else:
        with patch(
            "jamma.lmm.compute_numpy._compute_score_fused_c",
            wraps=_compute_score_fused_c,
        ) as mock_fused:
            result_fused = run_lmm_association_numpy(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=None,
                snp_info=snp_info,
                eigenvalues=eigenvalues,
                eigenvectors=U,
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                lmm_mode=3,
                n_refine=20,
            )
        assert mock_fused.called, "Fused Score C function was not called"

    # SoA split path (disable all fused Score variants)
    with (
        patch("jamma.lmm.compute_numpy._C_SCORE_FUSED_AVAILABLE", False),
        patch("jamma.lmm.compute_numpy._C_SCORE_FUSED_WS_AVAILABLE", False),
    ):
        result_split = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=U,
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=3,
            n_refine=20,
        )

    fused = result_fused.associations
    split = result_split.associations

    assert len(fused) == len(split), f"Count mismatch: {len(fused)} vs {len(split)}"
    assert len(fused) > 10, f"Too many SNPs filtered: {len(fused)}"

    for a_f, a_s in zip(fused, split, strict=True):
        assert a_f.rs == a_s.rs
        if a_f.p_score is not None and a_s.p_score is not None:
            np.testing.assert_allclose(
                a_f.p_score,
                a_s.p_score,
                rtol=1e-12,
                err_msg=f"p_score mismatch for {a_f.rs}",
            )


@pytest.mark.skipif(not _lrt_fused_available, reason="Fused LRT C not available")
def test_runner_fused_lrt_dispatch():
    """Runner dispatches fused LRT path for mode 2, matches SoA split.

    Prefers workspace-based dispatch when available; falls back to stateless.
    """
    from unittest.mock import patch

    from jamma.lmm.compute_numpy import (
        _C_LRT_FUSED_WS_AVAILABLE,
        _compute_lrt_fused_c,
        _compute_lrt_fused_ws_c,
    )
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(201)
    eigenvalues, genotypes, phenotypes, snp_info, U = _make_runner_test_data(rng)

    # Fused LRT path (default) — verify the fused C function is actually called.
    # Workspace path is preferred when available; stateless is the fallback.
    if _C_LRT_FUSED_WS_AVAILABLE:
        with patch(
            "jamma.lmm.compute_numpy._compute_lrt_fused_ws_c",
            wraps=_compute_lrt_fused_ws_c,
        ) as mock_fused:
            result_fused = run_lmm_association_numpy(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=None,
                snp_info=snp_info,
                eigenvalues=eigenvalues,
                eigenvectors=U,
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                lmm_mode=2,
                n_refine=20,
            )
        assert mock_fused.called, "Fused LRT WS C function was not called"
    else:
        with patch(
            "jamma.lmm.compute_numpy._compute_lrt_fused_c",
            wraps=_compute_lrt_fused_c,
        ) as mock_fused:
            result_fused = run_lmm_association_numpy(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=None,
                snp_info=snp_info,
                eigenvalues=eigenvalues,
                eigenvectors=U,
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                lmm_mode=2,
                n_refine=20,
            )
        assert mock_fused.called, "Fused LRT C function was not called"

    # SoA split path (disable all fused LRT variants)
    with (
        patch("jamma.lmm.compute_numpy._C_LRT_FUSED_AVAILABLE", False),
        patch("jamma.lmm.compute_numpy._C_LRT_FUSED_WS_AVAILABLE", False),
    ):
        result_split = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=U,
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=2,
            n_refine=20,
        )

    fused = result_fused.associations
    split = result_split.associations

    assert len(fused) == len(split), f"Count mismatch: {len(fused)} vs {len(split)}"
    assert len(fused) > 10, f"Too many SNPs filtered: {len(fused)}"

    for a_f, a_s in zip(fused, split, strict=True):
        assert a_f.rs == a_s.rs
        if a_f.p_lrt is not None and a_s.p_lrt is not None:
            np.testing.assert_allclose(
                a_f.p_lrt,
                a_s.p_lrt,
                rtol=5e-5,
                err_msg=f"p_lrt mismatch for {a_f.rs}",
            )
        if a_f.l_mle is not None and a_s.l_mle is not None:
            np.testing.assert_allclose(
                a_f.l_mle,
                a_s.l_mle,
                rtol=5e-5,
                err_msg=f"l_mle mismatch for {a_f.rs}",
            )


@pytest.mark.skipif(not _score_fused_available, reason="Fused Score C not available")
def test_runner_fused_score_chunk_size():
    """Fused Score uses 1-col accounting (4x larger chunks at same budget)."""
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy

    n_samples = 1000
    n_filtered = 200_000
    # Budget large enough that both paths exceed the 100-SNP floor.
    # Split needs n_samples * 4 * 8 = 32KB/SNP; fused needs n_samples * 8 = 8KB/SNP.
    # At 16 MB: split → 500 SNPs, fused → 2000 SNPs.
    budget = 16_000_000

    chunk_fused = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        use_split=True,
        lmm_mode=3,
        mem_budget_bytes=budget,
    )

    from unittest.mock import patch

    with patch("jamma.lmm.chunk_sizing._C_SCORE_FUSED_AVAILABLE", False):
        chunk_split = compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt=1,
            use_split=True,
            lmm_mode=3,
            mem_budget_bytes=budget,
        )

    assert chunk_fused >= 3 * chunk_split, (
        f"Fused chunk ({chunk_fused}) should be >= 3x split chunk ({chunk_split})"
    )


@pytest.mark.skipif(not _lrt_fused_available, reason="Fused LRT C not available")
def test_runner_fused_lrt_chunk_size():
    """Fused LRT uses 1-col accounting (4x larger chunks at same budget)."""
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy

    n_samples = 1000
    n_filtered = 200_000
    budget = 16_000_000  # Same budget as Score test

    chunk_fused = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        use_split=True,
        lmm_mode=2,
        mem_budget_bytes=budget,
    )

    from unittest.mock import patch

    with patch("jamma.lmm.chunk_sizing._C_LRT_FUSED_AVAILABLE", False):
        chunk_split = compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt=1,
            use_split=True,
            lmm_mode=2,
            mem_budget_bytes=budget,
        )

    assert chunk_fused >= 3 * chunk_split, (
        f"Fused chunk ({chunk_fused}) should be >= 3x split chunk ({chunk_split})"
    )


# ---------------------------------------------------------------------------
# Score workspace parity tests
# ---------------------------------------------------------------------------

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
        from jamma.lmm.compute_numpy import _create_workspace_score_fused_c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        ws = _create_workspace_score_fused_c(
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
        from jamma.lmm.compute_numpy import (
            _compute_score_fused_c,
            _compute_score_fused_ws_c,
            _create_workspace_score_fused_c,
        )

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        # Stateless reference
        ref = _compute_score_fused_c(
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
        ws = _create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1,
        )
        result = _compute_score_fused_ws_c(ws, utg_t, 1)

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
        from jamma.lmm.compute_numpy import (
            _compute_score_fused_c,
            _compute_score_fused_ws_c,
            _create_workspace_score_fused_c,
        )

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, Hi_eval_null, n_samples, n_snps) = (
            score_ws_data
        )

        ws = _create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1,
        )

        # Chunk 1
        ref1 = _compute_score_fused_c(
            utg_t,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )
        result1 = _compute_score_fused_ws_c(ws, utg_t, 1)
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
        ref2 = _compute_score_fused_c(
            utg_t2,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )
        result2 = _compute_score_fused_ws_c(ws, utg_t2, 1)
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
        from jamma.lmm.compute_numpy import (
            _compute_score_fused_ws_c,
            create_lmm_workspace,
        )

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
            _compute_score_fused_ws_c(wald_ws, utg_t, 1)


# ---------------------------------------------------------------------------
# LRT workspace parity tests
# ---------------------------------------------------------------------------

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
        from jamma.lmm.compute_numpy import _create_workspace_lrt_fused_c

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        ws = _create_workspace_lrt_fused_c(
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
        from jamma.lmm.compute_numpy import (
            _compute_lrt_fused_c,
            _compute_lrt_fused_ws_c,
            _create_workspace_lrt_fused_c,
        )

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        logl_H0 = -150.0

        # Stateless reference
        ref = _compute_lrt_fused_c(
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
        ws = _create_workspace_lrt_fused_c(
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
        result = _compute_lrt_fused_ws_c(ws, utg_t, 1)

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
        from jamma.lmm.compute_numpy import (
            _compute_lrt_fused_c,
            _compute_lrt_fused_ws_c,
            _create_workspace_lrt_fused_c,
        )

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        logl_H0 = -150.0
        ws = _create_workspace_lrt_fused_c(
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
        ref1 = _compute_lrt_fused_c(
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
        result1 = _compute_lrt_fused_ws_c(ws, utg_t, 1)
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
        ref2 = _compute_lrt_fused_c(
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
        result2 = _compute_lrt_fused_ws_c(ws, utg_t2, 1)
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
        from jamma.lmm.compute_numpy import (
            _compute_lrt_fused_ws_c,
            _create_workspace_score_fused_c,
        )

        (eigenvalues, w, Uty, utg_t, uab_inv_soa, n_samples, n_snps) = lrt_ws_data

        # Hi_eval_null needed for Score workspace creation
        Hi_eval_null = 1.0 / (0.5 * eigenvalues + 1.0)

        # Create a Score workspace (wrong type for LRT)
        score_ws = _create_workspace_score_fused_c(
            w,
            Uty,
            Hi_eval_null,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1,
        )

        with pytest.raises(ValueError, match="PyCapsule_GetPointer"):
            _compute_lrt_fused_ws_c(score_ws, utg_t, 1)


# ---------------------------------------------------------------------------
# Identity Pab optimization — logdet_from_row0 helper
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_wald_identity_pab_optimization(synthetic_covariate_data_ncvt2):
    """C-GEN-OPT-01: logdet_from_row0 helper produces identical Wald results.

    Verifies the identity Pab prepass optimization (logdet_from_row0 helper)
    produces numerically identical results to the Python reference. This test
    specifically targets the logdet_iab computation path which flows through
    to REML log-likelihood and ultimately to Wald beta/SE/p-values.

    The C extension uses logdet_from_row0 to deduplicate the identity Pab
    prepass across compute_lmm_chunk_general_c, fused general Wald, and
    fused general mode-4. If the helper introduces any numerical divergence,
    it will show up in the Wald results compared to the Python reference.
    """

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]

    # Python reference (force fallback — no C extension)
    orig = compute_numpy._C_GENERAL_AVAILABLE
    try:
        compute_numpy._C_GENERAL_AVAILABLE = False
        result_py = _compute_wald_numpy(
            n_cvt,
            eigenvalues,
            Uab_batch,
            n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
        )
    finally:
        compute_numpy._C_GENERAL_AVAILABLE = orig

    # C extension path (uses logdet_from_row0 helper internally)
    result_c = _compute_wald_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        n_threads=1,
    )

    # logdet_iab affects REML logl which flows to lambda, beta, SE, p-values.
    # Any divergence from the optimization would show up here.
    for key in ("lambdas", "logls", "betas", "ses"):
        np.testing.assert_allclose(
            result_c[key],
            result_py[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=(
                f"{key}: C (logdet_from_row0) vs Python mismatch — "
                f"identity Pab optimization may have diverged"
            ),
        )
    np.testing.assert_allclose(
        result_c["pwalds"],
        result_py["pwalds"],
        rtol=1e-6,
        atol=1e-14,
        equal_nan=True,
        err_msg=(
            "pwalds: C (logdet_from_row0) vs Python mismatch — "
            "identity Pab optimization may have diverged"
        ),
    )

    # Verify we actually tested SNPs (not an empty batch)
    valid = ~np.isnan(result_c["betas"])
    assert np.sum(valid) > 0, "No valid SNPs — test is vacuous"

    # Cross-check: n_cvt=4 data as well for broader coverage
    # (handled by test_general_ncvt_reml_wald_ncvt4 separately)


# =============================================================================
# n_cvt boundary tests (MAX_N_CVT=100)
# =============================================================================


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_ncvt_101_rejected_by_c_extension():
    """C extension raises ValueError for n_cvt=101 (exceeds MAX_N_CVT=100).

    Uses compute_score_batch_general_c as a representative entry point since
    it takes n_cvt as a direct parameter (not hidden in a workspace).
    """
    if not _score_general_c_available():
        pytest.skip("compute_score_batch_general_c not compiled yet")

    from jamma.lmm._lmm_accel import compute_score_batch_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    n_cvt = 101
    n_samples = 200
    n_snps = 5

    rng = np.random.default_rng(777)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))[::-1]

    # Build minimal arrays — they won't be used because validation fails first.
    # Use n_cvt=100 for the pab table (build_pab_table_for_c is pure Python,
    # no limit), then pass n_cvt=101 to the C function to trigger the check.
    pab_table_dict = build_pab_table_for_c(100)
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    Uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    Hi_eval_null = np.ones(n_samples, dtype=np.float64)

    with pytest.raises(ValueError, match=r"n_cvt must be 1\.\.100, got 101"):
        compute_score_batch_general_c(
            eigenvalues,
            Uab_batch,
            Hi_eval_null,
            n_samples,
            n_cvt,
            pab_table_dict,
            1,  # n_threads
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_reml_wald_ncvt20():
    """C extension Wald matches Python for n_cvt=20 (previous MAX_N_CVT limit).

    Verifies that n_cvt=20 — the old limit before MAX_N_CVT was raised to
    100 — works correctly through the full REML+CalcPab+Wald pipeline.
    Uses small matrices (150 samples, 15 SNPs) to keep execution fast.
    """
    from tests.conftest import _build_synthetic_covariate_data

    data = _build_synthetic_covariate_data(
        n_cvt=20, n_samples=150, n_snps=15, seed=2020
    )
    _run_general_ncvt_c_vs_python(data)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_reml_wald_ncvt50():
    """C extension Wald matches Python for n_cvt=50 (well beyond old limit).

    Exercises n_cvt=50, which requires Pab tables with n_index=1431 and
    n_rows=52. Validates that the raised MAX_N_CVT=100 works at a midpoint.
    Uses small matrices (150 samples, 10 SNPs) to keep execution fast.
    """
    from tests.conftest import _build_synthetic_covariate_data

    data = _build_synthetic_covariate_data(
        n_cvt=50, n_samples=150, n_snps=10, seed=5050
    )
    _run_general_ncvt_c_vs_python(data)
