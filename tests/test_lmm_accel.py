"""Tests for the C extension accelerator (_lmm_accel).

Validates that the C extension produces numerically identical results to the
pure-Python NumPy implementation. Also tests the fallback mechanism.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _C_SPLIT_AVAILABLE,
    _compute_lmm_chunk_numpy,
    _compute_wald_numpy,
    _compute_wald_split_c,
    compute_wald_split_c_ws,
    create_lmm_workspace,
)
from jamma.lmm.likelihood_numpy import (
    batch_compute_iab_numpy,
    batch_compute_iab_split_ncvt1,
    batch_compute_iab_split_ncvt1_soa,
    batch_compute_uab_split_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
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
    # Generate Uab-like data (not physically meaningful but numerically valid)
    Uab_batch = rng.standard_normal((n_snps, n_samples, 6))
    # Make ww column positive (required for Pab recursion)
    Uab_batch[:, :, 0] = np.abs(Uab_batch[:, :, 0]) + 0.1
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

    # --- Python path: temporarily disable C extension ---
    monkeypatch.setattr(compute_numpy, "_C_ACCEL_AVAILABLE", False)
    result_py = _compute_wald_numpy(
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

    # All outputs must agree within tight tolerances.
    # NaN entries (degenerate SNPs) are excluded from comparison via equal_nan=True.
    np.testing.assert_allclose(
        result_c["lambdas"],
        result_py["lambdas"],
        rtol=1e-10,
        atol=1e-14,
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
    # betas/ses: cached coarse-grid hi_eval changes FP accumulation order in
    # the Pab dot products, causing tiny differences that propagate to beta/SE.
    # Same root cause as logls tolerance — mathematically identical, different
    # FP operation ordering.
    np.testing.assert_allclose(
        result_c["betas"],
        result_py["betas"],
        rtol=1e-9,
        atol=1e-14,
        equal_nan=True,
        err_msg="betas: C vs Python mismatch",
    )
    np.testing.assert_allclose(
        result_c["ses"],
        result_py["ses"],
        rtol=1e-9,
        atol=1e-14,
        equal_nan=True,
        err_msg="ses: C vs Python mismatch",
    )
    # C path computes p-values via C-side Lentz CF betainc; Python path uses
    # betainc_batch in special.py. Same algorithm, different FP ordering.
    np.testing.assert_allclose(
        result_c["pwalds"],
        result_py["pwalds"],
        rtol=1e-8,
        atol=1e-14,
        equal_nan=True,
        err_msg="pwalds: C vs Python mismatch",
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_c_fallback_when_ncvt_gt1(synthetic_wald_data, monkeypatch):
    """With n_cvt=2, the C path must not be called (falls back to Python).

    Monkeypatches the C function to raise AssertionError if called.
    """
    eigenvalues, _Uab_batch_ncvt1, n_samples = synthetic_wald_data
    # Rebuild Uab for n_cvt=2 (n_index = (2+3)*(2+2)//2 = 10)
    rng = np.random.default_rng(0)
    n_snps = 10
    Uab_batch = rng.standard_normal((n_snps, n_samples, 10))
    Uab_batch[:, :, 0] = np.abs(Uab_batch[:, :, 0]) + 0.1

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("C extension should not be called for n_cvt > 1")

    monkeypatch.setattr(compute_numpy, "_compute_lmm_batch_c", should_not_be_called)

    # Should succeed via the Python path without calling the C function
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

    result = _compute_lmm_chunk_numpy(
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
    kwargs = dict(
        n_cvt=1,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        Iab_batch=Iab_batch,
    )

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

    with pytest.raises(ValueError, match="eigenvalues.*not finite"):
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
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)
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
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)
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

    with pytest.raises(ValueError, match="eigenvalues.*not finite"):
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
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)
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
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)

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
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)

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

    # Non-finite eigenvalues rejected (NaN, Inf, -Inf)
    for bad_value in [np.nan, np.inf, -np.inf]:
        bad_evals = eigenvalues.copy()
        bad_evals[0] = bad_value
        with pytest.raises(ValueError, match="eigenvalues.*not finite"):
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
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)
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
    from jamma.lmm.runner_numpy import _compute_chunk_size_numpy

    rng = np.random.default_rng(42)
    n_samples = 100
    # Use enough SNPs that we get at least 3 chunks
    chunk_size = _compute_chunk_size_numpy(
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
    results_pipeline = run_lmm_association_numpy(
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

    # Verify we got results for all SNPs (none filtered at maf=0)
    # Some may be filtered by the internal variance check, but most should pass
    assert len(results_pipeline) > n_snps * 0.8, (
        f"Too many SNPs filtered: got {len(results_pipeline)} of {n_snps}"
    )

    # Run with pipeline disabled: force single chunk by using sequential path
    # We do this by monkeypatching _C_SPLIT_AVAILABLE to False
    import jamma.lmm.runner_numpy as runner_mod

    orig_split = runner_mod._C_SPLIT_AVAILABLE
    try:
        runner_mod._C_SPLIT_AVAILABLE = False
        results_sequential = run_lmm_association_numpy(
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
    finally:
        runner_mod._C_SPLIT_AVAILABLE = orig_split

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


@pytest.mark.tier2
@pytest.mark.slow
@pytest.mark.benchmark
class TestCExtensionPerformance:
    """Benchmark C extension vs Python on realistic data.

    The C extension gains its advantage through OpenMP parallelism — at 1 thread
    it is slower than NumPy's vectorised batch path. At N physical cores it
    achieves the expected speedup.  The test uses get_physical_core_count() so
    it automatically picks the right thread count on any machine.

    A small warmup call is made before timing to amortise OpenMP thread-pool
    startup cost.
    """

    def test_c_faster_than_python(self, monkeypatch):
        """C extension with all physical cores is at least 2x faster than Python."""
        import time

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
            pytest.skip(
                f"Benchmark requires >=4 physical cores for reliable 2x speedup; "
                f"found {n_threads}"
            )

        # Use large synthetic data for timing
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

        # Time C path: take best of 3 runs (stable min)
        times_c = []
        result_c = None
        for _ in range(3):
            monkeypatch.setattr(cn, "_C_ACCEL_AVAILABLE", True)
            start_c = time.perf_counter()
            result_c = _compute_wald_numpy(
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
            times_c.append(time.perf_counter() - start_c)

        # Time Python path: take best of 3 runs
        times_py = []
        result_py = None
        for _ in range(3):
            monkeypatch.setattr(cn, "_C_ACCEL_AVAILABLE", False)
            start_py = time.perf_counter()
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
            times_py.append(time.perf_counter() - start_py)

        t_c = min(times_c)
        t_py = min(times_py)
        speedup = t_py / t_c if t_c > 0 else float("inf")
        print(
            f"\nC extension ({n_threads} threads): {t_c:.3f}s, "
            f"Python (vectorised): {t_py:.3f}s, speedup: {speedup:.1f}x"
        )

        # Assert speedup (conservative: 2x minimum when using all physical cores)
        assert speedup >= 2.0, (
            f"C extension only {speedup:.1f}x faster than Python "
            f"(C={t_c:.3f}s with {n_threads} threads, Python={t_py:.3f}s). "
            f"Expected at least 2x with {n_threads} cores."
        )

        # Verify numerical parity: C and Python golden section can produce
        # slightly different optima due to FP operation ordering, especially
        # on flat likelihood landscapes at extreme lambda.  Use 5e-5 rtol
        # (borrowed from JAX-vs-GEMMA tolerance, where GEMMA uses Brent and
        # JAMMA uses golden section) since 2000-SNP batches routinely
        # contain outliers at 2.3e-5 relative from FP ordering differences
        # on near-degenerate likelihoods.
        np.testing.assert_allclose(
            result_c["lambdas"], result_py["lambdas"], rtol=5e-5, atol=1e-14
        )
