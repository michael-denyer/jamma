"""_lmm_accel C extension tests: import, C-vs-Python parity, and input validation.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import _compute_wald_numpy, compute_lmm_chunk_numpy
from jamma.lmm.likelihood_numpy import (
    batch_compute_iab_numpy,
    golden_section_optimize_lambda_numpy,
)
from jamma.lmm.schema import MIN_N_GRID


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_c_extension_importable():
    """Verify the C extension module can be imported directly."""
    from jamma.lmm._lmm_accel import compute_lmm_batch_c

    assert callable(compute_lmm_batch_c)


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
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

    # --- C path (default when compute_numpy._accel is not None and n_cvt == 1) ---
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
    # With no extension loaded, _compute_wald_numpy dispatches n_cvt=1
    # to the split-Uab optimizer, which uses different FP accumulation
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
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_c_fallback_ncvt_gt1_when_general_unavailable(synthetic_wald_data, monkeypatch):
    """With n_cvt=2 and compute_numpy._accel is not None=False, falls back to Python.

    Monkeypatches compute_numpy._accel is not None to False and verifies the n_cvt=1
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
        compute_numpy._accel,
        "compute_lmm_batch_c",
        should_not_be_called,
    )  # allow-patch: sentinel asserts n_cvt=1 C kernel is NOT taken when n_cvt>1
    monkeypatch.setattr(compute_numpy, "_accel", None)

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
    """With no extension loaded, the Python path runs without error."""
    eigenvalues, Uab_batch, n_samples = synthetic_wald_data

    monkeypatch.setattr(compute_numpy, "_accel", None)

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
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
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
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
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
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
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
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
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
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
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
        """The kernel enforces the same minimum the config layer does.

        Anchored on MIN_N_GRID so the Python bound and the C bound in
        validate_batch_params cannot drift apart silently.
        """
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="n_grid"):
            fn(eigenvalues, Uab, Iab, n, 1e-5, 1e5, MIN_N_GRID - 1, 20, 1)

    def test_n_refine_too_small(self):
        fn, eigenvalues, Uab, Iab, n = _make_valid_c_inputs()
        with pytest.raises(ValueError, match="n_refine"):
            fn(eigenvalues, Uab, Iab, n, 1e-5, 1e5, 50, 0, 1)


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
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
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
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
