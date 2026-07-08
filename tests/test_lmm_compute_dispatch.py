"""Focused dispatch-boundary tests for jamma.lmm.compute_numpy."""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy


@pytest.mark.tier0
@pytest.mark.parametrize("n_cvt", [76, compute_numpy.MAX_C_N_CVT])
def test_wald_general_c_dispatch_uses_documented_ncvt_limit(monkeypatch, n_cvt):
    """_compute_wald_numpy uses the general C path through MAX_C_N_CVT."""
    n_samples = n_cvt + 5
    n_snps = 1
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    eigenvalues = np.ones(n_samples, dtype=np.float64)
    uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    iab_batch = np.zeros((n_snps, n_index), dtype=np.float64)
    workspace = object()
    calls = {}
    expected = {
        "lambdas": np.zeros(n_snps),
        "logls": np.zeros(n_snps),
        "betas": np.zeros(n_snps),
        "ses": np.zeros(n_snps),
        "pwalds": np.zeros(n_snps),
    }

    def fake_create_workspace(
        _eigenvalues,
        uab_invariant_soa,
        n_samples_arg,
        n_cvt_arg,
        _l_min,
        _l_max,
        _n_grid,
        _n_refine,
        n_threads_arg,
    ):
        calls["workspace"] = (uab_invariant_soa.shape, n_samples_arg, n_cvt_arg)
        calls["workspace_threads"] = n_threads_arg
        return workspace

    def fake_compute(workspace_arg, uab_varying_soa, n_threads_arg):
        calls["compute"] = (workspace_arg, uab_varying_soa.shape, n_threads_arg)
        calls["varying_is_contiguous"] = uab_varying_soa.flags.c_contiguous
        return expected

    def fail_python_fallback(*_args, **_kwargs):
        raise AssertionError("n_cvt within the C limit fell back to Python")

    monkeypatch.setattr(compute_numpy, "_C_GENERAL_AVAILABLE", True)
    monkeypatch.setattr(
        compute_numpy,
        "create_lmm_workspace_general",
        fake_create_workspace,
    )
    monkeypatch.setattr(  # allow-patch: sentinel proves boundary dispatch choice
        compute_numpy,
        "compute_wald_general_c_ws",
        fake_compute,
    )
    monkeypatch.setattr(
        compute_numpy,
        "golden_section_optimize_lambda_numpy",
        fail_python_fallback,
    )

    result = compute_numpy._compute_wald_numpy(
        n_cvt,
        eigenvalues,
        uab_batch,
        n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        Iab_batch=iab_batch,
        n_threads=3,
    )

    assert result is expected
    invariant_shape, workspace_n_samples, workspace_n_cvt = calls["workspace"]
    assert invariant_shape[1] == n_samples
    assert workspace_n_samples == n_samples
    assert workspace_n_cvt == n_cvt
    assert calls["workspace_threads"] == 3

    compute_workspace, varying_shape, compute_threads = calls["compute"]
    assert compute_workspace is workspace
    assert varying_shape[0] == n_snps
    assert varying_shape[2] == n_samples
    assert compute_threads == 3
    assert calls["varying_is_contiguous"] is True
