"""_lmm_accel C extension tests: import, edge cases, and input validation.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.

The kernel under test is the fused n_cvt=1 Wald workspace, which is what
``DispatchPath.FUSED`` reaches for lmm_mode 1. This module used to drive
``compute_lmm_batch_c``, a batch entry point no dispatch path selects, through
``_compute_wald_numpy``. That function's inner C ladder is unreachable by
construction: its only production caller is ``compute_lmm_chunk_numpy``, which
the runner reaches only on ``NUMPY_FALLBACK``, and that path is chosen only when
the extension is absent.

Numerical agreement with NumPy, thread determinism and the single-degenerate-SNP
case are checked on the fused kernel in test_lmm_accel_fused.py. What is here is
what that does not cover: that the extension imports, the two batch-shape edge
cases, and the validation the kernel does on its arguments.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import _c, compute_lmm_chunk_numpy
from jamma.lmm.schema import MIN_N_GRID


def _make_workspace(
    fused_data,
    *,
    eigenvalues=None,
    uab_invariant_soa=None,
    n_samples=None,
    l_min=1e-5,
    l_max=1e5,
    n_grid=50,
    n_refine=20,
):
    """Build the fused workspace, with any argument overridden by keyword.

    The overridable arguments are spelled out rather than merged from a dict so
    the call stays typed; a dict merge widens every value to the union of the
    dict's types and pyrefly rejects the call.
    """
    fixture_eigenvalues, w, Uty, _, fixture_inv_soa, _, fixture_n = fused_data
    return _c().create_workspace_ncvt1_c(
        fixture_eigenvalues if eigenvalues is None else eigenvalues,
        fixture_inv_soa if uab_invariant_soa is None else uab_invariant_soa,
        w,
        Uty,
        fixture_n if n_samples is None else n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        lmm_mode=1,
    )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_c_extension_importable():
    """The kernels the dispatch table names are importable and callable."""
    from jamma.lmm._lmm_accel import (
        compute_lmm_chunk_fused_c,
        compute_lmm_chunk_fused_general_c,
        create_workspace_fused_general_c,
        create_workspace_ncvt1_c,
    )

    for fn in (
        create_workspace_ncvt1_c,
        compute_lmm_chunk_fused_c,
        create_workspace_fused_general_c,
        compute_lmm_chunk_fused_general_c,
    ):
        assert callable(fn)


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
    assert not np.any(np.isnan(result["lambdas"])), (
        "Python fallback produced NaN lambdas"
    )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_c_extension_single_snp(fused_data):
    """Minimal case: n_snps=1 works without index errors."""
    _, _, _, utg_t, _, _, _ = fused_data

    result = _c().compute_lmm_chunk_fused_c(_make_workspace(fused_data), utg_t[:1], 1)

    for key in ("lambdas", "betas", "ses", "pwalds"):
        assert result[key].shape == (1,), f"{key} shape {result[key].shape} != (1,)"
    assert not np.isnan(result["lambdas"][0]), "Single SNP lambda should not be NaN"


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_c_extension_all_degenerate_snps(fused_data):
    """Every SNP degenerate: the whole output is NaN rather than a crash."""
    _, _, _, utg_t, _, _, _ = fused_data

    # A constant genotype rotates to an all-zero UtG column, driving xx and so
    # P_XX to zero. Zeroing every row makes the entire batch degenerate.
    utg_degen = np.zeros_like(utg_t)

    result = _c().compute_lmm_chunk_fused_c(_make_workspace(fused_data), utg_degen, 1)

    for key in ("betas", "ses", "pwalds"):
        assert np.all(np.isnan(result[key])), f"Expected all-NaN {key}"


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
class TestFusedWorkspaceInputValidation:
    """The kernel raises clean errors on invalid array arguments."""

    def test_wrong_eigenvalues_shape(self, fused_data):
        eigenvalues = fused_data[0]
        with pytest.raises(ValueError, match="eigenvalues"):
            _make_workspace(fused_data, eigenvalues=eigenvalues[:10])

    @pytest.mark.parametrize(
        "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"]
    )
    def test_nonfinite_eigenvalues(self, fused_data, bad_value):
        bad = fused_data[0].copy()
        bad[10] = bad_value
        with pytest.raises(ValueError, match=r"eigenvalues.*not finite"):
            _make_workspace(fused_data, eigenvalues=bad)

    def test_wrong_invariant_soa_shape(self, fused_data):
        uab_inv_soa = fused_data[4]
        with pytest.raises(ValueError, match="uab_invariant"):
            _make_workspace(fused_data, uab_invariant_soa=uab_inv_soa.T)

    def test_wrong_utg_t_n_samples(self, fused_data):
        _, _, _, utg_t, _, _, _ = fused_data
        ws = _make_workspace(fused_data)
        with pytest.raises(ValueError, match="utg_t"):
            _c().compute_lmm_chunk_fused_c(ws, np.ascontiguousarray(utg_t[:, :10]), 1)


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
class TestFusedWorkspaceScalarValidation:
    """The kernel validates its scalar parameters."""

    def test_n_samples_too_small(self, fused_data):
        with pytest.raises(ValueError, match="n_samples"):
            _make_workspace(fused_data, n_samples=2)

    def test_l_min_zero(self, fused_data):
        with pytest.raises(ValueError, match="l_min"):
            _make_workspace(fused_data, l_min=0.0)

    def test_l_max_le_l_min(self, fused_data):
        with pytest.raises(ValueError, match="l_min"):
            _make_workspace(fused_data, l_min=1.0, l_max=1.0)

    def test_n_grid_too_small(self, fused_data):
        """The kernel enforces the same minimum the config layer does.

        Anchored on MIN_N_GRID so the Python bound and the C bound in
        validate_batch_params cannot drift apart silently.
        """
        with pytest.raises(ValueError, match="n_grid"):
            _make_workspace(fused_data, n_grid=MIN_N_GRID - 1)

    def test_n_refine_too_small(self, fused_data):
        with pytest.raises(ValueError, match="n_refine"):
            _make_workspace(fused_data, n_refine=0)
