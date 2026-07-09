"""Guard/precondition tests for the shared NumPy LMM chunk runner.

These cover the cheap, isolated failure paths that the end-to-end parity suites
never exercise: the systemic-NaN abort, the ``run_lmm_chunk_source_numpy``
argument preconditions, and the ``dispatch_soa_split`` mode guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.lmm.chunk_dispatch import dispatch_soa_split
from jamma.lmm.chunk_runner_numpy import (
    _raise_if_systemic_nan,
    run_lmm_chunk_source_numpy,
)

# ---------------------------------------------------------------------------
# _raise_if_systemic_nan
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_systemic_nan_noop_when_no_nans():
    _raise_if_systemic_nan({}, n_filtered=100, nan_abort_fraction=1.0)


@pytest.mark.tier0
def test_systemic_nan_noop_for_partial_column():
    # A handful of per-SNP NaNs (e.g. P_xx <= 0 after projection) must not abort.
    _raise_if_systemic_nan(
        {"betas": 3, "pwalds": 1}, n_filtered=100, nan_abort_fraction=1.0
    )


@pytest.mark.tier0
def test_systemic_nan_aborts_on_fully_nan_column():
    with pytest.raises(RuntimeError, match="systemic failure"):
        _raise_if_systemic_nan(
            {"betas": 100, "pwalds": 40}, n_filtered=100, nan_abort_fraction=1.0
        )


@pytest.mark.tier0
def test_systemic_nan_respects_custom_fraction():
    # 60/100 NaN with a 0.5 threshold aborts; 40/100 does not.
    with pytest.raises(RuntimeError, match="abort threshold"):
        _raise_if_systemic_nan({"betas": 60}, n_filtered=100, nan_abort_fraction=0.5)
    _raise_if_systemic_nan({"betas": 40}, n_filtered=100, nan_abort_fraction=0.5)


# ---------------------------------------------------------------------------
# run_lmm_chunk_source_numpy preconditions
# ---------------------------------------------------------------------------


def _run_kwargs(**overrides):
    """Minimal valid kwargs minus ``lmm_mode`` (passed as a literal per call).

    The raw source / sink are never reached on the precondition paths, which
    raise before dispatch selection.
    """
    n_samples = 4
    base = {
        "raw_chunk_source_factory": lambda _chunk_size: lambda: None,
        "chunk_sink": lambda _arrays, _start, _end: None,
        "U": np.eye(n_samples),
        "eigenvalues_np": np.ones(n_samples),
        "UtW": np.ones((n_samples, 1)),
        "Uty": np.ones(n_samples),
        "Hi_eval_null": np.ones(n_samples),
        "logl_H0": -1.0,
        "n_samples": n_samples,
        "n_filtered": 5,
        "n_cvt": 1,
        "filtered_means": np.zeros(5),
        "l_min": 1e-5,
        "l_max": 1e5,
        "n_grid": 50,
        "n_refine": 20,
        "show_progress": False,
        "log_dispatch_choices": False,
    }
    base.update(overrides)
    return base


@pytest.mark.tier0
def test_requested_chunk_size_below_one_raises():
    with pytest.raises(ValueError, match="requested_chunk_size must be >= 1"):
        run_lmm_chunk_source_numpy(lmm_mode=1, **_run_kwargs(requested_chunk_size=0))


@pytest.mark.tier0
def test_filtered_means_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        run_lmm_chunk_source_numpy(
            lmm_mode=1, **_run_kwargs(filtered_means=np.zeros(3))
        )


@pytest.mark.tier0
def test_score_mode_without_hi_eval_raises():
    with pytest.raises(RuntimeError, match="Score/All mode requires Hi_eval_null"):
        run_lmm_chunk_source_numpy(lmm_mode=3, **_run_kwargs(Hi_eval_null=None))


@pytest.mark.tier0
def test_lrt_mode_without_logl_h0_raises():
    with pytest.raises(RuntimeError, match="LRT/All mode requires logl_H0"):
        run_lmm_chunk_source_numpy(lmm_mode=2, **_run_kwargs(logl_H0=None))


@pytest.mark.tier0
def test_empty_filtered_returns_zeroed_stats():
    stats = run_lmm_chunk_source_numpy(
        lmm_mode=1, **_run_kwargs(n_filtered=0, filtered_means=np.zeros(0))
    )
    assert stats.processed == 0
    assert stats.n_chunks == 0
    assert stats.nan_counts == {}


# ---------------------------------------------------------------------------
# dispatch_soa_split mode guard
# ---------------------------------------------------------------------------


def _dispatch_soa_split_args(**overrides):
    n_samples = 4
    base = {
        "lmm_mode": 4,
        "use_fused_mode4": False,
        "lmm_workspace": None,
        "n_cvt": 1,
        "eigenvalues_np": np.ones(n_samples),
        "uab_var_soa": np.ones((2, 3, n_samples)),
        "uab_invariant_soa": np.ones((3, n_samples)),
        "n_samples": n_samples,
        "Hi_eval_null": np.ones(n_samples),
        "l_min": 1e-5,
        "l_max": 1e5,
        "n_grid": 50,
        "n_refine": 20,
        "logl_H0": -1.0,
        "n_threads": 1,
    }
    base.update(overrides)
    return base


@pytest.mark.tier0
def test_dispatch_soa_split_mode4_without_workspace_raises():
    with pytest.raises(ValueError, match="Unexpected lmm_mode=4"):
        dispatch_soa_split(**_dispatch_soa_split_args(lmm_mode=4, lmm_workspace=None))


@pytest.mark.tier0
def test_dispatch_soa_split_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unexpected lmm_mode=99"):
        dispatch_soa_split(**_dispatch_soa_split_args(lmm_mode=99))
