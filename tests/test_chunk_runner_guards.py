"""Guard/precondition tests for the shared NumPy LMM chunk runner.

These cover the cheap, isolated failure paths that the end-to-end parity suites
never exercise: the ``run_lmm_chunk_source_numpy`` argument preconditions and
the SoA-split kernel's mode guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.lmm.chunk_kernel import RunInvariants, make_kernel
from jamma.lmm.chunk_runner_numpy import run_lmm_chunk_source_numpy
from jamma.lmm.dispatch import DispatchPath

pytestmark = pytest.mark.tier0

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


def test_max_chunk_size_below_one_raises():
    with pytest.raises(ValueError, match="max_chunk_size must be >= 1"):
        run_lmm_chunk_source_numpy(lmm_mode=1, **_run_kwargs(max_chunk_size=0))


def test_filtered_means_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        run_lmm_chunk_source_numpy(
            lmm_mode=1, **_run_kwargs(filtered_means=np.zeros(3))
        )


def test_score_mode_without_hi_eval_raises():
    with pytest.raises(RuntimeError, match="Score/All mode requires Hi_eval_null"):
        run_lmm_chunk_source_numpy(lmm_mode=3, **_run_kwargs(Hi_eval_null=None))


def test_lrt_mode_without_logl_h0_raises():
    with pytest.raises(RuntimeError, match="LRT/All mode requires logl_H0"):
        run_lmm_chunk_source_numpy(lmm_mode=2, **_run_kwargs(logl_H0=None))


def test_empty_filtered_returns_zeroed_stats():
    """No SNPs means no work and no time spent, on every field the caller reads."""
    stats = run_lmm_chunk_source_numpy(
        lmm_mode=1, **_run_kwargs(n_filtered=0, filtered_means=np.zeros(0))
    )
    assert stats == (0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# SoA-split kernel mode guard
# ---------------------------------------------------------------------------


def _forced_split_invariants(lmm_mode):
    """RunInvariants pinned to the split path, whatever the mode would select."""
    n_samples = 4
    return RunInvariants.build(
        dispatch=DispatchPath.SOA_SPLIT,
        lmm_mode=lmm_mode,
        n_cvt=1,
        n_samples=n_samples,
        n_filtered=5,
        eigenvalues=np.ones(n_samples),
        UtW=np.ones((n_samples, 1)),
        Uty=np.ones(n_samples),
        Hi_eval_null=np.ones(n_samples),
        logl_H0=-1.0,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
    )


def test_soa_split_kernel_rejects_modes_it_cannot_serve():
    """Only modes 2 and 3 reach this path; 1 and 4 take the fused general kernel.

    The guard fires when the kernel is built, before any chunk arrives, so a
    dispatch table that ever routed a Wald or mode-4 run here fails on the
    first line of the run rather than partway through the loop.
    """
    for mode in (1, 4, 99):
        with pytest.raises(ValueError, match=f"Unexpected lmm_mode={mode}"):
            make_kernel(_forced_split_invariants(mode), 1)


def test_shared_chunk_entry_resets_the_p_yy_warning():
    """Every runner reaches this entry, so the per-run reset belongs here.

    The flag deduplicates the negative-P_yy warning within one run. Resetting
    it in the batch runner only meant a streaming or LOCO run that followed
    one in the same process never warned again.
    """
    from jamma.lmm import likelihood

    likelihood._p_yy_state.warned = True
    run_lmm_chunk_source_numpy(
        lmm_mode=1, **_run_kwargs(n_filtered=0, filtered_means=np.zeros(0))
    )
    assert getattr(likelihood._p_yy_state, "warned", False) is False
