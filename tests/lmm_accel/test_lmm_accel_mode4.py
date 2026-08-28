"""_lmm_accel C extension tests: the mode-4 kernel that fuses Wald, Score and LRT.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.

The kernel under test is the fused mode-4 workspace, which is what
``DispatchPath.FUSED`` reaches for lmm_mode 4 at n_cvt=1. These checks used to
drive the SoA-split mode-4 workspace and compare it against
``_compose_mode4_from_split``. Neither is reachable: building an SoA-split
kernel for mode 1 or 4 raises, saying in as many words that they take the
fused kernel, and nothing in src called the compose helper at all. So the
comparison had a dead kernel on both sides.

Numerical agreement with NumPy across all eight outputs is checked in
test_lmm_accel_fused.py::TestFusedParity::test_mode4_parity. What is left here
is what that does not cover: the workspace API surface, thread determinism,
degenerate SNPs, capsule type safety, and that the shared coarse grid still
leaves REML and MLE with independent brackets.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import _c, compute_lmm_chunk_numpy
from tests.lmm_accel._helpers import _null_model_ncvt1

_MODE4_KEYS = (
    "lambdas",
    "logls",
    "betas",
    "ses",
    "pwalds",
    "lambdas_mle",
    "p_lrts",
    "p_scores",
)

# What lmm_mode=1 returns from the same compute: the Wald arrays without the
# Score and LRT three. Sliced from _MODE4_KEYS so the two cannot drift.
_WALD_KEYS = _MODE4_KEYS[:5]


def _mode4_workspace(fused_data):
    """Build the live fused mode-4 workspace and the genotypes to run it over.

    Driven from ``fused_data`` rather than ``score_lrt_data``. The latter builds
    a fresh w per SNP, so its Uab has no single invariant w for a fused kernel
    to take; feeding it one anyway pairs SNP 0's invariant columns with every
    other SNP's varying columns, which is a different problem from the one the
    fixture describes.
    """
    eigenvalues, w, Uty, utg_t, uab_inv_soa, _, n_samples = fused_data
    Hi_eval_null, logl_H0 = _null_model_ncvt1(eigenvalues, w, Uty)
    ws = _c().create_workspace_ncvt1_c(
        eigenvalues,
        uab_inv_soa,
        w,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        lmm_mode=4,
        hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
    )
    return ws, utg_t, utg_t.shape[0]


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_mode4_numpy_fallback_returns_all_keys(score_lrt_data, monkeypatch):
    """Mode 4 through the NumPy fallback returns all 8 keys, correctly shaped.

    ``compute_lmm_chunk_numpy`` is the full-Uab NumPy path, which the runner
    reaches only on ``NUMPY_FALLBACK``, and that is selected only when the
    extension is absent. The extension is cleared so this drives the path in the
    state production uses it in.
    """
    eigenvalues, Uab_batch, n_samples, Hi_eval_null, logl_H0 = score_lrt_data

    monkeypatch.setattr(compute_numpy, "_accel", None)

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

    for key in _MODE4_KEYS:
        arr = result[key]
        assert arr is not None, f"Mode 4 result['{key}'] should not be None"
        assert isinstance(arr, np.ndarray), f"result['{key}'] should be ndarray"
        assert arr.shape == (Uab_batch.shape[0],), (
            f"result['{key}'] shape mismatch: {arr.shape}"
        )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_mode4_fused_workspace_api(fused_data):
    """Fused mode-4 workspace creation and compute returns all 8 keys."""
    ws, utg_t, n_snps = _mode4_workspace(fused_data)
    assert ws is not None

    cr = _c().compute_lmm_chunk_fused_c(ws, utg_t, 1)

    for key in _MODE4_KEYS:
        assert key in cr, f"Missing key '{key}' in fused mode-4 result"
        assert isinstance(cr[key], np.ndarray), f"result['{key}'] should be ndarray"
        assert cr[key].shape == (n_snps,), (
            f"result['{key}'] shape {cr[key].shape} != ({n_snps},)"
        )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_mode4_shared_grid_preserves_distinct_reml_mle_brackets(fused_data):
    """The shared coarse grid still leaves REML and MLE with separate brackets.

    Mode 4 runs both optimisations in one pass over a single grid. If the two
    ever collapsed onto one bracket the kernel would be optimising once and
    reporting the answer twice, so the lambdas would come back exactly equal.
    Every SNP has to separate them.

    The floor is 1e-5 in log space. Measured separation on this fixture spans
    8.9e-4 to 2.7e-2, so that is roughly a hundredfold margin below the closest
    pair and eleven orders above the last-bit difference an exactly-equal pair
    would show. The direction is not asserted: REML exceeds MLE for only 8 of
    the 50 SNPs here, so there is no sign to pin.
    """
    ws, utg_t, _ = _mode4_workspace(fused_data)
    cr = _c().compute_lmm_chunk_fused_c(ws, utg_t, 1)

    log_separation = np.abs(np.log(cr["lambdas"]) - np.log(cr["lambdas_mle"]))

    assert np.all(log_separation > 1e-5), (
        f"{int(np.sum(log_separation <= 1e-5))} of {log_separation.size} SNPs "
        "give the same lambda for REML and MLE, which means the shared grid has "
        "collapsed the two brackets"
    )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_mode4_fused_degenerate_snps(fused_data):
    """A constant genotype gives NaN Wald and Score outputs, and p_lrt near 1."""
    ws, utg_t, _ = _mode4_workspace(fused_data)
    utg_degen = utg_t.copy()
    utg_degen[0, :] = 0.0

    cr = _c().compute_lmm_chunk_fused_c(ws, utg_degen, 1)

    for key in ("betas", "ses", "pwalds", "p_scores"):
        assert np.isnan(cr[key][0]), f"degenerate SNP should have NaN {key}"
    assert cr["p_lrts"][0] >= 0.99, (
        f"degenerate SNP p_lrt={cr['p_lrts'][0]}, expected near 1"
    )
    assert np.all(np.isfinite(cr["betas"][1:])), "non-degenerate betas should be finite"


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_wald_workspace_yields_wald_keys_only(fused_data):
    """One compute serves both modes, and the workspace decides what comes back.

    A workspace built with lmm_mode=1 gets the five Wald arrays; the same call
    against an lmm_mode=4 workspace gets those plus the Score and LRT three.
    Nothing but the workspace differs between the two calls.
    """
    eigenvalues, w, Uty, utg_t, uab_inv_soa, _, n_samples = fused_data

    wald_ws = _c().create_workspace_ncvt1_c(
        eigenvalues, uab_inv_soa, w, Uty, n_samples, 1e-5, 1e5, 50, 20, lmm_mode=1
    )
    wald_result = _c().compute_lmm_chunk_fused_c(wald_ws, utg_t, 1)
    assert set(wald_result) == set(_WALD_KEYS)

    mode4_ws, mode4_utg_t, _ = _mode4_workspace(fused_data)
    mode4_result = _c().compute_lmm_chunk_fused_c(mode4_ws, mode4_utg_t, 1)
    assert set(mode4_result) == set(_MODE4_KEYS)


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_mode4_fused_multithreaded_parity(fused_data):
    """Fused mode-4 is bitwise deterministic across thread counts.

    The kernel dispatches Wald, Score and LRT in a single pass over SNPs, so a
    race or thread-local state corruption in any of the three shows up here.
    """
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    ws, utg_t, _ = _mode4_workspace(fused_data)
    single = _c().compute_lmm_chunk_fused_c(ws, utg_t, 1)
    multi = _c().compute_lmm_chunk_fused_c(ws, utg_t, n_threads)

    for key in _MODE4_KEYS:
        np.testing.assert_array_equal(
            single[key],
            multi[key],
            err_msg=f"Mode-4 {key}: multi-thread vs single-thread mismatch",
        )
