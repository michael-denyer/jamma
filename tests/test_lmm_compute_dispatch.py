"""Focused dispatch-boundary tests for jamma.lmm.compute_numpy."""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.dispatch import DispatchPath

pytestmark = pytest.mark.tier0

# Stands in for a loaded extension. Only `is not None` is read on
# the paths under test, so the object's identity is all that matters.
_EXTENSION_LOADED = object()


@pytest.mark.parametrize("n_cvt", [2, 76, compute_numpy.MAX_C_N_CVT])
def test_wald_resolves_to_fused_general_through_ncvt_limit(monkeypatch, n_cvt):
    """Wald routes to the fused general C kernel for every n_cvt up to the limit.

    This used to assert that _compute_wald_numpy took a general C branch. That
    branch could not run: the runner reaches _compute_wald_numpy only on
    NUMPY_FALLBACK, which is selected only when the extension is absent. The
    decision the runner actually makes is this one.
    """
    monkeypatch.setattr(compute_numpy, "_accel", _EXTENSION_LOADED)

    assert (
        compute_numpy.select_current_dispatch_path(n_cvt, 1, log_choices=False)
        is DispatchPath.FUSED_GENERAL
    )


@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_ncvt_beyond_the_limit_is_rejected_by_the_kernel():
    """Past MAX_C_N_CVT the kernel refuses rather than the dispatcher diverting.

    Nothing in Python bounds n_cvt any more. The guard that did lived in
    _compute_wald_numpy, on a branch the runner never reached, so the C
    kernel's own check is the enforcement and this pins that it fires. The
    general workspace creator parses the Pab table before anything else, so
    it is the entry point that raises.
    """
    from jamma.lmm._lmm_accel import create_workspace_general_c
    from jamma.lmm.likelihood import build_pab_table_for_c

    n_cvt = compute_numpy.MAX_C_N_CVT + 1
    n_samples = 200

    rng = np.random.default_rng(777)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))[::-1]
    pab_table = build_pab_table_for_c(n_cvt)._asdict()

    with pytest.raises(ValueError, match=r"n_cvt must be 1\.\.100, got 101"):
        create_workspace_general_c(
            eigenvalues,
            np.zeros((pab_table["n_inv"], n_samples), dtype=np.float64),
            np.zeros((n_samples, n_cvt), dtype=np.float64),
            np.zeros(n_samples, dtype=np.float64),
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
            pab_table,
            lmm_mode=1,
        )


@pytest.mark.parametrize(
    "helper",
    ["_compute_wald_numpy", "_compute_lrt_numpy", "_compute_score_numpy"],
)
def test_full_uab_helpers_never_touch_the_extension(monkeypatch, helper):
    """The full-Uab helpers are pure NumPy, and must stay that way.

    Each used to open with an `if _accel is not None` ladder into a C kernel.
    None of those could run: the three are reached only through
    compute_lmm_chunk_numpy, the runner calls that only on NUMPY_FALLBACK, and
    that path is selected only when the extension is absent. So the ladders were
    dead by construction and have been removed.

    Reaching the extension is detected by making `_c` raise. It is the single
    accessor, so a helper that calls it fails loudly here rather than quietly
    reintroducing a branch no caller can reach.
    """

    def _extension_is_off_limits():
        raise AssertionError(
            f"{helper} reached the C extension. It is only ever called when the "
            "extension is absent, so a C branch here is unreachable."
        )

    monkeypatch.setattr(compute_numpy, "_accel", _EXTENSION_LOADED)
    # allow-patch: sentinel-on-call is the assertion. _c is the single
    # accessor, so raising there is how "never reaches C" is detected.
    monkeypatch.setattr(  # allow-patch: see above
        compute_numpy, "_c", _extension_is_off_limits
    )

    n_cvt, n_samples, n_snps = 2, 40, 3
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    rng = np.random.default_rng(4)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    UtW = np.abs(rng.standard_normal((n_samples, n_cvt))) + 0.5
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    from jamma.lmm.likelihood import compute_Uab

    Uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    for i in range(n_snps):
        Uab_batch[i] = compute_Uab(UtW, Uty, UtG[:, i])

    common = (n_cvt, eigenvalues)
    if helper == "_compute_wald_numpy":
        compute_numpy._compute_wald_numpy(
            *common, Uab_batch, n_samples, 1e-5, 1e5, 50, 20
        )
    elif helper == "_compute_lrt_numpy":
        compute_numpy._compute_lrt_numpy(*common, Uab_batch, 1e-5, 1e5, 50, 20, -100.0)
    else:
        compute_numpy._compute_score_numpy(
            *common, 1.0 / (0.5 * eigenvalues + 1.0), Uab_batch, n_samples
        )
