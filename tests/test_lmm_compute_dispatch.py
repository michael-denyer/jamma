"""Focused dispatch-boundary tests for jamma.lmm.compute_numpy."""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm import accel

pytestmark = pytest.mark.tier0

# Stands in for a loaded extension. Only `is not None` is read on
# the paths under test, so the object's identity is all that matters.
_EXTENSION_LOADED = object()


@pytest.mark.parametrize(
    "helper",
    ["compute_wald_numpy", "compute_lrt_numpy", "compute_score_numpy"],
)
def test_full_uab_helpers_never_touch_the_extension(monkeypatch, helper):
    """The full-Uab helpers are pure NumPy, and must stay that way.

    Each used to open with an `if _accel is not None` ladder into a C kernel.
    None of those could run: the three are reached only through
    compute_lmm_chunk_numpy, the runner calls that only on NUMPY_FALLBACK, and
    that path is selected only when the extension is absent. So the ladders were
    dead by construction and have been removed.

    Reaching the extension is detected by making `accel.require` raise. It is
    the single accessor, so a helper that calls it fails loudly here rather
    than quietly reintroducing a branch no caller can reach.
    """

    def _extension_is_off_limits():
        raise AssertionError(
            f"{helper} reached the C extension. It is only ever called when the "
            "extension is absent, so a C branch here is unreachable."
        )

    monkeypatch.setattr(accel, "_accel", _EXTENSION_LOADED)
    # allow-patch: sentinel-on-call is the assertion. accel.require is the
    # single accessor, so raising there is how "never reaches C" is detected.
    monkeypatch.setattr(
        accel, "require", _extension_is_off_limits
    )  # allow-patch: see above

    n_cvt, n_samples, n_snps = 2, 40, 3
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    rng = np.random.default_rng(4)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    UtW = np.abs(rng.standard_normal((n_samples, n_cvt))) + 0.5
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    from jamma.lmm.pab import compute_Uab

    Uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    for i in range(n_snps):
        Uab_batch[i] = compute_Uab(UtW, Uty, UtG[:, i])

    common = (n_cvt, eigenvalues)
    if helper == "compute_wald_numpy":
        compute_numpy.compute_wald_numpy(
            *common, Uab_batch, n_samples, 1e-5, 1e5, 50, 20
        )
    elif helper == "compute_lrt_numpy":
        compute_numpy.compute_lrt_numpy(*common, Uab_batch, 1e-5, 1e5, 50, 20, -100.0)
    else:
        compute_numpy.compute_score_numpy(
            *common, 1.0 / (0.5 * eigenvalues + 1.0), Uab_batch, n_samples
        )
