"""NumPy kernels consume rotated genotypes and preserve numerical results."""

import tracemalloc

import numpy as np
import pytest

from jamma.lmm.chunk_kernel import RunInvariants, make_kernel
from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy
from jamma.lmm.dispatch import select_dispatch_path
from jamma.lmm.prepare_common import PreparedLmmRun
from jamma.lmm.schema import LmmConfig
from jamma.lmm.uab import batch_compute_uab_numpy
from jamma.lmm.workspace import WorkspaceSpec

pytestmark = pytest.mark.tier0


def test_numpy_wald_raw_chunk_matches_full_product_reference():
    rng = np.random.default_rng(62)
    n, m = 256, 32
    eigenvalues = np.linspace(0.1, 2.0, n)
    u, _ = np.linalg.qr(rng.normal(size=(n, n)))
    w = u.T @ np.ones((n, 1))
    y = u.T @ rng.normal(size=n)
    utg = np.ascontiguousarray(rng.normal(size=(m, n)))
    prepared = PreparedLmmRun(
        eigenvalues=eigenvalues,
        U=u,
        UtW=w,
        Uty=y,
        logl_H0=-100.0,
        Hi_eval_null=1 / (1 + eigenvalues),
        pve=None,
        pve_se=None,
    )
    config = LmmConfig(show_progress=False)
    dispatch = select_dispatch_path(1, 1, accel=False)
    kernel = make_kernel(
        RunInvariants.build(dispatch, prepared, config, m),
        WorkspaceSpec.build(dispatch, 1, n, n, 1, config.n_grid, config.n_refine, 1),
    )
    tracemalloc.start()
    try:
        expected = compute_lmm_chunk_numpy(
            1, 1, eigenvalues, batch_compute_uab_numpy(1, w, y, utg), n
        )
        _, full_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    tracemalloc.start()
    try:
        actual = kernel.compute_chunk(utg, 1, 0)
        _, split_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Six full product columns no longer coexist with the varying tensor.
    assert full_peak - split_peak >= n * m * 6 * 8 * 0.9
    for key in ("betas", "ses", "pwalds", "lambdas", "logls"):
        np.testing.assert_array_equal(actual[key], expected[key])


def test_general_invariant_preparation_has_bounded_peak_memory():
    import json
    import subprocess
    import sys

    # A fresh process measures cold preparation, including any cached tables.
    script = """
import json
import tracemalloc
import numpy as np
from jamma.lmm.uab import compute_uab_invariant_soa
w = np.ones((16, 97))
y = np.arange(16.0)
tracemalloc.start()
result = compute_uab_invariant_soa(w, y, 97)
_, peak = tracemalloc.get_traced_memory()
print(json.dumps({"peak": peak, "payload": result.nbytes}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    measured = json.loads(result.stdout)
    assert measured["peak"] < 2 * measured["payload"] + 1_000_000
