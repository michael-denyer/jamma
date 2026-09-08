"""NumPy kernels consume rotated genotypes and preserve numerical results."""

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
    expected = compute_lmm_chunk_numpy(
        1, 1, eigenvalues, batch_compute_uab_numpy(1, w, y, utg), n
    )
    actual = kernel.compute_chunk(utg, 1, 0)
    for key in ("betas", "ses", "pwalds", "lambdas", "logls"):
        np.testing.assert_array_equal(actual[key], expected[key])
