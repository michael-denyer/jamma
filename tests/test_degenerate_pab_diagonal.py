"""A non-positive Pab diagonal makes the REML log-likelihood NaN in every family.

GEMMA's ``LogRL_f`` (lmm.cpp) takes ``log`` of every Pab and Iab diagonal
entry with no guard, so a constant-genotype SNP (P_xx = 0) never yields a
finite REML likelihood there. JAMMA's n_cvt=1 C path and the NumPy path used
to skip the term and report a finite value while the general C path returned
NaN.
"""

import numpy as np
import pytest

from jamma.lmm import accel
from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy
from jamma.lmm.uab import batch_compute_uab_numpy, compute_uab_invariant_soa
from tests.support import requires_c

pytestmark = pytest.mark.tier0

N_SAMPLES = 120
N_POLYMORPHIC = 3


def _inputs(n_cvt: int):
    rng = np.random.default_rng(7)
    a = rng.standard_normal((N_SAMPLES, 40))
    k = a @ a.T / 40.0
    eigenvalues, u = np.linalg.eigh(k)
    y = rng.standard_normal(N_SAMPLES)
    w = np.ones((N_SAMPLES, n_cvt))
    if n_cvt > 1:
        w[:, 1:] = rng.standard_normal((N_SAMPLES, n_cvt - 1))
    g = rng.integers(0, 3, size=(N_SAMPLES, N_POLYMORPHIC)).astype(np.float64)
    g = np.column_stack([g, np.zeros(N_SAMPLES)])
    utg_t = np.ascontiguousarray((u.T @ g).T)
    return eigenvalues, u.T @ w, u.T @ y, utg_t


def _wald_c(n_cvt: int) -> dict[str, np.ndarray]:
    eigenvalues, utw, uty, utg_t = _inputs(n_cvt)
    ws = accel.require().create_workspace_c(
        eigenvalues,
        compute_uab_invariant_soa(utw, uty, n_cvt=n_cvt),
        utw,
        uty,
        N_SAMPLES,
        1e-5,
        1e5,
        50,
        20,
        1,
        n_cvt,
        lmm_mode=1,
    )
    return accel.require().compute_lmm_chunk_c(ws, utg_t, 1)


def _wald_numpy(n_cvt: int) -> dict[str, np.ndarray]:
    eigenvalues, utw, uty, utg_t = _inputs(n_cvt)
    return compute_lmm_chunk_numpy(
        1,
        n_cvt,
        eigenvalues,
        batch_compute_uab_numpy(n_cvt, utw, uty, utg_t),
        N_SAMPLES,
        Hi_eval_null=np.ones(N_SAMPLES),
        logl_H0=0.0,
    )


_SKIPS_TERM = pytest.mark.xfail(
    strict=True, reason="skips the non-positive Pab diagonal, reports a finite logl"
)


@pytest.mark.parametrize(
    ("n_cvt", "wald"),
    [
        pytest.param(1, _wald_c, id="c-ncvt1", marks=[requires_c, _SKIPS_TERM]),
        pytest.param(2, _wald_c, id="c-ncvt2", marks=requires_c),
        pytest.param(1, _wald_numpy, id="numpy-ncvt1", marks=_SKIPS_TERM),
        pytest.param(2, _wald_numpy, id="numpy-ncvt2", marks=_SKIPS_TERM),
    ],
)
def test_constant_snp_reml_logl_is_nan(n_cvt, wald):
    result = wald(n_cvt)
    assert np.all(np.isfinite(result["logls"][:N_POLYMORPHIC]))
    for key in ("logls", "betas", "ses", "pwalds"):
        assert np.isnan(result[key][N_POLYMORPHIC]), key
