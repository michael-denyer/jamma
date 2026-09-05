"""_lmm_accel C extension tests: stateless fused Score and LRT kernels.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from jamma.lmm.likelihood_numpy import (
    golden_section_optimize_lambda_mle_numpy,
)
from jamma.lmm.schema import LmmConfig
from tests.conftest import requires_c
from tests.lmm_accel._helpers import assert_fused_matches_reference

pytestmark = pytest.mark.tier0


@pytest.fixture
def _fused_score_lrt_null_model(split_wald_data):
    """Compute null-model Hi_eval and logl_H0 from split_wald_data.

    Unlike score_lrt_data (which derives from synthetic_wald_data),
    this computes the null model from the same UtW/Uty/eigenvalues
    used by the fused Score/LRT tests.
    """
    from jamma.lmm.uab import batch_compute_uab_numpy

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    # Build null Uab from UtW/Uty (no genotype)
    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Uab_null = np.zeros((1, n_samples, 6), dtype=np.float64)
    Uab_null[0, :, 0] = full_uab[0, :, 0]  # ww (invariant)
    Uab_null[0, :, 2] = full_uab[0, :, 2]  # wy (invariant)
    Uab_null[0, :, 5] = full_uab[0, :, 5]  # yy (invariant)

    lambdas_null, logls_null = golden_section_optimize_lambda_mle_numpy(
        1,
        eigenvalues,
        Uab_null,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    lambda_null = float(lambdas_null[0])
    logl_H0 = float(logls_null[0])
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)
    return Hi_eval_null, logl_H0


@requires_c
def test_abi_version_19():
    """ABI 20 requires the mode 2 alternative-model likelihood output."""
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION == 20


def _make_runner_test_data(rng, n_samples=50, n_snps=20):
    """Create synthetic data for runner dispatch tests."""
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.4, 0.4, 0.2])
    phenotypes = rng.standard_normal(n_samples)
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]
    U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]
    return eigenvalues, genotypes, phenotypes, snp_info, U


def _run_fused_score_lrt_dispatch(
    eigenvalues, genotypes, phenotypes, snp_info, U, lmm_mode
):
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    return run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=None,
        snp_info=snp_info,
        eigenvalues=eigenvalues,
        eigenvectors=U,
        config=LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=lmm_mode,
            n_refine=20,
        ),
    )


@requires_c
def test_runner_fused_score_dispatch():
    """Runner dispatches fused Score WS path for mode 3, matches SoA split."""
    rng = np.random.default_rng(200)
    data = _make_runner_test_data(rng)

    assert_fused_matches_reference(
        lambda: _run_fused_score_lrt_dispatch(*data, lmm_mode=3),
        fields={"p_score": 1e-12},
        min_count=10,
        label=" Score WS",
    )


@requires_c
def test_runner_fused_lrt_dispatch():
    """Runner dispatches fused LRT WS path for mode 2, matches SoA split."""
    rng = np.random.default_rng(201)
    data = _make_runner_test_data(rng)

    assert_fused_matches_reference(
        lambda: _run_fused_score_lrt_dispatch(*data, lmm_mode=2),
        fields={"p_lrt": 5e-5, "l_mle": 5e-5},
        min_count=10,
        label=" LRT WS",
    )


@requires_c
def test_runner_fused_score_chunk_size():
    """Fused Score uses 1-col accounting (6x larger chunks than NumPy fallback)."""
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
    from jamma.lmm.dispatch import DispatchPath

    n_samples = 1000
    n_filtered = 200_000
    # Budget large enough that both paths exceed the 100-SNP floor.
    # NUMPY_FALLBACK needs n_samples * 6 * 8 = 48KB/SNP (n_cvt=1, 6 Uab
    # columns); fused needs n_samples * 8 = 8KB/SNP (utg_t alone).
    budget = 16_000_000

    chunk_fused = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=budget,
    )
    chunk_fallback = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=budget,
    )

    assert chunk_fused >= 5 * chunk_fallback, (
        f"Fused chunk ({chunk_fused}) should be >= 5x NumPy fallback chunk "
        f"({chunk_fallback})"
    )


@requires_c
def test_runner_fused_lrt_chunk_size():
    """Fused LRT uses 1-col accounting (6x larger chunks than NumPy fallback)."""
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
    from jamma.lmm.dispatch import DispatchPath

    n_samples = 1000
    n_filtered = 200_000
    budget = 16_000_000  # Same budget as Score test

    chunk_fused = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=budget,
    )
    chunk_fallback = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=budget,
    )

    assert chunk_fused >= 5 * chunk_fallback, (
        f"Fused chunk ({chunk_fused}) should be >= 5x NumPy fallback chunk "
        f"({chunk_fallback})"
    )
