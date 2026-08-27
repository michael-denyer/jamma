"""_lmm_accel C extension tests: stateless fused Score and LRT kernels.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.likelihood_numpy import (
    golden_section_optimize_lambda_mle_numpy,
)
from jamma.lmm.schema import LmmConfig

_score_fused_available = compute_numpy._accel is not None

_lrt_fused_available = compute_numpy._accel is not None


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


@pytest.mark.tier0
def test_abi_version_12():
    """ABI_VERSION is 12 after the unreachable entry points were removed."""
    if compute_numpy._accel is None:
        pytest.skip("C extension not available")
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION == 12


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


@pytest.mark.skipif(not _score_fused_available, reason="Fused Score C not available")
def test_runner_fused_score_dispatch():
    """Runner dispatches fused Score WS path for mode 3, matches SoA split."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy
    from jamma.lmm.compute_numpy import _c

    assert compute_numpy._accel is not None
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(200)
    eigenvalues, genotypes, phenotypes, snp_info, U = _make_runner_test_data(rng)

    with patch(
        "jamma.lmm.compute_numpy._accel.compute_score_fused_ws_c",
        wraps=_c().compute_score_fused_ws_c,
    ) as mock_fused:
        result_fused = run_lmm_association_numpy(
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
                lmm_mode=3,
                n_refine=20,
            ),
        )
    assert mock_fused.called, "Fused Score WS C function was not called"

    # SoA split path (disable all fused Score variants)
    with patch("jamma.lmm.compute_numpy._accel", None):
        result_split = run_lmm_association_numpy(
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
                lmm_mode=3,
                n_refine=20,
            ),
        )

    fused = result_fused.associations
    split = result_split.associations

    assert len(fused) == len(split), f"Count mismatch: {len(fused)} vs {len(split)}"
    assert len(fused) > 10, f"Too many SNPs filtered: {len(fused)}"

    for a_f, a_s in zip(fused, split, strict=True):
        assert a_f.rs == a_s.rs
        if a_f.p_score is not None and a_s.p_score is not None:
            np.testing.assert_allclose(
                a_f.p_score,
                a_s.p_score,
                rtol=1e-12,
                err_msg=f"p_score mismatch for {a_f.rs}",
            )


@pytest.mark.skipif(not _lrt_fused_available, reason="Fused LRT C not available")
def test_runner_fused_lrt_dispatch():
    """Runner dispatches fused LRT WS path for mode 2, matches SoA split."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy
    from jamma.lmm.compute_numpy import _c

    assert compute_numpy._accel is not None
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(201)
    eigenvalues, genotypes, phenotypes, snp_info, U = _make_runner_test_data(rng)

    with patch(
        "jamma.lmm.compute_numpy._accel.compute_lrt_fused_ws_c",
        wraps=_c().compute_lrt_fused_ws_c,
    ) as mock_fused:
        result_fused = run_lmm_association_numpy(
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
                lmm_mode=2,
                n_refine=20,
            ),
        )
    assert mock_fused.called, "Fused LRT WS C function was not called"

    # SoA split path (disable all fused LRT variants)
    with patch("jamma.lmm.compute_numpy._accel", None):
        result_split = run_lmm_association_numpy(
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
                lmm_mode=2,
                n_refine=20,
            ),
        )

    fused = result_fused.associations
    split = result_split.associations

    assert len(fused) == len(split), f"Count mismatch: {len(fused)} vs {len(split)}"
    assert len(fused) > 10, f"Too many SNPs filtered: {len(fused)}"

    for a_f, a_s in zip(fused, split, strict=True):
        assert a_f.rs == a_s.rs
        if a_f.p_lrt is not None and a_s.p_lrt is not None:
            np.testing.assert_allclose(
                a_f.p_lrt,
                a_s.p_lrt,
                rtol=5e-5,
                err_msg=f"p_lrt mismatch for {a_f.rs}",
            )
        if a_f.l_mle is not None and a_s.l_mle is not None:
            np.testing.assert_allclose(
                a_f.l_mle,
                a_s.l_mle,
                rtol=5e-5,
                err_msg=f"l_mle mismatch for {a_f.rs}",
            )


@pytest.mark.skipif(not _score_fused_available, reason="Fused Score C not available")
def test_runner_fused_score_chunk_size():
    """Fused Score uses 1-col accounting (4x larger chunks at same budget)."""
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
    from jamma.lmm.dispatch import DispatchPath

    n_samples = 1000
    n_filtered = 200_000
    # Budget large enough that both paths exceed the 100-SNP floor.
    # Split needs n_samples * 4 * 8 = 32KB/SNP; fused needs n_samples * 8 = 8KB/SNP.
    # At 16 MB: split → 500 SNPs, fused → 2000 SNPs.
    budget = 16_000_000

    chunk_fused = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        dispatch=DispatchPath.FUSED_SCORE_WS,
        mem_budget_bytes=budget,
    )
    chunk_split = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        dispatch=DispatchPath.SOA_SPLIT,
        mem_budget_bytes=budget,
    )

    assert chunk_fused >= 3 * chunk_split, (
        f"Fused chunk ({chunk_fused}) should be >= 3x split chunk ({chunk_split})"
    )


@pytest.mark.skipif(not _lrt_fused_available, reason="Fused LRT C not available")
def test_runner_fused_lrt_chunk_size():
    """Fused LRT uses 1-col accounting (4x larger chunks at same budget)."""
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
    from jamma.lmm.dispatch import DispatchPath

    n_samples = 1000
    n_filtered = 200_000
    budget = 16_000_000  # Same budget as Score test

    chunk_fused = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        dispatch=DispatchPath.FUSED_LRT_WS,
        mem_budget_bytes=budget,
    )
    chunk_split = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        dispatch=DispatchPath.SOA_SPLIT,
        mem_budget_bytes=budget,
    )

    assert chunk_fused >= 3 * chunk_split, (
        f"Fused chunk ({chunk_fused}) should be >= 3x split chunk ({chunk_split})"
    )
