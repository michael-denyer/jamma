"""_lmm_accel C extension tests: stateless fused Score and LRT kernels.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
)
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
    golden_section_optimize_lambda_mle_numpy,
)
from jamma.lmm.schema import LmmConfig

_score_fused_available = _C_ACCEL_AVAILABLE and getattr(
    compute_numpy, "_C_SCORE_FUSED_AVAILABLE", False
)

_lrt_fused_available = _C_ACCEL_AVAILABLE and getattr(
    compute_numpy, "_C_LRT_FUSED_AVAILABLE", False
)


@pytest.fixture
def _fused_score_lrt_null_model(split_wald_data):
    """Compute null-model Hi_eval and logl_H0 from split_wald_data.

    Unlike score_lrt_data (which derives from synthetic_wald_data),
    this computes the null model from the same UtW/Uty/eigenvalues
    used by the fused Score/LRT tests.
    """
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    # Build null Uab from UtW/Uty (no genotype)
    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG)
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


class TestFusedScoreParity:
    """Verify compute_score_fused_c matches compute_score_split_c."""

    @pytest.fixture
    def fused_score_data(self, split_wald_data, _fused_score_lrt_null_model):
        """Prepare data for Score fused vs split parity."""
        eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
        Hi_eval_null, logl_H0 = _fused_score_lrt_null_model

        w = UtW[:, 0].copy()
        utg_t = np.ascontiguousarray(UtG.T)
        uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
        uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

        return (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            Hi_eval_null,
            n_samples,
            n_snps,
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_available,
        reason="Score fused C not available",
    )
    def test_score_fused_parity(self, fused_score_data):
        """Fused Score matches split Score to rtol=1e-12."""
        from jamma.lmm.compute_numpy import (
            _compute_score_fused_c,
            _compute_score_split_c,
        )

        assert _compute_score_fused_c is not None
        assert _compute_score_split_c is not None

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            Hi_eval_null,
            n_samples,
            n_snps,
        ) = fused_score_data

        # Split reference
        split_result = _compute_score_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            Hi_eval_null,
            n_samples,
            1,
        )

        # Fused
        fused_result = _compute_score_fused_c(
            utg_t,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )

        for key in ("betas", "ses", "p_scores"):
            np.testing.assert_allclose(
                fused_result[key],
                split_result[key],
                rtol=1e-12,
                atol=0,
                err_msg=f"Score {key}: fused vs split mismatch",
            )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_available,
        reason="Score fused C not available",
    )
    def test_score_fused_degenerate_snps(self, fused_score_data):
        """Constant genotype produces NaN beta/se/p_score."""
        from jamma.lmm.compute_numpy import _compute_score_fused_c

        assert _compute_score_fused_c is not None  # narrowed: skipif gates this

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            _,
            Hi_eval_null,
            n_samples,
            n_snps,
        ) = fused_score_data

        utg_degen = utg_t.copy()
        utg_degen[0, :] = 0.0  # constant genotype

        result = _compute_score_fused_c(
            utg_degen,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            1,
        )

        assert np.isnan(result["betas"][0]), "degenerate SNP: NaN beta"
        assert np.isnan(result["ses"][0]), "degenerate SNP: NaN se"
        assert np.isnan(result["p_scores"][0]), "degenerate SNP: NaN p_score"

        # Non-degenerate SNPs should be finite
        assert np.all(np.isfinite(result["betas"][1:])), (
            "non-degenerate betas should be finite"
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _score_fused_available,
        reason="Score fused C not available",
    )
    def test_score_fused_multithreaded(self, fused_score_data):
        """Fused Score with n_threads=2 matches split Score."""
        from jamma.lmm.compute_numpy import (
            _compute_score_fused_c,
            _compute_score_split_c,
        )

        assert _compute_score_fused_c is not None
        assert _compute_score_split_c is not None

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            Hi_eval_null,
            n_samples,
            n_snps,
        ) = fused_score_data

        split_result = _compute_score_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            Hi_eval_null,
            n_samples,
            1,
        )

        fused_result = _compute_score_fused_c(
            utg_t,
            w,
            Uty,
            Hi_eval_null,
            uab_inv_soa,
            eigenvalues,
            n_samples,
            2,
        )

        for key in ("betas", "ses", "p_scores"):
            np.testing.assert_allclose(
                fused_result[key],
                split_result[key],
                rtol=1e-12,
                atol=0,
                err_msg=f"Score {key}: fused(2t) vs split mismatch",
            )


class TestFusedLrtParity:
    """Verify compute_lrt_fused_c matches compute_lrt_split_c."""

    @pytest.fixture
    def fused_lrt_data(self, split_wald_data, _fused_score_lrt_null_model):
        """Prepare data for LRT fused vs split parity."""
        eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
        _, logl_H0 = _fused_score_lrt_null_model

        w = UtW[:, 0].copy()
        utg_t = np.ascontiguousarray(UtG.T)
        uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
        uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

        return (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            logl_H0,
            n_samples,
            n_snps,
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_available,
        reason="LRT fused C not available",
    )
    def test_lrt_fused_parity(self, fused_lrt_data):
        """Fused LRT matches split LRT to rtol=5e-5."""
        from jamma.lmm.compute_numpy import (
            _compute_lrt_fused_c,
            _compute_lrt_split_c,
        )

        assert _compute_lrt_fused_c is not None
        assert _compute_lrt_split_c is not None

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            logl_H0,
            n_samples,
            n_snps,
        ) = fused_lrt_data

        # Split reference
        split_result = _compute_lrt_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            1,
        )

        # Fused
        fused_result = _compute_lrt_fused_c(
            utg_t,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            1,
        )

        np.testing.assert_allclose(
            fused_result["lambdas_mle"],
            split_result["lambdas_mle"],
            rtol=5e-5,
            atol=0,
            err_msg="LRT lambdas_mle: fused vs split mismatch",
        )
        np.testing.assert_allclose(
            fused_result["p_lrts"],
            split_result["p_lrts"],
            rtol=5e-5,
            atol=0,
            err_msg="LRT p_lrts: fused vs split mismatch",
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_available,
        reason="LRT fused C not available",
    )
    def test_lrt_fused_degenerate_snps(self, fused_lrt_data):
        """Constant genotype produces NaN lambda_mle and p_lrt=1.0."""
        from jamma.lmm.compute_numpy import _compute_lrt_fused_c

        assert _compute_lrt_fused_c is not None  # narrowed: skipif gates this

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            _,
            logl_H0,
            n_samples,
            n_snps,
        ) = fused_lrt_data

        utg_degen = utg_t.copy()
        utg_degen[0, :] = 0.0

        result = _compute_lrt_fused_c(
            utg_degen,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            1,
        )

        # Degenerate SNP: LRT stat ~ 0, so p_lrt ~ 1.0 (chi2_sf(0) = 1)
        # lambda_mle can be anything (optimization on flat surface)
        assert result["p_lrts"][0] >= 0.99, (
            f"degenerate SNP p_lrt={result['p_lrts'][0]}, expected ~1.0"
        )

        # Non-degenerate SNPs should be finite
        assert np.all(np.isfinite(result["p_lrts"][1:])), (
            "non-degenerate p_lrts should be finite"
        )

    @pytest.mark.tier0
    @pytest.mark.skipif(
        not _lrt_fused_available,
        reason="LRT fused C not available",
    )
    def test_lrt_fused_multithreaded(self, fused_lrt_data):
        """Fused LRT with n_threads=2 matches split LRT."""
        from jamma.lmm.compute_numpy import (
            _compute_lrt_fused_c,
            _compute_lrt_split_c,
        )

        assert _compute_lrt_fused_c is not None
        assert _compute_lrt_split_c is not None

        (
            eigenvalues,
            w,
            Uty,
            utg_t,
            uab_inv_soa,
            uab_var_soa,
            logl_H0,
            n_samples,
            n_snps,
        ) = fused_lrt_data

        split_result = _compute_lrt_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            1,
        )

        fused_result = _compute_lrt_fused_c(
            utg_t,
            w,
            Uty,
            eigenvalues,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            logl_H0,
            2,
        )

        np.testing.assert_allclose(
            fused_result["lambdas_mle"],
            split_result["lambdas_mle"],
            rtol=5e-5,
            atol=0,
            err_msg="LRT lambdas_mle: fused(2t) vs split mismatch",
        )
        np.testing.assert_allclose(
            fused_result["p_lrts"],
            split_result["p_lrts"],
            rtol=5e-5,
            atol=0,
            err_msg="LRT p_lrts: fused(2t) vs split mismatch",
        )


@pytest.mark.tier0
def test_abi_version_11():
    """ABI_VERSION is 11 after persistent Score/LRT workspace addition."""
    if not _C_ACCEL_AVAILABLE:
        pytest.skip("C extension not available")
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION == 11


@pytest.mark.tier0
def test_fused_score_available_flag():
    """_C_SCORE_FUSED_AVAILABLE is True when C extension loaded."""
    if not _C_ACCEL_AVAILABLE:
        pytest.skip("C extension not available")
    assert compute_numpy._C_SCORE_FUSED_AVAILABLE is True


@pytest.mark.tier0
def test_fused_lrt_available_flag():
    """_C_LRT_FUSED_AVAILABLE is True when C extension loaded."""
    if not _C_ACCEL_AVAILABLE:
        pytest.skip("C extension not available")
    assert compute_numpy._C_LRT_FUSED_AVAILABLE is True


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
    """Runner dispatches fused Score path for mode 3, matches SoA split.

    Prefers workspace-based dispatch when available; falls back to stateless.
    """
    from unittest.mock import patch

    from jamma.lmm.compute_numpy import (
        _C_SCORE_FUSED_WS_AVAILABLE,
        _compute_score_fused_c,
        _compute_score_fused_ws_c,
    )

    assert _C_SCORE_FUSED_WS_AVAILABLE is not None
    assert _compute_score_fused_c is not None
    assert _compute_score_fused_ws_c is not None
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(200)
    eigenvalues, genotypes, phenotypes, snp_info, U = _make_runner_test_data(rng)

    # Fused Score path (default) — verify the fused C function is actually called.
    # Workspace path is preferred when available; stateless is the fallback.
    if _C_SCORE_FUSED_WS_AVAILABLE:
        with patch(
            "jamma.lmm.compute_numpy._compute_score_fused_ws_c",
            wraps=_compute_score_fused_ws_c,
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
    else:
        with patch(
            "jamma.lmm.compute_numpy._compute_score_fused_c",
            wraps=_compute_score_fused_c,
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
        assert mock_fused.called, "Fused Score C function was not called"

    # SoA split path (disable all fused Score variants)
    with (
        patch("jamma.lmm.compute_numpy._C_SCORE_FUSED_AVAILABLE", False),
        patch("jamma.lmm.compute_numpy._C_SCORE_FUSED_WS_AVAILABLE", False),
    ):
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
    """Runner dispatches fused LRT path for mode 2, matches SoA split.

    Prefers workspace-based dispatch when available; falls back to stateless.
    """
    from unittest.mock import patch

    from jamma.lmm.compute_numpy import (
        _C_LRT_FUSED_WS_AVAILABLE,
        _compute_lrt_fused_c,
        _compute_lrt_fused_ws_c,
    )

    assert _C_LRT_FUSED_WS_AVAILABLE is not None
    assert _compute_lrt_fused_c is not None
    assert _compute_lrt_fused_ws_c is not None
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(201)
    eigenvalues, genotypes, phenotypes, snp_info, U = _make_runner_test_data(rng)

    # Fused LRT path (default) — verify the fused C function is actually called.
    # Workspace path is preferred when available; stateless is the fallback.
    if _C_LRT_FUSED_WS_AVAILABLE:
        with patch(
            "jamma.lmm.compute_numpy._compute_lrt_fused_ws_c",
            wraps=_compute_lrt_fused_ws_c,
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
    else:
        with patch(
            "jamma.lmm.compute_numpy._compute_lrt_fused_c",
            wraps=_compute_lrt_fused_c,
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
        assert mock_fused.called, "Fused LRT C function was not called"

    # SoA split path (disable all fused LRT variants)
    with (
        patch("jamma.lmm.compute_numpy._C_LRT_FUSED_AVAILABLE", False),
        patch("jamma.lmm.compute_numpy._C_LRT_FUSED_WS_AVAILABLE", False),
    ):
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
        use_split=True,
        lmm_mode=3,
        mem_budget_bytes=budget,
    )

    from unittest.mock import patch

    with patch("jamma.lmm.compute_numpy._C_SCORE_FUSED_AVAILABLE", False):
        chunk_split = compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt=1,
            use_split=True,
            lmm_mode=3,
            mem_budget_bytes=budget,
        )

    assert chunk_fused >= 3 * chunk_split, (
        f"Fused chunk ({chunk_fused}) should be >= 3x split chunk ({chunk_split})"
    )


@pytest.mark.skipif(not _lrt_fused_available, reason="Fused LRT C not available")
def test_runner_fused_lrt_chunk_size():
    """Fused LRT uses 1-col accounting (4x larger chunks at same budget)."""
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy

    n_samples = 1000
    n_filtered = 200_000
    budget = 16_000_000  # Same budget as Score test

    chunk_fused = compute_chunk_size_numpy(
        n_samples,
        n_filtered,
        n_cvt=1,
        use_split=True,
        lmm_mode=2,
        mem_budget_bytes=budget,
    )

    from unittest.mock import patch

    with patch("jamma.lmm.compute_numpy._C_LRT_FUSED_AVAILABLE", False):
        chunk_split = compute_chunk_size_numpy(
            n_samples,
            n_filtered,
            n_cvt=1,
            use_split=True,
            lmm_mode=2,
            mem_budget_bytes=budget,
        )

    assert chunk_fused >= 3 * chunk_split, (
        f"Fused chunk ({chunk_fused}) should be >= 3x split chunk ({chunk_split})"
    )
