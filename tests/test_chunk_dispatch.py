"""Dispatch selection and the kernels each path builds.

Covers which ``DispatchPath`` the runner picks for a given (n_cvt, lmm_mode)
pair and that the chosen kernel produces the same statistics as the NumPy
fallback. The split-Uab layout, the adaptive rotation/compute core split, and
the overlapped pipeline are all properties of a path rather than of a result,
so they live here rather than beside the GEMMA parity runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.lmm import compute_numpy
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from jamma.lmm.schema import LmmConfig
from jamma.lmm.stats import AssocResult  # noqa: F401
from jamma.validation import (
    ToleranceConfig,
    compare_assoc_results,
    load_gemma_assoc,
)
from tests.conftest import make_runner_synthetic_data
from tests.fixture_paths import SYNTHETIC


@pytest.mark.tier0
def test_runner_mode4_uses_fused_dispatch():
    """Mode 4 takes the fused path at n_cvt=1 and the fused general path at n_cvt>=2.

    This used to wrap _compose_mode4_from_split and assert it was never called.
    That helper has gone, and so has the standalone split dispatcher and its
    kernel-construction mode guard that replaced it as this test's second half:
    D2 gave the general workspace's one compute every lmm_mode, so there is no
    longer a split path for mode 4 to be refused by.
    """
    from jamma.lmm.dispatch import DispatchPath

    if compute_numpy._accel is None:
        pytest.skip("Fused mode-4 C extension not available")

    for n_cvt, expected in ((1, DispatchPath.FUSED), (2, DispatchPath.FUSED_GENERAL)):
        assert (
            compute_numpy.select_current_dispatch_path(n_cvt, 4, log_choices=False)
            is expected
        )


# ---------------------------------------------------------------------------
# Split-Uab all modes and reconstruct_uab_from_soa tests (RUN-01)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.parametrize("lmm_mode", [1, 2, 3, 4], ids=["Wald", "LRT", "Score", "All"])
def test_split_uab_all_modes(lmm_mode):
    """All LMM modes produce valid results with split-Uab layout (RUN-01)."""
    rng = np.random.default_rng(42)
    n_samples, n_snps = 100, 50

    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    phenotypes = rng.standard_normal(n_samples)
    kinship = np.corrcoef(genotypes) + np.eye(n_samples) * 0.1
    kinship = (kinship + kinship.T) / 2  # ensure symmetry
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
        for i in range(n_snps)
    ]

    run_result = run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=lmm_mode,
        ),
    )
    results = run_result.associations

    assert len(results) > 0, f"Mode {lmm_mode} produced no results"

    # Mode-specific output checks
    if lmm_mode in (1, 4):  # Wald or All
        for r in results[:5]:
            assert hasattr(r, "beta"), f"Wald result missing beta: {r}"
            assert np.isfinite(r.beta), f"Wald beta not finite: {r}"
            assert hasattr(r, "p_wald"), f"Wald result missing p_wald: {r}"
            assert np.isfinite(r.p_wald), f"Wald p not finite: {r}"
    if lmm_mode in (2, 4):  # LRT or All
        for r in results[:5]:
            assert hasattr(r, "p_lrt"), f"LRT result missing p_lrt: {r}"
            assert np.isfinite(r.p_lrt), f"LRT p not finite: {r}"
    if lmm_mode in (3, 4):  # Score or All
        for r in results[:5]:
            assert hasattr(r, "p_score"), f"Score result missing p_score: {r}"
            assert np.isfinite(r.p_score), f"Score p not finite: {r}"


@pytest.mark.tier1
def test_reconstruct_uab_from_soa_matches_direct():
    """reconstruct_uab_from_soa matches batch_compute_uab_numpy exactly (RUN-01)."""
    from jamma.lmm.uab import (
        batch_compute_uab_numpy,
        batch_compute_uab_varying_soa_numpy,
        compute_uab_invariant_soa,
        reconstruct_uab_from_soa,
    )

    rng = np.random.default_rng(42)
    n_samples, n_snps = 50, 20

    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Direct full Uab construction
    Uab_direct = batch_compute_uab_numpy(n_cvt=1, UtW=UtW, Uty=Uty, utg_t=UtG.T)

    # Split construction + reconstruction
    invariant = compute_uab_invariant_soa(UtW, Uty, 1)
    varying = batch_compute_uab_varying_soa_numpy(
        n_cvt=1, UtW=UtW, Uty=Uty, utg_t=UtG.T
    )
    Uab_reconstructed = reconstruct_uab_from_soa(invariant, varying, 1)

    np.testing.assert_allclose(
        Uab_reconstructed,
        Uab_direct,
        atol=1e-14,
        err_msg="Reconstructed Uab does not match direct construction",
    )


# ---------------------------------------------------------------------------
# Adaptive core split tests (RUN-05)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_adaptive_core_split():
    """Core split adapts rotation/compute ratio based on n_samples (RUN-05)."""
    from jamma.lmm.chunk_pipeline import compute_pipeline_core_split

    total_cores = 8

    # Large samples (>10k): rotation-heavy, gets 50% of cores
    rot, omp = compute_pipeline_core_split(50_000, total_cores)
    assert rot == 4, f"Large: rot={rot}, omp={omp}"
    assert omp == 4, f"Large: rot={rot}, omp={omp}"

    # Medium samples (1k-10k): balanced, rotation gets 33%
    rot, omp = compute_pipeline_core_split(5_000, total_cores)
    assert rot == 2, f"Medium: rot={rot}, omp={omp}"
    assert omp == 6, f"Medium: rot={rot}, omp={omp}"

    # Small samples (<1k): compute-heavy, rotation gets 25%
    rot, omp = compute_pipeline_core_split(500, total_cores)
    assert rot == 2, f"Small: rot={rot}, omp={omp}"
    assert omp == 6, f"Small: rot={rot}, omp={omp}"

    # Edge: 1 core — both get 1
    rot, omp = compute_pipeline_core_split(50_000, 1)
    assert rot >= 1, f"Single core: rot={rot}, omp={omp}"
    assert omp >= 1, f"Single core: rot={rot}, omp={omp}"


@pytest.mark.tier1
def test_compute_adaptive_core_split():
    """compute_adaptive_core_split allocates threads proportional to measured times."""
    from jamma.lmm.chunk_pipeline import compute_adaptive_core_split

    # Rotation-heavy: 80% rotation time -> ~80% of cores for rotation
    rot, omp = compute_adaptive_core_split(
        rot_time=0.8, compute_time=0.2, total_cores=8
    )
    assert rot == 6, f"Rotation-heavy: rot={rot}, omp={omp}"
    assert omp == 2, f"Rotation-heavy: rot={rot}, omp={omp}"

    # Compute-heavy: 20% rotation time -> ~20% of cores for rotation
    rot, omp = compute_adaptive_core_split(
        rot_time=0.2, compute_time=0.8, total_cores=8
    )
    assert rot == 2, f"Compute-heavy: rot={rot}, omp={omp}"
    assert omp == 6, f"Compute-heavy: rot={rot}, omp={omp}"

    # Balanced: equal times -> 50/50 split
    rot, omp = compute_adaptive_core_split(
        rot_time=0.5, compute_time=0.5, total_cores=8
    )
    assert rot == 4, f"Balanced: rot={rot}, omp={omp}"
    assert omp == 4, f"Balanced: rot={rot}, omp={omp}"

    # Degenerate: both times near zero -> static fallback
    rot, omp = compute_adaptive_core_split(
        rot_time=0.0, compute_time=0.0, total_cores=8, n_samples=50_000
    )
    # Static fallback for 50k samples: 50% -> (4, 4)
    assert rot == 4, f"Degenerate fallback: rot={rot}, omp={omp}"
    assert omp == 4, f"Degenerate fallback: rot={rot}, omp={omp}"

    # Always returns (rot >= 1, compute >= 1)
    for r, c, n in [(0.9, 0.1, 2), (0.1, 0.9, 2), (0.5, 0.5, 2)]:
        rot, omp = compute_adaptive_core_split(
            rot_time=r, compute_time=c, total_cores=n
        )
        assert rot >= 1, f"Min 1: rot={rot}, omp={omp} (r={r}, c={c}, n={n})"
        assert omp >= 1, f"Min 1: rot={rot}, omp={omp} (r={r}, c={c}, n={n})"

    # Clamped: 2 cores, rotation-heavy -> (1, 1) since both must be >= 1
    rot, omp = compute_adaptive_core_split(
        rot_time=0.9, compute_time=0.1, total_cores=2
    )
    assert rot == 1, f"Clamped 2-core: rot={rot}, omp={omp}"
    assert omp == 1, f"Clamped 2-core: rot={rot}, omp={omp}"


# ---------------------------------------------------------------------------
# End-to-end runner tests for LRT/Score/All modes with C extension (RUN-07)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_runner_lrt_mode_c_vs_python():
    """LRT mode (2) via C extension matches Python fallback (RUN-07)."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy

    genotypes, phenotypes, kinship, snp_info = make_runner_synthetic_data()

    kwargs = {
        "genotypes": genotypes,
        "phenotypes": phenotypes,
        "kinship": kinship.copy(),
        "snp_info": snp_info,
        "config": LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=2,
        ),
    }

    # Run with C extension
    result_c = run_lmm_association_numpy(**kwargs)

    # Run with C disabled (drop the loaded extension)
    with patch.object(compute_numpy, "_accel", None):
        kwargs["kinship"] = kinship.copy()
        result_py = run_lmm_association_numpy(**kwargs)

    assert len(result_c.associations) == len(result_py.associations)
    for rc, rp in zip(result_c.associations, result_py.associations, strict=True):
        assert np.isfinite(rc.p_lrt), f"C path p_lrt not finite: {rc}"
        np.testing.assert_allclose(
            rc.p_lrt,
            rp.p_lrt,
            rtol=1e-8,
            err_msg=f"LRT p-value mismatch for {rc.rs}",
        )


@pytest.mark.tier1
def test_runner_score_mode_c_vs_python():
    """Score mode (3) via C extension matches Python fallback (RUN-07)."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy

    genotypes, phenotypes, kinship, snp_info = make_runner_synthetic_data()

    kwargs = {
        "genotypes": genotypes,
        "phenotypes": phenotypes,
        "kinship": kinship.copy(),
        "snp_info": snp_info,
        "config": LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=3,
        ),
    }

    # Run with C extension
    result_c = run_lmm_association_numpy(**kwargs)

    # Run with C disabled
    with patch.object(compute_numpy, "_accel", None):
        kwargs["kinship"] = kinship.copy()
        result_py = run_lmm_association_numpy(**kwargs)

    assert len(result_c.associations) == len(result_py.associations)
    for rc, rp in zip(result_c.associations, result_py.associations, strict=True):
        assert np.isfinite(rc.p_score), f"C path p_score not finite: {rc}"
        np.testing.assert_allclose(
            rc.p_score,
            rp.p_score,
            rtol=1e-8,
            err_msg=f"Score p-value mismatch for {rc.rs}",
        )


@pytest.mark.tier1
def test_runner_all_mode_c_path():
    """All mode (4) produces all 8 result fields via C extension (RUN-07)."""
    genotypes, phenotypes, kinship, snp_info = make_runner_synthetic_data()

    result = run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=4,
        ),
    )

    assert len(result.associations) > 0
    for r in result.associations[:10]:
        # Wald fields
        assert np.isfinite(r.beta), f"beta not finite: {r}"
        assert np.isfinite(r.se), f"se not finite: {r}"
        assert np.isfinite(r.p_wald), f"p_wald not finite: {r}"
        assert np.isfinite(r.logl_H1), f"logl_H1 not finite: {r}"
        assert np.isfinite(r.l_remle), f"l_remle not finite: {r}"
        # LRT fields
        assert np.isfinite(r.p_lrt), f"p_lrt not finite: {r}"
        assert np.isfinite(r.l_mle), f"l_mle not finite: {r}"
        # Score field
        assert np.isfinite(r.p_score), f"p_score not finite: {r}"


@pytest.mark.tier1
def test_runner_pipeline_enabled_for_non_wald_modes(monkeypatch):
    """LRT and Score really do take the overlapped driver once chunks are enough.

    This used to patch runner_numpy._MIN_PIPELINE_CHUNKS down to 3 and then
    assert only that results came out. The chunk size was auto-sized on a
    200-SNP dataset, so the run was single-chunk and the pipeline never
    engaged; the assertion held either way. Forcing a small chunk gets past the
    real threshold, and the spy is what makes the claim in the name checkable.
    """
    from jamma.lmm import chunk_runner_numpy

    # The overlapped pipeline only engages on a split dispatch path, and
    # DispatchPath.use_split is False for the NumPy fallback (see dispatch.py).
    # With no C accelerator, every mode takes the full-Uab path and
    # _drive_pipeline is never called, so the spy count would be 0. Skip rather
    # than assert a C-accel-only behaviour. This is the case the ASAN workflow
    # hits: it sets JAMMA_FORCE_NUMPY_FALLBACK to keep dlopen away from ASAN.
    if compute_numpy._accel is None:
        pytest.skip("overlapped pipeline needs the C accelerator (split dispatch)")

    genotypes, phenotypes, kinship, snp_info = make_runner_synthetic_data(
        n_samples=50, n_snps=200
    )
    monkeypatch.setattr(
        "jamma.lmm.chunk_sizing.compute_chunk_size_numpy",
        lambda *args, **kwargs: 20,
    )

    drove = []
    real_driver = chunk_runner_numpy._drive_pipeline

    def spy(engine, **kwargs):
        drove.append(kwargs["n_chunks"])
        return real_driver(engine, **kwargs)

    monkeypatch.setattr(chunk_runner_numpy, "_drive_pipeline", spy)

    for mode in (2, 3):
        result = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=kinship.copy(),
            snp_info=snp_info,
            config=LmmConfig(
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                lmm_mode=mode,
            ),
        )
        assert len(result.associations) > 0, f"Mode {mode} should produce results"

    assert len(drove) == 2, f"pipeline ran {len(drove)} times, expected once per mode"
    assert all(n >= 8 for n in drove), drove


# ---------------------------------------------------------------------------
# n_cvt>1 C general path integration tests (Plan 70-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_runner_numpy_ncvt2_mode2_c_dispatch(synthetic_data_with_covariates):
    """LRT (mode 2) with n_cvt=2 uses C general path and matches GEMMA reference.

    Verifies the full path: FUSED_GENERAL dispatch -> a general workspace
    created with lmm_mode=2 -> compute_lmm_chunk_fused_general_c.
    """
    from jamma.lmm import compute_numpy as cn

    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates

    if cn._accel is None:
        pytest.skip("C extension not available")

    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        config=LmmConfig(lmm_mode=2, show_progress=False),
    )
    results = run_result.associations
    assert len(results) > 0, "Expected at least one LRT result with n_cvt=2"

    # Verify LRT fields are populated
    for r in results:
        assert r.p_lrt is not None, "p_lrt should be populated for mode 2"
        assert np.isfinite(r.p_lrt) or np.isnan(r.p_lrt), (
            f"p_lrt should be finite or NaN, got {r.p_lrt}"
        )

    # Compare against GEMMA reference
    reference = load_gemma_assoc(SYNTHETIC.ref("covar_lrt"))
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy mode 2+covar (n_cvt=2) vs GEMMA failed:\n{comparison}"
    )


@pytest.mark.tier0
def test_runner_numpy_ncvt2_mode3_c_dispatch(synthetic_data_with_covariates):
    """Score (mode 3) with n_cvt=2 uses C general path and matches GEMMA reference.

    Verifies the full path: FUSED_GENERAL dispatch -> a general workspace
    created with lmm_mode=3 -> compute_lmm_chunk_fused_general_c.
    """
    from jamma.lmm import compute_numpy as cn

    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates

    if cn._accel is None:
        pytest.skip("C extension not available")

    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        config=LmmConfig(lmm_mode=3, show_progress=False),
    )
    results = run_result.associations
    assert len(results) > 0, "Expected at least one Score result with n_cvt=2"

    # Verify Score fields are populated
    for r in results:
        assert r.p_score is not None, "p_score should be populated for mode 3"
        assert np.isfinite(r.p_score) or np.isnan(r.p_score), (
            f"p_score should be finite or NaN, got {r.p_score}"
        )

    # Compare against GEMMA reference
    reference = load_gemma_assoc(SYNTHETIC.ref("covar_score"))
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy mode 3+covar (n_cvt=2) vs GEMMA failed:\n{comparison}"
    )
