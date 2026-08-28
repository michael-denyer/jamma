"""Validation tests for the NumPy LMM runner against GEMMA reference output.

Validates that run_lmm_association_numpy produces GEMMA-equivalent p-values for all
four LMM modes (Wald, LRT, Score, All). Tests compare directly against GEMMA reference
files (see GEMMA_EQUIVALENCE.md for tolerance rationale).
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.io import load_plink_binary, read_fam_phenotypes
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm import compute_numpy
from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from jamma.lmm.schema import LmmConfig, LmmMode
from jamma.lmm.stats import AssocResult
from jamma.validation import (
    ToleranceConfig,
    compare_assoc_results,
    load_gemma_assoc,
)
from tests.conftest import require_fixture
from tests.fixture_paths import (
    MOUSE,
    NUMPY_GEMMA_TOLERANCES,
    SYNTHETIC,
    build_snp_info,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Load gemma_synthetic PLINK data, kinship, phenotypes, and snp_info."""
    plink = load_plink_binary(SYNTHETIC.bfile)
    kinship = read_kinship_matrix(SYNTHETIC.kinship)
    phenotypes = read_fam_phenotypes(SYNTHETIC.fam)
    snp_info = build_snp_info(plink)
    return plink, kinship, phenotypes, snp_info


@pytest.fixture
def mouse_hs1940_data():
    """Load mouse_hs1940 PLINK data, kinship, phenotypes, and snp_info."""
    plink = load_plink_binary(MOUSE.bfile)
    kinship = read_kinship_matrix(MOUSE.kinship)
    phenotypes = read_fam_phenotypes(MOUSE.fam)
    snp_info = build_snp_info(plink)
    return plink, kinship, phenotypes, snp_info


@pytest.fixture
def mouse_hs1940_data_with_covariates(mouse_hs1940_data):
    """Load mouse_hs1940 data plus covariates with intercept column prepended.

    The covariates.txt file contains only user-provided covariates (no intercept).
    GEMMA adds the intercept internally when -c is used, so we prepend a column
    of 1s to match GEMMA's internal representation.
    """
    plink, kinship, phenotypes, snp_info = mouse_hs1940_data
    raw_covariates = np.loadtxt(MOUSE.covariates)
    n_samples = raw_covariates.shape[0]
    covariates = np.hstack([np.ones((n_samples, 1)), raw_covariates])
    return plink, kinship, phenotypes, snp_info, covariates


# ---------------------------------------------------------------------------
# Fast unit and structural tests (always run)
# ---------------------------------------------------------------------------


def test_numpy_runner_returns_list_of_assoc_result(synthetic_data):
    """Type check: NumPy runner returns LmmRunResult with AssocResult items."""
    from jamma.lmm.schema import LmmConfig, LmmRunResult

    plink, kinship, phenotypes, snp_info = synthetic_data
    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(lmm_mode=1, show_progress=False),
    )
    results = run_result.associations
    assert isinstance(run_result, LmmRunResult), (
        f"Expected LmmRunResult, got {type(run_result)}"
    )
    assert len(results) > 0, "Expected at least one result"
    assert isinstance(results[0], AssocResult), (
        f"Expected AssocResult, got {type(results[0])}"
    )
    assert run_result.pve is not None, "PVE should be populated"
    assert 0 <= run_result.pve <= 1, f"PVE should be in [0, 1], got {run_result.pve}"
    assert run_result.pve_se is not None, (
        "PVE SE should be populated for synthetic data"
    )
    assert run_result.pve_se > 0, f"PVE SE should be positive, got {run_result.pve_se}"


def test_numpy_runner_empty_after_filter(synthetic_data):
    """Edge case: returns LmmRunResult with empty associations."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    # Constant genotypes are non-polymorphic, so the variance filter drops every
    # SNP whatever the thresholds are. A MAF threshold cannot express this: MAF
    # is min(af, 1-af) and so never exceeds 0.5.
    constant_genotypes = np.full_like(plink.genotypes, 2.0)
    run_result = run_lmm_association_numpy(
        genotypes=constant_genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(lmm_mode=1, show_progress=False),
    )
    results = run_result.associations
    assert len(results) == 0, f"Expected empty results, got {len(results)} results"
    assert run_result.pve is None, "PVE should be None when no SNPs pass filter"


# ---------------------------------------------------------------------------
# Chunk size computation
# ---------------------------------------------------------------------------


def test_compute_chunk_size_small_dataset():
    """Small dataset: chunk size = n_filtered (everything in one chunk)."""
    chunk = compute_chunk_size_numpy(
        n_samples=100,
        n_filtered=500,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    assert chunk == 500, f"Expected 500, got {chunk}"


def test_compute_chunk_size_large_dataset():
    """Large dataset: chunk capped by memory budget or _MAX_CHUNK."""
    chunk = compute_chunk_size_numpy(
        n_samples=10_000,
        n_filtered=200_000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    assert 100 <= chunk <= 200_000, f"Chunk {chunk} outside expected bounds"


def test_compute_chunk_size_zero_bytes():
    """bytes_per_snp=0 (n_samples=0): returns n_filtered directly."""
    chunk = compute_chunk_size_numpy(
        n_samples=0, n_filtered=1000, n_cvt=1, dispatch=DispatchPath.NUMPY_FALLBACK
    )
    assert chunk == 1000, f"Expected 1000, got {chunk}"


def test_compute_chunk_size_minimum():
    """Chunk size never drops below 100."""
    # Huge n_samples to force small chunk_from_memory, tiny n_filtered to avoid cap
    chunk = compute_chunk_size_numpy(
        n_samples=1_000_000,
        n_filtered=200,
        n_cvt=10,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    assert chunk >= 100, f"Chunk {chunk} below minimum 100"


def test_chunk_size_split_larger_than_full():
    """Split Uab accounting produces larger chunks than full Uab."""
    full = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(10e9),
    )
    split = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=int(10e9),
    )
    assert split > full, f"Split chunk ({split}) should exceed full ({full})"


def test_chunk_size_explicit_budget():
    """Explicit mem_budget_bytes overrides auto-scaling."""
    small_budget = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    large_budget = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(20e9),
    )
    assert large_budget > small_budget


def test_chunk_size_pipeline_halves_budget():
    """pipeline_buffers=2 produces roughly half the chunk size."""
    single = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=int(20e9),
    )
    double = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=int(20e9),
        pipeline_buffers=2,
    )
    # Double-buffering halves the budget, so chunk should be ~half
    assert double < single
    assert double >= single // 2 - 1  # allow rounding


def test_chunk_size_auto_scales_with_memory():
    """Auto-scaled budget uses 15% of available RAM between 2-40 GB bounds."""
    from unittest.mock import patch

    # 400 GB available → 15% = 60 GB (hits 40 GB ceiling)
    with patch("jamma.core.memory.available_ram_gb", return_value=400.0):
        chunk_big = compute_chunk_size_numpy(
            n_samples=50_000,
            n_filtered=100_000,
            n_cvt=1,
            dispatch=DispatchPath.FUSED,
        )

    # 10 GB available → 15% = 1.5 GB (hits 2 GB floor)
    with patch("jamma.core.memory.available_ram_gb", return_value=10.0):
        chunk_small = compute_chunk_size_numpy(
            n_samples=50_000,
            n_filtered=100_000,
            n_cvt=1,
            dispatch=DispatchPath.FUSED,
        )

    assert chunk_big > chunk_small


def test_chunk_size_accounting_by_dispatch_path():
    """Each path's column count, named by path rather than by mode.

    Every n_cvt=1 C path is in the fused family and hands ``utg_t`` straight to
    its kernel, so all three size identically at one column per SNP. The
    SoA-split accounting adds the three varying Uab columns beside it, and the
    NumPy fallback materialises the whole six-column table.

    This replaced a test that called the sizer three times with identical
    arguments and asserted the three results matched. It could not fail, and
    its "4-col" claim had been wrong since the C-availability flags collapsed
    to one bit: every n_cvt=1 C path had already moved to one column.
    """
    n_samples = 10_000
    budget = int(5e9)

    def size(dispatch):
        return compute_chunk_size_numpy(
            n_samples=n_samples,
            n_filtered=500_000,
            n_cvt=1,
            dispatch=dispatch,
            mem_budget_bytes=budget,
        )

    fused = [
        size(DispatchPath.FUSED),
        size(DispatchPath.FUSED_SCORE_WS),
        size(DispatchPath.FUSED_LRT_WS),
    ]
    assert len(set(fused)) == 1, f"fused family must size alike, got {fused}"

    # 1 column vs 4 (3 varying + utg_t) vs 6 ((n_cvt+3)(n_cvt+2)/2 at n_cvt=1).
    # Floor division on both sides: the sizer truncates budget/bytes_per_snp.
    assert size(DispatchPath.SOA_SPLIT) == fused[0] // 4
    assert size(DispatchPath.NUMPY_FALLBACK) == fused[0] // 6


def test_runner_mode4_uses_fused_dispatch():
    """Mode 4 takes a fused path, and the SoA-split kernel refuses to serve it.

    This used to wrap _compose_mode4_from_split and assert it was never called.
    That helper has gone, and so has the standalone split dispatcher that
    replaced it as this test's second half; the mode guard now lives in kernel
    construction, so a dispatch table that ever routed mode 4 to the split path
    fails before the chunk loop rather than on its first chunk.
    """
    from jamma.lmm.chunk_kernel import RunInvariants, make_kernel
    from jamma.lmm.dispatch import DispatchPath

    if compute_numpy._accel is None:
        pytest.skip("Fused mode-4 C extension not available")

    for n_cvt, expected in ((1, DispatchPath.FUSED), (2, DispatchPath.FUSED_GENERAL)):
        assert (
            compute_numpy.select_current_dispatch_path(n_cvt, 4, log_choices=False)
            is expected
        )

    n_samples = 8
    for refused_mode in (1, 4):
        forced_split = RunInvariants.build(
            dispatch=DispatchPath.SOA_SPLIT,
            lmm_mode=refused_mode,
            n_cvt=2,
            n_samples=n_samples,
            n_filtered=5,
            eigenvalues=np.linspace(0.1, 2.0, n_samples),
            UtW=np.ones((n_samples, 2)) * np.arange(1, 3),
            Uty=np.linspace(-1.0, 1.0, n_samples),
            Hi_eval_null=np.ones(n_samples),
            logl_H0=-1.0,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
        )
        with pytest.raises(ValueError, match="modes 1 and 4 take the fused"):
            make_kernel(forced_split, 1)


# ---------------------------------------------------------------------------
# GEMMA validation tests
# ---------------------------------------------------------------------------


_SYNTHETIC_MODE_REFS = [
    pytest.param(1, SYNTHETIC.ref("wald"), id="wald"),
    pytest.param(2, SYNTHETIC.ref("lrt"), id="lrt"),
    pytest.param(3, SYNTHETIC.ref("score"), id="score"),
    pytest.param(4, SYNTHETIC.ref("all"), id="all"),
]


@pytest.mark.tier1
@pytest.mark.parametrize("lmm_mode,reference_path", _SYNTHETIC_MODE_REFS)
def test_numpy_runner_synthetic(synthetic_data, lmm_mode, reference_path):
    """NumPy runner matches GEMMA reference on gemma_synthetic for each mode."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(lmm_mode=lmm_mode, show_progress=False),
    )
    results = run_result.associations
    reference = load_gemma_assoc(reference_path)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy mode {lmm_mode} (synthetic) vs GEMMA failed:\n{comparison}"
    )


_MOUSE_HS1940_MODE_REFS = [
    pytest.param(2, MOUSE.ref("lrt"), id="lrt"),
    pytest.param(3, MOUSE.ref("score"), id="score"),
    pytest.param(4, MOUSE.ref("all"), id="all"),
]


@pytest.mark.slow
@pytest.mark.tier2
@pytest.mark.parametrize("lmm_mode,reference_path", _MOUSE_HS1940_MODE_REFS)
def test_numpy_runner_mouse_hs1940(mouse_hs1940_data, lmm_mode, reference_path):
    """NumPy runner matches GEMMA on mouse_hs1940 for each mode."""
    plink, kinship, phenotypes, snp_info = mouse_hs1940_data
    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(lmm_mode=lmm_mode, show_progress=False),
    )
    results = run_result.associations
    reference = load_gemma_assoc(reference_path)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy mode {lmm_mode} (mouse_hs1940) vs GEMMA failed:\n{comparison}"
    )


# ---------------------------------------------------------------------------
# Covariate fixture and tests
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data_with_covariates(synthetic_data):
    """Load gemma_synthetic data plus covariates from gemma_covariate fixture.

    The covariates.txt file already includes the intercept column (first column
    is all 1.0), matching GEMMA's internal representation when -c is used.
    """
    plink, kinship, phenotypes, snp_info = synthetic_data
    covariates = np.loadtxt(SYNTHETIC.covariates)
    return plink, kinship, phenotypes, snp_info, covariates


_SYNTHETIC_COVAR_MODE_REFS = [
    pytest.param(1, SYNTHETIC.ref("covar_wald"), id="wald"),
    pytest.param(2, SYNTHETIC.ref("covar_lrt"), id="lrt"),
    pytest.param(3, SYNTHETIC.ref("covar_score"), id="score"),
    pytest.param(4, SYNTHETIC.ref("covar_all"), id="all"),
]


@pytest.mark.tier1
@pytest.mark.parametrize("lmm_mode,reference_path", _SYNTHETIC_COVAR_MODE_REFS)
def test_numpy_runner_covar_synthetic(
    synthetic_data_with_covariates, lmm_mode, reference_path
):
    """NumPy runner with covariates matches GEMMA reference on synthetic data."""
    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates
    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        config=LmmConfig(lmm_mode=lmm_mode, show_progress=False),
    )
    results = run_result.associations
    reference = load_gemma_assoc(reference_path)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy mode {lmm_mode}+covar (synthetic) vs GEMMA failed:\n{comparison}"
    )


# ---------------------------------------------------------------------------
# mouse_hs1940 covariate GEMMA validation tests (slow)
# ---------------------------------------------------------------------------

_MOUSE_HS1940_COVAR_MODE_REFS = [
    pytest.param(1, MOUSE.ref("covar_wald"), id="wald"),
    pytest.param(2, MOUSE.ref("covar_lrt"), id="lrt"),
    pytest.param(3, MOUSE.ref("covar_score"), id="score"),
    pytest.param(4, MOUSE.ref("covar_all"), id="all"),
]


@pytest.mark.slow
@pytest.mark.tier2
@pytest.mark.parametrize("lmm_mode,reference_path", _MOUSE_HS1940_COVAR_MODE_REFS)
def test_numpy_runner_covar_mouse_hs1940(
    mouse_hs1940_data_with_covariates, lmm_mode, reference_path
):
    """NumPy runner with covariates matches GEMMA on mouse_hs1940."""
    plink, kinship, phenotypes, snp_info, covariates = mouse_hs1940_data_with_covariates
    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        config=LmmConfig(lmm_mode=lmm_mode, show_progress=False),
    )
    results = run_result.associations
    reference = load_gemma_assoc(reference_path)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy mode {lmm_mode}+covar (mouse_hs1940) vs GEMMA failed:\n{comparison}"
    )


# ---------------------------------------------------------------------------
# Multi-chunk equivalence test
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_numpy_multi_chunk_pvalue_equivalence(monkeypatch):
    """p-values are identical regardless of chunk_size (single vs. multi-chunk).

    Forces multi-chunk mode by monkeypatching compute_chunk_size_numpy to
    return 50, then compares p_wald against a single-chunk run.  Pre-computed
    eigendecomp is passed to both calls to avoid repeated O(n^3) work and to
    ensure the only difference is the chunking path.
    """
    require_fixture(
        MOUSE.bed,
        MOUSE.fam,
        MOUSE.kinship,
    )

    plink = load_plink_binary(MOUSE.bfile)
    kinship = read_kinship_matrix(MOUSE.kinship)
    phenotypes = read_fam_phenotypes(MOUSE.fam)
    snp_info = build_snp_info(plink)

    # Filter to valid (non-NaN) samples then pre-compute eigendecomp once
    # on the filtered kinship — passed to both runs so the only variable is
    # the chunking path.
    valid_mask = ~np.isnan(phenotypes)
    kinship_filtered = kinship[np.ix_(valid_mask, valid_mask)]
    eigenvalues, eigenvectors = np.linalg.eigh(kinship_filtered)

    common_kwargs = {
        "genotypes": plink.genotypes,
        "phenotypes": phenotypes,
        "kinship": None,  # pre-computed eigen supplied; skip internal eigh
        "snp_info": snp_info,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "config": LmmConfig(lmm_mode=1, check_memory=False, show_progress=False),
        "output_path": None,
    }

    # Single-chunk run (no monkeypatching — default chunk_size fits all SNPs)
    result_single = run_lmm_association_numpy(**common_kwargs)

    # Multi-chunk run: force chunk_size=50 so the batch loop iterates many times.
    # The sizer is a RAM-budget policy, not a numerical routine, so pinning its
    # answer is the supported way to choose a chunking without a runner knob.
    monkeypatch.setattr(
        "jamma.lmm.chunk_sizing.compute_chunk_size_numpy",
        lambda *args, **kwargs: 50,
    )
    result_multi = run_lmm_association_numpy(**common_kwargs)

    p_single = np.array([r.p_wald for r in result_single.associations])
    p_multi = np.array([r.p_wald for r in result_multi.associations])

    np.testing.assert_allclose(
        p_single,
        p_multi,
        rtol=1e-10,
        err_msg="p_wald values differ between single-chunk and multi-chunk runs",
    )


# ---------------------------------------------------------------------------
# Lambda boundary diagnostic tests (REGR-03)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestLambdaBoundaryDiagnostics:
    """Unit tests for count_lambda_boundary_hits and log_lambda_boundary_warning.

    Verifies that flat-optima SNPs with lambda converging at l_min or l_max
    are correctly counted, and that the boundary warning logger path does not crash.
    """

    def test_mode1_lower_bound_count(self):
        """Mode 1 (Wald): count 3 lambdas at l_min, 0 at l_max."""
        from jamma.lmm.results import count_lambda_boundary_hits

        arrays = {"lambdas": np.array([1e-5, 1e-5, 0.5, 1e-5, 2.0])}
        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode=1, arrays=arrays, l_min=1e-5, l_max=1e5
        )
        assert n_at_lmin == 3
        assert n_at_lmax == 0

    def test_mode2_upper_bound_count(self):
        """Mode 2 (LRT): count 1 at l_min, 2 at l_max using lambdas_mle."""
        from jamma.lmm.results import count_lambda_boundary_hits

        arrays = {"lambdas_mle": np.array([1e-5, 1e5, 1e5])}
        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode=2, arrays=arrays, l_min=1e-5, l_max=1e5
        )
        assert n_at_lmin == 1
        assert n_at_lmax == 2

    def test_mode4_combines_reml_and_mle(self):
        """Mode 4 (All): counts from both lambdas (REML) and lambdas_mle (MLE)."""
        from jamma.lmm.results import count_lambda_boundary_hits

        arrays = {
            "lambdas": np.array([1e-5, 0.5]),
            "lambdas_mle": np.array([1e5, 0.5]),
        }
        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode=4, arrays=arrays, l_min=1e-5, l_max=1e5
        )
        assert n_at_lmin == 1  # one REML lambda at l_min
        assert n_at_lmax == 1  # one MLE lambda at l_max

    def test_empty_array_returns_zeros(self):
        """Empty lambda arrays return (0, 0) without error."""
        from jamma.lmm.results import count_lambda_boundary_hits

        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode=1, arrays={"lambdas": np.array([])}, l_min=1e-5, l_max=1e5
        )
        assert n_at_lmin == 0
        assert n_at_lmax == 0

    def test_warning_lower_bound_does_not_crash(self):
        """log_lambda_boundary_warning with lower-bound hits does not raise."""
        from jamma.lmm.results import log_lambda_boundary_warning

        log_lambda_boundary_warning(3, 0, 1e-5, 1e5)  # should not raise

    def test_warning_upper_bound_does_not_crash(self):
        """log_lambda_boundary_warning with upper-bound hits does not raise."""
        from jamma.lmm.results import log_lambda_boundary_warning

        log_lambda_boundary_warning(0, 2, 1e-5, 1e5)  # should not raise

    def test_warning_no_hits_is_noop(self):
        """log_lambda_boundary_warning with zero counts is a no-op."""
        from jamma.lmm.results import log_lambda_boundary_warning

        log_lambda_boundary_warning(0, 0, 1e-5, 1e5)  # should not raise


# ---------------------------------------------------------------------------
# Imputation guard tests (RUN-06)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_imputation_skipped_on_clean_data():
    """Imputation guard skips np.where when no NaN values present (RUN-06).

    Verifies that the imputation code path handles clean data correctly
    (the guard clause doesn't break the data flow).
    """
    rng = np.random.default_rng(42)
    n_samples, n_snps = 100, 50

    # Clean genotypes — no missing values
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    assert not np.any(np.isnan(genotypes)), "Test expects no missing values"

    phenotypes = rng.standard_normal(n_samples)
    kinship = np.eye(n_samples, dtype=np.float64)
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
            lmm_mode=1,
        ),
    )
    results = run_result.associations

    # Should complete without error and produce valid results
    assert len(results) > 0
    # Results should have finite values (no NaN from imputation issues)
    for r in results[:5]:  # spot check
        assert np.isfinite(r.beta), f"beta is not finite: {r.beta}"


@pytest.mark.tier1
def test_imputation_applies_on_missing_data():
    """Imputation guard correctly imputes when NaN values are present (RUN-06)."""
    rng = np.random.default_rng(42)
    n_samples, n_snps = 100, 50

    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    # Add some missing values
    genotypes[0, 0] = np.nan
    genotypes[5, 3] = np.nan
    genotypes[10, 10] = np.nan

    phenotypes = rng.standard_normal(n_samples)
    kinship = np.eye(n_samples, dtype=np.float64)
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
            lmm_mode=1,
        ),
    )
    results = run_result.associations

    assert len(results) > 0
    # With imputation, results should still be finite
    for r in results[:5]:
        assert np.isfinite(r.beta), f"beta is not finite: {r.beta}"


@pytest.mark.tier1
def test_inplace_imputation_does_not_corrupt_source():
    """In-place mean imputation on chunk must not mutate the source genotypes array.

    The batch runners slice genotypes with fancy indexing (integer array),
    which guarantees a copy.  This test documents that invariant so a future
    refactor to contiguous slicing doesn't silently corrupt subsequent chunks.
    """
    rng = np.random.default_rng(99)
    n_samples, n_snps = 80, 30

    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    # Sprinkle NaNs
    genotypes[0, 0] = np.nan
    genotypes[3, 5] = np.nan
    genotypes[7, 29] = np.nan

    original = genotypes.copy()

    phenotypes = rng.standard_normal(n_samples)
    kinship = np.eye(n_samples, dtype=np.float64)
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
        for i in range(n_snps)
    ]

    run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=1,
        ),
    )

    # Source array must be untouched — NaNs still present
    np.testing.assert_array_equal(
        genotypes,
        original,
        err_msg="In-place imputation corrupted the source genotypes array",
    )


@pytest.mark.tier1
def test_inplace_imputation_preserves_dtype():
    """In-place imputation must preserve the chunk's dtype.

    nanmean returns the same dtype as the input, so both chunk and means
    are float32 (or float64).  The in-place pattern must not change dtype.
    """
    for dtype in [np.float32, np.float64]:
        chunk = np.array(
            [[0.0, np.nan], [1.0, 2.0], [2.0, 1.0]],
            dtype=dtype,
        )
        chunk_means = np.nanmean(chunk, axis=0)
        assert chunk_means.dtype == dtype

        missing = np.isnan(chunk)
        chunk[missing] = np.take(chunk_means, np.where(missing)[1])

        assert chunk.dtype == dtype, (
            f"Expected {dtype} after imputation, got {chunk.dtype}"
        )
        np.testing.assert_allclose(chunk[0, 1], 1.5, rtol=1e-6)


@pytest.mark.tier1
def test_inplace_imputation_replaces_nan_with_column_mean():
    """Verify the np.take imputation pattern fills NaN positions with column means.

    Directly tests the pattern used by all LMM runners:
        chunk[missing] = np.take(chunk_means, np.where(missing)[1])
    """
    chunk = np.array(
        [
            [0.0, np.nan, 2.0],
            [1.0, 1.0, np.nan],
            [2.0, 2.0, 1.0],
            [np.nan, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    chunk_means = np.nanmean(chunk, axis=0)  # [1.0, 1.0, 1.0]

    missing = np.isnan(chunk)
    assert missing.any()
    chunk[missing] = np.take(chunk_means, np.where(missing)[1])

    # No NaNs remain
    assert not np.any(np.isnan(chunk)), "NaNs remain after imputation"

    # Each formerly-NaN position holds the column mean
    expected = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(chunk, expected)


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


def _make_synthetic_data(
    n_samples: int = 100, n_snps: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Create synthetic data for runner-level tests."""
    rng = np.random.default_rng(seed)
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    phenotypes = rng.standard_normal(n_samples)
    kinship = np.corrcoef(genotypes) + np.eye(n_samples) * 0.1
    kinship = (kinship + kinship.T) / 2
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
        for i in range(n_snps)
    ]
    return genotypes, phenotypes, kinship, snp_info


@pytest.mark.tier1
def test_runner_lrt_mode_c_vs_python():
    """LRT mode (2) via C extension matches Python fallback (RUN-07)."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy

    genotypes, phenotypes, kinship, snp_info = _make_synthetic_data()

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

    genotypes, phenotypes, kinship, snp_info = _make_synthetic_data()

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
    genotypes, phenotypes, kinship, snp_info = _make_synthetic_data()

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

    genotypes, phenotypes, kinship, snp_info = _make_synthetic_data(
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
# Output path streaming tests (65-03)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.parametrize("lmm_mode", [1, 2, 3, 4], ids=["wald", "lrt", "score", "all"])
def test_output_path_streaming_matches_inmemory(lmm_mode, tmp_path):
    """Streaming via output_path produces identical results to in-memory."""
    genotypes, phenotypes, kinship, snp_info = _make_synthetic_data()

    common_kwargs = {
        "genotypes": genotypes,
        "phenotypes": phenotypes,
        "snp_info": snp_info,
        "config": LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=lmm_mode,
        ),
    }

    # In-memory run
    result_mem = run_lmm_association_numpy(kinship=kinship.copy(), **common_kwargs)

    # Streaming run
    output_file = tmp_path / f"streamed_mode{lmm_mode}.assoc.txt"
    result_disk = run_lmm_association_numpy(
        kinship=kinship.copy(), output_path=output_file, **common_kwargs
    )

    # Streaming result has empty associations but populated metadata
    assert result_disk.associations == [], (
        "Streaming mode should return empty associations"
    )
    assert result_disk.n_tested == len(result_mem.associations), (
        f"n_tested mismatch: {result_disk.n_tested} vs {len(result_mem.associations)}"
    )

    # PVE and PVE SE should match
    assert result_disk.pve is not None
    assert result_mem.pve is not None
    np.testing.assert_allclose(
        result_disk.pve,
        result_mem.pve,
        rtol=1e-10,
        err_msg="PVE mismatch between streaming and in-memory",
    )
    if result_mem.pve_se is not None:
        assert result_disk.pve_se is not None, (
            "in-memory run reported pve_se but streaming run did not"
        )
        np.testing.assert_allclose(
            result_disk.pve_se,
            result_mem.pve_se,
            rtol=1e-10,
            err_msg="PVE SE mismatch between streaming and in-memory",
        )

    # Load streamed file and compare p-values
    assert output_file.exists(), f"Streamed output file not created: {output_file}"
    disk_results = load_gemma_assoc(output_file)
    assert len(disk_results) == len(result_mem.associations), (
        f"Streamed file has {len(disk_results)} SNPs, "
        f"expected {len(result_mem.associations)}"
    )

    # Compare SNP identifiers and p-values.
    # Text serialization loses ~7 digits of precision (%.6g format), so
    # use rtol=1e-6 for file-round-tripped values.
    file_rtol = 1e-6
    for r_mem, r_disk in zip(result_mem.associations, disk_results, strict=True):
        assert r_mem.rs == r_disk.rs, f"SNP order mismatch: {r_mem.rs} vs {r_disk.rs}"
        if lmm_mode in (1, 3, 4):
            np.testing.assert_allclose(
                r_disk.beta,
                r_mem.beta,
                rtol=file_rtol,
                err_msg=f"beta mismatch for {r_mem.rs}",
            )
        if lmm_mode in (1, 4):
            assert r_disk.p_wald is not None, f"p_wald absent on disk for {r_mem.rs}"
            assert r_mem.p_wald is not None, f"p_wald absent in memory for {r_mem.rs}"
            np.testing.assert_allclose(
                r_disk.p_wald,
                r_mem.p_wald,
                rtol=file_rtol,
                err_msg=f"p_wald mismatch for {r_mem.rs}",
            )
        if lmm_mode in (2, 4):
            assert r_disk.p_lrt is not None, f"p_lrt absent on disk for {r_mem.rs}"
            assert r_mem.p_lrt is not None, f"p_lrt absent in memory for {r_mem.rs}"
            np.testing.assert_allclose(
                r_disk.p_lrt,
                r_mem.p_lrt,
                rtol=file_rtol,
                err_msg=f"p_lrt mismatch for {r_mem.rs}",
            )
        if lmm_mode in (3, 4):
            assert r_disk.p_score is not None, f"p_score absent on disk for {r_mem.rs}"
            assert r_mem.p_score is not None, f"p_score absent in memory for {r_mem.rs}"
            np.testing.assert_allclose(
                r_disk.p_score,
                r_mem.p_score,
                rtol=file_rtol,
                err_msg=f"p_score mismatch for {r_mem.rs}",
            )


@pytest.mark.tier1
def test_output_path_streaming_all_filtered(tmp_path):
    """Streaming with all SNPs filtered returns empty result, no file created."""
    genotypes, phenotypes, kinship, snp_info = _make_synthetic_data()
    output_file = tmp_path / "filtered.assoc.txt"

    # Constant genotypes fail the polymorphism check, so nothing survives.
    result = run_lmm_association_numpy(
        genotypes=np.full_like(genotypes, 2.0),
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=1,
        ),
        output_path=output_file,
    )

    assert result.associations == []
    assert result.pve is None, "PVE should be None when no SNPs pass filter"


# ---------------------------------------------------------------------------
# Error message differentiation tests (68-02)
# ---------------------------------------------------------------------------


def _tiny_invariants(n_cvt: int, lmm_mode: LmmMode, n_samples: int = 8):
    """Smallest RunInvariants every dispatch path will build a kernel from."""
    from jamma.lmm.chunk_kernel import RunInvariants
    from jamma.lmm.dispatch import select_dispatch_path

    return RunInvariants.build(
        dispatch=select_dispatch_path(n_cvt, lmm_mode, accel=True, log_choices=False),
        lmm_mode=lmm_mode,
        n_cvt=n_cvt,
        n_samples=n_samples,
        n_filtered=500,
        eigenvalues=np.linspace(0.1, 2.0, n_samples),
        UtW=np.ones((n_samples, n_cvt)) * np.arange(1, n_cvt + 1),
        Uty=np.linspace(-1.0, 1.0, n_samples),
        Hi_eval_null=np.ones(n_samples),
        logl_H0=-10.0,
        l_min=1e-5,
        l_max=1e5,
        n_grid=20,
        n_refine=20,
    )


@pytest.mark.tier0
class TestErrorMessageDifferentiation:
    """A failing chunk must say which kernel failed, and where.

    These used to call the wrapper with an operation label the test invented,
    then assert the message contained it. All three labels they checked
    ("Wald C workspace compute" and friends) appear nowhere in src and never
    did, so the assertions only ever proved that an f-string interpolates.
    The labels below come from ``make_kernel``, so a renamed or duplicated one
    fails here.
    """

    def _failing_kernel(self, n_cvt: int, lmm_mode: LmmMode, exc: Exception):
        """A real kernel for this path, with its call swapped for a raise."""
        from jamma.lmm.chunk_kernel import Kernel, make_kernel

        built = make_kernel(_tiny_invariants(n_cvt, lmm_mode), 1)

        def _boom(_chunk, _threads):
            raise exc

        return Kernel(label=built.label, n_filtered=built.n_filtered, call=_boom)

    def test_every_path_has_its_own_label(self):
        """Seven labels over eight (n_cvt, mode) shapes, and none repeat a path.

        Eight shapes, seven labels: SoA-split serves modes 2 and 3 with one
        kernel, so those two share. Every other shape is distinguishable,
        including mode 4 against Wald within each fused family.
        """
        from jamma.lmm.chunk_kernel import make_kernel

        if compute_numpy._accel is None:
            pytest.skip("kernel construction needs the C extension")

        labels = {
            (n_cvt, mode): make_kernel(_tiny_invariants(n_cvt, mode), 1).label
            for n_cvt in (1, 2)
            for mode in (1, 2, 3, 4)
        }
        assert len(set(labels.values())) == 7, labels
        assert labels[1, 4] != labels[1, 1], "mode 4 must not report as Wald"
        assert labels[2, 4] != labels[2, 1], "mode 4 must not report as Wald"
        assert labels[2, 2] == labels[2, 3], "both are the one SoA-split kernel"

    @pytest.mark.parametrize(
        ("n_cvt", "lmm_mode"), [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2)]
    )
    def test_wrapped_error_names_the_kernel_and_the_offset(self, n_cvt, lmm_mode):
        """A segfault-shaped failure reports its own label, offset, and total."""
        if compute_numpy._accel is None:
            pytest.skip("kernel construction needs the C extension")

        kernel = self._failing_kernel(n_cvt, lmm_mode, OSError("segfault"))
        with pytest.raises(RuntimeError) as exc_info:
            kernel.compute_chunk(np.zeros((1, 8)), 1, 300)

        message = str(exc_info.value)
        assert kernel.label in message
        assert "300/500" in message
        assert "300 SNPs before failure" in message

    @pytest.mark.parametrize(
        "exc",
        [
            MemoryError("out of memory"),
            ValueError("bad value"),
            TypeError("wrong type"),
            OverflowError("overflow"),
        ],
    )
    def test_diagnosable_exceptions_pass_through_unwrapped(self, exc):
        """These four say what went wrong already; wrapping would bury them."""
        if compute_numpy._accel is None:
            pytest.skip("kernel construction needs the C extension")

        kernel = self._failing_kernel(1, 1, exc)
        with pytest.raises(type(exc), match=str(exc)):
            kernel.compute_chunk(np.zeros((1, 8)), 1, 0)

    def test_exception_chaining_preserved(self):
        """The original exception is chained via 'from exc'."""
        if compute_numpy._accel is None:
            pytest.skip("kernel construction needs the C extension")

        kernel = self._failing_kernel(1, 1, OSError("root cause"))
        with pytest.raises(RuntimeError) as exc_info:
            kernel.compute_chunk(np.zeros((1, 8)), 1, 0)

        assert isinstance(exc_info.value.__cause__, OSError)
        assert "root cause" in str(exc_info.value.__cause__)

    def test_successful_call_returns_result_unwrapped(self):
        """A kernel that succeeds hands its dict straight back."""
        from jamma.lmm.chunk_kernel import Kernel

        expected = {"betas": [1.0], "ses": [0.1]}
        kernel = Kernel(
            label="Fused Uab dispatch",
            n_filtered=100,
            call=lambda _chunk, _threads: expected,
        )
        assert kernel.compute_chunk(np.zeros((1, 8)), 1, 0) is expected


# ---------------------------------------------------------------------------
# n_cvt>1 C general path integration tests (Plan 70-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_runner_numpy_ncvt2_mode2_c_dispatch(synthetic_data_with_covariates):
    """LRT (mode 2) with n_cvt=2 uses C general path and matches GEMMA reference.

    Verifies the full path: use_split=True -> SoA split ->
    _compute_lrt_split_numpy -> compute_lrt_split_general_c.
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

    Verifies the full path: use_split=True -> SoA split ->
    _compute_score_split_numpy -> compute_score_split_general_c.
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
