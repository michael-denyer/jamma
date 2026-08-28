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
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from jamma.lmm.schema import LmmConfig
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


@pytest.mark.tier0
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


@pytest.mark.tier0
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
# Covariate tests
# ---------------------------------------------------------------------------

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
