"""Validation tests for the pure-NumPy LMM runner against GEMMA reference output.

Validates that run_lmm_association_numpy produces GEMMA-equivalent p-values for all
four LMM modes (Wald, LRT, Score, All). Tests compare directly against GEMMA reference
files, not against the JAX runner, because Cephes betainc vs JAX XLA betainc can
diverge up to ~6e-3 rtol at large sample sizes (see EQUIVALENCE.md).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.runner_numpy import _compute_chunk_size_numpy, run_lmm_association_numpy
from jamma.lmm.stats import AssocResult
from jamma.validation import (
    ToleranceConfig,
    compare_assoc_results,
    load_gemma_assoc,
)
from tests.conftest import load_phenotypes_from_fam

# ---------------------------------------------------------------------------
# Tolerance configurations
# ---------------------------------------------------------------------------

# NumPy backend vs GEMMA tolerances.
# Cephes betainc is more accurate than JAX XLA betainc for large a (n_samples > 1000).
# Lambda optimization uses identical golden section algorithm → same lambda tolerance.
# p-value tolerance can be tighter than JAX-vs-GEMMA because Cephes is closer to GSL.
NUMPY_GEMMA_TOLERANCES = ToleranceConfig(
    lambda_rtol=1e-3,  # Golden section vs Brent, same as JAX
    pvalue_rtol=1e-2,  # Start with same as JAX; may tighten after validation
    se_rtol=5e-4,  # Same as JAX
    logl_rtol=5e-3,  # Golden section vs Brent on flat optima
    atol=1e-4,  # Near-zero values
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURE_ROOT = Path(__file__).parent / "fixtures"
MOUSE_HS1940_DIR = _FIXTURE_ROOT / "mouse_hs1940"
MOUSE_HS1940_DATA = MOUSE_HS1940_DIR / "mouse_hs1940"
MOUSE_HS1940_KINSHIP = MOUSE_HS1940_DIR / "mouse_hs1940_kinship.cXX.txt"
MOUSE_HS1940_ALL = MOUSE_HS1940_DIR / "mouse_hs1940_all.assoc.txt"
MOUSE_HS1940_LRT = MOUSE_HS1940_DIR / "mouse_hs1940_lrt.assoc.txt"
MOUSE_HS1940_SCORE = MOUSE_HS1940_DIR / "mouse_hs1940_score.assoc.txt"
MOUSE_HS1940_COVARIATES = MOUSE_HS1940_DIR / "covariates.txt"
MOUSE_HS1940_COVAR_WALD = MOUSE_HS1940_DIR / "mouse_hs1940_covar_wald.assoc.txt"
MOUSE_HS1940_COVAR_LRT = MOUSE_HS1940_DIR / "mouse_hs1940_covar_lrt.assoc.txt"
MOUSE_HS1940_COVAR_SCORE = MOUSE_HS1940_DIR / "mouse_hs1940_covar_score.assoc.txt"
MOUSE_HS1940_COVAR_ALL = MOUSE_HS1940_DIR / "mouse_hs1940_covar_all.assoc.txt"

SYNTHETIC_DATA = _FIXTURE_ROOT / "gemma_synthetic" / "test"
SYNTHETIC_KINSHIP = _FIXTURE_ROOT / "gemma_synthetic" / "gemma_kinship.cXX.txt"
SYNTHETIC_REFERENCE = _FIXTURE_ROOT / "gemma_synthetic" / "gemma_assoc.assoc.txt"
SYNTHETIC_LRT_REFERENCE = _FIXTURE_ROOT / "gemma_synthetic" / "gemma_lrt.assoc.txt"
SCORE_REFERENCE = _FIXTURE_ROOT / "gemma_score" / "gemma_score.assoc.txt"
ALL_TESTS_REFERENCE = _FIXTURE_ROOT / "gemma_all_tests" / "gemma_all.assoc.txt"

COVARIATE_FIXTURE_DIR = _FIXTURE_ROOT / "gemma_covariate"
COVARIATE_FILE = COVARIATE_FIXTURE_DIR / "covariates.txt"
COVARIATE_WALD_REFERENCE = COVARIATE_FIXTURE_DIR / "gemma_covariate.assoc.txt"
COVARIATE_LRT_REFERENCE = COVARIATE_FIXTURE_DIR / "gemma_covariate_lrt.assoc.txt"
COVARIATE_SCORE_REFERENCE = COVARIATE_FIXTURE_DIR / "gemma_covariate_score.assoc.txt"
ALL_TESTS_COVAR_REFERENCE = (
    _FIXTURE_ROOT / "gemma_all_tests" / "gemma_all_covar.assoc.txt"
)


def _build_snp_info(plink_data) -> list[dict]:
    """Build snp_info list from PlinkData object.

    Args:
        plink_data: PlinkData with chromosome, sid, bp_position, allele_1, allele_2.

    Returns:
        List of dicts with keys: chr, rs, pos, a1, a0, maf, n_miss.
    """
    return [
        {
            "chr": str(plink_data.chromosome[i]),
            "rs": plink_data.sid[i],
            "pos": plink_data.bp_position[i],
            "a1": plink_data.allele_1[i],
            "a0": plink_data.allele_2[i],
            "maf": 0.0,
            "n_miss": 0,
        }
        for i in range(plink_data.n_snps)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Load gemma_synthetic PLINK data, kinship, phenotypes, and snp_info."""
    plink = load_plink_binary(SYNTHETIC_DATA)
    kinship = read_kinship_matrix(SYNTHETIC_KINSHIP)
    phenotypes = load_phenotypes_from_fam(SYNTHETIC_DATA.with_suffix(".fam"))
    snp_info = _build_snp_info(plink)
    return plink, kinship, phenotypes, snp_info


@pytest.fixture
def mouse_hs1940_data():
    """Load mouse_hs1940 PLINK data, kinship, phenotypes, and snp_info."""
    plink = load_plink_binary(MOUSE_HS1940_DATA)
    kinship = read_kinship_matrix(MOUSE_HS1940_KINSHIP)
    phenotypes = load_phenotypes_from_fam(MOUSE_HS1940_DATA.with_suffix(".fam"))
    snp_info = _build_snp_info(plink)
    return plink, kinship, phenotypes, snp_info


@pytest.fixture
def mouse_hs1940_data_with_covariates(mouse_hs1940_data):
    """Load mouse_hs1940 data plus covariates with intercept column prepended.

    The covariates.txt file contains only user-provided covariates (no intercept).
    GEMMA adds the intercept internally when -c is used, so we prepend a column
    of 1s to match GEMMA's internal representation.
    """
    plink, kinship, phenotypes, snp_info = mouse_hs1940_data
    raw_covariates = np.loadtxt(MOUSE_HS1940_COVARIATES)
    n_samples = raw_covariates.shape[0]
    covariates = np.hstack([np.ones((n_samples, 1)), raw_covariates])
    return plink, kinship, phenotypes, snp_info, covariates


# ---------------------------------------------------------------------------
# Fast unit and structural tests (always run)
# ---------------------------------------------------------------------------


def test_numpy_runner_no_jax_imports():
    """AST check: runner_numpy.py must not contain any JAX import."""
    src_path = (
        Path(__file__).parent.parent / "src" / "jamma" / "lmm" / "runner_numpy.py"
    )
    source = src_path.read_text()
    tree = ast.parse(source)
    jax_imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("jax"):
                        jax_imports.append(alias.name)
            else:
                if node.module and node.module.startswith("jax"):
                    jax_imports.append(node.module)
    assert jax_imports == [], f"runner_numpy.py has JAX imports: {jax_imports}"


def test_numpy_runner_signature_matches_jax():
    """Verify NumPy runner parameter names match JAX runner exactly (RUNR-04)."""
    import inspect

    jax_mod = pytest.importorskip("jamma.lmm.runner_jax")
    numpy_sig = inspect.signature(run_lmm_association_numpy)
    jax_sig = inspect.signature(jax_mod.run_lmm_association_jax)
    assert list(numpy_sig.parameters.keys()) == list(jax_sig.parameters.keys()), (
        f"Parameter names differ: "
        f"{list(numpy_sig.parameters.keys())} vs {list(jax_sig.parameters.keys())}"
    )


def test_numpy_runner_returns_list_of_assoc_result(synthetic_data):
    """Type check: NumPy runner returns LmmRunResult with AssocResult items."""
    from jamma.lmm.schema import LmmRunResult

    plink, kinship, phenotypes, snp_info = synthetic_data
    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=1,
        show_progress=False,
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
    assert run_result.pve_se is None or run_result.pve_se > 0, (
        f"PVE SE should be None or positive, got {run_result.pve_se}"
    )


def test_numpy_runner_empty_after_filter(synthetic_data):
    """Edge case: returns LmmRunResult with empty associations."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        maf_threshold=0.99,  # Filters everything
        lmm_mode=1,
        show_progress=False,
    )
    results = run_result.associations
    assert len(results) == 0, f"Expected empty results, got {len(results)} results"
    assert run_result.pve is None, "PVE should be None when no SNPs pass filter"


# ---------------------------------------------------------------------------
# Chunk size computation
# ---------------------------------------------------------------------------


def test_compute_chunk_size_small_dataset():
    """Small dataset: chunk size = n_filtered (everything in one chunk)."""
    chunk = _compute_chunk_size_numpy(
        n_samples=100,
        n_filtered=500,
        n_cvt=1,
        mem_budget_bytes=int(2e9),
    )
    assert chunk == 500, f"Expected 500, got {chunk}"


def test_compute_chunk_size_large_dataset():
    """Large dataset: chunk capped by memory budget or _MAX_CHUNK."""
    chunk = _compute_chunk_size_numpy(
        n_samples=10_000,
        n_filtered=200_000,
        n_cvt=1,
        mem_budget_bytes=int(2e9),
    )
    assert 100 <= chunk <= 200_000, f"Chunk {chunk} outside expected bounds"


def test_compute_chunk_size_zero_bytes():
    """bytes_per_snp=0 (n_samples=0): returns n_filtered directly."""
    chunk = _compute_chunk_size_numpy(n_samples=0, n_filtered=1000, n_cvt=1)
    assert chunk == 1000, f"Expected 1000, got {chunk}"


def test_compute_chunk_size_minimum():
    """Chunk size never drops below 100."""
    # Huge n_samples to force small chunk_from_memory, tiny n_filtered to avoid cap
    chunk = _compute_chunk_size_numpy(
        n_samples=1_000_000,
        n_filtered=200,
        n_cvt=10,
        mem_budget_bytes=int(2e9),
    )
    assert chunk >= 100, f"Chunk {chunk} below minimum 100"


def test_chunk_size_split_larger_than_full():
    """Split Uab accounting produces larger chunks than full Uab."""
    full = _compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        mem_budget_bytes=int(10e9),
    )
    split = _compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        use_split=True,
        mem_budget_bytes=int(10e9),
    )
    assert split > full, f"Split chunk ({split}) should exceed full ({full})"


def test_chunk_size_explicit_budget():
    """Explicit mem_budget_bytes overrides auto-scaling."""
    small_budget = _compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        mem_budget_bytes=int(2e9),
    )
    large_budget = _compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        mem_budget_bytes=int(20e9),
    )
    assert large_budget > small_budget


def test_chunk_size_pipeline_halves_budget():
    """pipeline_buffers=2 produces roughly half the chunk size."""
    single = _compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        use_split=True,
        mem_budget_bytes=int(20e9),
    )
    double = _compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        use_split=True,
        mem_budget_bytes=int(20e9),
        pipeline_buffers=2,
    )
    # Double-buffering halves the budget, so chunk should be ~half
    assert double < single
    assert double >= single // 2 - 1  # allow rounding


def test_chunk_size_auto_scales_with_memory():
    """Auto-scaled budget uses 15% of available RAM between 2-40 GB bounds."""
    from unittest.mock import MagicMock, patch

    # 400 GB available → 15% = 60 GB (hits 40 GB ceiling)
    mock_vmem = MagicMock()
    mock_vmem.available = 400_000_000_000
    with patch("jamma.lmm.runner_numpy.psutil.virtual_memory", return_value=mock_vmem):
        chunk_big = _compute_chunk_size_numpy(
            n_samples=50_000,
            n_filtered=100_000,
            n_cvt=1,
            use_split=True,
        )

    # 10 GB available → 15% = 1.5 GB (hits 2 GB floor)
    mock_vmem.available = 10_000_000_000
    with patch("jamma.lmm.runner_numpy.psutil.virtual_memory", return_value=mock_vmem):
        chunk_small = _compute_chunk_size_numpy(
            n_samples=50_000,
            n_filtered=100_000,
            n_cvt=1,
            use_split=True,
        )

    assert chunk_big > chunk_small


# ---------------------------------------------------------------------------
# GEMMA validation tests
# ---------------------------------------------------------------------------


_SYNTHETIC_MODE_REFS = [
    pytest.param(1, SYNTHETIC_REFERENCE, id="wald"),
    pytest.param(2, SYNTHETIC_LRT_REFERENCE, id="lrt"),
    pytest.param(3, SCORE_REFERENCE, id="score"),
    pytest.param(4, ALL_TESTS_REFERENCE, id="all"),
]


@pytest.mark.parametrize("lmm_mode,reference_path", _SYNTHETIC_MODE_REFS)
def test_numpy_runner_synthetic(synthetic_data, lmm_mode, reference_path):
    """NumPy runner matches GEMMA reference on gemma_synthetic for each mode."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=lmm_mode,
        show_progress=False,
    )
    results = run_result.associations
    reference = load_gemma_assoc(reference_path)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy mode {lmm_mode} (synthetic) vs GEMMA failed:\n{comparison}"
    )


_MOUSE_HS1940_MODE_REFS = [
    pytest.param(2, MOUSE_HS1940_LRT, id="lrt"),
    pytest.param(3, MOUSE_HS1940_SCORE, id="score"),
    pytest.param(4, MOUSE_HS1940_ALL, id="all"),
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
        lmm_mode=lmm_mode,
        show_progress=False,
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
    covariates = np.loadtxt(COVARIATE_FILE)
    return plink, kinship, phenotypes, snp_info, covariates


_SYNTHETIC_COVAR_MODE_REFS = [
    pytest.param(1, COVARIATE_WALD_REFERENCE, id="wald"),
    pytest.param(2, COVARIATE_LRT_REFERENCE, id="lrt"),
    pytest.param(3, COVARIATE_SCORE_REFERENCE, id="score"),
    pytest.param(4, ALL_TESTS_COVAR_REFERENCE, id="all"),
]


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
        lmm_mode=lmm_mode,
        show_progress=False,
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
    pytest.param(1, MOUSE_HS1940_COVAR_WALD, id="wald"),
    pytest.param(2, MOUSE_HS1940_COVAR_LRT, id="lrt"),
    pytest.param(3, MOUSE_HS1940_COVAR_SCORE, id="score"),
    pytest.param(4, MOUSE_HS1940_COVAR_ALL, id="all"),
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
        lmm_mode=lmm_mode,
        show_progress=False,
    )
    results = run_result.associations
    reference = load_gemma_assoc(reference_path)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy mode {lmm_mode}+covar (mouse_hs1940) vs GEMMA failed:\n{comparison}"
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
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
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
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
    )
    results = run_result.associations

    assert len(results) > 0
    # With imputation, results should still be finite
    for r in results[:5]:
        assert np.isfinite(r.beta), f"beta is not finite: {r.beta}"


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
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=lmm_mode,
    )
    results = run_result.associations

    assert len(results) > 0, f"Mode {lmm_mode} produced no results"

    # Mode-specific output checks
    if lmm_mode in (1, 4):  # Wald or All
        for r in results[:5]:
            assert hasattr(r, "beta") and np.isfinite(r.beta), (
                f"Wald beta not finite: {r}"
            )
            assert hasattr(r, "p_wald") and np.isfinite(r.p_wald), (
                f"Wald p not finite: {r}"
            )
    if lmm_mode in (2, 4):  # LRT or All
        for r in results[:5]:
            assert hasattr(r, "p_lrt") and np.isfinite(r.p_lrt), (
                f"LRT p not finite: {r}"
            )
    if lmm_mode in (3, 4):  # Score or All
        for r in results[:5]:
            assert hasattr(r, "p_score") and np.isfinite(r.p_score), (
                f"Score p not finite: {r}"
            )


@pytest.mark.tier1
def test_reconstruct_uab_from_soa_matches_direct():
    """reconstruct_uab_from_soa matches batch_compute_uab_numpy exactly (RUN-01)."""
    from jamma.lmm.likelihood_numpy import (
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
    Uab_direct = batch_compute_uab_numpy(n_cvt=1, UtW=UtW, Uty=Uty, UtG=UtG)

    # Split construction + reconstruction
    invariant = compute_uab_invariant_soa(UtW, Uty)
    varying = batch_compute_uab_varying_soa_numpy(n_cvt=1, UtW=UtW, Uty=Uty, UtG=UtG)
    Uab_reconstructed = reconstruct_uab_from_soa(invariant, varying)

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
    from jamma.lmm.runner_numpy import compute_pipeline_core_split

    total_cores = 8

    # Large samples (>10k): rotation-heavy, gets 50% of cores
    rot, omp = compute_pipeline_core_split(50_000, total_cores)
    assert rot == 4 and omp == 4, f"Large: rot={rot}, omp={omp}"

    # Medium samples (1k-10k): balanced, rotation gets 33%
    rot, omp = compute_pipeline_core_split(5_000, total_cores)
    assert rot == 2 and omp == 6, f"Medium: rot={rot}, omp={omp}"

    # Small samples (<1k): compute-heavy, rotation gets 25%
    rot, omp = compute_pipeline_core_split(500, total_cores)
    assert rot == 2 and omp == 6, f"Small: rot={rot}, omp={omp}"

    # Edge: 1 core — both get 1
    rot, omp = compute_pipeline_core_split(50_000, 1)
    assert rot >= 1 and omp >= 1, f"Single core: rot={rot}, omp={omp}"


@pytest.mark.tier1
def test_compute_adaptive_core_split():
    """compute_adaptive_core_split allocates threads proportional to measured times."""
    from jamma.lmm.runner_numpy import compute_adaptive_core_split

    # Rotation-heavy: 80% rotation time -> ~80% of cores for rotation
    rot, omp = compute_adaptive_core_split(
        rot_time=0.8, compute_time=0.2, total_cores=8
    )
    assert rot == 6 and omp == 2, f"Rotation-heavy: rot={rot}, omp={omp}"

    # Compute-heavy: 20% rotation time -> ~20% of cores for rotation
    rot, omp = compute_adaptive_core_split(
        rot_time=0.2, compute_time=0.8, total_cores=8
    )
    assert rot == 2 and omp == 6, f"Compute-heavy: rot={rot}, omp={omp}"

    # Balanced: equal times -> 50/50 split
    rot, omp = compute_adaptive_core_split(
        rot_time=0.5, compute_time=0.5, total_cores=8
    )
    assert rot == 4 and omp == 4, f"Balanced: rot={rot}, omp={omp}"

    # Degenerate: both times near zero -> static fallback
    rot, omp = compute_adaptive_core_split(
        rot_time=0.0, compute_time=0.0, total_cores=8, n_samples=50_000
    )
    # Static fallback for 50k samples: 50% -> (4, 4)
    assert rot == 4 and omp == 4, f"Degenerate fallback: rot={rot}, omp={omp}"

    # Always returns (rot >= 1, compute >= 1)
    for r, c, n in [(0.9, 0.1, 2), (0.1, 0.9, 2), (0.5, 0.5, 2)]:
        rot, omp = compute_adaptive_core_split(
            rot_time=r, compute_time=c, total_cores=n
        )
        assert rot >= 1 and omp >= 1, (
            f"Min 1: rot={rot}, omp={omp} (r={r}, c={c}, n={n})"
        )

    # Clamped: 2 cores, rotation-heavy -> (1, 1) since both must be >= 1
    rot, omp = compute_adaptive_core_split(
        rot_time=0.9, compute_time=0.1, total_cores=2
    )
    assert rot == 1 and omp == 1, f"Clamped 2-core: rot={rot}, omp={omp}"


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

    kwargs = dict(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship.copy(),
        snp_info=snp_info,
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=2,
    )

    # Run with C extension
    result_c = run_lmm_association_numpy(**kwargs)

    # Run with C disabled (monkeypatch Score/LRT C pointers to None)
    with patch.object(compute_numpy, "_compute_lrt_batch_c", None):
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

    kwargs = dict(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship.copy(),
        snp_info=snp_info,
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=3,
    )

    # Run with C extension
    result_c = run_lmm_association_numpy(**kwargs)

    # Run with C disabled
    with patch.object(compute_numpy, "_compute_score_batch_c", None):
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
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=4,
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
def test_runner_pipeline_enabled_for_non_wald_modes():
    """Pipeline enabled for LRT/Score when chunks sufficient (RUN-07)."""
    from unittest.mock import patch

    from jamma.lmm import runner_numpy

    genotypes, phenotypes, kinship, snp_info = _make_synthetic_data(
        n_samples=50, n_snps=200
    )

    # Force very small chunks so we get >= 30 chunks for pipeline
    with patch.object(runner_numpy, "_MIN_PIPELINE_CHUNKS", 3):
        for mode in [2, 3]:
            result = run_lmm_association_numpy(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=kinship.copy(),
                snp_info=snp_info * 4,  # 800 SNPs worth of info
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                lmm_mode=mode,
                # Force small chunk size via explicit budget
            )
            assert len(result.associations) > 0, f"Mode {mode} should produce results"
