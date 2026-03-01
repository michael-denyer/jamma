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
    """Type check: NumPy runner returns list[AssocResult]."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=1,
        show_progress=False,
    )
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, "Expected at least one result"
    assert isinstance(results[0], AssocResult), (
        f"Expected AssocResult, got {type(results[0])}"
    )


def test_numpy_runner_empty_after_filter(synthetic_data):
    """Edge case: returns empty list when all SNPs are filtered out."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        maf_threshold=0.99,  # Filters everything
        lmm_mode=1,
        show_progress=False,
    )
    assert results == [], f"Expected empty list, got {len(results)} results"


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
    """Auto-scaled budget uses 5% of available RAM between 2-20 GB bounds."""
    from unittest.mock import MagicMock, patch

    # 400 GB available → 5% = 20 GB (hits ceiling)
    mock_vmem = MagicMock()
    mock_vmem.available = 400_000_000_000
    with patch("jamma.lmm.runner_numpy.psutil.virtual_memory", return_value=mock_vmem):
        chunk_big = _compute_chunk_size_numpy(
            n_samples=50_000,
            n_filtered=100_000,
            n_cvt=1,
            use_split=True,
        )

    # 20 GB available → 5% = 1 GB (hits floor at 2 GB)
    mock_vmem.available = 20_000_000_000
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


def test_numpy_runner_wald_synthetic(synthetic_data):
    """Mode 1 (Wald): NumPy runner matches GEMMA reference on gemma_synthetic."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=1,
        show_progress=False,
    )
    reference = load_gemma_assoc(SYNTHETIC_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy Wald (synthetic) vs GEMMA comparison failed:\n{comparison}"
    )


@pytest.mark.slow
@pytest.mark.tier2
def test_numpy_runner_lrt_mouse_hs1940(mouse_hs1940_data):
    """Mode 2 (LRT): NumPy runner matches GEMMA on mouse_hs1940 (1410 samples)."""
    plink, kinship, phenotypes, snp_info = mouse_hs1940_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=2,
        show_progress=False,
    )
    reference = load_gemma_assoc(MOUSE_HS1940_LRT)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy LRT (mouse_hs1940) vs GEMMA comparison failed:\n{comparison}"
    )


@pytest.mark.slow
@pytest.mark.tier2
def test_numpy_runner_score_mouse_hs1940(mouse_hs1940_data):
    """Mode 3 (Score): NumPy runner matches GEMMA on mouse_hs1940."""
    plink, kinship, phenotypes, snp_info = mouse_hs1940_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=3,
        show_progress=False,
    )
    reference = load_gemma_assoc(MOUSE_HS1940_SCORE)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy Score (mouse_hs1940) vs GEMMA comparison failed:\n{comparison}"
    )


@pytest.mark.slow
@pytest.mark.tier2
def test_numpy_runner_all_mouse_hs1940(mouse_hs1940_data):
    """Mode 4 (All): NumPy runner matches GEMMA on mouse_hs1940 for Wald+LRT+Score."""
    plink, kinship, phenotypes, snp_info = mouse_hs1940_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=4,
        show_progress=False,
    )
    reference = load_gemma_assoc(MOUSE_HS1940_ALL)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy All (mouse_hs1940) vs GEMMA comparison failed:\n{comparison}"
    )


def test_numpy_runner_lrt_synthetic(synthetic_data):
    """Mode 2 (LRT): NumPy runner matches GEMMA gemma_lrt.assoc.txt reference."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=2,
        show_progress=False,
    )
    reference = load_gemma_assoc(SYNTHETIC_LRT_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, f"NumPy LRT (synthetic) vs GEMMA failed:\n{comparison}"


def test_numpy_runner_score_synthetic(synthetic_data):
    """Mode 3 (Score): NumPy runner matches GEMMA gemma_score.assoc.txt reference."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=3,
        show_progress=False,
    )
    reference = load_gemma_assoc(SCORE_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, f"NumPy Score (synthetic) vs GEMMA failed:\n{comparison}"


def test_numpy_runner_all_synthetic(synthetic_data):
    """Mode 4 (All): NumPy runner matches GEMMA gemma_all.assoc.txt reference."""
    plink, kinship, phenotypes, snp_info = synthetic_data
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        lmm_mode=4,
        show_progress=False,
    )
    reference = load_gemma_assoc(ALL_TESTS_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, f"NumPy All (synthetic) vs GEMMA failed:\n{comparison}"


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


def test_numpy_runner_wald_covar_synthetic(synthetic_data_with_covariates):
    """Mode 1 (Wald) with covariates: NumPy runner matches GEMMA reference."""
    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=1,
        show_progress=False,
    )
    reference = load_gemma_assoc(COVARIATE_WALD_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy Wald+covar (synthetic) vs GEMMA failed:\n{comparison}"
    )


def test_numpy_runner_lrt_covar_synthetic(synthetic_data_with_covariates):
    """Mode 2 (LRT) with covariates: NumPy runner matches GEMMA reference."""
    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=2,
        show_progress=False,
    )
    reference = load_gemma_assoc(COVARIATE_LRT_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy LRT+covar (synthetic) vs GEMMA failed:\n{comparison}"
    )


def test_numpy_runner_score_covar_synthetic(synthetic_data_with_covariates):
    """Mode 3 (Score) with covariates: NumPy runner matches GEMMA reference."""
    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=3,
        show_progress=False,
    )
    reference = load_gemma_assoc(COVARIATE_SCORE_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy Score+covar (synthetic) vs GEMMA failed:\n{comparison}"
    )


def test_numpy_runner_all_covar_synthetic(synthetic_data_with_covariates):
    """Mode 4 (All) with covariates: NumPy runner matches GEMMA reference."""
    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=4,
        show_progress=False,
    )
    reference = load_gemma_assoc(ALL_TESTS_COVAR_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy All+covar (synthetic) vs GEMMA failed:\n{comparison}"
    )


# ---------------------------------------------------------------------------
# mouse_hs1940 covariate GEMMA validation tests (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.tier2
def test_numpy_runner_wald_covar_mouse_hs1940(mouse_hs1940_data_with_covariates):
    """Mode 1 (Wald) with covariates: NumPy runner matches GEMMA on mouse_hs1940."""
    plink, kinship, phenotypes, snp_info, covariates = mouse_hs1940_data_with_covariates
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=1,
        show_progress=False,
    )
    reference = load_gemma_assoc(MOUSE_HS1940_COVAR_WALD)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy Wald+covar (mouse_hs1940) vs GEMMA failed:\n{comparison}"
    )


@pytest.mark.slow
@pytest.mark.tier2
def test_numpy_runner_lrt_covar_mouse_hs1940(mouse_hs1940_data_with_covariates):
    """Mode 2 (LRT) with covariates: NumPy runner matches GEMMA on mouse_hs1940."""
    plink, kinship, phenotypes, snp_info, covariates = mouse_hs1940_data_with_covariates
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=2,
        show_progress=False,
    )
    reference = load_gemma_assoc(MOUSE_HS1940_COVAR_LRT)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy LRT+covar (mouse_hs1940) vs GEMMA failed:\n{comparison}"
    )


@pytest.mark.slow
@pytest.mark.tier2
def test_numpy_runner_score_covar_mouse_hs1940(mouse_hs1940_data_with_covariates):
    """Mode 3 (Score) with covariates: NumPy runner matches GEMMA on mouse_hs1940."""
    plink, kinship, phenotypes, snp_info, covariates = mouse_hs1940_data_with_covariates
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=3,
        show_progress=False,
    )
    reference = load_gemma_assoc(MOUSE_HS1940_COVAR_SCORE)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy Score+covar (mouse_hs1940) vs GEMMA failed:\n{comparison}"
    )


@pytest.mark.slow
@pytest.mark.tier2
def test_numpy_runner_all_covar_mouse_hs1940(mouse_hs1940_data_with_covariates):
    """Mode 4 (All) with covariates: NumPy runner matches GEMMA on mouse_hs1940."""
    plink, kinship, phenotypes, snp_info, covariates = mouse_hs1940_data_with_covariates
    results = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=4,
        show_progress=False,
    )
    reference = load_gemma_assoc(MOUSE_HS1940_COVAR_ALL)
    comparison = compare_assoc_results(results, reference, NUMPY_GEMMA_TOLERANCES)
    assert comparison.passed, (
        f"NumPy All+covar (mouse_hs1940) vs GEMMA failed:\n{comparison}"
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
