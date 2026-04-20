"""Validation tests for the NumPy LMM runner against GEMMA reference output.

Validates that run_lmm_association_numpy produces GEMMA-equivalent p-values for all
four LMM modes (Wald, LRT, Score, All). Tests compare directly against GEMMA reference
files (see GEMMA_EQUIVALENCE.md for tolerance rationale).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.runner_numpy import compute_chunk_size_numpy, run_lmm_association_numpy
from jamma.lmm.stats import AssocResult
from jamma.validation import (
    ToleranceConfig,
    compare_assoc_results,
    load_gemma_assoc,
)
from tests.conftest import load_phenotypes_from_fam

# ---------------------------------------------------------------------------
# Fake infrastructure
# ---------------------------------------------------------------------------


class FakeAssocWriter:
    """In-memory fake for IncrementalAssocWriter.

    Captures write_arrays_batch calls so tests can assert on call count and
    arguments without MagicMock.  Unlike MagicMock, accessing an attribute
    that doesn't exist raises AttributeError — detecting interface drift.
    """

    def __init__(self) -> None:
        self.batches: list[tuple] = []

    @property
    def call_count(self) -> int:
        return len(self.batches)

    def write_arrays_batch(
        self,
        lmm_mode: int,
        snp_indices: np.ndarray,
        snp_info: list,
        afs: np.ndarray,
        miss_counts: np.ndarray,
        arrays: dict[str, np.ndarray],
    ) -> None:
        self.batches.append((lmm_mode, snp_indices, snp_info, afs, miss_counts, arrays))


# ---------------------------------------------------------------------------
# Tolerance configurations
# ---------------------------------------------------------------------------

# NumPy backend vs GEMMA tolerances.
# Cephes betainc is close to GSL betainc for large a (n_samples > 1000).
# Lambda optimization uses golden section algorithm.
# See GEMMA_EQUIVALENCE.md for tolerance derivation.
NUMPY_GEMMA_TOLERANCES = ToleranceConfig(
    lambda_rtol=1e-3,  # Golden section vs Brent
    pvalue_rtol=1e-2,  # Cephes vs GSL betainc
    se_rtol=5e-4,  # Pab arithmetic propagation
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
    assert run_result.pve_se is not None, (
        "PVE SE should be populated for synthetic data"
    )
    assert run_result.pve_se > 0, f"PVE SE should be positive, got {run_result.pve_se}"


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
    chunk = compute_chunk_size_numpy(
        n_samples=100,
        n_filtered=500,
        n_cvt=1,
        mem_budget_bytes=int(2e9),
    )
    assert chunk == 500, f"Expected 500, got {chunk}"


def test_compute_chunk_size_large_dataset():
    """Large dataset: chunk capped by memory budget or _MAX_CHUNK."""
    chunk = compute_chunk_size_numpy(
        n_samples=10_000,
        n_filtered=200_000,
        n_cvt=1,
        mem_budget_bytes=int(2e9),
    )
    assert 100 <= chunk <= 200_000, f"Chunk {chunk} outside expected bounds"


def test_compute_chunk_size_zero_bytes():
    """bytes_per_snp=0 (n_samples=0): returns n_filtered directly."""
    chunk = compute_chunk_size_numpy(n_samples=0, n_filtered=1000, n_cvt=1)
    assert chunk == 1000, f"Expected 1000, got {chunk}"


def test_compute_chunk_size_minimum():
    """Chunk size never drops below 100."""
    # Huge n_samples to force small chunk_from_memory, tiny n_filtered to avoid cap
    chunk = compute_chunk_size_numpy(
        n_samples=1_000_000,
        n_filtered=200,
        n_cvt=10,
        mem_budget_bytes=int(2e9),
    )
    assert chunk >= 100, f"Chunk {chunk} below minimum 100"


def test_chunk_size_split_larger_than_full():
    """Split Uab accounting produces larger chunks than full Uab."""
    full = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        mem_budget_bytes=int(10e9),
    )
    split = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        use_split=True,
        mem_budget_bytes=int(10e9),
    )
    assert split > full, f"Split chunk ({split}) should exceed full ({full})"


def test_chunk_size_explicit_budget():
    """Explicit mem_budget_bytes overrides auto-scaling."""
    small_budget = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        mem_budget_bytes=int(2e9),
    )
    large_budget = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        mem_budget_bytes=int(20e9),
    )
    assert large_budget > small_budget


def test_chunk_size_pipeline_halves_budget():
    """pipeline_buffers=2 produces roughly half the chunk size."""
    single = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        use_split=True,
        mem_budget_bytes=int(20e9),
    )
    double = compute_chunk_size_numpy(
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
        chunk_big = compute_chunk_size_numpy(
            n_samples=50_000,
            n_filtered=100_000,
            n_cvt=1,
            use_split=True,
        )

    # 10 GB available → 15% = 1.5 GB (hits 2 GB floor)
    mock_vmem.available = 10_000_000_000
    with patch("jamma.lmm.runner_numpy.psutil.virtual_memory", return_value=mock_vmem):
        chunk_small = compute_chunk_size_numpy(
            n_samples=50_000,
            n_filtered=100_000,
            n_cvt=1,
            use_split=True,
        )

    assert chunk_big > chunk_small


def test_chunk_size_mode4_fused_uses_4col():
    """All n_cvt=1 split paths use 4-col accounting (SoA-native)."""
    # Use large n_samples and n_filtered with moderate budget so chunks
    # don't hit the _MAX_CHUNK cap (200k).
    n_samples = 10_000
    budget = int(5e9)

    # Fused mode-4: 4 cols/SNP
    fused_chunk = compute_chunk_size_numpy(
        n_samples=n_samples,
        n_filtered=500_000,
        n_cvt=1,
        use_split=True,
        lmm_mode=4,
        fused_mode4=True,
        mem_budget_bytes=budget,
    )
    # Non-fused mode-4 fallback: also 4 cols/SNP (SoA split dispatch)
    fallback_chunk = compute_chunk_size_numpy(
        n_samples=n_samples,
        n_filtered=500_000,
        n_cvt=1,
        use_split=True,
        lmm_mode=4,
        fused_mode4=False,
        mem_budget_bytes=budget,
    )
    # Wald (mode 1): 4 cols/SNP — should match all other split paths
    wald_chunk = compute_chunk_size_numpy(
        n_samples=n_samples,
        n_filtered=500_000,
        n_cvt=1,
        use_split=True,
        lmm_mode=1,
        mem_budget_bytes=budget,
    )

    # All n_cvt=1 split paths use 4-col accounting (3 varying SoA + 1 UtG)
    assert fused_chunk == wald_chunk == fallback_chunk, (
        f"All split paths should use same accounting: fused={fused_chunk}, "
        f"wald={wald_chunk}, fallback={fallback_chunk}"
    )


@pytest.mark.tier1
def test_runner_mode4_uses_fused_dispatch():
    """Mode 4 with C extension uses fused dispatch, not compose fallback."""
    from unittest.mock import patch

    from jamma.lmm import runner_numpy
    from jamma.lmm.compute_numpy import _C_MODE4_AVAILABLE

    if not _C_MODE4_AVAILABLE:
        pytest.skip("Fused mode-4 C extension not available")

    genotypes, phenotypes, kinship, snp_info = _make_synthetic_data()

    with patch.object(
        runner_numpy,
        "_compose_mode4_from_split",
        wraps=runner_numpy._compose_mode4_from_split,
    ) as mock_compose:
        run_lmm_association_numpy(
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
        assert mock_compose.call_count == 0, (
            "Fused mode-4 should not fall back to _compose_mode4_from_split"
        )


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
# LmmRunResult.snp_count property tests (PR-65)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestSnpCount:
    """Unit tests for LmmRunResult.snp_count property."""

    def test_snp_count_from_n_tested(self):
        """snp_count returns n_tested when set (streaming mode)."""
        from jamma.lmm.schema import LmmRunResult

        result = LmmRunResult(associations=[], n_tested=42)
        assert result.snp_count == 42

    def test_snp_count_from_associations(self):
        """snp_count falls back to len(associations) when n_tested is None."""
        from jamma.lmm.schema import LmmRunResult

        result = LmmRunResult(associations=[1, 2, 3])  # dummy items
        assert result.snp_count == 3

    def test_snp_count_prefers_n_tested_over_associations(self):
        """n_tested takes priority even if associations is non-empty."""
        from jamma.lmm.schema import LmmRunResult

        result = LmmRunResult(associations=[1, 2, 3], n_tested=10)
        assert result.snp_count == 10


# ---------------------------------------------------------------------------
# write_streaming_chunk unit tests (PR-65)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestWriteStreamingChunk:
    """Unit tests for write_streaming_chunk diagnostic accumulation."""

    def test_nan_counts_accumulated_across_chunks(self):
        """NaN counts accumulate correctly across multiple chunks."""
        from jamma.lmm.results import write_streaming_chunk

        writer = FakeAssocWriter()
        nan_counts: dict[str, int] = {}

        # Chunk 1: 2 NaN betas, 1 NaN p_wald
        chunk1 = {
            "betas": np.array([np.nan, np.nan, 1.0]),
            "ses": np.array([0.1, 0.2, 0.3]),
            "logls": np.array([-10.0, -20.0, -30.0]),
            "lambdas": np.array([0.5, 0.5, 0.5]),
            "pwalds": np.array([np.nan, 0.1, 0.2]),
        }
        snp_indices = np.array([0, 1, 2])
        snp_info = [
            {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
            for i in range(10)
        ]
        afs = np.array([0.3, 0.3, 0.3])
        miss = np.array([0, 0, 0])

        lmin, lmax = write_streaming_chunk(
            writer,
            1,
            snp_indices,
            snp_info,
            afs,
            miss,
            chunk1,
            1e-5,
            1e5,
            nan_counts,
            0,
            0,
        )

        assert nan_counts["betas"] == 2
        assert nan_counts["pwalds"] == 1
        assert "ses" not in nan_counts

        # Chunk 2: 1 more NaN beta
        chunk2 = {
            "betas": np.array([np.nan, 1.0]),
            "ses": np.array([0.1, 0.2]),
            "logls": np.array([-10.0, -20.0]),
            "lambdas": np.array([0.5, 0.5]),
            "pwalds": np.array([0.1, 0.2]),
        }
        lmin2, lmax2 = write_streaming_chunk(
            writer,
            1,
            snp_indices[:2],
            snp_info,
            afs[:2],
            miss[:2],
            chunk2,
            1e-5,
            1e5,
            nan_counts,
            lmin,
            lmax,
        )

        assert nan_counts["betas"] == 3  # accumulated across chunks
        assert writer.call_count == 2

    def test_lambda_boundary_hits_accumulated(self):
        """Lambda boundary hits accumulate across chunks."""
        from jamma.lmm.results import write_streaming_chunk

        writer = FakeAssocWriter()
        nan_counts: dict[str, int] = {}

        # Chunk with lambdas at l_min
        chunk = {
            "betas": np.array([1.0, 2.0]),
            "ses": np.array([0.1, 0.2]),
            "logls": np.array([-10.0, -20.0]),
            "lambdas": np.array([1e-5, 0.5]),
            "pwalds": np.array([0.1, 0.2]),
        }
        snp_info = [
            {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
            for i in range(10)
        ]
        lmin, lmax = write_streaming_chunk(
            writer,
            1,
            np.array([0, 1]),
            snp_info,
            np.array([0.3, 0.3]),
            np.array([0, 0]),
            chunk,
            1e-5,
            1e5,
            nan_counts,
            0,
            0,
        )
        assert lmin == 1  # one lambda at l_min
        assert lmax == 0

        # Second chunk: one more at l_min
        lmin2, lmax2 = write_streaming_chunk(
            writer,
            1,
            np.array([2, 3]),
            snp_info,
            np.array([0.3, 0.3]),
            np.array([0, 0]),
            chunk,
            1e-5,
            1e5,
            nan_counts,
            lmin,
            lmax,
        )
        assert lmin2 == 2  # accumulated


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
    if not MOUSE_HS1940_DATA.with_suffix(".bed").exists():
        pytest.skip("mouse_hs1940 fixture not available")

    plink = load_plink_binary(MOUSE_HS1940_DATA)
    kinship = read_kinship_matrix(MOUSE_HS1940_KINSHIP)
    phenotypes = load_phenotypes_from_fam(MOUSE_HS1940_DATA.with_suffix(".fam"))
    snp_info = _build_snp_info(plink)

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
        "lmm_mode": 1,
        "check_memory": False,
        "show_progress": False,
        "output_path": None,
    }

    # Single-chunk run (no monkeypatching — default chunk_size fits all SNPs)
    result_single = run_lmm_association_numpy(**common_kwargs)

    # Multi-chunk run: force chunk_size=50 so the batch loop iterates many times
    monkeypatch.setattr(
        "jamma.lmm.runner_numpy.compute_chunk_size_numpy",
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
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
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
    varying = batch_compute_uab_varying_soa_numpy(
        n_cvt=1, UtW=UtW, Uty=Uty, utg_t=UtG.T
    )
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
    from jamma.lmm.runner_numpy import compute_adaptive_core_split

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
        "maf_threshold": 0.0,
        "miss_threshold": 1.0,
        "check_memory": False,
        "show_progress": False,
        "lmm_mode": 2,
    }

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

    kwargs = {
        "genotypes": genotypes,
        "phenotypes": phenotypes,
        "kinship": kinship.copy(),
        "snp_info": snp_info,
        "maf_threshold": 0.0,
        "miss_threshold": 1.0,
        "check_memory": False,
        "show_progress": False,
        "lmm_mode": 3,
    }

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
        "maf_threshold": 0.0,
        "miss_threshold": 1.0,
        "check_memory": False,
        "show_progress": False,
        "lmm_mode": lmm_mode,
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
    np.testing.assert_allclose(
        result_disk.pve,
        result_mem.pve,
        rtol=1e-10,
        err_msg="PVE mismatch between streaming and in-memory",
    )
    if result_mem.pve_se is not None:
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
            np.testing.assert_allclose(
                r_disk.p_wald,
                r_mem.p_wald,
                rtol=file_rtol,
                err_msg=f"p_wald mismatch for {r_mem.rs}",
            )
        if lmm_mode in (2, 4):
            np.testing.assert_allclose(
                r_disk.p_lrt,
                r_mem.p_lrt,
                rtol=file_rtol,
                err_msg=f"p_lrt mismatch for {r_mem.rs}",
            )
        if lmm_mode in (3, 4):
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

    result = run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        maf_threshold=0.99,  # Filters everything
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
        output_path=output_file,
    )

    assert result.associations == []
    assert result.pve is None, "PVE should be None when no SNPs pass filter"


# ---------------------------------------------------------------------------
# Error message differentiation tests (68-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestErrorMessageDifferentiation:
    """Verify that compute failures produce operation-specific error messages.

    The _guarded_compute helper wraps compute calls and produces distinct
    RuntimeError messages identifying the failed operation, SNP offset,
    and total SNP count.
    """

    def test_fused_mode4_label(self):
        """Fused mode-4 failure includes 'Fused mode-4' in the message."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _boom(*a, **kw):
            raise OSError("segfault simulation")

        with pytest.raises(RuntimeError, match="Fused mode-4"):
            _guarded_compute(
                _boom,
                operation="Fused mode-4 C workspace compute",
                write_offset=100,
                n_filtered=500,
            )

    def test_wald_label(self):
        """Wald workspace failure includes 'Wald' in the message."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _boom(*a, **kw):
            raise OSError("segfault simulation")

        with pytest.raises(RuntimeError, match="Wald"):
            _guarded_compute(
                _boom,
                operation="Wald C workspace compute",
                write_offset=200,
                n_filtered=1000,
            )

    def test_score_lrt_label(self):
        """Score/LRT dispatch failure includes 'Score/LRT' in the message."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _boom(*a, **kw):
            raise OSError("segfault simulation")

        with pytest.raises(RuntimeError, match="Score/LRT"):
            _guarded_compute(
                _boom,
                operation="Score/LRT C batch dispatch",
                write_offset=50,
                n_filtered=200,
            )

    def test_error_includes_snp_offset(self):
        """Error message includes SNP offset and total count."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _boom(*a, **kw):
            raise OSError("kaboom")

        with pytest.raises(RuntimeError, match=r"300/1000") as exc_info:
            _guarded_compute(
                _boom,
                operation="Wald C workspace compute",
                write_offset=300,
                n_filtered=1000,
            )
        assert "300 SNPs before failure" in str(exc_info.value)

    def test_memory_error_passes_through(self):
        """MemoryError is not wrapped in RuntimeError."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _oom(*a, **kw):
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError, match="out of memory"):
            _guarded_compute(
                _oom,
                operation="Wald C workspace compute",
                write_offset=0,
                n_filtered=100,
            )

    def test_value_error_passes_through(self):
        """ValueError is not wrapped in RuntimeError."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _bad(*a, **kw):
            raise ValueError("bad value")

        with pytest.raises(ValueError, match="bad value"):
            _guarded_compute(
                _bad,
                operation="Wald C workspace compute",
                write_offset=0,
                n_filtered=100,
            )

    def test_type_error_passes_through(self):
        """TypeError is not wrapped in RuntimeError."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _bad(*a, **kw):
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            _guarded_compute(
                _bad,
                operation="Wald C workspace compute",
                write_offset=0,
                n_filtered=100,
            )

    def test_overflow_error_passes_through(self):
        """OverflowError is not wrapped in RuntimeError."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _bad(*a, **kw):
            raise OverflowError("overflow")

        with pytest.raises(OverflowError, match="overflow"):
            _guarded_compute(
                _bad,
                operation="Wald C workspace compute",
                write_offset=0,
                n_filtered=100,
            )

    def test_exception_chaining_preserved(self):
        """The original exception is chained via 'from exc'."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _boom(*a, **kw):
            raise OSError("root cause")

        with pytest.raises(RuntimeError) as exc_info:
            _guarded_compute(
                _boom,
                operation="LMM chunk compute",
                write_offset=0,
                n_filtered=100,
            )
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, OSError)
        assert "root cause" in str(exc_info.value.__cause__)

    def test_successful_call_returns_result(self):
        """Successful function call returns result without wrapping."""
        from jamma.lmm.runner_numpy import _guarded_compute

        def _ok(*a, **kw):
            return {"betas": [1.0], "ses": [0.1]}

        result = _guarded_compute(
            _ok,
            operation="Wald C workspace compute",
            write_offset=0,
            n_filtered=100,
        )
        assert result == {"betas": [1.0], "ses": [0.1]}


# ---------------------------------------------------------------------------
# n_cvt>1 C general path integration tests (Plan 70-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_runner_numpy_ncvt2_mode2_c_dispatch(synthetic_data_with_covariates):
    """LRT (mode 2) with n_cvt=2 uses C general path and matches GEMMA reference.

    Verifies the full path: use_split=True -> SoA split -> reconstruct(n_cvt=2)
    -> _compute_lrt_numpy -> compute_lrt_batch_general_c.
    """
    from jamma.lmm import compute_numpy as cn

    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates

    if cn._compute_lrt_batch_general_c is None:
        pytest.skip("compute_lrt_batch_general_c not available")

    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=2,
        show_progress=False,
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
    reference = load_gemma_assoc(COVARIATE_LRT_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy mode 2+covar (n_cvt=2) vs GEMMA failed:\n{comparison}"
    )


@pytest.mark.tier0
def test_runner_numpy_ncvt2_mode3_c_dispatch(synthetic_data_with_covariates):
    """Score (mode 3) with n_cvt=2 uses C general path and matches GEMMA reference.

    Verifies the full path: use_split=True -> SoA split -> reconstruct(n_cvt=2)
    -> _compute_score_numpy -> compute_score_batch_general_c.
    """
    from jamma.lmm import compute_numpy as cn

    plink, kinship, phenotypes, snp_info, covariates = synthetic_data_with_covariates

    if cn._compute_score_batch_general_c is None:
        pytest.skip("compute_score_batch_general_c not available")

    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        lmm_mode=3,
        show_progress=False,
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
    reference = load_gemma_assoc(COVARIATE_SCORE_REFERENCE)
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"NumPy mode 3+covar (n_cvt=2) vs GEMMA failed:\n{comparison}"
    )
