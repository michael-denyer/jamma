"""Tests for the NumPy disk-streaming LMM runner.

Validates GEMMA parity (SC-01), batch equivalence (SC-02),
streaming mechanics (SC-03), and chunking edge cases (SC-04).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary, read_fam_phenotypes
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from jamma.lmm.runner_numpy_streaming import (
    run_lmm_association_numpy_streaming,
)
from jamma.lmm.schema import LmmConfig, SnpMeta
from jamma.lmm.stats import AssocResult
from jamma.validation import (
    ToleranceConfig,
    compare_assoc_results,
    load_gemma_assoc,
)
from tests.conftest import requires_c
from tests.fixture_paths import SYNTHETIC

# ---------------------------------------------------------------------------
# Sanitizer-aware tolerances
# ---------------------------------------------------------------------------

# Optimized builds compute batch and streaming mode-1/mode-4 results that are
# bitwise-identical (max relative difference 0.0). Under the ASAN/UBSAN build
# (JAMMA_SANITIZE set), the C extension is instrumented and compiled without
# vectorization, so the SoA-split and full-Uab accumulation orders drift by up
# to ~2e-10 on isolated elements. Widen the batch/streaming equivalence rtol
# for that build only; optimized CI keeps the strict bitwise-parity bound.
_SANITIZER_BUILD = bool(os.environ.get("JAMMA_SANITIZE", "").strip())
_FP_PARITY_RTOL_MODE1 = 1e-8 if _SANITIZER_BUILD else 1e-12
_FP_PARITY_RTOL_MODE4 = 1e-8 if _SANITIZER_BUILD else 1e-10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_shared_lmm_chunk_runner_avoids_transposed_u_copy_in_jlinalg_dgemm():
    """Hot path must use transa='T', not pass a transposed array into dgemm.

    jlinalg's C binding copies non-contiguous inputs to C-contiguous buffers.
    Passing U.T would therefore materialize an O(n^2) copy of the eigenvector
    matrix inside the shared LMM chunk loop used by batch, streaming, and LOCO.

    Structural (AST) guard, kept at tier0 per TESTING.md §2.4: it pins a
    performance invariant that no behavior test can catch, because a
    transposed copy changes speed, not results.

    The rule is about the shape of the call, not about what the operands are
    called. An earlier version only recognised the rotation when its second
    argument was a bare name, and only rejected a transpose when the array was
    named U, so moving either into an attribute would have quietly disarmed it.
    """
    source = (
        Path(__file__).parent.parent / "src" / "jamma" / "lmm" / "chunk_runner_numpy.py"
    ).read_text()
    tree = ast.parse(source)

    saw_rotation = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        is_dgemm = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "jlinalg"
            and func.attr == "dgemm"
        )
        if is_dgemm and node.args:
            assert not (
                isinstance(node.args[0], ast.Attribute) and node.args[0].attr == "T"
            ), (
                "chunk runner must not pass a transposed array into jlinalg.dgemm; "
                "use transa='T' so the C binding transposes in place"
            )
            if any(
                kw.arg == "transa"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "T"
                for kw in node.keywords
            ):
                saw_rotation = True

        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "np"
            and func.attr == "ascontiguousarray"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "T"
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "UtG"
        ):
            pytest.fail(
                "streaming runner must not materialize np.ascontiguousarray(UtG.T)"
            )

    assert saw_rotation, "no jlinalg.dgemm(..., transa='T') rotation found"


@pytest.mark.tier0
@pytest.mark.parametrize("bad", [0, -1])
def test_streaming_rejects_a_chunk_size_below_one_before_reading_the_bed(bad, tmp_path):
    """A bad chunk_size must fail before pass 1, not after it.

    None means "not specified"; zero does not. Deriving the statistics-pass
    block with ``chunk_size or DEFAULT_STATS_CHUNK`` would read zero as unset,
    stream the entire .bed at the default width, and only then raise from the
    chunk runner, which on a large dataset is minutes of I/O before the
    complaint.
    """
    with pytest.raises(ValueError, match="chunk_size must be >= 1 or None"):
        run_lmm_association_numpy_streaming(
            bed_path=tmp_path / "does-not-exist",
            phenotypes=np.zeros(4),
            kinship=np.eye(4),
            chunk_size=bad,
            config=LmmConfig(check_memory=False, show_progress=False),
        )


@pytest.fixture
def synthetic_data():
    """Load gemma_synthetic PLINK data, kinship, phenotypes."""
    plink = load_plink_binary(SYNTHETIC.bfile)
    kinship = read_kinship_matrix(SYNTHETIC.kinship)
    phenotypes = read_fam_phenotypes(SYNTHETIC.fam)
    return plink, kinship, phenotypes


@pytest.fixture
def synthetic_eigen(synthetic_data):
    """Pre-compute eigendecomposition on filtered kinship for synthetic data.

    Avoids redundant O(n^3) eigendecomposition across tests.
    """
    plink, kinship, phenotypes = synthetic_data
    valid_mask = ~np.isnan(phenotypes)
    kinship_filtered = kinship[np.ix_(valid_mask, valid_mask)]
    eigenvalues, eigenvectors = np.linalg.eigh(kinship_filtered)
    return plink, kinship, phenotypes, eigenvalues, eigenvectors


@pytest.fixture
def synthetic_data_with_covariates(synthetic_data):
    """Load gemma_synthetic data plus covariates from gemma_covariate fixture.

    The covariates.txt file already includes the intercept column (first column
    is all 1.0), matching GEMMA's internal representation when -c is used.

    Pre-computes eigendecomposition so callers share identical eigen and avoid
    the kinship in-place overwrite (DSYEVD-inplace) between sequential runner
    calls.
    """
    plink, kinship, phenotypes = synthetic_data
    covariates = np.loadtxt(SYNTHETIC.covariates)
    valid_mask = ~np.isnan(phenotypes)
    kinship_filtered = kinship[np.ix_(valid_mask, valid_mask)]
    eigenvalues, eigenvectors = np.linalg.eigh(kinship_filtered)
    return plink, phenotypes, covariates, eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# SC-01: GEMMA Parity Tests
# ---------------------------------------------------------------------------

_SYNTHETIC_MODE_REFS = [
    pytest.param(1, SYNTHETIC.ref("wald"), id="wald"),
    pytest.param(2, SYNTHETIC.ref("lrt"), id="lrt"),
    pytest.param(3, SYNTHETIC.ref("score"), id="score"),
    pytest.param(4, SYNTHETIC.ref("all"), id="all"),
]


@pytest.mark.tier1
class TestNumpyStreamingGemmaParity:
    """NumPy streaming runner produces GEMMA-equivalent results (SC-01)."""

    @pytest.mark.parametrize("lmm_mode,reference_path", _SYNTHETIC_MODE_REFS)
    def test_streaming_matches_gemma(self, synthetic_data, lmm_mode, reference_path):
        """Streaming runner matches GEMMA reference on synthetic data."""
        plink, kinship, phenotypes = synthetic_data
        run_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=kinship,
            config=LmmConfig(
                lmm_mode=lmm_mode, show_progress=False, check_memory=False
            ),
            chunk_size=200,  # Force multi-chunk (500 SNPs total)
        )
        results = run_result.associations
        assert len(results) > 0, "Expected results"
        assert run_result.n_tested == len(results)

        reference = load_gemma_assoc(reference_path)
        tolerances = ToleranceConfig(lambda_rtol=5e-5)
        comparison = compare_assoc_results(results, reference, tolerances)
        assert comparison.passed, (
            f"NumPy streaming mode {lmm_mode} (synthetic) vs GEMMA failed:\n"
            f"{comparison}"
        )


# ---------------------------------------------------------------------------
# SC-02: Batch Equivalence Tests
# ---------------------------------------------------------------------------


@pytest.mark.tier1
class TestBatchEquivalence:
    """NumPy streaming results match NumPy batch within BLAS tolerance (SC-02)."""

    def test_mode1_fp_identical(self, synthetic_eigen):
        """Wald results match between batch and streaming within BLAS tolerance."""
        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        # Batch run
        snp_info = SnpMeta.from_plink_meta(plink.meta)
        batch_result = run_lmm_association_numpy(
            genotypes=plink.genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
        )
        batch_assoc = batch_result.associations

        # Streaming run
        stream_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=200,
        )
        stream_assoc = stream_result.associations

        assert len(batch_assoc) == len(stream_assoc), (
            f"Count mismatch: {len(batch_assoc)} vs {len(stream_assoc)}"
        )

        for field in ("beta", "se", "p_wald"):
            batch_vals = np.array([getattr(r, field) for r in batch_assoc])
            stream_vals = np.array([getattr(r, field) for r in stream_assoc])
            np.testing.assert_allclose(
                batch_vals,
                stream_vals,
                atol=1e-14,
                rtol=_FP_PARITY_RTOL_MODE1,
                err_msg=f"{field} values differ",
            )

    @requires_c
    def test_mode4_fp_identical(self, synthetic_eigen):
        """Mode-4 results match between batch and streaming within BLAS tolerance."""
        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        snp_info = SnpMeta.from_plink_meta(plink.meta)
        batch_result = run_lmm_association_numpy(
            genotypes=plink.genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=4, show_progress=False, check_memory=False),
        )
        batch_assoc = batch_result.associations

        stream_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=4, show_progress=False, check_memory=False),
            chunk_size=200,
        )
        stream_assoc = stream_result.associations

        assert len(batch_assoc) == len(stream_assoc)

        for field in ("beta", "se", "p_wald", "p_score", "p_lrt"):
            batch_vals = np.array([getattr(r, field) for r in batch_assoc])
            stream_vals = np.array([getattr(r, field) for r in stream_assoc])
            np.testing.assert_allclose(
                batch_vals,
                stream_vals,
                atol=1e-14,
                # SoA-split dispatch accumulates Pab dot products in a
                # different order than pre-materialised full-Uab rows. In
                # optimized builds the two are bitwise-identical; under the
                # sanitizer build they drift up to ~2e-10 (see
                # _FP_PARITY_RTOL_MODE4).
                rtol=_FP_PARITY_RTOL_MODE4,
                err_msg=f"{field} values differ",
            )


@pytest.mark.tier1
class TestStreamingCovariates:
    """Streaming runner with covariates produces correct results (SC-02b)."""

    def test_streaming_covar_matches_batch_wald(self, synthetic_data_with_covariates):
        """Streaming + covariates matches batch + covariates (Wald mode)."""
        plink, phenotypes, covariates, eigenvalues, eigenvectors = (
            synthetic_data_with_covariates
        )

        # Batch run (pre-computed eigen for identical path)
        snp_info = SnpMeta.from_plink_meta(plink.meta)
        batch_result = run_lmm_association_numpy(
            genotypes=plink.genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
        )
        batch_assoc = batch_result.associations

        # Streaming run
        stream_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=200,
        )
        stream_assoc = stream_result.associations

        assert len(batch_assoc) == len(stream_assoc), (
            f"Count mismatch: {len(batch_assoc)} vs {len(stream_assoc)}"
        )

        for field in ("beta", "se", "p_wald"):
            batch_vals = np.array([getattr(r, field) for r in batch_assoc])
            stream_vals = np.array([getattr(r, field) for r in stream_assoc])
            np.testing.assert_allclose(
                batch_vals,
                stream_vals,
                atol=1e-14,
                rtol=1e-10,
                err_msg=f"{field} values differ (covariates)",
            )

    def test_streaming_covar_matches_gemma_wald(self, synthetic_data):
        """Streaming + covariates matches GEMMA reference (Wald mode)."""
        plink, kinship, phenotypes = synthetic_data
        covariates = np.loadtxt(SYNTHETIC.covariates)

        stream_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=kinship.copy(),
            covariates=covariates,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=200,
        )
        results = stream_result.associations
        assert len(results) > 0, "Expected results"
        assert stream_result.n_tested == len(results)

        reference = load_gemma_assoc(SYNTHETIC.ref("covar_wald"))
        tolerances = ToleranceConfig(lambda_rtol=5e-5)
        comparison = compare_assoc_results(results, reference, tolerances)
        assert comparison.passed, (
            f"Streaming covar mode 1 vs GEMMA failed:\n{comparison}"
        )

    def test_streaming_covar_multi_chunk(self, synthetic_data_with_covariates):
        """Chunk size independence with covariates."""
        plink, phenotypes, covariates, eigenvalues, eigenvectors = (
            synthetic_data_with_covariates
        )

        # Many small chunks
        small_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=50,
        )

        # Single chunk
        big_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=100_000,
        )

        assert small_result.n_tested == big_result.n_tested, (
            f"Count mismatch: {small_result.n_tested} vs {big_result.n_tested}"
        )

        small_p = np.array([r.p_wald for r in small_result.associations])
        big_p = np.array([r.p_wald for r in big_result.associations])
        np.testing.assert_allclose(
            small_p,
            big_p,
            atol=1e-14,
            rtol=1e-12,
            err_msg="p_wald differs across chunk sizes (covariates)",
        )


# ---------------------------------------------------------------------------
# SC-03: Streaming Mechanics Tests
# ---------------------------------------------------------------------------


@pytest.mark.tier1
class TestStreamingMechanics:
    """Disk-streaming write and in-memory accumulation (SC-03)."""

    def test_output_path_writes_to_disk(self, tmp_path, synthetic_data):
        """With output_path, results go to disk and associations is empty."""
        plink, kinship, phenotypes = synthetic_data
        out_file = tmp_path / "streaming_out.assoc.txt"

        result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=kinship,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=200,
            output_path=out_file,
        )

        # Result should have empty associations but n_tested populated
        assert result.associations == []
        assert result.n_tested > 0

        # Output file should exist with correct line count
        assert out_file.exists()
        lines = out_file.read_text().strip().split("\n")
        # First line is header, rest are data
        assert len(lines) == result.n_tested + 1

        # Load and verify count matches
        loaded = load_gemma_assoc(out_file)
        assert len(loaded) == result.n_tested

    def test_no_output_path_accumulates_in_memory(self, synthetic_data):
        """Without output_path, results accumulate in memory."""
        plink, kinship, phenotypes = synthetic_data

        result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=kinship,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=200,
        )

        assert len(result.associations) == result.n_tested
        assert result.n_tested > 0
        assert isinstance(result.associations[0], AssocResult)

    def test_result_carries_timing_breakdown(self, synthetic_data):
        """The returned result carries the run's timing breakdown."""
        plink, kinship, phenotypes = synthetic_data

        result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=kinship,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=200,
        )

        timing = result.timing
        assert timing.processed == result.n_tested
        assert timing.rotation_s >= 0
        assert timing.compute_s >= 0
        assert timing.result_write_s >= 0

    def test_pve_populated(self, synthetic_data):
        """PVE should be populated in the result."""
        plink, kinship, phenotypes = synthetic_data

        result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=kinship,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=200,
        )

        assert result.pve is not None
        assert 0 <= result.pve <= 1
        assert result.pve_se is not None, (
            "PVE SE should be populated for synthetic data"
        )
        assert result.pve_se > 0


# ---------------------------------------------------------------------------
# SC-04: Chunking Edge Case Tests
# ---------------------------------------------------------------------------


@pytest.mark.tier1
class TestChunkingEdgeCases:
    """Single-chunk and small-chunk edge cases produce identical results (SC-04)."""

    def test_single_chunk(self, synthetic_eigen):
        """chunk_size larger than total SNPs (single chunk) matches batch."""
        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        snp_info = SnpMeta.from_plink_meta(plink.meta)
        batch_result = run_lmm_association_numpy(
            genotypes=plink.genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
        )

        stream_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=100_000,  # Larger than total SNPs
        )

        batch_p = np.array([r.p_wald for r in batch_result.associations])
        stream_p = np.array([r.p_wald for r in stream_result.associations])
        np.testing.assert_allclose(batch_p, stream_p, atol=1e-14, rtol=1e-12)

    def test_small_chunks(self, synthetic_eigen):
        """chunk_size=50 (many small chunks) matches batch."""
        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        snp_info = SnpMeta.from_plink_meta(plink.meta)
        batch_result = run_lmm_association_numpy(
            genotypes=plink.genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
        )

        stream_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=50,  # Many small chunks
        )

        batch_p = np.array([r.p_wald for r in batch_result.associations])
        stream_p = np.array([r.p_wald for r in stream_result.associations])
        np.testing.assert_allclose(batch_p, stream_p, atol=1e-14, rtol=1e-12)

    def test_empty_after_filter(self, synthetic_data):
        """All SNPs filtered out returns empty result."""
        plink, kinship, phenotypes = synthetic_data

        # An empty -snps restriction leaves nothing to test. MAF cannot express
        # this: it is min(af, 1-af) and so never exceeds 0.5.
        result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=kinship,
            snps_indices=np.array([], dtype=np.int64),
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
        )

        assert result.associations == []
        assert result.n_tested == 0


# ---------------------------------------------------------------------------
# SC-05: Pipeline Machinery Tests
# ---------------------------------------------------------------------------


@pytest.mark.tier1
class TestStreamingPipeline:
    """Pipeline overlapping rotation/compute in NumPy streaming runner (SC-05)."""

    @requires_c
    def test_streaming_pipeline_activates(self, synthetic_eigen):
        """Pipeline activates with chunk_size=1 (forces >= 8 chunks)."""
        import io

        from loguru import logger as _logger

        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        # Capture DEBUG log output
        log_buffer = io.StringIO()
        sink_id = _logger.add(log_buffer, level="DEBUG", format="{message}")
        try:
            result = run_lmm_association_numpy_streaming(
                bed_path=SYNTHETIC.bfile,
                phenotypes=phenotypes,
                kinship=None,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
                chunk_size=1,
            )
        finally:
            _logger.remove(sink_id)

        log_text = log_buffer.getvalue()
        assert "Pipeline mode" in log_text, (
            f"Expected 'Pipeline mode' in log output, got: {log_text[:500]}"
        )
        assert result.n_tested > 0

    def test_streaming_pipeline_parity(self, synthetic_eigen):
        """Pipeline (chunk_size=1) matches sequential (chunk_size=100_000)."""
        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        # Sequential: single chunk (no pipeline)
        seq_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors.copy(),
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=100_000,
        )

        # Pipeline: many chunks
        pipe_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors.copy(),
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=1,
        )

        assert seq_result.n_tested == pipe_result.n_tested, (
            f"Count mismatch: {seq_result.n_tested} vs {pipe_result.n_tested}"
        )

        seq_assoc = seq_result.associations
        pipe_assoc = pipe_result.associations
        for field in ("beta", "se", "p_wald"):
            seq_vals = np.array([getattr(r, field) for r in seq_assoc])
            pipe_vals = np.array([getattr(r, field) for r in pipe_assoc])
            np.testing.assert_allclose(
                seq_vals,
                pipe_vals,
                atol=1e-14,
                rtol=1e-12,
                err_msg=f"{field} values differ between sequential and pipeline",
            )

    def test_streaming_auto_chunk_sizing(self, synthetic_eigen):
        """Auto chunk sizing runs without error (default chunk_size=10_000)."""
        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=10_000,
        )

        assert result.n_tested > 0

    def test_streaming_pipeline_output_path_parity(self, tmp_path, synthetic_eigen):
        """Pipeline with output_path produces same disk results as sequential."""
        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        # Sequential: single chunk -> disk
        seq_file = tmp_path / "seq.assoc.txt"
        seq_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors.copy(),
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=100_000,
            output_path=seq_file,
        )

        # Pipeline: many chunks -> disk
        pipe_file = tmp_path / "pipe.assoc.txt"
        pipe_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors.copy(),
            config=LmmConfig(lmm_mode=1, show_progress=False, check_memory=False),
            chunk_size=1,
            output_path=pipe_file,
        )

        assert seq_result.n_tested == pipe_result.n_tested
        assert seq_result.associations == []
        assert pipe_result.associations == []

        seq_loaded = load_gemma_assoc(seq_file)
        pipe_loaded = load_gemma_assoc(pipe_file)
        assert len(seq_loaded) == len(pipe_loaded) == seq_result.n_tested

        for field in ("beta", "se", "p_wald"):
            seq_vals = np.array([getattr(r, field) for r in seq_loaded])
            pipe_vals = np.array([getattr(r, field) for r in pipe_loaded])
            np.testing.assert_allclose(
                seq_vals,
                pipe_vals,
                atol=1e-14,
                rtol=1e-12,
                err_msg=f"{field} disk values differ between sequential and pipeline",
            )

    def test_streaming_pipeline_mode4_parity(self, synthetic_eigen):
        """Pipeline (chunk_size=1) with lmm_mode=4 matches sequential."""
        plink, _kinship, phenotypes, eigenvalues, eigenvectors = synthetic_eigen

        # Sequential: single chunk
        seq_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors.copy(),
            config=LmmConfig(lmm_mode=4, show_progress=False, check_memory=False),
            chunk_size=100_000,
        )

        # Pipeline: many chunks
        pipe_result = run_lmm_association_numpy_streaming(
            bed_path=SYNTHETIC.bfile,
            phenotypes=phenotypes,
            kinship=None,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors.copy(),
            config=LmmConfig(lmm_mode=4, show_progress=False, check_memory=False),
            chunk_size=1,
        )

        assert seq_result.n_tested == pipe_result.n_tested, (
            f"Count mismatch: {seq_result.n_tested} vs {pipe_result.n_tested}"
        )

        seq_assoc = seq_result.associations
        pipe_assoc = pipe_result.associations
        for field in ("beta", "se", "p_wald", "p_lrt", "p_score"):
            seq_vals = np.array([getattr(r, field) for r in seq_assoc])
            pipe_vals = np.array([getattr(r, field) for r in pipe_assoc])
            np.testing.assert_allclose(
                seq_vals,
                pipe_vals,
                atol=1e-14,
                rtol=1e-8,
                err_msg=f"{field} seq/pipeline mismatch (mode 4)",
            )
