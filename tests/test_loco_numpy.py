"""NumPy LOCO tests that run without JAX.

Kept in a separate file from test_loco.py because test_loco.py has
``pytest.importorskip("jax")`` at module level, which skips the entire
module when JAX is not installed. Tests here exercise the NumPy backend
only and must not import JAX.

Related LOCO test files:
- test_loco.py: Core LOCO tests (lmm_mode=1, cross-backend parity)
- test_gemma_loco_integration.py: GEMMA ref (mode 1),
  cross-backend parity (modes 2/3/4)
- test_loco_bugs.py: Regression tests for kinship aliasing, ordering, cleanup
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.io.plink import get_plink_metadata
from jamma.lmm.loco import run_lmm_loco
from tests.conftest import load_phenotypes_from_fam

# Fixture with 3 chromosomes — required for LOCO (needs >1 chromosome to leave one out)
_LOCO_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gemma_loco"
_LOCO_BFILE = _LOCO_FIXTURE_ROOT / "test"


@pytest.mark.tier0
class TestComputeLocoKinshipNumpy:
    """Tests for KIN-01: compute_loco_kinship works without JAX."""

    def test_compute_loco_kinship_no_jax_import(self):
        """compute_loco_kinship can be imported and called without JAX."""
        from jamma.kinship import compute_loco_kinship

        rng = np.random.default_rng(42)
        n_samples, n_snps = 20, 30
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
        chrs = np.array(["1"] * 10 + ["2"] * 10 + ["3"] * 10)

        results = list(
            compute_loco_kinship(genotypes, chrs, batch_size=15, check_memory=False)
        )

        assert len(results) == 3
        for _chr_name, K_loco in results:
            assert K_loco.shape == (n_samples, n_samples)
            np.testing.assert_allclose(K_loco, K_loco.T, atol=1e-14)

    def test_loco_subtraction_identity(self):
        """K_loco_c = (S_full - S_c) / (p - p_c) identity holds."""
        from jamma.kinship import compute_loco_kinship
        from jamma.kinship.missing import impute_and_center

        rng = np.random.default_rng(99)
        n_samples, n_snps = 15, 20
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
        chrs = np.array(["1"] * 8 + ["2"] * 12)

        # Compute reference: full kinship from centered genotypes
        X = genotypes.copy().astype(np.float64)
        X_centered = impute_and_center(X)
        S_full = X_centered @ X_centered.T

        results = dict(
            compute_loco_kinship(
                genotypes.copy(), chrs, batch_size=100, check_memory=False
            )
        )

        for chr_name in ["1", "2"]:
            chr_mask = chrs == chr_name
            X_chr = impute_and_center(genotypes[:, chr_mask].copy().astype(np.float64))
            S_chr = X_chr @ X_chr.T
            p_loco = int(np.sum(~chr_mask))
            K_loco_expected = (S_full - S_chr) / p_loco
            np.testing.assert_allclose(
                results[chr_name], K_loco_expected, rtol=1e-12, atol=1e-14
            )

    def test_loco_kinship_with_nan_genotypes(self):
        """LOCO kinship handles NaN (missing) genotypes correctly."""
        from jamma.kinship import compute_loco_kinship

        rng = np.random.default_rng(77)
        n_samples, n_snps = 15, 20
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
        # Inject missing values
        genotypes[0, 3] = np.nan
        genotypes[5, 10] = np.nan
        genotypes[12, 0] = np.nan
        chrs = np.array(["1"] * 10 + ["2"] * 10)

        results = list(
            compute_loco_kinship(
                genotypes.copy(), chrs, batch_size=100, check_memory=False
            )
        )

        assert len(results) == 2
        for _chr_name, K_loco in results:
            assert K_loco.shape == (n_samples, n_samples)
            assert K_loco.dtype == np.float64
            np.testing.assert_allclose(K_loco, K_loco.T, atol=1e-14)
            assert np.all(np.isfinite(K_loco))

    def test_loco_single_chromosome_raises(self):
        """LOCO with all SNPs on one chromosome raises ValueError."""
        from jamma.kinship import compute_loco_kinship

        rng = np.random.default_rng(42)
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(10, 20))
        chrs = np.array(["1"] * 20)  # all on chr 1

        with pytest.raises(ValueError, match="LOCO requires SNPs on multiple"):
            list(compute_loco_kinship(genotypes, chrs, check_memory=False))

    def test_loco_batch_size_invariant(self):
        """Different batch sizes produce identical LOCO kinship."""
        from jamma.kinship import compute_loco_kinship

        rng = np.random.default_rng(42)
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(15, 20))
        chrs = np.array(["1"] * 10 + ["2"] * 10)

        results_small = dict(
            compute_loco_kinship(
                genotypes.copy(), chrs, batch_size=3, check_memory=False
            )
        )
        results_large = dict(
            compute_loco_kinship(
                genotypes.copy(), chrs, batch_size=100, check_memory=False
            )
        )

        for chr_name in results_small:
            np.testing.assert_allclose(
                results_small[chr_name], results_large[chr_name], atol=1e-14
            )


@pytest.mark.tier1
def test_partitions_from_metadata_matches(sample_plink_data):
    """partitions_from_metadata produces identical output to get_chromosome_partitions.

    Verifies LOCO-04: derived partitions match direct BIM read without re-opening BED.
    """
    from jamma.io.plink import (
        get_chromosome_partitions,
        get_plink_metadata,
        partitions_from_metadata,
    )

    meta = get_plink_metadata(sample_plink_data)
    partitions_direct = get_chromosome_partitions(sample_plink_data)
    partitions_derived = partitions_from_metadata(meta)

    assert set(partitions_direct.keys()) == set(partitions_derived.keys())
    for chr_name in partitions_direct:
        np.testing.assert_array_equal(
            partitions_direct[chr_name], partitions_derived[chr_name]
        )


@pytest.mark.tier1
def test_apply_snp_list_mask_searchsorted():
    """apply_snp_list_mask searchsorted yields same result as boolean array approach."""
    from jamma.core.snp_filter import apply_snp_list_mask

    n_snps = 10000
    indices = np.sort(
        np.random.default_rng(42).choice(n_snps, size=500, replace=False)
    ).astype(np.intp)

    # Build expected result using the old boolean approach
    snp_mask_expected = np.ones(n_snps, dtype=bool)
    list_mask = np.zeros(n_snps, dtype=bool)
    list_mask[indices] = True
    snp_mask_expected &= list_mask

    # Build actual result using the new searchsorted approach
    snp_mask_actual = np.ones(n_snps, dtype=bool)
    apply_snp_list_mask(snp_mask_actual, indices, n_snps, "test")

    np.testing.assert_array_equal(snp_mask_actual, snp_mask_expected)


@pytest.mark.tier1
def test_get_loco_worker_count_env_var(monkeypatch):
    """JAMMA_LOCO_WORKERS env var is read and respected."""
    from jamma.core.threading import get_loco_worker_count

    # Default is 1
    monkeypatch.delenv("JAMMA_LOCO_WORKERS", raising=False)
    assert get_loco_worker_count() == 1

    # Env var sets count
    monkeypatch.setenv("JAMMA_LOCO_WORKERS", "4")
    assert get_loco_worker_count() == 4

    # Invalid env var falls back to 1
    monkeypatch.setenv("JAMMA_LOCO_WORKERS", "abc")
    assert get_loco_worker_count() == 1

    # Zero or negative clamps to 1
    monkeypatch.setenv("JAMMA_LOCO_WORKERS", "0")
    assert get_loco_worker_count() == 1

    monkeypatch.setenv("JAMMA_LOCO_WORKERS", "-2")
    assert get_loco_worker_count() == 1


@pytest.mark.tier1
def test_loco_numpy_no_per_chromosome_bed_reads():
    """NumPy LOCO stats cache eliminates per-chromosome BED re-reads.

    Verifies LOCO-01: without the cache, each chromosome needs an extra open_bed
    call for _collect_chr_snp_stats. With the cache, those calls are skipped.

    Counts: 2 metadata reads (run_lmm_loco + streaming_numpy) + 2 kinship reads
    (PASS 1 stats + PASS 2 accumulation) + n_chr assoc reads (genotypes per chr).
    With cache = 4 + n_chr. Without cache = 4 + 2*n_chr (extra stats reads).
    """
    if not _LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    from unittest.mock import patch

    import bed_reader

    call_count = 0
    original_open_bed = bed_reader.open_bed

    def counting_open_bed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_open_bed(*args, **kwargs)

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    # Patch at the plink.py import site (stream_genotype_chunks uses it for kinship
    # PASS 1 and PASS 2, and get_plink_metadata for metadata reads)
    # and at the loco.py import site (_run_lmm_for_chromosome_numpy uses it directly)
    with (
        patch("jamma.io.plink.open_bed", side_effect=counting_open_bed),
        patch("jamma.lmm.loco.open_bed", side_effect=counting_open_bed),
    ):
        results, n_tested, _ = run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            backend="numpy",
            check_memory=False,
            show_progress=False,
        )

    meta = get_plink_metadata(_LOCO_BFILE)
    n_chromosomes = len(set(meta["chromosome"].tolist()))

    # Expected: 2 metadata reads + 2 kinship reads + n_chr assoc reads
    # (no per-chromosome stats reads — eliminated by SnpStatsCache)
    # Without the cache this would be 4 + 2*n_chr
    expected_with_cache = 4 + n_chromosomes
    expected_without_cache = 4 + 2 * n_chromosomes
    assert call_count <= expected_with_cache, (
        f"Expected at most {expected_with_cache} BED opens "
        f"(2 metadata + 2 kinship + {n_chromosomes} assoc genotypes), "
        f"got {call_count}. "
        f"Without cache would be {expected_without_cache}. "
        f"Per-chromosome stats BED reads not fully eliminated."
    )
    assert n_tested > 0, "Expected SNPs to be tested"


@pytest.mark.tier1
def test_run_lmm_loco_reads_loco_workers_env(monkeypatch):
    """run_lmm_loco reads JAMMA_LOCO_WORKERS and logs a WARNING when workers > 1.

    Verifies LOCO-08 wiring: the env var is read at run_lmm_loco entry and
    a WARNING-level message is emitted when workers > 1 (not yet implemented).
    """
    if not _LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    from unittest.mock import patch

    monkeypatch.setenv("JAMMA_LOCO_WORKERS", "4")
    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    logged_warnings: list[str] = []

    # loguru does not integrate with pytest caplog; capture via the logger sink
    import jamma.lmm.loco as loco_module

    original_warning = loco_module.logger.warning

    def capture_warning(msg, *args, **kwargs):
        logged_warnings.append(str(msg))
        return original_warning(msg, *args, **kwargs)

    with patch.object(loco_module.logger, "warning", side_effect=capture_warning):
        results, n_tested, _ = run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            backend="numpy",
            check_memory=False,
            show_progress=False,
        )

    assert any("JAMMA_LOCO_WORKERS=4" in msg for msg in logged_warnings), (
        f"Expected 'JAMMA_LOCO_WORKERS=4' in warning messages, got: {logged_warnings}"
    )
    assert n_tested > 0, "Expected SNPs tested (workers > 1 falls back to sequential)"


@pytest.mark.tier1
def test_loco_numpy_multipass_equivalence():
    """Multi-pass and single-pass LOCO produce identical association results.

    Forces multi-pass mode via _max_batch_chrs=1 (one chromosome per disk pass)
    and verifies that all association statistics match the single-pass baseline.
    This tests LOCO-02: batch_size_chrs sizing and S_full accumulation correctness.
    """
    if not _LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    from jamma.lmm.loco import _compute_loco_kinship_streaming_numpy

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    # Single-pass baseline (default behaviour, all chromosomes fit in memory)
    results_single, n_single, _ = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        backend="numpy",
        check_memory=False,
        show_progress=False,
    )

    # Multi-pass: force batch_size_chrs=1 via debug override (_max_batch_chrs).
    # The fixture has 3 chromosomes, so this triggers 3 disk passes.
    # We patch _compute_loco_kinship_streaming_numpy to inject _max_batch_chrs=1.
    original_fn = _compute_loco_kinship_streaming_numpy

    def patched_fn(*args, **kwargs):
        kwargs["_max_batch_chrs"] = 1
        return original_fn(*args, **kwargs)

    from unittest.mock import patch

    import jamma.lmm.loco as loco_module

    with patch.object(
        loco_module,
        "_compute_loco_kinship_streaming_numpy",
        side_effect=patched_fn,
    ):
        results_multi, n_multi, _ = run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            backend="numpy",
            check_memory=False,
            show_progress=False,
        )

    assert n_single == n_multi, f"n_tested mismatch: {n_single} vs {n_multi}"
    assert len(results_single) == len(results_multi), (
        f"result count mismatch: {len(results_single)} vs {len(results_multi)}"
    )

    for r_single, r_multi in zip(results_single, results_multi, strict=True):
        assert r_single.rs == r_multi.rs, (
            f"SNP order differs: {r_single.rs} vs {r_multi.rs}"
        )
        np.testing.assert_allclose(r_single.beta, r_multi.beta, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(r_single.se, r_multi.se, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(
            r_single.p_wald, r_multi.p_wald, rtol=1e-10, atol=1e-14
        )


@pytest.mark.tier1
def test_loco_numpy_valid_sample_subsetting():
    """K_loco is computed at valid-sample size when valid_indices is provided.

    Verifies LOCO-07: _compute_loco_kinship_streaming_numpy returns n_valid x n_valid
    kinship matrices when valid_indices is provided, rather than n_samples x n_samples.
    """
    if not _LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    from jamma.lmm.loco import _compute_loco_kinship_streaming_numpy

    meta = get_plink_metadata(_LOCO_BFILE)
    n_samples = meta["n_samples"]

    # Exclude last 5 samples
    valid_indices = np.arange(0, n_samples - 5)
    n_valid = len(valid_indices)

    loco_iter, cache = _compute_loco_kinship_streaming_numpy(
        _LOCO_BFILE,
        check_memory=False,
        show_progress=False,
        valid_indices=valid_indices,
    )

    for chr_name, K_loco in loco_iter:
        assert K_loco.shape == (n_valid, n_valid), (
            f"K_loco for chr {chr_name} has shape {K_loco.shape}, "
            f"expected ({n_valid}, {n_valid})"
        )
        # Verify symmetry
        np.testing.assert_allclose(K_loco, K_loco.T, atol=1e-14)
        assert np.all(np.isfinite(K_loco)), (
            f"K_loco for chr {chr_name} has non-finite values"
        )


@pytest.mark.tier1
def test_loco_numpy_show_progress_true():
    """NumPy LOCO with show_progress=True completes without error.

    Exercises the tqdm progress bars and logger.info calls in
    _compute_loco_kinship_streaming_numpy and run_lmm_loco.
    Not marked @requires_jax — runs in NumPy-only CI.
    """
    if not _LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    results, n_tested, _ = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        lmm_mode=1,
        show_progress=True,
        check_memory=False,
        backend="numpy",
    )

    assert n_tested > 0, "Expected at least one SNP to be tested"
    assert len(results) > 0, "Expected at least one association result"
