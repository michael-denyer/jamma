"""NumPy LOCO tests.

Related LOCO test files:
- test_loco_orchestration.py: Kinship aliasing, biological ordering, cleanup
- test_loco_eigen_cache.py: LOCO eigen cache write/read round-trip
- legacy/tests/test_loco.py: Archived cross-backend parity tests
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.io.plink import get_plink_metadata
from jamma.lmm.loco import run_lmm_loco
from jamma.lmm.schema import LmmConfig
from jamma.validation.compare import compare_assoc_results, load_gemma_assoc
from jamma.validation.tolerances import ToleranceConfig
from tests.conftest import load_phenotypes_from_fam, require_fixture

# Fixture with 3 chromosomes — required for LOCO (needs >1 chromosome to leave one out)
_LOCO_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gemma_loco"
_LOCO_BFILE = _LOCO_FIXTURE_ROOT / "test"


@pytest.mark.tier1
def test_global_index_restriction_matches_boolean_mask():
    """The searchsorted intersection equals the boolean-mask approach.

    filter_snp_stats routes every -snps restriction through
    _apply_global_index_restriction; this pins its result against the
    O(n_snps)-memory boolean formulation it replaced.
    """
    from jamma.core.snp_stats import _apply_global_index_restriction

    n_snps = 10000
    indices = np.sort(
        np.random.default_rng(42).choice(n_snps, size=500, replace=False)
    ).astype(np.intp)

    # Expected result using the boolean approach over the full space
    snp_mask_expected = np.ones(n_snps, dtype=bool)
    list_mask = np.zeros(n_snps, dtype=bool)
    list_mask[indices] = True
    snp_mask_expected &= list_mask

    snp_mask_actual = np.ones(n_snps, dtype=bool)
    _apply_global_index_restriction(
        snp_mask_actual, np.arange(n_snps, dtype=np.intp), indices, "test"
    )

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
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

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
        loco = run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            config=LmmConfig(check_memory=False, show_progress=False),
        )

    meta = get_plink_metadata(_LOCO_BFILE)
    n_chromosomes = len(set(meta.chromosome.tolist()))

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
    assert loco.n_tested > 0, "Expected SNPs to be tested"


@pytest.mark.tier1
def test_run_lmm_loco_forwards_grid_params(monkeypatch):
    """run_lmm_loco forwards n_grid/n_refine all the way to the chunk optimizer.

    Regression: run_lmm_loco had no n_grid/n_refine parameters, so the pipeline
    could not configure LOCO's lambda grid — every LOCO run silently used the
    hard-coded defaults regardless of PipelineConfig.n_grid/n_refine.

    Asserting only n_tested > 0 would pass even if the values were dropped (a
    no-op body still tests SNPs). Instead, spy on run_lmm_chunk_source_numpy —
    the boundary where n_grid/n_refine are consumed, captured pre-clamp — and
    assert the configured non-default values actually arrive there. LOCO
    reaches that boundary through the shared run body in runner_numpy.
    """
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

    import jamma.lmm.runner_numpy as runner_mod

    real_chunk_runner = runner_mod.run_lmm_chunk_source_numpy
    captured: list[dict[str, int]] = []

    def spy(*args, **kwargs):
        captured.append({"n_grid": kwargs["n_grid"], "n_refine": kwargs["n_refine"]})
        return real_chunk_runner(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "run_lmm_chunk_source_numpy", spy)

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))
    loco = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        config=LmmConfig(
            n_grid=7, n_refine=25, check_memory=False, show_progress=False
        ),
    )
    assert loco.n_tested > 0, "Expected SNPs to be tested with an explicit grid"
    assert captured, "chunk runner was never called — cannot verify forwarding"
    # Every per-chromosome chunk call must receive the configured (non-default)
    # grid params, not the hard-coded 50/10 defaults the bug produced.
    assert all(c["n_grid"] == 7 for c in captured)
    assert all(c["n_refine"] == 25 for c in captured)


@pytest.mark.tier1
def test_run_lmm_loco_reads_loco_workers_env(monkeypatch):
    """run_lmm_loco reads JAMMA_LOCO_WORKERS and logs a WARNING when workers > 1.

    Verifies LOCO-08 wiring: the env var is read at run_lmm_loco entry and
    a WARNING-level message is emitted when workers > 1 (not yet implemented).
    """
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

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
        loco = run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            config=LmmConfig(check_memory=False, show_progress=False),
        )

    assert any("JAMMA_LOCO_WORKERS=4" in msg for msg in logged_warnings), (
        f"Expected 'JAMMA_LOCO_WORKERS=4' in warning messages, got: {logged_warnings}"
    )
    assert loco.n_tested > 0, (
        "Expected SNPs tested (workers > 1 falls back to sequential)"
    )


@pytest.mark.tier1
def test_loco_numpy_multipass_equivalence():
    """Multi-pass and single-pass LOCO produce identical association results.

    Forces multi-pass mode via _max_batch_chrs=1 (one chromosome per disk pass)
    and verifies that all association statistics match the single-pass baseline.
    This tests LOCO-02: batch_size_chrs sizing and S_full accumulation correctness.
    """
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

    from jamma.kinship import compute_loco_kinship_streaming

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    # Single-pass baseline (default behaviour, all chromosomes fit in memory)
    loco_single = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        config=LmmConfig(check_memory=False, show_progress=False),
    )

    # Multi-pass: force batch_size_chrs=1 via debug override (_max_batch_chrs).
    # The fixture has 3 chromosomes, so this triggers 3 disk passes.
    # We patch compute_loco_kinship_streaming (as imported into loco) to inject
    # _max_batch_chrs=1.
    original_fn = compute_loco_kinship_streaming

    def patched_fn(*args, **kwargs):
        kwargs["_max_batch_chrs"] = 1
        return original_fn(*args, **kwargs)

    from unittest.mock import patch

    import jamma.lmm.loco as loco_module

    with patch.object(
        loco_module,
        "compute_loco_kinship_streaming",
        side_effect=patched_fn,
    ):
        loco_multi = run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            config=LmmConfig(check_memory=False, show_progress=False),
        )
    results_single = loco_single.associations
    results_multi = loco_multi.associations

    assert loco_single.n_tested == loco_multi.n_tested, (
        f"n_tested mismatch: {loco_single.n_tested} vs {loco_multi.n_tested}"
    )
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
def test_loco_numpy_covariates_threaded_and_effective():
    """LOCO threads a covariate matrix (n_cvt=2) through per-chromosome UtW.

    No GEMMA covariate-LOCO reference exists, so this is a wiring/behavioural
    check rather than exact parity: adding a real covariate column must
    (a) leave the tested SNP set unchanged (covariates don't affect
    genotype-based filtering), (b) yield finite beta/se/p for every SNP, and
    (c) change the fit versus the intercept-only run — a covariate that was
    silently dropped or wrongly subset in LOCO's per-chromosome covariate handling
    would otherwise leave results identical.
    """
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))
    n = phenotypes.shape[0]
    rng = np.random.default_rng(0)
    # First column is the intercept (required); second is a real covariate.
    covariates = np.column_stack([np.ones(n), rng.standard_normal(n)])

    baseline = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        config=LmmConfig(check_memory=False, show_progress=False),
    )
    with_covar = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        covariates=covariates,
        config=LmmConfig(check_memory=False, show_progress=False),
    )

    assert with_covar.n_tested == baseline.n_tested > 0
    assert len(with_covar.associations) == len(baseline.associations)

    any_beta_differs = False
    for base_r, cov_r in zip(
        baseline.associations, with_covar.associations, strict=True
    ):
        assert base_r.rs == cov_r.rs  # same SNP set and order
        assert np.isfinite(cov_r.beta)
        assert np.isfinite(cov_r.se)
        assert cov_r.se > 0
        assert np.isfinite(cov_r.p_wald)
        if abs(cov_r.beta - base_r.beta) > 1e-8:
            any_beta_differs = True

    assert any_beta_differs, (
        "covariate had no effect on any SNP — it was likely dropped or "
        "wrongly wired in LOCO's per-chromosome covariate handling"
    )


def _all_sample_cache(bed_path):
    """Build an all-sample SnpStatsCache over every SNP, like kinship PASS 1."""
    from bed_reader import open_bed

    from jamma.core.snp_stats import SnpStatsCache, collect_snp_stats_from_chunks

    meta = get_plink_metadata(bed_path)
    n_total, n_snps = meta.n_samples, meta.n_snps
    with open_bed(Path(f"{bed_path}.bed")) as bed:
        geno_all = bed.read(index=np.s_[:, :], dtype=np.float64)
    stats = collect_snp_stats_from_chunks(
        [(geno_all, 0, n_snps)],
        n_snps=n_snps,
        n_samples=n_total,
        global_indices=np.arange(n_snps),
        sample_scope="all_samples",
    )
    return SnpStatsCache(
        col_means=stats.col_means,
        miss_counts=stats.miss_counts,
        col_vars=stats.col_vars,
        n_samples=n_total,
        n_unexpected=stats.n_unexpected,
        hwe_counts=stats.hwe_counts,
        global_indices=stats.global_indices,
        sample_scope="all_samples",
    )


@pytest.mark.tier1
def test_chr_snp_stats_for_loco_bypasses_all_sample_cache_when_analyzed_subset():
    """With a missing phenotype (analyzed != all), LOCO must recompute SNP stats
    over the analyzed samples and NOT reuse the all-sample kinship-pass cache.

    GEMMA computes each SNP's mean/MAF and imputes over analyzed individuals only,
    so the all-sample cache is the wrong basis here. Guards both approaches: the
    cached approach must fall through to the valid-sample recompute, and the
    non-cache approach must already use it.
    """
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

    from jamma.lmm.loco import _chr_snp_stats_for_loco, _collect_chr_snp_stats

    meta = get_plink_metadata(_LOCO_BFILE)
    n_total = meta.n_samples
    chrs = np.asarray(meta.chromosome)
    chr1 = np.where(chrs == chrs[0])[0]  # first chromosome's global SNP indices

    cache = _all_sample_cache(_LOCO_BFILE)

    # Drop 30 samples -> analyzed != all.
    valid_mask = np.ones(n_total, dtype=bool)
    valid_mask[:30] = False
    valid_indices = np.where(valid_mask)[0]

    valid_stats = _collect_chr_snp_stats(_LOCO_BFILE, chr1, valid_indices, 5000)
    all_sample_chr1 = cache.take(chr1)

    # Cached approach, analyzed != all: must return the valid-sample recompute...
    got = _chr_snp_stats_for_loco(
        cache,
        _LOCO_BFILE,
        chr1,
        valid_indices,
        all_samples_valid=False,
        col_chunk_size=5000,
    )
    np.testing.assert_allclose(got.col_means, valid_stats.col_means)
    # ...and that must differ from the all-sample cache (the bypass matters).
    assert np.max(np.abs(got.col_means - all_sample_chr1.col_means)) > 1e-6

    # Cached approach, analyzed == all: reuse the cache (free, exact match).
    got_all = _chr_snp_stats_for_loco(
        cache,
        _LOCO_BFILE,
        chr1,
        np.arange(n_total),
        all_samples_valid=True,
        col_chunk_size=5000,
    )
    np.testing.assert_allclose(got_all.col_means, all_sample_chr1.col_means)

    # Non-cache approach always uses the analyzed samples.
    got_none = _chr_snp_stats_for_loco(
        None,
        _LOCO_BFILE,
        chr1,
        valid_indices,
        all_samples_valid=False,
        col_chunk_size=5000,
    )
    np.testing.assert_allclose(got_none.col_means, valid_stats.col_means)


@pytest.mark.tier1
def test_loco_missing_phenotype_cache_and_noncache_agree():
    """With missing phenotypes, the cached and non-cache LOCO paths must produce
    identical association results — both on the analyzed-sample basis (GEMMA's).

    Before the cache was gated to analyzed==all, the cache path used all-sample
    stats and diverged from the non-cache path; this pins them together so the
    divergence cannot return in either approach.
    """
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

    from unittest.mock import patch

    import jamma.lmm.loco as loco_module
    from jamma.kinship import compute_loco_kinship_streaming

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))
    pheno = phenotypes.copy()
    pheno[::9] = np.nan  # drop ~12 samples -> analyzed != all

    loco_cache = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=pheno,
        config=LmmConfig(check_memory=False, show_progress=False),
    )

    original = compute_loco_kinship_streaming

    def _null_cache(*args, **kwargs):
        # Force the non-cache path: keep the kinship iterator, drop the cache.
        result = original(*args, **kwargs)
        if isinstance(result, tuple):
            loco_iter, _cache = result
            return loco_iter, None
        return result

    with patch.object(
        loco_module, "compute_loco_kinship_streaming", side_effect=_null_cache
    ):
        loco_nocache = run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=pheno,
            config=LmmConfig(check_memory=False, show_progress=False),
        )

    assert loco_cache.n_tested == loco_nocache.n_tested > 0
    for r_cache, r_nocache in zip(
        loco_cache.associations, loco_nocache.associations, strict=True
    ):
        assert r_cache.rs == r_nocache.rs
        # af is the direct fingerprint of the stats basis (all-sample vs analysed);
        # on a fixture with no missing genotypes it is where the bug shows.
        np.testing.assert_allclose(r_cache.af, r_nocache.af, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(r_cache.beta, r_nocache.beta, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(r_cache.se, r_nocache.se, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            r_cache.p_wald, r_nocache.p_wald, rtol=1e-9, atol=1e-12
        )


@pytest.mark.tier1
def test_loco_numpy_valid_sample_subsetting():
    """K_loco is computed at valid-sample size when valid_indices is provided.

    Verifies LOCO-07: compute_loco_kinship_streaming returns n_valid x n_valid
    kinship matrices when valid_indices is provided, rather than n_samples x n_samples.
    """
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

    from jamma.kinship import compute_loco_kinship_streaming

    meta = get_plink_metadata(_LOCO_BFILE)
    n_samples = meta.n_samples

    # Exclude last 5 samples
    valid_indices = np.arange(0, n_samples - 5)
    n_valid = len(valid_indices)

    loco_iter, cache = compute_loco_kinship_streaming(
        _LOCO_BFILE,
        check_memory=False,
        show_progress=False,
        valid_indices=valid_indices,
        return_snp_stats=True,
    )

    # The all-sample SNP-stats cache is not exported on a filtered-sample run:
    # PASS-1 stats are on the valid-sample basis (matching GEMMA and PASS-2), so
    # there is no all-sample cache to reuse, and the association pass re-derives.
    assert cache is None

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


@pytest.mark.tier0
def test_decide_loco_passes_reserves_eigendecomp_at_valid_size():
    """Multi-pass batch sizing reserves eigendecomp memory at n_mat, not n_samples.

    Regression: the multi-pass branch sized its eigendecomp workspace reserve
    with the full n_samples instead of n_mat (the valid-sample matrix size).
    On datasets with invalid samples that over-reservation shrinks usable RAM
    and can collapse batch_size to 1, forcing many redundant BED passes even
    though the live K_loco matrices are only n_valid x n_valid.

    Pure sizing math, so we drive it at realistic scale (no genotype data) where
    the n_mat-vs-n_samples reserve difference is material.
    """
    from jamma.core.eigen_plan import dsyevr_peak_gb
    from jamma.kinship.compute import _decide_loco_passes

    n_samples = 100_000
    n_mat = 70_000  # 30k samples filtered out
    n_chr = 22
    chunk_size = 10_000
    available_gb = 300.0  # forces multi-pass (single-pass needs ~950GB)

    plan = _decide_loco_passes(
        n_mat, n_samples, n_chr, chunk_size, available_gb, max_batch_chrs=None
    )

    assert not plan.single_pass, "scenario must exercise the multi-pass branch"

    # Re-derive both candidate batch sizes from the same public peak estimator.
    matrix_gb = n_mat**2 * 8 / 1e9
    chunk_buffer_gb = n_samples * chunk_size * 8 / 1e9
    budget = available_gb * 0.9 - 2 * matrix_gb - chunk_buffer_gb

    fixed_batch = max(1, int((budget - dsyevr_peak_gb(n_mat)) / matrix_gb))
    buggy_batch = max(1, int((budget - dsyevr_peak_gb(n_samples)) / matrix_gb))

    # The scenario must genuinely distinguish the two reserves, and the fix must
    # pick the (larger) n_mat-based batch size rather than collapsing to 1.
    assert buggy_batch < fixed_batch, "test scenario does not exercise the bug"
    assert buggy_batch == 1, "buggy n_samples reserve should collapse to batch_size=1"
    assert plan.batch_size == fixed_batch
    assert plan.batch_size > 1


@pytest.mark.tier0
def test_decide_loco_passes_unfiltered_matches_full_size():
    """With no sample filtering (n_mat == n_samples) the reserve fix is a no-op."""
    from jamma.kinship.compute import _decide_loco_passes

    plan = _decide_loco_passes(100_000, 100_000, 22, 10_000, 300.0, max_batch_chrs=None)
    assert not plan.single_pass
    assert plan.batch_size >= 1


@pytest.mark.tier1
def test_loco_numpy_show_progress_true():
    """NumPy LOCO with show_progress=True completes without error.

    Exercises the tqdm progress bars and logger.info calls in
    compute_loco_kinship_streaming and run_lmm_loco.
    Runs in NumPy-only CI.
    """
    require_fixture(_LOCO_BFILE.with_suffix(".bed"), _LOCO_BFILE.with_suffix(".fam"))

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    loco = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        config=LmmConfig(lmm_mode=1, show_progress=True, check_memory=False),
    )

    assert loco.n_tested > 0, "Expected at least one SNP to be tested"
    assert len(loco.associations) > 0, "Expected at least one association result"


@pytest.mark.tier2
def test_loco_gemma_equivalence():
    """LOCO per-chromosome results match GEMMA reference within calibrated tolerances.

    TEST-02: End-to-end integration test. Runs run_lmm_loco on the gemma_loco fixture
    (100 samples, 500 SNPs, 3 chromosomes) and compares per-chromosome association
    results against GEMMA reference files (gemma_loco_chr{1,2,3}.assoc.txt).

    Validates all numeric columns (beta, se, p_wald, logl_H1, l_remle, af)
    per SNP using calibrated tolerances from ToleranceConfig via
    compare_assoc_results().
    """
    require_fixture(
        _LOCO_BFILE.with_suffix(".bed"),
        _LOCO_BFILE.with_suffix(".fam"),
        *(_LOCO_FIXTURE_ROOT / f"gemma_loco_chr{c}.assoc.txt" for c in ("1", "2", "3")),
    )

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))
    # LOCO recomputes kinship per chromosome, amplifying Brent optimizer
    # divergence on lambda. Use 5e-5 (calibrated JAMMA-vs-GEMMA bound,
    # see GEMMA_EQUIVALENCE.md).
    tol = ToleranceConfig(lambda_rtol=5e-5)

    loco = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        config=LmmConfig(check_memory=False, show_progress=False),
    )

    assert loco.n_tested > 0, "Expected SNPs to be tested"

    # Group JAMMA results by chromosome
    by_chr: dict[str, list] = {}
    for r in loco.associations:
        by_chr.setdefault(r.chr, []).append(r)

    for chr_name in ["1", "2", "3"]:
        gemma_ref = load_gemma_assoc(
            _LOCO_FIXTURE_ROOT / f"gemma_loco_chr{chr_name}.assoc.txt"
        )
        jamma_chr = by_chr.get(chr_name, [])
        result = compare_assoc_results(jamma_chr, gemma_ref, config=tol)
        assert result.passed, (
            f"Chr {chr_name} GEMMA equivalence failed:\n"
            f"  beta: {result.beta.message}\n"
            f"  se: {result.se.message}\n"
            f"  p_wald: {result.p_wald.message}\n"
            f"  logl_H1: {result.logl_H1.message}\n"
            f"  l_remle: {result.l_remle.message}\n"
            f"  mismatched_snps: {result.mismatched_snps}"
        )
