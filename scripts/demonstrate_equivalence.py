"""Numerical equivalence and performance report: JAMMA vs GEMMA.

Runs JAMMA's JAX runner on gemma_synthetic reference data across all LMM modes
and produces per-field max difference tables, scientific equivalence metrics,
and per-section performance timing.

Usage:
    uv run python scripts/demonstrate_equivalence.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jamma.core import configure_jax  # noqa: E402
from jamma.io import load_plink_binary  # noqa: E402
from jamma.kinship import compute_centered_kinship  # noqa: E402
from jamma.kinship.io import read_kinship_matrix  # noqa: E402
from jamma.lmm.runner_jax import run_lmm_association_jax  # noqa: E402
from jamma.validation import (  # noqa: E402
    load_gemma_assoc,
    load_gemma_kinship,
)

# Configure JAX for float64
configure_jax(enable_x64=True)

# --- Paths ---
PLINK_PREFIX = ROOT / "tests/fixtures/gemma_synthetic/test"
GEMMA_KINSHIP = ROOT / "tests/fixtures/gemma_synthetic/gemma_kinship.cXX.txt"
GEMMA_WALD = ROOT / "tests/fixtures/gemma_synthetic/gemma_assoc.assoc.txt"
GEMMA_SCORE = ROOT / "tests/fixtures/gemma_score/gemma_score.assoc.txt"
GEMMA_LRT = ROOT / "tests/fixtures/gemma_synthetic/gemma_lrt.assoc.txt"
GEMMA_ALL = ROOT / "tests/fixtures/gemma_all_tests/gemma_all.assoc.txt"
GEMMA_COVAR = ROOT / "tests/fixtures/gemma_covariate/gemma_covariate.assoc.txt"
GEMMA_ALL_COVAR = ROOT / "tests/fixtures/gemma_all_tests/gemma_all_covar.assoc.txt"
COVARIATE_FILE = ROOT / "tests/fixtures/gemma_covariate/covariates.txt"

# --- Validated tolerances (calibrated from empirical GEMMA comparison) ---
TOLERANCES = {
    "kinship": 1e-8,  # observed: 4.66e-10
    "beta": 1e-2,  # observed: 7e-5 (large values from Pab projection sensitivity)
    "se": 1e-5,  # observed: ~2e-6
    "p_wald": 1e-4,  # observed: 2.2e-6
    "p_score": 1e-4,  # observed: 4.1e-7
    "p_lrt": 5e-3,  # observed: 1.6e-3 (chi2 CDF magnifies logl diffs)
    "logl": 1e-6,  # observed: 3.2e-7
    "lambda": 5e-5,  # observed: 3.8e-5
}

# Common JAX runner kwargs (suppress progress bars in report)
JAX_KWARGS = dict(n_grid=50, n_refine=20, show_progress=False, check_memory=False)


@dataclass
class FieldResult:
    """Per-field comparison result."""

    field: str
    n_values: int
    max_abs_diff: float
    max_rel_diff: float
    tolerance: float
    passed: bool


@dataclass
class ScientificEquivalence:
    """Scientific equivalence metrics."""

    n_snps: int
    spearman_rho: float
    sig_05: tuple[int, int]  # (agree, total)
    sig_01: tuple[int, int]
    sig_001: tuple[int, int]
    sig_5e8: tuple[int, int]
    effect_direction_agree: float  # fraction


@dataclass
class SectionTiming:
    """Per-section timing result."""

    name: str
    elapsed: float
    n_snps: int


def _compute_field_diffs(
    actual: np.ndarray, expected: np.ndarray, field: str, tolerance: float
) -> FieldResult:
    """Compute per-field max diffs."""
    mask = np.isfinite(actual) & np.isfinite(expected)
    a, e = actual[mask], expected[mask]
    if len(a) == 0:
        return FieldResult(field, 0, 0.0, 0.0, tolerance, True)

    abs_diff = np.abs(a - e)
    max_abs = float(np.max(abs_diff))

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / np.abs(e)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
    max_rel = float(np.max(rel_diff))

    return FieldResult(field, len(a), max_abs, max_rel, tolerance, max_rel <= tolerance)


def _scientific_equivalence(
    jamma: list, gemma: list, p_field: str
) -> ScientificEquivalence:
    """Compute scientific equivalence metrics."""
    j_by_rs = {r.rs: r for r in jamma}
    g_by_rs = {r.rs: r for r in gemma}
    common = sorted(set(j_by_rs) & set(g_by_rs))

    j_p = np.array([getattr(j_by_rs[rs], p_field) for rs in common])
    g_p = np.array([getattr(g_by_rs[rs], p_field) for rs in common])

    mask = np.isfinite(j_p) & np.isfinite(g_p) & (j_p > 0) & (g_p > 0)
    j_p, g_p = j_p[mask], g_p[mask]

    rho, _ = spearmanr(-np.log10(j_p), -np.log10(g_p))

    def _sig_agree(thresh):
        agree = int(np.sum((j_p < thresh) == (g_p < thresh)))
        return (agree, len(j_p))

    j_beta = np.array([j_by_rs[rs].beta for rs in common])
    g_beta = np.array([g_by_rs[rs].beta for rs in common])
    beta_mask = np.isfinite(j_beta) & np.isfinite(g_beta) & (np.abs(g_beta) > 1e-10)
    if np.sum(beta_mask) > 0:
        dir_agree = float(
            np.mean(np.sign(j_beta[beta_mask]) == np.sign(g_beta[beta_mask]))
        )
    else:
        dir_agree = 1.0

    return ScientificEquivalence(
        n_snps=len(j_p),
        spearman_rho=rho,
        sig_05=_sig_agree(0.05),
        sig_01=_sig_agree(0.01),
        sig_001=_sig_agree(0.001),
        sig_5e8=_sig_agree(5e-8),
        effect_direction_agree=dir_agree,
    )


def _build_snp_info(plink_data):
    """Build SNP info from PlinkData."""
    snp_info = []
    for i in range(plink_data.n_snps):
        snp_info.append(
            {
                "chr": str(plink_data.chromosome[i]),
                "rs": plink_data.sid[i],
                "pos": plink_data.bp_position[i],
                "a1": plink_data.allele_1[i],
                "a0": plink_data.allele_2[i],
                "maf": 0.0,
                "n_miss": 0,
            }
        )
    return snp_info


def _fmt_sci(val: float) -> str:
    """Format scientific notation."""
    if val == 0.0:
        return "0"
    return f"{val:.2e}"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_field_table(fields: list[FieldResult]):
    """Print field comparison table."""
    print(
        f"  {'Field':<14} {'N':>6}  {'Max Abs Diff':>14}  "
        f"{'Max Rel Diff':>14}  {'Tolerance':>12}  {'Result':>6}"
    )
    print(f"  {'-'*14} {'-'*6}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*6}")
    for f in fields:
        status = "PASS" if f.passed else "FAIL"
        print(
            f"  {f.field:<14} {f.n_values:>6}  {_fmt_sci(f.max_abs_diff):>14}  "
            f"{_fmt_sci(f.max_rel_diff):>14}  {_fmt_sci(f.tolerance):>12}  {status:>6}"
        )


def print_scientific(se: ScientificEquivalence, p_label: str):
    """Print scientific equivalence metrics."""
    print(f"\n  Scientific Equivalence ({p_label}):")
    print(f"    SNPs compared:              {se.n_snps}")
    print(f"    P-value rank correlation:   {se.spearman_rho:.6f}")
    print(f"    Significance (p < 0.05):    {se.sig_05[0]}/{se.sig_05[1]}")
    print(f"    Significance (p < 0.01):    {se.sig_01[0]}/{se.sig_01[1]}")
    print(f"    Significance (p < 0.001):   {se.sig_001[0]}/{se.sig_001[1]}")
    print(f"    Significance (p < 5e-8):    {se.sig_5e8[0]}/{se.sig_5e8[1]}")
    print(f"    Effect direction agreement: {se.effect_direction_agree * 100:.1f}%")


def _extract_fields(results, field_name):
    """Extract a field from results as numpy array, using NaN for missing."""
    return np.array(
        [getattr(r, field_name) if getattr(r, field_name) else np.nan for r in results]
    )


def main():
    total_start = time.perf_counter()
    timings: list[SectionTiming] = []

    print("=" * 70)
    print("  JAMMA vs GEMMA: Numerical Equivalence & Performance Report")
    print("=" * 70)
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset: gemma_synthetic (PLINK prefix: {PLINK_PREFIX.name})")
    print("  Runner: JAX (grid search + golden section)")

    # Load shared data
    plink_data = load_plink_binary(PLINK_PREFIX)
    fam_path = PLINK_PREFIX.with_suffix(".fam")
    phenotypes = []
    with open(fam_path) as f:
        for line in f:
            parts = line.strip().split()
            val = parts[5] if len(parts) >= 6 else "-9"
            phenotypes.append(np.nan if val in ("-9", "NA") else float(val))
    phenotypes = np.array(phenotypes)
    snp_info = _build_snp_info(plink_data)
    ref_kinship = read_kinship_matrix(GEMMA_KINSHIP)

    n_samples = plink_data.genotypes.shape[0]
    n_snps = plink_data.genotypes.shape[1]
    print(f"  Samples: {n_samples}, SNPs: {n_snps}")

    all_passed = True

    # ===== 1. KINSHIP =====
    print_section("1. Kinship Matrix (K = XX'/p)")

    t0 = time.perf_counter()
    gemma_K = load_gemma_kinship(GEMMA_KINSHIP)
    jamma_K = compute_centered_kinship(plink_data.genotypes)
    t_kinship = time.perf_counter() - t0
    timings.append(SectionTiming("Kinship", t_kinship, n_snps))

    abs_diff = np.abs(jamma_K - gemma_K)
    max_abs = float(np.max(abs_diff))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / np.abs(gemma_K)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
    max_rel = float(np.max(rel_diff))

    fields = [
        FieldResult(
            "kinship",
            n_samples * n_samples,
            max_abs,
            max_rel,
            TOLERANCES["kinship"],
            max_rel <= TOLERANCES["kinship"],
        )
    ]
    print_field_table(fields)
    print(f"\n  Symmetric: {np.allclose(jamma_K, jamma_K.T)}")
    print(f"  PSD: {np.all(np.linalg.eigvalsh(jamma_K) >= -1e-10)}")
    print(f"  Time: {t_kinship:.3f}s")
    if not fields[0].passed:
        all_passed = False

    # ===== 2. WALD TEST (-lmm 1) =====
    print_section("2. Wald Test (-lmm 1)")

    gemma_wald = load_gemma_assoc(GEMMA_WALD)

    t0 = time.perf_counter()
    jamma_wald = run_lmm_association_jax(
        genotypes=plink_data.genotypes,
        phenotypes=phenotypes,
        kinship=ref_kinship,
        snp_info=snp_info,
        lmm_mode=1,
        **JAX_KWARGS,
    )
    t_wald = time.perf_counter() - t0
    timings.append(SectionTiming("Wald (-lmm 1)", t_wald, len(jamma_wald)))

    g_beta = np.array([r.beta for r in gemma_wald])
    g_se = np.array([r.se for r in gemma_wald])
    g_pwald = np.array([r.p_wald for r in gemma_wald])
    g_logl = _extract_fields(gemma_wald, "logl_H1")
    g_lam = _extract_fields(gemma_wald, "l_remle")

    fields = [
        _compute_field_diffs(
            np.array([r.beta for r in jamma_wald]), g_beta, "beta", TOLERANCES["beta"]
        ),
        _compute_field_diffs(
            np.array([r.se for r in jamma_wald]), g_se, "se", TOLERANCES["se"]
        ),
        _compute_field_diffs(
            np.array([r.p_wald for r in jamma_wald]),
            g_pwald,
            "p_wald",
            TOLERANCES["p_wald"],
        ),
        _compute_field_diffs(
            _extract_fields(jamma_wald, "logl_H1"),
            g_logl,
            "logl_H1",
            TOLERANCES["logl"],
        ),
        _compute_field_diffs(
            _extract_fields(jamma_wald, "l_remle"),
            g_lam,
            "l_remle",
            TOLERANCES["lambda"],
        ),
    ]
    print_field_table(fields)
    print(f"\n  Time: {t_wald:.3f}s ({len(jamma_wald)} SNPs)")
    for f in fields:
        if not f.passed:
            all_passed = False

    se = _scientific_equivalence(jamma_wald, gemma_wald, "p_wald")
    print_scientific(se, "Wald p-value")

    # ===== 3. SCORE TEST (-lmm 3) =====
    if GEMMA_SCORE.exists():
        print_section("3. Score Test (-lmm 3)")

        gemma_score = load_gemma_assoc(GEMMA_SCORE)

        t0 = time.perf_counter()
        jamma_score = run_lmm_association_jax(
            genotypes=plink_data.genotypes,
            phenotypes=phenotypes,
            kinship=ref_kinship,
            snp_info=snp_info,
            lmm_mode=3,
            **JAX_KWARGS,
        )
        t_score = time.perf_counter() - t0
        timings.append(SectionTiming("Score (-lmm 3)", t_score, len(jamma_score)))

        fields = [
            _compute_field_diffs(
                np.array([r.p_score for r in jamma_score]),
                np.array([r.p_score for r in gemma_score]),
                "p_score",
                TOLERANCES["p_score"],
            ),
        ]
        print_field_table(fields)
        print(f"\n  Time: {t_score:.3f}s ({len(jamma_score)} SNPs)")
        for f in fields:
            if not f.passed:
                all_passed = False

        se_score = _scientific_equivalence(jamma_score, gemma_score, "p_score")
        print_scientific(se_score, "Score p-value")
    else:
        print("\n  [SKIPPED] Score test reference not found")

    # ===== 4. LRT (-lmm 2) =====
    if GEMMA_LRT.exists():
        print_section("4. Likelihood Ratio Test (-lmm 2)")

        gemma_lrt = load_gemma_assoc(GEMMA_LRT)

        t0 = time.perf_counter()
        jamma_lrt = run_lmm_association_jax(
            genotypes=plink_data.genotypes,
            phenotypes=phenotypes,
            kinship=ref_kinship,
            snp_info=snp_info,
            lmm_mode=2,
            **JAX_KWARGS,
        )
        t_lrt = time.perf_counter() - t0
        timings.append(SectionTiming("LRT (-lmm 2)", t_lrt, len(jamma_lrt)))

        fields = [
            _compute_field_diffs(
                np.array([r.p_lrt for r in jamma_lrt]),
                np.array([r.p_lrt for r in gemma_lrt]),
                "p_lrt",
                TOLERANCES["p_lrt"],
            ),
            _compute_field_diffs(
                _extract_fields(jamma_lrt, "l_mle"),
                _extract_fields(gemma_lrt, "l_mle"),
                "l_mle",
                TOLERANCES["lambda"],
            ),
        ]
        print_field_table(fields)
        print(f"\n  Time: {t_lrt:.3f}s ({len(jamma_lrt)} SNPs)")
        for f in fields:
            if not f.passed:
                all_passed = False

        se_lrt = _scientific_equivalence(jamma_lrt, gemma_lrt, "p_lrt")
        print_scientific(se_lrt, "LRT p-value")
    else:
        print("\n  [SKIPPED] LRT reference not found")

    # ===== 5. ALL TESTS (-lmm 4) =====
    if GEMMA_ALL.exists():
        print_section("5. All Tests Mode (-lmm 4)")

        gemma_all = load_gemma_assoc(GEMMA_ALL)

        t0 = time.perf_counter()
        jamma_all = run_lmm_association_jax(
            genotypes=plink_data.genotypes,
            phenotypes=phenotypes,
            kinship=ref_kinship,
            snp_info=snp_info,
            lmm_mode=4,
            **JAX_KWARGS,
        )
        t_all = time.perf_counter() - t0
        timings.append(SectionTiming("All tests (-lmm 4)", t_all, len(jamma_all)))

        fields = [
            _compute_field_diffs(
                np.array([r.beta for r in jamma_all]),
                np.array([r.beta for r in gemma_all]),
                "beta",
                TOLERANCES["beta"],
            ),
            _compute_field_diffs(
                np.array([r.p_wald for r in jamma_all]),
                np.array([r.p_wald for r in gemma_all]),
                "p_wald",
                TOLERANCES["p_wald"],
            ),
            _compute_field_diffs(
                _extract_fields(jamma_all, "p_lrt"),
                _extract_fields(gemma_all, "p_lrt"),
                "p_lrt",
                TOLERANCES["p_lrt"],
            ),
            _compute_field_diffs(
                _extract_fields(jamma_all, "p_score"),
                _extract_fields(gemma_all, "p_score"),
                "p_score",
                TOLERANCES["p_score"],
            ),
            _compute_field_diffs(
                _extract_fields(jamma_all, "l_remle"),
                _extract_fields(gemma_all, "l_remle"),
                "l_remle",
                TOLERANCES["lambda"],
            ),
            _compute_field_diffs(
                _extract_fields(jamma_all, "l_mle"),
                _extract_fields(gemma_all, "l_mle"),
                "l_mle",
                TOLERANCES["lambda"],
            ),
        ]
        print_field_table(fields)
        print(f"\n  Time: {t_all:.3f}s ({len(jamma_all)} SNPs)")
        for f in fields:
            if not f.passed:
                all_passed = False

        se_all = _scientific_equivalence(jamma_all, gemma_all, "p_wald")
        print_scientific(se_all, "All-tests Wald p-value")

    # ===== 6. COVARIATES (-lmm 1 -c) =====
    if GEMMA_COVAR.exists() and COVARIATE_FILE.exists():
        print_section("6. Wald Test with Covariates (-lmm 1 -c)")

        covariates = np.loadtxt(COVARIATE_FILE)
        gemma_covar = load_gemma_assoc(GEMMA_COVAR)

        t0 = time.perf_counter()
        jamma_covar = run_lmm_association_jax(
            genotypes=plink_data.genotypes,
            phenotypes=phenotypes,
            kinship=ref_kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=1,
            **JAX_KWARGS,
        )
        t_covar = time.perf_counter() - t0
        timings.append(
            SectionTiming("Wald+covar (-lmm 1 -c)", t_covar, len(jamma_covar))
        )

        fields = [
            _compute_field_diffs(
                np.array([r.beta for r in jamma_covar]),
                np.array([r.beta for r in gemma_covar]),
                "beta",
                TOLERANCES["beta"],
            ),
            _compute_field_diffs(
                np.array([r.p_wald for r in jamma_covar]),
                np.array([r.p_wald for r in gemma_covar]),
                "p_wald",
                TOLERANCES["p_wald"],
            ),
            _compute_field_diffs(
                _extract_fields(jamma_covar, "l_remle"),
                _extract_fields(gemma_covar, "l_remle"),
                "l_remle",
                TOLERANCES["lambda"],
            ),
        ]
        print_field_table(fields)
        print(f"\n  Time: {t_covar:.3f}s ({len(jamma_covar)} SNPs)")
        for f in fields:
            if not f.passed:
                all_passed = False

        se_covar = _scientific_equivalence(jamma_covar, gemma_covar, "p_wald")
        print_scientific(se_covar, "Wald p-value (with covariates)")

    # ===== 7. ALL TESTS + COVARIATES (-lmm 4 -c) =====
    if GEMMA_ALL_COVAR.exists() and COVARIATE_FILE.exists():
        print_section("7. All Tests with Covariates (-lmm 4 -c)")

        covariates = np.loadtxt(COVARIATE_FILE)
        gemma_all_covar = load_gemma_assoc(GEMMA_ALL_COVAR)

        t0 = time.perf_counter()
        jamma_all_covar = run_lmm_association_jax(
            genotypes=plink_data.genotypes,
            phenotypes=phenotypes,
            kinship=ref_kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=4,
            **JAX_KWARGS,
        )
        t_allc = time.perf_counter() - t0
        timings.append(
            SectionTiming("All+covar (-lmm 4 -c)", t_allc, len(jamma_all_covar))
        )

        fields = [
            _compute_field_diffs(
                np.array([r.beta for r in jamma_all_covar]),
                np.array([r.beta for r in gemma_all_covar]),
                "beta",
                TOLERANCES["beta"],
            ),
            _compute_field_diffs(
                np.array([r.p_wald for r in jamma_all_covar]),
                np.array([r.p_wald for r in gemma_all_covar]),
                "p_wald",
                TOLERANCES["p_wald"],
            ),
            _compute_field_diffs(
                _extract_fields(jamma_all_covar, "p_lrt"),
                _extract_fields(gemma_all_covar, "p_lrt"),
                "p_lrt",
                TOLERANCES["p_lrt"],
            ),
            _compute_field_diffs(
                _extract_fields(jamma_all_covar, "p_score"),
                _extract_fields(gemma_all_covar, "p_score"),
                "p_score",
                TOLERANCES["p_score"],
            ),
        ]
        print_field_table(fields)
        print(f"\n  Time: {t_allc:.3f}s ({len(jamma_all_covar)} SNPs)")
        for f in fields:
            if not f.passed:
                all_passed = False

        se_allc = _scientific_equivalence(jamma_all_covar, gemma_all_covar, "p_wald")
        print_scientific(se_allc, "All-tests Wald (covariates)")

    # ===== PERFORMANCE SUMMARY =====
    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'=' * 70}")
    print("  Performance Summary")
    print(f"{'=' * 70}")
    print(f"  {'Section':<28} {'SNPs':>6}  {'Time (s)':>10}  {'SNPs/sec':>10}")
    print(f"  {'-'*28} {'-'*6}  {'-'*10}  {'-'*10}")
    for t in timings:
        snps_per_sec = t.n_snps / t.elapsed if t.elapsed > 0 else 0
        print(
            f"  {t.name:<28} {t.n_snps:>6}  {t.elapsed:>10.3f}  {snps_per_sec:>10.0f}"
        )
    print(f"  {'-'*28} {'-'*6}  {'-'*10}  {'-'*10}")
    print(f"  {'Total':<28} {'':>6}  {total_elapsed:>10.3f}")

    # ===== VERDICT =====
    print(f"\n{'=' * 70}")
    if all_passed:
        print("  VERDICT: ALL FIELDS PASS TOLERANCES")
    else:
        print("  VERDICT: SOME FIELDS EXCEED TOLERANCES — SEE ABOVE")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
