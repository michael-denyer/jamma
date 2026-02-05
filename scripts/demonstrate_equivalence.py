"""Numerical equivalence and performance report: JAMMA vs GEMMA.

Runs JAMMA's JAX runner against GEMMA reference data on two datasets:
  1. gemma_synthetic (100 samples, 500 SNPs) — tight tolerances
  2. mouse_hs1940 (1940 samples, 12226 SNPs) — real data, wider tolerances

Produces per-field max difference tables, scientific equivalence metrics,
and per-section performance timing.

Usage:
    uv run python scripts/demonstrate_equivalence.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
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

configure_jax(enable_x64=True)

# Common JAX runner kwargs
JAX_KWARGS = dict(n_grid=50, n_refine=20, show_progress=False, check_memory=False)


# --- Dataset configurations ---
@dataclass
class DatasetConfig:
    """Configuration for one dataset's equivalence test."""

    name: str
    plink_prefix: Path
    kinship_path: Path
    covariate_path: Path | None
    tolerances: dict[str, float]
    compare_kinship: bool = True
    prepend_intercept: bool = False
    tests: list[TestSpec] = field(default_factory=list)


@dataclass
class TestSpec:
    """One LMM mode test to run."""

    name: str
    ref_path: Path
    lmm_mode: int
    use_covariates: bool = False


# Synthetic dataset
SYNTHETIC = DatasetConfig(
    name="gemma_synthetic (100 samples, 500 SNPs)",
    plink_prefix=ROOT / "tests/fixtures/gemma_synthetic/test",
    kinship_path=ROOT / "tests/fixtures/gemma_synthetic/gemma_kinship.cXX.txt",
    covariate_path=ROOT / "tests/fixtures/gemma_covariate/covariates.txt",
    tolerances={
        "kinship": 1e-8,
        "beta": 1e-2,
        "se": 1e-5,
        "p_wald": 1e-4,
        "p_score": 1e-4,
        "p_lrt": 5e-3,
        "logl": 1e-6,
        "lambda": 5e-5,
        "atol": 1e-12,
    },
    tests=[
        TestSpec(
            "Wald (-lmm 1)",
            ROOT / "tests/fixtures/gemma_synthetic/gemma_assoc.assoc.txt",
            1,
        ),
        TestSpec(
            "Score (-lmm 3)",
            ROOT / "tests/fixtures/gemma_score/gemma_score.assoc.txt",
            3,
        ),
        TestSpec(
            "LRT (-lmm 2)",
            ROOT / "tests/fixtures/gemma_synthetic/gemma_lrt.assoc.txt",
            2,
        ),
        TestSpec(
            "All tests (-lmm 4)",
            ROOT / "tests/fixtures/gemma_all_tests/gemma_all.assoc.txt",
            4,
        ),
        TestSpec(
            "Wald+covar (-lmm 1 -c)",
            ROOT / "tests/fixtures/gemma_covariate/gemma_covariate.assoc.txt",
            1,
            use_covariates=True,
        ),
        TestSpec(
            "All+covar (-lmm 4 -c)",
            ROOT / "tests/fixtures/gemma_all_tests/gemma_all_covar.assoc.txt",
            4,
            use_covariates=True,
        ),
    ],
)

# Mouse HS1940 dataset (wider tolerances for real data)
MOUSE_DIR = ROOT / "tests/fixtures/mouse_hs1940"
MOUSE_HS1940 = DatasetConfig(
    name="mouse_hs1940 (1940 samples, 12226 SNPs)",
    plink_prefix=MOUSE_DIR / "mouse_hs1940",
    kinship_path=MOUSE_DIR / "mouse_hs1940_kinship.cXX.txt",
    covariate_path=MOUSE_DIR / "covariates.txt",
    compare_kinship=False,  # GEMMA kinship used as input, not compared
    # covariates.txt lacks intercept column; CI tests prepend it
    prepend_intercept=True,
    tolerances={
        "kinship": 1e-8,
        "beta": 1e-2,
        "se": 5e-4,
        "p_wald": 1e-2,
        "p_score": 1e-2,
        "p_lrt": 1e-2,
        "logl": 5e-3,
        "lambda": 1e-3,
        "atol": 1e-4,
    },
    tests=[
        TestSpec("LRT (-lmm 2)", MOUSE_DIR / "mouse_hs1940_lrt.assoc.txt", 2),
        TestSpec("Score (-lmm 3)", MOUSE_DIR / "mouse_hs1940_score.assoc.txt", 3),
        TestSpec("All tests (-lmm 4)", MOUSE_DIR / "mouse_hs1940_all.assoc.txt", 4),
        TestSpec(
            "Wald+covar (-lmm 1 -c)",
            MOUSE_DIR / "mouse_hs1940_covar_wald.assoc.txt",
            1,
            use_covariates=True,
        ),
        TestSpec(
            "LRT+covar (-lmm 2 -c)",
            MOUSE_DIR / "mouse_hs1940_covar_lrt.assoc.txt",
            2,
            use_covariates=True,
        ),
        TestSpec(
            "Score+covar (-lmm 3 -c)",
            MOUSE_DIR / "mouse_hs1940_covar_score.assoc.txt",
            3,
            use_covariates=True,
        ),
        TestSpec(
            "All+covar (-lmm 4 -c)",
            MOUSE_DIR / "mouse_hs1940_covar_all.assoc.txt",
            4,
            use_covariates=True,
        ),
    ],
)


# --- Data classes ---
@dataclass
class FieldResult:
    field: str
    n_values: int
    max_abs_diff: float
    max_rel_diff: float
    tolerance: float
    passed: bool


@dataclass
class ScientificEquivalence:
    n_snps: int
    spearman_rho: float
    sig_05: tuple[int, int]
    sig_01: tuple[int, int]
    sig_001: tuple[int, int]
    sig_5e8: tuple[int, int]
    effect_direction_agree: float


@dataclass
class SectionTiming:
    name: str
    elapsed: float
    n_snps: int


# --- Computation helpers ---
def _compute_field_diffs(
    actual: np.ndarray,
    expected: np.ndarray,
    field_name: str,
    rtol: float,
    atol: float = 0.0,
) -> FieldResult:
    mask = np.isfinite(actual) & np.isfinite(expected)
    a, e = actual[mask], expected[mask]
    if len(a) == 0:
        return FieldResult(field_name, 0, 0.0, 0.0, rtol, True)

    abs_diff = np.abs(a - e)
    max_abs = float(np.max(abs_diff))

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / np.abs(e)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
    max_rel = float(np.max(rel_diff))

    # Match numpy.testing.assert_allclose: |a-e| <= atol + rtol * |e|
    passed = bool(np.all(abs_diff <= atol + rtol * np.abs(e)))

    return FieldResult(field_name, len(a), max_abs, max_rel, rtol, passed)


def _scientific_equivalence(
    jamma: list, gemma: list, p_field: str
) -> ScientificEquivalence:
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
    dir_agree = (
        float(np.mean(np.sign(j_beta[beta_mask]) == np.sign(g_beta[beta_mask])))
        if np.sum(beta_mask) > 0
        else 1.0
    )

    return ScientificEquivalence(
        n_snps=len(j_p),
        spearman_rho=rho,
        sig_05=_sig_agree(0.05),
        sig_01=_sig_agree(0.01),
        sig_001=_sig_agree(0.001),
        sig_5e8=_sig_agree(5e-8),
        effect_direction_agree=dir_agree,
    )


def _extract(results, field_name):
    """Extract a field as numpy array, NaN for missing."""
    return np.array(
        [getattr(r, field_name) if getattr(r, field_name) else np.nan for r in results]
    )


def _build_snp_info(plink_data):
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


def _load_phenotypes(fam_path: Path) -> np.ndarray:
    phenotypes = []
    with open(fam_path) as f:
        for line in f:
            parts = line.strip().split()
            val = parts[5] if len(parts) >= 6 else "-9"
            phenotypes.append(np.nan if val in ("-9", "NA") else float(val))
    return np.array(phenotypes)


# --- Output helpers ---
def _fmt_sci(val: float) -> str:
    if val == 0.0:
        return "0"
    return f"{val:.2e}"


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_field_table(fields: list[FieldResult]):
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
    print(f"\n  Scientific Equivalence ({p_label}):")
    print(f"    SNPs compared:              {se.n_snps}")
    print(f"    P-value rank correlation:   {se.spearman_rho:.6f}")
    print(f"    Significance (p < 0.05):    {se.sig_05[0]}/{se.sig_05[1]}")
    print(f"    Significance (p < 0.01):    {se.sig_01[0]}/{se.sig_01[1]}")
    print(f"    Significance (p < 0.001):   {se.sig_001[0]}/{se.sig_001[1]}")
    print(f"    Significance (p < 5e-8):    {se.sig_5e8[0]}/{se.sig_5e8[1]}")
    print(f"    Effect direction agreement: {se.effect_direction_agree * 100:.1f}%")


def print_performance_summary(timings: list[SectionTiming], total: float):
    print(f"\n{'=' * 70}")
    print("  Performance Summary")
    print(f"{'=' * 70}")
    print(f"  {'Section':<32} {'SNPs':>6}  {'Time (s)':>10}  {'SNPs/sec':>10}")
    print(f"  {'-'*32} {'-'*6}  {'-'*10}  {'-'*10}")
    for t in timings:
        snps_per_sec = t.n_snps / t.elapsed if t.elapsed > 0 else 0
        print(
            f"  {t.name:<32} {t.n_snps:>6}  {t.elapsed:>10.3f}  {snps_per_sec:>10.0f}"
        )
    print(f"  {'-'*32} {'-'*6}  {'-'*10}  {'-'*10}")
    print(f"  {'Total':<32} {'':>6}  {total:>10.3f}")


# --- Per-mode field comparisons ---
def _compare_fields(
    lmm_mode: int, jamma: list, gemma: list, tol: dict
) -> list[FieldResult]:
    """Build field comparison list based on LMM mode."""
    fields = []
    atol = tol.get("atol", 0.0)

    # Wald fields (modes 1, 4)
    if lmm_mode in (1, 4):
        fields.append(
            _compute_field_diffs(
                np.array([r.beta for r in jamma]),
                np.array([r.beta for r in gemma]),
                "beta",
                tol["beta"],
                atol,
            )
        )
        fields.append(
            _compute_field_diffs(
                np.array([r.p_wald for r in jamma]),
                np.array([r.p_wald for r in gemma]),
                "p_wald",
                tol["p_wald"],
                atol,
            )
        )

    # SE (mode 1 only — mode 4 doesn't need separate check)
    if lmm_mode == 1:
        fields.append(
            _compute_field_diffs(
                np.array([r.se for r in jamma]),
                np.array([r.se for r in gemma]),
                "se",
                tol["se"],
                atol,
            )
        )

    # LRT fields (modes 2, 4)
    if lmm_mode in (2, 4):
        fields.append(
            _compute_field_diffs(
                _extract(jamma, "p_lrt"),
                _extract(gemma, "p_lrt"),
                "p_lrt",
                tol["p_lrt"],
                atol,
            )
        )

    # Score fields (modes 3, 4)
    if lmm_mode in (3, 4):
        fields.append(
            _compute_field_diffs(
                _extract(jamma, "p_score"),
                _extract(gemma, "p_score"),
                "p_score",
                tol["p_score"],
                atol,
            )
        )

    # Lambda REML (modes 1, 4)
    if lmm_mode in (1, 4):
        fields.append(
            _compute_field_diffs(
                _extract(jamma, "l_remle"),
                _extract(gemma, "l_remle"),
                "l_remle",
                tol["lambda"],
                atol,
            )
        )

    # Lambda MLE (modes 2, 4)
    if lmm_mode in (2, 4):
        fields.append(
            _compute_field_diffs(
                _extract(jamma, "l_mle"),
                _extract(gemma, "l_mle"),
                "l_mle",
                tol["lambda"],
                atol,
            )
        )

    # logl_H1 (mode 1 — the Wald reference sometimes includes it)
    if lmm_mode == 1:
        fields.append(
            _compute_field_diffs(
                _extract(jamma, "logl_H1"),
                _extract(gemma, "logl_H1"),
                "logl_H1",
                tol["logl"],
                atol,
            )
        )

    return fields


def _primary_p_field(lmm_mode: int) -> str:
    """Which p-value field to use for scientific equivalence."""
    if lmm_mode == 2:
        return "p_lrt"
    if lmm_mode == 3:
        return "p_score"
    return "p_wald"


# --- Main ---
def run_dataset(
    config: DatasetConfig, section_offset: int
) -> tuple[bool, list[SectionTiming]]:
    """Run equivalence tests for one dataset. Returns (all_passed, timings)."""
    all_passed = True
    timings: list[SectionTiming] = []

    # Check if dataset exists
    bed_path = config.plink_prefix.with_suffix(".bed")
    if not bed_path.exists():
        print(
            f"\n  [SKIPPED] {config.name} — "
            f"PLINK files not found at {config.plink_prefix}"
        )
        return True, []

    print(f"\n{'#' * 70}")
    print(f"  DATASET: {config.name}")
    print(f"{'#' * 70}")

    # Load data
    plink_data = load_plink_binary(config.plink_prefix)
    phenotypes = _load_phenotypes(config.plink_prefix.with_suffix(".fam"))
    snp_info = _build_snp_info(plink_data)
    ref_kinship = read_kinship_matrix(config.kinship_path)
    n_samples = plink_data.genotypes.shape[0]
    n_snps = plink_data.genotypes.shape[1]
    print(f"  Samples: {n_samples}, SNPs: {n_snps}")

    covariates = None
    if config.covariate_path and config.covariate_path.exists():
        covariates = np.loadtxt(config.covariate_path)
        if config.prepend_intercept:
            covariates = np.hstack([np.ones((covariates.shape[0], 1)), covariates])

    # Kinship comparison (only for datasets where we validated it)
    section_num = section_offset + 1
    if config.compare_kinship:
        print_section(f"{section_num}. Kinship Matrix")

        t0 = time.perf_counter()
        gemma_K = load_gemma_kinship(config.kinship_path)
        jamma_K = compute_centered_kinship(plink_data.genotypes)
        t_kinship = time.perf_counter() - t0
        timings.append(SectionTiming(f"[{config.name[:8]}] Kinship", t_kinship, n_snps))

        abs_diff = np.abs(jamma_K - gemma_K)
        max_abs = float(np.max(abs_diff))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / np.abs(gemma_K)
            rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
        max_rel = float(np.max(rel_diff))

        k_tol = config.tolerances["kinship"]
        fields = [
            FieldResult(
                "kinship",
                n_samples * n_samples,
                max_abs,
                max_rel,
                k_tol,
                max_rel <= k_tol,
            )
        ]
        print_field_table(fields)
        print(f"\n  Time: {t_kinship:.3f}s")
        if not fields[0].passed:
            all_passed = False

    # LMM mode tests
    # Note: eigendecomp is NOT pre-computed because the runner filters samples
    # (removing missing phenotypes) before eigendecomposing. Passing pre-computed
    # eigenvalues from the full kinship causes dimension mismatches.
    for i, spec in enumerate(config.tests):
        section_num = section_offset + 2 + i

        if not spec.ref_path.exists():
            print(f"\n  [{section_num}. {spec.name}] SKIPPED — fixture not found")
            continue

        print_section(f"{section_num}. {spec.name}")

        gemma_ref = load_gemma_assoc(spec.ref_path)
        covar = covariates if spec.use_covariates else None

        t0 = time.perf_counter()
        jamma_results = run_lmm_association_jax(
            genotypes=plink_data.genotypes,
            phenotypes=phenotypes,
            kinship=ref_kinship,
            snp_info=snp_info,
            covariates=covar,
            lmm_mode=spec.lmm_mode,
            **JAX_KWARGS,
        )
        t_elapsed = time.perf_counter() - t0
        timings.append(
            SectionTiming(
                f"[{config.name[:8]}] {spec.name[:20]}", t_elapsed, len(jamma_results)
            )
        )

        fields = _compare_fields(
            spec.lmm_mode, jamma_results, gemma_ref, config.tolerances
        )
        print_field_table(fields)
        print(f"\n  Time: {t_elapsed:.3f}s ({len(jamma_results)} SNPs)")

        for f in fields:
            if not f.passed:
                all_passed = False

        p_field = _primary_p_field(spec.lmm_mode)
        se = _scientific_equivalence(jamma_results, gemma_ref, p_field)
        print_scientific(se, f"{p_field}")

    return all_passed, timings


def main():
    total_start = time.perf_counter()

    print("=" * 70)
    print("  JAMMA vs GEMMA: Numerical Equivalence & Performance Report")
    print("=" * 70)
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Runner: JAX (grid search + golden section)")

    all_timings: list[SectionTiming] = []
    all_passed = True

    # Run both datasets
    for i, config in enumerate([SYNTHETIC, MOUSE_HS1940]):
        offset = sum(1 + len(c.tests) for c in [SYNTHETIC, MOUSE_HS1940][:i])
        passed, timings = run_dataset(config, offset)
        all_timings.extend(timings)
        if not passed:
            all_passed = False

    # Performance summary
    total_elapsed = time.perf_counter() - total_start
    print_performance_summary(all_timings, total_elapsed)

    # Verdict
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
