"""Numerical equivalence and performance report: JAMMA vs GEMMA.

Runs JAMMA's NumPy runner against GEMMA reference data on two datasets:
  1. gemma_synthetic (100 samples, 500 SNPs) — tight tolerances
  2. mouse_hs1940 (1940 samples, 12226 SNPs) — real data, wider tolerances

Produces per-field max difference tables, scientific equivalence metrics,
and per-section performance timing. The per-field comparison is the same
``compare_assoc_results`` the tier1 parity tests use, with the same
``ToleranceConfig`` overrides, so the report and the suite cannot disagree
about what "within tolerance" means.

Usage:
    uv run python scripts/demonstrate_equivalence.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jamma.io import load_plink_binary  # noqa: E402
from jamma.kinship import compute_centered_kinship  # noqa: E402
from jamma.kinship.io import read_kinship_matrix  # noqa: E402
from jamma.lmm.runner_numpy import run_lmm_association_numpy  # noqa: E402
from jamma.lmm.schema import LmmConfig, LmmMode  # noqa: E402
from jamma.validation import (  # noqa: E402
    ToleranceConfig,
    compare_assoc_results,
    compare_kinship_matrices,
    load_gemma_assoc,
    load_gemma_kinship,
)
from jamma.validation.compare import ComparisonResult  # noqa: E402

# Common runner config knobs, merged with each spec's mode at the call site.
RUNNER_CONFIG_KWARGS = {
    "n_grid": 50,
    "n_refine": 20,
    "show_progress": False,
    "check_memory": False,
}


# --- Dataset configurations ---
@dataclass
class TestSpec:
    """One LMM mode test to run."""

    name: str
    ref_path: Path
    lmm_mode: LmmMode
    use_covariates: bool = False


@dataclass
class DatasetConfig:
    """Configuration for one dataset's equivalence test."""

    name: str
    plink_prefix: Path
    kinship_path: Path
    covariate_path: Path | None
    tolerances: ToleranceConfig
    tests: list[TestSpec]
    compare_kinship: bool = True
    prepend_intercept: bool = False


# Synthetic dataset
SYNTHETIC = DatasetConfig(
    name="gemma_synthetic (100 samples, 500 SNPs)",
    plink_prefix=ROOT / "tests/fixtures/gemma_synthetic/test",
    kinship_path=ROOT / "tests/fixtures/gemma_synthetic/gemma_kinship.cXX.txt",
    covariate_path=ROOT / "tests/fixtures/gemma_covariate/covariates.txt",
    # Golden section vs Brent: ~6.6e-5 per 20-iteration bracket, same value
    # the tier1 suite uses on this dataset.
    tolerances=ToleranceConfig(lambda_rtol=5e-5),
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
    # The same overrides tests/test_runner_numpy.py::NUMPY_GEMMA_TOLERANCES
    # applies to this dataset.
    tolerances=ToleranceConfig(
        lambda_rtol=1e-3,
        pvalue_rtol=1e-2,
        se_rtol=5e-4,
        logl_rtol=5e-3,
        atol=1e-4,
    ),
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


@dataclass
class SectionTiming:
    name: str
    elapsed: float
    n_snps: int


# Report order for the AssocComparisonResult columns, with the ToleranceConfig
# field each one is judged against.
_ASSOC_COLUMNS = (
    ("beta", "beta_rtol"),
    ("se", "se_rtol"),
    ("af", "af_rtol"),
    ("p_wald", "pvalue_rtol"),
    ("p_score", "pvalue_rtol"),
    ("p_lrt", "p_lrt_rtol"),
    ("l_remle", "lambda_rtol"),
    ("l_mle", "lambda_rtol"),
    ("logl_H1", "logl_rtol"),
)

# Which p-value the scientific-equivalence block reads, per LMM mode.
_PRIMARY_P_FIELD = {2: "p_lrt", 3: "p_score"}


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman's rho on ordinal ranks (ties ranked by position).

    Written over ``np.argsort`` so this script imports no scipy: installing
    scipy overwrites the ILP64 numpy build (CLAUDE.md, "No scipy at runtime"),
    and the tier1 suite runs this script. ``-log10(p)`` inputs are continuous,
    so ordinal and average ranks agree for all practical purposes.
    """
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def _print_scientific_equivalence(jamma: list, gemma: list, p_field: str) -> None:
    """Compute and print scientific equivalence metrics."""
    j_by_rs = {r.rs: r for r in jamma}
    g_by_rs = {r.rs: r for r in gemma}
    common = sorted(set(j_by_rs) & set(g_by_rs))

    j_p = np.array([getattr(j_by_rs[rs], p_field) for rs in common])
    g_p = np.array([getattr(g_by_rs[rs], p_field) for rs in common])

    mask = np.isfinite(j_p) & np.isfinite(g_p) & (j_p > 0) & (g_p > 0)
    j_p, g_p = j_p[mask], g_p[mask]

    rho = _rank_correlation(-np.log10(j_p), -np.log10(g_p))

    def sig_agree(thresh):
        agree = int(np.sum((j_p < thresh) == (g_p < thresh)))
        return f"{agree}/{len(j_p)}"

    j_beta = np.array([j_by_rs[rs].beta for rs in common])
    g_beta = np.array([g_by_rs[rs].beta for rs in common])
    beta_mask = np.isfinite(j_beta) & np.isfinite(g_beta) & (np.abs(g_beta) > 1e-10)
    dir_agree = (
        float(np.mean(np.sign(j_beta[beta_mask]) == np.sign(g_beta[beta_mask])))
        if np.sum(beta_mask) > 0
        else 1.0
    )

    print(f"\n  Scientific Equivalence ({p_field}):")
    print(f"    SNPs compared:              {len(j_p)}")
    print(f"    P-value rank correlation:   {rho:.6f}")
    print(f"    Significance (p < 0.05):    {sig_agree(0.05)}")
    print(f"    Significance (p < 0.01):    {sig_agree(0.01)}")
    print(f"    Significance (p < 0.001):   {sig_agree(0.001)}")
    print(f"    Significance (p < 5e-8):    {sig_agree(5e-8)}")
    print(f"    Effect direction agreement: {dir_agree * 100:.1f}%")


def _build_snp_info(plink_data):
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


def _load_phenotypes(fam_path: Path) -> np.ndarray:
    """Load phenotypes from column 6 of a .fam file (NaN for missing)."""
    with open(fam_path) as f:
        parts = [line.strip().split() for line in f]
    vals = [p[5] if len(p) >= 6 else "-9" for p in parts]
    return np.array([np.nan if v in ("-9", "NA") else float(v) for v in vals])


def _fmt_sci(val: float) -> str:
    if val == 0.0:
        return "0"
    return f"{val:.2e}"


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_field_table(rows: list[tuple[str, float, ComparisonResult]]) -> None:
    """Print one line per compared column: (field, tolerance, result)."""
    print(
        f"  {'Field':<14}  {'Max Abs Diff':>14}  "
        f"{'Max Rel Diff':>14}  {'Tolerance':>12}  {'Result':>6}"
    )
    print(f"  {'-' * 14}  {'-' * 14}  {'-' * 14}  {'-' * 12}  {'-' * 6}")
    for field, tolerance, r in rows:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {field:<14}  {_fmt_sci(r.max_abs_diff):>14}  "
            f"{_fmt_sci(r.max_rel_diff):>14}  {_fmt_sci(tolerance):>12}  {status:>6}"
        )


def print_performance_summary(timings: list[SectionTiming], total: float):
    print(f"\n{'=' * 70}")
    print("  Performance Summary")
    print(f"{'=' * 70}")
    print(f"  {'Section':<32} {'SNPs':>6}  {'Time (s)':>10}  {'SNPs/sec':>10}")
    print(f"  {'-' * 32} {'-' * 6}  {'-' * 10}  {'-' * 10}")
    for t in timings:
        snps_per_sec = t.n_snps / t.elapsed if t.elapsed > 0 else 0
        print(
            f"  {t.name:<32} {t.n_snps:>6}  {t.elapsed:>10.3f}  {snps_per_sec:>10.0f}"
        )
    print(f"  {'-' * 32} {'-' * 6}  {'-' * 10}  {'-' * 10}")
    print(f"  {'Total':<32} {'':>6}  {total:>10.3f}")


def _assoc_rows(
    jamma: list, gemma: list, tol: ToleranceConfig
) -> tuple[bool, list[tuple[str, float, ComparisonResult]]]:
    """Compare one mode's results and lay the active columns out as rows.

    Columns compare_assoc_results skipped for the detected mode (a
    vacuously-passing result with no measured difference) stay out of the
    table, so each mode shows only the fields GEMMA wrote for it.
    """
    comparison = compare_assoc_results(jamma, gemma, tol)
    rows = []
    for field, tol_field in _ASSOC_COLUMNS:
        r = getattr(comparison, field)
        if r is None or (
            r.passed and r.worst_location is None and "skipped" in r.message
        ):
            continue
        rows.append((field, getattr(tol, tol_field), r))
    if comparison.mismatched_snps:
        print(f"  SNP id mismatches: {len(comparison.mismatched_snps)}")
    return comparison.passed, rows


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

        k_result = compare_kinship_matrices(jamma_K, gemma_K, config.tolerances)
        print_field_table([("kinship", config.tolerances.kinship_rtol, k_result)])
        print(f"\n  Time: {t_kinship:.3f}s")
        if not k_result.passed:
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
        run_result = run_lmm_association_numpy(
            genotypes=plink_data.genotypes,
            phenotypes=phenotypes,
            # eigendecompose_kinship consumes its input, reusing the buffer for
            # the eigenvectors, so each section needs its own copy. Sharing one
            # array made every section after the first decompose eigenvectors.
            kinship=ref_kinship.copy(),
            snp_info=snp_info,
            covariates=covar,
            config=LmmConfig(lmm_mode=spec.lmm_mode, **RUNNER_CONFIG_KWARGS),
        )
        jamma_results = run_result.associations
        t_elapsed = time.perf_counter() - t0
        timings.append(
            SectionTiming(
                f"[{config.name[:8]}] {spec.name[:20]}", t_elapsed, len(jamma_results)
            )
        )

        passed, rows = _assoc_rows(jamma_results, gemma_ref, config.tolerances)
        print_field_table(rows)
        print(f"\n  Time: {t_elapsed:.3f}s ({len(jamma_results)} SNPs)")

        if not passed:
            all_passed = False

        _print_scientific_equivalence(
            jamma_results, gemma_ref, _PRIMARY_P_FIELD.get(spec.lmm_mode, "p_wald")
        )

    return all_passed, timings


def main():
    total_start = time.perf_counter()

    print("=" * 70)
    print("  JAMMA vs GEMMA: Numerical Equivalence & Performance Report")
    print("=" * 70)
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Runner: NumPy (grid search + golden section)")

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
