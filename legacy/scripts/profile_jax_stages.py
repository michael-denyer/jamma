#!/usr/bin/env python3
"""Profile JAX LMM compute stages and overhead at multiple scales.

Instruments each stage of the Wald compute path plus host transfer,
pass-1 SNP stats, kinship text load, and result building overhead.

Usage:
    uv run python scripts/profile_jax_stages.py
    uv run python scripts/profile_jax_stages.py --runs 3
    uv run python scripts/profile_jax_stages.py --scale 5000  # synthetic 5k samples
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jamma.core.jax_config import ensure_jax_configured

ensure_jax_configured()

from jamma.lmm.likelihood_jax import (  # noqa: E402
    batch_calc_wald_stats,
    batch_compute_iab,
    batch_compute_uab,
)
from jamma.lmm.prepare import _grid_optimize_lambda_batched  # noqa: E402
from jamma.lmm.runner_jax import _build_covariate_matrix  # noqa: E402

from jamma.core.snp_filter import (  # noqa: E402
    compute_snp_filter_mask,
    compute_snp_stats,
)
from jamma.io import load_plink_binary  # noqa: E402
from jamma.kinship.io import read_kinship_matrix  # noqa: E402
from jamma.lmm.eigen import eigendecompose_kinship  # noqa: E402

# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOUSE_DIR = _REPO_ROOT / "tests" / "fixtures" / "mouse_hs1940"
_MOUSE_PREFIX = _MOUSE_DIR / "mouse_hs1940"
_MOUSE_KINSHIP = _MOUSE_DIR / "mouse_hs1940_kinship.cXX.txt"

# Result field keys per mode (from runner_jax.py)
_RESULT_FIELDS = {
    1: ("lambdas", "logls", "betas", "ses", "pwalds"),
}


def profile_compute(
    eigenvalues: jax.Array,
    UtW: jax.Array,
    Uty: jax.Array,
    utg_t: jax.Array,
    n_samples: int,
    n_cvt: int,
    warmup: bool = False,
) -> dict[str, float]:
    """Profile the core compute stages."""

    # Stage 1: batch_compute_uab
    t0 = time.perf_counter()
    Uab_batch = batch_compute_uab(n_cvt, UtW, Uty, utg_t)
    Uab_batch.block_until_ready()
    t_uab = time.perf_counter() - t0

    # Stage 2: batch_compute_iab
    t0 = time.perf_counter()
    Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
    Iab_batch.block_until_ready()
    t_iab = time.perf_counter() - t0

    # Stage 3: lambda optimization
    t0 = time.perf_counter()
    lambdas, logls = _grid_optimize_lambda_batched(
        n_cvt,
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=10,
    )
    lambdas.block_until_ready()
    logls.block_until_ready()
    t_optimize = time.perf_counter() - t0

    # Stage 4: Wald stats
    t0 = time.perf_counter()
    betas, ses, pwalds = batch_calc_wald_stats(
        n_cvt, lambdas, eigenvalues, Uab_batch, n_samples
    )
    pwalds.block_until_ready()
    t_wald = time.perf_counter() - t0

    # Stage 5: Host transfer (np.asarray on each array separately)
    t0 = time.perf_counter()
    arrays = {
        "lambdas": lambdas,
        "logls": logls,
        "betas": betas,
        "ses": ses,
        "pwalds": pwalds,
    }
    host_arrays = {k: np.asarray(v) for k, v in arrays.items()}
    t_transfer = time.perf_counter() - t0

    # Stage 6: Result object creation (simulate _build_results overhead)
    t0 = time.perf_counter()
    n_snps = host_arrays["betas"].shape[0]
    results = []
    for i in range(n_snps):
        results.append(
            {
                "beta": float(host_arrays["betas"][i]),
                "se": float(host_arrays["ses"][i]),
                "p_wald": float(host_arrays["pwalds"][i]),
                "logl": float(host_arrays["logls"][i]),
                "lambda": float(host_arrays["lambdas"][i]),
            }
        )
    t_results = time.perf_counter() - t0

    total = t_uab + t_iab + t_optimize + t_wald + t_transfer + t_results

    return {
        "uab": t_uab,
        "iab": t_iab,
        "optimize": t_optimize,
        "wald": t_wald,
        "transfer": t_transfer,
        "results": t_results,
        "total": total,
    }


def profile_snp_stats(genotypes: np.ndarray, n_runs: int = 3) -> float:
    """Profile pass-1 SNP statistics computation."""
    best = float("inf")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        compute_snp_stats(genotypes)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def profile_kinship_text_load(path: Path, n_runs: int = 1) -> float:
    """Profile kinship text loading with np.loadtxt."""
    best = float("inf")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        np.loadtxt(path, dtype=np.float64)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def generate_synthetic(n_samples: int, n_snps: int = 10_000, seed: int = 42):
    """Generate synthetic genotypes, phenotypes, kinship for profiling."""
    rng = np.random.default_rng(seed)
    # Genotypes: random 0/1/2 with ~5% missing
    genotypes = rng.choice(
        [0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.5, 0.35, 0.15]
    )
    missing_mask = rng.random((n_samples, n_snps)) < 0.05
    genotypes[missing_mask] = np.nan

    # Phenotypes
    phenotypes = rng.standard_normal(n_samples)

    # Kinship (symmetric positive semi-definite)
    X = rng.standard_normal((n_samples, 500))
    kinship = X @ X.T / 500
    kinship = (kinship + kinship.T) / 2

    return genotypes, phenotypes, kinship


def run_profile(label, genotypes, phenotypes, kinship, n_runs):
    """Run full profile for a given dataset."""
    n_samples = phenotypes.shape[0]
    n_snps_raw = genotypes.shape[1]
    print(f"\n{'=' * 60}")
    print(f"  {label}: {n_samples:,} samples x {n_snps_raw:,} SNPs")
    print(f"{'=' * 60}")

    # SNP filtering
    col_means, missing_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, _, _ = compute_snp_filter_mask(
        col_means, missing_counts, col_vars, n_samples, 1e-2, 0.05
    )
    snp_indices = np.where(snp_mask)[0]
    genotypes_filtered = genotypes[:, snp_indices]

    # Eigendecomposition
    eigenvalues_np, U = eigendecompose_kinship(kinship, check_memory=False)

    # Rotate
    W, n_cvt = _build_covariate_matrix(None, n_samples)
    UtW = U.T @ W
    Uty = U.T @ phenotypes
    utg_t = (U.T @ genotypes_filtered).T  # (n_snps, n_samples)

    n_filtered = utg_t.shape[0]
    uab_cols = (n_cvt + 3) * (n_cvt + 2) // 2
    uab_mb = n_filtered * n_samples * uab_cols * 8 / 1e6
    print(f"  Filtered SNPs: {n_filtered:,}, n_cvt={n_cvt}")
    print(f"  Uab: ({n_filtered}, {n_samples}, {uab_cols}) = {uab_mb:.0f} MB")

    eigenvalues_jax = jnp.array(eigenvalues_np)
    UtW_jax = jnp.array(UtW)
    Uty_jax = jnp.array(Uty)
    utg_t_jax = jnp.array(utg_t)

    # Warmup
    print("  Warmup (JIT)...", end=" ", flush=True)
    profile_compute(
        eigenvalues_jax, UtW_jax, Uty_jax, utg_t_jax, n_samples, n_cvt, warmup=True
    )
    print("done")

    # Profile: SNP stats
    t_stats = profile_snp_stats(genotypes, n_runs=n_runs)
    print(f"\n  Pass-1 SNP stats:     {t_stats:7.3f}s")

    # Profile: UT@G rotation
    t0 = time.perf_counter()
    _utg_t = (U.T @ genotypes_filtered).T
    t_rotation = time.perf_counter() - t0
    print(f"  UT@G rotation:        {t_rotation:7.3f}s")

    # Profile: compute stages
    all_results = []
    for _i in range(n_runs):
        r = profile_compute(
            eigenvalues_jax, UtW_jax, Uty_jax, utg_t_jax, n_samples, n_cvt
        )
        all_results.append(r)

    best = min(all_results, key=lambda r: r["total"])

    print(f"\n  COMPUTE STAGES (best of {n_runs}):")
    total = best["total"]
    for key in ("uab", "iab", "optimize", "wald", "transfer", "results"):
        pct = 100 * best[key] / total if total > 0 else 0
        print(f"    {key:22s} {best[key]:7.3f}s  ({pct:5.1f}%)")
    print(f"    {'TOTAL':22s} {total:7.3f}s")

    # Full pipeline estimate
    pipeline = t_stats + t_rotation + total
    print(f"\n  PIPELINE ESTIMATE:    {pipeline:7.3f}s")
    print(
        f"    SNP stats:          {t_stats:7.3f}s  ({100 * t_stats / pipeline:5.1f}%)"
    )
    print(
        f"    UT@G rotation:      {t_rotation:7.3f}s"
        f"  ({100 * t_rotation / pipeline:5.1f}%)"
    )
    print(f"    JAX compute:        {total:7.3f}s  ({100 * total / pipeline:5.1f}%)")

    return best


def main():
    parser = argparse.ArgumentParser(
        description="Profile JAX LMM stages at multiple scales"
    )
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per config")
    parser.add_argument(
        "--scale",
        type=int,
        nargs="*",
        default=None,
        help="Synthetic sample counts (e.g. --scale 5000 10000)",
    )
    parser.add_argument(
        "--kinship-text", action="store_true", help="Also profile kinship text loading"
    )
    args = parser.parse_args()

    # Always run mouse_hs1940
    print("Loading mouse_hs1940...")
    plink = load_plink_binary(_MOUSE_PREFIX)
    kinship = read_kinship_matrix(_MOUSE_KINSHIP)
    genotypes = plink.genotypes

    from jamma.core.constants import PHENOTYPE_MISSING

    fam_data = np.loadtxt(_MOUSE_PREFIX.with_suffix(".fam"), usecols=5, dtype=str)
    missing = np.isin(fam_data, [str(int(PHENOTYPE_MISSING)), "NA"])
    phenotypes = np.where(missing, "0", fam_data).astype(np.float64)
    phenotypes[missing] = np.nan

    valid = ~np.isnan(phenotypes)
    genotypes = genotypes[valid]
    phenotypes = phenotypes[valid]
    kinship = kinship[np.ix_(valid, valid)]

    run_profile("mouse_hs1940", genotypes, phenotypes, kinship, args.runs)

    # Kinship text load timing
    if args.kinship_text:
        print("\n  Kinship text load (np.loadtxt): ", end="", flush=True)
        t_kin = profile_kinship_text_load(_MOUSE_KINSHIP)
        print(f"{t_kin:.3f}s ({kinship.shape[0]}x{kinship.shape[0]})")

    # Synthetic scales
    if args.scale:
        for n in args.scale:
            n_snps = max(10_000, n * 2)  # scale SNPs with samples
            geno, pheno, kin = generate_synthetic(n, n_snps)
            run_profile(f"synthetic {n:,}x{n_snps:,}", geno, pheno, kin, args.runs)


if __name__ == "__main__":
    main()
