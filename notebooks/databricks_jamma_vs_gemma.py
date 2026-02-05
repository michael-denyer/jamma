# Databricks notebook source
# MAGIC %md
# MAGIC # JAMMA vs GEMMA: Benchmark & Accuracy Comparison
# MAGIC
# MAGIC Point this notebook at any PLINK dataset and run both JAMMA and GEMMA
# MAGIC to get runtime benchmarks and accuracy comparison.
# MAGIC
# MAGIC **Widgets** (set in sidebar):
# MAGIC | Parameter | Description |
# MAGIC |-----------|-------------|
# MAGIC | `bfile` | PLINK binary prefix (e.g. `/dbfs/data/mouse_hs1940`) |
# MAGIC | `kinship_file` | Pre-computed kinship matrix. Empty = compute from genotypes |
# MAGIC | `covariate_file` | Covariate file (whitespace-delimited). Empty = none |
# MAGIC | `lmm_mode` | 1=Wald, 2=LRT, 3=Score, 4=All |
# MAGIC | `maf` | MAF threshold (default 0.01) |
# MAGIC | `miss` | Missing rate threshold (default 0.05) |
# MAGIC | `run_gemma` | Run GEMMA for comparison (yes/no) |
# MAGIC | `output_dir` | Output directory for results |
# MAGIC
# MAGIC **Cluster requirements:** DBR 15.4+, memory-optimized instance for large datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Dependencies

# COMMAND ----------

# MAGIC %sh # Purge non-MKL BLAS providers
# MAGIC apt-get purge -y libopenblas* libblas* libatlas* liblapack* python3-numpy 2>/dev/null; echo "Non-MKL BLAS purged"

# COMMAND ----------

# MAGIC %pip install mkl

# COMMAND ----------

# MAGIC %pip install numpy --extra-index-url https://michael-denyer.github.io/numpy-mkl --force-reinstall --upgrade

# COMMAND ----------

# MAGIC %pip install psutil loguru threadpoolctl jax jaxlib jaxtyping typer progressbar2 bed-reader

# COMMAND ----------

# MAGIC %pip install git+https://github.com/michael-denyer/jamma.git --no-deps

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widget Setup

# COMMAND ----------

dbutils.widgets.text("bfile", "", "PLINK binary prefix path")  # noqa: F821
dbutils.widgets.text("kinship_file", "", "Kinship matrix file (empty = compute)")  # noqa: F821
dbutils.widgets.text("covariate_file", "", "Covariate file (empty = none)")  # noqa: F821
dbutils.widgets.dropdown(  # noqa: F821
    "lmm_mode", "1", ["1", "2", "3", "4"], "LMM mode (1=Wald 2=LRT 3=Score 4=All)"
)
dbutils.widgets.text("maf", "0.01", "MAF threshold")  # noqa: F821
dbutils.widgets.text("miss", "0.05", "Missing rate threshold")  # noqa: F821
dbutils.widgets.dropdown("run_gemma", "yes", ["yes", "no"], "Run GEMMA for comparison")  # noqa: F821
dbutils.widgets.text("output_dir", "/dbfs/tmp/jamma_benchmark", "Output directory")  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports & Configuration

# COMMAND ----------

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

from jamma.core import configure_jax  # noqa: E402

configure_jax(enable_x64=True)

from jamma.io import load_plink_binary  # noqa: E402
from jamma.kinship.compute import compute_centered_kinship  # noqa: E402
from jamma.kinship.io import read_kinship_matrix, write_kinship_matrix  # noqa: E402
from jamma.lmm.io import write_assoc_results  # noqa: E402
from jamma.lmm.runner_jax import run_lmm_association_jax  # noqa: E402
from jamma.validation import load_gemma_assoc  # noqa: E402

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Widgets & Validate

# COMMAND ----------

BFILE = dbutils.widgets.get("bfile").strip()  # noqa: F821
KINSHIP_FILE = dbutils.widgets.get("kinship_file").strip()  # noqa: F821
COVARIATE_FILE = dbutils.widgets.get("covariate_file").strip()  # noqa: F821
LMM_MODE = int(dbutils.widgets.get("lmm_mode"))  # noqa: F821
MAF = float(dbutils.widgets.get("maf"))  # noqa: F821
MISS = float(dbutils.widgets.get("miss"))  # noqa: F821
RUN_GEMMA = dbutils.widgets.get("run_gemma") == "yes"  # noqa: F821
OUTPUT_DIR = Path(dbutils.widgets.get("output_dir").strip())  # noqa: F821

# Validate required inputs
assert BFILE, "bfile widget is required — set it to your PLINK binary prefix"
bed = Path(f"{BFILE}.bed")
bim = Path(f"{BFILE}.bim")
fam = Path(f"{BFILE}.fam")
assert bed.exists(), f".bed not found: {bed}"
assert bim.exists(), f".bim not found: {bim}"
assert fam.exists(), f".fam not found: {fam}"

if KINSHIP_FILE:
    assert Path(KINSHIP_FILE).exists(), f"Kinship file not found: {KINSHIP_FILE}"
if COVARIATE_FILE:
    assert Path(COVARIATE_FILE).exists(), f"Covariate file not found: {COVARIATE_FILE}"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODE_NAMES = {1: "Wald", 2: "LRT", 3: "Score", 4: "All tests"}
print(f"Dataset:    {BFILE}")
print(f"Kinship:    {KINSHIP_FILE or '(compute from genotypes)'}")
print(f"Covariates: {COVARIATE_FILE or '(none)'}")
print(f"LMM mode:   {LMM_MODE} ({MODE_NAMES[LMM_MODE]})")
print(f"MAF:        {MAF}")
print(f"Miss:       {MISS}")
print(f"Run GEMMA:  {RUN_GEMMA}")
print(f"Output:     {OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### GEMMA Setup (download binary if needed)

# COMMAND ----------

GEMMA_BIN = Path("/tmp/gemma")
GEMMA_URL = "https://github.com/genetics-statistics/GEMMA/releases/download/v0.98.5/gemma-0.98.5-linux-static-AMD64.gz"

if RUN_GEMMA:
    if GEMMA_BIN.exists():
        print(f"GEMMA binary already at {GEMMA_BIN}")
    else:
        print("Downloading GEMMA 0.98.5 static binary...")
        subprocess.run(
            [
                "bash",
                "-c",
                f"curl -sL {GEMMA_URL} | gunzip > {GEMMA_BIN} && chmod +x {GEMMA_BIN}",
            ],
            check=True,
        )
        print(f"GEMMA installed at {GEMMA_BIN}")
    # Verify
    result = subprocess.run([str(GEMMA_BIN)], capture_output=True, text=True)
    # GEMMA prints version info to stderr even on success
    version_line = [line for line in result.stderr.splitlines() if "GEMMA" in line]
    if version_line:
        print(version_line[0].strip())
    else:
        print("GEMMA binary verified (no version string found)")
else:
    print("GEMMA comparison disabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

print("Loading PLINK data...")
t0 = time.perf_counter()
plink_data = load_plink_binary(BFILE)
t_plink = time.perf_counter() - t0
print(f"  {plink_data.n_samples} samples, {plink_data.n_snps} SNPs ({t_plink:.2f}s)")

# Phenotypes from .fam file
with open(fam) as f:
    parts = [line.strip().split() for line in f]
phenotypes = np.array([np.nan if p[5] in ("-9", "NA") else float(p[5]) for p in parts])
n_valid = int(np.sum(np.isfinite(phenotypes)))
print(f"  {n_valid}/{len(phenotypes)} samples with valid phenotype")

# Kinship
if KINSHIP_FILE:
    print(f"Loading kinship from {KINSHIP_FILE}...")
    t0 = time.perf_counter()
    kinship = read_kinship_matrix(KINSHIP_FILE)
    t_kin = time.perf_counter() - t0
    print(f"  {kinship.shape[0]}x{kinship.shape[1]} ({t_kin:.2f}s)")
else:
    print("Computing kinship from genotypes...")
    t0 = time.perf_counter()
    kinship = compute_centered_kinship(plink_data.genotypes, check_memory=False)
    t_kin = time.perf_counter() - t0
    print(f"  {kinship.shape[0]}x{kinship.shape[1]} ({t_kin:.2f}s)")
    # Save JAMMA kinship
    jamma_kin_path = OUTPUT_DIR / "jamma_kinship.cXX.txt"
    write_kinship_matrix(kinship, jamma_kin_path)
    print(f"  Saved to {jamma_kin_path}")

# SNP info for JAMMA
snp_info = [
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

# Covariates
covariates = None
if COVARIATE_FILE:
    raw_cov = np.loadtxt(COVARIATE_FILE)
    if raw_cov.ndim == 1:
        raw_cov = raw_cov.reshape(-1, 1)
    # Prepend intercept column (GEMMA -c convention)
    covariates = np.hstack([np.ones((raw_cov.shape[0], 1)), raw_cov])
    print(
        f"  Covariates: {covariates.shape[1]} columns (1 intercept + {raw_cov.shape[1]} from file)"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run JAMMA

# COMMAND ----------

# JIT warmup on small subset
print("Warming up JAX JIT...", flush=True)
_ = run_lmm_association_jax(
    genotypes=plink_data.genotypes,
    phenotypes=phenotypes,
    kinship=kinship,
    snp_info=snp_info,
    covariates=None,
    lmm_mode=1,
    n_grid=50,
    n_refine=20,
    show_progress=False,
    check_memory=False,
)

# Timed JAMMA run
print(f"\nRunning JAMMA (mode={LMM_MODE})...", flush=True)
t0 = time.perf_counter()
jamma_results = run_lmm_association_jax(
    genotypes=plink_data.genotypes,
    phenotypes=phenotypes,
    kinship=kinship,
    snp_info=snp_info,
    covariates=covariates,
    lmm_mode=LMM_MODE,
    maf_threshold=MAF,
    miss_threshold=MISS,
    n_grid=50,
    n_refine=20,
    show_progress=True,
    check_memory=False,
)
jamma_time = time.perf_counter() - t0
print(
    f"JAMMA: {len(jamma_results)} SNPs in {jamma_time:.2f}s ({len(jamma_results) / jamma_time:.0f} SNPs/sec)"
)

# Save JAMMA results
jamma_out = OUTPUT_DIR / "jamma_results.assoc.txt"
write_assoc_results(jamma_results, jamma_out)
print(f"Saved to {jamma_out}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run GEMMA

# COMMAND ----------

gemma_time = None
gemma_kinship_time = None
gemma_results = None

if RUN_GEMMA:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Compute kinship if no file provided (GEMMA -gk 1)
        if KINSHIP_FILE:
            gemma_kin_path = KINSHIP_FILE
        else:
            print("Computing kinship with GEMMA (-gk 1)...")
            gk_cmd = [
                str(GEMMA_BIN),
                "-bfile",
                BFILE,
                "-gk",
                "1",
                "-o",
                "kinship",
                "-outdir",
                tmpdir,
            ]
            print(f"  Command: {' '.join(gk_cmd)}")
            t0 = time.perf_counter()
            gk_result = subprocess.run(gk_cmd, capture_output=True, text=True)
            gemma_kinship_time = time.perf_counter() - t0

            if gk_result.returncode != 0:
                print(f"GEMMA kinship FAILED (exit {gk_result.returncode}):")
                print(gk_result.stderr[:500])
                RUN_GEMMA = False
            else:
                gemma_kin_path = str(Path(tmpdir) / "kinship.cXX.txt")
                print(f"  GEMMA kinship: {gemma_kinship_time:.2f}s")

        # Step 2: Run LMM with GEMMA's own (or user-provided) kinship
        if not RUN_GEMMA:
            print("Skipping GEMMA LMM (kinship step failed)")
        else:
            cmd = [
                str(GEMMA_BIN),
                "-bfile",
                BFILE,
                "-k",
                gemma_kin_path,
                "-lmm",
                str(LMM_MODE),
                "-maf",
                str(MAF),
                "-miss",
                str(MISS),
                "-o",
                "bench",
                "-outdir",
                tmpdir,
            ]

            # Add covariates if present
            if COVARIATE_FILE:
                cmd.extend(["-c", COVARIATE_FILE])

            print(f"Running GEMMA (mode={LMM_MODE})...")
            print(f"  Command: {' '.join(cmd)}")
            t0 = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True)
            gemma_time = time.perf_counter() - t0

            if result.returncode != 0:
                print(f"GEMMA FAILED (exit {result.returncode}):")
                print(result.stderr[:500])
            else:
                print(f"GEMMA: {gemma_time:.2f}s")
                # Load GEMMA output
                gemma_outfile = Path(tmpdir) / "bench.assoc.txt"
                if gemma_outfile.exists():
                    gemma_results = load_gemma_assoc(gemma_outfile)
                    print(f"  {len(gemma_results)} SNPs")
                    # Copy to output dir
                    gemma_save = OUTPUT_DIR / "gemma_results.assoc.txt"
                    shutil.copy2(gemma_outfile, gemma_save)
                    print(f"  Saved to {gemma_save}")
                else:
                    print(f"  WARNING: Expected output not found at {gemma_outfile}")
                    print(f"  Files in tmpdir: {os.listdir(tmpdir)}")
else:
    print("GEMMA comparison skipped")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Accuracy Comparison

# COMMAND ----------

if gemma_results is not None:
    # Determine p-value field based on LMM mode
    if LMM_MODE == 2:
        p_field = "p_lrt"
    elif LMM_MODE == 3:
        p_field = "p_score"
    else:
        p_field = "p_wald"

    # Match SNPs by rs ID
    j_by_rs = {r.rs: r for r in jamma_results}
    g_by_rs = {r.rs: r for r in gemma_results}
    common = sorted(set(j_by_rs) & set(g_by_rs))

    # P-values
    j_p = np.array([getattr(j_by_rs[rs], p_field) for rs in common])
    g_p = np.array([getattr(g_by_rs[rs], p_field) for rs in common])
    mask = np.isfinite(j_p) & np.isfinite(g_p) & (j_p > 0) & (g_p > 0)
    j_p, g_p = j_p[mask], g_p[mask]
    common_masked = [rs for rs, m in zip(common, mask, strict=True) if m]

    # Spearman correlation on -log10(p)
    rho, _ = spearmanr(-np.log10(j_p), -np.log10(g_p))

    # Significance agreement
    sig_05 = int(np.sum((j_p < 0.05) == (g_p < 0.05)))
    sig_5e8 = int(np.sum((j_p < 5e-8) == (g_p < 5e-8)))

    # Effect direction agreement
    j_beta = np.array([j_by_rs[rs].beta for rs in common_masked])
    g_beta = np.array([g_by_rs[rs].beta for rs in common_masked])
    beta_mask = np.isfinite(j_beta) & np.isfinite(g_beta) & (np.abs(g_beta) > 1e-10)
    dir_agree = float(np.mean(np.sign(j_beta[beta_mask]) == np.sign(g_beta[beta_mask])))

    # Max relative p-value difference
    abs_diff = np.abs(j_p - g_p)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = abs_diff / np.abs(g_p)
        rel = np.where(np.isfinite(rel), rel, 0.0)
    max_p_rel = float(np.max(rel))

    print("=" * 70)
    print("  ACCURACY: JAMMA vs GEMMA")
    print("=" * 70)
    print(f"  P-value field:              {p_field}")
    print(f"  Common SNPs:                {len(j_p)}")
    print(f"  Spearman rho (-log10 p):    {rho:.6f}")
    print(f"  Significance agree (p<0.05):{sig_05}/{len(j_p)}")
    print(f"  Significance agree (p<5e-8):{sig_5e8}/{len(j_p)}")
    print(f"  Effect direction agreement: {dir_agree * 100:.1f}%")
    print(f"  Max relative p diff:        {max_p_rel:.2e}")
    print("=" * 70)
else:
    print("No GEMMA results — accuracy comparison skipped")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Summary

# COMMAND ----------

print("=" * 70)
print("  PERFORMANCE SUMMARY")
print("=" * 70)
print(f"  Dataset:      {BFILE}")
print(f"  Samples:      {plink_data.n_samples}")
print(f"  SNPs:         {plink_data.n_snps}")
print(f"  LMM mode:     {LMM_MODE} ({MODE_NAMES[LMM_MODE]})")
print(f"  Covariates:   {'yes' if COVARIATE_FILE else 'no'}")
print()

# Kinship computation timing
if not KINSHIP_FILE:
    print(f"  JAMMA kinship:  {t_kin:.2f}s")
    if gemma_kinship_time is not None:
        print(f"  GEMMA kinship:  {gemma_kinship_time:.2f}s")
        kin_speedup = gemma_kinship_time / t_kin
        print(
            f"  Kinship speedup:{kin_speedup:.1f}x ({'JAMMA' if kin_speedup > 1 else 'GEMMA'} faster)"
        )
    print()

# LMM timing
print(
    f"  JAMMA LMM:    {jamma_time:.2f}s ({len(jamma_results) / jamma_time:.0f} SNPs/sec)"
)

if gemma_time is not None:
    print(
        f"  GEMMA LMM:    {gemma_time:.2f}s ({len(gemma_results) / gemma_time:.0f} SNPs/sec)"
    )
    speedup = gemma_time / jamma_time
    label = "JAMMA faster" if speedup > 1 else "GEMMA faster"
    print(f"  LMM speedup:  {speedup:.1f}x ({label})")
print("=" * 70)

# Save summary JSON
summary = {
    "dataset": BFILE,
    "n_samples": plink_data.n_samples,
    "n_snps": plink_data.n_snps,
    "lmm_mode": LMM_MODE,
    "maf": MAF,
    "miss": MISS,
    "covariates": bool(COVARIATE_FILE),
    "jamma_lmm_time_s": round(jamma_time, 3),
    "jamma_snps_tested": len(jamma_results),
    "kinship_source": KINSHIP_FILE if KINSHIP_FILE else "computed",
}
if not KINSHIP_FILE:
    summary["jamma_kinship_time_s"] = round(t_kin, 3)
    if gemma_kinship_time is not None:
        summary["gemma_kinship_time_s"] = round(gemma_kinship_time, 3)
if gemma_time is not None:
    summary["gemma_lmm_time_s"] = round(gemma_time, 3)
    summary["lmm_speedup"] = round(gemma_time / jamma_time, 2)
if gemma_results is not None:
    summary["accuracy"] = {
        "spearman_rho": round(rho, 6),
        "sig_agree_p05": f"{sig_05}/{len(j_p)}",
        "sig_agree_p5e8": f"{sig_5e8}/{len(j_p)}",
        "direction_agree": round(dir_agree, 4),
        "max_relative_p_diff": f"{max_p_rel:.2e}",
    }

summary_path = OUTPUT_DIR / "benchmark_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {summary_path}")
