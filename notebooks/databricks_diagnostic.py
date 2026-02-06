# Databricks notebook source
# MAGIC %md
# MAGIC # JAMMA Phase 19: Diagnostic Notebook
# MAGIC
# MAGIC Captures three empirical measurements for v1.4 performance optimization:
# MAGIC 1. MKL thread count before/after JAMMA import (PROF-03)
# MAGIC 2. LAPACK driver identification via MKL_VERBOSE=1 (PROF-04)
# MAGIC 3. Baseline 90k timing breakdown (PROF-01, PROF-02)
# MAGIC
# MAGIC **Cluster:** Memory-optimized instance with 512GB+ RAM, DBR 15.4 LTS+

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Dependencies
# MAGIC
# MAGIC NumPy with MKL ILP64 (64-bit integers) for stable eigendecomp at 100k+ scale.
# MAGIC - OpenBLAS segfaults on large matrices (50k+)
# MAGIC - MKL LP64 hits int32 overflow at ~46k samples
# MAGIC - **MKL ILP64** uses 64-bit integers - no practical limit
# MAGIC
# MAGIC **IMPORTANT:** Run `dbutils.library.restartPython()` after pip installs.

# COMMAND ----------

# MAGIC %sh # Purge all non-MKL BLAS/LAPACK providers and system numpy
# MAGIC apt-get purge -y libopenblas* libblas* libatlas* liblapack* python3-numpy 2>/dev/null; echo "Non-MKL BLAS purged"

# COMMAND ----------

# Install MKL runtime — provides libmkl_def.so.2 and other computational kernels
# that auditwheel can't bundle (loaded via dlopen, not ELF NEEDED)
# MAGIC %pip install mkl

# COMMAND ----------

# Install numpy with MKL ILP64 from forked wheel repository
# ILP64 uses 64-bit integers, supporting matrices >46k x 46k
# MAGIC %pip install numpy --extra-index-url https://michael-denyer.github.io/numpy-mkl --force-reinstall --upgrade

# COMMAND ----------

# Install jamma dependencies (except numpy which is ILP64 above)
# MAGIC %pip install psutil loguru threadpoolctl jax jaxlib jaxtyping typer progressbar2 bed-reader

# COMMAND ----------

# Install jamma WITHOUT dependencies (numpy already installed with ILP64)
# MAGIC %pip install git+https://github.com/michael-denyer/jamma.git --no-deps

# COMMAND ----------

# Restart Python kernel to load new packages - MUST run this cell
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# IMPORTANT: Set MKL_VERBOSE=1 BEFORE any numpy import
# This enables LAPACK driver identification (PROF-04)
import os

os.environ["MKL_VERBOSE"] = "1"

# COMMAND ----------

# PROF-03 part 1: Capture threadpool_info BEFORE jamma import
import json

import numpy as np
from threadpoolctl import threadpool_info

print("=" * 60)
print("THREADPOOL STATE: BEFORE JAMMA IMPORT")
print("=" * 60)
info_before = threadpool_info()
for lib in info_before:
    if lib.get("user_api") == "blas":
        print(f"  Backend:         {lib.get('internal_api', 'unknown')}")
        print(f"  Threads:         {lib.get('num_threads', '?')}")
        print(f"  Threading layer: {lib.get('threading_layer', 'unknown')}")
        print(f"  Library:         {lib.get('filepath', 'unknown')}")
print(f"\nFull threadpool_info:\n{json.dumps(info_before, indent=2)}")

# Also capture env vars for comparison
for var in ["MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    print(f"  {var} = {os.environ.get(var, '<NOT SET>')}")

# COMMAND ----------

# Import jamma (triggers _pin_blas_threads(1) via jamma.core.jax_config module load)
from jamma.core import configure_jax

configure_jax(enable_x64=True)

# COMMAND ----------

# PROF-03 part 2: Capture threadpool_info AFTER jamma import
print("=" * 60)
print("THREADPOOL STATE: AFTER JAMMA IMPORT")
print("=" * 60)
info_after = threadpool_info()
for lib in info_after:
    if lib.get("user_api") == "blas":
        print(f"  Backend:         {lib.get('internal_api', 'unknown')}")
        print(f"  Threads:         {lib.get('num_threads', '?')}")
        print(f"  Threading layer: {lib.get('threading_layer', 'unknown')}")
        print(f"  Library:         {lib.get('filepath', 'unknown')}")
print(f"\nFull threadpool_info:\n{json.dumps(info_after, indent=2)}")

# Show env vars after import
for var in ["MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    print(f"  {var} = {os.environ.get(var, '<NOT SET>')}")

# Diagnose thread pinning bug
before_threads = next(
    (lib["num_threads"] for lib in info_before if lib.get("user_api") == "blas"), None
)
after_threads = next(
    (lib["num_threads"] for lib in info_after if lib.get("user_api") == "blas"), None
)
print(f"\n{'=' * 60}")
print(
    f"DIAGNOSIS: Thread count {'CHANGED' if before_threads != after_threads else 'UNCHANGED'} by JAMMA import"
)
print(f"  Before: {before_threads} threads")
print(f"  After:  {after_threads} threads")
if before_threads != after_threads and after_threads == 1:
    print(
        "  CONCLUSION: Thread-pinning bug IS ACTIVE. _pin_blas_threads(1) reduced MKL threads."
    )
    print(
        "  Phase 20 fix will deliver significant speedup (potentially 8-32x for eigendecomp)."
    )
elif before_threads == after_threads and after_threads == 1:
    print("  CONCLUSION: Threads were ALREADY 1 before JAMMA import.")
    print("  Databricks may be setting MKL_NUM_THREADS=1 via cluster config.")
    print(
        "  Check: spark.conf.get('spark.executor.extraJavaOptions') and cluster env vars."
    )
elif before_threads == after_threads and after_threads > 1:
    print("  CONCLUSION: Thread-pinning bug is INACTIVE (setdefault was a no-op).")
    print("  MKL is already running multi-threaded. Phase 20 gains capped at ~7%.")
print(f"{'=' * 60}")

# COMMAND ----------

# PROF-04: LAPACK driver identification (MKL_VERBOSE=1 was set in Cell 3)
# MKL verbose output goes to stderr, which Databricks captures
import sys

print("=" * 60)
print("LAPACK DRIVER IDENTIFICATION (MKL_VERBOSE=1)")
print("=" * 60)
print("Running numpy.linalg.eigh on 500x500 matrix...")
print("Look for DSYEVD or DSYEVR in the MKL_VERBOSE output below:")
print()

sys.stderr.flush()

K_test = np.random.default_rng(42).standard_normal((500, 500))
K_test = K_test @ K_test.T
vals, vecs = np.linalg.eigh(K_test)

sys.stderr.flush()
print()
print(f"Eigenvalues range: [{vals.min():.6g}, {vals.max():.6g}]")
print("If no MKL_VERBOSE output appeared above, MKL_VERBOSE was not active.")
print(
    "This can happen if numpy was imported before Cell 3 ran (restart kernel and re-run)."
)

# Disable MKL_VERBOSE for the 90k run (avoid huge log output)
os.environ.pop("MKL_VERBOSE", None)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify NumPy ILP64 + MKL Backend

# COMMAND ----------

# ILP64 verification (same checks as databricks_benchmark.py)
print("=== NumPy ILP64 Backend Verification ===")
print(f"NumPy version: {np.__version__}")
print(f"NumPy location: {np.__file__}")

# --- Step 1: Check build config for ILP64 ---
ilp64_ok = False
try:
    config = np.show_config(mode="dicts")
    blas_info = config.get("Build Dependencies", {}).get("blas", {})
    lapack_info = config.get("Build Dependencies", {}).get("lapack", {})
    blas_name = blas_info.get("name", "unknown")
    lapack_name = lapack_info.get("name", "unknown")
    print(f"  BLAS:   {blas_name}")
    print(f"  LAPACK: {lapack_name}")

    blas_ilp64 = "ilp64" in blas_name.lower()
    lapack_ilp64 = "ilp64" in lapack_name.lower()
    ilp64_ok = blas_ilp64 and lapack_ilp64

    if ilp64_ok:
        print("  ILP64:  CONFIRMED (both BLAS and LAPACK)")
    else:
        print(f"  ILP64:  NOT DETECTED (BLAS={blas_ilp64}, LAPACK={lapack_ilp64})")
        print("  WARNING: Eigendecomp WILL FAIL at >46k samples without ILP64.")
        print("  Troubleshooting:")
        print("    1. Check pip install output above for errors")
        print(
            "    2. Verify wheel was fetched from michael-denyer.github.io (not PyPI)"
        )
        print("    3. Run: pip show numpy | grep Location")
        print("    4. Confirm the fork index has wheels for this Python version")
except Exception as e:
    print(f"  ERROR reading build config: {e}")
    np.show_config()

# --- Step 2: Check runtime BLAS via threadpoolctl ---
print("\nBLAS runtime (threadpoolctl):")
blas_libs = [lib for lib in threadpool_info() if lib.get("user_api") == "blas"]
detected_mkl = False
if blas_libs:
    for lib in blas_libs:
        internal_api = lib.get("internal_api", "unknown")
        filepath = lib.get("filepath", "unknown")
        num_threads = lib.get("num_threads", "?")
        threading_layer = lib.get("threading_layer", "unknown")
        print(f"  Backend: {internal_api}")
        print(f"  Library: {filepath}")
        print(f"  Threads: {num_threads}")
        print(f"  Threading: {threading_layer}")
        if "mkl" in internal_api.lower() or "mkl" in filepath.lower():
            detected_mkl = True
else:
    print("  No BLAS libraries detected by threadpoolctl!")
    print(
        "  This usually means numpy was built without BLAS or the library failed to load."
    )
    print("  Check: python -c 'import numpy; numpy.show_config()'")

if detected_mkl:
    print("\nMKL runtime: DETECTED")
else:
    print("\nMKL runtime: NOT DETECTED")
    print("  Eigendecomp may segfault at 50k+ samples with OpenBLAS.")
    print("  If ILP64 build config was confirmed above but MKL isn't loading,")
    print("  check if mkl-service is installed: pip show mkl-service")

# --- Step 3: Numerical sanity check ---
# This catches LP64 mismatch: garbage eigenvalues when ILP64 header but LP64 runtime
print("\nNumerical sanity check:")
test_k = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
vals, vecs = np.linalg.eigh(test_k)
expected = np.array([1.0, 3.0])
eigenval_error = np.abs(vals - expected).max()
print(f"  Eigenvalues: {vals} (expected {expected})")
print(f"  Error: {eigenval_error:.2e}")
assert eigenval_error < 1e-10, f"Basic eigendecomp FAILED: error={eigenval_error:.2e}"

# Reconstruction error catches the LP64/ILP64 mismatch that gave error=6.54e+01
M = np.random.default_rng(42).standard_normal((100, 100))
M = M @ M.T
evals, evecs = np.linalg.eigh(M)
reconstructed = evecs @ np.diag(evals) @ evecs.T
recon_error = np.abs(M - reconstructed).max()
print(f"  100x100 reconstruction error: {recon_error:.2e}")
assert recon_error < 1e-8, (
    f"Eigendecomp reconstruction error {recon_error:.2e} too large. "
    "This usually means ILP64 headers but LP64 runtime (symbol suffix mismatch). "
    "See: https://github.com/michael-denyer/numpy-mkl build notes."
)
print("  Sanity checks: PASSED")

# --- Summary ---
print(f"\n{'=' * 40}")
if ilp64_ok and detected_mkl and recon_error < 1e-8:
    print("ILP64 + MKL: READY for large-scale eigendecomp")
elif not ilp64_ok:
    print("WARNING: ILP64 NOT confirmed - 46k sample limit applies")
elif not detected_mkl:
    print("WARNING: MKL not detected at runtime - stability risk at 50k+")
print(f"{'=' * 40}")

# COMMAND ----------

# Generate 90k synthetic data
import gc
import time

import psutil

print("=" * 60)
print("GENERATING 90K SYNTHETIC DATA")
print("=" * 60)

N_SAMPLES = 90_000
N_SNPS = 90_000
N_CAUSAL = 100
HERITABILITY = 0.5
SEED = 42

rng = np.random.default_rng(SEED)

# Genotype matrix (float32 for memory efficiency during generation)
t_gen_start = time.perf_counter()
mafs = rng.uniform(0.05, 0.5, N_SNPS)
chunk_size = 10_000
genotypes = np.zeros((N_SAMPLES, N_SNPS), dtype=np.float32)
for i in range(0, N_SNPS, chunk_size):
    end = min(i + chunk_size, N_SNPS)
    p = mafs[i:end][np.newaxis, :]
    u = rng.random((N_SAMPLES, end - i))
    genotypes[:, i:end] = np.where(
        u < (1 - p) ** 2, 0, np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2)
    ).astype(np.float32)

# Phenotype
causal_idx = rng.choice(N_SNPS, N_CAUSAL, replace=False)
true_betas = rng.standard_normal(N_CAUSAL)
G_causal = genotypes[:, causal_idx]
G_causal_std = (G_causal - G_causal.mean(axis=0)) / (G_causal.std(axis=0) + 1e-8)
genetic_value = G_causal_std @ true_betas
var_g = np.var(genetic_value)
var_e = var_g * (1 - HERITABILITY) / HERITABILITY
noise = rng.standard_normal(N_SAMPLES) * np.sqrt(var_e)
phenotype = genetic_value + noise
phenotype = ((phenotype - phenotype.mean()) / phenotype.std()).astype(np.float64)

t_gen_end = time.perf_counter()
geno_gb = genotypes.nbytes / 1e9
print(
    f"Generated {N_SAMPLES:,} x {N_SNPS:,} in {t_gen_end - t_gen_start:.1f}s ({geno_gb:.1f}GB)"
)

mem = psutil.virtual_memory()
print(f"RAM: {mem.available / 1e9:.1f}GB available / {mem.total / 1e9:.1f}GB total")
print(f"RSS: {psutil.Process().memory_info().rss / 1e9:.1f}GB")

# COMMAND ----------

# Kinship + eigendecomp (timed) - 90k baseline
from jamma.kinship import compute_centered_kinship
from jamma.lmm.eigen import eigendecompose_kinship

print("=" * 60)
print("KINSHIP + EIGENDECOMP (90K BASELINE)")
print("=" * 60)

t_kinship_start = time.perf_counter()
K = compute_centered_kinship(genotypes)
t_kinship_end = time.perf_counter()
print(f"Kinship: {t_kinship_end - t_kinship_start:.1f}s")

mem = psutil.virtual_memory()
print(
    f"RAM after kinship: {mem.available / 1e9:.1f}GB available, RSS={psutil.Process().memory_info().rss / 1e9:.1f}GB"
)

t_eigen_start = time.perf_counter()
eigenvalues, eigenvectors = eigendecompose_kinship(K)
t_eigen_end = time.perf_counter()
print(f"Eigendecomp: {t_eigen_end - t_eigen_start:.1f}s")
print(f"Eigenvalues: min={eigenvalues.min():.6g}, max={eigenvalues.max():.6g}")

# Reconstruction sanity check
k = min(50, N_SAMPLES)
recon = (eigenvectors[:k, :] * eigenvalues) @ eigenvectors[:k, :].T
recon_error = np.abs(K[:k, :k] - recon).max()
print(f"Reconstruction error (50x50): {recon_error:.2e}")
assert recon_error < 1e-6, f"Eigendecomp failed: reconstruction error {recon_error:.2e}"

# COMMAND ----------

# LMM association baseline with timing breakdown (PROF-01, PROF-02)
from jamma.lmm.runner_jax import run_lmm_association_jax

print("=" * 60)
print("LMM ASSOCIATION BASELINE (90K, WALD TEST)")
print("=" * 60)
print(
    "The timing breakdown from runner_streaming.py will appear in the log output below."
)
print()

# Generate SNP info
snp_info = [
    {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
    for j in range(N_SNPS)
]

gc.collect()
mem = psutil.virtual_memory()
print(
    f"RAM before LMM: {mem.available / 1e9:.1f}GB available, RSS={psutil.Process().memory_info().rss / 1e9:.1f}GB"
)

t_lmm_start = time.perf_counter()
results = run_lmm_association_jax(
    genotypes=genotypes,
    phenotypes=phenotype,
    kinship=K,
    snp_info=snp_info,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    show_progress=True,
    lmm_mode=1,
)
t_lmm_end = time.perf_counter()

print(f"\nLMM total: {t_lmm_end - t_lmm_start:.1f}s")
print(f"Results: {len(results)} SNPs tested")
if results:
    p_vals = [
        r.p_wald for r in results if r.p_wald is not None and not np.isnan(r.p_wald)
    ]
    if p_vals:
        print(f"P-value range: [{min(p_vals):.2e}, {max(p_vals):.2e}]")
        n_sig = sum(1 for p in p_vals if p < 5e-8)
        print(f"Genome-wide significant (p < 5e-8): {n_sig}")

# COMMAND ----------

# Phase 19 diagnostic summary
print("\n" + "=" * 60)
print("PHASE 19 DIAGNOSTIC SUMMARY")
print("=" * 60)
print(f"\n{'Measurement':<35} {'Value'}")
print("-" * 60)
print(f"{'MKL threads (before import)':<35} {before_threads}")
print(f"{'MKL threads (after import)':<35} {after_threads}")
print(
    f"{'Thread pinning active?':<35} {'YES' if before_threads != after_threads else 'NO'}"
)
print(f"{'Kinship time':<35} {t_kinship_end - t_kinship_start:.1f}s")
print(f"{'Eigendecomp time':<35} {t_eigen_end - t_eigen_start:.1f}s")
print(f"{'LMM total time':<35} {t_lmm_end - t_lmm_start:.1f}s")
print(
    f"{'Total (kinship+eigen+LMM)':<35} {(t_kinship_end - t_kinship_start) + (t_eigen_end - t_eigen_start) + (t_lmm_end - t_lmm_start):.1f}s"
)
print("\n(See LMM log output above for the 6-phase timing breakdown)")
print("(See MKL_VERBOSE output above for LAPACK driver identification)")

# COMMAND ----------

# Cleanup
del genotypes, K, eigenvalues, eigenvectors, results, phenotype, snp_info
gc.collect()
from jamma.core import cleanup_memory

cleanup_memory(clear_jax=True, verbose=True)
