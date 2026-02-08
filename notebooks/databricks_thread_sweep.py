# Databricks notebook source
# MAGIC %md
# MAGIC # JAMMA Thread Count Sweep
# MAGIC
# MAGIC Empirical benchmark of BLAS thread counts for eigendecomp and UT@G rotation.
# MAGIC Tests 1/4/8/16/32/64 threads on configurable matrix size to find optimal MKL configuration.
# MAGIC
# MAGIC **Requires:** Plan 20-01 code (threadpool_limits context managers + JAMMA_BLAS_THREADS env var)
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

# MAGIC %sh # Purge all non-MKL BLAS/LAPACK providers
# MAGIC apt-get purge -y libopenblas* libblas* libatlas* liblapack* 2>/dev/null; echo "Non-MKL BLAS purged"

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

# MAGIC %md
# MAGIC ### Verify NumPy ILP64 + MKL Backend

# COMMAND ----------

import numpy as np
from threadpoolctl import threadpool_info

print("=== NumPy ILP64 Backend Verification ===")
print(f"NumPy version: {np.__version__}")
print(f"NumPy location: {np.__file__}")

# Check build config
try:
    config = np.show_config(mode="dicts")
    blas_info = config.get("Build Dependencies", {}).get("blas", {})
    lapack_info = config.get("Build Dependencies", {}).get("lapack", {})
    blas_name = blas_info.get("name", "unknown")
    lapack_name = lapack_info.get("name", "unknown")
    print(f"  BLAS:   {blas_name}")
    print(f"  LAPACK: {lapack_name}")

    ilp64_ok = "ilp64" in blas_name.lower() and "ilp64" in lapack_name.lower()
    if ilp64_ok:
        print("  ILP64:  CONFIRMED (both BLAS and LAPACK)")
    else:
        print("  WARNING: ILP64 NOT DETECTED. Eigendecomp will fail at >46k samples.")
except Exception as e:
    print(f"  ERROR reading build config: {e}")

# Runtime BLAS check
print("\nBLAS runtime (threadpoolctl):")
for lib in threadpool_info():
    if lib.get("user_api") == "blas":
        print(
            f"  {lib.get('internal_api', 'unknown')}: "
            f"{lib.get('num_threads', '?')} threads, "
            f"prefix={lib.get('prefix', 'N/A')}, "
            f"threading={lib.get('threading_layer', 'unknown')}"
        )

# Numerical sanity check
test_k = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
vals, vecs = np.linalg.eigh(test_k)
expected = np.array([1.0, 3.0])
eigenval_error = np.abs(vals - expected).max()
print(f"\nSanity check: eigenvalue error = {eigenval_error:.2e}")
assert eigenval_error < 1e-10, f"Basic eigendecomp FAILED: error={eigenval_error:.2e}"
print("Sanity check: PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eigendecomp Thread Sweep
# MAGIC
# MAGIC Sweep MKL thread counts for `np.linalg.eigh` (DSYEVD).
# MAGIC Eigendecomp is **memory-bandwidth-bound** in its tridiagonalization step,
# MAGIC so optimal thread count may be lower than total vCPUs.
# MAGIC
# MAGIC **Default size:** 5000 (fast sweep, ~2 min total). Change `N_SAMPLES` to 90000
# MAGIC for production-representative results (~1 hour total).

# COMMAND ----------

import statistics
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

# --- Configuration ---
# Change N_SAMPLES to 90000 for production-representative timing.
# At 5000, each eigendecomp takes ~1-3s (good for verifying thread control works).
# At 90000, each eigendecomp takes ~30-60min (real production measurement).
N_SAMPLES = 5_000
N_REPS = 3  # Repetitions per thread count (median reported)
THREAD_COUNTS = [1, 4, 8, 16, 32, 64]

print(f"Matrix size: {N_SAMPLES:,} x {N_SAMPLES:,}")
print(f"Matrix memory: {N_SAMPLES * N_SAMPLES * 8 / 1e9:.1f} GB")
print(f"Repetitions per thread count: {N_REPS}")
print(f"Thread counts to test: {THREAD_COUNTS}")

mem = psutil.virtual_memory()
physical_cores = psutil.cpu_count(logical=False) or 0
logical_cores = psutil.cpu_count(logical=True) or 0
print(f"\nSystem: {physical_cores} physical cores, {logical_cores} logical CPUs")
print(f"RAM: {mem.available / 1e9:.1f} GB available / {mem.total / 1e9:.1f} GB total")

# Generate symmetric PSD matrix
print(f"\nGenerating {N_SAMPLES:,} x {N_SAMPLES:,} PSD matrix...")
rng = np.random.default_rng(42)
A = rng.standard_normal((N_SAMPLES, N_SAMPLES))
K = A @ A.T  # Symmetric positive semi-definite
del A
print("Matrix ready.")

# COMMAND ----------

# Run eigendecomp sweep
print("=" * 70)
print("EIGENDECOMP THREAD SWEEP")
print("=" * 70)

eigen_results = {}

for n_threads in THREAD_COUNTS:
    timings = []

    for _rep in range(N_REPS):
        with threadpool_limits(limits=n_threads, user_api="blas"):
            # Verify actual thread count inside context manager
            actual_threads = None
            for lib in threadpool_info():
                if lib.get("user_api") == "blas":
                    actual_threads = lib.get("num_threads")
                    break

            t_start = time.perf_counter()
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            t_end = time.perf_counter()

        elapsed = t_end - t_start
        timings.append(elapsed)

        # Free results to avoid memory accumulation
        del eigenvalues, eigenvectors

    median_time = statistics.median(timings)
    eigen_results[n_threads] = {
        "median_s": median_time,
        "all_timings": timings,
        "verified_threads": actual_threads,
    }

    print(
        f"  Threads={n_threads:2d} (actual={actual_threads}): "
        f"median={median_time:.2f}s  "
        f"[{', '.join(f'{t:.2f}' for t in timings)}]"
    )

# Find optimal
optimal_eigen = min(eigen_results, key=lambda t: eigen_results[t]["median_s"])
baseline_32 = eigen_results.get(32, {}).get("median_s", float("inf"))
optimal_time = eigen_results[optimal_eigen]["median_s"]

print(f"\n{'=' * 70}")
print(f"EIGENDECOMP OPTIMAL: {optimal_eigen} threads ({optimal_time:.2f}s)")
if baseline_32 < float("inf"):
    speedup = baseline_32 / optimal_time
    print(
        f"  vs 32 threads: {baseline_32:.2f}s -> {optimal_time:.2f}s ({speedup:.2f}x)"
    )
print(f"{'=' * 70}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### UT@G Rotation Thread Sweep
# MAGIC
# MAGIC Sweep MKL thread counts for dense matrix multiply `UT @ G` (DGEMM).
# MAGIC Rotation is **compute-bound**, so optimal thread count may differ from eigendecomp.
# MAGIC
# MAGIC Uses the eigenvectors from the first eigendecomp as UT, and a random
# MAGIC genotype-shaped matrix as G.

# COMMAND ----------

# Generate UT and G for rotation sweep
# UT: (N_SAMPLES, N_SAMPLES) -- eigenvectors transposed
# G:  (N_SAMPLES, N_SNPS_CHUNK) -- genotype chunk
N_SNPS_CHUNK = 10_000  # Typical JAMMA chunk size

print(
    f"Rotation: UT ({N_SAMPLES:,} x {N_SAMPLES:,}) @ G ({N_SAMPLES:,} x {N_SNPS_CHUNK:,})"
)
print(f"Output: ({N_SAMPLES:,} x {N_SNPS_CHUNK:,})")

# Get eigenvectors for UT (run one eigendecomp to get realistic UT)
with threadpool_limits(limits=32, user_api="blas"):
    _, UT = np.linalg.eigh(K)
UT = np.ascontiguousarray(UT.T)  # Transpose for UT @ G

# Random genotype chunk
G = rng.standard_normal((N_SAMPLES, N_SNPS_CHUNK)).astype(np.float64)
print("Matrices ready.")

# COMMAND ----------

# Run rotation sweep
print("=" * 70)
print("UT@G ROTATION THREAD SWEEP")
print("=" * 70)

rotation_results = {}

for n_threads in THREAD_COUNTS:
    timings = []

    for _rep in range(N_REPS):
        with threadpool_limits(limits=n_threads, user_api="blas"):
            # Verify actual thread count
            actual_threads = None
            for lib in threadpool_info():
                if lib.get("user_api") == "blas":
                    actual_threads = lib.get("num_threads")
                    break

            t_start = time.perf_counter()
            result = np.ascontiguousarray(UT @ G)
            t_end = time.perf_counter()

        elapsed = t_end - t_start
        timings.append(elapsed)
        del result

    median_time = statistics.median(timings)
    rotation_results[n_threads] = {
        "median_s": median_time,
        "all_timings": timings,
        "verified_threads": actual_threads,
    }

    print(
        f"  Threads={n_threads:2d} (actual={actual_threads}): "
        f"median={median_time:.2f}s  "
        f"[{', '.join(f'{t:.2f}' for t in timings)}]"
    )

# Find optimal
optimal_rotation = min(rotation_results, key=lambda t: rotation_results[t]["median_s"])
baseline_32_rot = rotation_results.get(32, {}).get("median_s", float("inf"))
optimal_rot_time = rotation_results[optimal_rotation]["median_s"]

print(f"\n{'=' * 70}")
print(f"ROTATION OPTIMAL: {optimal_rotation} threads ({optimal_rot_time:.2f}s)")
if baseline_32_rot < float("inf"):
    speedup_rot = baseline_32_rot / optimal_rot_time
    print(
        f"  vs 32 threads: {baseline_32_rot:.2f}s -> {optimal_rot_time:.2f}s ({speedup_rot:.2f}x)"
    )
print(f"{'=' * 70}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full JAMMA Pipeline Comparison
# MAGIC
# MAGIC Run the complete JAMMA pipeline (kinship + eigendecomp + LMM) at different
# MAGIC thread counts using the `JAMMA_BLAS_THREADS` env var.
# MAGIC
# MAGIC This exercises the actual `blas_threads()` context managers added in Plan 20-01.
# MAGIC
# MAGIC **Note:** This cell uses the small matrix (N_SAMPLES above). For production
# MAGIC comparison, increase `N_SAMPLES` in the Configuration cell and re-run.

# COMMAND ----------

import gc
import os

from jamma.core import configure_jax

configure_jax(enable_x64=True)

from jamma.kinship import compute_centered_kinship  # noqa: E402
from jamma.lmm.eigen import eigendecompose_kinship  # noqa: E402
from jamma.lmm.runner_jax import run_lmm_association_jax  # noqa: E402

# Generate small dataset for pipeline comparison
N_PIPELINE_SNPS = 1_000
geno_pipeline = rng.standard_normal((N_SAMPLES, N_PIPELINE_SNPS)).astype(np.float32)
pheno_pipeline = rng.standard_normal(N_SAMPLES).astype(np.float64)
snp_info = [
    {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
    for j in range(N_PIPELINE_SNPS)
]

# Only test a subset of thread counts for full pipeline (slower)
PIPELINE_THREADS = [8, 16, 32]

print("=" * 70)
print(f"FULL PIPELINE COMPARISON (n={N_SAMPLES:,}, SNPs={N_PIPELINE_SNPS:,})")
print("=" * 70)

pipeline_results = {}

for n_threads in PIPELINE_THREADS:
    os.environ["JAMMA_BLAS_THREADS"] = str(n_threads)

    t_start = time.perf_counter()

    K_pipe = compute_centered_kinship(geno_pipeline)
    eigenvalues_pipe, eigenvectors_pipe = eigendecompose_kinship(K_pipe)

    _ = run_lmm_association_jax(
        genotypes=geno_pipeline,
        phenotypes=pheno_pipeline,
        kinship=K_pipe,
        snp_info=snp_info,
        eigenvalues=eigenvalues_pipe,
        eigenvectors=eigenvectors_pipe,
        show_progress=False,
    )

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    pipeline_results[n_threads] = elapsed
    print(f"  JAMMA_BLAS_THREADS={n_threads:2d}: {elapsed:.2f}s total")

    del K_pipe, eigenvalues_pipe, eigenvectors_pipe
    gc.collect()

# Clean up env var
os.environ.pop("JAMMA_BLAS_THREADS", None)

optimal_pipeline = min(pipeline_results, key=pipeline_results.get)
print(
    f"\nPipeline optimal: {optimal_pipeline} threads ({pipeline_results[optimal_pipeline]:.2f}s)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Large-Scale Sweep (Optional)
# MAGIC
# MAGIC To run production-representative benchmarks at 90k samples:
# MAGIC 1. Change `N_SAMPLES = 90_000` in the Configuration cell above
# MAGIC 2. Re-run from the Configuration cell onwards
# MAGIC 3. **Expected runtime:** ~1 hour for eigendecomp sweep (6 thread counts x 3 reps x ~3 min each)
# MAGIC 4. Rotation sweep will be faster (~15 min total)
# MAGIC
# MAGIC At 90k scale, eigendecomp memory = ~65 GB per matrix. Ensure the VM has
# MAGIC sufficient RAM (512 GB recommended, 256 GB minimum).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary and Recommendation

# COMMAND ----------

# Summary table
print("=" * 70)
print("THREAD SWEEP SUMMARY")
print("=" * 70)

print(f"\n{'Operation':<20} {'Threads':<10} {'Time (s)':<12} {'Speedup vs 32'}")
print("-" * 60)

# Eigendecomp results
for t in THREAD_COUNTS:
    r = eigen_results[t]
    speedup_str = ""
    if baseline_32 < float("inf") and baseline_32 > 0:
        speedup = baseline_32 / r["median_s"]
        speedup_str = f"{speedup:.2f}x"
    verified = r["verified_threads"]
    marker = " <-- OPTIMAL" if t == optimal_eigen else ""
    print(
        f"  eigendecomp       {t:>2d} ({verified:>2d})   {r['median_s']:>8.2f}     {speedup_str}{marker}"
    )

print()

# Rotation results
for t in THREAD_COUNTS:
    r = rotation_results[t]
    speedup_str = ""
    if baseline_32_rot < float("inf") and baseline_32_rot > 0:
        speedup = baseline_32_rot / r["median_s"]
        speedup_str = f"{speedup:.2f}x"
    verified = r["verified_threads"]
    marker = " <-- OPTIMAL" if t == optimal_rotation else ""
    print(
        f"  UT@G rotation     {t:>2d} ({verified:>2d})   {r['median_s']:>8.2f}     {speedup_str}{marker}"
    )

print()

# Pipeline results
if pipeline_results:
    baseline_pipe_32 = pipeline_results.get(32, float("inf"))
    for t in sorted(pipeline_results):
        pipe_time = pipeline_results[t]
        speedup_str = ""
        if baseline_pipe_32 < float("inf") and baseline_pipe_32 > 0:
            speedup = baseline_pipe_32 / pipe_time
            speedup_str = f"{speedup:.2f}x"
        marker = " <-- OPTIMAL" if t == optimal_pipeline else ""
        print(
            f"  full pipeline     {t:>2d}        {pipe_time:>8.2f}     {speedup_str}{marker}"
        )

print(f"\n{'=' * 70}")
print("RECOMMENDATION")
print(f"{'=' * 70}")
print(f"  Eigendecomp optimal:  {optimal_eigen} threads")
print(f"  Rotation optimal:     {optimal_rotation} threads")
print(f"  Pipeline optimal:     {optimal_pipeline} threads")
print()

# Check if eigendecomp and rotation agree
if optimal_eigen == optimal_rotation:
    print(f"  Both operations optimal at {optimal_eigen} threads.")
    print(f"  Recommendation: set JAMMA_BLAS_THREADS={optimal_eigen}")
else:
    print("  Eigendecomp and rotation prefer different thread counts.")
    print("  Since eigendecomp dominates runtime (54% of total),")
    print(f"  recommendation: set JAMMA_BLAS_THREADS={optimal_eigen}")
    print("  (optimise for the bottleneck)")

print(f"\n  To apply: export JAMMA_BLAS_THREADS={optimal_eigen}")
print(f"  Or modify jamma.core.threading default to {optimal_eigen}")

# COMMAND ----------

# Cleanup
del K, UT, G, geno_pipeline, pheno_pipeline, snp_info
gc.collect()
from jamma.core import cleanup_memory  # noqa: E402

cleanup_memory(clear_jax=True, verbose=True)
