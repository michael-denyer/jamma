# Databricks notebook source
# MAGIC %md
# MAGIC # NumPy/MKL Eigendecomposition Benchmark
# MAGIC
# MAGIC Tests numpy.linalg.eigh at scale (10k, 50k, 100k samples).
# MAGIC NumPy uses LAPACK via MKL on Databricks.

# COMMAND ----------

# MAGIC %pip install psutil numpy loguru

# COMMAND ----------

# MAGIC %pip install --force-reinstall git+https://github.com/michael-denyer/jamma.git

# COMMAND ----------

# Restart to pick up new packages
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify NumPy Backend

# COMMAND ----------

import numpy as np

print("=== NumPy Backend Verification ===")
print(f"NumPy version: {np.__version__}")

# Check BLAS config - verify MKL is being used
print("\nBLAS configuration:")
try:
    config = np.show_config(mode="dicts")
    blas_info = config.get("Build Dependencies", {}).get("blas", {})
    lapack_info = config.get("Build Dependencies", {}).get("lapack", {})
    print(f"  BLAS: {blas_info.get('name', 'unknown')}")
    print(f"  LAPACK: {lapack_info.get('name', 'unknown')}")

    # Verify MKL
    blas_name = blas_info.get("name", "").lower()
    if "mkl" in blas_name:
        print("\n✓ MKL detected - 64-bit safe, multi-threaded")
    elif "openblas" in blas_name:
        print("\n⚠ OpenBLAS detected - may need thread limiting for large matrices")
    elif "accelerate" in blas_name:
        print("\n✓ Accelerate detected (macOS) - 64-bit safe")
    else:
        print(f"\n? Unknown BLAS: {blas_name}")
except Exception:
    # Fallback for older numpy
    np.show_config()

# Quick sanity check
test_k = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
vals, vecs = np.linalg.eigh(test_k)
print(f"\nTest eigenvalues: {vals}")
print("NumPy working!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Benchmark Function

# COMMAND ----------

import gc
import time

import psutil


def benchmark_numpy(n_samples: int):
    """Benchmark numpy.linalg.eigh (uses MKL on Databricks).

    Args:
        n_samples: Matrix dimension
    """
    print(f"\n{'='*60}")
    print(f"NumPy/MKL Benchmark: {n_samples:,} x {n_samples:,}")
    matrix_gb = n_samples * n_samples * 8 / 1e9
    print(f"Matrix memory: {matrix_gb:.1f} GB")
    print(f"{'='*60}")

    np.random.seed(42)
    A = np.random.randn(n_samples, n_samples).astype(np.float64)
    K = A @ A.T / n_samples
    del A
    gc.collect()

    mem_before = psutil.Process().memory_info().rss / 1e9
    print(f"RSS before eigendecomp: {mem_before:.1f} GB")

    print("\nRunning numpy.linalg.eigh...")
    start = time.perf_counter()
    try:
        vals, vecs = np.linalg.eigh(K)
        elapsed = time.perf_counter() - start
        mem_after = psutil.Process().memory_info().rss / 1e9
        print(f"  Time: {elapsed:.2f}s")
        print(
            f"  RSS after: {mem_after:.1f} GB (delta: {mem_after - mem_before:.1f} GB)"
        )
        print(f"  Eigenvalue range: [{vals.min():.6f}, {vals.max():.6f}]")
        result = {"time": elapsed, "n_samples": n_samples, "success": True}
    except Exception as e:
        print(f"  FAILED: {e}")
        result = {
            "time": None,
            "n_samples": n_samples,
            "success": False,
            "error": str(e),
        }

    del K
    gc.collect()
    return result


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run Benchmarks

# COMMAND ----------

# 10k warmup
print("=" * 60)
print("10k BENCHMARK")
print("=" * 60)
results_10k = benchmark_numpy(10_000)

# COMMAND ----------

# 50k
print("=" * 60)
print("50k BENCHMARK")
print("=" * 60)
results_50k = benchmark_numpy(50_000)

# COMMAND ----------

# 100k - check memory first
available_gb = psutil.virtual_memory().available / 1e9
matrix_gb = (100_000**2) * 8 / 1e9
numpy_peak = 3 * matrix_gb  # K + U + workspace ≈ 240 GB

print(f"100k: matrix={matrix_gb:.0f}GB, available={available_gb:.0f}GB")
print(f"  NumPy peak ≈ {numpy_peak:.0f}GB")

results_100k = None
if available_gb > numpy_peak * 1.1:
    results_100k = benchmark_numpy(100_000)
else:
    print("Skipping 100k - insufficient memory")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY: NumPy/MKL Eigendecomposition")
print("=" * 60)

all_results = [
    ("10k", results_10k),
    ("50k", results_50k),
    ("100k", results_100k),
]

print(f"\n{'Scale':<8} {'Samples':<12} {'Time (s)':<12} {'Status':<10}")
print("-" * 44)
for scale, res in all_results:
    if res is None:
        print(f"{scale:<8} {'N/A':<12} {'N/A':<12} {'skipped':<10}")
    elif res.get("success"):
        print(f"{scale:<8} {res['n_samples']:<12,} {res['time']:<12.2f} {'OK':<10}")
    else:
        print(f"{scale:<8} {res['n_samples']:<12,} {'N/A':<12} {'FAILED':<10}")

print("\nConclusion:")
print("- NumPy/MKL eigendecomposition via numpy.linalg.eigh")
print("- Memory: ~3x matrix size (K + U + workspace)")
print("- 100k samples needs ~240 GB RAM")
