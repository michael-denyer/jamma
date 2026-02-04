# Databricks notebook source
# MAGIC %md
# MAGIC # NumPy/MKL Eigendecomposition Benchmark
# MAGIC
# MAGIC Tests numpy.linalg.eigh at scale (10k, 50k, 100k samples).
# MAGIC NumPy uses LAPACK via MKL on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install NumPy with MKL
# MAGIC
# MAGIC Databricks uses OpenBLAS by default. MKL is needed for stable eigendecomp at scale.
# MAGIC OpenBLAS can segfault on large matrices (50k+) due to threading/memory issues.
# MAGIC
# MAGIC **Approach**: Use pre-built MKL wheels from urob/numpy-mkl repository.

# COMMAND ----------

# Install pre-built numpy/scipy with MKL from urob's wheel repository
# MAGIC %pip install numpy scipy --extra-index-url https://urob.github.io/numpy-mkl --force-reinstall --upgrade

# COMMAND ----------

# MAGIC %pip install psutil loguru threadpoolctl

# COMMAND ----------

# Install jamma and dependencies
# MAGIC %pip install git+https://github.com/michael-denyer/jamma.git

# COMMAND ----------

# Restart to pick up new packages
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify NumPy Backend

# COMMAND ----------

import numpy as np
from threadpoolctl import threadpool_info

print("=== NumPy Backend Verification ===")
print(f"NumPy version: {np.__version__}")

# Check BLAS config via threadpoolctl (most reliable method)
print("\nBLAS configuration (threadpoolctl):")
blas_info = [lib for lib in threadpool_info() if lib.get("user_api") == "blas"]
detected_mkl = False
if blas_info:
    for lib in blas_info:
        internal_api = lib.get("internal_api", "unknown")
        filepath = lib.get("filepath", "unknown")
        num_threads = lib.get("num_threads", "?")
        print(f"  Backend: {internal_api}")
        print(f"  Library: {filepath}")
        print(f"  Threads: {num_threads}")
        if "mkl" in internal_api.lower() or "mkl" in filepath.lower():
            detected_mkl = True

if detected_mkl:
    print("\n✓ MKL detected - 64-bit safe, multi-threaded")
else:
    print("\n⚠ MKL NOT detected - eigendecomp may be unstable at scale")
    print("  Check build output above for errors")

# Also show numpy's built-in config
print("\nNumPy build config:")
try:
    config = np.show_config(mode="dicts")
    blas_info = config.get("Build Dependencies", {}).get("blas", {})
    lapack_info = config.get("Build Dependencies", {}).get("lapack", {})
    print(f"  BLAS: {blas_info.get('name', 'unknown')}")
    print(f"  LAPACK: {lapack_info.get('name', 'unknown')}")
except Exception:
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
    print(
        f"  Available memory before: {psutil.virtual_memory().available / 1e9:.1f} GB"
    )
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
        print(f"  Eigenvector shape: {vecs.shape}")
        result = {"time": elapsed, "n_samples": n_samples, "success": True}
    except Exception as e:
        import traceback

        elapsed = time.perf_counter() - start
        error_str = str(e) if str(e) else "(no message)"
        print(f"  FAILED after {elapsed:.2f}s")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {error_str}")
        print(f"  Full traceback:\n{traceback.format_exc()}")
        avail_after = psutil.virtual_memory().available / 1e9
        print(f"  Available memory after: {avail_after:.1f} GB")
        result = {
            "time": None,
            "n_samples": n_samples,
            "success": False,
            "error": f"{type(e).__name__}: {error_str}",
        }

    del K
    gc.collect()
    return result


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run Benchmarks

# COMMAND ----------

# 10k warmup - always runs (only ~2.4GB peak)
available_gb = psutil.virtual_memory().available / 1e9
matrix_gb = (10_000**2) * 8 / 1e9
numpy_peak = 3 * matrix_gb  # K + U + workspace ≈ 2.4 GB

print("=" * 60)
print("10k BENCHMARK")
print("=" * 60)
print(f"10k: matrix={matrix_gb:.1f}GB, available={available_gb:.0f}GB")
print(f"  NumPy peak ≈ {numpy_peak:.1f}GB")

results_10k = benchmark_numpy(10_000)

# COMMAND ----------

# 50k - check memory first
available_gb = psutil.virtual_memory().available / 1e9
matrix_gb = (50_000**2) * 8 / 1e9
numpy_peak = 3 * matrix_gb  # K + U + workspace ≈ 60 GB

print("=" * 60)
print("50k BENCHMARK")
print("=" * 60)
print(f"50k: matrix={matrix_gb:.0f}GB, available={available_gb:.0f}GB")
print(f"  NumPy peak ≈ {numpy_peak:.0f}GB")

# Initialize with crash marker - if this persists, cell crashed before completing
results_50k = {
    "n_samples": 50_000,
    "success": False,
    "error": "Cell crashed (OOM kill or segfault) - check driver logs",
}

if available_gb > numpy_peak * 1.1:
    results_50k = benchmark_numpy(50_000)
else:
    print("Skipping 50k - insufficient memory")
    results_50k = {
        "n_samples": 50_000,
        "success": False,
        "error": f"Insufficient memory: need {numpy_peak:.0f}GB, have {available_gb:.0f}GB",  # noqa: E501
    }

# COMMAND ----------

# 100k - check memory first
available_gb = psutil.virtual_memory().available / 1e9
matrix_gb = (100_000**2) * 8 / 1e9
numpy_peak = 3 * matrix_gb  # K + U + workspace ≈ 240 GB

print(f"100k: matrix={matrix_gb:.0f}GB, available={available_gb:.0f}GB")
print(f"  NumPy peak ≈ {numpy_peak:.0f}GB")

# Initialize with crash marker
results_100k = {
    "n_samples": 100_000,
    "success": False,
    "error": "Cell crashed (OOM kill or segfault) - check driver logs",
}

if available_gb > numpy_peak * 1.1:
    results_100k = benchmark_numpy(100_000)
else:
    print("Skipping 100k - insufficient memory")
    results_100k = {
        "n_samples": 100_000,
        "success": False,
        "error": f"Insufficient memory: need {numpy_peak:.0f}GB, have {available_gb:.0f}GB",  # noqa: E501
    }

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
        error_msg = res.get("error", "unknown error")
        print(f"{scale:<8} {res['n_samples']:<12,} {'N/A':<12} {'FAILED':<10}")
        print(f"         Error: {error_msg}")

print("\nConclusion:")
print("- NumPy/MKL eigendecomposition via numpy.linalg.eigh")
print("- Memory: ~3x matrix size (K + U + workspace)")
print("- 100k samples needs ~240 GB RAM")
