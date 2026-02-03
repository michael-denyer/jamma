# Databricks notebook source
# MAGIC %md
# MAGIC # Eigendecomposition Stress Test
# MAGIC
# MAGIC Focused test for eigendecomposition at 100k+ scale.
# MAGIC Tests MKL vs OpenBLAS and thread limiting strategies.
# MAGIC
# MAGIC Run this notebook to diagnose SIGSEGV crashes without
# MAGIC regenerating all benchmark data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install MKL (Intel Math Kernel Library)
# MAGIC
# MAGIC MKL is stable with multi-threading unlike OpenBLAS which
# MAGIC segfaults at 100k+ scale.

# COMMAND ----------

# Install MKL from conda-forge
# This replaces OpenBLAS as the BLAS/LAPACK backend for numpy/scipy
# fmt: off
# ruff: noqa
import subprocess
subprocess.run(["pip", "install", "mkl", "mkl-service"], check=True)
# fmt: on

# COMMAND ----------

# Restart Python to pick up MKL
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify BLAS Backend

# COMMAND ----------

import numpy as np
from scipy import linalg

# Check numpy config
print("=== NumPy Configuration ===")
np.show_config()

# COMMAND ----------

# Check scipy LAPACK
print("\n=== SciPy LAPACK ===")
try:
    # Try to get LAPACK version info
    info = linalg.lapack.get_lapack_funcs(("syevd",), (np.zeros((1, 1)),))
    print(f"LAPACK syevd function: {info}")
except Exception as e:
    print(f"Could not get LAPACK info: {e}")

# COMMAND ----------

# Verify with threadpoolctl
try:
    from threadpoolctl import threadpool_info
    import json

    info = threadpool_info()
    print("\n=== Threadpool Info ===")
    print(json.dumps(info, indent=2))

    # Check for BLAS
    blas_libs = [lib for lib in info if lib.get("user_api") == "blas"]
    if blas_libs:
        for lib in blas_libs:
            backend = lib.get("internal_api", "unknown")
            prefix = lib.get("prefix", "unknown")
            filepath = lib.get("filepath", "unknown")
            print(f"\nBLAS Backend: {backend}")
            print(f"Prefix: {prefix}")
            print(f"Library: {filepath}")
    else:
        print("\nWARNING: No BLAS library detected!")
        print("Eigendecomposition may use OpenBLAS and crash at 100k scale.")
except ImportError:
    print("threadpoolctl not installed - cannot verify BLAS backend")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Small Eigendecomp Test (Verify Working)

# COMMAND ----------

import time
import gc


def test_eigendecomp(
    n_samples: int, use_thread_limit: bool = False, thread_limit: int = 1
):
    """Test eigendecomposition at given scale."""
    from scipy import linalg

    print(f"\n{'='*60}")
    print(f"Testing eigendecomp: {n_samples:,} x {n_samples:,}")
    print(f"Matrix memory: {n_samples * n_samples * 8 / 1e9:.1f} GB")
    print(f"Thread limiting: {use_thread_limit} (limit={thread_limit})")
    print(f"{'='*60}")

    # Generate random symmetric matrix
    print("Generating random symmetric matrix...")
    gen_start = time.time()
    np.random.seed(42)
    A = np.random.randn(n_samples, n_samples).astype(np.float64)
    K = A @ A.T / n_samples  # Make positive semi-definite
    del A
    gc.collect()
    print(f"Matrix generated in {time.time() - gen_start:.1f}s")

    # Log memory
    try:
        import psutil

        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1e9
        print(f"Process memory (RSS): {mem_gb:.1f} GB")
    except ImportError:
        pass

    # Run eigendecomp
    print("\nStarting eigendecomposition...")
    start = time.time()

    try:
        eigh_kwargs = {"overwrite_a": True, "check_finite": False}
        if use_thread_limit:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=thread_limit, user_api="blas"):
                eigenvalues, eigenvectors = linalg.eigh(K, **eigh_kwargs)
        else:
            eigenvalues, eigenvectors = linalg.eigh(K, **eigh_kwargs)

        elapsed = time.time() - start
        print(f"SUCCESS: Eigendecomp completed in {elapsed:.1f}s")
        print(f"Eigenvalue range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
        print(f"Eigenvector shape: {eigenvectors.shape}")

        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        return False, elapsed
    finally:
        gc.collect()


# COMMAND ----------

# Quick sanity check with 1k
test_eigendecomp(1_000)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Scale Testing
# MAGIC
# MAGIC Run progressively larger tests to find the breaking point.

# COMMAND ----------

# 10k - should be fast
test_eigendecomp(10_000)

# COMMAND ----------

# 50k - should complete in ~35 min with OpenBLAS (multi-threaded)
# With MKL this should be faster and stable
test_eigendecomp(50_000)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The 100k Test
# MAGIC
# MAGIC This is where OpenBLAS crashes. With MKL it should work.

# COMMAND ----------

# 100k - the crash scale
# Try with MKL first (no thread limiting needed if MKL is installed)
success, elapsed = test_eigendecomp(100_000, use_thread_limit=False)

if not success:
    print("\n\nRetrying with single-threaded BLAS...")
    test_eigendecomp(100_000, use_thread_limit=True, thread_limit=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Thread Limiting Comparison
# MAGIC
# MAGIC If MKL isn't working, test different thread limits.

# COMMAND ----------

# Skip this section if 100k worked above
# Uncomment to test thread limiting strategies

# print("Testing thread limit = 4")
# test_eigendecomp(100_000, use_thread_limit=True, thread_limit=4)

# print("\nTesting thread limit = 2")
# test_eigendecomp(100_000, use_thread_limit=True, thread_limit=2)

# print("\nTesting thread limit = 1")
# test_eigendecomp(100_000, use_thread_limit=True, thread_limit=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary
# MAGIC
# MAGIC - If MKL installed correctly: 100k should work with full threading
# MAGIC - If still OpenBLAS: Need to try single-thread or alternative approaches
