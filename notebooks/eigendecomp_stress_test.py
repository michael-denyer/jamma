# Databricks notebook source
# MAGIC %md
# MAGIC # Eigendecomposition Stress Test
# MAGIC
# MAGIC Focused test for eigendecomposition at 100k+ scale.
# MAGIC Tests numpy eigendecomposition at large scale.
# MAGIC
# MAGIC Run this notebook to diagnose SIGSEGV crashes without
# MAGIC regenerating all benchmark data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install JAMMA

# COMMAND ----------

# MAGIC %pip install psutil threadpoolctl loguru numpy "jax>=0.8" "jaxlib>=0.8"

# COMMAND ----------

# MAGIC %pip install --force-reinstall git+https://github.com/michael-denyer/jamma.git

# COMMAND ----------

# Restart Python to pick up new packages
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify Backend Configuration

# COMMAND ----------

import numpy as np

# Check JAMMA backend configuration
print("=== JAMMA Eigendecomp Backend ===")
from jamma.core import get_backend_info

info = get_backend_info()
print(f"Selected backend: {info['selected']}")
print(f"GPU available: {info['gpu_available']}")

# COMMAND ----------

# Verify threadpool configuration
try:
    import json

    from threadpoolctl import threadpool_info

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
        print("\nNo BLAS library detected")
except ImportError:
    print("threadpoolctl not installed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Small Eigendecomp Test (Verify Working)

# COMMAND ----------

import gc
import time

from jamma.lmm.eigen import eigendecompose_kinship


def test_eigendecomp(n_samples: int):
    """Test eigendecomposition at given scale using numpy/LAPACK.

    Args:
        n_samples: Matrix dimension (n_samples x n_samples)
    """
    print(f"\n{'='*60}")
    print(f"Testing eigendecomp: {n_samples:,} x {n_samples:,}")
    print(f"Matrix memory: {n_samples * n_samples * 8 / 1e9:.1f} GB")
    print("Backend: numpy/LAPACK")
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

    # Run eigendecomp using JAMMA's eigendecompose_kinship
    # JAMMA auto-selects backend and handles thread limiting
    print("\nStarting eigendecomposition...")
    start = time.time()

    try:
        eigenvalues, eigenvectors = eigendecompose_kinship(K)

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
# MAGIC Run progressively larger tests.

# COMMAND ----------

# 10k - should be fast
test_eigendecomp(10_000)

# COMMAND ----------

# 50k - numpy with thread limiting should work
test_eigendecomp(50_000)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The 100k Test
# MAGIC
# MAGIC This is where OpenBLAS crashes. numpy with MKL and thread limiting should work.

# COMMAND ----------

# 100k - the crash scale for OpenBLAS
# numpy with MKL and thread limiting should handle this
test_eigendecomp(100_000)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary
# MAGIC
# MAGIC - **numpy/MKL**: Stable up to ~46k samples (LP64 int32 limit)
# MAGIC - **numpy/OpenBLAS**: Crashes at ~50k+ without thread limiting
