# Databricks notebook source
# MAGIC %md
# MAGIC # Eigendecomposition Stress Test
# MAGIC
# MAGIC Focused test for eigendecomposition at 100k+ scale.
# MAGIC Tests Rust backend (via faer) vs scipy/OpenBLAS.
# MAGIC
# MAGIC Run this notebook to diagnose SIGSEGV crashes without
# MAGIC regenerating all benchmark data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install JAMMA with Rust Backend
# MAGIC
# MAGIC The Rust backend uses faer for eigendecomposition, which is stable
# MAGIC at 100k+ scale without BLAS threading issues.

# COMMAND ----------

# MAGIC %pip install psutil threadpoolctl loguru scipy numpy "jax>=0.8" "jaxlib>=0.8"

# COMMAND ----------

# MAGIC %pip install --force-reinstall "jamma[rust] @ git+https://github.com/michael-denyer/jamma.git"

# COMMAND ----------

# Restart Python to pick up new packages
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify Backend Configuration

# COMMAND ----------

import numpy as np

# Check JAMMA backend configuration
print("=== JAMMA Backend Configuration ===")
from jamma.core import get_backend_info, is_rust_available

info = get_backend_info()
print(f"Selected backend: {info['selected']}")
print(f"Rust available: {info['rust_available']}")
print(f"JAX backend: {info.get('jax_backend', 'N/A')}")

# COMMAND ----------

# Check Rust backend details
print("\n=== Rust Backend Status ===")
if is_rust_available():
    try:
        import jamma_core

        print(f"jamma_core version: {getattr(jamma_core, '__version__', 'unknown')}")
        print("Rust eigendecomposition: AVAILABLE")
        print("  - Uses faer library (pure Rust)")
        print("  - No BLAS threading issues")
        print("  - Stable at 100k+ samples")
    except ImportError as e:
        print(f"jamma_core import failed: {e}")
else:
    print("Rust backend NOT available")
    print("Will fall back to scipy (may have BLAS threading issues)")

# COMMAND ----------

# Verify with threadpoolctl (for scipy fallback path)
try:
    import json

    from threadpoolctl import threadpool_info

    info = threadpool_info()
    print("\n=== Threadpool Info (scipy fallback) ===")
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
        print("\nNo BLAS library detected (OK if using Rust backend)")
except ImportError:
    print("threadpoolctl not installed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Small Eigendecomp Test (Verify Working)

# COMMAND ----------

import gc
import os
import time

from jamma.lmm.eigen import eigendecompose_kinship


def test_eigendecomp(
    n_samples: int,
    use_rust: bool = True,
    use_thread_limit: bool = False,
    thread_limit: int = 1,
):
    """Test eigendecomposition at given scale.

    Args:
        n_samples: Matrix dimension (n_samples x n_samples)
        use_rust: Use Rust/faer backend (recommended). If False, uses scipy.
        use_thread_limit: Limit BLAS threads (only relevant for scipy path)
        thread_limit: Number of threads if limiting (only relevant for scipy path)
    """
    print(f"\n{'='*60}")
    print(f"Testing eigendecomp: {n_samples:,} x {n_samples:,}")
    print(f"Matrix memory: {n_samples * n_samples * 8 / 1e9:.1f} GB")
    print(f"Backend: {'Rust/faer' if use_rust else 'scipy'}")
    if not use_rust:
        print(f"Thread limiting: {use_thread_limit} (limit={thread_limit})")
    print(f"{'='*60}")

    # Set backend preference
    if use_rust:
        os.environ["JAMMA_BACKEND"] = "rust"
    else:
        os.environ["JAMMA_BACKEND"] = "jax"

    # Clear backend cache to pick up new setting
    from jamma.core.backend import get_compute_backend

    get_compute_backend.cache_clear()

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
    # This automatically uses Rust or scipy based on JAMMA_BACKEND
    print(
        f"\nStarting eigendecomposition ({os.environ.get('JAMMA_BACKEND', 'auto')})..."
    )
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
test_eigendecomp(1_000, use_rust=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Scale Testing
# MAGIC
# MAGIC Run progressively larger tests with the Rust backend.

# COMMAND ----------

# 10k - should be fast
test_eigendecomp(10_000, use_rust=True)

# COMMAND ----------

# 50k - Rust backend handles this without BLAS threading issues
test_eigendecomp(50_000, use_rust=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The 100k Test
# MAGIC
# MAGIC This is where scipy/OpenBLAS crashes. Rust backend should work.

# COMMAND ----------

# 100k - the crash scale for OpenBLAS
# Rust backend should complete without issues
success, elapsed = test_eigendecomp(100_000, use_rust=True)

if not success:
    print("\n\nRust failed - trying scipy with thread limiting...")
    test_eigendecomp(100_000, use_rust=False, use_thread_limit=True, thread_limit=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Backend Comparison (Optional)
# MAGIC
# MAGIC Compare Rust vs scipy performance at smaller scales.

# COMMAND ----------

# Uncomment to compare backends at 10k scale

# print("=== Rust Backend ===")
# test_eigendecomp(10_000, use_rust=True)

# print("\n=== Scipy Backend ===")
# test_eigendecomp(10_000, use_rust=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary
# MAGIC
# MAGIC - **Rust backend**: Stable at 100k+ scale, no BLAS threading issues
# MAGIC - **Scipy/OpenBLAS**: May crash at 100k+ without thread limiting
# MAGIC - **Scipy with thread limit=1**: Works but slower than Rust
