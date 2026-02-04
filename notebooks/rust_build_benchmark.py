# Databricks notebook source
# MAGIC %md
# MAGIC # Eigendecomposition Benchmark: NumPy/MKL vs faer
# MAGIC
# MAGIC Compares eigendecomposition backends:
# MAGIC - **NumPy/MKL** - Direct LAPACK via numpy.linalg.eigh (no copy overhead)
# MAGIC - **faer 0.22** - Pure Rust with rayon parallelism (requires data copy)
# MAGIC
# MAGIC NumPy/MKL should be more memory-efficient at large scale.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Build Dependencies

# COMMAND ----------

# MAGIC %sh
# MAGIC # Install C toolchain (needed for Rust linker)
# MAGIC apt-get update && apt-get install -y build-essential
# MAGIC
# MAGIC # Install Rust if not present
# MAGIC if ! command -v rustc &> /dev/null; then
# MAGIC     echo "Installing Rust..."
# MAGIC     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# MAGIC     source "$HOME/.cargo/env"
# MAGIC fi
# MAGIC rustc --version
# MAGIC cargo --version

# COMMAND ----------

# MAGIC %pip install maturin psutil numpy loguru

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clone and Build jamma_core

# COMMAND ----------

# MAGIC %sh
# MAGIC source "$HOME/.cargo/env"
# MAGIC
# MAGIC # Clone jamma repo (or update if exists)
# MAGIC if [ -d "/tmp/jamma" ]; then
# MAGIC     cd /tmp/jamma && git pull
# MAGIC else
# MAGIC     git clone https://github.com/michael-denyer/jamma.git /tmp/jamma
# MAGIC fi
# MAGIC
# MAGIC # Show what we're building
# MAGIC cd /tmp/jamma/rust/jamma-core
# MAGIC echo "=== Cargo.toml ==="
# MAGIC cat Cargo.toml
# MAGIC echo ""
# MAGIC echo "=== .cargo/config.toml ==="
# MAGIC cat .cargo/config.toml 2>/dev/null || echo "(no config.toml)"

# COMMAND ----------

# MAGIC %sh
# MAGIC source "$HOME/.cargo/env"
# MAGIC cd /tmp/jamma/rust/jamma-core
# MAGIC
# MAGIC # Clean previous build
# MAGIC cargo clean
# MAGIC
# MAGIC # Build with maturin
# MAGIC echo "Building jamma_core with faer + scirs2-linalg + target-cpu=native..."
# MAGIC maturin build --release
# MAGIC
# MAGIC echo ""
# MAGIC echo "=== Built wheel ==="
# MAGIC ls -la target/wheels/

# COMMAND ----------

# Install the locally-built wheel
import glob
import subprocess

wheels = glob.glob("/tmp/jamma/rust/jamma-core/target/wheels/*.whl")
if wheels:
    wheel_path = wheels[0]
    print(f"Installing: {wheel_path}")
    subprocess.run(["pip", "install", "--force-reinstall", wheel_path], check=True)
else:
    raise RuntimeError("No wheel found! Build may have failed.")

# COMMAND ----------

# MAGIC %pip install --force-reinstall git+https://github.com/michael-denyer/jamma.git

# COMMAND ----------

# Restart to pick up new packages
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verify Backends

# COMMAND ----------

import numpy as np

print("=== Backend Verification ===")

# Check NumPy BLAS config
print("\nNumPy BLAS configuration:")
np.show_config()

# Check faer
import jamma_core

print(f"\njamma_core version: {getattr(jamma_core, '__version__', 'unknown')}")
has_faer = hasattr(jamma_core, "eigendecompose_kinship")
print(f"faer backend available: {has_faer}")

# Quick sanity check with tiny matrix
test_k = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)

print("\nTesting numpy.linalg.eigh...")
vals_np, vecs_np = np.linalg.eigh(test_k)
print(f"  eigenvalues: {vals_np}")

print("\nTesting faer backend...")
vals_faer, vecs_faer = jamma_core.eigendecompose_kinship(test_k, 0.0)
print(f"  eigenvalues: {vals_faer}")

print("\nBoth backends working!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Benchmark Functions

# COMMAND ----------

import gc
import time

import psutil


def benchmark_numpy(n_samples: int):
    """Benchmark numpy.linalg.eigh (uses MKL if available).

    Args:
        n_samples: Matrix dimension
    """
    print(f"\n{'='*60}")
    print(f"NumPy/MKL Benchmark: {n_samples:,} x {n_samples:,}")
    print(f"Matrix memory: {n_samples * n_samples * 8 / 1e9:.1f} GB")
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
        result = {"time": elapsed, "n_samples": n_samples, "backend": "numpy"}
    except Exception as e:
        print(f"  FAILED: {e}")
        result = {
            "time": None,
            "n_samples": n_samples,
            "backend": "numpy",
            "error": str(e),
        }

    del K, vals, vecs
    gc.collect()
    return result


def benchmark_faer(n_samples: int):
    """Benchmark faer eigendecomposition.

    Args:
        n_samples: Matrix dimension
    """
    print(f"\n{'='*60}")
    print(f"faer Benchmark: {n_samples:,} x {n_samples:,}")
    print(f"Matrix memory: {n_samples * n_samples * 8 / 1e9:.1f} GB")
    print(f"{'='*60}")

    np.random.seed(42)
    A = np.random.randn(n_samples, n_samples).astype(np.float64)
    K = A @ A.T / n_samples
    del A
    gc.collect()

    mem_before = psutil.Process().memory_info().rss / 1e9
    print(f"RSS before eigendecomp: {mem_before:.1f} GB")

    print("\nRunning faer eigendecomposition...")
    start = time.perf_counter()
    try:
        vals, vecs = jamma_core.eigendecompose_kinship(K, 0.0)
        elapsed = time.perf_counter() - start
        mem_after = psutil.Process().memory_info().rss / 1e9
        print(f"  Time: {elapsed:.2f}s")
        print(
            f"  RSS after: {mem_after:.1f} GB (delta: {mem_after - mem_before:.1f} GB)"
        )
        print(f"  Eigenvalue range: [{vals.min():.6f}, {vals.max():.6f}]")
        result = {"time": elapsed, "n_samples": n_samples, "backend": "faer"}
    except Exception as e:
        print(f"  FAILED: {e}")
        result = {
            "time": None,
            "n_samples": n_samples,
            "backend": "faer",
            "error": str(e),
        }

    del K, vals, vecs
    gc.collect()
    return result


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Benchmarks at Multiple Scales

# COMMAND ----------

# 10k warmup - compare both backends
print("=" * 60)
print("10k COMPARISON: NumPy/MKL vs faer")
print("=" * 60)
results_10k_numpy = benchmark_numpy(10_000)
results_10k_faer = benchmark_faer(10_000)

# COMMAND ----------

# 50k - production scale comparison
print("=" * 60)
print("50k COMPARISON: NumPy/MKL vs faer")
print("=" * 60)
results_50k_numpy = benchmark_numpy(50_000)
results_50k_faer = benchmark_faer(50_000)

# COMMAND ----------

# 100k - NumPy/MKL only (faer needs too much memory due to copies)
# NumPy peak: K + U + workspace ≈ 3x matrix size (~240 GB)
# faer peak: K + k_vec + faer_mat + U + workspace ≈ 5x matrix size (~400 GB)
available_gb = psutil.virtual_memory().available / 1e9
matrix_gb = (100_000**2) * 8 / 1e9
numpy_peak = 3 * matrix_gb  # ~240 GB
faer_peak = 5 * matrix_gb  # ~400 GB

print(f"100k: matrix={matrix_gb:.0f}GB, available={available_gb:.0f}GB")
print(f"  NumPy peak ≈ {numpy_peak:.0f}GB, faer peak ≈ {faer_peak:.0f}GB")

results_100k_numpy = None
results_100k_faer = None

if available_gb > numpy_peak * 1.1:
    results_100k_numpy = benchmark_numpy(100_000)
else:
    print("Skipping 100k NumPy - insufficient memory")

if available_gb > faer_peak * 1.1:
    results_100k_faer = benchmark_faer(100_000)
else:
    print("Skipping 100k faer - insufficient memory for copy overhead")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY: NumPy/MKL vs faer")
print("=" * 60)

# Collect results
all_results = []
for scale, np_res, faer_res in [
    ("10k", results_10k_numpy, results_10k_faer),
    ("50k", results_50k_numpy, results_50k_faer),
    ("100k", results_100k_numpy, results_100k_faer),
]:
    np_time = np_res.get("time") if np_res else None
    faer_time = faer_res.get("time") if faer_res else None
    if np_time or faer_time:
        n = np_res.get("n_samples") if np_res else faer_res.get("n_samples")
        all_results.append((scale, n, np_time, faer_time))

if all_results:
    print(f"\n{'Scale':<8} {'Samples':<12} {'NumPy':<12} {'faer':<12} {'Winner':<10}")
    print("-" * 56)
    for scale, n, np_time, faer_time in all_results:
        np_str = f"{np_time:.2f}" if np_time else "N/A"
        faer_str = f"{faer_time:.2f}" if faer_time else "N/A"
        if np_time and faer_time:
            winner = "NumPy" if np_time < faer_time else "faer"
            ratio = max(np_time, faer_time) / min(np_time, faer_time)
            winner = f"{winner} ({ratio:.1f}x)"
        elif np_time:
            winner = "NumPy (only)"
        elif faer_time:
            winner = "faer (only)"
        else:
            winner = "N/A"
        print(f"{scale:<8} {n:<12,} {np_str:<12} {faer_str:<12} {winner:<10}")

    print("\nConclusion:")
    print("- NumPy/MKL: Lower memory overhead, can handle larger matrices")
    print("- faer: Pure Rust, but copy overhead limits scale")
else:
    print("No results to summarize")
