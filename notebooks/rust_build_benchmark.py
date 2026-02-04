# Databricks notebook source
# MAGIC %md
# MAGIC # Rust Backend Build & Benchmark: faer Performance
# MAGIC
# MAGIC Builds jamma_core from source ON Databricks to benchmark:
# MAGIC - **faer 0.22** - Pure Rust eigendecomposition with rayon parallelism
# MAGIC - Compared against **scipy.linalg.eigh** baseline
# MAGIC
# MAGIC Uses target-cpu=native for CPU-specific SIMD optimization (AVX2/AVX-512).

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
# MAGIC ## 3. Verify faer Backend

# COMMAND ----------

import numpy as np
from scipy import linalg as scipy_linalg

print("=== JAMMA Backend Verification ===")

import jamma_core

print(f"jamma_core version: {getattr(jamma_core, '__version__', 'unknown')}")

# Check faer function exists
has_faer = hasattr(jamma_core, "eigendecompose_kinship")
print(f"faer backend: eigendecompose_kinship = {has_faer}")

# Quick sanity check with tiny matrix
test_k = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)

print("\nTesting faer backend...")
vals_faer, vecs_faer = jamma_core.eigendecompose_kinship(test_k, 0.0)
print(f"  eigenvalues: {vals_faer}")

print("\nTesting scipy baseline...")
vals_scipy, vecs_scipy = scipy_linalg.eigh(test_k)
print(f"  eigenvalues: {vals_scipy}")

print("\nBoth backends working!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Benchmark: faer vs scipy

# COMMAND ----------

import gc
import time

import psutil


def benchmark_faer_vs_scipy(n_samples: int):
    """Benchmark faer vs scipy eigendecomposition.

    Args:
        n_samples: Matrix dimension
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_samples:,} x {n_samples:,}")
    print(f"Matrix memory: {n_samples * n_samples * 8 / 1e9:.1f} GB")
    print(f"{'='*60}")

    # Generate random symmetric positive semi-definite matrix
    np.random.seed(42)
    A = np.random.randn(n_samples, n_samples).astype(np.float64)
    K = A @ A.T / n_samples
    del A
    gc.collect()

    mem_gb = psutil.Process().memory_info().rss / 1e9
    print(f"RSS after matrix gen: {mem_gb:.1f} GB")

    results = {}

    # Test scipy (baseline)
    print("\nRunning scipy baseline...")
    start = time.perf_counter()
    try:
        vals, vecs = scipy_linalg.eigh(K)
        elapsed = time.perf_counter() - start
        print(f"  scipy: {elapsed:.2f}s")
        print(f"  eigenvalue range: [{vals.min():.6f}, {vals.max():.6f}]")
        results["scipy"] = elapsed
    except Exception as e:
        print(f"  scipy FAILED: {e}")
        results["scipy"] = None

    gc.collect()

    # Test faer
    print("\nRunning faer backend...")
    start = time.perf_counter()
    try:
        vals, vecs = jamma_core.eigendecompose_kinship(K, 0.0)
        elapsed = time.perf_counter() - start
        print(f"  faer: {elapsed:.2f}s")
        print(f"  eigenvalue range: [{vals.min():.6f}, {vals.max():.6f}]")
        results["faer"] = elapsed
    except Exception as e:
        print(f"  faer FAILED: {e}")
        results["faer"] = None

    del K
    gc.collect()

    # Summary
    if results.get("scipy") and results.get("faer"):
        ratio = results["scipy"] / results["faer"]
        scale = n_samples // 1000
        scipy_t, faer_t = results["scipy"], results["faer"]
        print(f"\n{scale}k Summary: scipy={scipy_t:.2f}s, faer={faer_t:.2f}s")
        if ratio > 1:
            print(f"  faer is {ratio:.2f}x FASTER than scipy")
        else:
            print(f"  scipy is {1/ratio:.2f}x faster than faer")

    return results


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Benchmarks at Multiple Scales

# COMMAND ----------

# 1k warmup / verification
results_1k = benchmark_faer_vs_scipy(1_000)

# COMMAND ----------

# 10k - key performance test
results_10k = benchmark_faer_vs_scipy(10_000)

# COMMAND ----------

# 50k - production scale
results_50k = benchmark_faer_vs_scipy(50_000)

# COMMAND ----------

# 100k - target scale (if memory allows)

available_gb = psutil.virtual_memory().available / 1e9
required_gb = 2 * (100_000**2) * 8 / 1e9  # K + U

print(f"100k requires ~{required_gb:.0f}GB, available: {available_gb:.0f}GB")

if available_gb > required_gb * 1.1:
    results_100k = benchmark_faer_vs_scipy(100_000)
else:
    print("Skipping 100k - insufficient memory")
    results_100k = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY: faer vs scipy")
print("=" * 60)

all_results = []
for name, res in [("1k", results_1k), ("10k", results_10k), ("50k", results_50k)]:
    if res and res.get("scipy") and res.get("faer"):
        all_results.append((name, res["scipy"], res["faer"]))

if results_100k and results_100k.get("scipy") and results_100k.get("faer"):
    all_results.append(("100k", results_100k["scipy"], results_100k["faer"]))

if all_results:
    print(f"\n{'Scale':<8} {'scipy':<12} {'faer':<12} {'Speedup':<10} {'Winner':<10}")
    print("-" * 54)
    for scale, scipy_time, faer_time in all_results:
        ratio = scipy_time / faer_time
        winner = "faer" if ratio > 1 else "scipy"
        speedup = ratio if ratio > 1 else 1 / ratio
        row = f"{scale:<8} {scipy_time:<12.2f} {faer_time:<12.2f} {speedup:<10.2f}x"
        print(f"{row} {winner:<10}")

    # Overall summary
    avg_ratio = sum(s / f for _, s, f in all_results) / len(all_results)
    print(f"\nAverage speedup: {avg_ratio:.2f}x")
    if avg_ratio > 1:
        print(f"RESULT: faer is {avg_ratio:.1f}x FASTER than scipy on average")
    else:
        print(f"RESULT: scipy is {1/avg_ratio:.1f}x faster than faer on average")
else:
    print("No comparable results to summarize")
