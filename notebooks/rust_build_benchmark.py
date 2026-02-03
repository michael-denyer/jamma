# Databricks notebook source
# MAGIC %md
# MAGIC # Rust Backend Build & Benchmark: faer vs OxiBLAS
# MAGIC
# MAGIC Builds jamma_core from source ON Databricks to compare:
# MAGIC - **faer 0.22** - Pure Rust eigendecomposition
# MAGIC - **scirs2-linalg/OxiBLAS** - Pure Rust BLAS with SIMD
# MAGIC
# MAGIC Both use target-cpu=native for CPU-specific SIMD optimization.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Build Dependencies

# COMMAND ----------

# MAGIC %sh
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
# MAGIC ## 3. Verify Both Backends Available

# COMMAND ----------

import numpy as np

print("=== JAMMA Backend Verification ===")

import jamma_core

print(f"jamma_core version: {getattr(jamma_core, '__version__', 'unknown')}")

# Check both functions exist
has_faer = hasattr(jamma_core, "eigendecompose_kinship")
print(f"faer backend: eigendecompose_kinship = {has_faer}")
has_scirs2 = hasattr(jamma_core, "eigendecompose_kinship_scirs2")
print(f"OxiBLAS backend: eigendecompose_kinship_scirs2 = {has_scirs2}")

# Quick sanity check with tiny matrix
test_k = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)

print("\nTesting faer backend...")
vals_faer, vecs_faer = jamma_core.eigendecompose_kinship(test_k, 0.0)
print(f"  eigenvalues: {vals_faer}")

print("\nTesting OxiBLAS backend...")
vals_oxi, vecs_oxi = jamma_core.eigendecompose_kinship_scirs2(test_k, 0.0)
print(f"  eigenvalues: {vals_oxi}")

print("\nBoth backends working!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Benchmark: faer vs OxiBLAS (Head-to-Head)

# COMMAND ----------

import gc
import time

import psutil


def benchmark_rust_backends(n_samples: int):
    """Benchmark faer vs OxiBLAS eigendecomposition.

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

    gc.collect()

    # Test OxiBLAS
    print("\nRunning OxiBLAS backend...")
    start = time.perf_counter()
    try:
        vals, vecs = jamma_core.eigendecompose_kinship_scirs2(K, 0.0)
        elapsed = time.perf_counter() - start
        print(f"  OxiBLAS: {elapsed:.2f}s")
        print(f"  eigenvalue range: [{vals.min():.6f}, {vals.max():.6f}]")
        results["oxiblas"] = elapsed
    except Exception as e:
        print(f"  OxiBLAS FAILED: {e}")
        results["oxiblas"] = None

    del K
    gc.collect()

    # Summary
    if results.get("faer") and results.get("oxiblas"):
        ratio = results["faer"] / results["oxiblas"]
        faster = "OxiBLAS" if ratio > 1 else "faer"
        scale = n_samples // 1000
        faer_t, oxi_t = results["faer"], results["oxiblas"]
        print(f"\n{scale}k Summary: faer={faer_t:.2f}s, OxiBLAS={oxi_t:.2f}s")
        print(f"  ratio: {ratio:.2f}x ({faster} is faster)")

    return results


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Benchmarks at Multiple Scales

# COMMAND ----------

# 1k warmup / verification
results_1k = benchmark_rust_backends(1_000)

# COMMAND ----------

# 10k - key performance test
results_10k = benchmark_rust_backends(10_000)

# COMMAND ----------

# 50k - production scale
results_50k = benchmark_rust_backends(50_000)

# COMMAND ----------

# 100k - target scale (if memory allows)

available_gb = psutil.virtual_memory().available / 1e9
required_gb = 2 * (100_000**2) * 8 / 1e9  # K + U

print(f"100k requires ~{required_gb:.0f}GB, available: {available_gb:.0f}GB")

if available_gb > required_gb * 1.1:
    results_100k = benchmark_rust_backends(100_000)
else:
    print("Skipping 100k - insufficient memory")
    results_100k = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY: faer vs OxiBLAS")
print("=" * 60)

all_results = []
for name, res in [("1k", results_1k), ("10k", results_10k), ("50k", results_50k)]:
    if res and res.get("faer") and res.get("oxiblas"):
        all_results.append((name, res["faer"], res["oxiblas"]))

if results_100k and results_100k.get("faer") and results_100k.get("oxiblas"):
    all_results.append(("100k", results_100k["faer"], results_100k["oxiblas"]))

if all_results:
    print(f"\n{'Scale':<8} {'faer':<12} {'OxiBLAS':<12} {'Ratio':<10} {'Faster':<10}")
    print("-" * 54)
    for scale, faer_time, oxi_time in all_results:
        ratio = faer_time / oxi_time
        faster = "OxiBLAS" if ratio > 1 else "faer"
        speedup = ratio if ratio > 1 else 1 / ratio
        row = f"{scale:<8} {faer_time:<12.2f} {oxi_time:<12.2f} {speedup:<10.2f}"
        print(f"{row} {faster:<10}")

    # Overall winner
    avg_ratio = sum(f / o for _, f, o in all_results) / len(all_results)
    print(f"\nAverage ratio: {avg_ratio:.2f}x")
    if avg_ratio > 1:
        print(f"WINNER: OxiBLAS is {avg_ratio:.1f}x faster on average")
    else:
        print(f"WINNER: faer is {1/avg_ratio:.1f}x faster on average")
else:
    print("No comparable results to summarize")
