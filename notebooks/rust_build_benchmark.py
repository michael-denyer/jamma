# Databricks notebook source
# MAGIC %md
# MAGIC # Rust Backend Build & Benchmark: faer Performance
# MAGIC
# MAGIC Builds jamma_core from source ON Databricks to benchmark:
# MAGIC - **faer 0.22** - Pure Rust eigendecomposition with rayon parallelism
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

print("\nfaer backend working!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Benchmark: faer Performance

# COMMAND ----------

import gc
import time

import psutil


def benchmark_faer(n_samples: int):
    """Benchmark faer eigendecomposition at given scale.

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

    # Run faer eigendecomposition
    print("\nRunning faer eigendecomposition...")
    start = time.perf_counter()
    try:
        vals, vecs = jamma_core.eigendecompose_kinship(K, 0.0)
        elapsed = time.perf_counter() - start
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Eigenvalue range: [{vals.min():.6f}, {vals.max():.6f}]")
        result = {"time": elapsed, "n_samples": n_samples}
    except Exception as e:
        print(f"  FAILED: {e}")
        result = {"time": None, "n_samples": n_samples, "error": str(e)}

    del K
    gc.collect()

    return result


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Benchmarks at Multiple Scales

# COMMAND ----------

# 1k warmup / verification
results_1k = benchmark_faer(1_000)

# COMMAND ----------

# 10k - key performance test
results_10k = benchmark_faer(10_000)

# COMMAND ----------

# 50k - production scale
results_50k = benchmark_faer(50_000)

# COMMAND ----------

# 100k - target scale (if memory allows)
# Peak memory: K + k_vec_copy + faer_mat + U + workspace ≈ 5x matrix size
available_gb = psutil.virtual_memory().available / 1e9
matrix_gb = (100_000**2) * 8 / 1e9
required_gb = 5 * matrix_gb  # ~400 GB peak

print(f"100k: matrix={matrix_gb:.0f}GB, peak≈{required_gb:.0f}GB", end="")
print(f", available={available_gb:.0f}GB")

if available_gb > required_gb * 1.1:
    results_100k = benchmark_faer(100_000)
else:
    print("Skipping 100k - insufficient memory for peak usage")
    results_100k = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY: faer Performance")
print("=" * 60)

all_results = []
for name, res in [("1k", results_1k), ("10k", results_10k), ("50k", results_50k)]:
    if res and res.get("time"):
        all_results.append((name, res["n_samples"], res["time"]))

if results_100k and results_100k.get("time"):
    all_results.append(("100k", results_100k["n_samples"], results_100k["time"]))

if all_results:
    print(f"\n{'Scale':<8} {'Samples':<12} {'Time (s)':<12} {'Matrix GB':<12}")
    print("-" * 46)
    for scale, n, elapsed in all_results:
        matrix_gb = n * n * 8 / 1e9
        print(f"{scale:<8} {n:<12,} {elapsed:<12.2f} {matrix_gb:<12.1f}")

    # Performance scaling
    if len(all_results) >= 2:
        print("\nScaling analysis:")
        for i in range(1, len(all_results)):
            prev_scale, prev_n, prev_t = all_results[i - 1]
            curr_scale, curr_n, curr_t = all_results[i]
            n_ratio = curr_n / prev_n
            t_ratio = curr_t / prev_t
            # O(n^3) would give t_ratio = n_ratio^3
            expected_cubic = n_ratio**3
            print(f"  {prev_scale} → {curr_scale}: {t_ratio:.1f}x slower", end="")
            print(f" (O(n³) predicts {expected_cubic:.1f}x)")
else:
    print("No results to summarize")
