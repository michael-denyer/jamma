# Databricks notebook source
# MAGIC %md
# MAGIC # Rust Backend Build & Benchmark
# MAGIC
# MAGIC Builds jamma_core from source ON Databricks to test:
# MAGIC - faer 0.24 (upgraded from 0.21)
# MAGIC - target-cpu=native (enables AVX2/AVX-512 on Databricks workers)
# MAGIC
# MAGIC This is the only way to properly benchmark performance since
# MAGIC target-cpu=native generates CPU-specific SIMD code.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Build Dependencies
# MAGIC
# MAGIC Rust toolchain + maturin required for building.

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

# MAGIC %pip install maturin psutil scipy numpy "jax>=0.8" "jaxlib>=0.8" loguru

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
# MAGIC head -30 Cargo.toml
# MAGIC echo ""
# MAGIC echo "=== .cargo/config.toml ==="
# MAGIC cat .cargo/config.toml 2>/dev/null || echo "(no config.toml - using defaults)"

# COMMAND ----------

# MAGIC %sh
# MAGIC source "$HOME/.cargo/env"
# MAGIC
# MAGIC cd /tmp/jamma/rust/jamma-core
# MAGIC
# MAGIC # ============================================================
# MAGIC # FAER VERSION SELECTION
# MAGIC # ============================================================
# MAGIC # 0.24 is SLOW (15x slower than scipy on 1k test)
# MAGIC # Test 0.21, 0.22, 0.23 to find the best version
# MAGIC #
# MAGIC # CHANGE THIS LINE to test different versions:
# MAGIC FAER_VERSION="0.21"
# MAGIC # ============================================================
# MAGIC
# MAGIC echo "=== Setting faer version to ${FAER_VERSION} ==="
# MAGIC SED_PAT='s/faer = { version = "[0-9.]*"/faer = { version = "'
# MAGIC sed -i "${SED_PAT}${FAER_VERSION}\"/" Cargo.toml
# MAGIC grep faer Cargo.toml
# MAGIC
# MAGIC # Clean previous build to ensure fresh compilation
# MAGIC cargo clean
# MAGIC
# MAGIC # Build with maturin (will use .cargo/config.toml settings)
# MAGIC echo ""
# MAGIC echo "Building jamma_core with faer ${FAER_VERSION} + target-cpu=native..."
# MAGIC maturin build --release
# MAGIC
# MAGIC # Show the built wheel
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
# MAGIC ## 3. Verify Backend

# COMMAND ----------

import numpy as np

print("=== JAMMA Backend Verification ===")

from jamma.core import get_backend_info, is_rust_available

info = get_backend_info()
print(f"Selected backend: {info['selected']}")
print(f"Rust available: {info['rust_available']}")

if is_rust_available():
    import jamma_core

    print(f"jamma_core version: {getattr(jamma_core, '__version__', 'unknown')}")
    print("SUCCESS: Rust backend built from source")
else:
    print("ERROR: Rust backend not available after build!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Benchmark: Rust vs scipy

# COMMAND ----------

import gc
import time

import psutil
from scipy import linalg


def benchmark_eigendecomp(n_samples: int, backend: str = "rust"):
    """Benchmark eigendecomposition.

    Args:
        n_samples: Matrix dimension
        backend: "rust" or "scipy"
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_samples:,} x {n_samples:,} ({backend})")
    print(f"Matrix memory: {n_samples * n_samples * 8 / 1e9:.1f} GB")
    print(f"{'='*60}")

    # Generate random symmetric positive semi-definite matrix
    np.random.seed(42)
    A = np.random.randn(n_samples, n_samples).astype(np.float64)
    K = A @ A.T / n_samples
    del A
    gc.collect()

    mem_gb = psutil.Process().memory_info().rss / 1e9
    print(f"RSS before eigendecomp: {mem_gb:.1f} GB")

    print(f"\nStarting {backend} eigendecomposition...")
    start = time.perf_counter()

    try:
        if backend == "scipy":
            eigenvalues, eigenvectors = linalg.eigh(K)
        else:
            import jamma_core

            eigenvalues, eigenvectors = jamma_core.eigendecompose_kinship(
                K, threshold=0.0
            )

        elapsed = time.perf_counter() - start
        print(f"SUCCESS: {elapsed:.2f}s")
        print(f"Eigenvalue range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")

        return elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"FAILED after {elapsed:.1f}s: {e}")
        return None
    finally:
        del K
        gc.collect()


# COMMAND ----------

# Small test to verify both work
print("=== Verification at 1k ===")
scipy_1k = benchmark_eigendecomp(1_000, "scipy")
rust_1k = benchmark_eigendecomp(1_000, "rust")

if scipy_1k and rust_1k:
    r = scipy_1k / rust_1k
    print(f"\n1k Summary: scipy={scipy_1k:.2f}s, rust={rust_1k:.2f}s, ratio={r:.2f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 10k Benchmark (Key Performance Test)

# COMMAND ----------

print("=== 10k Benchmark ===")
scipy_10k = benchmark_eigendecomp(10_000, "scipy")
rust_10k = benchmark_eigendecomp(10_000, "rust")

if scipy_10k and rust_10k:
    ratio = scipy_10k / rust_10k
    faster = "rust" if ratio > 1 else "scipy"
    print("\n10k Summary:")
    print(f"  scipy: {scipy_10k:.2f}s")
    print(f"  rust:  {rust_10k:.2f}s")
    print(f"  ratio: {ratio:.2f}x ({faster} is faster)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Scale Tests (50k, 100k)

# COMMAND ----------

print("=== 50k Benchmark ===")
scipy_50k = benchmark_eigendecomp(50_000, "scipy")
rust_50k = benchmark_eigendecomp(50_000, "rust")

if scipy_50k and rust_50k:
    ratio = scipy_50k / rust_50k
    faster = "rust" if ratio > 1 else "scipy"
    print(f"\n50k Summary: scipy={scipy_50k:.1f}s, rust={rust_50k:.1f}s")
    print(f"  ratio: {ratio:.2f}x ({faster} faster)")

# COMMAND ----------

# 100k - requires ~160GB for K + U
# Only run if you have enough memory

available_gb = psutil.virtual_memory().available / 1e9
required_gb = 2 * (100_000**2) * 8 / 1e9  # K + U

print(f"100k requires ~{required_gb:.0f}GB, available: {available_gb:.0f}GB")

if available_gb > required_gb * 1.1:
    print("\n=== 100k Benchmark ===")
    # scipy_100k = benchmark_eigendecomp(100_000, "scipy")  # Often crashes
    rust_100k = benchmark_eigendecomp(100_000, "rust")
else:
    print("Skipping 100k - insufficient memory")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary
# MAGIC
# MAGIC Compare faer 0.24 + target-cpu=native performance against scipy baseline.

# COMMAND ----------

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY: faer 0.24 + target-cpu=native")
print("=" * 60)

results = []
if "scipy_1k" in dir() and "rust_1k" in dir() and scipy_1k and rust_1k:
    results.append(("1k", scipy_1k, rust_1k))
if "scipy_10k" in dir() and "rust_10k" in dir() and scipy_10k and rust_10k:
    results.append(("10k", scipy_10k, rust_10k))
if "scipy_50k" in dir() and "rust_50k" in dir() and scipy_50k and rust_50k:
    results.append(("50k", scipy_50k, rust_50k))

if results:
    print(f"\n{'Scale':<8} {'scipy':<10} {'rust':<10} {'Ratio':<8} {'Faster':<8}")
    print("-" * 46)
    for scale, scipy_time, rust_time in results:
        r = scipy_time / rust_time
        faster = "rust" if r > 1 else "scipy"
        print(
            f"{scale:<8} {scipy_time:<10.2f} {rust_time:<10.2f} {r:<8.2f} {faster:<8}"
        )

    # Average improvement
    avg_ratio = sum(s / r for _, s, r in results) / len(results)
    print(f"\nAverage ratio: {avg_ratio:.2f}x")
    if avg_ratio >= 1.0:
        print(f"RESULT: rust/faer is {avg_ratio:.1f}x faster than scipy on average")
    else:
        print(f"RESULT: scipy is {1/avg_ratio:.1f}x faster than rust/faer on average")
        print(
            "NOTE: This may indicate faer optimization issues on this CPU architecture"
        )
else:
    print("No comparable results to summarize")
