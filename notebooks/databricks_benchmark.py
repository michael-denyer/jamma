# Databricks notebook source
# MAGIC %md
# MAGIC # JAMMA Large-Scale Benchmark - Databricks Edition
# MAGIC
# MAGIC Benchmark JAMMA performance at scale (up to 160K samples) on Databricks.
# MAGIC
# MAGIC **Memory Constraint:** Eigendecomp needs K + U simultaneously:
# MAGIC - 160K samples: ~410GB peak (fits 512GB node)
# MAGIC - 200K samples: ~640GB peak (requires 768GB+ node)
# MAGIC
# MAGIC **Cluster Requirements:**
# MAGIC - Memory-optimized instance with 512GB+ RAM (e.g., `Standard_M64s`)
# MAGIC - DBR 15.4 LTS+ (Python 3.11+ required)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Dependencies
# MAGIC
# MAGIC NumPy with MKL ILP64 (64-bit integers) for stable eigendecomp at 100k+ scale.
# MAGIC - OpenBLAS segfaults on large matrices (50k+)
# MAGIC - MKL LP64 hits int32 overflow at ~46k samples
# MAGIC - **MKL ILP64** uses 64-bit integers - no practical limit
# MAGIC
# MAGIC **IMPORTANT:** Run `dbutils.library.restartPython()` after pip installs.

# COMMAND ----------

# MAGIC %sh # Purge all non-MKL BLAS/LAPACK providers and system numpy
# MAGIC apt-get purge -y libopenblas* libblas* libatlas* liblapack* python3-numpy 2>/dev/null; echo "Non-MKL BLAS purged"

# COMMAND ----------

# Install MKL runtime — provides libmkl_def.so.2 and other computational kernels
# that auditwheel can't bundle (loaded via dlopen, not ELF NEEDED)
# MAGIC %pip install mkl

# COMMAND ----------

# Install numpy with MKL ILP64 from forked wheel repository
# ILP64 uses 64-bit integers, supporting matrices >46k x 46k
# MAGIC %pip install numpy --extra-index-url https://michael-denyer.github.io/numpy-mkl --force-reinstall --upgrade

# COMMAND ----------

# Install jamma dependencies (except numpy which is ILP64 above)
# MAGIC %pip install psutil loguru threadpoolctl jax jaxlib jaxtyping typer progressbar2 bed-reader

# COMMAND ----------

# Install jamma WITHOUT dependencies (numpy already installed with ILP64)
# MAGIC %pip install git+https://github.com/michael-denyer/jamma.git --no-deps

# COMMAND ----------

# Restart Python kernel to load new packages - MUST run this cell
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from loguru import logger as loguru_logger
from threadpoolctl import threadpool_info

# JAX configuration - must be before import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

from jamma.core import configure_jax  # noqa: E402

configure_jax(enable_x64=True)

import jax  # noqa: E402

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify NumPy ILP64 + MKL Backend

# COMMAND ----------

print("=== NumPy ILP64 Backend Verification ===")
print(f"NumPy version: {np.__version__}")
print(f"NumPy location: {np.__file__}")

# --- Step 1: Check build config for ILP64 ---
ilp64_ok = False
try:
    config = np.show_config(mode="dicts")
    blas_info = config.get("Build Dependencies", {}).get("blas", {})
    lapack_info = config.get("Build Dependencies", {}).get("lapack", {})
    blas_name = blas_info.get("name", "unknown")
    lapack_name = lapack_info.get("name", "unknown")
    print(f"  BLAS:   {blas_name}")
    print(f"  LAPACK: {lapack_name}")

    blas_ilp64 = "ilp64" in blas_name.lower()
    lapack_ilp64 = "ilp64" in lapack_name.lower()
    ilp64_ok = blas_ilp64 and lapack_ilp64

    if ilp64_ok:
        print("  ILP64:  CONFIRMED (both BLAS and LAPACK)")
    else:
        print(f"  ILP64:  NOT DETECTED (BLAS={blas_ilp64}, LAPACK={lapack_ilp64})")
        print("  WARNING: Eigendecomp WILL FAIL at >46k samples without ILP64.")
        print("  Troubleshooting:")
        print("    1. Check pip install output above for errors")
        print(
            "    2. Verify wheel was fetched from michael-denyer.github.io (not PyPI)"
        )
        print("    3. Run: pip show numpy | grep Location")
        print("    4. Confirm the fork index has wheels for this Python version")
except Exception as e:
    print(f"  ERROR reading build config: {e}")
    np.show_config()

# --- Step 2: Check runtime BLAS via threadpoolctl ---
print("\nBLAS runtime (threadpoolctl):")
blas_libs = [lib for lib in threadpool_info() if lib.get("user_api") == "blas"]
detected_mkl = False
if blas_libs:
    for lib in blas_libs:
        internal_api = lib.get("internal_api", "unknown")
        filepath = lib.get("filepath", "unknown")
        num_threads = lib.get("num_threads", "?")
        threading_layer = lib.get("threading_layer", "unknown")
        print(f"  Backend: {internal_api}")
        print(f"  Library: {filepath}")
        print(f"  Threads: {num_threads}")
        print(f"  Threading: {threading_layer}")
        if "mkl" in internal_api.lower() or "mkl" in filepath.lower():
            detected_mkl = True
else:
    print("  No BLAS libraries detected by threadpoolctl!")
    print(
        "  This usually means numpy was built without BLAS or the library failed to load."
    )
    print("  Check: python -c 'import numpy; numpy.show_config()'")

if detected_mkl:
    print("\nMKL runtime: DETECTED")
else:
    print("\nMKL runtime: NOT DETECTED")
    print("  Eigendecomp may segfault at 50k+ samples with OpenBLAS.")
    print("  If ILP64 build config was confirmed above but MKL isn't loading,")
    print("  check if mkl-service is installed: pip show mkl-service")

# --- Step 3: Numerical sanity check ---
# This catches LP64 mismatch: garbage eigenvalues when ILP64 header but LP64 runtime
print("\nNumerical sanity check:")
test_k = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
vals, vecs = np.linalg.eigh(test_k)
expected = np.array([1.0, 3.0])
eigenval_error = np.abs(vals - expected).max()
print(f"  Eigenvalues: {vals} (expected {expected})")
print(f"  Error: {eigenval_error:.2e}")
assert eigenval_error < 1e-10, f"Basic eigendecomp FAILED: error={eigenval_error:.2e}"

# Reconstruction error catches the LP64/ILP64 mismatch that gave error=6.54e+01
M = np.random.default_rng(42).standard_normal((100, 100))
M = M @ M.T
evals, evecs = np.linalg.eigh(M)
reconstructed = evecs @ np.diag(evals) @ evecs.T
recon_error = np.abs(M - reconstructed).max()
print(f"  100x100 reconstruction error: {recon_error:.2e}")
assert recon_error < 1e-8, (
    f"Eigendecomp reconstruction error {recon_error:.2e} too large. "
    "This usually means ILP64 headers but LP64 runtime (symbol suffix mismatch). "
    "See: https://github.com/michael-denyer/numpy-mkl build notes."
)
print("  Sanity checks: PASSED")

# --- Summary ---
print(f"\n{'=' * 40}")
if ilp64_ok and detected_mkl and recon_error < 1e-8:
    print("ILP64 + MKL: READY for large-scale eigendecomp")
elif not ilp64_ok:
    print("WARNING: ILP64 NOT confirmed - 46k sample limit applies")
elif not detected_mkl:
    print("WARNING: MKL not detected at runtime - stability risk at 50k+")
print(f"{'=' * 40}")

# COMMAND ----------

# Configuration - Legacy Hive Metastore (single-level namespace)
OUTPUT_SCHEMA = "jamma_benchmarks"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jamma_benchmark")

loguru_logger.remove()
loguru_logger.add(
    sys.stdout,
    level="INFO",
    format="{time:HH:mm:ss} | <level>{level: <8}</level> | {message}",
)

logger.info(f"Python: {sys.version}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"JAX version: {jax.__version__}")
logger.info(f"JAX devices: {jax.devices()}")
logger.info(f"JAX device type: {jax.default_backend()}")

mem = psutil.virtual_memory()
logger.info(f"RAM: {mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB avail")
logger.info(
    f"BLAS: {'MKL ILP64' if (detected_mkl and ilp64_ok) else 'OpenBLAS (limited)'}"
)

# COMMAND ----------


@dataclass
class ProfileEvent:
    """Single profiling event."""

    timestamp: str
    event_type: str
    component: str
    phase: str
    elapsed_ms: float = 0.0
    memory_gb: float = 0.0
    memory_delta_gb: float = 0.0
    metadata: dict = field(default_factory=dict)


class BenchmarkProfiler:
    """Profiler for tracking benchmark execution."""

    def __init__(self, run_id: str, config_name: str):
        self.run_id = run_id
        self.config_name = config_name
        self.events: list[ProfileEvent] = []
        self._start_times: dict[str, float] = {}
        self._start_memory: dict[str, float] = {}

    def _get_memory_gb(self) -> float:
        return psutil.Process().memory_info().rss / 1e9

    def start(self, component: str, phase: str = "main", **metadata):
        key = f"{component}:{phase}"
        self._start_times[key] = time.perf_counter()
        self._start_memory[key] = self._get_memory_gb()

        event = ProfileEvent(
            timestamp=datetime.now(UTC).isoformat(),
            event_type="start",
            component=component,
            phase=phase,
            memory_gb=self._start_memory[key],
            metadata=metadata,
        )
        self.events.append(event)
        logger.info(f"START {component}:{phase} | mem={self._start_memory[key]:.2f}GB")

    def end(self, component: str, phase: str = "main", **metadata):
        key = f"{component}:{phase}"
        end_time = time.perf_counter()
        end_memory = self._get_memory_gb()

        elapsed_ms = (end_time - self._start_times.get(key, end_time)) * 1000
        memory_delta = end_memory - self._start_memory.get(key, end_memory)

        event = ProfileEvent(
            timestamp=datetime.now(UTC).isoformat(),
            event_type="end",
            component=component,
            phase=phase,
            elapsed_ms=elapsed_ms,
            memory_gb=end_memory,
            memory_delta_gb=memory_delta,
            metadata=metadata,
        )
        self.events.append(event)
        logger.info(
            f"END {component}:{phase} | {elapsed_ms:.0f}ms | "
            f"mem={end_memory:.2f}GB (delta={memory_delta:+.2f}GB)"
        )
        return elapsed_ms / 1000

    def error(self, component: str, phase: str, error: Exception):
        event = ProfileEvent(
            timestamp=datetime.now(UTC).isoformat(),
            event_type="error",
            component=component,
            phase=phase,
            memory_gb=self._get_memory_gb(),
            metadata={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        self.events.append(event)
        logger.error(f"ERROR {component}:{phase} | {type(error).__name__}: {error}")

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for e in self.events:
            record = {
                "run_id": self.run_id,
                "config": self.config_name,
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "component": e.component,
                "phase": e.phase,
                "elapsed_ms": e.elapsed_ms,
                "memory_gb": e.memory_gb,
                "memory_delta_gb": e.memory_delta_gb,
                "metadata_json": json.dumps(e.metadata),
            }
            records.append(record)
        return pd.DataFrame(records)


# COMMAND ----------


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    name: str
    n_samples: int
    n_snps: int
    n_snps_lmm: int
    n_causal: int = 100
    heritability: float = 0.5
    maf_min: float = 0.05
    maf_max: float = 0.5
    seed: int = 42

    @property
    def memory_estimate_gb(self) -> float:
        """Get total memory estimate using jamma.core.memory module."""
        from jamma.core.memory import estimate_workflow_memory

        est = estimate_workflow_memory(self.n_samples, self.n_snps)
        return est.total_gb

    def memory_breakdown(self):
        """Get detailed memory breakdown."""
        from jamma.core.memory import estimate_workflow_memory

        return estimate_workflow_memory(self.n_samples, self.n_snps)

    def validate_memory(self) -> None:
        """Raise MemoryError if insufficient memory for this config."""
        from jamma.core.memory import check_memory_available

        check_memory_available(
            self.memory_estimate_gb,
            safety_margin=0.1,
            operation=f"benchmark {self.name}",
        )


# Benchmark configurations for 512GB VM
#
# Memory constraint: Eigendecomposition requires K + U simultaneously
#   Peak = 2 * n_samples^2 * 8 bytes
#   512GB node (~460GB usable) supports max ~160k samples safely
#
# Original target was 200k but that requires ~640GB (eigendecomp bottleneck)
CONFIGS = {
    "small": BenchmarkConfig("small", 1_000, 10_000, 10_000),
    "medium": BenchmarkConfig("medium", 10_000, 100_000, 100_000),
    "large": BenchmarkConfig("large", 50_000, 95_000, 10_000),
    "xlarge": BenchmarkConfig("xlarge", 100_000, 95_000, 10_000),
    "target": BenchmarkConfig(
        "target", 160_000, 95_000, 95_000
    ),  # Max for 512GB node (~410GB peak during eigendecomp)
}

for name, cfg in CONFIGS.items():
    est_gb = cfg.memory_estimate_gb
    logger.info(f"  {name}: {cfg.n_samples:,} x {cfg.n_snps:,} SNPs (~{est_gb:.0f} GB)")

# COMMAND ----------


def generate_synthetic_data(config: BenchmarkConfig, profiler: BenchmarkProfiler):
    """Generate synthetic GWAS data with profiling."""
    profiler.start("data_gen", "init", n_samples=config.n_samples, n_snps=config.n_snps)

    rng = np.random.default_rng(config.seed)
    mafs = rng.uniform(config.maf_min, config.maf_max, config.n_snps)

    geno_gb = config.n_samples * config.n_snps * 4 / 1e9  # float32
    logger.info(
        f"Allocating genotype matrix: {config.n_samples:,} x {config.n_snps:,} "
        f"({geno_gb:.1f}GB, float32)"
    )

    profiler.start("data_gen", "genotypes")
    chunk_size = 10_000
    genotypes = np.zeros((config.n_samples, config.n_snps), dtype=np.float32)

    for i in range(0, config.n_snps, chunk_size):
        end = min(i + chunk_size, config.n_snps)
        chunk_mafs = mafs[i:end]
        p = chunk_mafs[np.newaxis, :]
        u = rng.random((config.n_samples, end - i))
        genotypes[:, i:end] = np.where(
            u < (1 - p) ** 2, 0, np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2)
        ).astype(np.float32)

    profiler.end("data_gen", "genotypes", shape=genotypes.shape)

    # Phenotype
    profiler.start("data_gen", "phenotype")
    causal_indices = rng.choice(config.n_snps, config.n_causal, replace=False)
    true_betas = rng.standard_normal(config.n_causal)

    G_causal = genotypes[:, causal_indices]
    G_causal_std = (G_causal - G_causal.mean(axis=0)) / (G_causal.std(axis=0) + 1e-8)
    genetic_value = G_causal_std @ true_betas

    var_g = np.var(genetic_value)
    var_e = var_g * (1 - config.heritability) / config.heritability
    noise = rng.standard_normal(config.n_samples) * np.sqrt(var_e)

    phenotype = genetic_value + noise
    phenotype = (phenotype - phenotype.mean()) / phenotype.std()
    profiler.end("data_gen", "phenotype")

    profiler.end("data_gen", "init")

    return genotypes, phenotype.astype(np.float64), causal_indices, true_betas


# COMMAND ----------

from jamma.kinship import compute_centered_kinship  # noqa: E402
from jamma.lmm.eigen import eigendecompose_kinship  # noqa: E402
from jamma.lmm.runner_jax import run_lmm_association_jax  # noqa: E402


def benchmark_kinship(genotypes: np.ndarray, profiler: BenchmarkProfiler):
    """Benchmark kinship computation and eigendecomposition.

    Returns:
        Tuple of (K, eigenvalues, eigenvectors, kinship_time, eigen_time)
    """
    n_samples, n_snps = genotypes.shape

    profiler.start("kinship", "compute", n_samples=n_samples, n_snps=n_snps)
    gc.collect()

    K = compute_centered_kinship(genotypes)
    kinship_elapsed = profiler.end("kinship", "compute", output_shape=K.shape)

    # Eigendecomposition — the most likely crash point.
    # Segfaults (OpenBLAS >50k) and OOM kills bypass Python exceptions,
    # so log memory state before entering to help diagnose silent restarts.
    mem = psutil.virtual_memory()
    matrix_gb = n_samples * n_samples * 8 / 1e9
    eigen_peak_gb = 3 * matrix_gb  # K + U + LAPACK workspace
    logger.info(
        f"Eigendecomp: {n_samples:,}x{n_samples:,} "
        f"(matrix={matrix_gb:.1f}GB, peak~{eigen_peak_gb:.0f}GB, "
        f"avail={mem.available / 1e9:.0f}GB, "
        f"RSS={psutil.Process().memory_info().rss / 1e9:.1f}GB)"
    )
    if mem.available < eigen_peak_gb * 1e9 * 1.1:
        logger.warning(
            f"Tight memory: need ~{eigen_peak_gb:.0f}GB, "
            f"only {mem.available / 1e9:.0f}GB available. OOM kill likely."
        )

    profiler.start("kinship", "eigendecomp", n_samples=n_samples)
    eigenvalues, eigenvectors = eigendecompose_kinship(K)
    eigen_elapsed = profiler.end("kinship", "eigendecomp")

    # Validate eigendecomposition results — catches LP64/ILP64 mismatch
    # which produces garbage eigenvalues without crashing (error ~1e+1 vs ~1e-14)
    logger.info(
        f"Eigenvalues: min={eigenvalues.min():.6g}, max={eigenvalues.max():.6g}, "
        f"n_negative={int((eigenvalues < -1e-6).sum())}"
    )
    # Spot-check reconstruction on a small submatrix.
    # Use broadcasting instead of np.diag() to avoid allocating a full n×n matrix.
    k = min(50, n_samples)
    K_sub = K[:k, :k]
    recon = (eigenvectors[:k, :] * eigenvalues) @ eigenvectors[:k, :].T
    recon_error = np.abs(K_sub - recon).max()
    logger.info(
        f"Eigendecomp reconstruction check ({k}x{k} submatrix): error={recon_error:.2e}"
    )
    if recon_error > 1e-6:
        logger.error(
            f"Reconstruction error {recon_error:.2e} is suspiciously large. "
            "Possible LP64/ILP64 mismatch — eigenvalues may be garbage. "
            "Check numpy ILP64 verification cell output above."
        )

    return K, eigenvalues, eigenvectors, kinship_elapsed, eigen_elapsed


def benchmark_lmm_jax(
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    kinship: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    snp_info: list,
    profiler: BenchmarkProfiler,
    n_snps: int | None = None,
):
    """Benchmark JAX LMM path.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps)
        phenotype: Phenotype vector (n_samples,)
        kinship: Kinship matrix (n_samples, n_samples)
        eigenvalues: Pre-computed eigenvalues from kinship decomposition
        eigenvectors: Pre-computed eigenvectors from kinship decomposition
        snp_info: SNP metadata list
        profiler: Benchmark profiler
        n_snps: Optional limit on number of SNPs to test
    """
    if n_snps:
        genotypes = genotypes[:, :n_snps]
        snp_info = snp_info[:n_snps]

    n_samples, actual_snps = genotypes.shape

    # Warmup (use subset of snp_info too, disable progress output)
    # Pass pre-computed eigendecomposition to avoid redundant 35-min computation
    warmup_snps = min(100, actual_snps)
    profiler.start("lmm_jax", "warmup")
    try:
        _ = run_lmm_association_jax(
            genotypes=genotypes[:, :warmup_snps],
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info[:warmup_snps],
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            show_progress=False,
        )
        profiler.end("lmm_jax", "warmup")
    except Exception as e:
        profiler.error("lmm_jax", "warmup", e)
        logger.error(f"JAX warmup failed: {type(e).__name__}: {e}")
        raise

    # Run with progress output enabled
    # Pass pre-computed eigendecomposition to avoid redundant computation
    profiler.start("lmm_jax", "association", n_samples=n_samples, n_snps=actual_snps)
    gc.collect()

    results = run_lmm_association_jax(
        genotypes=genotypes,
        phenotypes=phenotype,
        kinship=kinship,
        snp_info=snp_info,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        show_progress=True,
    )

    # Results is a list[AssocResult], no need for block_until_ready
    # JAX computation is already complete when run_lmm_association_jax returns

    elapsed = profiler.end("lmm_jax", "association", device=str(jax.default_backend()))
    throughput = actual_snps / elapsed if elapsed > 0 else 0
    logger.info(f"JAX throughput: {throughput:.0f} SNPs/sec")

    return results, elapsed


# COMMAND ----------


def run_benchmark(config_name: str, run_id: str) -> tuple[dict, pd.DataFrame]:
    """Run complete benchmark for a configuration."""
    from jamma.core import log_memory_snapshot

    config = CONFIGS[config_name]
    profiler = BenchmarkProfiler(run_id, config_name)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BENCHMARK: {config_name.upper()}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Samples: {config.n_samples:,}, SNPs: {config.n_snps:,}")

    # Log memory at start of each benchmark run
    log_memory_snapshot(f"start_{config_name}")

    # Memory breakdown
    breakdown = config.memory_breakdown()
    logger.info(
        f"Memory estimate: {breakdown.total_gb:.0f}GB (peak during eigendecomp)"
    )
    logger.info(f"  Kinship: {breakdown.kinship_gb:.0f}GB")
    logger.info(f"  Eigenvectors: {breakdown.eigenvectors_gb:.0f}GB")
    logger.info(f"  Genotypes: {breakdown.genotypes_gb:.0f}GB")
    logger.info(f"  Available: {breakdown.available_gb:.0f}GB")

    # Memory validation using jamma.core.memory
    try:
        config.validate_memory()
    except MemoryError as e:
        logger.warning(str(e))
        profiler.error("system", "memory_check", e)
        return {"config": config_name, "error": "memory"}, profiler.to_dataframe()

    results = {"config": config_name, "run_id": run_id}

    try:
        # Generate data
        genotypes, phenotype, _, _ = generate_synthetic_data(config, profiler)

        # Generate SNP info (required by LMM functions)
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(config.n_snps)
        ]

        # JAMMA: Kinship + eigendecomposition + LMM (wall clock time)
        jamma_start = time.time()

        K, eigenvalues, eigenvectors, kinship_time, eigen_time = benchmark_kinship(
            genotypes, profiler
        )
        results["kinship_time"] = kinship_time
        results["eigendecomp_time"] = eigen_time

        # LMM JAX - store results for validation
        # Pass pre-computed eigendecomposition to avoid redundant computation
        jax_assoc_results = None
        try:
            jax_assoc_results, jax_time = benchmark_lmm_jax(
                genotypes,
                phenotype,
                K,
                eigenvalues,
                eigenvectors,
                snp_info,
                profiler,
                n_snps=config.n_snps_lmm,
            )
            results["lmm_jax_time"] = jax_time
            results["lmm_jax_snps_per_sec"] = config.n_snps_lmm / jax_time
        except Exception as e:
            import traceback

            profiler.error("lmm_jax", "association", e)
            logger.error(
                f"LMM JAX failed: {type(e).__name__}: {e}\n"
                f"  n_samples={config.n_samples:,}, n_snps_lmm={config.n_snps_lmm:,}\n"
                f"  RSS={psutil.Process().memory_info().rss / 1e9:.1f}GB\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            results["error"] = f"lmm_jax: {type(e).__name__}: {e}"

        # JAMMA total = wall clock from start to end (not sum of components)
        jamma_total = time.time() - jamma_start
        results["jamma_total_time"] = jamma_total
        logger.info(f"JAMMA total time: {jamma_total:.1f}s (wall clock)")

    except Exception as e:
        profiler.error("benchmark", "main", e)
        mem = psutil.virtual_memory()
        logger.exception(
            f"Benchmark {config_name} FAILED: {type(e).__name__}: {e}\n"
            f"  RAM: {mem.available / 1e9:.0f}GB avail / {mem.total / 1e9:.0f}GB total\n"
            f"  RSS: {psutil.Process().memory_info().rss / 1e9:.1f}GB\n"
            f"  If this was a MemoryError, try: reduce config or use larger VM.\n"
            f"  If 'Killed' or no message, it was OOM-killed by the OS — check driver logs."
        )
        results["error"] = f"{type(e).__name__}: {e}"

    # Clean up memory after each benchmark run to prevent accumulation
    # This is critical for sequential runs - without cleanup, memory from
    # previous runs causes OOM/SIGSEGV on larger configs
    #
    # IMPORTANT: Must explicitly delete large arrays before cleanup_memory()
    # because Python keeps local variables alive until function returns.
    profiler.start("cleanup", "memory")
    from jamma.core import cleanup_memory

    # Delete large arrays explicitly - these hold 90%+ of memory
    # genotypes: n_samples × n_snps × 4 bytes (e.g., 100k × 95k = 38GB)
    # K: n_samples² × 8 bytes (e.g., 100k² = 80GB)
    # eigenvalues: n_samples × 8 bytes (negligible)
    # eigenvectors: n_samples² × 8 bytes (e.g., 100k² = 80GB)
    try:
        del genotypes
    except NameError:
        pass
    try:
        del K
    except NameError:
        pass
    try:
        del eigenvalues
    except NameError:
        pass
    try:
        del eigenvectors
    except NameError:
        pass
    try:
        del jax_assoc_results
    except NameError:
        pass
    try:
        del phenotype
    except NameError:
        pass
    try:
        del snp_info
    except NameError:
        pass

    cleanup_memory(clear_jax=True, verbose=True)
    profiler.end("cleanup", "memory")

    return results, profiler.to_dataframe()


# COMMAND ----------

# Execute benchmarks
# To skip scales that already completed, set SKIP_SCALES (e.g., ["small", "medium"])
# To resume a previous run, set RESUME_RUN_ID to the previous run_id
BENCHMARK_SCALES = ["small", "medium", "large", "xlarge", "target"]
SKIP_SCALES: list[
    str
] = []  # Add scales to skip here, e.g., ["small", "medium", "large"]
RESUME_RUN_ID: str | None = None  # Set to previous run_id to append results

all_results = []
all_events = []

# Use existing run_id if resuming, otherwise create new
if RESUME_RUN_ID:
    run_id = RESUME_RUN_ID
    # Try to load existing results
    try:
        existing_results = pd.read_parquet(
            f"/dbfs/tmp/jamma_benchmark_{run_id}_results.parquet"
        )
        existing_events = pd.read_parquet(
            f"/dbfs/tmp/jamma_benchmark_{run_id}_events.parquet"
        )
        all_results = existing_results.to_dict("records")
        all_events = [existing_events]
        completed_scales = set(existing_results["config"].tolist())
        SKIP_SCALES = list(completed_scales)
        logger.info(f"Resumed run {run_id}, skipping completed: {SKIP_SCALES}")
    except Exception as e:
        logger.warning(f"Could not load existing results for {run_id}: {e}")
else:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"benchmark_{timestamp}"

logger.info(f"Run ID: {run_id}")
logger.info(f"Scales to run: {[s for s in BENCHMARK_SCALES if s not in SKIP_SCALES]}")

# Track total benchmark suite time
benchmark_start_time = time.time()
logger.info(f"Starting benchmark suite at {datetime.now(UTC).isoformat()}")

for scale in BENCHMARK_SCALES:
    if scale in SKIP_SCALES:
        logger.info(f"Skipping {scale} (already completed or in SKIP_SCALES)")
        continue

    # Crash marker: if the kernel dies (OOM kill / segfault), this persists.
    # On Databricks, silent restart means the cell appears to complete but
    # variables from before the crash are gone. This marker survives if saved
    # in the incremental checkpoint before the crash.
    crash_marker = {
        "config": scale,
        "run_id": run_id,
        "error": f"Kernel crashed during {scale} (OOM kill or segfault). "
        "Check Databricks driver logs for SIGKILL/SIGSEGV.",
    }

    mem = psutil.virtual_memory()
    logger.info(
        f"\n--- Starting {scale} | "
        f"RAM: {mem.available / 1e9:.0f}GB avail / {mem.total / 1e9:.0f}GB total | "
        f"RSS: {psutil.Process().memory_info().rss / 1e9:.1f}GB ---"
    )

    scale_start = time.time()
    results, events_df = run_benchmark(scale, run_id)
    scale_elapsed = time.time() - scale_start

    # If we get here, no crash — overwrite the marker
    results["total_time"] = scale_elapsed
    all_results.append(results)
    all_events.append(events_df)
    logger.info(f"Completed {scale} in {scale_elapsed:.1f}s")

    # === INCREMENTAL SAVE after each scale ===
    # Save results immediately so we don't lose data if later scales crash
    results_df = pd.DataFrame(all_results)
    events_df_combined = pd.concat(all_events, ignore_index=True)

    # Save to Delta tables (overwrites each time with cumulative results)
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUTPUT_SCHEMA}")  # noqa: F821
        results_df.to_parquet(f"/dbfs/tmp/jamma_benchmark_{run_id}_results.parquet")
        events_df_combined.to_parquet(
            f"/dbfs/tmp/jamma_benchmark_{run_id}_events.parquet"
        )
        logger.info(
            f"Saved checkpoint after {scale}: "
            f"/dbfs/tmp/jamma_benchmark_{run_id}_*.parquet"
        )
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")

    # Also display incremental results
    print(f"\n=== Results after {scale} ===")
    display(results_df)  # noqa: F821

results_df = pd.DataFrame(all_results)
events_df = pd.concat(all_events, ignore_index=True)

benchmark_total_time = time.time() - benchmark_start_time
logger.info(f"\nBenchmark suite complete at {datetime.now(UTC).isoformat()}")
total_min = benchmark_total_time / 60
logger.info(f"Total benchmark time: {benchmark_total_time:.1f}s ({total_min:.1f} min)")
logger.info(f"{len(results_df)} configurations tested.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

display(results_df)  # noqa: F821 - Databricks built-in

# COMMAND ----------

# Performance Summary
print("\n" + "=" * 60)
print("JAMMA PERFORMANCE SUMMARY")
print("=" * 60)

for _, row in results_df.iterrows():
    cfg = row["config"]
    print(f"\n{cfg}:")

    jax_lmm = row.get("lmm_jax_time")
    if jax_lmm:
        print(f"  JAMMA JAX:  {jax_lmm:7.2f}s")
        snps_per_sec = row.get("jax_snps_per_sec")
        if snps_per_sec:
            print(f"  Throughput: {snps_per_sec:,.0f} SNPs/sec")

# COMMAND ----------

# Bottleneck Analysis
print("\n" + "=" * 60)
print("BOTTLENECK ANALYSIS")
print("=" * 60)

end_events = events_df[events_df["event_type"] == "end"].copy()
end_events["elapsed_sec"] = end_events["elapsed_ms"] / 1000

for config in results_df["config"].unique():
    config_events = end_events[end_events["config"] == config]
    total_time = config_events["elapsed_sec"].sum()

    print(f"\n{config}:")
    for _, row in config_events.nlargest(5, "elapsed_sec").iterrows():
        pct = row["elapsed_sec"] / total_time * 100 if total_time > 0 else 0
        comp_phase = f"{row['component']}:{row['phase']}"
        print(f"  {comp_phase}: {row['elapsed_sec']:.2f}s ({pct:.1f}%)")

# COMMAND ----------

# Save to Delta (Databricks) or CSV (local)
try:
    # Databricks with legacy Hive Metastore (single-level namespace)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUTPUT_SCHEMA}")  # noqa: F821

    # Use mergeSchema to handle evolving result columns
    spark.createDataFrame(results_df).write.mode("append").option(  # noqa: F821
        "mergeSchema", "true"
    ).saveAsTable(f"{OUTPUT_SCHEMA}.benchmark_results")

    spark.createDataFrame(events_df).write.mode("append").option(  # noqa: F821
        "mergeSchema", "true"
    ).saveAsTable(f"{OUTPUT_SCHEMA}.benchmark_events")
    logger.info(f"Results saved to {OUTPUT_SCHEMA}")
except NameError:
    # Local
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / f"results_{run_id}.csv", index=False)
    events_df.to_csv(output_dir / f"events_{run_id}.csv", index=False)
    logger.info(f"Results saved to {output_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimization Recommendations

# COMMAND ----------

print("\n" + "=" * 60)
print("OPTIMIZATION RECOMMENDATIONS")
print("=" * 60)

recommendations = []

# Check kinship bottleneck
end_events = events_df[events_df["event_type"] == "end"]
kinship_events = end_events[end_events["component"] == "kinship"]
total_events = end_events.groupby("config")["elapsed_ms"].sum()

for config in results_df["config"].unique():
    k_time = kinship_events[kinship_events["config"] == config]["elapsed_ms"].sum()
    total_time = total_events.get(config, 1)

    if k_time / total_time > 0.5:
        pct = k_time / total_time * 100
        recommendations.append(
            f"{config}: Kinship dominates ({pct:.0f}% of time). "
            f"Consider JAX-accelerated kinship or chunked computation."
        )

# Check JAX speedup
for _, row in results_df.iterrows():
    speedup = row.get("jax_speedup")
    if speedup and speedup < 1.0:
        recommendations.append(
            f"{row['config']}: JAX slower than CPU ({speedup:.2f}x). "
            f"JIT overhead may dominate for small workloads."
        )

# Output
if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
else:
    print("\nNo critical issues detected.")
