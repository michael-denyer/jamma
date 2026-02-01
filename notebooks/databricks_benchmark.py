# Databricks notebook source
# MAGIC %md
# MAGIC # JAMMA Large-Scale Benchmark - Databricks Edition
# MAGIC
# MAGIC Benchmark JAMMA performance at scale (up to 200K samples) on Databricks.
# MAGIC
# MAGIC **Cluster Requirements:**
# MAGIC - Memory-optimized instance (e.g., `Standard_E64s_v3` with 432GB RAM)
# MAGIC - Or GPU instance for JAX acceleration
# MAGIC - DBR 14.0+ (Python 3.10+)

# COMMAND ----------

# MAGIC %pip install git+https://github.com/michael-denyer/jamma.git psutil

# COMMAND ----------

import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# JAX configuration - must be before import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

from jamma.core import configure_jax  # noqa: E402

configure_jax(enable_x64=True)

import jax  # noqa: E402

# COMMAND ----------

# Configuration
OUTPUT_CATALOG = "main"
OUTPUT_SCHEMA = "jamma_benchmarks"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jamma_benchmark")

logger.info(f"Python: {sys.version}")
logger.info(f"JAX version: {jax.__version__}")
logger.info(f"JAX devices: {jax.devices()}")
logger.info(f"JAX backend: {jax.default_backend()}")

# System memory
mem = psutil.virtual_memory()
logger.info(f"RAM: {mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB avail")

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
            timestamp=datetime.utcnow().isoformat(),
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
            timestamp=datetime.utcnow().isoformat(),
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
            timestamp=datetime.utcnow().isoformat(),
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


# Benchmark configurations - target is 200k x 95k (fits in 512GB VM)
CONFIGS = {
    "small": BenchmarkConfig("small", 1_000, 10_000, 10_000),
    "medium": BenchmarkConfig("medium", 10_000, 100_000, 10_000),
    "large": BenchmarkConfig("large", 50_000, 95_000, 10_000),
    "xlarge": BenchmarkConfig("xlarge", 100_000, 95_000, 10_000),
    "target": BenchmarkConfig(
        "target", 200_000, 95_000, 95_000
    ),  # Full 95k SNPs for LMM
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
from jamma.lmm import run_lmm_association  # noqa: E402
from jamma.lmm.runner_jax import run_lmm_association_jax  # noqa: E402


def benchmark_kinship(genotypes: np.ndarray, profiler: BenchmarkProfiler):
    """Benchmark kinship computation."""
    n_samples, n_snps = genotypes.shape

    profiler.start("kinship", "compute", n_samples=n_samples, n_snps=n_snps)
    gc.collect()

    K = compute_centered_kinship(genotypes)
    elapsed = profiler.end("kinship", "compute", output_shape=K.shape)

    return K, elapsed


def benchmark_lmm_cpu(
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    kinship: np.ndarray,
    profiler: BenchmarkProfiler,
    n_snps: int | None = None,
):
    """Benchmark CPU LMM path."""
    if n_snps:
        genotypes = genotypes[:, :n_snps]

    n_samples, actual_snps = genotypes.shape

    profiler.start("lmm_cpu", "association", n_samples=n_samples, n_snps=actual_snps)
    gc.collect()

    results = run_lmm_association(
        genotypes=genotypes,
        phenotype=phenotype,
        kinship=kinship,
    )

    elapsed = profiler.end("lmm_cpu", "association")
    throughput = actual_snps / elapsed if elapsed > 0 else 0
    logger.info(f"CPU throughput: {throughput:.0f} SNPs/sec")

    return results, elapsed


def benchmark_lmm_jax(
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    kinship: np.ndarray,
    profiler: BenchmarkProfiler,
    n_snps: int | None = None,
):
    """Benchmark JAX LMM path."""
    if n_snps:
        genotypes = genotypes[:, :n_snps]

    n_samples, actual_snps = genotypes.shape

    # Warmup
    profiler.start("lmm_jax", "warmup")
    _ = run_lmm_association_jax(
        genotypes=genotypes[:, : min(100, actual_snps)],
        phenotype=phenotype,
        kinship=kinship,
    )
    profiler.end("lmm_jax", "warmup")

    # Run
    profiler.start("lmm_jax", "association", n_samples=n_samples, n_snps=actual_snps)
    gc.collect()

    results = run_lmm_association_jax(
        genotypes=genotypes,
        phenotype=phenotype,
        kinship=kinship,
    )

    if hasattr(results.get("beta"), "block_until_ready"):
        results["beta"].block_until_ready()

    elapsed = profiler.end("lmm_jax", "association", device=str(jax.default_backend()))
    throughput = actual_snps / elapsed if elapsed > 0 else 0
    logger.info(f"JAX throughput: {throughput:.0f} SNPs/sec")

    return results, elapsed


# COMMAND ----------


def run_benchmark(config_name: str, run_id: str) -> tuple[dict, pd.DataFrame]:
    """Run complete benchmark for a configuration."""
    config = CONFIGS[config_name]
    profiler = BenchmarkProfiler(run_id, config_name)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BENCHMARK: {config_name.upper()}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Samples: {config.n_samples:,}, SNPs: {config.n_snps:,}")

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

        # Kinship
        K, kinship_time = benchmark_kinship(genotypes, profiler)
        results["kinship_time"] = kinship_time

        # LMM CPU
        try:
            _, cpu_time = benchmark_lmm_cpu(
                genotypes, phenotype, K, profiler, n_snps=config.n_snps_lmm
            )
            results["lmm_cpu_time"] = cpu_time
            results["lmm_cpu_snps_per_sec"] = config.n_snps_lmm / cpu_time
        except Exception as e:
            profiler.error("lmm_cpu", "association", e)

        # LMM JAX
        try:
            _, jax_time = benchmark_lmm_jax(
                genotypes, phenotype, K, profiler, n_snps=config.n_snps_lmm
            )
            results["lmm_jax_time"] = jax_time
            results["lmm_jax_snps_per_sec"] = config.n_snps_lmm / jax_time
        except Exception as e:
            profiler.error("lmm_jax", "association", e)

        # Speedup
        if results.get("lmm_cpu_time") and results.get("lmm_jax_time"):
            results["jax_speedup"] = results["lmm_cpu_time"] / results["lmm_jax_time"]

    except Exception as e:
        profiler.error("benchmark", "main", e)
        logger.exception(f"Benchmark failed: {e}")
    finally:
        gc.collect()

    return results, profiler.to_dataframe()


# COMMAND ----------

# Execute benchmarks
BENCHMARK_SCALES = ["small", "medium", "large", "xlarge", "target"]

all_results = []
all_events = []

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
run_id = f"benchmark_{timestamp}"

for scale in BENCHMARK_SCALES:
    results, events_df = run_benchmark(scale, run_id)
    all_results.append(results)
    all_events.append(events_df)
    logger.info(f"Completed {scale}: {results}")

results_df = pd.DataFrame(all_results)
events_df = pd.concat(all_events, ignore_index=True)

logger.info(f"\nBenchmark complete. {len(results_df)} configurations tested.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

display(results_df)  # noqa: F821 - Databricks built-in

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
    # Databricks - spark is a built-in in Databricks notebooks
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}")  # noqa: F821

    spark.createDataFrame(results_df).write.mode("append").saveAsTable(  # noqa: F821
        f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.benchmark_results"
    )
    spark.createDataFrame(events_df).write.mode("append").saveAsTable(  # noqa: F821
        f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.benchmark_events"
    )
    logger.info(f"Results saved to {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}")
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
