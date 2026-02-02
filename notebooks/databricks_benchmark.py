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
# MAGIC - Or GPU instance for JAX acceleration
# MAGIC - DBR 15.4 LTS+ (Python 3.11+ required)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Dependencies
# MAGIC
# MAGIC JAX 0.8+ requires NumPy 2.0+. Force upgrade numpy before installing JAX.
# MAGIC
# MAGIC **IMPORTANT:** Run `dbutils.library.restartPython()` after pip installs.

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall "numpy>=2.0,<2.4"

# COMMAND ----------

# MAGIC %pip install "jax>=0.8" "jaxlib>=0.8" psutil

# COMMAND ----------

# MAGIC %pip install --no-deps --force-reinstall git+https://github.com/michael-denyer/jamma.git

# COMMAND ----------

# MAGIC %pip install bed-reader jaxtyping typer loguru progressbar2

# COMMAND ----------

# Restart Python kernel to load new numpy - MUST run this cell
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

# JAX configuration - must be before import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

from jamma.core import configure_jax  # noqa: E402

configure_jax(enable_x64=True)

import jax  # noqa: E402

# COMMAND ----------

# Configuration - Legacy Hive Metastore (single-level namespace)
OUTPUT_SCHEMA = "jamma_benchmarks"

# Setup logging - configure both standard logging and loguru for JAMMA output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jamma_benchmark")

# Configure loguru to show JAMMA logs (JAMMA uses loguru internally)
# Remove default handler and add one with INFO level to stdout
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
logger.info(f"JAX backend: {jax.default_backend()}")

# System memory
mem = psutil.virtual_memory()
logger.info(f"RAM: {mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB avail")

# GEMMA binary path - default is the Databricks micromamba environment
# Must be defined before benchmark_gemma() which uses it as a default
GEMMA_PATH = os.environ.get("GEMMA_PATH", "/opt/micromamba/envs/disco/bin/gemma")

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

import shutil
import subprocess
import tempfile

from jamma.kinship import compute_centered_kinship  # noqa: E402
from jamma.lmm.eigen import eigendecompose_kinship  # noqa: E402
from jamma.lmm.runner_jax import run_lmm_association_jax  # noqa: E402


def _write_bed_file(path: str, genotypes: np.ndarray) -> None:
    """Write genotypes to PLINK .bed format.

    Args:
        path: Output path for .bed file.
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2 or NaN.
                   Values represent counts of A1 (BIM col5) allele.
    """
    n_samples, n_snps = genotypes.shape

    # BED file magic bytes: PLINK 1.9 format, SNP-major
    magic = bytes([0x6C, 0x1B, 0x01])

    # Pack genotypes: 4 samples per byte
    # PLINK encoding (genotype = count of A1 allele):
    #   00 = hom A1 (genotype 2)
    #   01 = missing
    #   10 = het (genotype 1)
    #   11 = hom A2 (genotype 0)
    geno_t = genotypes.T  # SNP-major
    bytes_per_snp = (n_samples + 3) // 4
    data = bytearray(len(magic) + n_snps * bytes_per_snp)
    data[:3] = magic

    for j in range(n_snps):
        for i in range(n_samples):
            byte_idx = 3 + j * bytes_per_snp + i // 4
            bit_idx = (i % 4) * 2
            g = geno_t[j, i]
            if np.isnan(g):
                code = 0b01  # Missing
            elif g == 0:
                code = 0b11  # Hom A2 (0 copies of A1)
            elif g == 1:
                code = 0b10  # Het
            else:
                code = 0b00  # Hom A1 (2 copies of A1)
            data[byte_idx] |= code << bit_idx

    with open(path, "wb") as f:
        f.write(data)


def _write_plink_files(
    prefix: str, genotypes: np.ndarray, phenotype: np.ndarray
) -> None:
    """Write genotypes and phenotypes to PLINK binary format.

    Creates .fam, .bim, and .bed files at the given prefix.
    """
    n_samples, n_snps = genotypes.shape

    # Write .fam file
    with open(f"{prefix}.fam", "w") as f:
        for i in range(n_samples):
            f.write(f"FAM{i} IND{i} 0 0 0 {phenotype[i]:.6f}\n")

    # Write .bim file
    # Genotype values count copies of A1 (BIM col5, minor allele)
    # A1 (col5) = G = minor allele, A2 (col6) = A = major allele
    with open(f"{prefix}.bim", "w") as f:
        for j in range(n_snps):
            f.write(f"1 rs{j} 0 {j * 1000} G A\n")

    # Write .bed file
    _write_bed_file(f"{prefix}.bed", genotypes)


@dataclass
class GemmaResult:
    """GEMMA benchmark result with timing and output for validation."""

    kinship_time: float | None = None
    lmm_time: float | None = None
    assoc_df: pd.DataFrame | None = None  # GEMMA association results


def benchmark_gemma(
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    profiler: BenchmarkProfiler,
    n_snps_lmm: int,
    gemma_path: str | None = None,
) -> GemmaResult:
    """Benchmark GEMMA for direct comparison.

    Returns:
        GemmaResult with timing and association results for validation.
    """
    result = GemmaResult()

    # Find GEMMA binary
    gemma_bin = gemma_path or GEMMA_PATH or shutil.which("gemma")
    if not gemma_bin or not Path(gemma_bin).exists():
        logger.warning("GEMMA not found - skipping GEMMA benchmark")
        return result

    n_samples, n_snps = genotypes.shape

    # Only benchmark smaller configs (GEMMA is slow at scale)
    if n_samples > 20_000:
        logger.info(f"Skipping GEMMA benchmark for {n_samples:,} samples (too large)")
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        prefix = tmpdir / "bench"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Write PLINK files (only n_snps_lmm SNPs for fair comparison)
        _write_plink_files(str(prefix), genotypes[:, :n_snps_lmm], phenotype)

        # Benchmark GEMMA kinship - stream output in real-time (like lab_disco)
        profiler.start("gemma", "kinship", n_samples=n_samples, n_snps=n_snps_lmm)
        try:
            proc = subprocess.Popen(
                [gemma_bin, "-bfile", str(prefix), "-gk", "1", "-o", "kinship"],
                cwd=str(output_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            while True:
                output = proc.stdout.readline()
                if output == "" and proc.poll() is not None:
                    break
                if output:
                    print(output.strip())
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, "gemma -gk")
            result.kinship_time = profiler.end("gemma", "kinship")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            profiler.error("gemma", "kinship", e)
            return result

        # Benchmark GEMMA LMM - stream output in real-time (like lab_disco)
        kinship_file = output_dir / "output" / "kinship.cXX.txt"
        profiler.start("gemma", "lmm", n_samples=n_samples, n_snps=n_snps_lmm)
        try:
            proc = subprocess.Popen(
                [
                    gemma_bin,
                    "-bfile",
                    str(prefix),
                    "-k",
                    str(kinship_file),
                    "-lmm",
                    "1",
                    "-o",
                    "assoc",
                ],
                cwd=str(output_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            while True:
                output = proc.stdout.readline()
                if output == "" and proc.poll() is not None:
                    break
                if output:
                    print(output.strip())
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, "gemma -lmm")
            result.lmm_time = profiler.end("gemma", "lmm")
            throughput = n_snps_lmm / result.lmm_time if result.lmm_time > 0 else 0
            logger.info(f"GEMMA throughput: {throughput:.0f} SNPs/sec")

            # Load GEMMA results for validation
            assoc_file = output_dir / "output" / "assoc.assoc.txt"
            if assoc_file.exists():
                result.assoc_df = pd.read_csv(assoc_file, sep="\t")

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            profiler.error("gemma", "lmm", e)

    return result


def validate_jamma_vs_gemma(
    jamma_results: list, gemma_df: pd.DataFrame
) -> dict[str, float]:
    """Validate JAMMA results against GEMMA reference.

    Args:
        jamma_results: List of AssocResult from JAMMA
        gemma_df: DataFrame from GEMMA assoc.txt

    Returns:
        Dict with validation metrics (max relative differences, pass/fail flags)
    """
    # Convert JAMMA results to DataFrame
    jamma_df = pd.DataFrame([vars(r) for r in jamma_results])

    # Merge on rs ID
    merged = pd.merge(
        gemma_df, jamma_df, left_on="rs", right_on="rs", suffixes=("_gemma", "_jamma")
    )

    if len(merged) == 0:
        logger.warning("No matching SNPs between JAMMA and GEMMA results")
        return {"error": "no_matches"}

    # Compute relative differences
    beta_diff = np.abs(merged["beta_gemma"] - merged["beta_jamma"])
    beta_rel = beta_diff / (np.abs(merged["beta_gemma"]) + 1e-10)

    pval_diff = np.abs(merged["p_wald_gemma"] - merged["p_wald_jamma"])
    pval_rel = pval_diff / (merged["p_wald_gemma"] + 1e-10)

    # Tolerances from CLAUDE.md
    beta_tol = 1e-6
    pval_tol = 1e-8

    validation = {
        "n_snps_compared": len(merged),
        "beta_max_rel_diff": float(beta_rel.max()),
        "beta_mean_rel_diff": float(beta_rel.mean()),
        "pval_max_rel_diff": float(pval_rel.max()),
        "pval_mean_rel_diff": float(pval_rel.mean()),
        "beta_pass": bool(beta_rel.max() < beta_tol),
        "pval_pass": bool(pval_rel.max() < pval_tol),
    }
    validation["overall_pass"] = validation["beta_pass"] and validation["pval_pass"]

    # Log results
    status = "PASS" if validation["overall_pass"] else "FAIL"
    logger.info(
        f"GEMMA validation: {status} | "
        f"beta_max_rel={validation['beta_max_rel_diff']:.2e} | "
        f"pval_max_rel={validation['pval_max_rel_diff']:.2e}"
    )

    return validation


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

    # Eigendecompose once here - reused by warmup and main LMM runs
    profiler.start("kinship", "eigendecomp", n_samples=n_samples)
    eigenvalues, eigenvectors = eigendecompose_kinship(K)
    eigen_elapsed = profiler.end("kinship", "eigendecomp")

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

        # Kinship and eigendecomposition (computed once, reused by LMM)
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
            profiler.error("lmm_jax", "association", e)

        # GEMMA benchmark (for smaller configs only)
        gemma_result = benchmark_gemma(
            genotypes, phenotype, profiler, config.n_snps_lmm
        )
        if gemma_result.kinship_time is not None:
            results["gemma_kinship_time"] = gemma_result.kinship_time
        if gemma_result.lmm_time is not None:
            results["gemma_lmm_time"] = gemma_result.lmm_time
            results["gemma_snps_per_sec"] = config.n_snps_lmm / gemma_result.lmm_time

        # Validate JAMMA results against GEMMA
        if gemma_result.assoc_df is not None and jax_assoc_results is not None:
            gemma_df = gemma_result.assoc_df
            validation = validate_jamma_vs_gemma(jax_assoc_results, gemma_df)
            results["gemma_validation_pass"] = validation.get("overall_pass", False)
            results["gemma_beta_max_rel_diff"] = validation.get("beta_max_rel_diff")
            results["gemma_pval_max_rel_diff"] = validation.get("pval_max_rel_diff")

        # Speedups (JAMMA JAX vs GEMMA only)
        if results.get("gemma_lmm_time") and results.get("lmm_jax_time"):
            gemma_t = results["gemma_lmm_time"]
            results["jax_vs_gemma"] = gemma_t / results["lmm_jax_time"]

    except Exception as e:
        profiler.error("benchmark", "main", e)
        logger.exception(f"Benchmark failed: {e}")

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
BENCHMARK_SCALES = ["small", "medium", "large", "xlarge", "target"]

all_results = []
all_events = []

timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
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

# Performance Comparison: JAMMA vs GEMMA
print("\n" + "=" * 60)
print("JAMMA vs GEMMA COMPARISON")
print("=" * 60)

for _, row in results_df.iterrows():
    cfg = row["config"]
    print(f"\n{cfg}:")

    gemma_lmm = row.get("gemma_lmm_time")
    jax_lmm = row.get("lmm_jax_time")
    validation_pass = row.get("gemma_validation_pass")

    if gemma_lmm:
        print(f"  GEMMA:      {gemma_lmm:7.2f}s")
        # Show validation status
        if validation_pass is not None:
            status = "PASS" if validation_pass else "FAIL"
            beta_diff = row.get("gemma_beta_max_rel_diff", 0)
            pval_diff = row.get("gemma_pval_max_rel_diff", 0)
            print(f"  Validation: {status} (β={beta_diff:.2e}, p={pval_diff:.2e})")
    else:
        print("  GEMMA:      (skipped - too large or unavailable)")

    if jax_lmm:
        print(f"  JAMMA JAX:  {jax_lmm:7.2f}s", end="")
        if gemma_lmm:
            ratio = gemma_lmm / jax_lmm
            print(f"  ({ratio:.2f}x vs GEMMA)")
        else:
            print()

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## GEMMA Validation (10K Scale)
# MAGIC
# MAGIC Compare JAMMA results against GEMMA reference at 10K samples.
# MAGIC This validates numerical equivalence before trusting large-scale results.
# MAGIC
# MAGIC **GEMMA Binary:**
# MAGIC Set `GEMMA_PATH` env var or modify the value at the top of the notebook.

# COMMAND ----------


def validate_vs_gemma(
    n_samples: int = 10_000,
    n_snps: int = 1_000,
    seed: int = 42,
    gemma_path: str | None = None,
) -> dict | None:
    """Validate JAMMA against GEMMA at small scale.

    Runs both tools on synthetic data and compares results.
    Requires GEMMA binary in PATH or specified via gemma_path.

    Args:
        n_samples: Number of samples for validation.
        n_snps: Number of SNPs for validation.
        seed: Random seed for reproducibility.
        gemma_path: Path to GEMMA binary. If None, searches PATH and GEMMA_PATH.

    Returns:
        Dict with comparison metrics, or None if GEMMA not available.
    """
    # Find GEMMA binary
    gemma_bin = gemma_path or GEMMA_PATH or shutil.which("gemma")
    if not gemma_bin or not Path(gemma_bin).exists():
        logger.warning("GEMMA not found - skipping validation")
        logger.info("Set GEMMA_PATH variable or pass gemma_path argument")
        return None

    logger.info(f"Running GEMMA validation: {n_samples:,} samples x {n_snps:,} SNPs")

    # Generate synthetic data
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, n_snps)

    genotypes = np.zeros((n_samples, n_snps), dtype=np.float32)
    for j in range(n_snps):
        p = maf[j]
        probs = [(1 - p) ** 2, 2 * p * (1 - p), p**2]
        genotypes[:, j] = rng.choice([0, 1, 2], size=n_samples, p=probs)

    phenotype = rng.standard_normal(n_samples)
    phenotype = (phenotype - phenotype.mean()) / phenotype.std()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        prefix = tmpdir / "test"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Write PLINK files using shared function
        _write_plink_files(str(prefix), genotypes, phenotype)

        # Run GEMMA kinship and LMM - stream output in real-time
        def run_gemma_streaming(cmd, cwd, timeout=300):
            """Run GEMMA with real-time output streaming."""
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            while True:
                output = proc.stdout.readline()
                if output == "" and proc.poll() is not None:
                    break
                if output:
                    print(output.strip())
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd[0])

        try:
            run_gemma_streaming(
                [gemma_bin, "-bfile", str(prefix), "-gk", "1", "-o", "kinship"],
                output_dir,
            )

            kinship_file = output_dir / "output" / "kinship.cXX.txt"
            run_gemma_streaming(
                [
                    gemma_bin,
                    "-bfile",
                    str(prefix),
                    "-k",
                    str(kinship_file),
                    "-lmm",
                    "1",
                    "-o",
                    "assoc",
                ],
                output_dir,
            )
        except subprocess.TimeoutExpired:
            logger.error("GEMMA timed out")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"GEMMA failed with return code {e.returncode}")
            return None

        # Load GEMMA results
        gemma_kinship = np.loadtxt(output_dir / "output" / "kinship.cXX.txt")
        gemma_assoc = pd.read_csv(output_dir / "output" / "assoc.assoc.txt", sep="\t")

    # Run JAMMA
    logger.info("Running JAMMA...")
    snp_info = [
        {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
        for j in range(n_snps)
    ]

    jamma_kinship = compute_centered_kinship(genotypes, check_memory=False)
    jamma_results = run_lmm_association_jax(
        genotypes=genotypes,
        phenotypes=phenotype,
        kinship=jamma_kinship,
        snp_info=snp_info,
        show_progress=True,
    )

    # Compare kinship
    kinship_max_diff = np.max(np.abs(jamma_kinship - gemma_kinship))
    kinship_rel_diff = kinship_max_diff / (np.abs(gemma_kinship).max() + 1e-10)

    # Compare association results using shared validation logic
    validation = validate_jamma_vs_gemma(jamma_results, gemma_assoc)

    results = {
        "n_samples": n_samples,
        "n_snps": n_snps,
        "kinship_max_abs_diff": float(kinship_max_diff),
        "kinship_max_rel_diff": float(kinship_rel_diff),
        "beta_max_rel_diff": validation.get("beta_max_rel_diff", float("nan")),
        "beta_mean_rel_diff": validation.get("beta_mean_rel_diff", float("nan")),
        "pval_max_rel_diff": validation.get("pval_max_rel_diff", float("nan")),
        "pval_mean_rel_diff": validation.get("pval_mean_rel_diff", float("nan")),
        "kinship_pass": kinship_rel_diff < 1e-8,
        "beta_pass": validation.get("beta_pass", False),
        "pval_pass": validation.get("pval_pass", False),
    }
    results["overall_pass"] = all(
        [results["kinship_pass"], results["beta_pass"], results["pval_pass"]]
    )

    # Log results
    logger.info("=" * 60)
    logger.info("GEMMA VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Kinship max rel diff: {results['kinship_max_rel_diff']:.2e}")
    logger.info(f"Beta max rel diff:    {results['beta_max_rel_diff']:.2e}")
    logger.info(f"P-value max rel diff: {results['pval_max_rel_diff']:.2e}")
    logger.info(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")

    return results


# COMMAND ----------

# Run GEMMA validation (uncomment to execute)
# gemma_results = validate_vs_gemma(n_samples=10_000, n_snps=1_000)
# if gemma_results:
#     display(pd.DataFrame([gemma_results]))  # noqa: F821
