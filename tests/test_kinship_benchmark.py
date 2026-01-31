"""Performance benchmarks for kinship computation.

These benchmarks compare JAMMA's JAX implementation against:
1. GEMMA (the reference C++ implementation) - the target to beat
2. Naive NumPy baseline - for understanding JAX JIT improvement

Run with: uv run pytest tests/test_kinship_benchmark.py -v -n0 --benchmark-only

For GEMMA comparison, requires Docker:
  docker pull quay.io/biocontainers/gemma:0.98.5--ha36d3ea_0
"""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


GEMMA_DOCKER_IMAGE = "quay.io/biocontainers/gemma:0.98.5--ha36d3ea_0"
EXAMPLE_DATA = Path("legacy/example/mouse_hs1940")


@pytest.fixture
def medium_genotypes():
    """Medium-size genotype data for benchmarking (500 samples, 5000 SNPs)."""
    rng = np.random.default_rng(42)
    # Use smaller size for faster CI runs
    return rng.integers(0, 3, size=(500, 5000)).astype(np.float64)


@pytest.fixture
def genotypes_with_missing(medium_genotypes):
    """Same genotypes with 5% missing data."""
    rng = np.random.default_rng(43)
    mask = rng.random(medium_genotypes.shape) < 0.05
    result = medium_genotypes.copy()
    result[mask] = np.nan
    return result


def numpy_kinship(X):
    """Naive NumPy kinship computation for baseline comparison.

    Implements the same centered kinship algorithm as JAMMA,
    but using pure NumPy without JAX JIT compilation.
    """
    # Center (impute missing to mean first)
    means = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), means, X)
    X_centered = X_filled - means
    # Compute K = X_c @ X_c.T / n_snps
    K = X_centered @ X_centered.T / X.shape[1]
    return K


class TestKinshipBenchmarks:
    """Benchmark kinship computation performance."""

    def test_kinship_jax_performance(self, benchmark, medium_genotypes):
        """Benchmark JAX kinship computation."""
        from jamma.kinship import compute_centered_kinship

        # Use pedantic mode with warmup to exclude JIT compilation time
        result = benchmark.pedantic(
            compute_centered_kinship,
            args=(medium_genotypes,),
            warmup_rounds=1,
            rounds=3,
            iterations=1,
        )

        # Store metadata for reporting
        benchmark.extra_info['n_samples'] = medium_genotypes.shape[0]
        benchmark.extra_info['n_snps'] = medium_genotypes.shape[1]
        benchmark.extra_info['implementation'] = 'jax'

        # Basic sanity check
        assert result.shape == (500, 500)

    def test_kinship_numpy_baseline(self, benchmark, medium_genotypes):
        """Benchmark naive NumPy baseline for comparison."""
        result = benchmark.pedantic(
            numpy_kinship,
            args=(medium_genotypes,),
            warmup_rounds=0,
            rounds=3,
            iterations=1,
        )

        benchmark.extra_info['n_samples'] = medium_genotypes.shape[0]
        benchmark.extra_info['n_snps'] = medium_genotypes.shape[1]
        benchmark.extra_info['implementation'] = 'numpy'

        assert result.shape == (500, 500)

    def test_kinship_with_missing_data(self, benchmark, genotypes_with_missing):
        """Benchmark with missing data handling."""
        from jamma.kinship import compute_centered_kinship

        result = benchmark.pedantic(
            compute_centered_kinship,
            args=(genotypes_with_missing,),
            warmup_rounds=1,
            rounds=3,
            iterations=1,
        )

        benchmark.extra_info['n_samples'] = genotypes_with_missing.shape[0]
        benchmark.extra_info['n_snps'] = genotypes_with_missing.shape[1]
        benchmark.extra_info['missing_rate'] = '5%'

        assert result.shape == (500, 500)


class TestKinshipScaling:
    """Test kinship computation scaling with data size."""

    @pytest.fixture
    def small_genotypes(self):
        """Small genotype data (100 samples, 1000 SNPs)."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 3, size=(100, 1000)).astype(np.float64)

    def test_small_jax_benchmark(self, benchmark, small_genotypes):
        """Benchmark JAX on small data."""
        from jamma.kinship import compute_centered_kinship

        result = benchmark.pedantic(
            compute_centered_kinship,
            args=(small_genotypes,),
            warmup_rounds=1,
            rounds=5,
            iterations=1,
        )

        benchmark.extra_info['n_samples'] = small_genotypes.shape[0]
        benchmark.extra_info['n_snps'] = small_genotypes.shape[1]
        benchmark.extra_info['implementation'] = 'jax'
        benchmark.extra_info['size'] = 'small'

        assert result.shape == (100, 100)

    def test_small_numpy_benchmark(self, benchmark, small_genotypes):
        """Benchmark NumPy on small data."""
        result = benchmark.pedantic(
            numpy_kinship,
            args=(small_genotypes,),
            warmup_rounds=0,
            rounds=5,
            iterations=1,
        )

        benchmark.extra_info['n_samples'] = small_genotypes.shape[0]
        benchmark.extra_info['n_snps'] = small_genotypes.shape[1]
        benchmark.extra_info['implementation'] = 'numpy'
        benchmark.extra_info['size'] = 'small'

        assert result.shape == (100, 100)


class TestJammaVsGemma:
    """Compare JAMMA performance against GEMMA (the target to beat)."""

    @pytest.fixture
    def mouse_genotypes(self):
        """Load mouse_hs1940 genotypes for real-world comparison."""
        from jamma.io import load_plink_binary

        plink_data = load_plink_binary(EXAMPLE_DATA)
        return plink_data.genotypes

    def _run_gemma_docker(self, tmpdir: Path) -> float:
        """Run GEMMA via Docker and return execution time in seconds."""
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{Path.cwd()}:/data",
            "-v", f"{tmpdir}:/output",
            GEMMA_DOCKER_IMAGE,
            "gemma",
            "-bfile", "/data/legacy/example/mouse_hs1940",
            "-gk", "1",
            "-o", "bench",
            "-outdir", "/output",
        ]

        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            pytest.skip(f"GEMMA Docker failed: {result.stderr}")

        return elapsed

    @pytest.mark.slow
    def test_jamma_vs_gemma_mouse_hs1940(self, mouse_genotypes):
        """Compare JAMMA and GEMMA on mouse_hs1940 dataset.

        This is the critical benchmark - JAMMA must be faster than GEMMA.
        """
        from jamma.kinship import compute_centered_kinship

        # Check Docker is available
        docker_check = subprocess.run(
            ["docker", "info"], capture_output=True, text=True
        )
        if docker_check.returncode != 0:
            pytest.skip("Docker not available")

        # Warm up JAMMA (JIT compilation)
        _ = compute_centered_kinship(mouse_genotypes[:100, :1000])

        # Time JAMMA
        jamma_start = time.perf_counter()
        jamma_result = compute_centered_kinship(mouse_genotypes)
        jamma_time = time.perf_counter() - jamma_start

        # Time GEMMA
        with tempfile.TemporaryDirectory() as tmpdir:
            gemma_time = self._run_gemma_docker(Path(tmpdir))

        # Report results
        speedup = gemma_time / jamma_time
        print(f"\n{'='*60}")
        print("JAMMA vs GEMMA Performance Comparison")
        print(f"{'='*60}")
        print(f"Dataset: mouse_hs1940 (1940 samples, 12226 SNPs)")
        print(f"GEMMA time:  {gemma_time:.3f}s")
        print(f"JAMMA time:  {jamma_time:.3f}s")
        print(f"Speedup:     {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")
        print(f"{'='*60}\n")

        # Store for reporting
        assert jamma_result.shape == (1940, 1940)

        # CRITICAL: JAMMA must be faster than GEMMA
        assert jamma_time < gemma_time, (
            f"JAMMA ({jamma_time:.3f}s) must be faster than GEMMA ({gemma_time:.3f}s)"
        )
