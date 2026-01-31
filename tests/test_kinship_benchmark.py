"""Performance benchmarks for kinship computation.

These benchmarks compare JAMMA's JAX implementation against a naive NumPy
baseline to measure the performance improvement from JAX JIT compilation.

Run with: uv run pytest tests/test_kinship_benchmark.py -v -n0 --benchmark-only
"""

import numpy as np
import pytest


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
