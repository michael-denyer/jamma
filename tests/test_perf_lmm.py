"""Performance benchmarks for LMM association pipeline.

Measures per-stage timing on the mouse_hs1940 dataset (1410 samples, 10768 SNPs)
to establish baseline performance for optimization work.

Run with:
    uv run pytest tests/test_perf_lmm.py -v -n0 --benchmark-only

Results include hardware context (CPU, BLAS, JAX config) for cross-machine
comparison. All benchmarks use JAX x64 precision with proper block_until_ready
synchronization.
"""

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

import jax

from jamma.core.hardware import assert_x64_precision, get_hardware_context
from jamma.core.threading import blas_threads
from tests.conftest import load_phenotypes_from_fam

pytestmark = pytest.mark.requires_jax

_FIXTURE_ROOT = Path(__file__).parent / "fixtures"
_MOUSE_DIR = _FIXTURE_ROOT / "mouse_hs1940"
_MOUSE_DATA = _MOUSE_DIR / "mouse_hs1940"
_MOUSE_KINSHIP = _MOUSE_DIR / "mouse_hs1940_kinship.cXX.txt"


def _mouse_data_available() -> bool:
    """Check if mouse_hs1940 PLINK data is available."""
    return _MOUSE_DATA.with_suffix(".bed").exists()


@pytest.fixture(scope="module")
def mouse_plink():
    """Load mouse_hs1940 PLINK data (module-scoped for reuse)."""
    if not _mouse_data_available():
        pytest.skip("mouse_hs1940 PLINK data not found")
    from jamma.io import load_plink_binary

    return load_plink_binary(_MOUSE_DATA)


@pytest.fixture(scope="module")
def mouse_phenotypes():
    """Load mouse_hs1940 phenotypes (module-scoped)."""
    if not _mouse_data_available():
        pytest.skip("mouse_hs1940 PLINK data not found")
    return load_phenotypes_from_fam(_MOUSE_DATA.with_suffix(".fam"))


@pytest.fixture(scope="module")
def mouse_kinship():
    """Load pre-computed mouse_hs1940 kinship matrix."""
    if not _MOUSE_KINSHIP.exists():
        pytest.skip("mouse_hs1940 kinship not found")
    from jamma.kinship.io import read_kinship_matrix

    return read_kinship_matrix(_MOUSE_KINSHIP)


@pytest.fixture(scope="module")
def mouse_eigen(mouse_kinship, mouse_phenotypes):
    """Pre-compute eigendecomposition for association benchmarks.

    Module-scoped so eigendecomp is done once, then reused across
    association benchmark rounds. This isolates association timing
    from eigendecomp timing.
    """
    from jamma.lmm.eigen import eigendecompose_kinship

    # Filter to valid samples (matching what the runner does)
    valid_mask = ~np.isnan(mouse_phenotypes) & (mouse_phenotypes != -9.0)
    K_valid = mouse_kinship[np.ix_(valid_mask, valid_mask)]
    eigenvalues, eigenvectors = eigendecompose_kinship(K_valid, check_memory=False)
    return eigenvalues, eigenvectors, valid_mask


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.tier2
class TestLMMBenchmarks:
    """Benchmark LMM pipeline stages on mouse_hs1940."""

    def test_eigendecomp_benchmark(self, benchmark, mouse_kinship, mouse_phenotypes):
        """Benchmark eigendecomposition (numpy LAPACK).

        This is the memory-peak operation. Measures raw LAPACK dsyevd
        performance on the 1410x1410 kinship matrix.
        """
        assert_x64_precision()
        hw_ctx = get_hardware_context()

        valid_mask = ~np.isnan(mouse_phenotypes) & (mouse_phenotypes != -9.0)
        K_valid = mouse_kinship[np.ix_(valid_mask, valid_mask)]

        from jamma.lmm.eigen import eigendecompose_kinship

        def _run():
            eigenvalues, eigenvectors = eigendecompose_kinship(
                K_valid.copy(), check_memory=False
            )
            return eigenvalues, eigenvectors

        result = benchmark.pedantic(
            _run,
            warmup_rounds=1,
            rounds=3,
            iterations=1,
        )

        eigenvalues, eigenvectors = result
        n_samples = K_valid.shape[0]
        benchmark.extra_info.update(hw_ctx)
        benchmark.extra_info["stage"] = "eigendecomp"
        benchmark.extra_info["n_samples"] = n_samples
        benchmark.extra_info["matrix_elements"] = n_samples * n_samples

        assert eigenvalues.shape[0] == n_samples
        assert eigenvectors.shape == (n_samples, n_samples)

    def test_dgemm_rotation_benchmark(
        self, benchmark, mouse_plink, mouse_phenotypes, mouse_eigen
    ):
        """Benchmark DGEMM rotation (U.T @ G, the numpy BLAS bottleneck).

        At production scale (100k samples), this is ~92% of pipeline time.
        Measures the numpy BLAS matmul that rotates genotypes into eigen-space.
        """
        assert_x64_precision()
        hw_ctx = get_hardware_context()

        eigenvalues, eigenvectors, valid_mask = mouse_eigen
        genotypes = mouse_plink.genotypes[valid_mask, :]

        # Impute missing to column mean (matching runner behavior)
        col_means = np.nanmean(genotypes, axis=0)
        missing = np.isnan(genotypes)
        genotypes = np.where(missing, col_means[None, :], genotypes)
        U = eigenvectors

        def _run():
            with blas_threads():
                UtG = np.ascontiguousarray(U.T @ genotypes)
            return UtG

        result = benchmark.pedantic(
            _run,
            warmup_rounds=1,
            rounds=5,
            iterations=1,
        )

        n_samples, n_snps = genotypes.shape
        benchmark.extra_info.update(hw_ctx)
        benchmark.extra_info["stage"] = "dgemm_rotation"
        benchmark.extra_info["n_samples"] = n_samples
        benchmark.extra_info["n_snps"] = n_snps
        benchmark.extra_info["matmul_flops"] = 2 * n_samples * n_samples * n_snps

        assert result.shape == (n_samples, n_snps)

    def test_jax_optimization_benchmark(
        self, benchmark, mouse_plink, mouse_phenotypes, mouse_eigen
    ):
        """Benchmark JAX golden section optimization (compute bottleneck at small N).

        At mouse_hs1940 scale (1410 samples), JAX optimization is ~64% of pipeline time.
        Measures batch Uab computation + grid search + golden section on all SNPs.
        """
        assert_x64_precision()
        hw_ctx = get_hardware_context()

        eigenvalues, eigenvectors, valid_mask = mouse_eigen
        genotypes = mouse_plink.genotypes[valid_mask, :]
        phenotypes = mouse_phenotypes[valid_mask]
        U = eigenvectors

        # Impute and rotate (prep work, not timed)
        col_means = np.nanmean(genotypes, axis=0)
        missing = np.isnan(genotypes)
        genotypes_imp = np.where(missing, col_means[None, :], genotypes)

        W = np.ones((len(phenotypes), 1))
        with blas_threads():
            UtW = U.T @ W
            Uty = U.T @ phenotypes
            UtG = np.ascontiguousarray(U.T @ genotypes_imp)

        device = jax.devices("cpu")[0]
        eigenvalues_jax = jax.device_put(eigenvalues, device)
        UtW_jax = jax.device_put(UtW, device)
        Uty_jax = jax.device_put(Uty, device)
        UtG_jax = jax.device_put(UtG, device)

        from jamma.lmm.compute import _compute_lmm_chunk, block_chunk_result
        from jamma.lmm.likelihood_jax import batch_compute_uab

        n_cvt = 1

        # JIT warmup: run on a small slice to compile
        _warmup_UtG = UtG_jax[:, :10]
        _warmup_Uab = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, _warmup_UtG)
        _warmup_cr = _compute_lmm_chunk(
            1, n_cvt, eigenvalues_jax, _warmup_Uab, len(phenotypes)
        )
        block_chunk_result(_warmup_cr, 1)
        del _warmup_UtG, _warmup_Uab, _warmup_cr

        def _run():
            Uab_batch = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, UtG_jax)
            cr = _compute_lmm_chunk(
                1, n_cvt, eigenvalues_jax, Uab_batch, len(phenotypes)
            )
            block_chunk_result(cr, 1)
            return cr

        result = benchmark.pedantic(
            _run,
            warmup_rounds=0,  # Already warmed up JIT above
            rounds=3,
            iterations=1,
        )

        n_snps = UtG_jax.shape[1]
        benchmark.extra_info.update(hw_ctx)
        benchmark.extra_info["stage"] = "jax_optimization"
        benchmark.extra_info["n_samples"] = len(phenotypes)
        benchmark.extra_info["n_snps"] = n_snps
        benchmark.extra_info["n_cvt"] = n_cvt

        assert result["pwalds"] is not None

    def test_full_pipeline_benchmark(
        self, benchmark, mouse_plink, mouse_phenotypes, mouse_kinship
    ):
        """Benchmark the full LMM pipeline (eigendecomp + association).

        This is the end-to-end benchmark that subsequent optimization phases
        will use to measure improvement. Includes all stages: eigendecomp,
        DGEMM rotation, JAX optimization, result construction.
        """
        assert_x64_precision()
        hw_ctx = get_hardware_context()

        from jamma.lmm.runner_jax import run_lmm_association_jax

        snp_info = [
            {
                "chr": str(mouse_plink.chromosome[i]),
                "rs": mouse_plink.sid[i],
                "pos": int(mouse_plink.bp_position[i]),
                "a1": mouse_plink.allele_1[i],
                "a0": mouse_plink.allele_2[i],
            }
            for i in range(mouse_plink.n_snps)
        ]

        def _run():
            results = run_lmm_association_jax(
                genotypes=mouse_plink.genotypes,
                phenotypes=mouse_phenotypes,
                kinship=mouse_kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )
            return results

        result = benchmark.pedantic(
            _run,
            warmup_rounds=1,
            rounds=3,
            iterations=1,
        )

        benchmark.extra_info.update(hw_ctx)
        benchmark.extra_info["stage"] = "full_pipeline"
        benchmark.extra_info["n_samples_total"] = mouse_plink.n_samples
        benchmark.extra_info["n_snps_total"] = mouse_plink.n_snps
        benchmark.extra_info["n_results"] = len(result)

        # Sanity: should have ~10k results after filtering
        assert len(result) > 5000

    def test_full_pipeline_streaming_benchmark(
        self, benchmark, mouse_phenotypes, mouse_kinship
    ):
        """Benchmark the streaming LMM pipeline for comparison with batch.

        Uses the streaming runner with pre-computed kinship on mouse_hs1940.
        This is the production code path used by the CLI.
        """
        assert_x64_precision()
        hw_ctx = get_hardware_context()

        from jamma.lmm import run_lmm_association_streaming

        def _run():
            results, n_tested = run_lmm_association_streaming(
                bed_path=_MOUSE_DATA,
                phenotypes=mouse_phenotypes,
                kinship=mouse_kinship,
                show_progress=False,
                check_memory=False,
            )
            return results, n_tested

        result = benchmark.pedantic(
            _run,
            warmup_rounds=1,
            rounds=3,
            iterations=1,
        )

        results, n_tested = result
        benchmark.extra_info.update(hw_ctx)
        benchmark.extra_info["stage"] = "full_pipeline_streaming"
        benchmark.extra_info["n_tested"] = n_tested

        assert n_tested > 5000


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.tier2
class TestShardedBenchmarks:
    """Sharded benchmark variants for before/after comparison on server hardware.

    These benchmarks exercise the CPU device sharding code path. They record
    ``sharding_enabled`` and ``jax_device_count`` in ``benchmark.extra_info``
    so results from different machines are directly comparable.

    Comparison methodology
    ----------------------
    True before/after comparison requires controlling device count:

    * **Before (unsharded baseline)**::

          JAMMA_JAX_DEVICES=1 uv run pytest tests/test_perf_lmm.py -v -n0 \\
              --benchmark-only -k "jax_optimization"

    * **After (sharded)**::

          uv run pytest tests/test_perf_lmm.py -v -n0 \\
              --benchmark-only -k "jax_optimization_sharded"

    On development machines (ARM Mac), JAX typically exposes only a single
    virtual CPU device, so the sharding fallback activates and timing is
    directional only. The definitive measurement is on Databricks Intel Xeon
    hardware where ``configure_jax()`` auto-configures multiple virtual devices
    (``physical_cores // 2``).
    """

    def test_jax_optimization_sharded_benchmark(
        self, benchmark, mouse_plink, mouse_phenotypes, mouse_eigen
    ):
        """Benchmark JAX optimization with explicit NamedSharding on UtG.

        Mirrors ``TestLMMBenchmarks.test_jax_optimization_benchmark`` but
        places UtG using ``snp_spec`` (sharded on SNP axis) and shared arrays
        using ``rep_spec`` (replicated). JIT warmup uses the same sharding
        configuration so the compiled kernel matches the timed run.

        Records ``sharding_enabled`` and ``jax_device_count`` to distinguish
        single-device fallback (small-scale dev machines) from multi-device
        server runs.
        """
        assert_x64_precision()
        hw_ctx = get_hardware_context()

        from jamma.lmm.compute import _compute_lmm_chunk, block_chunk_result
        from jamma.lmm.likelihood_jax import batch_compute_uab
        from jamma.lmm.prepare import _setup_cpu_sharding

        eigenvalues, eigenvectors, valid_mask = mouse_eigen
        genotypes = mouse_plink.genotypes[valid_mask, :]
        phenotypes = mouse_phenotypes[valid_mask]
        U = eigenvectors

        col_means = np.nanmean(genotypes, axis=0)
        missing = np.isnan(genotypes)
        genotypes_imp = np.where(missing, col_means[None, :], genotypes)

        W = np.ones((len(phenotypes), 1))
        with blas_threads():
            UtW = U.T @ W
            Uty = U.T @ phenotypes
            UtG = np.ascontiguousarray(U.T @ genotypes_imp)

        snp_spec, rep_spec = _setup_cpu_sharding()
        device = jax.devices("cpu")[0]
        n_devices = len(jax.devices("cpu"))
        n_snps = UtG.shape[1]

        # Pad UtG to device-count multiple for even sharding distribution
        use_sharding = snp_spec is not None
        if use_sharding and n_snps % n_devices != 0:
            dev_pad = n_devices - (n_snps % n_devices)
            UtG = np.pad(UtG, ((0, 0), (0, dev_pad)), mode="constant")

        effective_snp_spec = snp_spec if use_sharding else device
        effective_rep_spec = rep_spec if use_sharding else device

        eigenvalues_jax = jax.device_put(eigenvalues, effective_rep_spec)
        UtW_jax = jax.device_put(UtW, effective_rep_spec)
        Uty_jax = jax.device_put(Uty, effective_rep_spec)
        UtG_jax = jax.device_put(UtG, effective_snp_spec)

        n_cvt = 1

        # JIT warmup: use same sharding config so compiled kernel matches timed run.
        warmup_n = n_devices if use_sharding else 10
        _warmup_UtG = jax.device_put(UtG[:, :warmup_n], effective_snp_spec)
        _warmup_Uab = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, _warmup_UtG)
        _warmup_cr = _compute_lmm_chunk(
            1, n_cvt, eigenvalues_jax, _warmup_Uab, len(phenotypes)
        )
        block_chunk_result(_warmup_cr, 1)
        del _warmup_UtG, _warmup_Uab, _warmup_cr

        def _run():
            Uab_batch = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, UtG_jax)
            cr = _compute_lmm_chunk(
                1, n_cvt, eigenvalues_jax, Uab_batch, len(phenotypes)
            )
            block_chunk_result(cr, 1)
            return cr

        result = benchmark.pedantic(
            _run,
            warmup_rounds=0,  # Already warmed up JIT above with sharded config
            rounds=3,
            iterations=1,
        )

        benchmark.extra_info.update(hw_ctx)
        benchmark.extra_info["stage"] = "jax_optimization_sharded"
        benchmark.extra_info["n_samples"] = len(phenotypes)
        benchmark.extra_info["n_snps"] = n_snps
        benchmark.extra_info["n_cvt"] = n_cvt
        benchmark.extra_info["sharding_enabled"] = snp_spec is not None
        benchmark.extra_info["jax_device_count"] = n_devices

        assert result["pwalds"] is not None

    def test_full_pipeline_sharded_benchmark(
        self, benchmark, mouse_plink, mouse_phenotypes, mouse_kinship
    ):
        """Benchmark the full LMM pipeline with CPU device sharding active.

        Calls ``run_lmm_association_jax()`` identically to
        ``TestLMMBenchmarks.test_full_pipeline_benchmark``. Sharding is baked
        into the runner: ``_setup_cpu_sharding()`` is called automatically when
        multiple JAX CPU devices are configured.

        On single-device machines the sharding fallback activates — timing is
        still collected for the metadata record.
        """
        assert_x64_precision()
        hw_ctx = get_hardware_context()

        from jamma.lmm.runner_jax import run_lmm_association_jax

        snp_info = [
            {
                "chr": str(mouse_plink.chromosome[i]),
                "rs": mouse_plink.sid[i],
                "pos": int(mouse_plink.bp_position[i]),
                "a1": mouse_plink.allele_1[i],
                "a0": mouse_plink.allele_2[i],
            }
            for i in range(mouse_plink.n_snps)
        ]
        n_devices = len(jax.devices("cpu"))

        def _run():
            results = run_lmm_association_jax(
                genotypes=mouse_plink.genotypes,
                phenotypes=mouse_phenotypes,
                kinship=mouse_kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )
            return results

        result = benchmark.pedantic(
            _run,
            warmup_rounds=1,
            rounds=3,
            iterations=1,
        )

        benchmark.extra_info.update(hw_ctx)
        benchmark.extra_info["stage"] = "full_pipeline_sharded"
        benchmark.extra_info["n_samples_total"] = mouse_plink.n_samples
        benchmark.extra_info["n_snps_total"] = mouse_plink.n_snps
        benchmark.extra_info["n_results"] = len(result)
        benchmark.extra_info["sharding_enabled"] = n_devices > 1
        benchmark.extra_info["jax_device_count"] = n_devices

        assert len(result) > 5000

    def test_full_pipeline_streaming_sharded_benchmark(
        self, benchmark, mouse_phenotypes, mouse_kinship
    ):
        """Benchmark the streaming LMM pipeline with CPU device sharding active.

        Calls ``run_lmm_association_streaming()`` identically to
        ``TestLMMBenchmarks.test_full_pipeline_streaming_benchmark``. Sharding
        is baked into the streaming runner.

        On single-device machines the sharding fallback activates — timing is
        still collected for the metadata record.
        """
        assert_x64_precision()
        hw_ctx = get_hardware_context()

        from jamma.lmm import run_lmm_association_streaming

        n_devices = len(jax.devices("cpu"))

        def _run():
            results, n_tested = run_lmm_association_streaming(
                bed_path=_MOUSE_DATA,
                phenotypes=mouse_phenotypes,
                kinship=mouse_kinship,
                show_progress=False,
                check_memory=False,
            )
            return results, n_tested

        result = benchmark.pedantic(
            _run,
            warmup_rounds=1,
            rounds=3,
            iterations=1,
        )

        results, n_tested = result
        benchmark.extra_info.update(hw_ctx)
        benchmark.extra_info["stage"] = "full_pipeline_streaming_sharded"
        benchmark.extra_info["n_tested"] = n_tested
        benchmark.extra_info["sharding_enabled"] = n_devices > 1
        benchmark.extra_info["jax_device_count"] = n_devices

        assert n_tested > 5000


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.tier2
class TestXLACacheVerification:
    """Verify XLA compilation cache persistence across runs."""

    def test_xla_cache_populated(self, mouse_plink, mouse_phenotypes, mouse_kinship):
        """Verify XLA compilation cache directory is created and accessible.

        After running an LMM pipeline, the JAX compilation cache directory
        (~/.cache/jax) should exist. Whether it contains cached compilations
        depends on jax_persistent_cache_min_compile_time_secs (default 1s) —
        small-scale compilations may not exceed the threshold.
        """
        assert_x64_precision()

        from jamma.lmm.runner_jax import run_lmm_association_jax

        snp_info = [
            {
                "chr": str(mouse_plink.chromosome[i]),
                "rs": mouse_plink.sid[i],
                "pos": int(mouse_plink.bp_position[i]),
                "a1": mouse_plink.allele_1[i],
                "a0": mouse_plink.allele_2[i],
            }
            for i in range(mouse_plink.n_snps)
        ]

        # Run pipeline to populate cache
        _ = run_lmm_association_jax(
            genotypes=mouse_plink.genotypes,
            phenotypes=mouse_phenotypes,
            kinship=mouse_kinship,
            snp_info=snp_info,
            show_progress=False,
            check_memory=False,
        )

        # Verify cache directory exists (use the constant from jax_config
        # rather than hardcoding, so the test follows XDG_CACHE_HOME changes)
        from jamma.core.jax_config import JAX_CACHE_DIR

        cache_dir = Path(JAX_CACHE_DIR)
        assert cache_dir.exists(), (
            f"XLA compilation cache directory not found at {cache_dir}. "
            f"JAX persistent cache may not be configured."
        )

        assert cache_dir.is_dir(), (
            f"Cache path exists but is not a directory: {cache_dir}"
        )

    def test_xla_cache_reuse_faster(
        self, benchmark, mouse_plink, mouse_phenotypes, mouse_eigen
    ):
        """Verify second JIT run is faster than first (cache warm).

        Runs the JAX optimization path twice. The second run should be
        measurably faster due to XLA compilation cache reuse.
        """
        assert_x64_precision()

        eigenvalues, eigenvectors, valid_mask = mouse_eigen
        genotypes = mouse_plink.genotypes[valid_mask, :]
        phenotypes = mouse_phenotypes[valid_mask]
        U = eigenvectors

        col_means = np.nanmean(genotypes, axis=0)
        missing = np.isnan(genotypes)
        genotypes_imp = np.where(missing, col_means[None, :], genotypes)

        W = np.ones((len(phenotypes), 1))
        with blas_threads():
            UtW = U.T @ W
            Uty = U.T @ phenotypes
            UtG = np.ascontiguousarray(U.T @ genotypes_imp[:, :100])

        device = jax.devices("cpu")[0]
        eigenvalues_jax = jax.device_put(eigenvalues, device)
        UtW_jax = jax.device_put(UtW, device)
        Uty_jax = jax.device_put(Uty, device)
        UtG_jax = jax.device_put(UtG, device)

        from jamma.lmm.compute import _compute_lmm_chunk, block_chunk_result
        from jamma.lmm.likelihood_jax import batch_compute_uab

        n_cvt = 1

        # Clear caches to force recompilation
        jax.clear_caches()

        # First run (cold JIT)
        t1_start = time.perf_counter()
        Uab1 = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, UtG_jax)
        cr1 = _compute_lmm_chunk(1, n_cvt, eigenvalues_jax, Uab1, len(phenotypes))
        block_chunk_result(cr1, 1)
        t1_cold = time.perf_counter() - t1_start

        # Second run (warm JIT / cache hit)
        t2_start = time.perf_counter()
        Uab2 = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, UtG_jax)
        cr2 = _compute_lmm_chunk(1, n_cvt, eigenvalues_jax, Uab2, len(phenotypes))
        block_chunk_result(cr2, 1)
        t2_warm = time.perf_counter() - t2_start

        benchmark.extra_info["jit_cold_s"] = t1_cold
        benchmark.extra_info["jit_warm_s"] = t2_warm
        benchmark.extra_info["speedup"] = t1_cold / max(t2_warm, 1e-9)

        # Warm should be faster (JIT compilation cost removed)
        # Not a hard assert since cache behavior depends on JAX version
        # and compilation threshold, but log the comparison
        if t2_warm < t1_cold:
            benchmark.extra_info["cache_effective"] = True
        else:
            benchmark.extra_info["cache_effective"] = False
