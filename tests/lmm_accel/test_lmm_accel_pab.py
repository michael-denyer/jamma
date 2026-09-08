"""_lmm_accel C extension tests: Pab table construction and kernel performance.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel/_helpers.py.
"""

import numpy as np
import pytest


@pytest.mark.benchmark
class TestCExtensionPerformance:
    """Benchmark the C extension on realistic valid data.

    Hardware-sensitive — `2x` speedup is not a correctness invariant. Runs
    only under `--benchmark-only`; on machines with <4 physical cores it
    skips. Numerical correctness lives in ordinary tier-0 tests.
    """

    def test_native_wald_latency(self, benchmark):
        """Time the actual native fused Wald kernel on valid shared inputs."""
        from jamma.core.threading import get_physical_core_count
        from jamma.lmm import accel

        if not accel.available():
            pytest.skip("C extension not compiled")

        n_threads = get_physical_core_count()
        if n_threads < 4:
            pytest.skip(f"Benchmark needs >=4 physical cores; found {n_threads}")

        rng = np.random.default_rng(42)
        n_samples, n_snps = 500, 2000
        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))

        w = rng.standard_normal(n_samples)
        y = rng.standard_normal(n_samples)
        utg_t = np.ascontiguousarray(rng.standard_normal((n_snps, n_samples)))
        invariant = np.stack((w * w, w * y, y * y))
        workspace = accel.require().create_workspace_ncvt1_c(
            eigenvalues, invariant, w, y, n_samples, 1e-5, 1e5, 50, 20, lmm_mode=1
        )

        # Warmup: amortise OpenMP thread-pool startup before timing
        accel.require().compute_lmm_chunk_ncvt1_c(workspace, utg_t[:50], n_threads)

        # pytest-benchmark tracks native latency history; it asserts no speed ratio.
        benchmark(
            accel.require().compute_lmm_chunk_ncvt1_c,
            workspace,
            utg_t,
            n_threads,
        )
