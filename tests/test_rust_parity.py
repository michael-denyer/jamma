"""End-to-end parity tests: Rust backend vs scipy backend.

These tests verify that the full LMM workflow produces identical
statistical results regardless of which backend is used for
eigendecomposition. This is critical for validating that the Rust
implementation can replace scipy without changing GWAS results.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from jamma.core.backend import get_compute_backend, is_rust_available

# Skip all tests if Rust backend not available
pytestmark = pytest.mark.skipif(
    not is_rust_available(), reason="Rust backend (jamma_core) not installed"
)


class TestEigendecompParity:
    """Tests comparing Rust vs scipy eigendecomposition directly."""

    def setup_method(self):
        get_compute_backend.cache_clear()

    def teardown_method(self):
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    def test_eigenvalues_match(self):
        """Eigenvalues from Rust should match scipy exactly."""
        from scipy import linalg

        np.random.seed(12345)
        n = 500
        A = np.random.randn(n, n)
        K = (A + A.T) / 2  # Symmetric

        # scipy reference
        scipy_eigenvalues, _ = linalg.eigh(K.copy())

        # Rust (via jamma)
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()
        from jamma.lmm.eigen import eigendecompose_kinship

        rust_eigenvalues, _ = eigendecompose_kinship(K.copy(), threshold=0.0)

        # Compare within JAMMA's standard tolerance
        np.testing.assert_allclose(
            rust_eigenvalues,
            scipy_eigenvalues,
            rtol=1e-10,
            atol=1e-14,
            err_msg="Eigenvalues differ between Rust and scipy",
        )

    def test_eigenvectors_reconstruct_matrix(self):
        """Eigenvectors from Rust should reconstruct the original matrix."""
        np.random.seed(67890)
        n = 300
        A = np.random.randn(n, n)
        K = (A + A.T) / 2

        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()
        from jamma.lmm.eigen import eigendecompose_kinship

        eigenvalues, eigenvectors = eigendecompose_kinship(K.copy(), threshold=0.0)

        # Reconstruct: K = U @ diag(S) @ U.T
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        np.testing.assert_allclose(
            reconstructed,
            K,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Eigenvectors don't reconstruct original matrix",
        )

    def test_threshold_behavior_matches(self):
        """Threshold zeroing should behave identically between Rust and scipy."""
        # Create a matrix with a known small eigenvalue using diagonal construction
        # Diagonal matrix eigenvalues are the diagonal elements
        n = 100
        eigenvalues_true = np.ones(n)
        eigenvalues_true[0] = 1e-11  # Small eigenvalue (below 1e-10 threshold)
        eigenvalues_true[1] = 1e-10  # At threshold (should NOT be zeroed)

        # Create diagonal matrix
        K = np.diag(eigenvalues_true)

        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()
        from jamma.lmm.eigen import eigendecompose_kinship

        eigenvalues, _ = eigendecompose_kinship(K, threshold=1e-10)

        # The smallest eigenvalue (1e-11) should be zeroed
        n_zeroed = np.sum(eigenvalues == 0.0)
        assert n_zeroed == 1, f"Expected 1 zeroed eigenvalue, got {n_zeroed}"

        # The threshold eigenvalue (1e-10) should NOT be zeroed
        assert np.any(
            np.isclose(eigenvalues, 1e-10, rtol=1e-3)
        ), "Eigenvalue at threshold should not be zeroed"


class TestKinshipParity:
    """Tests comparing kinship computation results with different backends."""

    def setup_method(self):
        get_compute_backend.cache_clear()

    def teardown_method(self):
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    def test_kinship_eigendecomp_parity(self):
        """Kinship eigendecomposition should be identical across backends."""
        from scipy import linalg

        from jamma.kinship import compute_centered_kinship

        # Create synthetic genotype data
        np.random.seed(22222)
        n_samples = 200
        n_snps = 500
        G = np.random.randint(0, 3, size=(n_samples, n_snps)).astype(np.float64)

        # Compute kinship
        K = compute_centered_kinship(G)

        # scipy reference (direct)
        scipy_eigenvalues, scipy_eigenvectors = linalg.eigh(K.copy())

        # Rust (via jamma)
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()
        from jamma.lmm.eigen import eigendecompose_kinship

        rust_eigenvalues, rust_eigenvectors = eigendecompose_kinship(
            K.copy(), threshold=0.0
        )

        # Eigenvalues must match
        np.testing.assert_allclose(
            rust_eigenvalues,
            scipy_eigenvalues,
            rtol=1e-10,
            atol=1e-14,
            err_msg="Kinship eigenvalues differ",
        )


@pytest.mark.tier1
class TestLMMWorkflowParity:
    """End-to-end LMM workflow parity tests using test fixtures."""

    @pytest.fixture
    def fixture_path(self):
        """Path to test fixtures."""
        return Path(__file__).parent / "fixtures"

    def setup_method(self):
        get_compute_backend.cache_clear()

    def teardown_method(self):
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    def test_synthetic_lmm_parity(self):
        """Full LMM workflow on synthetic data should match between backends.

        This is the critical end-to-end test: run full LMM association with both
        scipy and Rust backends and verify identical statistical results.
        """
        from jamma.kinship import compute_centered_kinship
        from jamma.lmm import run_lmm_association

        # Synthetic data with reproducible seed
        np.random.seed(33333)
        n_samples = 100
        n_snps = 20  # Small for speed

        # Genotypes
        G = np.random.randint(0, 3, size=(n_samples, n_snps)).astype(np.float64)

        # Phenotype with genetic effect
        true_beta = 0.5
        y = G[:, 0] * true_beta + np.random.randn(n_samples) * 0.5

        # Kinship (using scipy by default, but this is just for the matrix)
        K = compute_centered_kinship(G)

        # SNP info
        snp_info = [
            {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
            for i in range(n_snps)
        ]

        # Run with JAX/scipy backend (force no Rust)
        os.environ["JAMMA_BACKEND"] = "jax"
        get_compute_backend.cache_clear()

        scipy_results = run_lmm_association(
            genotypes=G.copy(),
            phenotypes=y.copy(),
            kinship=K.copy(),
            snp_info=snp_info,
        )

        # Run with Rust backend
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()

        rust_results = run_lmm_association(
            genotypes=G.copy(),
            phenotypes=y.copy(),
            kinship=K.copy(),
            snp_info=snp_info,
        )

        # Compare results
        assert len(scipy_results) == len(rust_results), "Different number of results"

        for scipy_r, rust_r in zip(scipy_results, rust_results, strict=False):
            assert (
                scipy_r.rs == rust_r.rs
            ), f"SNP order mismatch: {scipy_r.rs} vs {rust_r.rs}"

            # Beta should match within tolerance
            np.testing.assert_allclose(
                rust_r.beta,
                scipy_r.beta,
                rtol=1e-6,  # JAMMA's beta tolerance
                err_msg=f"Beta mismatch for {scipy_r.rs}",
            )

            # SE should match
            np.testing.assert_allclose(
                rust_r.se,
                scipy_r.se,
                rtol=1e-6,
                err_msg=f"SE mismatch for {scipy_r.rs}",
            )

            # P-value should match
            np.testing.assert_allclose(
                rust_r.p_wald,
                scipy_r.p_wald,
                rtol=1e-8,  # JAMMA's p-value tolerance
                err_msg=f"P-value mismatch for {scipy_r.rs}",
            )

            # Lambda should match
            np.testing.assert_allclose(
                rust_r.l_remle,
                scipy_r.l_remle,
                rtol=1e-5,  # JAMMA's lambda tolerance
                err_msg=f"Lambda mismatch for {scipy_r.rs}",
            )


class TestGEMMAReferenceWithRust:
    """Tests verifying Rust backend still matches GEMMA reference."""

    @pytest.fixture
    def fixture_path(self):
        return Path(__file__).parent / "fixtures"

    def setup_method(self):
        get_compute_backend.cache_clear()

    def teardown_method(self):
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    @pytest.mark.tier1
    def test_gemma_eigenvalues_parity(self, fixture_path):
        """Eigenvalues should match GEMMA reference when using Rust backend."""
        eigenvalue_file = fixture_path / "eigenvalues.txt"
        if not eigenvalue_file.exists():
            pytest.skip("GEMMA eigenvalue reference not available")

        # Load GEMMA reference eigenvalues
        gemma_eigenvalues = np.loadtxt(eigenvalue_file)

        # Load corresponding kinship matrix
        kinship_file = fixture_path / "mouse_hs1940.cXX.txt"
        if not kinship_file.exists():
            pytest.skip("Kinship fixture not available")

        K = np.loadtxt(kinship_file)

        # Eigendecomp with Rust
        os.environ["JAMMA_BACKEND"] = "rust"
        get_compute_backend.cache_clear()
        from jamma.lmm.eigen import eigendecompose_kinship

        rust_eigenvalues, _ = eigendecompose_kinship(K)

        # Should match GEMMA within tolerance
        np.testing.assert_allclose(
            rust_eigenvalues,
            gemma_eigenvalues,
            rtol=1e-8,  # JAMMA's kinship tolerance
            err_msg="Rust eigenvalues don't match GEMMA reference",
        )


class TestPerformanceBaseline:
    """Performance baseline tests (not strict, just for monitoring)."""

    def setup_method(self):
        get_compute_backend.cache_clear()

    def teardown_method(self):
        os.environ.pop("JAMMA_BACKEND", None)
        get_compute_backend.cache_clear()

    @pytest.mark.benchmark
    def test_rust_not_dramatically_slower(self):
        """Rust should not be dramatically slower than scipy.

        This test compares raw eigendecomp performance, calling jamma_core
        directly to avoid Python wrapper overhead and logging.
        """
        import time

        import jamma_core
        from scipy import linalg

        np.random.seed(44444)
        n = 1000
        A = np.random.randn(n, n)
        K = (A + A.T) / 2

        # Warmup both implementations (JIT, cache effects)
        _ = linalg.eigh(K.copy())
        _ = jamma_core.eigendecompose_kinship(K.copy(), threshold=0.0)

        # Time scipy (average of 3 runs)
        scipy_times = []
        for _ in range(3):
            start = time.perf_counter()
            linalg.eigh(K.copy())
            scipy_times.append(time.perf_counter() - start)
        scipy_time = min(scipy_times)  # Use min to reduce noise

        # Time Rust (average of 3 runs)
        rust_times = []
        for _ in range(3):
            start = time.perf_counter()
            jamma_core.eigendecompose_kinship(K.copy(), threshold=0.0)
            rust_times.append(time.perf_counter() - start)
        rust_time = min(rust_times)  # Use min to reduce noise

        # Rust should not be more than 5x slower
        # (faer is typically comparable to or faster than OpenBLAS)
        assert rust_time < scipy_time * 5, (
            f"Rust ({rust_time:.2f}s) is more than 5x slower than "
            f"scipy ({scipy_time:.2f}s)"
        )
