"""Tests for JAX configuration and verification.

These tests verify that JAX is properly installed and configured
for JAMMA numerical computations, including 64-bit precision,
JIT compilation, and linear algebra operations.
"""

import jax
import jax.numpy as jnp
import pytest

from jamma.core import configure_jax, get_jax_info, verify_jax_installation


@pytest.fixture(autouse=True)
def setup_jax():
    """Configure JAX with 64-bit precision before each test."""
    configure_jax(enable_x64=True)


class TestJaxImports:
    """Test that JAX imports correctly."""

    def test_jax_imports(self):
        """Verify jax and jax.numpy import without errors."""
        assert jax is not None
        assert jnp is not None
        assert hasattr(jax, "jit")
        assert hasattr(jnp, "array")


class TestJax64Bit:
    """Test JAX 64-bit precision configuration."""

    def test_jax_64bit_enabled(self):
        """Verify 64-bit precision is enabled after configure_jax."""
        # Create array with float64 literal
        arr = jnp.array([1.0])
        assert arr.dtype == jnp.float64, f"Expected float64, got {arr.dtype}"

    def test_jax_64bit_operations_preserve_precision(self):
        """Verify operations maintain 64-bit precision."""
        a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        b = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float64)
        result = a + b
        assert result.dtype == jnp.float64


class TestJaxJit:
    """Test JAX JIT compilation."""

    def test_jax_jit_works(self):
        """Verify JIT-decorated function executes correctly."""

        @jax.jit
        def add_arrays(x, y):
            return x + y

        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        result = add_arrays(a, b)

        expected = jnp.array([5.0, 7.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_jax_jit_second_call_uses_cache(self):
        """Verify JIT function can be called multiple times."""

        @jax.jit
        def multiply(x, y):
            return x * y

        a = jnp.array([2.0, 3.0])
        b = jnp.array([4.0, 5.0])

        # First call compiles
        result1 = multiply(a, b)
        # Second call uses cached compilation
        result2 = multiply(a, b)

        assert jnp.allclose(result1, result2)
        assert jnp.allclose(result1, jnp.array([8.0, 15.0]))


class TestJaxLinearAlgebra:
    """Test JAX linear algebra operations needed for GEMMA."""

    def test_jax_matmul(self):
        """Verify matrix multiplication works correctly."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = jnp.matmul(a, b)

        assert result.shape == (2, 2)
        expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
        assert jnp.allclose(result, expected)

    def test_jax_eigendecomposition(self):
        """Verify eigendecomposition works on symmetric matrix.

        This is critical for GEMMA kinship matrix operations.
        """
        # Create symmetric positive definite matrix
        a = jnp.array([[4.0, 2.0], [2.0, 3.0]])

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = jnp.linalg.eigh(a)

        # Verify eigenvalues are returned (shape check)
        assert eigenvalues.shape == (2,)
        assert eigenvectors.shape == (2, 2)

        # Verify eigenvalues are real and positive for this PD matrix
        assert jnp.all(eigenvalues > 0)

        # Verify decomposition: A = V @ diag(eigenvalues) @ V.T
        reconstructed = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
        assert jnp.allclose(a, reconstructed)

    def test_jax_solve_linear_system(self):
        """Verify linear system solver works."""
        # Solve Ax = b
        a = jnp.array([[3.0, 1.0], [1.0, 2.0]])
        b = jnp.array([9.0, 8.0])

        x = jnp.linalg.solve(a, b)

        # Verify solution: Ax should equal b
        assert jnp.allclose(jnp.matmul(a, x), b)


class TestJaxConfigFunctions:
    """Test the JAX configuration utility functions."""

    def test_verify_jax_installation(self):
        """Verify verify_jax_installation returns True."""
        result = verify_jax_installation()
        assert result is True

    def test_get_jax_info(self):
        """Verify get_jax_info returns dict with expected keys."""
        info = get_jax_info()

        assert isinstance(info, dict)
        assert "version" in info
        assert "backend" in info
        assert "devices" in info
        assert "x64_enabled" in info

        # Verify types
        assert isinstance(info["version"], str)
        assert isinstance(info["backend"], str)
        assert isinstance(info["devices"], list)
        assert isinstance(info["x64_enabled"], bool)

    def test_get_jax_info_backend_is_valid(self):
        """Verify backend is one of the expected values."""
        info = get_jax_info()
        assert info["backend"] in ("cpu", "gpu", "tpu")

    def test_get_jax_info_has_at_least_one_device(self):
        """Verify at least one device is available."""
        info = get_jax_info()
        assert len(info["devices"]) >= 1
