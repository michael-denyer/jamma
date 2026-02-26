"""Tests for BLAS thread management module."""

import os

import numpy as np
import pytest

from jamma.core.threading import blas_threads, get_blas_thread_count


@pytest.mark.tier0
class TestGetBlasThreadCount:
    """Tests for get_blas_thread_count()."""

    def test_returns_positive(self):
        result = get_blas_thread_count()
        assert result > 0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("JAMMA_BLAS_THREADS", "2")
        assert get_blas_thread_count() == 2

    def test_env_capped_at_cpu_count(self, monkeypatch):
        monkeypatch.setenv("JAMMA_BLAS_THREADS", "9999")
        max_threads = os.cpu_count() or 64
        assert get_blas_thread_count() == max_threads

    def test_env_floored_at_one(self, monkeypatch):
        monkeypatch.setenv("JAMMA_BLAS_THREADS", "0")
        assert get_blas_thread_count() == 1

    def test_env_negative_floored(self, monkeypatch):
        monkeypatch.setenv("JAMMA_BLAS_THREADS", "-5")
        assert get_blas_thread_count() == 1


@pytest.mark.tier0
class TestBlasThreads:
    """Tests for blas_threads() context manager."""

    def test_context_manager_with_explicit_count(self):
        with blas_threads(2):
            pass  # enters and exits without error

    def test_context_manager_default(self):
        with blas_threads():
            pass  # enters and exits without error

    def test_context_manager_returns_none(self):
        with blas_threads(2) as result:
            assert result is None


@pytest.mark.tier0
class TestEigendecompThreading:
    """Eigendecomp uses all physical cores, not get_blas_thread_count().

    get_blas_thread_count() divides by n_jax_devices, which is correct for
    LMM association (JAX competing for cores) but wrong for eigendecomp
    (pure LAPACK, no JAX contention).
    """

    def test_eigendecomp_uses_all_physical_cores(self, monkeypatch):
        """eigendecomp_kinship sets n_threads to physical core count, not reduced.

        On a 48-physical-core machine with 24 JAX devices:
        - get_blas_thread_count() returns 48 // 24 = 2  (WRONG for eigendecomp)
        - eigendecomp should use all 48 physical cores

        Regression test for the bug where eigendecomp ran 24x slower than
        expected on Databricks (2 threads instead of 48).
        """
        # Mock get_physical_core_count to report 48 physical cores
        monkeypatch.setattr("jamma.lmm.eigen.get_physical_core_count", lambda: 48)

        # Track what thread count blas_threads() is called with
        captured_threads = []
        original_blas_threads = blas_threads

        from contextlib import contextmanager

        @contextmanager
        def capturing_blas_threads(n):
            captured_threads.append(n)
            with original_blas_threads(n):
                yield

        monkeypatch.setattr("jamma.lmm.eigen.blas_threads", capturing_blas_threads)

        # Run eigendecomp on a small matrix
        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = rng.standard_normal((50, 50))
        K = (K + K.T) / 2

        eigendecompose_kinship(K, check_memory=False)

        assert len(captured_threads) == 1, (
            f"Expected 1 blas_threads call, got {len(captured_threads)}"
        )
        assert captured_threads[0] == 48, (
            f"Eigendecomp should use all 48 physical cores, got {captured_threads[0]}. "
            f"If this is 2, it's dividing by JAX device count (the old bug)."
        )

    def test_eigendecomp_falls_back_to_os_cpu_count(self, monkeypatch):
        """Falls back to os.cpu_count() when psutil returns None."""
        # get_physical_core_count uses psutil then os.cpu_count as fallback.
        # Mock at the threading module level where it's defined.
        monkeypatch.setattr(
            "jamma.core.threading.psutil.cpu_count", lambda logical=False: None
        )
        monkeypatch.setattr("jamma.core.threading.os.cpu_count", lambda: 64)

        captured_threads = []
        original_blas_threads = blas_threads

        from contextlib import contextmanager

        @contextmanager
        def capturing_blas_threads(n):
            captured_threads.append(n)
            with original_blas_threads(n):
                yield

        monkeypatch.setattr("jamma.lmm.eigen.blas_threads", capturing_blas_threads)

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = rng.standard_normal((30, 30))
        K = (K + K.T) / 2

        eigendecompose_kinship(K, check_memory=False)

        assert captured_threads[0] == 64, (
            f"Should fall back to os.cpu_count()=64, got {captured_threads[0]}"
        )
