"""Tests for BLAS thread management module."""

import os

import numpy as np
import pytest

from jamma.core.threading import (
    blas_threads,
    get_blas_thread_count,
    get_c_extension_thread_count,
    jlinalg_threads,
)


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
class TestCExtensionThreads:
    """Tests for C-extension OpenMP thread sizing."""

    def test_serial_c_extension_is_single_threaded(self):
        assert get_c_extension_thread_count(True, False) == 1

    def test_missing_c_extension_is_single_threaded(self):
        assert get_c_extension_thread_count(False, False) == 1

    def test_openmp_c_extension_uses_physical_cores(self, monkeypatch):
        monkeypatch.setattr("jamma.core.threading.get_physical_core_count", lambda: 48)
        monkeypatch.setattr("jamma.core.threading.is_blas_controllable", lambda: True)
        assert get_c_extension_thread_count(True, True) == 48

    def test_openmp_c_extension_halves_cores_when_blas_uncontrollable(
        self, monkeypatch
    ):
        monkeypatch.setattr("jamma.core.threading.get_physical_core_count", lambda: 48)
        monkeypatch.setattr("jamma.core.threading.is_blas_controllable", lambda: False)
        assert get_c_extension_thread_count(True, True) == 24


@pytest.mark.tier0
class TestJlinalgThreads:
    """Tests for jlinalg thread scoping."""

    def test_sets_and_restores_thread_count(self, monkeypatch):
        state = {"threads": 8}
        calls: list[int] = []

        def fake_set_n_threads(n: int) -> int:
            old = state["threads"]
            state["threads"] = n
            calls.append(n)
            return old

        monkeypatch.setattr("jamma.jlinalg.set_n_threads", fake_set_n_threads)

        with jlinalg_threads(3):
            assert state["threads"] == 3

        assert state["threads"] == 8
        assert calls == [3, 8]

    def test_restores_thread_count_on_exception(self, monkeypatch):
        state = {"threads": 8}
        calls: list[int] = []

        def fake_set_n_threads(n: int) -> int:
            old = state["threads"]
            state["threads"] = n
            calls.append(n)
            return old

        monkeypatch.setattr("jamma.jlinalg.set_n_threads", fake_set_n_threads)

        with pytest.raises(RuntimeError, match="boom"):
            with jlinalg_threads(4):
                assert state["threads"] == 4
                raise RuntimeError("boom")

        assert state["threads"] == 8
        assert calls == [4, 8]


@pytest.mark.tier0
class TestEigendecompThreading:
    """Eigendecomp uses all physical cores, not a reduced thread count.

    Eigendecomp is pure LAPACK and should use all available physical cores.
    """

    def test_eigendecomp_uses_all_physical_cores(self, monkeypatch):
        """eigendecomp_kinship sets n_threads to physical core count, not reduced.

        On a 48-physical-core machine, eigendecomp should use all 48 cores.

        Regression test for a bug where eigendecomp ran with a reduced thread
        count on Databricks (2 threads instead of 48).
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
            f"Eigendecomp should use all 48 physical cores, got {captured_threads[0]}."
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
