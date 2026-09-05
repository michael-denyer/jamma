"""Tests for BLAS thread management module."""

import os

import numpy as np
import pytest

from jamma.core.threading import (
    blas_threads,
    get_blas_thread_count,
    get_c_extension_thread_count,
)
from jamma.jlinalg import HAS_C_EXTENSION, get_n_threads, set_n_threads

pytestmark = pytest.mark.tier0


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

    def test_openmp_c_extension_uses_physical_cores_when_blas_uncontrollable(
        self, monkeypatch
    ):
        monkeypatch.setattr("jamma.core.threading.get_physical_core_count", lambda: 48)
        monkeypatch.setattr("jamma.core.threading.is_blas_controllable", lambda: False)
        assert get_c_extension_thread_count(True, True) == 48


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
        # Mock the thread-count source to report 48 (physical cores, no env
        # override) — eigen reads it through get_blas_thread_count so the
        # documented JAMMA_BLAS_THREADS knob also reaches this path.
        monkeypatch.setattr("jamma.lmm.eigen.get_blas_thread_count", lambda: 48)
        # Force vendor path so blas_threads is actually called
        monkeypatch.setattr("jamma.lmm.eigen.jlinalg.blas_has_dsyevd", 1)
        monkeypatch.setattr("jamma.lmm.eigen.jlinalg.blas_has_dsyevr", 0)

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
        # Force vendor path so blas_threads is actually called
        monkeypatch.setattr("jamma.lmm.eigen.jlinalg.blas_has_dsyevd", 1)
        monkeypatch.setattr("jamma.lmm.eigen.jlinalg.blas_has_dsyevr", 0)

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


class TestBlasThreadsKnobReachesRotation:
    """JAMMA_BLAS_THREADS must govern the rotation's thread context.

    Regression for the documented-but-ignored knob: eigen and the shared
    preparation called blas_threads(get_physical_core_count()) directly,
    so the env override never reached the paths users benchmark with it.
    """

    def test_prepare_rotation_uses_env_thread_count(self, monkeypatch):
        import numpy as np

        from jamma.lmm import prepare_common

        monkeypatch.setenv("JAMMA_BLAS_THREADS", "2")
        seen: list[int] = []
        real = prepare_common.blas_threads

        def spy(n: int):
            seen.append(n)
            return real(n)

        monkeypatch.setattr(prepare_common, "blas_threads", spy)

        rng = np.random.default_rng(7)
        n = 30
        x = rng.standard_normal((n, n))
        kinship = np.ascontiguousarray(x @ x.T / n)
        prepare_common.prepare_lmm_run(
            eigen_input=prepare_common.KinshipMatrix(kinship),
            phenotypes=rng.standard_normal(n),
            W=np.ones((n, 1)),
            n_cvt=1,
            l_min=1e-5,
            l_max=1e5,
            show_progress=False,
            check_memory=False,
            label="test",
        )

        assert seen == [2], f"rotation must run under JAMMA_BLAS_THREADS=2, saw {seen}"


class TestThreadControl:
    """Thread control API: get/set_n_threads with init-time clamping."""

    def test_get_n_threads_returns_positive(self) -> None:
        """get_n_threads returns a positive integer."""
        n = get_n_threads()
        assert isinstance(n, int)
        assert n >= 1, f"get_n_threads returned {n}, expected >= 1"

    def test_set_n_threads_returns_old_count(self) -> None:
        """set_n_threads returns the previous thread count."""
        original = get_n_threads()
        old = set_n_threads(1)
        assert old == original, f"set_n_threads returned {old}, expected {original}"
        # Restore
        set_n_threads(original)

    @pytest.mark.skipif(
        not HAS_C_EXTENSION,
        reason="the NumPy fallback clamps to os.cpu_count; unclamped storage is C-only",
    )
    def test_set_n_threads_accepts_large(self) -> None:
        """set_n_threads(9999) stores the value (no clamping after own-BLAS removal)."""
        original = get_n_threads()
        set_n_threads(9999)
        assert get_n_threads() == 9999
        # Restore
        set_n_threads(original)

    def test_set_n_threads_rejects_zero(self) -> None:
        """set_n_threads(0) raises ValueError."""
        with pytest.raises(ValueError):
            set_n_threads(0)

    def test_set_n_threads_rejects_negative(self) -> None:
        """set_n_threads(-1) raises ValueError."""
        with pytest.raises(ValueError):
            set_n_threads(-1)
