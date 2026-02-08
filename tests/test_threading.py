"""Tests for BLAS thread management module."""

import os

from jamma.core.threading import blas_threads, get_blas_thread_count


class TestGetBlasThreadCount:
    """Tests for get_blas_thread_count()."""

    def test_returns_positive(self):
        result = get_blas_thread_count()
        assert result > 0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("JAMMA_BLAS_THREADS", "4")
        assert get_blas_thread_count() == 4

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
