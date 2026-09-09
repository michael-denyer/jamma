"""Tests for the consolidated pipeline startup banner."""

from __future__ import annotations

import sys

import pytest

from jamma.pipeline_banner import format_pipeline_banner

pytestmark = pytest.mark.tier0


class TestFormatPipelineBanner:
    """Unit tests for format_pipeline_banner()."""

    def test_numpy_batch_with_c_ext(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=True,
        )
        assert result == "Pipeline: numpy-batch | MKL | DSYEVD | C-ext"

    def test_numpy_batch_without_c_ext(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="openblas",
            eigen_driver="DSYEVR",
            c_ext=False,
        )
        assert result == "Pipeline: numpy-batch | OpenBLAS | DSYEVR | no C-ext"

    def test_numpy_streaming(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-streaming",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=True,
        )
        assert result == "Pipeline: numpy-streaming | MKL | DSYEVD | C-ext"

    def test_unknown_blas_backend(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="unknown",
            eigen_driver="DSYEVD",
            c_ext=True,
        )
        assert result == "Pipeline: numpy-batch | Unknown | DSYEVD | C-ext"

    def test_accelerate_blas(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="accelerate",
            eigen_driver="DSYEVD",
            c_ext=True,
        )
        assert result == "Pipeline: numpy-batch | Accelerate | DSYEVD | C-ext"

    def test_extra_kwargs_rejected(self) -> None:
        """Extra keyword arguments raise TypeError (fail-fast, no silent swallow)."""
        import pytest

        with pytest.raises(TypeError):
            format_pipeline_banner(
                runner="numpy-batch",
                blas="mkl",
                eigen_driver="DSYEVD",
                c_ext=False,
                some_extra_param=4,  # type: ignore[unexpected-keyword]
            )

    def test_jlinalg_backend_appended_when_given(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=True,
            jlinalg_backend="MKL-ILP64",
        )
        assert result == (
            "Pipeline: numpy-batch | MKL | DSYEVD | C-ext | jlinalg: MKL-ILP64"
        )

    def test_jlinalg_numpy_fallback_shown_even_with_c_ext_loaded(self) -> None:
        """jlinalg can report numpy-fallback with its C extension loaded
        (JLINALG_NO_VENDOR_DGEMM) — the banner must surface that combination
        rather than let c_ext=True imply jlinalg is vendor-backed.
        """
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=True,
            jlinalg_backend="numpy-fallback",
        )
        assert "jlinalg: numpy-fallback" in result

    def test_jlinalg_backend_omitted_when_none(self) -> None:
        """Omitting jlinalg_backend must not append a trailing separator."""
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=True,
        )
        assert "jlinalg" not in result
        assert result == "Pipeline: numpy-batch | MKL | DSYEVD | C-ext"


class TestLogPipelineBanner:
    """Tests for log_pipeline_banner()'s end-to-end wiring of jlinalg."""

    def test_startup_does_not_claim_unresolved_thread_counts(self):
        from loguru import logger

        from jamma.lmm.association_plan import ExecutionPlan
        from jamma.pipeline_banner import log_pipeline_banner

        captured: list[str] = []
        sink = logger.add(captured.append, level="INFO", format="{message}")
        try:
            log_pipeline_banner(ExecutionPlan(mode="batch", reason="test"))
        finally:
            logger.remove(sink)
        assert len(captured) == 1
        assert " threads)" not in captured[0]

    def test_banner_includes_real_jlinalg_backend(self, capsys):
        """log_pipeline_banner must read jamma.jlinalg.blas_backend and put
        it in the emitted line — the gap the P0-P8 review flagged: jlinalg
        can be numpy-fallback while _lmm_accel (c_ext) is loaded, and
        without this the banner cannot show that combination.
        """
        from loguru import logger as _logger

        import jamma.jlinalg as jlinalg
        from jamma.lmm.association_plan import plan_association
        from jamma.pipeline_banner import log_pipeline_banner

        sink_id = _logger.add(sys.stderr, level="INFO")
        try:
            log_pipeline_banner(plan_association(1_000, 10_000, n_cvt=3).summary)
        finally:
            _logger.remove(sink_id)

        captured = capsys.readouterr()
        assert f"jlinalg: {jlinalg.blas_backend}" in captured.err

    @pytest.mark.parametrize("native", [False, True])
    def test_accelerate_is_named_without_threadpoolctl(self, monkeypatch, native):
        from loguru import logger

        import jamma.jlinalg as jlinalg
        from jamma.core import threading as core_threading
        from jamma.lmm import accel
        from jamma.lmm.association_plan import ExecutionPlan
        from jamma.pipeline_banner import log_pipeline_banner

        monkeypatch.setattr(core_threading, "threadpool_info", list)
        monkeypatch.setattr(accel, "available", lambda: native)
        monkeypatch.setattr(jlinalg, "blas_backend", "Accelerate-ILP64")
        captured: list[str] = []
        sink = logger.add(captured.append, level="INFO", format="{message}")
        try:
            log_pipeline_banner(ExecutionPlan(mode="batch", reason="test"))
        finally:
            logger.remove(sink)
        extension = "C-ext" if native else "no C-ext"
        assert captured == [
            f"Pipeline: numpy-batch | Accelerate | pending | {extension}"
            " | jlinalg: Accelerate-ILP64\n"
        ]
