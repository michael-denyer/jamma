"""Tests for the consolidated pipeline startup banner."""

from __future__ import annotations

import pytest

from jamma.core.backend import format_pipeline_banner

pytestmark = pytest.mark.tier0


class TestFormatPipelineBanner:
    """Unit tests for format_pipeline_banner()."""

    def test_numpy_batch_with_c_ext(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=True,
            threads=48,
        )
        assert result == "Pipeline: numpy-batch | MKL | DSYEVD | C-ext (48 threads)"

    def test_numpy_batch_without_c_ext(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="openblas",
            eigen_driver="DSYEVR",
            c_ext=False,
            threads=8,
        )
        assert (
            result == "Pipeline: numpy-batch | OpenBLAS | DSYEVR | no C-ext (8 threads)"
        )

    def test_numpy_streaming(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-streaming",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=True,
            threads=16,
        )
        assert result == "Pipeline: numpy-streaming | MKL | DSYEVD | C-ext (16 threads)"

    def test_unknown_blas_backend(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="unknown",
            eigen_driver="DSYEVD",
            c_ext=True,
            threads=4,
        )
        assert result == "Pipeline: numpy-batch | Unknown | DSYEVD | C-ext (4 threads)"

    def test_accelerate_blas(self) -> None:
        result = format_pipeline_banner(
            runner="numpy-batch",
            blas="accelerate",
            eigen_driver="DSYEVD",
            c_ext=True,
            threads=10,
        )
        assert (
            result == "Pipeline: numpy-batch | Accelerate | DSYEVD | C-ext (10 threads)"
        )

    def test_extra_kwargs_rejected(self) -> None:
        """Extra keyword arguments raise TypeError (fail-fast, no silent swallow)."""
        import pytest

        with pytest.raises(TypeError):
            format_pipeline_banner(
                runner="numpy-batch",
                blas="mkl",
                eigen_driver="DSYEVD",
                c_ext=False,
                threads=16,
                some_extra_param=4,  # type: ignore[unexpected-keyword]
            )
