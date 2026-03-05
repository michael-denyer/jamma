"""Tests for the consolidated pipeline startup banner."""

from __future__ import annotations

from jamma.core.backend import format_pipeline_banner


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

    def test_jax_streaming_with_devices(self) -> None:
        result = format_pipeline_banner(
            runner="jax-streaming",
            blas="openblas",
            eigen_driver="DSYEVR",
            c_ext=False,
            threads=8,
            jax_devices=4,
        )
        expected = (
            "Pipeline: jax-streaming | OpenBLAS"
            " | DSYEVR | no C-ext (8 threads, 4 JAX devices)"
        )
        assert result == expected

    def test_jax_batch_with_devices(self) -> None:
        result = format_pipeline_banner(
            runner="jax-batch",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=False,
            threads=16,
            jax_devices=4,
        )
        expected = (
            "Pipeline: jax-batch | MKL | DSYEVD | no C-ext (16 threads, 4 JAX devices)"
        )
        assert result == expected

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

    def test_jax_without_devices_no_suffix(self) -> None:
        """JAX runner with jax_devices=0 should not append device count."""
        result = format_pipeline_banner(
            runner="jax-batch",
            blas="mkl",
            eigen_driver="DSYEVD",
            c_ext=False,
            threads=16,
            jax_devices=0,
        )
        assert result == "Pipeline: jax-batch | MKL | DSYEVD | no C-ext (16 threads)"
