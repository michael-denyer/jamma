"""Eigendecomposition time estimates follow the active BLAS backend."""

from __future__ import annotations

import pytest

from jamma import estimates

pytestmark = pytest.mark.tier0

# Median wall seconds per sample count, as scripts/bench_eigendecomp_scaling.py
# printed them on an Apple M5 Pro with Accelerate-ILP64.
_ACCELERATE_M5_PRO_MEDIANS = {
    4_000: 3.50,
    6_000: 11.82,
    8_000: 29.60,
    10_000: 55.70,
    14_000: 153.18,
    20_000: 449.35,
}


@pytest.fixture
def blas_backend(monkeypatch):
    def set_backend(name: str) -> None:
        monkeypatch.setattr(estimates, "_blas_backend_name", lambda: name)

    return set_backend


def test_accelerate_estimate_reproduces_the_measured_timings(blas_backend):
    blas_backend("Accelerate-ILP64")

    for n_samples, measured in _ACCELERATE_M5_PRO_MEDIANS.items():
        estimate = estimates.estimate_eigendecomp_seconds(n_samples)
        assert estimate == pytest.approx(measured, rel=0.05), n_samples


def test_accelerate_estimate_does_not_scale_with_core_count(blas_backend):
    blas_backend("Accelerate-ILP64")

    few_cores = estimates.estimate_eigendecomp_seconds(20_000, 8)
    many_cores = estimates.estimate_eigendecomp_seconds(20_000, 48)

    assert few_cores == many_cores


@pytest.mark.parametrize("name", ["MKL-ILP64", "OpenBLAS-ILP64", "numpy-fallback"])
def test_other_backends_keep_the_core_scaled_mkl_model(blas_backend, name):
    blas_backend(name)

    # 46.1 s is the 20k measurement on the 48-core MKL calibration machine.
    at_calibration = estimates.estimate_eigendecomp_seconds(20_000, 48)
    assert at_calibration == pytest.approx(46.1, rel=0.13)
    assert estimates.estimate_eigendecomp_seconds(20_000, 24) > at_calibration


def test_accelerate_eigendecomp_caveat_names_its_own_calibration(blas_backend):
    blas_backend("Accelerate-ILP64")

    eigendecomp = estimates.estimate_eigendecomp_time(20_000)

    assert "MKL" not in eigendecomp
    assert "Apple M5 Pro" in eigendecomp
    # The kinship model has no Accelerate fit, so its caveat still says so.
    assert "calibrated to MKL" in estimates.estimate_kinship_time(20_000, 95_000)
