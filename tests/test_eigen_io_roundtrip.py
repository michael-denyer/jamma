"""Property tests for eigendecomposition file round trips."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from tests.eigen_io_helpers import genotype_matrix
from tests.reference.kinship import compute_centered_kinship


def _roundtrip(genotypes, tmp_path_factory, *, legacy_text=False):
    kinship = compute_centered_kinship(genotypes, check_memory=False)
    original = kinship.copy()
    eigenvalues, eigenvectors = eigendecompose_kinship(kinship, threshold=0)
    d_path, u_path = write_eigen_files(
        eigenvalues,
        eigenvectors,
        tmp_path_factory.mktemp("eigen_roundtrip"),
        prefix="test",
        legacy_text=legacy_text,
    )
    read_values, read_vectors = read_eigen_files(d_path, u_path)
    return original, eigenvalues, read_values, read_vectors


@pytest.mark.tier0
class TestEigenIoRoundTrip:
    """Written eigenpairs retain reconstruction and precision invariants."""

    @given(genotypes=genotype_matrix(10, 30, 20, 40))
    @settings(
        max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_eigen_roundtrip_reconstruction(self, genotypes, tmp_path_factory):
        original, _values, read_values, read_vectors = _roundtrip(
            genotypes, tmp_path_factory
        )
        reconstructed = read_vectors @ np.diag(read_values) @ read_vectors.T
        scale = max(np.abs(original).max(), 1e-10)
        np.testing.assert_allclose(
            original, reconstructed, rtol=1e-7, atol=scale * 1e-8
        )

    @given(genotypes=genotype_matrix(10, 30, 20, 40))
    @settings(
        max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_eigen_roundtrip_orthonormality(self, genotypes, tmp_path_factory):
        _original, _values, _read_values, vectors = _roundtrip(
            genotypes, tmp_path_factory
        )
        n = vectors.shape[0]
        np.testing.assert_allclose(
            vectors.T @ vectors, np.eye(n), rtol=1e-6, atol=n * 1e-9
        )

    @given(genotypes=genotype_matrix(10, 30, 20, 40))
    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_eigen_roundtrip_eigenvalue_precision(self, genotypes, tmp_path_factory):
        _original, values, read_values, _vectors = _roundtrip(
            genotypes, tmp_path_factory
        )
        np.testing.assert_allclose(values, read_values, rtol=1e-9)

    @given(genotypes=genotype_matrix(10, 30, 20, 40))
    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_eigen_roundtrip_text_precision(self, genotypes, tmp_path_factory):
        original, values, read_values, vectors = _roundtrip(
            genotypes, tmp_path_factory, legacy_text=True
        )
        np.testing.assert_allclose(values, read_values, rtol=5e-10)
        reconstructed = vectors @ np.diag(read_values) @ vectors.T
        scale = max(np.abs(original).max(), 1e-10)
        np.testing.assert_allclose(
            original, reconstructed, rtol=1e-6, atol=scale * 1e-7
        )
