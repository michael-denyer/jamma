"""Lazy eigendecomposition -- rotate targets without materializing U.

LazyEigen holds the factored eigendecomposition state (Householder vectors
from dsytrd + tridiagonal eigenvectors from dstedc) and provides a rotate()
method that computes V.T @ (Q^T @ target) == U.T @ target without ever
forming the N x N eigenvector matrix U.

This is only available when the jlinalg D&C pipeline is active (not vendor
LAPACK). When vendor LAPACK is used, the D&C pipeline is not reachable, and
eigendecompose_kinship_lazy raises NotImplementedError. Callers should fall
back to eigendecompose_kinship().
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from jamma import jlinalg


class LazyEigen:
    """Factored eigendecomposition state for on-demand rotation.

    Holds Householder vectors (in K_householder), tau scalars, and
    tridiagonal eigenvectors V. Provides rotate() to compute
    V.T @ (Q^T @ target) without materializing U.

    Memory: K_householder (N^2) + V (N^2) + tau (N-1) + eigenvalues (N) ~ 2N^2 + 2N.
    After free_tridiag_eigenvectors(): K_householder (N^2) + tau (N-1) + evals (N).

    Attributes:
        eigenvalues: (N,) ascending eigenvalues.
        n: Matrix dimension N (read-only, derived from eigenvalues).
    """

    __slots__ = (
        "eigenvalues",
        "_n",
        "_K_householder",
        "_tau",
        "_V",
    )

    def __init__(
        self,
        eigenvalues: np.ndarray,
        K_householder: np.ndarray,
        tau: np.ndarray,
        V: np.ndarray,
    ) -> None:
        n = len(eigenvalues)
        # Validate shapes to catch misuse before it reaches the C extension
        if K_householder.shape != (n, n):
            raise ValueError(
                f"K_householder must be ({n}, {n}), got {K_householder.shape}"
            )
        expected_tau = max(n - 1, 0)
        if tau.shape != (expected_tau,):
            raise ValueError(f"tau must be ({expected_tau},), got {tau.shape}")
        if V.shape != (n, n):
            raise ValueError(f"V must be ({n}, {n}), got {V.shape}")

        self.eigenvalues = eigenvalues
        self._n = n
        self._K_householder = K_householder
        self._tau = tau
        self._V = V

    def rotate(self, target: np.ndarray) -> np.ndarray:
        """Compute V.T @ (Q^T @ target) == U.T @ target.

        Args:
            target: (N,) vector or (N, M) matrix to rotate.

        Returns:
            Rotated result, same shape as target.

        Raises:
            RuntimeError: If Householder state has been freed via
                free_householder(), or if V has been freed via
                free_tridiag_eigenvectors().
        """
        if self._K_householder is None or self._tau is None:
            raise RuntimeError(
                "Householder state has been freed. "
                "Cannot rotate targets after free_householder()."
            )
        if self._V is None:
            raise RuntimeError(
                "Tridiagonal eigenvectors V have been freed. "
                "Cannot rotate targets after free_tridiag_eigenvectors()."
            )

        # Handle 1D input
        squeeze = target.ndim == 1
        if squeeze:
            target = target[:, np.newaxis]

        # Ensure C-contiguous float64
        target = np.ascontiguousarray(target, dtype=np.float64)

        result = jlinalg.rotate_via_householder(
            self._K_householder, self._tau, self._V, target
        )

        if squeeze:
            result = result.ravel()

        return result

    def free_tridiag_eigenvectors(self) -> None:
        """Free V (N x N) to reclaim memory.

        After this call, rotate() will raise RuntimeError.
        Call only after all rotations (UtW, Uty, and all UtG chunks)
        are complete.
        """
        self._V = None
        logger.debug(
            "LazyEigen: freed tridiag eigenvectors V (reclaimed {:.1f} GB)",
            self.n * self.n * 8 / 1e9,
        )

    def free_householder(self) -> None:
        """Free K_householder (N x N) and tau to reclaim all memory.

        Also frees V if still present, since V is unusable without
        Householder state.
        """
        n_matrices = 1 + (1 if self._V is not None else 0)
        self._K_householder = None
        self._tau = None
        self._V = None
        reclaimed_gb = n_matrices * self.n * self.n * 8 / 1e9
        logger.debug(
            "LazyEigen: freed Householder state (reclaimed {:.1f} GB)",
            reclaimed_gb,
        )

    @property
    def n(self) -> int:
        """Matrix dimension N."""
        return self._n

    @property
    def memory_gb(self) -> float:
        """Current memory usage in GB."""
        total = self.eigenvalues.nbytes
        if self._K_householder is not None:
            total += self._K_householder.nbytes
        if self._tau is not None:
            total += self._tau.nbytes
        if self._V is not None:
            total += self._V.nbytes
        return total / 1e9
