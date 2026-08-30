"""Shared numerical assertions for eigendecomposition tests."""

from __future__ import annotations

import numpy as np


def _assert_reconstruction(
    K: np.ndarray,
    w: np.ndarray,
    v: np.ndarray,
    tol: float,
    label: str = "",
) -> float:
    """Assert ||K - V diag(w) V.T||_F / ||K||_F < tol.

    Args:
        K: Original matrix (before eigh overwrites it).
        w: Eigenvalues from eigh.
        v: Eigenvectors from eigh.
        tol: Maximum allowed relative reconstruction error.
        label: Optional label for the assertion message.

    Returns:
        The computed relative reconstruction error.
    """
    K_recon = v @ np.diag(w) @ v.T
    norm_K = np.linalg.norm(K, "fro")
    if norm_K == 0.0:
        ratio = np.linalg.norm(K_recon, "fro")
    else:
        ratio = np.linalg.norm(K - K_recon, "fro") / norm_K
    msg = f"Reconstruction error {ratio:.2e} > {tol}"
    if label:
        msg = f"{label}: {msg}"
    assert ratio < tol, msg
    return float(ratio)


def _assert_orthogonality(
    v: np.ndarray,
    tol: float,
    label: str = "",
) -> float:
    """Assert ||V.T @ V - I||_F < tol.

    Args:
        v: Eigenvectors from eigh, shape (N, N).
        tol: Maximum allowed orthogonality error.
        label: Optional label for the assertion message.

    Returns:
        The computed orthogonality error.
    """
    N = v.shape[1]
    norm_off = np.linalg.norm(v.T @ v - np.eye(N), "fro")
    msg = f"Orthogonality error {norm_off:.2e} > {tol}"
    if label:
        msg = f"{label}: {msg}"
    assert norm_off < tol, msg
    return float(norm_off)


# ---------------------------------------------------------------------------
# Helper: generate random symmetric positive semi-definite matrix
# ---------------------------------------------------------------------------


def _random_spd(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric positive semi-definite matrix.

    Args:
        N: Matrix dimension.
        rng: NumPy random generator instance.

    Returns:
        N x N symmetric PSD matrix, float64.
    """
    A = rng.standard_normal((N, N))
    K = A @ A.T / N
    return K
