"""Read GEMMA individual weights and apply kinship and observation transforms.

A ``-widv`` file has one weight per line, no header, in FAM sample order.
For positive weights, divide centered K[i,j] by sqrt(w_i * w_j), then
multiply each eigenvector row by sqrt(w_i) before association rotations.
Nonpositive weights zero the corresponding kinship rows and columns and
observation scales. Persist raw eigenvectors before scaling their rows.
"""

from pathlib import Path

import numpy as np
from loguru import logger


def read_weight_file(path: Path) -> np.ndarray:
    """Read a GEMMA-format individual weight file.

    Parses a single-column text file with one weight per line. Each row
    corresponds to a sample in positional order (matching .fam file).

    Args:
        path: Path to the weight file.

    Returns:
        1-D float64 array of weights.

    Raises:
        ValueError: If file is empty or contains no valid weights.
    """
    try:
        weights = np.loadtxt(path, dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"Cannot parse weight file {path}: {e}") from e

    if weights.size == 0:
        raise ValueError(f"Weight file is empty: {path}")

    # Validate single-column format before flattening
    if weights.ndim > 1 and weights.shape[1] != 1:
        raise ValueError(
            f"Weight file has {weights.shape[1]} columns but expected 1 "
            f"(single column of weights): {path}"
        )

    # Ensure 1-D even for single-value files
    weights = weights.ravel()

    # Reject NaN weights — they would silently bypass scaling in
    # apply_individual_weights (NaN comparisons are always False)
    n_nan = int(np.sum(np.isnan(weights)))
    if n_nan > 0:
        raise ValueError(f"Weight file contains {n_nan} NaN value(s): {path}")

    return weights


def read_analysis_weights(
    path: Path, n_samples: int, valid_indices: np.ndarray | None
) -> np.ndarray:
    """Read weights and return them in analyzed-sample order."""
    weights = read_weight_file(path)
    if len(weights) != n_samples:
        raise ValueError(
            f"Weight file has {len(weights)} entries but expected "
            f"{n_samples} (matching sample count)"
        )
    return weights if valid_indices is None else weights[valid_indices]


def _row_scale(weights: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(weights, 0.0))


def apply_individual_weights(K: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply individual weights to kinship matrix in-place.

    Transforms kinship via K[i,j] /= sqrt(w_i * w_j) for positive weights,
    and zeros out entries where either weight is non-positive. This matches
    GEMMA's individual-weight transform.

    Uses two-pass row/column broadcasting to avoid allocating an n x n
    temporary matrix. At 100k samples the outer product would be 80GB;
    this approach uses O(n) memory.

    Args:
        K: Kinship matrix of shape (n, n). Modified in-place.
        weights: 1-D array of individual weights, length n.

    Returns:
        The modified kinship matrix (same object as input K).

    Raises:
        ValueError: If weights length does not match K dimensions.
    """
    if weights.shape[0] != K.shape[0]:
        raise ValueError(
            f"Weight array has {weights.shape[0]} entries but kinship matrix "
            f"has {K.shape[0]} samples"
        )

    # Identify non-positive weights once (used for zeroing and warning)
    invalid = weights <= 0
    n_invalid = int(np.sum(invalid))
    if n_invalid > 0:
        logger.warning(
            f"{n_invalid} sample(s) have non-positive weight; "
            f"their kinship rows/columns will be zeroed out"
        )

    # Compute sqrt of positive weights; non-positive get 0
    sqrt_w = _row_scale(weights)

    # Two-pass in-place scaling: row then column (no n x n temporary)
    # Pass 1: K[i,j] /= sqrt(w_i) for each row i
    np.divide(K, sqrt_w[:, None], out=K, where=sqrt_w[:, None] > 0)
    # Pass 2: K[i,j] /= sqrt(w_j) for each column j
    np.divide(K, sqrt_w[None, :], out=K, where=sqrt_w[None, :] > 0)

    # Zero out rows/columns for non-positive weights
    K[invalid, :] = 0.0
    K[:, invalid] = 0.0

    return K


def apply_weights_to_eigenvectors(
    eigenvectors: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Apply GEMMA ``-widv`` row scaling to analysis eigenvectors in-place.

    GEMMA sets the multiplier to zero when a weight is nonpositive; otherwise
    it uses its square root.  Scaling eigenvector rows makes the existing
    rotations apply the same transform to phenotype, covariates, and markers.
    """
    if weights.shape[0] != eigenvectors.shape[0]:
        raise ValueError(
            f"Weight array has {weights.shape[0]} entries but eigenvectors have "
            f"{eigenvectors.shape[0]} sample rows"
        )
    eigenvectors *= _row_scale(weights)[:, None]
    return eigenvectors
