"""Weight file I/O and kinship weight application.

This module provides reading of GEMMA-format individual weight files and
vectorized application of individual weights to kinship matrices.

GEMMA individual weight file format (-widv):
- Single column of floating-point weights, one per line
- No header row
- Row order matches .fam file (positional matching)
- Positive weights scale the kinship matrix; zero/negative weights zero out entries

The weight application transforms the kinship matrix as:
    K[i,j] /= sqrt(w_i * w_j) for positive weights
    K[i,j] = 0 when either weight <= 0

This matches GEMMA's gemma.cpp behavior (lines ~2635-2655).
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


def apply_individual_weights(K: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply individual weights to kinship matrix in-place.

    Transforms kinship via K[i,j] /= sqrt(w_i * w_j) for positive weights,
    and zeros out entries where either weight is non-positive. This matches
    GEMMA's exact behavior (gemma.cpp lines ~2635-2655).

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
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))
    sqrt_w[invalid] = 0.0

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
    row_scale = np.sqrt(np.maximum(weights, 0.0))
    eigenvectors *= row_scale[:, None]
    return eigenvectors
