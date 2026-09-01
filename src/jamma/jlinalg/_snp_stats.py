"""NumPy fallback for the native single-pass SNP statistics kernel."""

from __future__ import annotations

import warnings

import numpy as np


def compute_snp_stats_chunk(
    data: np.ndarray,
    means: np.ndarray,
    miss_counts: np.ndarray,
    variances: np.ndarray,
    n_aa: np.ndarray | None = None,
    n_ab: np.ndarray | None = None,
    n_bb: np.ndarray | None = None,
) -> None:
    """Compute per-SNP statistics into preallocated output arrays."""
    is_nan = np.isnan(data)
    missing = np.sum(is_nan, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(data, axis=0)
        variance = np.nanvar(data, axis=0)
    means[:] = np.nan_to_num(mean, nan=0.0)
    miss_counts[:] = missing
    variances[:] = np.nan_to_num(variance, nan=0.0)
    if n_aa is not None and n_ab is not None and n_bb is not None:
        valid = ~is_nan
        n_aa[:] = np.sum((data == 0) & valid, axis=0)
        n_ab[:] = np.sum((data == 1) & valid, axis=0)
        n_bb[:] = np.sum((data == 2) & valid, axis=0)
