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
    # NumPy's summation order follows memory order, so reduce every chunk in
    # one layout: equal values then give equal statistics, as in the C kernel.
    data = np.asfortranarray(data)
    is_nan = np.isnan(data)
    missing = np.sum(is_nan, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(data, axis=0, dtype=np.float64)
        # np.nanvar holds the deviations in the input dtype whatever dtype= says,
        # so float32 genotypes would not get the C kernel's double accumulation.
        squares = np.subtract(data, mean, dtype=np.float64)
        np.copyto(squares, 0.0, where=is_nan)
        np.square(squares, out=squares)
        variance = squares.sum(axis=0) / (data.shape[0] - missing)
    means[:] = np.nan_to_num(mean, nan=0.0)
    miss_counts[:] = missing
    variances[:] = np.nan_to_num(variance, nan=0.0)
    if n_aa is not None and n_ab is not None and n_bb is not None:
        valid = ~is_nan
        n_aa[:] = np.sum((data == 0) & valid, axis=0)
        n_ab[:] = np.sum((data == 1) & valid, axis=0)
        n_bb[:] = np.sum((data == 2) & valid, axis=0)
