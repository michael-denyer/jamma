"""Shared SNP selection, chunk preprocessing, and symmetric kinship accumulation.

Streaming and LOCO own their pass scheduling and matrix lifetimes. These helpers
preserve BED chunk boundaries and preprocess all samples before output-row
selection, so both consumers share the same numerical contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple

import numpy as np

from jamma import jlinalg
from jamma.core.snp_stats import SnpFilterSpec, SnpSelection, SnpStats, filter_snp_stats
from jamma.kinship.missing import impute_and_center


def accumulate_kinship(K: np.ndarray, X_centered: np.ndarray) -> None:
    """Accumulate kinship contribution from centered SNP batch.

    Uses jlinalg.dsyrk (symmetric rank-k update) with in-place accumulation.
    The non-LOCO kinship path uses this exclusively.

    Args:
        K: Current kinship matrix accumulator (n_samples, n_samples)
        X_centered: Centered genotype batch (n_samples, batch_snps)

    The accumulator is mutated in place.
    """
    jlinalg.dsyrk(X_centered, out=K, beta=1.0)


class CenteredChunk(NamedTuple):
    """One file chunk after column selection, preprocessing, and row selection.

    X: float64 (n_out, n_sel), imputed and centered per column.
    global_idx: global BED SNP indices of X's columns, in column order, sorted
        ascending. ``X.shape[1] == len(global_idx)``. LOCO maps these to chromosomes.
    """

    X: np.ndarray
    global_idx: np.ndarray


def selected_chunks(
    chunk_iter: Iterator[tuple[np.ndarray, int, int]],
    snp_indices: np.ndarray,
    valid_indices: np.ndarray | None,
    *,
    keep: Callable[[np.ndarray], bool] | None = None,
    transform: Callable[[np.ndarray], np.ndarray] = impute_and_center,
) -> Iterator[CenteredChunk]:
    """Select columns, transform all samples, then select output rows per file chunk.

    Unifies the streaming (PASS 2) and LOCO accumulation loops, which share one
    mechanism. Pick the filtered columns of each BED chunk via searchsorted against
    the sorted global ``snp_indices``, apply ``transform`` over all samples, then
    subset rows to ``valid_indices``. The single-pass monomorphism loop selects
    columns by a per-chunk variance mask instead.

    Args:
        chunk_iter: Yields ``(chunk, file_start, file_end)`` from the genotype stream.
            ``chunk`` is float64 ``(n_samples, chunk_cols)`` over full BED rows.
        snp_indices: Global indices of SNPs that passed filtering, sorted ascending.
        valid_indices: Sample indices to keep, or None for all samples.
        keep: Optional predicate on a chunk's global indices, evaluated before any
            transform. Returning False skips the chunk with no work done, preserving
            LOCO's "skip chunks that contribute nothing" optimisation. None keeps every
            chunk with at least one selected column.
        transform: Per-chunk preprocessing applied to selected columns over all samples.
            Defaults to ``impute_and_center`` (GEMMA -gk 1). Pass
            ``impute_center_and_standardize`` for -gk 2; it self-computes each column's
            variance over the chunk's rows, which equals the full-sample variance since
            every retained row is present in the chunk.

    Yields:
        One CenteredChunk per surviving file chunk. Chunks with no selected columns (or
        that fail ``keep``) yield nothing.

    Numerics contract:
        Exactly one yield per file chunk, never re-batching selected columns across
        chunks and never splitting one chunk's selection. So one ``accumulate_kinship``
        per yield reproduces the pre-refactor dsyrk column grouping, which splitting
        would not (bit-level). ``searchsorted`` runs on full BED chunk boundaries.
        Preprocessing precedes row selection so means, missing-value imputation,
        and standardization do not depend on which matrix rows a caller needs.
    """
    assert snp_indices.ndim == 1, "snp_indices must be 1-D"
    assert len(snp_indices) < 2 or np.all(np.diff(snp_indices) > 0), (
        "snp_indices must be sorted ascending for searchsorted selection"
    )

    for chunk, file_start, file_end in chunk_iter:
        left = np.searchsorted(snp_indices, file_start, side="left")
        right = np.searchsorted(snp_indices, file_end, side="left")
        global_idx = snp_indices[left:right]
        if len(global_idx) == 0:
            continue
        if keep is not None and not keep(global_idx):
            continue

        X_chunk = chunk[:, global_idx - file_start]
        assert X_chunk.dtype == np.float64, (
            f"kinship accumulation requires float64 chunks (got {X_chunk.dtype}); "
            "check stream_genotype_chunks dtype arg"
        )
        X_chunk = transform(X_chunk)
        if valid_indices is not None:
            X_chunk = X_chunk[valid_indices, :]
        yield CenteredChunk(X_chunk, global_idx)


def select_kinship_snps(
    stats: SnpStats,
    maf_threshold: float,
    miss_threshold: float,
    ksnps_indices: np.ndarray | None,
    n_snps: int,
) -> SnpSelection:
    """Apply the kinship MAF/missing/monomorphism filter, raising if none pass.

    The streaming and LOCO kinship passes share this filter step exactly. Both
    build the same SnpFilterSpec (no HWE, "Kinship SNP list" restriction label)
    and raise the same message when every SNP is removed. Callers log their own
    retained/removed line afterwards, since the wording differs between passes.

    Args:
        stats: Per-SNP statistics from collect_streamed_snp_stats.
        maf_threshold: Minimum MAF for inclusion.
        miss_threshold: Maximum missing rate for inclusion.
        ksnps_indices: Optional -ksnps restriction, or None.
        n_snps: Total SNP count, for the error message.

    Returns:
        The SnpSelection of SNPs that passed filtering.

    Raises:
        ValueError: If no SNPs pass filtering.
    """
    selection = filter_snp_stats(
        stats,
        SnpFilterSpec(
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            restrict_indices=ksnps_indices,
            restrict_label="Kinship SNP list",
        ),
    )
    if len(selection.indices) == 0:
        raise ValueError(
            f"No SNPs passed filtering (maf>={maf_threshold}, "
            f"miss<={miss_threshold}, polymorphic). "
            f"Original SNP count: {n_snps}"
        )
    return selection


def validate_valid_indices(valid_indices: np.ndarray, n_samples: int) -> None:
    """Validate valid_indices for emptiness, bounds, duplicates, and ordering.

    The single source of truth for the sample-subset invariant. Called once per
    public entry path (``compute_kinship_streaming``,
    ``compute_loco_kinship_streaming``, and ``PipelineRunner._load_or_compute_kinship``
    before its ``np.ix_`` file subset). Internal helpers below a validating boundary
    trust the value and do not re-check.

    Args:
        valid_indices: Array of sample indices to keep.
        n_samples: Total number of samples (upper bound for indices).

    Raises:
        ValueError: If indices are empty, out of bounds, duplicated, or unsorted.
    """
    if len(valid_indices) == 0:
        raise ValueError("valid_indices must not be empty")
    if valid_indices.min() < 0 or valid_indices.max() >= n_samples:
        raise ValueError(
            f"valid_indices out of bounds: min={valid_indices.min()}, "
            f"max={valid_indices.max()}, n_samples={n_samples}"
        )
    n_unique = len(np.unique(valid_indices))
    if len(valid_indices) != n_unique:
        raise ValueError(
            f"valid_indices contains {len(valid_indices) - n_unique} duplicates"
        )
    if not np.all(np.diff(valid_indices) > 0):
        raise ValueError(
            "valid_indices must be strictly increasing (sorted, no duplicates)"
        )
