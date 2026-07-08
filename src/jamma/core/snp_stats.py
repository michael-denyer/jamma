"""Shared streamed SNP statistics and filtering.

This module owns the arrays and denominator metadata produced by streamed
SNP-statistics passes. Callers still own where genotype chunks come from;
mean, missingness, variance, HWE, validation counts, and SNP-list filtering
live here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import DTypeLike

from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import (
    apply_snp_list_mask,
    compute_hwe_pvalues,
    compute_snp_filter_mask,
)
from jamma.io.plink import stream_genotype_chunks, validate_genotype_values
from jamma.jlinalg import compute_snp_stats_chunk

SampleScope = Literal["all_samples", "valid_samples"]


def _readonly_1d(
    name: str, values: np.ndarray, dtype: DTypeLike | None = None
) -> np.ndarray:
    arr = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={arr.ndim}")
    arr.flags.writeable = False
    return arr


def _same_shape(name: str, expected: tuple[int, ...], arr: np.ndarray) -> None:
    if arr.shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {arr.shape}")


@dataclass(frozen=True, slots=True)
class HweCounts:
    """Per-SNP genotype counts for HWE filtering."""

    n_aa: np.ndarray
    n_ab: np.ndarray
    n_bb: np.ndarray

    def __post_init__(self) -> None:
        n_aa = _readonly_1d("n_aa", self.n_aa, np.int64)
        n_ab = _readonly_1d("n_ab", self.n_ab, np.int64)
        n_bb = _readonly_1d("n_bb", self.n_bb, np.int64)
        _same_shape("n_ab", n_aa.shape, n_ab)
        _same_shape("n_bb", n_aa.shape, n_bb)
        object.__setattr__(self, "n_aa", n_aa)
        object.__setattr__(self, "n_ab", n_ab)
        object.__setattr__(self, "n_bb", n_bb)


@dataclass(frozen=True, slots=True)
class SnpStats:
    """Per-SNP statistics over one explicit sample population.

    ``n_samples`` is the denominator for missingness. Streaming LMM and regular
    kinship may compute stats over valid samples, while LOCO kinship caches
    stats over all samples for later association use.
    """

    col_means: np.ndarray
    miss_counts: np.ndarray
    col_vars: np.ndarray
    n_samples: int
    n_unexpected: int = 0
    hwe_counts: HweCounts | None = None
    global_indices: np.ndarray | None = None
    sample_scope: SampleScope = "all_samples"

    def __post_init__(self) -> None:
        col_means = _readonly_1d("col_means", self.col_means, np.float64)
        miss_counts = _readonly_1d("miss_counts", self.miss_counts, np.intp)
        col_vars = _readonly_1d("col_vars", self.col_vars, np.float64)
        _same_shape("miss_counts", col_means.shape, miss_counts)
        _same_shape("col_vars", col_means.shape, col_vars)

        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.sample_scope not in ("all_samples", "valid_samples"):
            raise ValueError(f"unexpected sample_scope: {self.sample_scope!r}")
        if self.hwe_counts is not None:
            _same_shape("hwe_counts", col_means.shape, self.hwe_counts.n_aa)

        if self.global_indices is None:
            global_indices = np.arange(col_means.shape[0], dtype=np.intp)
        else:
            global_indices = _readonly_1d(
                "global_indices", self.global_indices, np.intp
            )
            _same_shape("global_indices", col_means.shape, global_indices)

        global_indices.flags.writeable = False
        object.__setattr__(self, "col_means", col_means)
        object.__setattr__(self, "miss_counts", miss_counts)
        object.__setattr__(self, "col_vars", col_vars)
        object.__setattr__(self, "global_indices", global_indices)

    @property
    def n_snps(self) -> int:
        return self.col_means.shape[0]

    def take(self, local_indices: np.ndarray) -> SnpStats:
        """Return a local-position slice with global SNP identities preserved.

        Validation counts are aggregate pass diagnostics, so slices reset them.
        """
        positions = np.asarray(local_indices, dtype=np.intp)
        global_indices = self.global_indices
        assert global_indices is not None
        hwe_counts = None
        if self.hwe_counts is not None:
            hwe_counts = HweCounts(
                self.hwe_counts.n_aa[positions],
                self.hwe_counts.n_ab[positions],
                self.hwe_counts.n_bb[positions],
            )
        return SnpStats(
            col_means=self.col_means[positions],
            miss_counts=self.miss_counts[positions],
            col_vars=self.col_vars[positions],
            n_samples=self.n_samples,
            n_unexpected=0,
            hwe_counts=hwe_counts,
            global_indices=global_indices[positions],
            sample_scope=self.sample_scope,
        )


SnpStatsCache = SnpStats


@dataclass(frozen=True, slots=True)
class SnpFilterSpec:
    """SNP filter parameters applied to a ``SnpStats`` population."""

    maf_threshold: float
    miss_threshold: float
    restrict_indices: np.ndarray | None = None
    restrict_global_mask: np.ndarray | None = None
    hwe_threshold: float = 0.0
    restrict_label: str = "SNP list"

    def __post_init__(self) -> None:
        if self.restrict_indices is not None and self.restrict_global_mask is not None:
            raise ValueError("use restrict_indices or restrict_global_mask, not both")
        if self.hwe_threshold < 0:
            raise ValueError("hwe_threshold must be >= 0")
        if self.restrict_indices is not None:
            indices = _readonly_1d("restrict_indices", self.restrict_indices, np.intp)
            if len(indices) > 1 and np.any(np.diff(indices) <= 0):
                raise ValueError("restrict_indices must be strictly increasing")
            object.__setattr__(self, "restrict_indices", indices)
        if self.restrict_global_mask is not None:
            mask = _readonly_1d("restrict_global_mask", self.restrict_global_mask, bool)
            object.__setattr__(self, "restrict_global_mask", mask)


@dataclass(frozen=True, slots=True)
class SnpSelection:
    """Filtered SNP set and output statistics aligned to filtered SNP order."""

    indices: np.ndarray
    local_indices: np.ndarray
    mask: np.ndarray
    filtered_afs: np.ndarray
    filtered_miss: np.ndarray
    filtered_means: np.ndarray
    n_hwe_removed: int = 0

    def __post_init__(self) -> None:
        indices = _readonly_1d("indices", self.indices, np.intp)
        local_indices = _readonly_1d("local_indices", self.local_indices, np.intp)
        mask = _readonly_1d("mask", self.mask, bool)
        filtered_afs = _readonly_1d("filtered_afs", self.filtered_afs, np.float64)
        filtered_miss = _readonly_1d("filtered_miss", self.filtered_miss, int)
        filtered_means = _readonly_1d("filtered_means", self.filtered_means, np.float64)
        _same_shape("local_indices", indices.shape, local_indices)
        _same_shape("filtered_afs", indices.shape, filtered_afs)
        _same_shape("filtered_miss", indices.shape, filtered_miss)
        _same_shape("filtered_means", indices.shape, filtered_means)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "local_indices", local_indices)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "filtered_afs", filtered_afs)
        object.__setattr__(self, "filtered_miss", filtered_miss)
        object.__setattr__(self, "filtered_means", filtered_means)


def _is_identity_indices(indices: np.ndarray) -> bool:
    return bool(np.array_equal(indices, np.arange(len(indices), dtype=np.intp)))


def _apply_global_index_restriction(
    snp_mask: np.ndarray,
    global_indices: np.ndarray,
    restrict_indices: np.ndarray,
    label: str,
) -> None:
    if _is_identity_indices(global_indices):
        apply_snp_list_mask(snp_mask, restrict_indices, len(global_indices), label)
        return

    if len(restrict_indices) == 0:
        snp_mask[:] = False
        logger.info(f"{label}: restricting to 0 requested SNPs (all filtered)")
        return

    active = np.where(snp_mask)[0]
    pos = np.searchsorted(restrict_indices, global_indices[active])
    pos = np.clip(pos, 0, len(restrict_indices) - 1)
    in_list = restrict_indices[pos] == global_indices[active]
    snp_mask[active[~in_list]] = False
    retained = int(np.sum(snp_mask))
    logger.info(
        f"{label}: restricting to {len(restrict_indices)} requested SNPs "
        f"({retained} retained after intersection)"
    )


def collect_snp_stats_from_chunks(
    chunks: Iterable[tuple[np.ndarray, int, int]],
    *,
    n_snps: int,
    n_samples: int,
    global_indices: np.ndarray | None = None,
    include_hwe: bool = False,
    validate_genotypes: bool = False,
    sample_scope: SampleScope = "all_samples",
) -> SnpStats:
    """Collect SNP stats from chunks whose start/end are local SNP offsets."""
    col_means = np.zeros(n_snps, dtype=np.float64)
    miss_counts = np.zeros(n_snps, dtype=np.intp)
    col_vars = np.zeros(n_snps, dtype=np.float64)
    n_aa = np.zeros(n_snps, dtype=np.int64) if include_hwe else None
    n_ab = np.zeros(n_snps, dtype=np.int64) if include_hwe else None
    n_bb = np.zeros(n_snps, dtype=np.int64) if include_hwe else None
    n_unexpected = 0

    for chunk, start, end in chunks:
        if end <= start:
            continue
        expected_width = end - start
        if chunk.shape != (n_samples, expected_width):
            raise ValueError(
                "SNP stats chunk shape mismatch: expected "
                f"({n_samples}, {expected_width}), got {chunk.shape}"
            )

        chunk = np.ascontiguousarray(chunk)
        if include_hwe:
            assert n_aa is not None
            assert n_ab is not None
            assert n_bb is not None
            compute_snp_stats_chunk(
                chunk,
                col_means[start:end],
                miss_counts[start:end],
                col_vars[start:end],
                n_aa[start:end],
                n_ab[start:end],
                n_bb[start:end],
            )
        else:
            compute_snp_stats_chunk(
                chunk,
                col_means[start:end],
                miss_counts[start:end],
                col_vars[start:end],
            )
        if validate_genotypes:
            n_unexpected += validate_genotype_values(chunk)
        del chunk

    hwe_counts = None
    if include_hwe:
        assert n_aa is not None
        assert n_ab is not None
        assert n_bb is not None
        hwe_counts = HweCounts(n_aa, n_ab, n_bb)

    return SnpStats(
        col_means=col_means,
        miss_counts=miss_counts,
        col_vars=col_vars,
        n_samples=n_samples,
        n_unexpected=n_unexpected,
        hwe_counts=hwe_counts,
        global_indices=global_indices,
        sample_scope=sample_scope,
    )


def collect_streamed_snp_stats(
    bed_path: Path,
    *,
    n_snps: int,
    n_samples: int,
    chunk_size: int,
    sample_indices: np.ndarray | None = None,
    snp_indices: np.ndarray | None = None,
    include_hwe: bool = False,
    validate_genotypes: bool = False,
    show_progress: bool = True,
    progress_label: str = "Computing SNP statistics",
    dtype: type = np.float32,
    sample_scope: SampleScope = "all_samples",
) -> SnpStats:
    """Collect SNP statistics by streaming PLINK BED chunks."""
    sample_indices = (
        None if sample_indices is None else np.asarray(sample_indices, dtype=np.intp)
    )
    snp_indices = (
        None if snp_indices is None else np.asarray(snp_indices, dtype=np.intp)
    )
    stats_n_snps = n_snps if snp_indices is None else len(snp_indices)
    stats_n_samples = n_samples if sample_indices is None else len(sample_indices)
    global_indices = (
        None
        if snp_indices is None
        else np.ascontiguousarray(snp_indices, dtype=np.intp)
    )

    raw_chunks = stream_genotype_chunks(
        bed_path,
        chunk_size=chunk_size,
        dtype=dtype,
        show_progress=False,
        snp_indices=snp_indices,
    )

    def _chunks():
        for chunk, start, end in raw_chunks:
            if sample_indices is not None:
                chunk = chunk[sample_indices, :]
            yield chunk, start, end

    chunks: Iterator[tuple[np.ndarray, int, int]] = _chunks()
    if show_progress:
        n_chunks = (stats_n_snps + chunk_size - 1) // chunk_size
        chunks = progress_iterator(chunks, total=n_chunks, desc=progress_label)

    return collect_snp_stats_from_chunks(
        chunks,
        n_snps=stats_n_snps,
        n_samples=stats_n_samples,
        global_indices=global_indices,
        include_hwe=include_hwe,
        validate_genotypes=validate_genotypes,
        sample_scope=sample_scope,
    )


def filter_snp_stats(stats: SnpStats, spec: SnpFilterSpec) -> SnpSelection:
    """Apply MAF, missingness, monomorphism, SNP-list, and HWE filters."""
    global_indices = stats.global_indices
    assert global_indices is not None
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        stats.col_means,
        stats.miss_counts,
        stats.col_vars,
        stats.n_samples,
        spec.maf_threshold,
        spec.miss_threshold,
    )

    if spec.restrict_indices is not None:
        _apply_global_index_restriction(
            snp_mask, global_indices, spec.restrict_indices, spec.restrict_label
        )

    if spec.restrict_global_mask is not None:
        if len(global_indices) > 0 and (
            global_indices[0] < 0
            or global_indices[-1] >= len(spec.restrict_global_mask)
        ):
            raise ValueError(
                f"restrict_global_mask is too short for SNP index {global_indices[-1]}"
            )
        snp_mask &= spec.restrict_global_mask[global_indices]

    n_hwe_removed = 0
    if spec.hwe_threshold > 0:
        if stats.hwe_counts is None:
            raise ValueError("hwe_threshold requires HWE counts in SnpStats")
        hwe_pvalues = compute_hwe_pvalues(
            stats.hwe_counts.n_aa, stats.hwe_counts.n_ab, stats.hwe_counts.n_bb
        )
        hwe_pass = hwe_pvalues >= spec.hwe_threshold
        n_hwe_removed = int(np.sum(~hwe_pass & snp_mask))
        snp_mask &= hwe_pass
        logger.info(
            f"HWE filter: {n_hwe_removed} SNPs removed (p < {spec.hwe_threshold})"
        )

    local_indices = np.where(snp_mask)[0]
    return SnpSelection(
        indices=global_indices[local_indices],
        local_indices=local_indices,
        mask=snp_mask,
        filtered_afs=allele_freqs[local_indices],
        filtered_miss=stats.miss_counts[local_indices].astype(int),
        filtered_means=stats.col_means[local_indices],
        n_hwe_removed=n_hwe_removed,
    )
