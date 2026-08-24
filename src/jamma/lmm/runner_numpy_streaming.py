"""Disk-streaming NumPy LMM association runner.

``BedSource`` streams a PLINK .bed twice (float32 statistics pass, float64
association pass) without ever allocating the full genotype matrix; the run
itself is the shared body in ``runner_numpy``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from jamma.core.snp_stats import SnpStats, collect_streamed_snp_stats
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.lmm.chunk_runner_numpy import RawLmmChunk
from jamma.lmm.runner_numpy import _run_numpy_lmm
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    LmmConfig,
    LmmRunResult,
    SnpMeta,
)

# SNPs per block in the statistics pass when the caller names no chunk size.
# Pass 1 reads the .bed and accumulates per-SNP counts, so its footprint is one
# block of genotypes rather than the rotation and grid buffers the association
# pass carries; it needs no RAM-budgeted sizing of its own.
_DEFAULT_STATS_CHUNK = 10_000


class BedSource:
    """A PLINK .bed file as a :class:`~jamma.lmm.runner_numpy.GenotypeSource`.

    The statistics pass reads float32 blocks (lightweight, counts only); the
    association pass streams float64 chunks and row-filters each one, since
    the file cannot be row-filtered up front the way a matrix can.
    """

    def __init__(
        self,
        bed_path: Path,
        *,
        n_samples: int,
        n_snps: int,
        stats_chunk_size: int,
        validate_genotypes: bool,
        show_progress: bool,
    ) -> None:
        self._bed_path = bed_path
        self._n_samples = n_samples
        self._n_snps = n_snps
        self._stats_chunk_size = stats_chunk_size
        self._validate_genotypes = validate_genotypes
        self._show_progress = show_progress

    @property
    def n_snps(self) -> int:
        return self._n_snps

    def snp_stats(self, valid_mask: np.ndarray, *, include_hwe: bool) -> SnpStats:
        needs_filter = not bool(np.all(valid_mask))
        return collect_streamed_snp_stats(
            self._bed_path,
            n_snps=self._n_snps,
            n_samples=self._n_samples,
            chunk_size=self._stats_chunk_size,
            sample_indices=np.where(valid_mask)[0] if needs_filter else None,
            include_hwe=include_hwe,
            validate_genotypes=self._validate_genotypes,
            show_progress=self._show_progress,
            progress_label="Computing SNP statistics",
            dtype=np.float32,
            sample_scope="valid_samples" if needs_filter else "all_samples",
        )

    def chunks(
        self, chunk_size: int, snp_indices: np.ndarray, valid_mask: np.ndarray
    ) -> Callable[[], RawLmmChunk | None]:
        needs_filter = not bool(np.all(valid_mask))
        chunk_iter = iter(
            stream_genotype_chunks(
                self._bed_path,
                chunk_size=chunk_size,
                dtype=np.float64,
                show_progress=False,
                snp_indices=snp_indices,
            )
        )

        def _next_chunk() -> RawLmmChunk | None:
            try:
                chunk, filt_start, filt_end = next(chunk_iter)
            except StopIteration:
                return None

            if needs_filter:
                chunk = chunk[valid_mask, :]

            return RawLmmChunk(np.ascontiguousarray(chunk), filt_start, filt_end)

        return _next_chunk


def run_lmm_association_numpy_streaming(
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None = None,
    snp_info: list | None = None,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    chunk_size: int | None = None,
    output_path: Path | None = None,
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    validate_genotypes: bool = True,
    config: LmmConfig = DEFAULT_LMM_CONFIG,
) -> LmmRunResult:
    """Run LMM association tests by streaming genotypes from disk.

    Two-pass disk streaming: pass 1 computes SNP statistics for filtering,
    pass 2 runs the shared chunk engine per chunk. Never allocates the full
    genotype matrix.

    Args:
        bed_path: PLINK file prefix (without .bed/.bim/.fam extension).
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues and eigenvectors are provided.
        snp_info: List of SNP metadata dicts, or None to build from PLINK.
        covariates: Covariate matrix (n_samples, n_cvt) or None for
            intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        chunk_size: Cap on SNPs per chunk, for both the statistics pass and
            the association pass. None (default) reads statistics in
            _DEFAULT_STATS_CHUNK blocks and lets the chunk engine size the
            association chunks against the RAM budget.
        output_path: Path for incremental result writing, or None for
            in-memory.
        snps_indices: Pre-resolved column indices for -snps restriction,
            or None.
        hwe_threshold: HWE p-value threshold; SNPs with p < threshold are
            removed. 0.0 disables HWE filtering (default).
        validate_genotypes: Check for unexpected genotype values during
            pass 1.
        config: LmmConfig with thresholds, lambda bounds, test type,
            memory check and progress settings.

    Returns:
        LmmRunResult with associations (empty if output_path is set --
        results on disk), PVE from the null model, n_tested counting the
        SNPs that passed filtering and were tested, and the run's timing
        breakdown.
    """
    # Checked here, not in the shared body, because even the metadata read
    # touches disk: a bad value must fail before any file I/O.
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1 or None, got {chunk_size}")

    meta = get_plink_metadata(bed_path)

    # Caller-supplied list, or the PLINK metadata, parsed once into columns.
    snp_meta = (
        SnpMeta.from_plink_meta(meta)
        if snp_info is None
        else SnpMeta.from_dicts(snp_info)
    )

    source = BedSource(
        bed_path,
        n_samples=meta["n_samples"],
        n_snps=meta["n_snps"],
        stats_chunk_size=_DEFAULT_STATS_CHUNK if chunk_size is None else chunk_size,
        validate_genotypes=validate_genotypes,
        show_progress=config.show_progress,
    )
    return _run_numpy_lmm(
        source,
        phenotypes=phenotypes,
        kinship=kinship,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        snp_meta=snp_meta,
        config=config,
        output_path=output_path,
        snps_indices=snps_indices,
        hwe_threshold=hwe_threshold,
        max_chunk_size=chunk_size,
        banner="NumPy streaming",
        label="lmm_numpy_streaming",
        progress_label="LMM association (streaming)",
        log_dispatch_choices=False,
    )
