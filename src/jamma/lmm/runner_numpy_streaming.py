"""Disk-streaming NumPy LMM association runner.

``BedSource`` streams a PLINK .bed twice (float32 statistics pass, float64
association pass) without ever allocating the full genotype matrix; the run
itself is the shared body in ``runner_numpy``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from jamma.core.snp_filter import validate_snp_indices
from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpSelection,
    collect_streamed_snp_stats,
)
from jamma.io.plink import PlinkMetadata, get_plink_metadata, stream_genotype_chunks
from jamma.lmm.association_plan import (
    ExecutableAssociationPlan,
    plan_association,
)
from jamma.lmm.chunk_runner_numpy import RawLmmChunk
from jamma.lmm.genotype_source import (
    PreparedGenotypes,
    SampleBasis,
    bind_prepared_genotypes,
)
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
    """A PLINK .bed file as a genotype source.

    The statistics pass reads float32 blocks (lightweight, counts only); the
    association pass streams float64 chunks and row-filters each one, since
    the file cannot be row-filtered up front the way a matrix can.
    """

    def __init__(
        self,
        bed_path: Path,
        *,
        snp_meta: SnpMeta,
        n_samples: int,
        n_snps: int,
        stats_chunk_size: int,
        validate_genotypes: bool,
        show_progress: bool,
    ) -> None:
        if len(snp_meta) != n_snps:
            raise ValueError(
                "BED SNP count must match paired SnpMeta: "
                f"got {n_snps} SNPs and {len(snp_meta)} metadata rows"
            )
        self._bed_path = bed_path
        self._snp_meta = snp_meta
        self._n_samples = n_samples
        self._n_snps = n_snps
        self._stats_chunk_size = stats_chunk_size
        self._validate_genotypes = validate_genotypes
        self._show_progress = show_progress

    @property
    def n_snps(self) -> int:
        return self._n_snps

    def prepare(
        self, samples: SampleBasis, filters: SnpFilterSpec
    ) -> PreparedGenotypes:
        if samples.source_row_count != self._n_samples:
            raise ValueError(
                "sample basis row count must match BED rows: "
                f"got {samples.source_row_count} and {self._n_samples}"
            )
        stats = collect_streamed_snp_stats(
            self._bed_path,
            n_snps=self._n_snps,
            n_samples=self._n_samples,
            chunk_size=self._stats_chunk_size,
            sample_indices=None if samples.is_all_samples else samples.positions,
            include_hwe=filters.hwe_threshold > 0,
            validate_genotypes=self._validate_genotypes,
            show_progress=self._show_progress,
            progress_label="Computing SNP statistics",
            dtype=np.float32,
            sample_scope="all_samples" if samples.is_all_samples else "valid_samples",
        )

        def _iter_chunks(
            selection: SnpSelection, chunk_size: int
        ) -> Iterator[RawLmmChunk]:
            for chunk, filt_start, filt_end in stream_genotype_chunks(
                self._bed_path,
                chunk_size=chunk_size,
                dtype=np.float64,
                show_progress=False,
                snp_indices=selection.indices,
            ):
                if not samples.is_all_samples:
                    chunk = chunk[samples.positions, :]
                yield RawLmmChunk(np.ascontiguousarray(chunk), filt_start, filt_end)

        return bind_prepared_genotypes(
            snp_meta=self._snp_meta,
            stats=stats,
            filters=filters,
            chunk_source=_iter_chunks,
        )


def run_lmm_association_numpy_streaming(
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None = None,
    snp_info: list | SnpMeta | None = None,
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
        snp_info: SnpMeta, a list of SNP metadata dicts, or None to build
            from PLINK.
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
    n_cvt = covariates.shape[1] if covariates is not None else 1
    execution = plan_association(
        meta.n_samples,
        meta.n_snps,
        requested="numpy-streaming",
        n_cvt=n_cvt,
        lmm_mode=config.lmm_mode,
        mem_budget=config.mem_budget,
        max_chunk_size=chunk_size,
    )
    return _run_lmm_association_numpy_streaming_planned(
        bed_path=bed_path,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        chunk_size=chunk_size,
        output_path=output_path,
        snps_indices=snps_indices,
        hwe_threshold=hwe_threshold,
        validate_genotypes=validate_genotypes,
        config=config,
        execution=execution,
        _meta=meta,
    )


def _run_lmm_association_numpy_streaming_planned(
    *,
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    snp_info: list | SnpMeta | None,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    chunk_size: int | None,
    output_path: Path | None,
    snps_indices: np.ndarray | None,
    hwe_threshold: float,
    validate_genotypes: bool,
    config: LmmConfig,
    execution: ExecutableAssociationPlan,
    _meta: PlinkMetadata | None = None,
) -> LmmRunResult:
    """Run the streaming boundary with policy supplied by the pipeline."""
    meta = get_plink_metadata(bed_path) if _meta is None else _meta
    validate_snp_indices(snps_indices, meta.n_snps)

    # Caller-supplied SnpMeta or list, or the PLINK metadata parsed once.
    if snp_info is None:
        snp_meta = SnpMeta.from_plink_meta(meta)
    elif isinstance(snp_info, SnpMeta):
        snp_meta = snp_info
    else:
        snp_meta = SnpMeta.from_dicts(snp_info)

    source = BedSource(
        bed_path,
        snp_meta=snp_meta,
        n_samples=meta.n_samples,
        n_snps=meta.n_snps,
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
        config=config,
        output_path=output_path,
        snps_indices=snps_indices,
        hwe_threshold=hwe_threshold,
        execution=execution,
        banner="NumPy streaming",
        label="lmm_numpy_streaming",
        progress_label="LMM association (streaming)",
    )
