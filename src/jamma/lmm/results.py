"""Result building functions for LMM association tests.

Constructs AssocResult objects from computed statistics for each
test mode (Wald, Score, LRT, All). Used by batch runners (NumPy)
in both in-memory and output_path streaming modes via ``write_streaming_chunk``
(which wraps ``IncrementalAssocWriter.write_arrays_batch`` with diagnostic
accumulation).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.stats import AssocResult

if TYPE_CHECKING:
    from jamma.lmm.io import IncrementalAssocWriter

# Per-chunk result sink handed to the shared NumPy LMM chunk runner:
# (chunk_arrays, filtered_start, filtered_end) -> None.
ChunkSink = Callable[[dict[str, np.ndarray], int, int], None]

# Relative tolerance for detecting lambda convergence at optimization bounds
LAMBDA_BOUND_TOL = 1e-3


def _snp_metadata(snp_info: dict, af: float, n_miss: int) -> dict:
    """Extract common SNP metadata fields for AssocResult construction.

    Args:
        snp_info: SNP metadata dict with keys: chr, rs, pos/ps, a1/allele1, a0/allele0.
        af: Allele frequency of counted allele (BIM A1), can be > 0.5.
        n_miss: Missing genotype count.

    Returns:
        Dict of shared AssocResult fields.
    """
    return {
        "chr": snp_info["chr"],
        "rs": snp_info["rs"],
        "ps": snp_info.get("pos", snp_info.get("ps", 0)),
        "n_miss": n_miss,
        "allele1": snp_info.get("a1", snp_info.get("allele1", "")),
        "allele0": snp_info.get("a0", snp_info.get("allele0", "")),
        "af": af,
    }


def _build_results(
    lmm_mode: int,
    snp_indices: np.ndarray,
    filtered_afs: np.ndarray,
    filtered_miss: np.ndarray,
    snp_info: list,
    arrays: dict[str, np.ndarray],
) -> list[AssocResult]:
    """Build AssocResult objects for any LMM test mode.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        snp_indices: Indices of SNPs that passed filtering.
        filtered_afs: Allele frequencies for filtered SNPs.
        filtered_miss: Missing counts for filtered SNPs.
        snp_info: Full SNP metadata list.
        arrays: Dict mapping stat name -> numpy array of values.

    Returns:
        List of AssocResult objects.
    """
    if lmm_mode not in _RESULT_FIELDS:
        raise ValueError(
            f"Unknown lmm_mode={lmm_mode}; expected one of {list(_RESULT_FIELDS)}"
        )
    field_map = _RESULT_FIELDS[lmm_mode]
    missing_keys = set(field_map.keys()) - set(arrays.keys())
    if missing_keys:
        raise ValueError(
            f"Missing arrays for lmm_mode={lmm_mode}: {missing_keys}. "
            f"Expected keys: {set(field_map.keys())}, got: {set(arrays.keys())}"
        )
    # Convert stat arrays to Python lists in one C call each
    # (avoids per-element float() conversion overhead)
    stat_lists = {
        field_name: arrays[array_key].tolist()
        for array_key, field_name in field_map.items()
    }
    af_list = filtered_afs.tolist()
    miss_list = filtered_miss.tolist()

    nan = float("nan")
    is_lrt = lmm_mode == 2
    results = []
    for j, snp_idx in enumerate(snp_indices):
        meta = _snp_metadata(snp_info[snp_idx], af_list[j], int(miss_list[j]))

        if is_lrt:
            meta["beta"] = nan
            meta["se"] = nan

        for field_name, vals in stat_lists.items():
            meta[field_name] = vals[j]

        results.append(AssocResult(**meta))
    return results


def make_writer_sink(
    writer: IncrementalAssocWriter,
    lmm_mode: int,
    snp_info: list,
    snp_indices: np.ndarray,
    filtered_afs: np.ndarray,
    filtered_miss: np.ndarray,
) -> ChunkSink:
    """Build a chunk sink that streams each result chunk to disk.

    Shared by the batch, streaming, and LOCO NumPy runners, which previously
    each inlined a byte-identical ``writer.write_arrays_batch`` call. The
    ``snp_indices`` / ``filtered_afs`` / ``filtered_miss`` arrays are the full
    filtered-order arrays; the returned sink slices them by the
    ``[filtered_start, filtered_end)`` range it receives per chunk.
    """

    def _sink(
        chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
    ) -> None:
        writer.write_arrays_batch(
            lmm_mode,
            snp_indices[filtered_start:filtered_end],
            snp_info,
            filtered_afs[filtered_start:filtered_end],
            filtered_miss[filtered_start:filtered_end],
            chunk_arrays,
        )

    return _sink


def make_result_list_sink(
    results: list[AssocResult],
    lmm_mode: int,
    snp_info: list,
    snp_indices: np.ndarray,
    filtered_afs: np.ndarray,
    filtered_miss: np.ndarray,
) -> ChunkSink:
    """Build a chunk sink that appends built ``AssocResult`` objects to ``results``.

    Shared by the streaming and LOCO NumPy runners on their in-memory
    (no ``output_path``) path.
    """

    def _sink(
        chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
    ) -> None:
        results.extend(
            _build_results(
                lmm_mode,
                snp_indices[filtered_start:filtered_end],
                filtered_afs[filtered_start:filtered_end],
                filtered_miss[filtered_start:filtered_end],
                snp_info,
                chunk_arrays,
            )
        )

    return _sink


def _count_boundary_hits(
    lambdas: np.ndarray, l_min: float, l_max: float
) -> tuple[int, int]:
    """Count how many lambda values sit at the lower/upper optimization bound."""
    if len(lambdas) == 0:
        return 0, 0
    at_min = int(np.sum(lambdas / l_min < 1 + LAMBDA_BOUND_TOL))
    at_max = int(np.sum(lambdas / l_max > 1 - LAMBDA_BOUND_TOL))
    return at_min, at_max


def count_lambda_boundary_hits(
    lmm_mode: int,
    arrays: dict[str, np.ndarray],
    l_min: float,
    l_max: float,
) -> tuple[int, int]:
    """Count SNPs with lambda converging at optimization bounds.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        arrays: Dict of numpy arrays with per-SNP statistics.
        l_min: Lower lambda bound.
        l_max: Upper lambda bound.

    Returns:
        Tuple of (n_at_lmin, n_at_lmax).
    """
    n_at_lmin = 0
    n_at_lmax = 0
    if lmm_mode in (1, 4):
        lmin, lmax = _count_boundary_hits(
            np.asarray(arrays.get("lambdas", [])), l_min, l_max
        )
        n_at_lmin += lmin
        n_at_lmax += lmax
    if lmm_mode in (2, 4):
        lmin, lmax = _count_boundary_hits(
            np.asarray(arrays.get("lambdas_mle", [])), l_min, l_max
        )
        n_at_lmin += lmin
        n_at_lmax += lmax
    return n_at_lmin, n_at_lmax


def write_streaming_chunk(
    writer,
    lmm_mode: int,
    snp_indices: np.ndarray,
    snp_info: list,
    filtered_afs: np.ndarray,
    filtered_miss: np.ndarray,
    chunk_arrays: dict[str, np.ndarray],
    l_min: float,
    l_max: float,
    nan_counts: dict[str, int],
    n_at_lmin_accum: int,
    n_at_lmax_accum: int,
) -> tuple[int, int]:
    """Write a chunk to disk and accumulate diagnostics.

    Used by the batch and streaming runners when output_path is set.
    The disk-streaming runner handles its own write/diagnostic loop inline.

    Args:
        writer: IncrementalAssocWriter instance.
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        snp_indices: SNP column indices for this chunk.
        snp_info: Full SNP metadata list.
        filtered_afs: Allele frequencies for this chunk.
        filtered_miss: Missing counts for this chunk.
        chunk_arrays: Dict of per-SNP statistic arrays for this chunk.
        l_min: Lower lambda bound.
        l_max: Upper lambda bound.
        nan_counts: Mutable dict accumulating NaN counts per field.
        n_at_lmin_accum: Running count of SNPs at lower lambda bound.
        n_at_lmax_accum: Running count of SNPs at upper lambda bound.

    Returns:
        Updated (n_at_lmin_accum, n_at_lmax_accum).
    """
    writer.write_arrays_batch(
        lmm_mode, snp_indices, snp_info, filtered_afs, filtered_miss, chunk_arrays
    )
    chunk_lmin, chunk_lmax = count_lambda_boundary_hits(
        lmm_mode, chunk_arrays, l_min, l_max
    )
    for key, arr in chunk_arrays.items():
        n_nan = int(np.sum(np.isnan(arr)))
        if n_nan > 0:
            nan_counts[key] = nan_counts.get(key, 0) + n_nan
    return n_at_lmin_accum + chunk_lmin, n_at_lmax_accum + chunk_lmax


def log_lambda_boundary_warning(
    n_at_lmin: int,
    n_at_lmax: int,
    l_min: float,
    l_max: float,
    prefix: str = "",
) -> None:
    """Emit a warning if any SNPs converged at lambda bounds.

    Args:
        n_at_lmin: Count of SNPs at lower bound.
        n_at_lmax: Count of SNPs at upper bound.
        l_min: Lower lambda bound.
        l_max: Upper lambda bound.
        prefix: Optional prefix for log message (e.g. "LOCO ").
    """
    if n_at_lmin > 0 or n_at_lmax > 0:
        parts = []
        if n_at_lmin > 0:
            parts.append(f"{n_at_lmin} SNPs at l_min={l_min:.1e}")
        if n_at_lmax > 0:
            parts.append(f"{n_at_lmax} SNPs at l_max={l_max:.1e}")
        logger.warning(f"{prefix}Lambda bound convergence: {', '.join(parts)}")
