"""Result building and per-chunk sinks for LMM association tests.

Builds ``AssocResult`` objects from computed statistics for each test mode
(Wald, Score, LRT, All) via ``_build_results``, and exposes the per-chunk sink
factories (``make_writer_sink``, ``make_result_list_sink``) that the batch,
streaming, and LOCO NumPy runners share to route each result chunk to disk or an
in-memory list. Per-chunk NaN and lambda-boundary diagnostics are accumulated by
the shared chunk runner itself (``chunk_runner_numpy``), not here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import SnpMeta
from jamma.lmm.stats import AssocResult

if TYPE_CHECKING:
    from jamma.lmm.genotype_source import PreparedGenotypes
    from jamma.lmm.io import IncrementalAssocWriter

# Per-chunk result sink handed to the shared NumPy LMM chunk runner:
# (chunk_arrays, filtered_start, filtered_end) -> None.
ChunkSink = Callable[[dict[str, np.ndarray], int, int], None]

# Relative tolerance for detecting lambda convergence at optimization bounds
LAMBDA_BOUND_TOL = 1e-3


def _build_results(
    lmm_mode: int,
    snp_indices: np.ndarray,
    filtered_afs: np.ndarray,
    filtered_miss: np.ndarray,
    snp_info: SnpMeta,
    arrays: dict[str, np.ndarray],
) -> list[AssocResult]:
    """Build AssocResult objects for any LMM test mode.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        snp_indices: Indices of SNPs that passed filtering.
        filtered_afs: Allele frequencies for filtered SNPs.
        filtered_miss: Missing counts for filtered SNPs.
        snp_info: SNP metadata columns, indexed by global SNP index.
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
    chr_list = snp_info.chr[snp_indices].tolist()
    rs_list = snp_info.rs[snp_indices].tolist()
    pos_list = snp_info.pos[snp_indices].tolist()
    a1_list = snp_info.a1[snp_indices].tolist()
    a0_list = snp_info.a0[snp_indices].tolist()

    nan = float("nan")
    is_lrt = lmm_mode == 2
    results = []
    for j in range(len(snp_indices)):
        meta: dict[str, Any] = {
            "chr": chr_list[j],
            "rs": rs_list[j],
            "ps": pos_list[j],
            "n_miss": int(miss_list[j]),
            "allele1": a1_list[j],
            "allele0": a0_list[j],
            "af": af_list[j],
        }

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
    genotypes: PreparedGenotypes,
) -> ChunkSink:
    """Build a chunk sink that streams each result chunk to disk.

    The returned sink slices the prepared source's bound selection by the
    ``[filtered_start, filtered_end)`` range it receives per chunk.
    """
    selection = genotypes.selection

    def _sink(
        chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
    ) -> None:
        writer.write_arrays_batch(
            lmm_mode,
            selection.indices[filtered_start:filtered_end],
            genotypes.snp_meta,
            selection.filtered_afs[filtered_start:filtered_end],
            selection.filtered_miss[filtered_start:filtered_end],
            chunk_arrays,
        )

    return _sink


def make_result_list_sink(
    results: list[AssocResult],
    lmm_mode: int,
    genotypes: PreparedGenotypes,
) -> ChunkSink:
    """Build a chunk sink that appends built ``AssocResult`` objects to ``results``.

    Shared by the streaming and LOCO NumPy runners on their in-memory
    (no ``output_path``) path.
    """

    selection = genotypes.selection

    def _sink(
        chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
    ) -> None:
        results.extend(
            _build_results(
                lmm_mode,
                selection.indices[filtered_start:filtered_end],
                selection.filtered_afs[filtered_start:filtered_end],
                selection.filtered_miss[filtered_start:filtered_end],
                genotypes.snp_meta,
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
        lmin, lmax = _count_boundary_hits(np.asarray(arrays["lambdas"]), l_min, l_max)
        n_at_lmin += lmin
        n_at_lmax += lmax
    if lmm_mode in (2, 4):
        lmin, lmax = _count_boundary_hits(
            np.asarray(arrays["lambdas_mle"]), l_min, l_max
        )
        n_at_lmin += lmin
        n_at_lmax += lmax
    return n_at_lmin, n_at_lmax


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
