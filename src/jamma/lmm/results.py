"""Result building functions for LMM association tests.

Constructs AssocResult objects from computed statistics for each
test mode (Wald, Score, LRT, All). Used by both batch and streaming runners.
Note: the disk-write hot path bypasses this module entirely — see
``IncrementalAssocWriter.write_arrays_batch`` in ``io.py``.
"""

from collections.abc import Generator

import jax.numpy as jnp
import numpy as np
from loguru import logger

from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.stats import AssocResult

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
    results = []
    for j, snp_idx in enumerate(snp_indices):
        af = float(filtered_afs[j])
        n_miss = int(filtered_miss[j])
        meta = _snp_metadata(snp_info[snp_idx], af, n_miss)

        # LRT mode: beta and se are NaN (not computed)
        if lmm_mode == 2:
            meta["beta"] = float("nan")
            meta["se"] = float("nan")

        # Populate stat fields from arrays
        for array_key, field_name in field_map.items():
            meta[field_name] = float(arrays[array_key][j])

        results.append(AssocResult(**meta))
    return results


def _concat_jax_accumulators(
    lmm_mode: int,
    accumulators: dict[str, list],
) -> dict[str, np.ndarray]:
    """Concatenate JAX accumulator lists and transfer to host numpy arrays.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        accumulators: Dict mapping stat names to lists of JAX arrays.

    Returns:
        Dict mapping stat names to concatenated numpy arrays.
    """
    return {k: np.asarray(jnp.concatenate(v)) for k, v in accumulators.items()}


def _yield_chunk_results(
    lmm_mode: int,
    chunk_filtered_local_idx: list[int],
    snp_indices: np.ndarray,
    filtered_afs: np.ndarray,
    filtered_miss: np.ndarray,
    snp_info: list,
    arrays: dict[str, np.ndarray],
) -> Generator[AssocResult, None, None]:
    """Yield AssocResult objects for a streaming file chunk.

    Builds one result per filtered SNP in the chunk from the concatenated
    numpy arrays returned by _concat_jax_accumulators.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        chunk_filtered_local_idx: Indices within the filtered SNP arrays.
        snp_indices: Full array of filtered SNP indices.
        filtered_afs: Allele frequencies for filtered SNPs (numpy array).
        filtered_miss: Missing counts for filtered SNPs (numpy int array).
        snp_info: Full SNP metadata list.
        arrays: Dict of numpy arrays from _concat_jax_accumulators.

    Yields:
        AssocResult for each SNP in the chunk.
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
    for j, local_idx in enumerate(chunk_filtered_local_idx):
        snp_idx = snp_indices[local_idx]
        af = float(filtered_afs[local_idx])
        n_miss = int(filtered_miss[local_idx])
        meta = _snp_metadata(snp_info[snp_idx], af, n_miss)

        # LRT mode: beta and se are NaN (not computed)
        if lmm_mode == 2:
            meta["beta"] = float("nan")
            meta["se"] = float("nan")

        # Populate stat fields from arrays
        for array_key, field_name in field_map.items():
            meta[field_name] = float(arrays[array_key][j])

        yield AssocResult(**meta)


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
        arrays: Dict of numpy arrays from _concat_jax_accumulators.
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
