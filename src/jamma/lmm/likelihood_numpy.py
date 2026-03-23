"""Pure-NumPy batch LMM likelihood computation.

Replaces JAX vmap/lax.fori_loop with numpy broadcasting and Python loops.
All functions process a chunk (batch) of SNPs at once without JAX.

Design:
- batch_compute_uab_numpy: vectorized Uab for n_snps SNPs at once
- batch_compute_pab_numpy / _batch_compute_pab_varying_numpy: Pab for a batch
- batch_compute_iab_numpy: Iab (identity-weighted Pab)
- _compute_reml_const / compute_iab_invariant_scalars_ncvt1: precomputed constants
- golden_section_optimize_lambda_numpy / _mle: batch lambda optimization
- golden_section_optimize_lambda_split_ncvt1_numpy: split-Uab optimizer for n_cvt=1
- _batch_grid_reml_split_ncvt1_numpy / _batch_reml_at_lambda_split_ncvt1_numpy:
    split-Uab REML evaluation (invariant/varying separation for n_cvt=1)
- batch_calc_wald_stats_numpy / _from_pab / score / lrt: batch test statistics

No JAX imports anywhere in this module. Compatible with JAX-free environments.

Reference: likelihood_jax.py (ported to NumPy in this module).
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma.lmm.likelihood import _P_YY_MIN, build_index_table
from jamma.lmm.special import betainc_batch, chi2_sf_batch


class SplitUab(NamedTuple):
    """Split Uab components for n_cvt=1 — separates SNP-varying from invariant columns.

    Zero runtime cost (NamedTuple is just a tuple subclass). Prevents silent
    argument swaps between varying and invariant at call sites.
    """

    varying: np.ndarray
    """SNP-varying columns (n_snps, n_samples, 3) with order [wx, xx, xy]."""
    invariant: np.ndarray
    """SNP-invariant columns (n_samples, 3) with order [ww, wy, yy]."""


class SplitUabSoA(NamedTuple):
    """Split Uab in SoA layout for n_cvt=1 — optimised for SIMD C inner loops.

    SoA (Structure-of-Arrays) layout gives stride-1 access to each column,
    enabling AVX-512 contiguous loads instead of stride-N gathers.

    Zero runtime cost (NamedTuple is just a tuple subclass).
    """

    varying: np.ndarray
    """SNP-varying columns (n_snps, 3, n_samples) with axis-1 order [wx, xx, xy].

    Axis-1 columns are contiguous in memory (stride-1 over n_samples).
    """
    invariant: np.ndarray
    """SNP-invariant columns (3, n_samples) with axis-0 order [ww, wy, yy].

    Each row is contiguous in memory (stride-1 over n_samples).
    """


# Thread-local flag to deduplicate _guard_P_yy warning — fires hundreds of
# times per run (once per grid eval + golden section iter per chunk) which
# buries the meaningful first warning under identical log spam.
_p_yy_state = threading.local()


def reset_p_yy_warned() -> None:
    """Reset the P_yy warning flag so each LMM run gets its own warning."""
    _p_yy_state.warned = False


def _guard_P_yy(P_yy: np.ndarray) -> np.ndarray:
    """Clamp P_yy: negative -> NaN, near-zero -> _P_YY_MIN.

    Prevents NaN/Inf from log(P_yy) in degenerate SNPs. Downstream code
    detects NaN to mark those SNPs as invalid.

    Args:
        P_yy: Projected phenotype variance, any shape.

    Returns:
        Guarded P_yy with same shape.
    """
    n_negative = int(np.sum(P_yy < 0.0))
    if n_negative > 0 and not getattr(_p_yy_state, "warned", False):
        logger.warning(
            f"{n_negative} SNPs have negative P_yy — numerical breakdown. "
            "Kinship matrix may not be positive semi-definite."
        )
        _p_yy_state.warned = True
    P_yy = np.where(P_yy < 0.0, np.nan, P_yy)
    return np.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)


# ---------------------------------------------------------------------------
# Uab batch computation
# ---------------------------------------------------------------------------


def batch_compute_uab_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> np.ndarray:
    """Compute Uab matrices for all SNPs in a chunk.

    Direct port of likelihood_jax.py::batch_compute_uab with jnp replaced
    by np. Shape: (n_snps, n_samples, n_index).

    Args:
        n_cvt: Number of covariates. If 1, uses explicit fast-path broadcasting.
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        UtG: Rotated genotypes for all SNPs in this chunk (n_samples, n_snps).

    Returns:
        Uab matrices (n_snps, n_samples, n_index).
    """
    if UtG.shape[0] != UtW.shape[0]:
        raise ValueError(
            f"UtG shape {UtG.shape} has {UtG.shape[0]} rows but expected "
            f"{UtW.shape[0]} (n_samples from UtW). "
            f"Pass (n_samples, n_snps), not (n_snps, n_samples)."
        )
    if n_cvt == 1:
        return _batch_compute_uab_ncvt1_numpy(UtW, Uty, UtG)
    return _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG)


def _batch_compute_uab_ncvt1_numpy(
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> np.ndarray:
    """Fast path batch Uab for n_cvt=1 (intercept only).

    Indices for n_cvt=1:
      0: ww  (1,1), 1: wx  (1,2), 2: wy  (1,3)
      3: xx  (2,2), 4: xy  (2,3), 5: yy  (3,3)

    Args:
        UtW: Rotated covariates (n_samples, 1).
        Uty: Rotated phenotype (n_samples,).
        UtG: Rotated genotypes (n_samples, n_snps).

    Returns:
        Uab batch (n_snps, n_samples, 6).
    """
    n_samples, n_snps = UtG.shape
    w = UtW[:, 0]  # (n_samples,)
    UtG_T = UtG.T  # (n_snps, n_samples)

    # Pre-allocate and fill directly — avoids 2x memory spike from np.stack
    out = np.empty((n_snps, n_samples, 6), dtype=np.float64)

    # SNP-invariant fields (broadcast into pre-allocated slices)
    out[:, :, 0] = (w * w)[None, :]  # ww
    out[:, :, 2] = (w * Uty)[None, :]  # wy
    out[:, :, 5] = (Uty * Uty)[None, :]  # yy

    # SNP-varying fields
    out[:, :, 1] = w[None, :] * UtG_T  # wx
    out[:, :, 3] = UtG_T * UtG_T  # xx
    out[:, :, 4] = UtG_T * Uty[None, :]  # xy

    return out


def _batch_compute_uab_general_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> np.ndarray:
    """General batch Uab for arbitrary n_cvt -- fully vectorized over SNPs.

    Builds a (n_snps, n_samples, n_cvt+2) vectors array where only the
    genotype column (index n_cvt) varies per SNP. All other columns are
    broadcast from UtW/Uty. Eliminates per-SNP copy entirely.

    Args:
        n_cvt: Number of covariates (> 1).
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        UtG: Rotated genotypes (n_samples, n_snps).

    Returns:
        Uab batch (n_snps, n_samples, n_index).
    """
    table = build_index_table(n_cvt)
    n_samples, n_snps = UtG.shape
    n_index = table["n_index"]
    genotype_col = n_cvt  # 0-based index of X in vectors array

    # Build vectors_all: (n_snps, n_samples, n_cvt+2)
    # Columns 0..n_cvt-1: covariates (broadcast across SNPs)
    # Column n_cvt: genotype (per-SNP)
    # Column n_cvt+1: phenotype (broadcast across SNPs)
    vectors_all = np.empty((n_snps, n_samples, n_cvt + 2), dtype=np.float64)
    for j in range(n_cvt):
        vectors_all[:, :, j] = UtW[:, j][None, :]  # broadcast
    vectors_all[:, :, genotype_col] = UtG.T  # (n_snps, n_samples)
    vectors_all[:, :, n_cvt + 1] = Uty[None, :]  # broadcast

    # Compute all Uab columns vectorized over SNPs
    Uab_batch = np.empty((n_snps, n_samples, n_index), dtype=np.float64)
    for a_col, b_col, idx in table["uab_pairs"]:
        Uab_batch[:, :, idx] = vectors_all[:, :, a_col] * vectors_all[:, :, b_col]

    return Uab_batch


def _batch_compute_uab_varying_general_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    utg_t: np.ndarray,
) -> np.ndarray:
    """Direct SoA varying Uab for general n_cvt -- no full Uab materialization.

    Computes only the SNP-varying Uab columns directly in SoA layout
    (n_snps, n_var, n_samples). A column is varying if its (a_col, b_col)
    pair involves the genotype (0-based index = n_cvt).

    For each varying pair:
    - Both involve genotype (xx): utg_t * utg_t
    - One involves genotype (wx_i, xy): covariate/phenotype * utg_t

    This avoids materializing the full (n_snps, n_samples, n_index) Uab.

    Args:
        n_cvt: Number of covariates (> 1).
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        utg_t: Rotated genotypes (n_snps, n_samples). C-contiguous layout
            from jlinalg.dgemm(chunk, U, transa="T").

    Returns:
        Varying Uab in SoA layout (n_snps, n_var, n_samples).
    """
    from jamma.lmm.likelihood import classify_uab_columns

    _inv_indices, var_indices = classify_uab_columns(n_cvt)
    table = build_index_table(n_cvt)
    n_snps, n_samples = utg_t.shape
    n_var = len(var_indices)
    genotype_col = n_cvt  # 0-based index of X in vectors array

    # Map linear index -> position in var_indices for output placement
    var_index_to_row = {idx: row for row, idx in enumerate(var_indices)}

    # Build the non-genotype vectors for lookup: covariates and phenotype
    # vectors[j] = UtW[:, j] for j < n_cvt, vectors[n_cvt+1] = Uty
    vectors = np.column_stack([UtW, np.zeros(n_samples), Uty])  # (n_samples, n_cvt+2)

    result = np.empty((n_snps, n_var, n_samples), dtype=np.float64)

    for a_col, b_col, linear_idx in table["uab_pairs"]:
        if linear_idx not in var_index_to_row:
            continue  # invariant column, skip
        row = var_index_to_row[linear_idx]

        if a_col == genotype_col and b_col == genotype_col:
            # xx case: genotype * genotype
            result[:, row, :] = utg_t * utg_t
        elif a_col == genotype_col:
            # genotype * other (b_col is covariate or phenotype)
            result[:, row, :] = utg_t * vectors[:, b_col][None, :]
        else:
            # other * genotype (a_col is covariate or phenotype, b_col is genotype)
            result[:, row, :] = vectors[:, a_col][None, :] * utg_t

    return result


# ---------------------------------------------------------------------------
# Pab batch computation
# ---------------------------------------------------------------------------


def batch_compute_pab_numpy(
    n_cvt: int,
    Hi_eval: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Compute Pab for all SNPs using a shared Hi_eval vector.

    Used for Score test (fixed null lambda for all SNPs) and Iab computation.

    Row-0: tensordot(Hi_eval[n], Uab_batch[p,n,i]) -> (p, i) via BLAS.
    Rows 1..n_cvt+1: recursive projection vectorized over all SNPs.

    Args:
        n_cvt: Number of covariates.
        Hi_eval: Shared 1/(lambda*eval+1) vector (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Pab batch (n_snps, n_cvt+2, n_index).
    """
    table = build_index_table(n_cvt)
    n_snps, n_samples, n_index = Uab_batch.shape
    Pab_batch = np.zeros((n_snps, n_cvt + 2, n_index), dtype=np.float64)

    # Row 0: batched weighted dot product
    # tensordot contracts Hi_eval (axis 0) with Uab_batch (axis 1)
    # Result shape: (n_snps, n_index)
    Pab_batch[:, 0, :] = np.tensordot(Hi_eval, Uab_batch, axes=([0], [1]))

    # Rows 1..n_cvt+1: recursive projection (same for all SNPs)
    _fill_pab_recursion(Pab_batch, table, n_cvt)

    return Pab_batch


def _batch_compute_pab_varying_numpy(
    n_cvt: int,
    Hi_eval_batch: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Compute Pab for all SNPs with per-SNP Hi_eval vectors.

    Used during lambda optimization (each SNP has its own lambda -> own Hi_eval).

    Row-0: einsum('pn,pni->pi', Hi_eval_batch, Uab_batch).
    Rows 1..n_cvt+1: same recursive projection as batch_compute_pab_numpy.

    Args:
        n_cvt: Number of covariates.
        Hi_eval_batch: Per-SNP Hi_eval (n_snps, n_samples).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Pab batch (n_snps, n_cvt+2, n_index).
    """
    table = build_index_table(n_cvt)
    n_snps, n_samples, n_index = Uab_batch.shape
    Pab_batch = np.zeros((n_snps, n_cvt + 2, n_index), dtype=np.float64)

    # Row 0: per-SNP einsum
    # Hi_eval_batch: (n_snps, n_samples)
    # Uab_batch: (n_snps, n_samples, n_index)
    # -> (n_snps, n_index)
    Pab_batch[:, 0, :] = np.einsum("pn,pni->pi", Hi_eval_batch, Uab_batch)

    # Rows 1..n_cvt+1: recursive projection
    _fill_pab_recursion(Pab_batch, table, n_cvt)

    return Pab_batch


def _fill_pab_recursion(
    Pab_batch: np.ndarray,
    table: dict,
    n_cvt: int,
) -> None:
    """Fill rows 1..n_cvt+1 of Pab_batch using GEMMA's recursive formula.

    Mutates Pab_batch in-place. Row 0 must already be filled.
    All SNPs are processed simultaneously (vectorized over axis 0).

    Args:
        Pab_batch: (n_snps, n_cvt+2, n_index) array, row 0 pre-filled.
        table: Index table from build_index_table.
        n_cvt: Number of covariates.
    """
    n_degenerate = 0
    for p in range(1, n_cvt + 2):
        for _a, _b, index_ab, index_aw, index_bw, index_ww in table["pab_recursion"][p]:
            ps_ww = Pab_batch[..., p - 1, index_ww]
            # Guard: ps_ww=0 means degenerate covariate projection. Use 0
            # to prevent NaN/Inf propagation to valid SNPs in the same batch.
            # Degenerate SNPs are caught downstream by P_XX > 0 checks.
            n_degenerate += int(np.sum(ps_ww == 0))
            with np.errstate(divide="ignore"):
                safe_inv = np.where(ps_ww != 0, 1.0 / ps_ww, 0.0)
            Pab_batch[..., p, index_ab] = (
                Pab_batch[..., p - 1, index_ab]
                - Pab_batch[..., p - 1, index_aw]
                * Pab_batch[..., p - 1, index_bw]
                * safe_inv
            )
    if n_degenerate > 0:
        logger.debug(
            f"Pab recursion: {n_degenerate} degenerate ps_ww=0 entries guarded"
        )


def batch_compute_iab_numpy(
    n_cvt: int,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Compute identity-weighted Pab (Iab) for all SNPs.

    Iab = Pab with Hi_eval = ones. Used for logdet_hiw term in REML.
    Lambda-independent: precompute once per chunk.

    Args:
        n_cvt: Number of covariates.
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Iab batch (n_snps, n_cvt+2, n_index).
    """
    n_samples = Uab_batch.shape[1]
    ones = np.ones(n_samples, dtype=np.float64)
    return batch_compute_pab_numpy(n_cvt, ones, Uab_batch)


# ---------------------------------------------------------------------------
# Split Uab/Iab for n_cvt=1 — separates SNP-invariant columns
# ---------------------------------------------------------------------------


def batch_compute_uab_split_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> SplitUab:
    """Compute split Uab: SNP-varying and SNP-invariant components.

    For n_cvt=1, columns ww(0), wy(2), yy(5) are identical across all SNPs.
    This function returns them as a shared (n_samples, 3) array instead of
    broadcasting into every SNP row — halving Uab memory.

    Args:
        n_cvt: Number of covariates (must be 1).
        UtW: Rotated covariates (n_samples, 1).
        Uty: Rotated phenotype (n_samples,).
        UtG: Rotated genotypes (n_samples, n_snps).

    Returns:
        SplitUab(varying, invariant) where:
        - varying: (n_snps, n_samples, 3) — wx, xx, xy columns.
        - invariant: (n_samples, 3) — ww, wy, yy (shared).
    """
    if n_cvt != 1:
        raise ValueError("batch_compute_uab_split_numpy requires n_cvt=1")
    return _batch_compute_uab_split_ncvt1_numpy(UtW, Uty, UtG)


def _batch_compute_uab_split_ncvt1_numpy(
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> SplitUab:
    """Fast-path split Uab for n_cvt=1.

    Returns:
        SplitUab(varying, invariant):
        - varying: (n_snps, n_samples, 3) with col order [wx, xx, xy].
        - invariant: (n_samples, 3) with col order [ww, wy, yy].
    """
    n_samples, n_snps = UtG.shape
    w = UtW[:, 0]
    UtG_T = UtG.T  # (n_snps, n_samples)

    uab_varying = np.empty((n_snps, n_samples, 3), dtype=np.float64)
    uab_varying[:, :, 0] = w[None, :] * UtG_T  # wx
    uab_varying[:, :, 1] = UtG_T * UtG_T  # xx
    uab_varying[:, :, 2] = UtG_T * Uty[None, :]  # xy

    uab_invariant = np.empty((n_samples, 3), dtype=np.float64)
    uab_invariant[:, 0] = w * w  # ww
    uab_invariant[:, 1] = w * Uty  # wy
    uab_invariant[:, 2] = Uty * Uty  # yy

    return SplitUab(uab_varying, uab_invariant)


def batch_compute_uab_split_soa_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    utg_t: np.ndarray,
) -> SplitUabSoA:
    """Compute split Uab in SoA layout — eliminates per-chunk AoS->SoA transpose.

    Produces the SoA layout (n_snps, 3, n_samples) for varying and
    (3, n_samples) for invariant directly, without intermediate AoS allocation.
    The C extension's inner loops read stride-1 columns, enabling SIMD loads.

    Args:
        n_cvt: Number of covariates (must be 1).
        UtW: Rotated covariates (n_samples, 1).
        Uty: Rotated phenotype (n_samples,).
        utg_t: Rotated genotypes (n_snps, n_samples). C-contiguous layout
            from jlinalg.dgemm(chunk, U, transa="T").

    Returns:
        SplitUabSoA(varying, invariant) where:
        - varying: (n_snps, 3, n_samples) — rows are wx, xx, xy.
        - invariant: (3, n_samples) — rows are ww, wy, yy (shared).
    """
    if n_cvt != 1:
        raise ValueError("batch_compute_uab_split_soa_numpy requires n_cvt=1")
    inv = compute_uab_invariant_soa(UtW, Uty)
    var = batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, utg_t)
    return SplitUabSoA(var, inv)


def compute_uab_invariant_soa(
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int = 1,
) -> np.ndarray:
    """Compute SNP-invariant Uab columns in SoA layout (n_inv, n_samples).

    For n_cvt=1, rows are [ww, wy, yy] (3 rows). For n_cvt>1, invariant
    columns are identified via classify_uab_columns and extracted from a
    single representative Uab computed with a zero genotype vector.

    These columns depend only on UtW and Uty, so they can be computed once
    per run (before the chunk loop) rather than once per chunk.

    Args:
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        n_cvt: Number of covariates (default 1 for backwards compatibility).

    Returns:
        Invariant array (n_inv, n_samples) — SoA layout.
    """
    if n_cvt == 1:
        n_samples = Uty.shape[0]
        w = UtW[:, 0]
        uab_invariant_soa = np.empty((3, n_samples), dtype=np.float64)
        uab_invariant_soa[0, :] = w * w  # ww
        uab_invariant_soa[1, :] = w * Uty  # wy
        uab_invariant_soa[2, :] = Uty * Uty  # yy
        return uab_invariant_soa

    # General n_cvt: compute full Uab for a zero genotype vector,
    # then extract invariant columns.
    from jamma.lmm.likelihood import classify_uab_columns

    inv_indices, _var_indices = classify_uab_columns(n_cvt)
    n_samples = Uty.shape[0]

    # Build a single Uab with zero genotype (invariant columns are
    # independent of genotype, so the genotype value doesn't matter).
    UtG_zero = np.zeros((n_samples, 1), dtype=np.float64)
    Uab_single = _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG_zero)
    # Uab_single shape: (1, n_samples, n_index)
    # Extract invariant columns: advanced indexing a[0, :, list] groups the
    # integer (0) and list indices at front -> (n_inv, n_samples) SoA layout.
    uab_invariant_soa = np.ascontiguousarray(Uab_single[0, :, list(inv_indices)])
    return uab_invariant_soa


def batch_compute_uab_varying_soa_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    utg_t: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute SNP-varying Uab columns in SoA layout (n_snps, n_var, n_samples).

    For n_cvt=1, n_var=3 with rows [wx, xx, xy]. For n_cvt>1, varying columns
    are identified via classify_uab_columns and extracted from the full Uab batch.

    Args:
        n_cvt: Number of covariates.
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        utg_t: Rotated genotypes (n_snps, n_samples). C-contiguous layout
            from jlinalg.dgemm(chunk, U, transa="T").
        out: Optional pre-allocated output buffer (n_snps, n_var, n_samples).
            When provided and shape matches, writes directly into it to avoid
            allocation. Only used for n_cvt=1 path.

    Returns:
        Varying array (n_snps, n_var, n_samples) — SoA layout.
    """
    if utg_t.shape[1] != UtW.shape[0]:
        raise ValueError(
            f"utg_t shape {utg_t.shape} has {utg_t.shape[1]} columns but "
            f"expected {UtW.shape[0]} (n_samples from UtW). "
            f"Pass (n_snps, n_samples), not (n_samples, n_snps)."
        )
    if n_cvt == 1:
        n_snps, n_samples = utg_t.shape
        w = UtW[:, 0]

        expected_shape = (n_snps, 3, n_samples)
        if out is not None:
            if out.shape != expected_shape:
                raise ValueError(
                    f"batch_compute_uab_varying_soa_numpy: out shape {out.shape} "
                    f"doesn't match expected {expected_shape}"
                )
            uab_varying_soa = out
        else:
            uab_varying_soa = np.empty(expected_shape, dtype=np.float64)
        uab_varying_soa[:, 0, :] = w[None, :] * utg_t  # wx row
        uab_varying_soa[:, 1, :] = utg_t * utg_t  # xx row
        uab_varying_soa[:, 2, :] = utg_t * Uty[None, :]  # xy row
        return uab_varying_soa

    # General n_cvt: direct SoA varying without full Uab materialization
    if out is not None:
        raise ValueError(
            "batch_compute_uab_varying_soa_numpy: out= buffer not supported "
            f"for n_cvt={n_cvt} (only n_cvt=1)"
        )
    return _batch_compute_uab_varying_general_numpy(n_cvt, UtW, Uty, utg_t)


def reconstruct_uab_from_soa(
    uab_invariant_soa: np.ndarray,
    uab_varying_soa: np.ndarray,
    n_cvt: int = 1,
) -> np.ndarray:
    """Reconstruct full Uab matrix from split SoA components.

    For n_cvt=1: combines invariant columns [ww, wy, yy] with per-SNP varying
    columns [wx, xx, xy] into the standard layout (n_snps, n_samples, 6).

    For n_cvt>1: uses classify_uab_columns(n_cvt) to determine which linear
    column indices are invariant vs varying, then places each column at its
    correct index in the output array. Invariant columns are broadcast across
    all SNPs; varying columns are placed per-SNP from uab_varying_soa.

    Args:
        uab_invariant_soa: Shape (n_inv, n_samples) — one row per invariant column.
        uab_varying_soa: Shape (n_snps, n_var, n_samples) — one axis-1 row per
            varying column.
        n_cvt: Number of covariates. Default 1 preserves backward compatibility.

    Returns:
        Full Uab array (n_snps, n_samples, n_index) matching
        batch_compute_uab_numpy layout.
    """
    n_snps, _, n_samples = uab_varying_soa.shape

    if n_cvt == 1:
        # Fast path: hardcoded 6-column layout — zero overhead for common case.
        # n_cvt=1 column order: 0=ww, 1=wx, 2=wy, 3=xx, 4=xy, 5=yy
        Uab = np.empty((n_snps, n_samples, 6), dtype=np.float64)

        # Invariant columns — broadcast across all SNPs
        Uab[:, :, 0] = uab_invariant_soa[0]  # ww
        Uab[:, :, 2] = uab_invariant_soa[1]  # wy
        Uab[:, :, 5] = uab_invariant_soa[2]  # yy

        # Varying columns — per-SNP
        Uab[:, :, 1] = uab_varying_soa[:, 0, :]  # wx
        Uab[:, :, 3] = uab_varying_soa[:, 1, :]  # xx
        Uab[:, :, 4] = uab_varying_soa[:, 2, :]  # xy

        return Uab

    # General path for n_cvt > 1: use classify_uab_columns to get index mapping.
    # Function-level import avoids circular import (Phase 60 decision).
    from jamma.lmm.likelihood import classify_uab_columns

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2  # total Uab columns
    Uab = np.empty((n_snps, n_samples, n_index), dtype=np.float64)

    # Place invariant columns (broadcast across all SNPs)
    for row_i, col_idx in enumerate(inv_indices):
        Uab[:, :, col_idx] = uab_invariant_soa[row_i]

    # Place varying columns (per-SNP from SoA axis-1)
    for row_i, col_idx in enumerate(var_indices):
        Uab[:, :, col_idx] = uab_varying_soa[:, row_i, :]

    return Uab


def batch_compute_iab_split_ncvt1_soa(
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
) -> np.ndarray:
    """Compute Iab from split Uab in SoA layout (n_cvt=1 only).

    SoA variant of batch_compute_iab_split_ncvt1. Sums over axis=2 (n_samples)
    instead of axis=1 because SoA columns are on axis=2 rather than axis=1.
    Produces identical numerical results.

    Args:
        uab_varying_soa: (n_snps, 3, n_samples) — rows [wx, xx, xy].
        uab_invariant_soa: (3, n_samples) — rows [ww, wy, yy].

    Returns:
        Iab batch (n_snps, 3, 6).
    """
    n_snps = uab_varying_soa.shape[0]

    # Row 0: column sums (Hi_eval = ones -> just sum over samples, axis=2 for SoA)
    s_ww = uab_invariant_soa[0, :].sum()
    s_wy = uab_invariant_soa[1, :].sum()
    s_yy = uab_invariant_soa[2, :].sum()
    s_wx = uab_varying_soa[:, 0, :].sum(axis=1)  # (n_snps,)
    s_xx = uab_varying_soa[:, 1, :].sum(axis=1)
    s_xy = uab_varying_soa[:, 2, :].sum(axis=1)

    iab = np.zeros((n_snps, 3, 6), dtype=np.float64)
    iab[:, 0, 0] = s_ww
    iab[:, 0, 1] = s_wx
    iab[:, 0, 2] = s_wy
    iab[:, 0, 3] = s_xx
    iab[:, 0, 4] = s_xy
    iab[:, 0, 5] = s_yy

    # Row 1: project out W (Schur complement)
    inv_ww = 1.0 / s_ww if s_ww != 0 else 0.0
    iab[:, 1, 3] = s_xx - s_wx * s_wx * inv_ww
    iab[:, 1, 4] = s_xy - s_wx * s_wy * inv_ww
    iab[:, 1, 5] = s_yy - s_wy * s_wy * inv_ww

    # Row 2: project out X
    ps_xx = iab[:, 1, 3]
    with np.errstate(divide="ignore"):
        inv_xx = np.where(ps_xx != 0, 1.0 / ps_xx, 0.0)
    iab[:, 2, 5] = iab[:, 1, 5] - iab[:, 1, 4] * iab[:, 1, 4] * inv_xx

    return iab


def batch_compute_iab_split_ncvt1(
    uab_varying: np.ndarray,
    uab_invariant: np.ndarray,
) -> np.ndarray:
    """Compute Iab from split Uab components (n_cvt=1 only).

    Equivalent to batch_compute_iab_numpy(1, full_uab) but avoids
    constructing the full 6-column Uab.

    Args:
        uab_varying: (n_snps, n_samples, 3) — [wx, xx, xy].
        uab_invariant: (n_samples, 3) — [ww, wy, yy].

    Returns:
        Iab batch (n_snps, 3, 6).
    """
    n_snps = uab_varying.shape[0]

    # Row 0: column sums (Hi_eval = ones → just sum over samples)
    s_ww, s_wy, s_yy = (
        uab_invariant[:, 0].sum(),
        uab_invariant[:, 1].sum(),
        uab_invariant[:, 2].sum(),
    )
    s_wx = uab_varying[:, :, 0].sum(axis=1)  # (n_snps,)
    s_xx = uab_varying[:, :, 1].sum(axis=1)
    s_xy = uab_varying[:, :, 2].sum(axis=1)

    iab = np.zeros((n_snps, 3, 6), dtype=np.float64)
    iab[:, 0, 0] = s_ww
    iab[:, 0, 1] = s_wx
    iab[:, 0, 2] = s_wy
    iab[:, 0, 3] = s_xx
    iab[:, 0, 4] = s_xy
    iab[:, 0, 5] = s_yy

    # Row 1: project out W (Schur complement)
    inv_ww = 1.0 / s_ww if s_ww != 0 else 0.0
    iab[:, 1, 3] = s_xx - s_wx * s_wx * inv_ww
    iab[:, 1, 4] = s_xy - s_wx * s_wy * inv_ww
    iab[:, 1, 5] = s_yy - s_wy * s_wy * inv_ww

    # Row 2: project out X
    ps_xx = iab[:, 1, 3]
    with np.errstate(divide="ignore"):
        inv_xx = np.where(ps_xx != 0, 1.0 / ps_xx, 0.0)
    iab[:, 2, 5] = iab[:, 1, 5] - iab[:, 1, 4] * iab[:, 1, 4] * inv_xx

    return iab


# ---------------------------------------------------------------------------
# Precomputed REML/MLE constants and Iab invariant scalars
# ---------------------------------------------------------------------------


def _compute_reml_const(df: int) -> float:
    """Precompute REML normalizing constant: 0.5 * df * (log(df) - log(2*pi) - 1).

    Constant across all SNPs and lambda values — compute once per run.

    Args:
        df: Degrees of freedom (n_samples - n_cvt - 1).

    Returns:
        REML normalizing constant.
    """
    return 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)


def compute_iab_invariant_scalars_ncvt1(
    uab_invariant_soa: np.ndarray,
) -> tuple[float, float, float, float]:
    """Precompute Iab invariant scalars for n_cvt=1.

    These are the simple sums of the invariant Uab columns (Hi_eval = ones),
    constant across all chunks and all lambda values. Compute once at run start.

    Args:
        uab_invariant_soa: (3, n_samples) — rows [ww, wy, yy].

    Returns:
        (iab_s_ww, iab_s_wy, iab_s_yy, logdet_iab) where:
        - iab_s_ww/wy/yy: simple sums of invariant columns
        - logdet_iab: log(iab_s_ww) — the Iab diagonal for REML logdet_hiw
    """
    iab_s_ww = float(uab_invariant_soa[0, :].sum())
    iab_s_wy = float(uab_invariant_soa[1, :].sum())
    iab_s_yy = float(uab_invariant_soa[2, :].sum())
    logdet_iab = np.log(iab_s_ww) if iab_s_ww > 0 else 0.0
    return iab_s_ww, iab_s_wy, iab_s_yy, logdet_iab


# ---------------------------------------------------------------------------
# Batch REML / MLE log-likelihood evaluation
# ---------------------------------------------------------------------------


def _batch_reml_at_lambda_numpy(
    n_cvt: int,
    lambda_vals: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Iab_batch: np.ndarray,
    reml_const: float | None = None,
    return_pab: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate REML log-likelihood for each SNP at its own lambda value.

    Args:
        n_cvt: Number of covariates.
        lambda_vals: Per-SNP lambda values (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).
        reml_const: Precomputed 0.5*df*(log(df)-log(2*pi)-1). If None, computed here.
        return_pab: If True, also return Pab_batch for downstream Wald stats.

    Returns:
        If return_pab=False: REML log-likelihoods (n_snps,).
        If return_pab=True: (log-likelihoods, Pab_batch) where Pab is
        (n_snps, n_cvt+2, n_index).
    """
    table = build_index_table(n_cvt)
    n_snps = Uab_batch.shape[0]
    n = eigenvalues.shape[0]
    nc_total = n_cvt + 1
    df = n - n_cvt - 1

    # Per-SNP H-inv weights: (n_snps, n_samples)
    v_temp = lambda_vals[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_batch = 1.0 / v_temp  # (n_snps, n_samples)

    # Log determinant of H per SNP: (n_snps,)
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)

    # Pab with per-SNP Hi_eval
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)

    # logdet_hiw per SNP: sum over diagonal indices
    # Guard: non-positive diagonal Pab/Iab entries (degenerate SNPs) use 0.0
    # instead of log to prevent NaN/Inf from corrupting the batch. Degenerate
    # SNPs are caught downstream by the P_yy < 0 → NaN guard.
    logdet_hiw = np.zeros(n_snps, dtype=np.float64)
    for row, col in table["logdet_diag_indices"]:
        d_pab = Pab_batch[:, row, col]  # (n_snps,)
        d_iab = Iab_batch[:, row, col]  # (n_snps,)
        with np.errstate(divide="ignore", invalid="ignore"):
            logdet_hiw += np.where(d_pab > 0, np.log(d_pab), 0.0)
            logdet_hiw -= np.where(d_iab > 0, np.log(d_iab), 0.0)

    # P_yy per SNP with guards
    P_yy = _guard_P_yy(Pab_batch[:, nc_total, table["idx_yy"]])

    # REML log-likelihood per SNP
    if reml_const is None:
        reml_const = 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)
    logl = reml_const - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)
    if return_pab:
        return logl, Pab_batch
    return logl


def _batch_mle_at_lambda_numpy(
    n_cvt: int,
    lambda_vals: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate MLE log-likelihood for each SNP at its own lambda value.

    Key differences from REML: no logdet_hiw term, uses n instead of df.

    Args:
        n_cvt: Number of covariates.
        lambda_vals: Per-SNP lambda values (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        MLE log-likelihoods (n_snps,).
    """
    table = build_index_table(n_cvt)
    n = eigenvalues.shape[0]
    nc_total = n_cvt + 1

    # Per-SNP H-inv weights: (n_snps, n_samples)
    v_temp = lambda_vals[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_batch = 1.0 / v_temp

    # Log determinant of H per SNP: (n_snps,)
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)

    # Pab with per-SNP Hi_eval
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)

    # P_yy per SNP with guards
    P_yy = _guard_P_yy(Pab_batch[:, nc_total, table["idx_yy"]])

    # MLE log-likelihood per SNP (no logdet_hiw, uses n not df)
    c = 0.5 * n * (np.log(n) - np.log(2.0 * np.pi) - 1.0)
    return c - 0.5 * logdet_h - 0.5 * n * np.log(P_yy)


# ---------------------------------------------------------------------------
# Grid evaluations
# ---------------------------------------------------------------------------


def _batch_grid_pab_numpy(
    n_cvt: int,
    lambdas_grid: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute grid-level Pab, logdet(H), and guarded P_yy.

    Shared computation for both REML and MLE grid evaluation. Since all SNPs
    share the same lambda at each grid point, Hi_eval is (n_grid, n_samples)
    not (n_snps, n_samples) -- eliminates the dominant memory allocation at
    scale.

    Args:
        n_cvt: Number of covariates.
        lambdas_grid: Grid of lambda values (n_grid,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Tuple of (Pab, logdet_h, P_yy):
            Pab: (n_grid, n_snps, n_cvt+2, n_index)
            logdet_h: (n_grid,)
            P_yy: (n_grid, n_snps) -- guarded
    """
    table = build_index_table(n_cvt)
    n_snps, _n_samples, n_index = Uab_batch.shape
    n_grid = len(lambdas_grid)
    nc_total = n_cvt + 1

    # All grid lambdas at once: (n_grid, n_samples)
    v_temp = lambdas_grid[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_grid = 1.0 / v_temp

    # logdet(H) per grid lambda -- shared across SNPs: (n_grid,)
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)

    # Pab row 0 via single tensordot (BLAS gemm):
    # Hi_eval_grid (n_grid, n_samples) x Uab_batch (n_snps, n_samples, n_index)
    # contracts n_samples -> (n_grid, n_snps, n_index)
    Pab = np.zeros((n_grid, n_snps, n_cvt + 2, n_index), dtype=np.float64)
    Pab[:, :, 0, :] = np.tensordot(Hi_eval_grid, Uab_batch, axes=([1], [1]))

    # Recursive rows 1..n_cvt+1 (uses ... indexing for 4D)
    _fill_pab_recursion(Pab, table, n_cvt)

    P_yy = _guard_P_yy(Pab[:, :, nc_total, table["idx_yy"]])

    return Pab, logdet_h, P_yy


def _batch_grid_reml_numpy(
    n_cvt: int,
    lambdas_grid: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Iab_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate REML at all grid lambda values for all SNPs.

    Args:
        n_cvt: Number of covariates.
        lambdas_grid: Grid of lambda values (n_grid,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed Iab (n_snps, n_cvt+2, n_index).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """
    table = build_index_table(n_cvt)
    n_snps = Uab_batch.shape[0]
    n_samples = eigenvalues.shape[0]
    df = n_samples - n_cvt - 1

    Pab, logdet_h, P_yy = _batch_grid_pab_numpy(
        n_cvt, lambdas_grid, eigenvalues, Uab_batch
    )
    n_grid = len(lambdas_grid)

    # logdet_hiw: Iab part is lambda-independent, precompute once
    logdet_iab = np.zeros(n_snps, dtype=np.float64)
    logdet_pab = np.zeros((n_grid, n_snps), dtype=np.float64)
    for row, col in table["logdet_diag_indices"]:
        d_pab = Pab[:, :, row, col]  # (n_grid, n_snps)
        d_iab = Iab_batch[:, row, col]  # (n_snps,)
        with np.errstate(divide="ignore", invalid="ignore"):
            logdet_pab += np.where(d_pab > 0, np.log(d_pab), 0.0)
            logdet_iab += np.where(d_iab > 0, np.log(d_iab), 0.0)
    logdet_hiw = logdet_pab - logdet_iab[None, :]

    # REML log-likelihood: (n_grid, n_snps)
    c = 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)
    return c - 0.5 * logdet_h[:, None] - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)


def _batch_grid_mle_numpy(
    n_cvt: int,
    lambdas_grid: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate MLE at all grid lambda values for all SNPs.

    Args:
        n_cvt: Number of covariates.
        lambdas_grid: Grid of lambda values (n_grid,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """
    n_samples = eigenvalues.shape[0]

    _Pab, logdet_h, P_yy = _batch_grid_pab_numpy(
        n_cvt, lambdas_grid, eigenvalues, Uab_batch
    )

    # MLE log-likelihood: (n_grid, n_snps) -- no logdet_hiw, uses n not df
    c = 0.5 * n_samples * (np.log(n_samples) - np.log(2.0 * np.pi) - 1.0)
    return c - 0.5 * logdet_h[:, None] - 0.5 * n_samples * np.log(P_yy)


# ---------------------------------------------------------------------------
# Golden section optimizer
# ---------------------------------------------------------------------------


def _batch_golden_section_numpy(
    compute_batch_fn,
    grid_logls: np.ndarray,
    log_lambdas: np.ndarray,
    n_iter: int,
    compute_batch_with_pab_fn=None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid-to-golden-section refinement for lambda optimization.

    Direct translation of likelihood_jax.py::_golden_section_refine
    with jnp -> np and lax.fori_loop -> Python for loop.

    All operations are vectorized over SNPs (axis 0).
    After 20 iterations: 0.618^20 ~ 6.6e-5 relative tolerance.

    Args:
        compute_batch_fn: callable(log_lambdas_per_snp: (n_snps,)) -> (n_snps,).
        grid_logls: Grid log-likelihoods (n_grid, n_snps).
        log_lambdas: Log-scale grid points (n_grid,).
        n_iter: Golden section iterations (should be >= 20).
        compute_batch_with_pab_fn: Optional callable(log_lambdas: (n_snps,)) ->
            (logls (n_snps,), Pab_batch (n_snps, n_cvt+2, n_index)).
            If provided, performs a final evaluation at the optimal midpoint
            using this function and returns Pab alongside lambdas/logls.

    Returns:
        If compute_batch_with_pab_fn is None: (optimal_lambdas, optimal_logls).
        If provided: (optimal_lambdas, optimal_logls, Pab_final).
    """
    phi = 0.6180339887498949  # golden ratio - 1

    # Find best grid point per SNP and bracket
    safe_logls = np.where(np.isnan(grid_logls), -np.inf, grid_logls)
    best_idx = np.argmax(safe_logls, axis=0)  # (n_snps,)
    idx_low = np.maximum(best_idx - 1, 0)
    idx_high = np.minimum(best_idx + 1, len(log_lambdas) - 1)

    a = log_lambdas[idx_low]  # (n_snps,)
    b = log_lambdas[idx_high]  # (n_snps,)

    # Initial probe points
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = compute_batch_fn(c)
    fd = compute_batch_fn(d)

    # Golden section iterations (Python for loop, vectorized over SNPs)
    for _ in range(n_iter):
        keep_left = fc > fd  # (n_snps,) boolean

        new_a = np.where(keep_left, a, c)
        new_b = np.where(keep_left, d, b)
        new_c = new_b - phi * (new_b - new_a)
        new_d = new_a + phi * (new_b - new_a)

        new_logl = compute_batch_fn(np.where(keep_left, new_c, new_d))
        new_fc = np.where(keep_left, new_logl, fd)
        new_fd = np.where(keep_left, fc, new_logl)

        a, b, c, d, fc, fd = new_a, new_b, new_c, new_d, new_fc, new_fd

    log_opt = (a + b) / 2.0

    if compute_batch_with_pab_fn is not None:
        # Final eval at midpoint — captures Pab for downstream Wald stats.
        # This makes the Wald path's "final evaluation" productive (its Pab
        # is reused) rather than the optimizer returning best-of-fc/fd and
        # then batch_calc_wald_stats_numpy reconstructing Hi_eval + Pab again.
        opt_logl, Pab_final = compute_batch_with_pab_fn(log_opt)
        return np.exp(log_opt), opt_logl, Pab_final

    # Evaluate logl at the midpoint to match lambda — ensures the returned
    # (lambda, logl) pair is from the same evaluation point. This matches
    # the JAX path (likelihood_jax.py line 531) which also evaluates at midpoint.
    opt_logl = compute_batch_fn(log_opt)
    return np.exp(log_opt), opt_logl


def golden_section_optimize_lambda_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Iab_batch: np.ndarray,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
    return_pab: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize REML lambda using grid search + golden section refinement.

    Port of likelihood_jax.py::golden_section_optimize_lambda. Replaces
    jnp/vmap with np broadcasting, and lax.fori_loop with Python for loop.

    Enforces minimum of 20 golden section iterations to guarantee
    lambda relative tolerance < 1e-5 (matching GEMMA Brent tolerance).

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations (should be >= 20 for 1e-5 tolerance;
            runner-level code enforces the minimum).
        return_pab: If True, return (lambdas, logls, Pab_final) where Pab_final
            is the Pab batch at the optimal lambda. Avoids redundant Hi_eval +
            Pab reconstruction in the Wald stats step.

    Returns:
        If return_pab=False: (optimal_lambdas, optimal_logls) both (n_snps,).
        If return_pab=True: (optimal_lambdas, optimal_logls, Pab_final) where
        Pab_final is (n_snps, n_cvt+2, n_index).
    """
    log_l_min = np.log(l_min)
    log_l_max = np.log(l_max)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)
    lambdas_grid = np.exp(log_lambdas)

    n = eigenvalues.shape[0]
    df = n - n_cvt - 1
    reml_const = _compute_reml_const(df)

    # Stage 1: Coarse grid search
    grid_logls = _batch_grid_reml_numpy(
        n_cvt, lambdas_grid, eigenvalues, Uab_batch, Iab_batch
    )

    # REML batch evaluator closure (over precomputed Iab and reml_const)
    def compute_reml_batch(log_lams: np.ndarray) -> np.ndarray:
        lams = np.exp(log_lams)
        return _batch_reml_at_lambda_numpy(
            n_cvt, lams, eigenvalues, Uab_batch, Iab_batch, reml_const=reml_const
        )

    # Stage 2: Golden section refinement
    pab_fn = None
    if return_pab:

        def pab_fn(log_lams: np.ndarray) -> tuple:
            lams = np.exp(log_lams)
            return _batch_reml_at_lambda_numpy(
                n_cvt,
                lams,
                eigenvalues,
                Uab_batch,
                Iab_batch,
                reml_const=reml_const,
                return_pab=True,
            )

    return _batch_golden_section_numpy(
        compute_reml_batch,
        grid_logls,
        log_lambdas,
        n_iter,
        compute_batch_with_pab_fn=pab_fn,
    )


def golden_section_optimize_lambda_mle_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimize MLE lambda using grid search + golden section refinement.

    Port of likelihood_jax.py::golden_section_optimize_lambda_mle.
    No Iab argument needed (MLE has no logdet_hiw term).

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations (should be >= 20 for 1e-5 tolerance;
            runner-level code enforces the minimum).

    Returns:
        (optimal_lambdas, optimal_logls_mle) both shape (n_snps,).
    """
    log_l_min = np.log(l_min)
    log_l_max = np.log(l_max)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)
    lambdas_grid = np.exp(log_lambdas)

    # Stage 1: Coarse grid search
    grid_logls = _batch_grid_mle_numpy(n_cvt, lambdas_grid, eigenvalues, Uab_batch)

    # MLE batch evaluator closure
    def compute_mle_batch(log_lams: np.ndarray) -> np.ndarray:
        lams = np.exp(log_lams)
        return _batch_mle_at_lambda_numpy(n_cvt, lams, eigenvalues, Uab_batch)

    # Stage 2: Golden section refinement
    return _batch_golden_section_numpy(
        compute_mle_batch, grid_logls, log_lambdas, n_iter
    )


# ---------------------------------------------------------------------------
# Split-Uab REML path for n_cvt=1 (grid + refinement + optimizer)
# ---------------------------------------------------------------------------


def _compute_iab_varying_ncvt1(
    uab_varying_soa: np.ndarray,
    iab_inv_s_ww: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-SNP Iab varying quantities for n_cvt=1.

    These are lambda-independent (Iab uses Hi_eval=ones) and constant
    across all grid/refinement evaluations. Compute once per optimizer call.

    Args:
        uab_varying_soa: (n_snps, 3, n_samples) — [wx, xx, xy].
        iab_inv_s_ww: Precomputed 1/iab_s_ww.

    Returns:
        (iab_p1_xx, iab_logdet_var) both (n_snps,).
    """
    iab_s_wx = uab_varying_soa[:, 0, :].sum(axis=1)  # (n_snps,)
    iab_s_xx = uab_varying_soa[:, 1, :].sum(axis=1)
    iab_p1_xx = iab_s_xx - iab_s_wx * iab_s_wx * iab_inv_s_ww
    with np.errstate(divide="ignore", invalid="ignore"):
        iab_logdet_var = np.where(iab_p1_xx > 0, np.log(iab_p1_xx), 0.0)
    return iab_p1_xx, iab_logdet_var


def _batch_grid_reml_split_ncvt1_numpy(
    lambdas_grid: np.ndarray,
    eigenvalues: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    iab_logdet: float,
    iab_inv_s_ww: float,
    iab_p1_xx: np.ndarray,
    iab_logdet_var: np.ndarray,
    reml_const: float,
) -> np.ndarray:
    """Evaluate REML at grid lambda values using split-Uab for n_cvt=1.

    Invariant quantities (s_ww, s_wy, s_yy and their Schur complements) are
    computed once per grid point — O(n_grid * n_samples), not
    O(n_grid * n_snps * n_samples).

    Only the varying columns (wx, xx, xy) are contracted per-SNP.

    Args:
        lambdas_grid: Grid lambda values (n_grid,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_varying_soa: (n_snps, 3, n_samples) — [wx, xx, xy].
        uab_invariant_soa: (3, n_samples) — [ww, wy, yy].
        iab_logdet: Precomputed log(iab_s_ww) for logdet_hiw.
        iab_inv_s_ww: Precomputed 1/iab_s_ww for Iab Schur complement.
        iab_p1_xx: Precomputed per-SNP Iab p1_xx (n_snps,).
        iab_logdet_var: Precomputed per-SNP log(iab_p1_xx) (n_snps,).
        reml_const: Precomputed 0.5 * df * (log(df) - log(2*pi) - 1).

    Returns:
        REML log-likelihoods (n_grid, n_snps).
    """
    n_samples = eigenvalues.shape[0]
    df = n_samples - 2  # n_cvt=1 -> df = n - 1 - 1

    # Hi_eval_grid: (n_grid, n_samples)
    v_temp = lambdas_grid[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_grid = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)  # (n_grid,)

    # --- Invariant dot products: (n_grid,) — once per grid point ---
    s_ww_grid = Hi_eval_grid @ uab_invariant_soa[0]  # (n_grid,)
    s_wy_grid = Hi_eval_grid @ uab_invariant_soa[1]  # (n_grid,)
    s_yy_grid = Hi_eval_grid @ uab_invariant_soa[2]  # (n_grid,)

    # Invariant Pab row 1: project out W
    with np.errstate(divide="ignore"):
        inv_s_ww_grid = np.where(s_ww_grid != 0, 1.0 / s_ww_grid, 0.0)
    p1_yy_grid = s_yy_grid - s_wy_grid * s_wy_grid * inv_s_ww_grid  # (n_grid,)

    # logdet_hiw invariant part: log(s_ww) - log(iab_s_ww) per grid point
    with np.errstate(divide="ignore", invalid="ignore"):
        logdet_pab_inv = np.where(s_ww_grid > 0, np.log(s_ww_grid), 0.0)
    logdet_hiw_inv = logdet_pab_inv - iab_logdet  # (n_grid,)

    # --- Varying dot products: (n_grid, n_snps, 3) ---
    # Hi_eval_grid: (n_grid, n_samples), uab_varying_soa: (n_snps, 3, n_samples)
    # Contract over n_samples -> (n_grid, n_snps, 3)
    s_varying = np.einsum("gn,pjn->gpj", Hi_eval_grid, uab_varying_soa)
    s_wx = s_varying[:, :, 0]  # (n_grid, n_snps)
    s_xx = s_varying[:, :, 1]
    s_xy = s_varying[:, :, 2]

    # --- Full Pab recursion using invariant + varying ---
    # Row 1 varying: p1_xx, p1_xy (broadcast inv_s_ww_grid)
    p1_xx = s_xx - s_wx * s_wx * inv_s_ww_grid[:, None]
    p1_xy = s_xy - s_wx * s_wy_grid[:, None] * inv_s_ww_grid[:, None]

    # Row 2: P_yy = p1_yy - p1_xy^2 / p1_xx
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_p1_xx = np.where(p1_xx != 0, 1.0 / p1_xx, 0.0)
    P_yy = p1_yy_grid[:, None] - p1_xy * p1_xy * inv_p1_xx  # (n_grid, n_snps)
    P_yy = _guard_P_yy(P_yy)

    # logdet_hiw = (log(s_ww) - log(iab_s_ww)) + (log(p1_xx) - log(iab_p1_xx))
    with np.errstate(divide="ignore", invalid="ignore"):
        logdet_pab_var = np.where(p1_xx > 0, np.log(p1_xx), 0.0)  # (n_grid, n_snps)
    logdet_hiw = logdet_hiw_inv[:, None] + logdet_pab_var - iab_logdet_var[None, :]

    return (
        reml_const
        - 0.5 * logdet_h[:, None]
        - 0.5 * logdet_hiw
        - 0.5 * df * np.log(P_yy)
    )


def _batch_reml_at_lambda_split_ncvt1_numpy(
    lambda_vals: np.ndarray,
    eigenvalues: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    iab_logdet: float,
    iab_inv_s_ww: float,
    iab_p1_xx: np.ndarray,
    iab_logdet_var: np.ndarray,
    reml_const: float,
    return_pab: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate REML for each SNP at its own lambda using split-Uab (n_cvt=1).

    Same split logic as grid version but with per-SNP lambda values.

    Args:
        lambda_vals: Per-SNP lambda values (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_varying_soa: (n_snps, 3, n_samples).
        uab_invariant_soa: (3, n_samples).
        iab_logdet: Precomputed log(iab_s_ww).
        iab_inv_s_ww: Precomputed 1/iab_s_ww.
        iab_p1_xx: Precomputed per-SNP Iab p1_xx (n_snps,).
        iab_logdet_var: Precomputed per-SNP log(iab_p1_xx) (n_snps,).
        reml_const: Precomputed REML constant.
        return_pab: If True, also return the full Pab batch (n_snps, 3, 6).

    Returns:
        If return_pab=False: REML log-likelihoods (n_snps,).
        If return_pab=True: (log-likelihoods, Pab_batch (n_snps, 3, 6)).
    """
    n_samples = eigenvalues.shape[0]
    n_snps = uab_varying_soa.shape[0]
    df = n_samples - 2

    # Per-SNP Hi_eval: (n_snps, n_samples)
    v_temp = lambda_vals[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_batch = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)  # (n_snps,)

    # Invariant dot products: (n_snps,) — per-SNP lambda, but shared invariant cols
    s_ww = Hi_eval_batch @ uab_invariant_soa[0]  # (n_snps,)
    s_wy = Hi_eval_batch @ uab_invariant_soa[1]
    s_yy = Hi_eval_batch @ uab_invariant_soa[2]

    # Varying dot products: einsum for per-SNP contraction
    # Hi_eval_batch: (n_snps, n_samples), uab_varying_soa: (n_snps, 3, n_samples)
    s_varying = np.einsum("pn,pjn->pj", Hi_eval_batch, uab_varying_soa)
    s_wx = s_varying[:, 0]
    s_xx = s_varying[:, 1]
    s_xy = s_varying[:, 2]

    # Pab recursion
    with np.errstate(divide="ignore"):
        inv_s_ww = np.where(s_ww != 0, 1.0 / s_ww, 0.0)
    p1_xx = s_xx - s_wx * s_wx * inv_s_ww
    p1_xy = s_xy - s_wx * s_wy * inv_s_ww
    p1_yy = s_yy - s_wy * s_wy * inv_s_ww

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_p1_xx = np.where(p1_xx != 0, 1.0 / p1_xx, 0.0)
    P_yy = _guard_P_yy(p1_yy - p1_xy * p1_xy * inv_p1_xx)

    # logdet_hiw
    with np.errstate(divide="ignore", invalid="ignore"):
        logdet_pab_inv = np.where(s_ww > 0, np.log(s_ww), 0.0)
        logdet_pab_var = np.where(p1_xx > 0, np.log(p1_xx), 0.0)
    logdet_hiw = (logdet_pab_inv - iab_logdet) + (logdet_pab_var - iab_logdet_var)

    logl = reml_const - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)

    if not return_pab:
        return logl

    # Reconstruct full Pab (n_snps, 3, 6) for n_cvt=1:
    # Row 0: Hi_eval-weighted dot products — [ww, wx, wy, xx, xy, yy]
    # Row 1: Schur complement projecting out W — [xx, xy, yy] (cols 3,4,5)
    # Row 2: Schur complement projecting out X — [yy] (col 5)
    Pab_batch = np.zeros((n_snps, 3, 6), dtype=np.float64)
    Pab_batch[:, 0, 0] = s_ww
    Pab_batch[:, 0, 1] = s_wx
    Pab_batch[:, 0, 2] = s_wy
    Pab_batch[:, 0, 3] = s_xx
    Pab_batch[:, 0, 4] = s_xy
    Pab_batch[:, 0, 5] = s_yy
    Pab_batch[:, 1, 3] = p1_xx
    Pab_batch[:, 1, 4] = p1_xy
    Pab_batch[:, 1, 5] = p1_yy
    Pab_batch[:, 2, 5] = P_yy  # already guarded

    return logl, Pab_batch


def golden_section_optimize_lambda_split_ncvt1_numpy(
    eigenvalues: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    iab_s_ww: float,
    iab_s_wy: float,
    iab_s_yy: float,
    iab_logdet: float,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
    return_pab: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize REML lambda using split-Uab for n_cvt=1.

    Uses invariant/varying split to reduce per-SNP computation.
    Precomputes all Iab-derived quantities once.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,).
        uab_varying_soa: (n_snps, 3, n_samples).
        uab_invariant_soa: (3, n_samples).
        iab_s_ww: Precomputed Iab s_ww scalar.
        iab_s_wy: Precomputed Iab s_wy scalar.
        iab_s_yy: Precomputed Iab s_yy scalar.
        iab_logdet: Precomputed log(iab_s_ww).
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations.
        return_pab: If True, return (lambdas, logls, Pab_final) where Pab_final
            is the full Pab batch (n_snps, 3, 6) at optimal lambda. Avoids
            redundant Hi_eval + Pab reconstruction in the Wald stats step.

    Returns:
        If return_pab=False: (optimal_lambdas, optimal_logls) both (n_snps,).
        If return_pab=True: (optimal_lambdas, optimal_logls, Pab_final) where
        Pab_final is (n_snps, 3, 6).
    """
    n_samples = eigenvalues.shape[0]
    df = n_samples - 2
    reml_const = _compute_reml_const(df)

    # Precompute Iab quantities
    iab_inv_s_ww = 1.0 / iab_s_ww if iab_s_ww != 0 else 0.0

    # Per-SNP Iab varying quantities (constant across lambda)
    iab_p1_xx, iab_logdet_var = _compute_iab_varying_ncvt1(
        uab_varying_soa, iab_inv_s_ww
    )

    log_l_min = np.log(l_min)
    log_l_max = np.log(l_max)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)
    lambdas_grid = np.exp(log_lambdas)

    # Grid search
    grid_logls = _batch_grid_reml_split_ncvt1_numpy(
        lambdas_grid,
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_logdet,
        iab_inv_s_ww,
        iab_p1_xx,
        iab_logdet_var,
        reml_const,
    )

    # Refinement closure (scalar logls only — no Pab)
    def compute_reml_split(log_lams: np.ndarray) -> np.ndarray:
        lams = np.exp(log_lams)
        return _batch_reml_at_lambda_split_ncvt1_numpy(
            lams,
            eigenvalues,
            uab_varying_soa,
            uab_invariant_soa,
            iab_logdet,
            iab_inv_s_ww,
            iab_p1_xx,
            iab_logdet_var,
            reml_const,
        )

    pab_fn = None
    if return_pab:

        def pab_fn(log_lams: np.ndarray) -> tuple:
            lams = np.exp(log_lams)
            return _batch_reml_at_lambda_split_ncvt1_numpy(
                lams,
                eigenvalues,
                uab_varying_soa,
                uab_invariant_soa,
                iab_logdet,
                iab_inv_s_ww,
                iab_p1_xx,
                iab_logdet_var,
                reml_const,
                return_pab=True,
            )

    return _batch_golden_section_numpy(
        compute_reml_split,
        grid_logls,
        log_lambdas,
        n_iter,
        compute_batch_with_pab_fn=pab_fn,
    )


# ---------------------------------------------------------------------------
# Shared helpers for batch test statistics
# ---------------------------------------------------------------------------


def _beta_se_from_pab(
    P_XX: np.ndarray,
    P_XY: np.ndarray,
    Px_YY: np.ndarray,
    df: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute beta, SE, and validity mask from Pab projections.

    Shared by Wald and Score tests. Handles degenerate SNPs (P_XX <= 0)
    by setting beta/SE to NaN and GEMMA-compatible safe_sqrt for tiny
    variance values.

    Args:
        P_XX: Projected genotype variance per SNP (n_snps,).
        P_XY: Projected genotype-phenotype covariance per SNP (n_snps,).
        Px_YY: Projected phenotype variance at n_cvt+1 level (n_snps,).
        df: Degrees of freedom (n_samples - n_cvt - 1).

    Returns:
        Tuple of (beta, se, is_valid) each shape (n_snps,).
    """
    is_valid = P_XX > 0
    safe_P_XX = np.where(is_valid, P_XX, 1.0)

    beta = np.where(is_valid, P_XY / safe_P_XX, np.nan)
    tau = df / Px_YY
    variance_beta = np.where(is_valid, 1.0 / (tau * safe_P_XX), np.nan)
    # safe_sqrt: for |v| < 0.001, use abs(v) to avoid sqrt of tiny negative
    # values from FP rounding (matches GEMMA lmm.cpp safe_sqrt behaviour)
    variance_safe = np.where(
        np.abs(variance_beta) < 0.001,
        np.abs(variance_beta),
        variance_beta,
    )
    # np.where evaluates sqrt on all elements including NaN/negative variance_safe
    # from invalid SNPs; those results are discarded by the is_valid mask
    with np.errstate(invalid="ignore"):
        se = np.where(is_valid, np.sqrt(variance_safe), np.nan)

    return beta, se, is_valid


def _f_to_pvalue(f_stat: np.ndarray, df: int, is_valid: np.ndarray) -> np.ndarray:
    """Convert F-statistics to p-values via regularized incomplete beta.

    Shared by Wald and Score test computations. Uses algebraically exact
    complement_z = f_safe / (df + f_safe) to avoid cancellation near z = 1.

    Args:
        f_stat: F-statistics per SNP (n_snps,).
        df: Degrees of freedom (n_samples - n_cvt - 1).
        is_valid: Boolean mask of valid (non-degenerate) SNPs.

    Returns:
        P-values (n_snps,), NaN for invalid SNPs.
    """
    f_safe = np.maximum(f_stat, 1e-10)
    denom = df + f_safe
    z = np.clip(df / denom, 0.0, 1.0)
    complement_z = f_safe / denom  # algebraically exact 1-z, avoids cancellation
    a_arr = np.full_like(z, df / 2.0)
    b_arr = np.full_like(z, 0.5)
    p_val = betainc_batch(a_arr, b_arr, z, complement_z)
    p_val = np.where(f_stat <= 0, 1.0, p_val)
    return np.where(is_valid, p_val, np.nan)


# ---------------------------------------------------------------------------
# Batch test statistics
# ---------------------------------------------------------------------------


def batch_calc_wald_stats_numpy(
    n_cvt: int,
    lambdas: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Wald test statistics for a batch of SNPs.

    Port of likelihood_jax.py::batch_calc_wald_stats. Computes per-SNP
    Hi_eval from optimized lambdas, constructs Pab, then delegates to
    batch_calc_wald_stats_from_pab_numpy for the statistics.

    Args:
        n_cvt: Number of covariates.
        lambdas: Optimized REML lambda per SNP (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_walds) each shape (n_snps,).
    """
    Hi_eval_batch = 1.0 / (lambdas[:, None] * eigenvalues[None, :] + 1.0)
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)
    return batch_calc_wald_stats_from_pab_numpy(n_cvt, Pab_batch, n_samples)


def batch_calc_wald_stats_from_pab_numpy(
    n_cvt: int,
    Pab_batch: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Wald test statistics from pre-computed Pab batch.

    Used when the REML optimizer has already computed Pab at the optimal lambda
    and returned it via return_pab=True. Avoids the redundant Hi_eval + Pab
    construction in batch_calc_wald_stats_numpy.

    Args:
        n_cvt: Number of covariates.
        Pab_batch: Pre-computed Pab (n_snps, n_cvt+2, n_index) at optimal lambdas.
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_walds) each shape (n_snps,).
    """
    table = build_index_table(n_cvt)
    idx_xx = table["idx_xx"]
    idx_xy = table["idx_xy"]
    idx_yy = table["idx_yy"]
    df = n_samples - n_cvt - 1

    P_XX = Pab_batch[:, n_cvt, idx_xx]
    P_XY = Pab_batch[:, n_cvt, idx_xy]
    P_YY = Pab_batch[:, n_cvt, idx_yy]
    Px_YY = Pab_batch[:, n_cvt + 1, idx_yy]

    # Clamp Px_YY (matches batch_calc_wald_stats_numpy behaviour)
    Px_YY = np.where((Px_YY >= 0.0) & (Px_YY < _P_YY_MIN), _P_YY_MIN, Px_YY)

    beta, se, is_valid = _beta_se_from_pab(P_XX, P_XY, Px_YY, df)

    tau = df / Px_YY
    f_stat = (P_YY - Px_YY) * tau
    p_wald = _f_to_pvalue(f_stat, df, is_valid)

    return beta, se, p_wald


def batch_calc_score_stats_numpy(
    n_cvt: int,
    Hi_eval_null: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Score test statistics for a batch of SNPs.

    Port of likelihood_jax.py::batch_calc_score_stats. Uses fixed null-model
    Hi_eval shared across all SNPs (cheaper than Wald — no per-SNP optimization).

    Score F-statistic uses n_samples (not df) in numerator and P_yy*P_xx
    denominator (not Px_yy). Matches GEMMA CalcRLScore exactly.

    Args:
        n_cvt: Number of covariates.
        Hi_eval_null: Null-model 1/(lambda_null*eval+1) vector (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_scores) each shape (n_snps,).
    """
    table = build_index_table(n_cvt)
    idx_xx = table["idx_xx"]
    idx_xy = table["idx_xy"]
    idx_yy = table["idx_yy"]
    df = n_samples - n_cvt - 1

    # Batch Pab with shared null Hi_eval
    Pab_batch = batch_compute_pab_numpy(n_cvt, Hi_eval_null, Uab_batch)

    # Score test: extract at level n_cvt (covariates only, NOT genotype)
    P_yy = Pab_batch[:, n_cvt, idx_yy]
    P_yy = np.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)
    P_xx = Pab_batch[:, n_cvt, idx_xx]
    P_xy = Pab_batch[:, n_cvt, idx_xy]

    # Px_yy for beta/se computation
    Px_yy = Pab_batch[:, n_cvt + 1, idx_yy]
    Px_yy = np.where((Px_yy >= 0.0) & (Px_yy < _P_YY_MIN), _P_YY_MIN, Px_yy)

    beta, se, is_valid = _beta_se_from_pab(P_xx, P_xy, Px_yy, df)

    # Score F-statistic: F = n * P_xy^2 / (P_yy * P_xx)
    safe_P_xx = np.where(is_valid, P_xx, 1.0)
    f_stat = n_samples * (P_xy * P_xy) / (P_yy * safe_P_xx)

    # p_score via Cephes betainc
    p_score = _f_to_pvalue(f_stat, df, is_valid)

    return beta, se, p_score


def _batch_lrt_pvalues_numpy(
    logls_mle: np.ndarray,
    logl_H0: float,
) -> np.ndarray:
    """Compute LRT p-values for a batch of SNPs.

    Port of likelihood_jax.py::calc_lrt_pvalue_jax for batch use.
    LRT statistic = 2 * (logl_H1 - logl_H0), chi2 with df=1.

    Uses special.chi2_sf_batch (erfc-based, stdlib-only).

    Args:
        logls_mle: Per-SNP MLE log-likelihoods under alternative (n_snps,).
        logl_H0: Null model MLE log-likelihood (scalar).

    Returns:
        LRT p-values (n_snps,).
    """
    lrt_stats = 2.0 * (logls_mle - logl_H0)
    lrt_stats = np.maximum(lrt_stats, 0.0)
    p_lrts = chi2_sf_batch(lrt_stats)
    # Propagate NaN from MLE optimization failures (degenerate SNPs)
    p_lrts = np.where(np.isnan(logls_mle), np.nan, p_lrts)
    return p_lrts
