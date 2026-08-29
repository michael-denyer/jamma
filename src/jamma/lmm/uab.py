"""Uab, Pab and Iab batches: the products every likelihood evaluation reads.

Uab holds the element-wise products of the rotated vectors (covariates,
genotype, phenotype) for a chunk of SNPs; Pab is the H-inverse weighted
projection GEMMA's CalcPab recurses over; Iab is Pab with unit weights. The
SoA layout separates the columns that vary per SNP from the ones that do
not, so the C kernels and the n_cvt=1 fallback read stride-1 rows.

Every walker here iterates the ``PabIndexTable`` from ``build_index_table``,
the same integers in the same order as the scalar ``calc_pab``.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from jamma.lmm.likelihood import (
    _NCVT1,
    PabIndexTable,
    build_index_table,
    classify_uab_columns,
)


def batch_compute_uab_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    utg_t: np.ndarray,
) -> np.ndarray:
    """Compute Uab matrices for all SNPs in a chunk.

    Batch computation of Uab matrices using NumPy broadcasting.
    Shape: (n_snps, n_samples, n_index).

    Args:
        n_cvt: Number of covariates. If 1, uses explicit fast-path broadcasting.
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        utg_t: Rotated genotypes (n_snps, n_samples). C-contiguous layout
            from jlinalg.dgemm(chunk, U, transa="T"), the same layout
            batch_compute_uab_varying_soa_numpy takes.

    Returns:
        Uab matrices (n_snps, n_samples, n_index).
    """
    _check_utg_t(utg_t, UtW)
    if n_cvt == 1:
        return _batch_compute_uab_ncvt1_numpy(UtW, Uty, utg_t)
    return _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, utg_t)


def _check_utg_t(utg_t: np.ndarray, UtW: np.ndarray) -> None:
    if utg_t.shape[1] != UtW.shape[0]:
        raise ValueError(
            f"utg_t shape {utg_t.shape} has {utg_t.shape[1]} columns but "
            f"expected {UtW.shape[0]} (n_samples from UtW). "
            f"Pass (n_snps, n_samples), not (n_samples, n_snps)."
        )


def _batch_compute_uab_ncvt1_numpy(
    UtW: np.ndarray,
    Uty: np.ndarray,
    utg_t: np.ndarray,
) -> np.ndarray:
    """Fast path batch Uab for n_cvt=1 (intercept only). Columns follow _NCVT1.

    Args:
        UtW: Rotated covariates (n_samples, 1).
        Uty: Rotated phenotype (n_samples,).
        utg_t: Rotated genotypes (n_snps, n_samples).

    Returns:
        Uab batch (n_snps, n_samples, 6).
    """
    n_snps, n_samples = utg_t.shape
    w = UtW[:, 0]  # (n_samples,)

    # Pre-allocate and fill directly — avoids 2x memory spike from np.stack
    out = np.empty((n_snps, n_samples, 6), dtype=np.float64)

    # SNP-invariant fields (broadcast into pre-allocated slices)
    out[:, :, _NCVT1.ww] = (w * w)[None, :]
    out[:, :, _NCVT1.wy] = (w * Uty)[None, :]
    out[:, :, _NCVT1.yy] = (Uty * Uty)[None, :]

    # SNP-varying fields
    out[:, :, _NCVT1.wx] = w[None, :] * utg_t
    out[:, :, _NCVT1.xx] = utg_t * utg_t
    out[:, :, _NCVT1.xy] = utg_t * Uty[None, :]

    return out


def _batch_compute_uab_general_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    utg_t: np.ndarray,
) -> np.ndarray:
    """General batch Uab for arbitrary n_cvt -- fully vectorized over SNPs.

    Builds a (n_snps, n_samples, n_cvt+2) vectors array where only the
    genotype column (index n_cvt) varies per SNP. All other columns are
    broadcast from UtW/Uty. Eliminates per-SNP copy entirely.

    Args:
        n_cvt: Number of covariates (> 1).
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        utg_t: Rotated genotypes (n_snps, n_samples).

    Returns:
        Uab batch (n_snps, n_samples, n_index).
    """
    table = build_index_table(n_cvt)
    n_snps, n_samples = utg_t.shape
    n_index = table.n_index
    genotype_col = n_cvt  # 0-based index of X in vectors array

    # Build vectors_all: (n_snps, n_samples, n_cvt+2)
    # Columns 0..n_cvt-1: covariates (broadcast across SNPs)
    # Column n_cvt: genotype (per-SNP)
    # Column n_cvt+1: phenotype (broadcast across SNPs)
    vectors_all = np.empty((n_snps, n_samples, n_cvt + 2), dtype=np.float64)
    for j in range(n_cvt):
        vectors_all[:, :, j] = UtW[:, j][None, :]  # broadcast
    vectors_all[:, :, genotype_col] = utg_t
    vectors_all[:, :, n_cvt + 1] = Uty[None, :]  # broadcast

    # Compute all Uab columns vectorized over SNPs
    Uab_batch = np.empty((n_snps, n_samples, n_index), dtype=np.float64)
    for a_col, b_col, idx in table.uab_pairs:
        Uab_batch[:, :, idx] = vectors_all[:, :, a_col] * vectors_all[:, :, b_col]

    return Uab_batch


def _batch_compute_uab_varying_general_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    utg_t: np.ndarray,
    out: np.ndarray,
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
        out: Output buffer (n_snps, n_var, n_samples), already validated by
            batch_compute_uab_varying_soa_numpy.

    Returns:
        ``out``, filled.
    """
    _inv_indices, var_indices = classify_uab_columns(n_cvt)
    table = build_index_table(n_cvt)
    n_samples = utg_t.shape[1]
    genotype_col = n_cvt  # 0-based index of X in vectors array

    # Map linear index -> position in var_indices for output placement
    var_index_to_row = {idx: row for row, idx in enumerate(var_indices)}

    # Build the non-genotype vectors for lookup: covariates and phenotype
    # vectors[j] = UtW[:, j] for j < n_cvt, vectors[n_cvt+1] = Uty
    vectors = np.column_stack([UtW, np.zeros(n_samples), Uty])  # (n_samples, n_cvt+2)

    for a_col, b_col, linear_idx in table.uab_pairs:
        if linear_idx not in var_index_to_row:
            continue  # invariant column, skip
        row = var_index_to_row[linear_idx]

        if a_col == b_col == genotype_col:
            # xx case: genotype * genotype
            out[:, row, :] = utg_t * utg_t
        elif a_col == genotype_col:
            # genotype * other (b_col is covariate or phenotype)
            out[:, row, :] = utg_t * vectors[:, b_col][None, :]
        else:
            # other * genotype (a_col is covariate or phenotype, b_col is genotype)
            out[:, row, :] = vectors[:, a_col][None, :] * utg_t

    return out


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
    table: PabIndexTable,
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
        for _a, _b, index_ab, index_aw, index_bw, index_ww in table.pab_recursion[p]:
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


def compute_uab_invariant_soa(
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
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
        n_cvt: Number of covariates.

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

    inv_indices, _var_indices = classify_uab_columns(n_cvt)
    n_samples = Uty.shape[0]

    # Build a single Uab with zero genotype (invariant columns are
    # independent of genotype, so the genotype value doesn't matter).
    utg_t_zero = np.zeros((1, n_samples), dtype=np.float64)
    Uab_single = _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, utg_t_zero)
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
            allocation.

    Returns:
        Varying array (n_snps, n_var, n_samples) — SoA layout.
    """
    _check_utg_t(utg_t, UtW)
    n_snps, n_samples = utg_t.shape
    n_var = 3 if n_cvt == 1 else len(classify_uab_columns(n_cvt)[1])
    expected_shape = (n_snps, n_var, n_samples)
    if out is None:
        out = np.empty(expected_shape, dtype=np.float64)
    else:
        if out.shape != expected_shape:
            raise ValueError(
                f"batch_compute_uab_varying_soa_numpy: out shape {out.shape} "
                f"doesn't match expected {expected_shape}"
            )
        if out.dtype != np.float64:
            raise ValueError(
                f"batch_compute_uab_varying_soa_numpy: out dtype {out.dtype} "
                f"must be float64"
            )
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "batch_compute_uab_varying_soa_numpy: out must be C-contiguous"
            )

    if n_cvt == 1:
        w = UtW[:, 0]
        out[:, 0, :] = w[None, :] * utg_t  # wx row
        out[:, 1, :] = utg_t * utg_t  # xx row
        out[:, 2, :] = utg_t * Uty[None, :]  # xy row
        return out

    # General n_cvt: direct SoA varying without full Uab materialization
    return _batch_compute_uab_varying_general_numpy(n_cvt, UtW, Uty, utg_t, out)


def reconstruct_uab_from_soa(
    uab_invariant_soa: np.ndarray,
    uab_varying_soa: np.ndarray,
    n_cvt: int,
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
        n_cvt: Number of covariates.

    Returns:
        Full Uab array (n_snps, n_samples, n_index) matching
        batch_compute_uab_numpy layout.
    """
    n_snps, _, n_samples = uab_varying_soa.shape

    if n_cvt == 1:
        # Fast path: the six-column layout, zero overhead for the common case.
        Uab = np.empty((n_snps, n_samples, 6), dtype=np.float64)

        # Invariant columns — broadcast across all SNPs
        Uab[:, :, _NCVT1.ww] = uab_invariant_soa[0]
        Uab[:, :, _NCVT1.wy] = uab_invariant_soa[1]
        Uab[:, :, _NCVT1.yy] = uab_invariant_soa[2]

        # Varying columns — per-SNP
        Uab[:, :, _NCVT1.wx] = uab_varying_soa[:, 0, :]
        Uab[:, :, _NCVT1.xx] = uab_varying_soa[:, 1, :]
        Uab[:, :, _NCVT1.xy] = uab_varying_soa[:, 2, :]

        return Uab

    # General path for n_cvt > 1: use classify_uab_columns to get index mapping.

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    Uab = np.empty(
        (n_snps, n_samples, build_index_table(n_cvt).n_index), dtype=np.float64
    )

    # Place invariant columns (broadcast across all SNPs)
    for row_i, col_idx in enumerate(inv_indices):
        Uab[:, :, col_idx] = uab_invariant_soa[row_i]

    # Place varying columns (per-SNP from SoA axis-1)
    for row_i, col_idx in enumerate(var_indices):
        Uab[:, :, col_idx] = uab_varying_soa[:, row_i, :]

    return Uab


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
