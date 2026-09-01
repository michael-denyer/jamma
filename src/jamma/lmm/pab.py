"""Packed Uab/Pab/Iab representation and table construction.

This module owns GEMMA's packed projection indexing and the C-friendly table
derived from it. Likelihood evaluation consumes these values, but it does not
own their shape.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from jamma.core.constants import n_index

_P_YY_MIN = 1e-8


class PabIndexTable(NamedTuple):
    """Index mappings for one n_cvt: GEMMA's 1-based (a, b) packing made concrete.

    Built once per n_cvt by ``build_index_table`` and walked by every Uab and
    Pab builder, scalar or batch, so the recursion visits the same integers
    in the same order everywhere.
    """

    n_index: int
    """Total (a, b) pairs: (n_cvt+3)*(n_cvt+2)//2."""
    idx_yy: int
    idx_xx: int
    idx_xy: int
    uab_pairs: tuple[tuple[int, int, int], ...]
    """(0-based col_a, 0-based col_b, linear index) for Uab construction."""
    pab_recursion: tuple[tuple[tuple[int, int, int, int, int, int], ...], ...]
    """Per projection level p, the (a, b, index_ab, index_aw, index_bw, index_ww)
    tuples GEMMA's CalcPab visits. Level 0 is empty: row 0 comes from dot
    products."""
    logdet_diag_indices: tuple[tuple[int, int], ...]
    """(row, col) of Pab[i, (i+1, i+1)] for i in 0..n_cvt, the logdet_hiw diagonal."""


class PabCTable(NamedTuple):
    """``build_pab_table_for_c``'s product: the recursion flattened for the C kernels.

    The ``entries`` array is stride-4: each entry is
    [index_ab, index_aw, index_bw, index_ww]. Level 0 has no entries (row 0
    comes from dot products); levels 1..n_cvt+1 have recursion entries.
    ``_asdict()`` is the dict the C table parser reads.
    """

    n_cvt: int
    n_index: int
    n_rows: int
    n_inv: int
    n_var: int
    idx_xx: int
    idx_xy: int
    idx_yy: int
    invariant_indices: np.ndarray
    varying_indices: np.ndarray
    logdet_diag_rows: np.ndarray
    logdet_diag_cols: np.ndarray
    level_offsets: np.ndarray
    level_counts: np.ndarray
    entries: np.ndarray
    var_a_cols: np.ndarray
    var_b_cols: np.ndarray


class _Ncvt1Layout(NamedTuple):
    """Column order of the six-column Uab/Pab for n_cvt=1 (intercept only).

    ``build_index_table(1)`` packs the (a, b) pairs in this order; the n_cvt=1
    fast paths spell the columns by name instead of by literal.
    """

    ww: int
    wx: int
    wy: int
    xx: int
    xy: int
    yy: int


_NCVT1 = _Ncvt1Layout(ww=0, wx=1, wy=2, xx=3, xy=4, yy=5)


def get_ab_index(a: int, b: int, n_cvt: int) -> int:
    """Compute index for accessing Uab/Pab elements using GEMMA's GetabIndex.

    GEMMA uses upper triangular storage with 1-based indices:
    index = (2 * cols - a1 + 2) * (a1 - 1) / 2 + b1 - a1

    where cols = n_cvt + 2, and a1 <= b1 (swapped if necessary).
    """
    cols = n_cvt + 2
    a1, b1 = (a, b) if a <= b else (b, a)
    return (2 * cols - a1 + 2) * (a1 - 1) // 2 + b1 - a1


@functools.lru_cache(maxsize=8)
def build_index_table(n_cvt: int) -> PabIndexTable:
    """Precompute all index mappings for a given n_cvt."""
    idx_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    idx_xx = get_ab_index(n_cvt + 1, n_cvt + 1, n_cvt)
    idx_xy = get_ab_index(n_cvt + 2, n_cvt + 1, n_cvt)

    uab_pairs = []
    for a in range(1, n_cvt + 3):
        for b in range(a, n_cvt + 3):
            idx = get_ab_index(a, b, n_cvt)
            uab_pairs.append((a - 1, b - 1, idx))

    pab_recursion: list[tuple[tuple[int, int, int, int, int, int], ...]] = [()]
    for p in range(1, n_cvt + 2):
        entries = []
        for a in range(p + 1, n_cvt + 3):
            for b in range(a, n_cvt + 3):
                index_ab = get_ab_index(a, b, n_cvt)
                index_aw = get_ab_index(a, p, n_cvt)
                index_bw = get_ab_index(b, p, n_cvt)
                index_ww = get_ab_index(p, p, n_cvt)
                entries.append((a, b, index_ab, index_aw, index_bw, index_ww))
        pab_recursion.append(tuple(entries))

    logdet_diag_indices = []
    for i in range(n_cvt + 1):
        col = get_ab_index(i + 1, i + 1, n_cvt)
        logdet_diag_indices.append((i, col))

    return PabIndexTable(
        n_index=n_index(n_cvt),
        idx_yy=idx_yy,
        idx_xx=idx_xx,
        idx_xy=idx_xy,
        uab_pairs=tuple(uab_pairs),
        pab_recursion=tuple(pab_recursion),
        logdet_diag_indices=tuple(logdet_diag_indices),
    )


def compute_Uab(
    UtW: np.ndarray, Uty: np.ndarray, Utx: np.ndarray | None = None
) -> np.ndarray:
    """Compute Uab matrix following GEMMA's CalcUab exactly."""
    n = len(Uty)
    n_cvt = UtW.shape[1] if UtW.ndim > 1 else 1
    UtW = UtW.reshape(n, -1) if UtW.ndim == 1 else UtW

    if n_cvt == 1:
        return _compute_Uab_ncvt1(UtW, Uty, Utx)
    return _compute_Uab_general(UtW, Uty, Utx, n_cvt)


def _compute_Uab_ncvt1(
    UtW: np.ndarray, Uty: np.ndarray, Utx: np.ndarray | None
) -> np.ndarray:
    """Optimized Uab computation for n_cvt=1 (intercept only)."""
    n = len(Uty)
    w = UtW[:, 0]
    Uab = np.zeros((n, 6), dtype=np.float64)

    Uab[:, _NCVT1.ww] = w * w
    Uab[:, _NCVT1.wy] = w * Uty
    Uab[:, _NCVT1.yy] = Uty * Uty

    if Utx is not None:
        Uab[:, _NCVT1.wx] = w * Utx
        Uab[:, _NCVT1.xx] = Utx * Utx
        Uab[:, _NCVT1.xy] = Utx * Uty

    return Uab


def _compute_Uab_general(
    UtW: np.ndarray, Uty: np.ndarray, Utx: np.ndarray | None, n_cvt: int
) -> np.ndarray:
    """General Uab computation for arbitrary n_cvt."""
    n = len(Uty)
    table = build_index_table(n_cvt)
    Uab = np.zeros((n, table.n_index), dtype=np.float64)

    if Utx is not None:
        vectors = np.column_stack([UtW, Utx, Uty])
    else:
        vectors = np.column_stack([UtW, np.zeros(n), Uty])

    genotype_col = n_cvt
    for a_col, b_col, idx in table.uab_pairs:
        if Utx is None and genotype_col in (a_col, b_col):
            continue
        Uab[:, idx] = vectors[:, a_col] * vectors[:, b_col]

    return Uab


def calc_pab(
    n_cvt: int,
    Hi_eval: np.ndarray,
    Uab: np.ndarray,
) -> np.ndarray:
    """Compute Pab matrix following GEMMA's CalcPab exactly."""
    table = build_index_table(n_cvt)
    Pab = np.zeros((n_cvt + 2, table.n_index), dtype=np.float64)

    Pab[0, :] = Hi_eval @ Uab

    for p in range(1, n_cvt + 2):
        for _a, _b, index_ab, index_aw, index_bw, index_ww in table.pab_recursion[p]:
            ps_ab = Pab[p - 1, index_ab]
            ps_aw = Pab[p - 1, index_aw]
            ps_bw = Pab[p - 1, index_bw]
            ps_ww = Pab[p - 1, index_ww]

            if ps_ww != 0:
                Pab[p, index_ab] = ps_ab - ps_aw * ps_bw / ps_ww
            else:
                Pab[p, index_ab] = ps_ab

    return Pab


def calc_iab(
    n_cvt: int,
    Uab: np.ndarray,
) -> np.ndarray:
    """Compute Iab matrix: ``calc_pab`` with identity weights."""
    n_samples = Uab.shape[0]
    ones = np.ones(n_samples, dtype=np.float64)
    return calc_pab(n_cvt, ones, Uab)


@functools.lru_cache(maxsize=8)
def classify_uab_columns(n_cvt: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Classify Uab columns as invariant or SNP-varying."""
    table = build_index_table(n_cvt)
    genotype_col = n_cvt
    invariant = []
    varying = []
    for a_col, b_col, linear_idx in table.uab_pairs:
        if genotype_col in (a_col, b_col):
            varying.append(linear_idx)
        else:
            invariant.append(linear_idx)
    return tuple(invariant), tuple(varying)


@functools.lru_cache(maxsize=8)
def build_pab_table_for_c(n_cvt: int) -> PabCTable:
    """Build flat C-friendly arrays from build_index_table recursion data."""
    table = build_index_table(n_cvt)
    inv_indices, var_indices = classify_uab_columns(n_cvt)

    level_counts_list = [len(level) for level in table.pab_recursion]
    all_entries = []
    for level_entries in table.pab_recursion:
        for _, _, idx_ab, idx_aw, idx_bw, idx_ww in level_entries:
            all_entries.extend([idx_ab, idx_aw, idx_bw, idx_ww])

    level_offsets_list = []
    running = 0
    for count in level_counts_list:
        level_offsets_list.append(running)
        running += count

    diag_rows = [r for r, _ in table.logdet_diag_indices]
    diag_cols = [c for _, c in table.logdet_diag_indices]

    def _frozen(data: Sequence[int]) -> np.ndarray:
        arr = np.array(data, dtype=np.int32)
        arr.flags.writeable = False
        return arr

    genotype_col = n_cvt
    var_a_list = []
    var_b_list = []
    for a_col, b_col, _linear_idx in table.uab_pairs:
        if genotype_col in (a_col, b_col):
            var_a_list.append(a_col)
            var_b_list.append(b_col)

    return PabCTable(
        n_cvt=n_cvt,
        n_index=table.n_index,
        n_rows=n_cvt + 2,
        n_inv=len(inv_indices),
        n_var=len(var_indices),
        idx_xx=table.idx_xx,
        idx_xy=table.idx_xy,
        idx_yy=table.idx_yy,
        invariant_indices=_frozen(inv_indices),
        varying_indices=_frozen(var_indices),
        logdet_diag_rows=_frozen(diag_rows),
        logdet_diag_cols=_frozen(diag_cols),
        level_offsets=_frozen(level_offsets_list),
        level_counts=_frozen(level_counts_list),
        entries=_frozen(all_entries),
        var_a_cols=_frozen(var_a_list),
        var_b_cols=_frozen(var_b_list),
    )
