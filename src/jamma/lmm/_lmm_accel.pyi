"""Type stubs for the _lmm_accel C extension."""

from typing import NewType

import numpy as np
import numpy.typing as npt

from jamma.lmm.compute_numpy import WaldResult

LmmWorkspace = NewType("LmmWorkspace", object)
LmmWorkspaceGeneral = NewType("LmmWorkspaceGeneral", object)
Mode4Workspace = NewType("Mode4Workspace", object)

ABI_VERSION: int
HAS_OPENMP: int

def compute_lmm_batch_c(
    eigenvalues: npt.NDArray[np.float64],
    Uab_batch: npt.NDArray[np.float64],
    Iab_batch: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> WaldResult: ...
def compute_lmm_batch_split_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_varying: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    Iab_batch: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> WaldResult: ...
def create_workspace_split_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> LmmWorkspace: ...
def compute_lmm_chunk_split_c(
    workspace: LmmWorkspace,
    uab_varying: npt.NDArray[np.float64],
    n_threads: int,
) -> WaldResult: ...
def create_workspace_general_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    n_cvt: int,
    invariant_indices: npt.NDArray[np.int32],
    varying_indices: npt.NDArray[np.int32],
    logdet_diag_rows: npt.NDArray[np.int32],
    logdet_diag_cols: npt.NDArray[np.int32],
    level_offsets: npt.NDArray[np.int32],
    level_counts: npt.NDArray[np.int32],
    entries: npt.NDArray[np.int32],
    idx_xx: int,
    idx_xy: int,
    idx_yy: int,
) -> LmmWorkspaceGeneral: ...
def compute_lmm_chunk_general_c(
    workspace: LmmWorkspaceGeneral,
    uab_varying: npt.NDArray[np.float64],
    n_threads: int,
) -> WaldResult: ...
def compute_score_fused_c(
    utg_t: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    Hi_eval_null: npt.NDArray[np.float64],
    uab_invariant_soa: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.float64],
    n_samples: int,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def compute_lrt_fused_c(
    utg_t: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant_soa: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def compute_score_batch_c(
    eigenvalues: npt.NDArray[np.float64],
    Uab_batch: npt.NDArray[np.float64],
    Hi_eval_null: npt.NDArray[np.float64],
    n_samples: int,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def compute_lrt_batch_c(
    eigenvalues: npt.NDArray[np.float64],
    Uab_batch: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def create_workspace_mode4_split_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    hi_eval_null: npt.NDArray[np.float64],
    logl_H0: float,
) -> Mode4Workspace: ...
def compute_mode4_chunk_split_c(
    workspace: Mode4Workspace,
    uab_varying: npt.NDArray[np.float64],
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...

FusedWorkspace = NewType("FusedWorkspace", object)
FusedMode4Workspace = NewType("FusedMode4Workspace", object)

def create_workspace_fused_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
) -> FusedWorkspace: ...
def compute_lmm_chunk_fused_c(
    workspace: FusedWorkspace,
    utg_t: npt.NDArray[np.float64],
    n_threads: int,
) -> WaldResult: ...
def create_workspace_mode4_fused_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    hi_eval_null: npt.NDArray[np.float64],
    logl_H0: float,
) -> FusedMode4Workspace: ...
def compute_mode4_chunk_fused_c(
    workspace: FusedMode4Workspace,
    utg_t: npt.NDArray[np.float64],
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...

FusedGeneralWorkspace = NewType("FusedGeneralWorkspace", object)
FusedGeneralMode4Workspace = NewType("FusedGeneralMode4Workspace", object)

def create_workspace_fused_general_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    UtW: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    n_cvt: int,
    invariant_indices: npt.NDArray[np.int32],
    varying_indices: npt.NDArray[np.int32],
    logdet_diag_rows: npt.NDArray[np.int32],
    logdet_diag_cols: npt.NDArray[np.int32],
    level_offsets: npt.NDArray[np.int32],
    level_counts: npt.NDArray[np.int32],
    entries: npt.NDArray[np.int32],
    idx_xx: int,
    idx_xy: int,
    idx_yy: int,
    var_a_cols: npt.NDArray[np.int32],
    var_b_cols: npt.NDArray[np.int32],
) -> FusedGeneralWorkspace: ...
def compute_lmm_chunk_fused_general_c(
    workspace: FusedGeneralWorkspace,
    utg_t: npt.NDArray[np.float64],
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def create_workspace_mode4_fused_general_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    UtW: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    n_threads: int,
    n_cvt: int,
    invariant_indices: npt.NDArray[np.int32],
    varying_indices: npt.NDArray[np.int32],
    logdet_diag_rows: npt.NDArray[np.int32],
    logdet_diag_cols: npt.NDArray[np.int32],
    level_offsets: npt.NDArray[np.int32],
    level_counts: npt.NDArray[np.int32],
    entries: npt.NDArray[np.int32],
    idx_xx: int,
    idx_xy: int,
    idx_yy: int,
    var_a_cols: npt.NDArray[np.int32],
    var_b_cols: npt.NDArray[np.int32],
    hi_eval_null: npt.NDArray[np.float64],
    logl_H0: float,
) -> FusedGeneralMode4Workspace: ...
def compute_mode4_chunk_fused_general_c(
    workspace: FusedGeneralMode4Workspace,
    utg_t: npt.NDArray[np.float64],
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def compute_score_split_general_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_varying_soa: npt.NDArray[np.float64],
    uab_invariant_soa: npt.NDArray[np.float64],
    Hi_eval_null: npt.NDArray[np.float64],
    n_samples: int,
    n_cvt: int,
    pab_table_dict: dict,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def compute_lrt_split_general_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_varying_soa: npt.NDArray[np.float64],
    uab_invariant_soa: npt.NDArray[np.float64],
    n_samples: int,
    n_cvt: int,
    pab_table_dict: dict,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...

# Batch Score/LRT for arbitrary n_cvt (table-driven Pab recursion). The
# *_general_c variants take n_cvt + a pab_table_dict from
# build_pab_table_for_c(n_cvt); the non-general batch siblings above are
# the n_cvt=1 fast path.
def compute_score_batch_general_c(
    eigenvalues: npt.NDArray[np.float64],
    Uab_batch: npt.NDArray[np.float64],
    Hi_eval_null: npt.NDArray[np.float64],
    n_samples: int,
    n_cvt: int,
    pab_table_dict: dict,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def compute_lrt_batch_general_c(
    eigenvalues: npt.NDArray[np.float64],
    Uab_batch: npt.NDArray[np.float64],
    n_samples: int,
    n_cvt: int,
    pab_table_dict: dict,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...

# SoA-native Score/LRT for n_cvt=1 — accept split SoA data (varying
# [wx, xx, xy] + invariant [ww, wy, yy]) instead of a full Uab batch.
def compute_score_split_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_varying_soa: npt.NDArray[np.float64],
    uab_invariant_soa: npt.NDArray[np.float64],
    Hi_eval_null: npt.NDArray[np.float64],
    n_samples: int,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def compute_lrt_split_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_varying_soa: npt.NDArray[np.float64],
    uab_invariant_soa: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...

ScoreFusedWorkspace = NewType("ScoreFusedWorkspace", object)
LrtFusedWorkspace = NewType("LrtFusedWorkspace", object)

def create_workspace_score_fused_c(
    w: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    Hi_eval_null: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant_soa: npt.NDArray[np.float64],
    n_samples: int,
    n_threads: int,
) -> ScoreFusedWorkspace: ...
def compute_score_fused_ws_c(
    workspace: ScoreFusedWorkspace,
    utg_t: npt.NDArray[np.float64],
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
def create_workspace_lrt_fused_c(
    w: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant_soa: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
    n_threads: int,
) -> LrtFusedWorkspace: ...
def compute_lrt_fused_ws_c(
    workspace: LrtFusedWorkspace,
    utg_t: npt.NDArray[np.float64],
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...

# Test-only entry points. Not part of the computational API.
def _get_aligned_alloc_test_ptr(n: int) -> int:
    """Return the address of an ``alloc_aligned_doubles(n)`` buffer.

    Used by tests/lmm_accel/test_lmm_accel_split.py to assert 32-byte
    alignment. Always compiled.
    """

def jamma_sentinel_oob() -> int:
    """Deliberately read one byte past a heap allocation.

    Compiled **only** when ``-DJAMMA_SENTINEL_UB`` is set, which just the
    sanitizers workflow does. Absent from every normal build, so callers must
    guard on ``hasattr`` first — tests/test_sanitizer_sentinel.py does.
    """
