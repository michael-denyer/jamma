"""Type stubs for the _lmm_accel C extension."""

import numpy as np
import numpy.typing as npt

from jamma.lmm.compute_numpy import WaldResult

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
) -> object: ...
def compute_lmm_chunk_split_c(
    workspace: object,
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
) -> object: ...
def compute_lmm_chunk_general_c(
    workspace: object,
    uab_varying: npt.NDArray[np.float64],
    n_threads: int,
) -> WaldResult: ...
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
) -> object: ...
def compute_mode4_chunk_split_c(
    workspace: object,
    uab_varying: npt.NDArray[np.float64],
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...
