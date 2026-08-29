"""Type stubs for the _lmm_accel C extension."""

from typing import NewType

import numpy as np
import numpy.typing as npt

ABI_VERSION: int
HAS_OPENMP: int

# One n_cvt=1 workspace type. The creator's lmm_mode picks which loop
# compute_lmm_chunk_ncvt1_c runs: 1 Wald, 2 LRT, 3 Score, 4 all three. Under
# lmm_mode 4 the returned dict carries three extra keys.
NcvtOneWorkspace = NewType("NcvtOneWorkspace", object)

def create_workspace_ncvt1_c(
    eigenvalues: npt.NDArray[np.float64],
    uab_invariant: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    Uty: npt.NDArray[np.float64],
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    *,
    lmm_mode: int,
    hi_eval_null: npt.NDArray[np.float64] | None = None,
    logl_H0: float | None = None,
) -> NcvtOneWorkspace: ...
def compute_lmm_chunk_ncvt1_c(
    workspace: NcvtOneWorkspace,
    utg_t: npt.NDArray[np.float64],
    n_threads: int,
) -> dict[str, npt.NDArray[np.float64]]: ...

# One general (n_cvt >= 2) workspace type, for any lmm_mode. pab_table is
# the dict PabCTable._asdict() returns.
GeneralWorkspace = NewType("GeneralWorkspace", object)

def create_workspace_general_c(
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
    pab_table: dict,
    *,
    lmm_mode: int,
    hi_eval_null: npt.NDArray[np.float64] | None = None,
    logl_H0: float | None = None,
) -> GeneralWorkspace: ...
def compute_lmm_chunk_fused_general_c(
    workspace: GeneralWorkspace,
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
