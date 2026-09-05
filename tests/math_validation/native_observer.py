"""Read exported ncvt1 numerical kernels from the loaded binary, without a fork.

Only the diagnostic driver uses ctypes. Ordinary parity tests use the Python C
entry point. A binary without exported observer symbols is an explicit error.
"""

import ctypes as ct

import numpy as np

from jamma.lmm import accel

_POINTER = ct.POINTER(ct.c_double)


def native_wald(eigenvalues, w, x, y, *, n_refine=20):
    """Actual fused C entry point, one phenotype/covariate vector per workspace."""
    w, y = np.ascontiguousarray(w), np.ascontiguousarray(y)
    inv = np.ascontiguousarray([w * w, w * y, y * y])
    ws = accel.require().create_workspace_ncvt1_c(
        np.ascontiguousarray(eigenvalues),
        inv,
        w,
        y,
        len(w),
        1e-5,
        1e5,
        50,
        n_refine,
        lmm_mode=1,
    )
    return accel.require().compute_lmm_chunk_ncvt1_c(ws, np.ascontiguousarray(x), 1)


def native_objectives(eigenvalues, uab, lambdas):
    """Observe all three Pab levels from the binary's actual REML evaluator."""
    library = ct.CDLL(accel.require().__file__)
    function = library.reml_logl_ncvt1_split
    function.argtypes = [_POINTER] * 7 + [
        ct.c_double,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        _POINTER,
    ]
    function.restype = ct.c_double
    calc = library.calc_pab_ncvt1_split
    calc.argtypes = [ct.c_double] * 6 + [_POINTER]
    calc.restype = None
    columns = [np.ascontiguousarray(uab[:, i]) for i in (1, 3, 4, 0, 2, 5)]
    columns.append(np.ascontiguousarray(eigenvalues))
    pointers = [a.ctypes.data_as(_POINTER) for a in columns]
    iab = np.zeros((3, 6))
    calc(*uab.sum(axis=0), iab.ctypes.data_as(_POINTER))
    logdet_iab = sum(float(np.log(v)) if v > 0 else 0.0 for v in (iab[0, 0], iab[1, 3]))
    df = len(eigenvalues) - 2
    constant = float(0.5 * df * (np.log(df) - np.log(2 * np.pi) - 1))
    objectives, levels = [], []
    for lam in lambdas:
        pab = np.zeros((3, 6))
        objectives.append(
            function(
                *pointers,
                logdet_iab,
                len(eigenvalues),
                float(lam),
                constant,
                pab.ctypes.data_as(_POINTER),
            )
        )
        levels.append(pab)
    return np.array(objectives), np.array(levels), iab


def native_grid_bracket(eigenvalues, uab):
    """Run the actual cached grid builder and selector, observing the chosen cell."""

    class GridInvariant(ct.Structure):
        _fields_ = [(name, ct.c_double) for name in ("ww", "wy", "yy", "log_ww")]

    library = ct.CDLL(accel.require().__file__)
    build = library.build_grid_ncvt1
    build.argtypes = (
        [ct.c_int, ct.c_int, ct.c_double, ct.c_double]
        + [_POINTER] * 7
        + [ct.POINTER(GridInvariant)]
    )
    build.restype = None
    coarse = library.coarse_grid_mode4_ncvt1_split
    coarse.argtypes = [_POINTER] * 3 + [
        ct.c_int,
        _POINTER,
        _POINTER,
        ct.POINTER(GridInvariant),
        ct.c_int,
        ct.c_double,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_int),
    ]
    coarse.restype = None
    ev = np.ascontiguousarray(eigenvalues)
    cols = [np.ascontiguousarray(uab[:, i]) for i in (0, 2, 5, 1, 3, 4)]
    pointers = [a.ctypes.data_as(_POINTER) for a in cols]
    grid = np.zeros(50)
    hi = np.zeros((50, len(ev)))
    logdet = np.zeros(50)
    inv = (GridInvariant * 50)()
    low, step = float(np.log(1e-5)), float(np.log(1e10) / 49)
    build(
        50,
        len(ev),
        low,
        step,
        ev.ctypes.data_as(_POINTER),
        *pointers[:3],
        grid.ctypes.data_as(_POINTER),
        hi.ctypes.data_as(_POINTER),
        logdet.ctypes.data_as(_POINTER),
        inv,
    )
    _, _, iab = native_objectives(ev, uab, [])
    logdet_iab = float(np.log(iab[0, 0]) + np.log(iab[1, 3]))
    df = len(ev) - 2
    constant = float(0.5 * df * (np.log(df) - np.log(2 * np.pi) - 1))
    best_reml, best_mle = ct.c_int(), ct.c_int()
    coarse(
        *pointers[3:],
        len(ev),
        hi.ctypes.data_as(_POINTER),
        logdet.ctypes.data_as(_POINTER),
        inv,
        50,
        logdet_iab,
        df,
        constant,
        0.0,
        ct.byref(best_reml),
        ct.byref(best_mle),
    )
    return best_reml.value, grid
