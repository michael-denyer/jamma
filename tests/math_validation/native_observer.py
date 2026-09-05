"""Call the supported native Wald entry point."""

import numpy as np

from jamma.lmm import accel


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
