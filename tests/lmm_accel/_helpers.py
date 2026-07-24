"""Non-fixture helpers shared across the _lmm_accel test modules."""

import numpy as np

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _compute_wald_numpy,
)
from jamma.lmm.likelihood_numpy import (
    golden_section_optimize_lambda_mle_numpy,
)


def _run_general_ncvt_c_vs_python(data: dict) -> None:
    """Helper: compare C extension general n_cvt results against Python path.

    Monkeypatches _C_GENERAL_AVAILABLE to False for the Python reference,
    then compares against C extension results.
    """
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]

    # Python reference path (force fallback)
    orig = compute_numpy._C_GENERAL_AVAILABLE
    try:
        compute_numpy._C_GENERAL_AVAILABLE = False
        result_py = _compute_wald_numpy(
            n_cvt,
            eigenvalues,
            Uab_batch,
            n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
        )
    finally:
        compute_numpy._C_GENERAL_AVAILABLE = orig

    # C extension path
    result_c = _compute_wald_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        n_threads=1,
    )

    for key in ("lambdas", "logls", "betas", "ses"):
        np.testing.assert_allclose(
            result_c[key],
            result_py[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: C vs Python mismatch for n_cvt={n_cvt}",
        )
    np.testing.assert_allclose(
        result_c["pwalds"],
        result_py["pwalds"],
        rtol=1e-6,
        atol=1e-14,
        equal_nan=True,
        err_msg=f"pwalds: C vs Python mismatch for n_cvt={n_cvt}",
    )


def _make_general_score_lrt_data(data: dict) -> dict:
    """Extend synthetic covariate data with null-model Hi_eval and logl_H0.

    Computes the null-model MLE lambda via golden section on the null Uab
    (zero genotype), then derives Hi_eval_null and logl_H0.

    Args:
        data: Dict from _build_synthetic_covariate_data.

    Returns:
        Dict with all original keys plus Hi_eval_null and logl_H0.
    """
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_cvt = data["n_cvt"]
    n_samples = data["n_samples"]
    n_index = Uab_batch.shape[2]  # (n_cvt+3)*(n_cvt+2)//2

    # Null Uab: zero all varying (genotype) columns.
    from jamma.lmm.likelihood import classify_uab_columns

    inv_indices, _ = classify_uab_columns(n_cvt)
    Uab_null = np.zeros((1, n_samples, n_index), dtype=np.float64)
    for idx in inv_indices:
        Uab_null[0, :, idx] = Uab_batch[0, :, idx]

    lambdas_null, logls_null = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_null,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    lambda_null = float(lambdas_null[0])
    logl_H0 = float(logls_null[0])
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)

    return {**data, "Hi_eval_null": Hi_eval_null, "logl_H0": logl_H0}


def _score_general_c_available() -> bool:
    """Check if compute_score_batch_general_c is available from the C extension."""
    if not _C_ACCEL_AVAILABLE:
        return False
    try:
        from jamma.lmm._lmm_accel import compute_score_batch_general_c  # noqa: F401

        return True
    except ImportError:
        return False
