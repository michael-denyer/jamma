"""Non-fixture helpers shared across the _lmm_accel test modules."""

import numpy as np

import jamma.lmm.compute_numpy as compute_numpy
from jamma.lmm.compute_numpy import (
    WaldResult,
    _compute_wald_numpy,
    compute_wald_fused_general_c_ws,
    create_lmm_workspace_fused_general,
    create_lmm_workspace_mode4_fused_general,
)
from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns
from jamma.lmm.likelihood_numpy import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    batch_compute_uab_numpy,
    golden_section_optimize_lambda_mle_numpy,
)

_PAB_KWARG_NAMES = (
    "invariant_indices",
    "varying_indices",
    "logdet_diag_rows",
    "logdet_diag_cols",
    "level_offsets",
    "level_counts",
    "entries",
    "idx_xx",
    "idx_xy",
    "idx_yy",
    "var_a_cols",
    "var_b_cols",
)


def _prepare_fused_general_data(data: dict) -> dict:
    """Add the invariant SoA, varying SoA, UtG_T and Pab table the kernels need.

    Args:
        data: Dict from _build_synthetic_covariate_data.

    Returns:
        Dict with uab_inv_soa, uab_var_soa, utg_t, pab_c, and the original keys.
    """
    n_cvt = data["n_cvt"]
    Uab_batch = data["Uab_batch"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    return {
        **data,
        "uab_inv_soa": np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)]),
        "uab_var_soa": np.ascontiguousarray(
            Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
        ),
        "utg_t": np.ascontiguousarray(data["UtG"].T),
        "pab_c": build_pab_table_for_c(n_cvt),
    }


def _fused_general_workspace(data: dict, n_threads: int = 1) -> object:
    """Build the live fused-general Wald workspace for *data*.

    This is what ``DispatchPath.FUSED_GENERAL`` reaches for n_cvt>=2 in mode 1.
    Accepts either a raw _build_synthetic_covariate_data dict or one already
    through _prepare_fused_general_data.
    """
    if "pab_c" not in data:
        data = _prepare_fused_general_data(data)
    return create_lmm_workspace_fused_general(
        data["eigenvalues"],
        data["uab_inv_soa"],
        data["UtW"],
        data["Uty"],
        data["n_samples"],
        1e-5,
        1e5,
        50,
        20,
        n_threads,
        n_cvt=data["n_cvt"],
        **{k: data["pab_c"][k] for k in _PAB_KWARG_NAMES},
    )


def _fused_general_mode4_workspace(data: dict, n_threads: int = 1) -> object:
    """Build the live fused-general mode-4 workspace for *data*.

    *data* must carry Hi_eval_null and logl_H0, so it has to have been through
    _make_general_score_lrt_data.
    """
    if "pab_c" not in data:
        data = _prepare_fused_general_data(data)
    return create_lmm_workspace_mode4_fused_general(
        data["eigenvalues"],
        data["uab_inv_soa"],
        data["UtW"],
        data["Uty"],
        data["n_samples"],
        1e-5,
        1e5,
        50,
        20,
        n_threads,
        n_cvt=data["n_cvt"],
        **{k: data["pab_c"][k] for k in _PAB_KWARG_NAMES},
        hi_eval_null=data["Hi_eval_null"],
        logl_H0=data["logl_H0"],
    )


def _numpy_general_score(data: dict) -> dict:
    """NumPy Score statistics for a general n_cvt fixture."""
    betas, ses, p_scores = batch_calc_score_stats_numpy(
        data["n_cvt"], data["Hi_eval_null"], data["Uab_batch"], data["n_samples"]
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _numpy_general_lrt(data: dict) -> dict:
    """NumPy MLE lambdas and LRT p-values for a general n_cvt fixture."""
    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle_numpy(
        data["n_cvt"],
        data["eigenvalues"],
        data["Uab_batch"],
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    return {
        "lambdas_mle": lambdas_mle,
        "p_lrts": _batch_lrt_pvalues_numpy(logls_mle, data["logl_H0"]),
    }


def _fused_general_wald(data: dict, n_threads: int = 1) -> WaldResult:
    """Run the live fused-general Wald kernel over *data*."""
    if "pab_c" not in data:
        data = _prepare_fused_general_data(data)
    ws = _fused_general_workspace(data, n_threads)
    return compute_wald_fused_general_c_ws(ws, data["utg_t"], n_threads)


def _numpy_general_wald(data: dict) -> WaldResult:
    """Run the NumPy Wald path over *data*, with the extension held out.

    ``_compute_wald_numpy`` consults ``compute_numpy._accel`` at call time and
    takes a C branch when it is set, so the attribute has to be cleared rather
    than the argument changed.
    """
    orig = compute_numpy._accel
    try:
        compute_numpy._accel = None
        return _compute_wald_numpy(
            data["n_cvt"],
            data["eigenvalues"],
            data["Uab_batch"],
            data["n_samples"],
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
        )
    finally:
        compute_numpy._accel = orig


def _run_general_ncvt_c_vs_python(data: dict) -> None:
    """Compare the fused-general C Wald kernel against the NumPy Wald path.

    The C side used to be ``_compute_wald_numpy`` with the extension loaded,
    which took an inner C ladder that no dispatch path reaches: the only
    production caller of that function runs when ``_accel`` is None. Comparing
    it against the same function with ``_accel`` cleared would have gone
    NumPy-versus-NumPy, and still passed, once the ladder was removed.

    Tolerances are the ones this comparison already used. The fused-general
    kernel is bitwise identical to the non-fused general kernel it replaces as
    the subject here, so the deviation from NumPy is unchanged.
    """
    n_cvt = data["n_cvt"]
    result_c = _fused_general_wald(data)
    result_py = _numpy_general_wald(data)

    for key in ("lambdas", "logls", "betas", "ses"):
        np.testing.assert_allclose(
            result_c[key],
            result_py[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: C vs NumPy mismatch for n_cvt={n_cvt}",
        )
    np.testing.assert_allclose(
        result_c["pwalds"],
        result_py["pwalds"],
        rtol=1e-6,
        atol=1e-14,
        equal_nan=True,
        err_msg=f"pwalds: C vs NumPy mismatch for n_cvt={n_cvt}",
    )


# Deviation of every C kernel here from its NumPy counterpart, measured on the
# fixtures in this package, peaks at 1.1e-13. 1e-10 leaves three orders of
# headroom for a different compiler and CPU in CI.
C_VS_NUMPY_RTOL = 1e-10
# Except the MLE lambda. It is an argmin on a surface that is flat for
# weak-signal SNPs, so the two golden-section implementations land 2.4e-5 to
# 3.8e-5 apart while the p-value they feed still agrees to 1e-12. This is the
# band CLAUDE.md records as lambda_rtol.
LAMBDA_MLE_RTOL = 5e-5


def _uab_from_fused_inputs(w, Uty, utg_t):
    """Rebuild the full Uab batch the NumPy kernels take from the fused SoA inputs."""
    return batch_compute_uab_numpy(1, w[:, None], Uty, utg_t.T)


def _numpy_ncvt1_wald(eigenvalues, w, Uty, utg_t, n_samples) -> WaldResult:
    """NumPy REML Wald for the fused kernel's n_cvt=1 inputs."""
    orig = compute_numpy._accel
    try:
        compute_numpy._accel = None
        return _compute_wald_numpy(
            1,
            eigenvalues,
            _uab_from_fused_inputs(w, Uty, utg_t),
            n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
        )
    finally:
        compute_numpy._accel = orig


def _numpy_ncvt1_score(w, Uty, utg_t, Hi_eval_null, n_samples) -> dict:
    """NumPy Score statistics for the fused kernel's n_cvt=1 inputs."""
    betas, ses, p_scores = batch_calc_score_stats_numpy(
        1, Hi_eval_null, _uab_from_fused_inputs(w, Uty, utg_t), n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _numpy_ncvt1_lrt(eigenvalues, w, Uty, utg_t, logl_H0, n_refine=20) -> dict:
    """NumPy MLE lambdas and LRT p-values for the fused kernel's n_cvt=1 inputs."""
    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle_numpy(
        1,
        eigenvalues,
        _uab_from_fused_inputs(w, Uty, utg_t),
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=n_refine,
    )
    return {
        "lambdas_mle": lambdas_mle,
        "p_lrts": _batch_lrt_pvalues_numpy(logls_mle, logl_H0),
    }


def assert_matches_numpy(result, reference, label) -> None:
    """Assert every key of *reference* matches *result* at the measured tolerance."""
    for key, ref in reference.items():
        np.testing.assert_allclose(
            result[key],
            ref,
            rtol=LAMBDA_MLE_RTOL if key == "lambdas_mle" else C_VS_NUMPY_RTOL,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{label} {key} does not match the NumPy reference",
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
