"""Non-fixture helpers shared across the _lmm_accel test modules."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np

from jamma.lmm import accel
from jamma.lmm.compute_numpy import _compute_wald_numpy
from jamma.lmm.likelihood_numpy import golden_section_optimize_lambda_mle_numpy
from jamma.lmm.pab import build_index_table
from jamma.lmm.stats import _batch_lrt_pvalues_numpy, batch_calc_score_stats_numpy
from jamma.lmm.uab import batch_compute_uab_numpy
from tests.builders import LmmInputs


@functools.lru_cache(maxsize=8)
def classify_uab_columns(n_cvt: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Classify Uab columns as invariant or SNP-varying.

    Args:
        n_cvt: Number of covariates.

    Returns:
        ``(invariant, varying)``, each a tuple of linear Uab column indices.
    """
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


def assert_fused_matches_reference(
    run: Callable[[], Any],
    *,
    fields: dict[str, float],
    kernel: str | None = "compute_lmm_chunk_c",
    min_count: float | None = None,
    atol: float = 0.0,
    label: str = "",
) -> None:
    """Run a fused-kernel dispatch once live, once with the C extension off.

    ``run`` is a zero-argument callable returning a result with an
    ``.associations`` sequence, each entry exposing ``.rs`` plus the field
    names in ``fields``, whose values are its per-field ``rtol``. When
    ``kernel`` is given, the live run is spied on ``jamma.lmm.accel._accel``
    to assert dispatch actually reached that C entry point rather than
    silently falling through to the reference path; pass ``kernel=None`` to
    skip that assertion when the comparison has no single spy target (the
    n_cvt=2 general case compares fused output against the NumPy path
    directly, since dropping the extension leaves no second C path to spy on).
    A field is compared only where both sides report a value, since a
    statistic can be ``None`` on one side for a mode that skips it.
    """
    if kernel is not None:
        with patch(
            f"jamma.lmm.accel._accel.{kernel}",
            wraps=getattr(accel.require(), kernel),
        ) as mock_fused:
            fused_result = run()
        assert mock_fused.called, f"Fused{label} C function was not called"
    else:
        fused_result = run()

    with patch("jamma.lmm.accel._accel", None):
        reference_result = run()

    fused = fused_result.associations
    reference = reference_result.associations
    assert len(fused) == len(reference), (
        f"Count mismatch: {len(fused)} vs {len(reference)}"
    )
    if min_count is not None:
        assert len(fused) > min_count, f"Too many SNPs filtered: {len(fused)}"

    for a_f, a_r in zip(fused, reference, strict=True):
        assert a_f.rs == a_r.rs, f"SNP order mismatch: {a_f.rs} vs {a_r.rs}"
        for field, rtol in fields.items():
            v_f, v_r = getattr(a_f, field), getattr(a_r, field)
            if v_f is not None and v_r is not None:
                np.testing.assert_allclose(
                    v_f,
                    v_r,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"{field} mismatch for {a_f.rs}{label}",
                )


@dataclass(frozen=True)
class GeneralCase:
    """Synthetic inputs plus the derived arrays the general (n_cvt >= 2) kernels read.

    Each derived array is computed on first read and cached.
    """

    inputs: LmmInputs

    @property
    def n_cvt(self) -> int:
        return self.inputs.n_cvt

    @property
    def n_samples(self) -> int:
        return self.inputs.n_samples

    @functools.cached_property
    def uab_batch(self) -> np.ndarray:
        return self.inputs.uab_batch()

    @functools.cached_property
    def uab_inv_soa(self) -> np.ndarray:
        inv_indices, _ = classify_uab_columns(self.n_cvt)
        return np.ascontiguousarray(self.uab_batch[0, :, list(inv_indices)])

    @functools.cached_property
    def uab_var_soa(self) -> np.ndarray:
        _, var_indices = classify_uab_columns(self.n_cvt)
        return np.ascontiguousarray(
            self.uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
        )

    @functools.cached_property
    def utg_t(self) -> np.ndarray:
        return np.ascontiguousarray(self.inputs.UtG.T)

    @functools.cached_property
    def _null_model(self) -> tuple[np.ndarray, float]:
        """Fit the null model: the same Uab with every genotype column zeroed."""
        eigenvalues = self.inputs.eigenvalues
        inv_indices, _ = classify_uab_columns(self.n_cvt)
        Uab_null = np.zeros((1, *self.uab_batch.shape[1:]), dtype=np.float64)
        for idx in inv_indices:
            Uab_null[0, :, idx] = self.uab_batch[0, :, idx]
        lambdas_null, logls_null = golden_section_optimize_lambda_mle_numpy(
            self.n_cvt,
            eigenvalues,
            Uab_null,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_iter=20,
        )
        lambda_null = float(lambdas_null[0])
        return 1.0 / (lambda_null * eigenvalues + 1.0), float(logls_null[0])

    @property
    def Hi_eval_null(self) -> np.ndarray:
        return self._null_model[0]

    @property
    def logl_H0(self) -> float:
        return self._null_model[1]


def _fused_general_workspace(case: GeneralCase, n_threads: int = 1) -> object:
    """Build the live fused-general Wald workspace for *case*.

    This is what ``DispatchPath.FUSED`` reaches for n_cvt>=2 in mode 1.
    """
    return accel.require().create_workspace_c(
        case.inputs.eigenvalues,
        case.uab_inv_soa,
        case.inputs.UtW,
        case.inputs.Uty,
        case.n_samples,
        1e-5,
        1e5,
        50,
        20,
        n_threads,
        case.n_cvt,
        lmm_mode=1,
    )


def _fused_general_mode4_workspace(case: GeneralCase, n_threads: int = 1) -> object:
    """Build the live fused-general mode-4 workspace for *case*."""
    return accel.require().create_workspace_c(
        case.inputs.eigenvalues,
        case.uab_inv_soa,
        case.inputs.UtW,
        case.inputs.Uty,
        case.n_samples,
        1e-5,
        1e5,
        50,
        20,
        n_threads,
        case.n_cvt,
        lmm_mode=4,
        hi_eval_null=case.Hi_eval_null,
        logl_H0=case.logl_H0,
    )


def _numpy_general_score(case: GeneralCase) -> dict:
    """NumPy Score statistics for a general n_cvt case."""
    betas, ses, p_scores = batch_calc_score_stats_numpy(
        case.n_cvt, case.Hi_eval_null, case.uab_batch, case.n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _numpy_general_lrt(case: GeneralCase) -> dict:
    """NumPy mode-4 MLE likelihoods, lambdas, and LRT p-values."""
    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle_numpy(
        case.n_cvt,
        case.inputs.eigenvalues,
        case.uab_batch,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    return {
        "logls": logls_mle,
        "lambdas_mle": lambdas_mle,
        "p_lrts": _batch_lrt_pvalues_numpy(logls_mle, case.logl_H0),
    }


def _fused_general_wald(case: GeneralCase, n_threads: int = 1) -> dict[str, np.ndarray]:
    """Run the live fused-general Wald kernel over *case*."""
    ws = _fused_general_workspace(case, n_threads)
    return accel.require().compute_lmm_chunk_c(ws, case.utg_t, n_threads)


def _numpy_general_wald(case: GeneralCase) -> dict[str, np.ndarray]:
    """Run the NumPy Wald path over *case*, with the extension held out.

    ``_compute_wald_numpy`` consults ``accel._accel`` at call time and
    takes a C branch when it is set, so the attribute has to be cleared rather
    than the argument changed.
    """
    orig = accel._accel
    try:
        accel._accel = None
        return _compute_wald_numpy(
            case.n_cvt,
            case.inputs.eigenvalues,
            case.uab_batch,
            case.n_samples,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_refine=20,
        )
    finally:
        accel._accel = orig


def _run_general_ncvt_c_vs_python(case: GeneralCase) -> None:
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
    n_cvt = case.n_cvt
    result_c = _fused_general_wald(case)
    result_py = _numpy_general_wald(case)

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
    return batch_compute_uab_numpy(1, w[:, None], Uty, utg_t)


def _numpy_ncvt1_wald(eigenvalues, w, Uty, utg_t, n_samples) -> dict[str, np.ndarray]:
    """NumPy REML Wald for the fused kernel's n_cvt=1 inputs."""
    orig = accel._accel
    try:
        accel._accel = None
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
        accel._accel = orig


def _fused_inputs_from_uab_ncvt1(Uab_batch):
    """Recover (w, Uty, utg_t) from an n_cvt=1 Uab batch.

    The fused kernels build Uab from the rotated vectors themselves, so a
    fixture that hands over a prebuilt Uab has to be inverted. Column layout is
    0=ww, 1=wx, 2=wy, 3=xx, 4=xy, 5=yy, and this package's fixtures build every
    column from a positive w, so the recovery is exact.
    """
    w = np.sqrt(Uab_batch[0, :, 0])
    return w, Uab_batch[0, :, 2] / w, np.ascontiguousarray(Uab_batch[:, :, 1] / w)


def _null_model_ncvt1(eigenvalues, w, Uty):
    """Fit the n_cvt=1 null model, returning (Hi_eval_null, logl_H0).

    The null model is the same Uab with the genotype columns zeroed. An LRT
    p-value is only interpretable against the real logl_H0, so any test that
    asserts a p_lrt value rather than comparing two implementations needs this
    rather than a stand-in constant.
    """
    n_samples = eigenvalues.shape[0]
    Uab_null = np.zeros((1, n_samples, 6), dtype=np.float64)
    Uab_null[0, :, 0] = w * w
    Uab_null[0, :, 2] = w * Uty
    Uab_null[0, :, 5] = Uty * Uty

    lambdas_null, logls_null = golden_section_optimize_lambda_mle_numpy(
        1, eigenvalues, Uab_null, l_min=1e-5, l_max=1e5, n_grid=50, n_iter=20
    )
    lambda_null = float(lambdas_null[0])
    return 1.0 / (lambda_null * eigenvalues + 1.0), float(logls_null[0])


def _numpy_ncvt1_score(w, Uty, utg_t, Hi_eval_null, n_samples) -> dict:
    """NumPy Score statistics for the fused kernel's n_cvt=1 inputs."""
    betas, ses, p_scores = batch_calc_score_stats_numpy(
        1, Hi_eval_null, _uab_from_fused_inputs(w, Uty, utg_t), n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _numpy_ncvt1_lrt(eigenvalues, w, Uty, utg_t, logl_H0, n_refine=20) -> dict:
    """NumPy mode-4 MLE likelihoods, lambdas, and LRT p-values for n_cvt=1."""
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
        "logls": logls_mle,
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
