"""_lmm_accel C extension tests: fused Uab kernels, n_cvt=1 and general.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from jamma.lmm import accel
from jamma.lmm.schema import LmmConfig
from jamma.lmm.uab import compute_uab_invariant_soa
from tests.builders import rotated_lmm_inputs
from tests.conftest import requires_c
from tests.lmm_accel._helpers import (
    _fused_general_mode4_workspace,
    _fused_general_workspace,
    _numpy_general_lrt,
    _numpy_general_score,
    _numpy_general_wald,
    _numpy_ncvt1_lrt,
    _numpy_ncvt1_score,
    _numpy_ncvt1_wald,
    _prepare_fused_general_data,
    assert_fused_matches_reference,
    assert_matches_numpy,
)

_WALD_KEYS = ("lambdas", "logls", "betas", "ses", "pwalds")


def _ncvt1_workspace(fused_data, **kwargs):
    """Build an n_cvt=1 workspace from the fixture, with the grid defaults.

    Every mode shares one creator, so the mode and its extra inputs are the
    only thing a caller varies.
    """
    eigenvalues, w, Uty, _, uab_inv_soa, _, n_samples = fused_data
    return accel.require().create_workspace_ncvt1_c(
        eigenvalues, uab_inv_soa, w, Uty, n_samples, 1e-5, 1e5, 50, 20, **kwargs
    )


@pytest.mark.tier0
@requires_c
class TestHiEvalNullPositivity:
    """C extension rejects non-positive hi_eval_null at every site that takes it.

    hi_eval_null is 1/(lambda_null*eval + 1), so a zero or negative entry means
    the null model is broken upstream. The kernels divide by it, so the check has
    to be at the boundary rather than left to produce infinities downstream.

    The two n_cvt=1 sites are one creator called in mode 4 and in mode 3, plus
    the general split kernel. Earlier unreachable entry points have gone; the
    checks they covered live in the same C validation helper.
    """

    @pytest.mark.parametrize("bad", [0.0, -0.5], ids=["zero", "negative"])
    def test_mode4_fused_workspace_rejects(self, fused_data, score_lrt_data, bad):
        """create_workspace_ncvt1_c rejects a non-positive hi_eval_null in mode 4."""
        _, _, _, Hi_eval_null, logl_H0 = score_lrt_data

        hi_bad = Hi_eval_null.copy()
        hi_bad[5] = bad

        with pytest.raises(ValueError, match="positive"):
            _ncvt1_workspace(
                fused_data, lmm_mode=4, hi_eval_null=hi_bad, logl_H0=logl_H0
            )

    @pytest.mark.parametrize("bad", [0.0, -1.0], ids=["zero", "negative"])
    def test_score_fused_workspace_rejects(self, fused_data, score_lrt_data, bad):
        """create_workspace_ncvt1_c rejects a non-positive hi_eval_null in mode 3."""
        _, _, _, Hi_eval_null, _ = score_lrt_data

        hi_bad = Hi_eval_null.copy()
        hi_bad[3] = bad

        with pytest.raises(ValueError, match="positive"):
            _ncvt1_workspace(fused_data, lmm_mode=3, hi_eval_null=hi_bad)

    @pytest.mark.parametrize("bad", [0.0, -2.0], ids=["zero", "negative"])
    def test_general_score_workspace_rejects(self, synthetic_covariate_data_ncvt2, bad):
        """create_workspace_general_c rejects a non-positive hi_eval_null in mode 3."""
        from jamma.lmm._lmm_accel import create_workspace_general_c

        data = _prepare_fused_general_data(synthetic_covariate_data_ncvt2)
        eigenvalues = data["eigenvalues"]

        hi_bad = 1.0 / (0.5 * eigenvalues + 1.0)
        hi_bad[0] = bad

        with pytest.raises(ValueError, match="positive"):
            create_workspace_general_c(
                eigenvalues,
                data["uab_inv_soa"],
                data["UtW"],
                data["Uty"],
                data["n_samples"],
                1e-5,
                1e5,
                50,
                20,
                1,
                data["pab_c"]._asdict(),
                lmm_mode=3,
                hi_eval_null=hi_bad,
            )


_fused_c_available = accel.available()


@pytest.mark.tier0
@requires_c
@pytest.mark.parametrize(
    ("lmm_mode", "extra"),
    [
        (1, {"hi_eval_null": True}),
        (1, {"logl_H0": True}),
        (2, {}),
        (3, {}),
    ],
    ids=[
        "wald_given_hi_eval_null",
        "wald_given_logl_H0",
        "lrt_no_logl_H0",
        "score_no_hi_eval_null",
    ],
)
def test_ncvt1_creator_rejects_wrong_inputs_for_mode(fused_data, lmm_mode, extra):
    """The one creator takes exactly the inputs its lmm_mode uses.

    Mode 1 takes neither hi_eval_null nor logl_H0, mode 2 requires logl_H0 and
    mode 3 requires hi_eval_null. Anything else is a caller confusing two modes,
    which the single capsule type can no longer catch on its own.
    """
    eigenvalues = fused_data[0]
    kwargs = {}
    if extra.get("hi_eval_null"):
        kwargs["hi_eval_null"] = 1.0 / (0.5 * eigenvalues + 1.0)
    if extra.get("logl_H0"):
        kwargs["logl_H0"] = -100.0

    with pytest.raises(ValueError, match="lmm_mode="):
        _ncvt1_workspace(fused_data, lmm_mode=lmm_mode, **kwargs)


@pytest.mark.tier0
@pytest.mark.skipif(not _fused_c_available, reason="Fused C extension not available")
class TestFusedParity:
    """Verify the fused Uab path against the NumPy implementations."""

    def test_fused_workspace_creation(self, fused_data):
        """create_workspace_ncvt1_c returns a PyCapsule."""
        assert _ncvt1_workspace(fused_data, lmm_mode=1) is not None

    def test_wald_parity(self, fused_data):
        """Fused Wald matches the NumPy REML Wald path.

        The reference was the SoA workspace kernel, which let this be bitwise,
        but no dispatch path reaches that kernel. NumPy is an independent
        implementation, so the assertion carries the measured tolerance.
        """
        from jamma.lmm import accel

        eigenvalues, w, Uty, utg_t, uab_inv_soa, uab_var_soa, n_samples = fused_data

        ws_fused = _ncvt1_workspace(fused_data, lmm_mode=1)
        result = accel.require().compute_lmm_chunk_ncvt1_c(ws_fused, utg_t, 1)
        reference = _numpy_ncvt1_wald(eigenvalues, w, Uty, utg_t, n_samples)

        assert_matches_numpy(
            result, {k: reference[k] for k in _WALD_KEYS}, "Fused Wald"
        )

    def test_wald_parity_multithreaded(self, fused_data):
        """Fused Wald is bitwise deterministic across thread counts.

        The OpenMP parallel-for partitions SNPs across threads, so a race or
        thread-local state corruption shows up as a difference here. This stays
        a bitwise comparison because both sides are the same kernel.
        """
        from jamma.lmm import accel

        _, _, _, utg_t, _, _, _ = fused_data

        ws_fused = _ncvt1_workspace(fused_data, lmm_mode=1)
        single = accel.require().compute_lmm_chunk_ncvt1_c(ws_fused, utg_t, 1)
        multi = accel.require().compute_lmm_chunk_ncvt1_c(ws_fused, utg_t, 4)

        for key in _WALD_KEYS:
            np.testing.assert_array_equal(
                single[key],
                multi[key],
                err_msg=f"Wald {key}: fused 4-thread vs 1-thread mismatch",
            )

    def test_mode4_fused_workspace_creation(self, fused_data, score_lrt_data):
        """create_workspace_ncvt1_c returns a PyCapsule in mode 4."""
        _, _, _, Hi_eval_null, logl_H0 = score_lrt_data

        ws = _ncvt1_workspace(
            fused_data, lmm_mode=4, hi_eval_null=Hi_eval_null, logl_H0=logl_H0
        )
        assert ws is not None

    def test_mode4_parity(self, fused_data, score_lrt_data):
        """Fused mode-4 matches the NumPy Wald, Score and LRT statistics."""
        from jamma.lmm import accel

        eigenvalues, w, Uty, utg_t, uab_inv_soa, uab_var_soa, n_samples = fused_data
        _, _, _, Hi_eval_null, logl_H0 = score_lrt_data

        ws_fused = _ncvt1_workspace(
            fused_data, lmm_mode=4, hi_eval_null=Hi_eval_null, logl_H0=logl_H0
        )
        result = accel.require().compute_lmm_chunk_ncvt1_c(ws_fused, utg_t, 1)

        wald = _numpy_ncvt1_wald(eigenvalues, w, Uty, utg_t, n_samples)
        reference = {k: wald[k] for k in _WALD_KEYS}
        reference["p_scores"] = _numpy_ncvt1_score(
            w, Uty, utg_t, Hi_eval_null, n_samples
        )["p_scores"]
        reference.update(_numpy_ncvt1_lrt(eigenvalues, w, Uty, utg_t, logl_H0))

        assert_matches_numpy(result, reference, "Fused mode-4")

    def test_fused_wrong_utg_t_shape(self, fused_data):
        """Fused compute raises ValueError for wrong UtG_T shape."""
        from jamma.lmm import accel

        _, _, _, utg_t, _, _, _ = fused_data

        ws = _ncvt1_workspace(fused_data, lmm_mode=1)

        # 3D instead of 2D
        bad_utg = utg_t.reshape(utg_t.shape[0], 1, utg_t.shape[1])
        with pytest.raises(ValueError, match="utg_t"):
            accel.require().compute_lmm_chunk_ncvt1_c(ws, bad_utg, 1)

    def test_fused_workspace_refcount(self, fused_data):
        """w and Uty arrays not garbage collected while workspace alive."""
        import gc
        import sys

        from jamma.lmm import accel

        eigenvalues, w, Uty, _, uab_inv_soa, _, n_samples = fused_data

        # Make copies that we can track
        w_tracked = w.copy()
        Uty_tracked = Uty.copy()
        initial_w_ref = sys.getrefcount(w_tracked)
        initial_Uty_ref = sys.getrefcount(Uty_tracked)

        ws = accel.require().create_workspace_ncvt1_c(
            eigenvalues,
            uab_inv_soa,
            w_tracked,
            Uty_tracked,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            lmm_mode=1,
        )

        # Workspace should hold a reference to w and Uty
        assert sys.getrefcount(w_tracked) > initial_w_ref
        assert sys.getrefcount(Uty_tracked) > initial_Uty_ref

        del ws
        gc.collect()

        # After workspace destruction, refcounts should be back to initial
        assert sys.getrefcount(w_tracked) == initial_w_ref
        assert sys.getrefcount(Uty_tracked) == initial_Uty_ref

    def test_fused_degenerate_snps(self, fused_data):
        """Fused Wald handles degenerate (constant) SNPs: NaN beta/se/pwald."""
        from jamma.lmm import accel

        _, _, _, utg_t, _, _, _ = fused_data

        # Make first SNP degenerate: constant genotype -> all zeros after rotation
        utg_t_degen = utg_t.copy()
        utg_t_degen[0, :] = 0.0

        ws = _ncvt1_workspace(fused_data, lmm_mode=1)
        cr = accel.require().compute_lmm_chunk_ncvt1_c(ws, utg_t_degen, 1)

        # Degenerate SNP: should produce NaN
        assert np.isnan(cr["betas"][0]), "degenerate SNP should have NaN beta"
        assert np.isnan(cr["ses"][0]), "degenerate SNP should have NaN se"
        assert np.isnan(cr["pwalds"][0]), "degenerate SNP should have NaN p_wald"

        # Non-degenerate SNPs should still be valid (compare against reference)
        ws_ref = _ncvt1_workspace(fused_data, lmm_mode=1)
        cr_ref = accel.require().compute_lmm_chunk_ncvt1_c(ws_ref, utg_t, 1)
        finite_mask = np.isfinite(cr_ref["betas"][1:])
        assert np.all(np.isfinite(cr["betas"][1:][finite_mask])), (
            "non-degenerate betas should be finite"
        )

    def test_ncvt1_dispatches_by_workspace_mode(self, fused_data):
        """compute_lmm_chunk_ncvt1_c reads the loop to run off the workspace.

        One entry point now serves every n_cvt=1 workspace mode; the mode the
        creator recorded picks the loop, not which name the caller used.
        """
        from jamma.lmm import accel

        eigenvalues, w, Uty, utg_t, uab_inv_soa, uab_var_soa, n_samples = fused_data
        Hi_eval_null = 1.0 / (0.5 * eigenvalues + 1.0)

        score_ws = _ncvt1_workspace(fused_data, lmm_mode=3, hi_eval_null=Hi_eval_null)
        result = accel.require().compute_lmm_chunk_ncvt1_c(score_ws, utg_t, 1)

        reference = _numpy_ncvt1_score(w, Uty, utg_t, Hi_eval_null, n_samples)
        assert_matches_numpy(result, reference, "ncvt1 dispatch, mode 3")


def _run_fused_general_wald_vs_numpy(data: dict) -> None:
    """Compare the fused general Wald kernel against the NumPy Wald path.

    The reference was the non-fused general workspace, which let this be
    bitwise. No dispatch path reaches that kernel, so it has gone and the
    reference is now an independent implementation with a tolerance.
    """
    prepared = _prepare_fused_general_data(data)
    result = accel.require().compute_lmm_chunk_fused_general_c(
        _fused_general_workspace(prepared), prepared["utg_t"], 1
    )
    reference = _numpy_general_wald(prepared)

    assert_matches_numpy(
        result,
        {k: reference[k] for k in _WALD_KEYS},
        f"Fused general Wald n_cvt={data['n_cvt']}",
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="Fused general C not available",
)
def test_fused_general_ncvt2_wald(synthetic_covariate_data_ncvt2):
    """FGEN-04: Fused general Wald bitwise matches non-fused general for n_cvt=2."""
    _run_fused_general_wald_vs_numpy(
        _prepare_fused_general_data(synthetic_covariate_data_ncvt2)
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="Fused general C not available",
)
def test_fused_general_ncvt4_wald(synthetic_covariate_data_ncvt4):
    """FGEN-04: Fused general Wald bitwise matches non-fused general for n_cvt=4."""
    _run_fused_general_wald_vs_numpy(
        _prepare_fused_general_data(synthetic_covariate_data_ncvt4)
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="Mode-4 fused general C not available",
)
def test_fused_general_ncvt2_mode4(general_score_lrt_ncvt2):
    """FGEN-07: Fused general mode-4 Wald matches the NumPy Wald for n_cvt=2.

    The Wald component is checked against NumPy; Score and LRT are checked for
    shape and range here, and against NumPy in the two tests below.
    """
    data = _prepare_fused_general_data(general_score_lrt_ncvt2)
    result = accel.require().compute_lmm_chunk_fused_general_c(
        _fused_general_mode4_workspace(data), data["utg_t"], 1
    )
    reference = _numpy_general_wald(data)

    assert_matches_numpy(
        result,
        {k: reference[k] for k in _WALD_KEYS},
        "Fused general mode-4 Wald n_cvt=2",
    )

    n_snps = data["UtG"].shape[1]
    for key in ("p_scores", "p_lrts", "lambdas_mle"):
        assert result[key].shape == (n_snps,), f"{key} shape mismatch"

    for key in ("p_scores", "p_lrts"):
        finite = result[key][np.isfinite(result[key])]
        assert np.all((finite >= 0) & (finite <= 1)), f"{key} out of range [0, 1]"


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="Fused general C not available",
)
def test_fused_general_mode4_nan_lambda_regression(general_score_lrt_ncvt2):
    """FGEN-08: Regression test — fused general mode-4 produces finite lambda_mle.

    Previously, fused general mode-4 produced NaN lambda_mle due to missing
    mle_const in the workspace. This test verifies the fix: all non-degenerate
    SNPs must have finite lambda_mle values.
    """
    from jamma.lmm.likelihood import build_pab_table_for_c, classify_uab_columns

    data = general_score_lrt_ncvt2
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    Uab_batch = data["Uab_batch"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]
    Hi_eval_null = data["Hi_eval_null"]
    logl_H0 = data["logl_H0"]

    inv_indices, _ = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    utg_t = np.ascontiguousarray(UtG.T)
    ws_fused = accel.require().create_workspace_general_c(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        build_pab_table_for_c(n_cvt)._asdict(),
        lmm_mode=4,
        hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
    )
    result = accel.require().compute_lmm_chunk_fused_general_c(ws_fused, utg_t, 1)

    # All non-degenerate SNPs must have finite lambda_mle
    lambdas_mle = result["lambdas_mle"]
    # Degenerate SNPs (constant genotype) may produce NaN — check non-degenerate
    non_degen = np.isfinite(result["betas"])  # Wald beta finite => non-degenerate
    assert np.all(np.isfinite(lambdas_mle[non_degen])), (
        f"NaN lambda_mle found for {np.sum(~np.isfinite(lambdas_mle[non_degen]))} "
        f"non-degenerate SNPs (regression: mode-4 fused general NaN bug)"
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="Fused general C not available",
)
def test_fused_general_mode4_lrt_parity_ncvt2(general_score_lrt_ncvt2):
    """FGEN-08: Fused general mode-4 LRT matches the NumPy MLE lambdas and p-values."""
    data = _prepare_fused_general_data(general_score_lrt_ncvt2)
    result = accel.require().compute_lmm_chunk_fused_general_c(
        _fused_general_mode4_workspace(data), data["utg_t"], 1
    )

    assert_matches_numpy(
        result, _numpy_general_lrt(data), "Fused general mode-4 LRT n_cvt=2"
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="Fused general C not available",
)
def test_fused_general_mode4_all_statistics_ncvt2(general_score_lrt_ncvt2):
    """FGEN-09: every mode-4 statistic from the fused general kernel matches NumPy.

    Mode 4 composes Wald, Score and LRT in one pass over the workspace, so a
    mix-up between the three shows here and not in the single-mode tests.
    """
    data = _prepare_fused_general_data(general_score_lrt_ncvt2)
    result = accel.require().compute_lmm_chunk_fused_general_c(
        _fused_general_mode4_workspace(data), data["utg_t"], 1
    )

    wald = _numpy_general_wald(data)
    reference = {k: wald[k] for k in _WALD_KEYS}
    reference["p_scores"] = _numpy_general_score(data)["p_scores"]
    reference.update(_numpy_general_lrt(data))

    assert_matches_numpy(result, reference, "Fused general mode-4 n_cvt=2")


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="Fused general C not available",
)
def test_fused_general_workspace_lifecycle(synthetic_covariate_data_ncvt2):
    """FGEN-04: Fused general workspace creates, computes, and destroys cleanly."""
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = synthetic_covariate_data_ncvt2
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    n_cvt = data["n_cvt"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty, n_cvt)
    utg_t = np.ascontiguousarray(UtG.T)
    ws = accel.require().create_workspace_general_c(
        eigenvalues,
        uab_inv_soa,
        UtW,
        Uty,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
        build_pab_table_for_c(n_cvt)._asdict(),
        lmm_mode=1,
    )
    assert ws is not None

    # Compute first half
    mid = UtG.shape[1] // 2
    r1 = accel.require().compute_lmm_chunk_fused_general_c(ws, utg_t[:mid], 1)
    assert r1["lambdas"].shape == (mid,)

    # Reuse workspace for second half
    r2 = accel.require().compute_lmm_chunk_fused_general_c(ws, utg_t[mid:], 1)
    assert r2["lambdas"].shape == (UtG.shape[1] - mid,)

    # Full batch
    r_full = accel.require().compute_lmm_chunk_fused_general_c(ws, utg_t, 1)
    combined = np.concatenate([r1["lambdas"], r2["lambdas"]])
    np.testing.assert_allclose(
        combined,
        r_full["lambdas"],
        rtol=1e-12,
        atol=1e-14,
        err_msg="Chunked vs full fused general workspace mismatch",
    )

    # Destroy (PyCapsule GC)
    del ws


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="Fused general C not available",
)
def test_fused_general_degenerate_snps(synthetic_covariate_data_ncvt2):
    """FGEN-05: constant genotypes give NaN, and the rest still match NumPy.

    A constant genotype rotates to an all-zero UtG column, which drives xx and
    so P_XX to zero. Both sides are given the same degenerate input, so the
    comparison stays between two implementations of one problem.
    """
    from jamma.lmm.likelihood import compute_Uab

    data = dict(synthetic_covariate_data_ncvt2)
    UtG = data["UtG"].copy()
    UtG[:, :2] = 0.0
    data["UtG"] = UtG
    data["Uab_batch"] = np.stack(
        [compute_Uab(data["UtW"], data["Uty"], UtG[:, i]) for i in range(UtG.shape[1])]
    )

    prepared = _prepare_fused_general_data(data)
    result = accel.require().compute_lmm_chunk_fused_general_c(
        _fused_general_workspace(prepared), prepared["utg_t"], 1
    )
    reference = _numpy_general_wald(prepared)

    for key in ("betas", "ses", "pwalds"):
        assert np.all(np.isnan(result[key][:2])), (
            f"{key}: degenerate SNPs should be NaN"
        )
        np.testing.assert_array_equal(
            np.isnan(result[key][:2]),
            np.isnan(reference[key][:2]),
            err_msg=f"{key}: NaN pattern differs from NumPy on the degenerate SNPs",
        )

    assert_matches_numpy(
        {k: result[k][2:] for k in _WALD_KEYS},
        {k: reference[k][2:] for k in _WALD_KEYS},
        "Fused general non-degenerate",
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    not accel.available(),
    reason="C extension not available",
)
def test_fused_general_abi_version_9():
    """FGEN-06: ABI_VERSION is >= 9 for fused general kernel support."""
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION >= 9, f"Expected ABI_VERSION>=9, got {ABI_VERSION}"


@pytest.mark.tier1
@pytest.mark.skipif(
    not accel.available(),
    reason="Fused general C not available",
)
def test_runner_fused_general_ncvt2_dispatch():
    """Runner integration: n_cvt=2 dispatches fused general path end-to-end.

    Exercises the full build_pab_table_for_c → create_workspace_fused_general →
    compute_lmm_chunk_fused_general_c pipeline through run_lmm_association_numpy.
    Compares fused general results (n_cvt=2 with the C extension) against the
    NumPy path, reached by dropping the extension. Not bitwise: the reference
    run is the NumPy path, not a second C path. Dropping the fused general
    kernel used to leave the general split kernel in place, and the two agreed
    to the last bit; no build exports one without the other, so the honest
    reference is NumPy, which accumulates in a different order.
    """
    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    rng = np.random.default_rng(77)
    n_samples = 100
    n_snps = 80
    n_cvt = 2

    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.4, 0.4, 0.2])
    phenotypes = rng.standard_normal(n_samples)
    covariates = rng.standard_normal((n_samples, n_cvt))
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]
    U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]

    def run():
        return run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=U,
            config=LmmConfig(
                maf_threshold=0.0,
                miss_threshold=1.0,
                check_memory=False,
                show_progress=False,
                lmm_mode=1,
                n_refine=20,
            ),
        )

    # No single spy target: dropping the extension leaves no second C path,
    # so the reference run is the NumPy path itself (kernel=None skips the
    # dispatch-reached assertion the other fused tests make).
    assert_fused_matches_reference(
        run,
        fields={"p_wald": 1e-8, "beta": 1e-8},
        kernel=None,
        min_count=n_snps * 0.8,
        atol=1e-14,
    )


@pytest.mark.tier0
@requires_c
@pytest.mark.parametrize(
    "key",
    [
        "invariant_indices",
        "varying_indices",
        "logdet_diag_rows",
        "logdet_diag_cols",
        "var_a_cols",
    ],
)
def test_general_creator_rejects_out_of_range_table(
    synthetic_covariate_data_ncvt2, key
):
    """The Pab table parser range-checks every index array it is handed.

    The general creator takes the table as one dict, so a corrupt entry has
    to be caught there rather than by the workspace filling code that used
    to read the arrays one by one.
    """
    from jamma.lmm.likelihood import build_pab_table_for_c

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    table = build_pab_table_for_c(n_cvt)._asdict()
    bad = np.array(table[key], dtype=np.int32).copy()
    bad[0] = 10**6
    table[key] = bad

    with pytest.raises(ValueError, match=rf"{key}\[0\].*out of range"):
        accel.require().create_workspace_general_c(
            data["eigenvalues"],
            compute_uab_invariant_soa(data["UtW"], data["Uty"], n_cvt),
            data["UtW"],
            data["Uty"],
            data["n_samples"],
            1e-5,
            1e5,
            50,
            20,
            1,
            table,
            lmm_mode=1,
        )


@pytest.mark.tier0
@requires_c
def test_general_creator_rejects_mode_5():
    """lmm_mode outside 1..4 is rejected, whatever n_cvt.

    D2 gave the general creator every lmm_mode 1..4 (previously 1 and 4
    only, with modes 2 and 3 served by the now-deleted SOA_SPLIT entry
    points); mode 2/3 acceptance is covered directly in
    tests/lmm_accel/test_lmm_accel_workspace_score_lrt.py. This pins the
    bound still enforced past 4.
    """
    from jamma.lmm.likelihood import build_pab_table_for_c

    n_cvt, n_samples = 2, 20
    inputs = rotated_lmm_inputs(
        n_samples=n_samples, n_cvt=n_cvt, seed=1, n_snps=1, eig_range=(0.1, 2.0)
    )
    eigenvalues, UtW, Uty = inputs.eigenvalues, inputs.UtW, inputs.Uty
    with pytest.raises(ValueError, match="lmm_mode must be 1, 2, 3 or 4"):
        accel.require().create_workspace_general_c(
            eigenvalues,
            compute_uab_invariant_soa(UtW, Uty, n_cvt),
            UtW,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
            build_pab_table_for_c(n_cvt)._asdict(),
            lmm_mode=5,
        )
