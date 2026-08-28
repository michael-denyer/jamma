"""_lmm_accel C extension tests: general n_cvt Wald kernels.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from jamma.lmm import compute_numpy
from jamma.lmm.compute_numpy import (
    _c,
    compute_lmm_chunk_numpy,
    compute_wald_fused_general_c_ws,
)
from jamma.lmm.schema import LmmConfig
from tests.lmm_accel._helpers import (
    _fused_general_workspace,
    _prepare_fused_general_data,
    _run_general_ncvt_c_vs_python,
)


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_general_ncvt_reml_wald_matches_python_ncvt2(
    synthetic_covariate_data_ncvt2,
):
    """C-GEN-01: C extension Wald results match Python for n_cvt=2."""
    _run_general_ncvt_c_vs_python(synthetic_covariate_data_ncvt2)


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_general_ncvt_reml_wald_ncvt4(
    synthetic_covariate_data_ncvt4,
):
    """C-GEN-01: C extension Wald results match Python for n_cvt=4."""
    _run_general_ncvt_c_vs_python(synthetic_covariate_data_ncvt4)


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_general_ncvt_workspace_lifecycle(synthetic_covariate_data_ncvt2):
    """C-GEN-02: Workspace create/compute/destroy cycle works for n_cvt>1."""
    data = _prepare_fused_general_data(synthetic_covariate_data_ncvt2)
    utg_t = data["utg_t"]
    n_snps = utg_t.shape[0]

    ws = _fused_general_workspace(data)
    assert ws is not None

    mid = n_snps // 2
    r1 = compute_wald_fused_general_c_ws(ws, utg_t[:mid], 1)
    assert r1["lambdas"].shape == (mid,)

    # Reuse the same workspace for the second chunk.
    r2 = compute_wald_fused_general_c_ws(ws, utg_t[mid:], 1)
    assert r2["lambdas"].shape == (n_snps - mid,)

    r_full = compute_wald_fused_general_c_ws(ws, utg_t, 1)
    np.testing.assert_allclose(
        np.concatenate([r1["lambdas"], r2["lambdas"]]),
        r_full["lambdas"],
        rtol=1e-12,
        atol=1e-14,
        err_msg="Chunked vs full workspace mismatch",
    )

    del ws


@pytest.mark.tier1
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_general_ncvt_gemma_covariate_match():
    """C-GEN-03: C extension Wald results match GEMMA reference with covariates.

    End-to-end test: loads gemma_synthetic PLINK data + covariates, runs the
    NumPy runner (which uses the general C workspace for n_cvt=2 Wald), and
    compares against GEMMA's covariate reference output.
    """
    from jamma.io import load_plink_binary, read_fam_phenotypes
    from jamma.kinship.io import read_kinship_matrix
    from jamma.lmm.runner_numpy import run_lmm_association_numpy
    from jamma.validation import (
        ToleranceConfig,
        compare_assoc_results,
        load_gemma_assoc,
    )
    from tests.fixture_paths import SYNTHETIC

    plink = load_plink_binary(SYNTHETIC.bfile)
    kinship = read_kinship_matrix(SYNTHETIC.kinship)
    phenotypes = read_fam_phenotypes(SYNTHETIC.fam)
    covariates = np.loadtxt(SYNTHETIC.covariates)
    snp_info = [
        {
            "chr": str(plink.chromosome[i]),
            "rs": plink.sid[i],
            "pos": plink.bp_position[i],
            "a1": plink.allele_1[i],
            "a0": plink.allele_2[i],
            "maf": 0.0,
            "n_miss": 0,
        }
        for i in range(plink.n_snps)
    ]

    run_result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        config=LmmConfig(lmm_mode=1, show_progress=False),
    )
    results = run_result.associations

    reference = load_gemma_assoc(SYNTHETIC.ref("covar_wald"))
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"C extension Wald+covariate vs GEMMA failed:\n{comparison}"
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_general_ncvt_all_modes(synthetic_covariate_data_ncvt2, monkeypatch):
    """C-GEN-04: All 4 LMM modes produce results with n_cvt=2 covariates.

    ``compute_lmm_chunk_numpy`` is the full-Uab NumPy path, and the runner
    reaches it only on ``DispatchPath.NUMPY_FALLBACK``, which is selected only
    when the extension is absent. The extension is cleared here so the test
    drives the path in the state production actually uses it in; left loaded, it
    would take an inner C ladder no dispatch path selects.
    """
    from jamma.lmm.prepare_common import _compute_null_model_common
    from jamma.lmm.uab import batch_compute_uab_numpy

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG.T)
    n_snps = Uab_batch.shape[0]

    logl_H0, _lambda_mle, Hi_eval_null = _compute_null_model_common(
        4, eigenvalues, UtW, Uty, n_cvt, False
    )

    monkeypatch.setattr(compute_numpy, "_accel", None)

    common = {
        "n_cvt": n_cvt,
        "eigenvalues": eigenvalues,
        "Uab_batch": Uab_batch,
        "n_samples": n_samples,
        "n_threads": 1,
    }

    result = compute_lmm_chunk_numpy(
        lmm_mode=4, logl_H0=logl_H0, Hi_eval_null=Hi_eval_null, **common
    )

    for key in (
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "p_lrts",
        "p_scores",
    ):
        arr = result[key]
        assert arr is not None, f"{key} is None in mode 4"
        assert arr.shape == (n_snps,), f"{key} shape mismatch: {arr.shape}"

    for key in ("betas", "ses", "pwalds"):
        arr = result[key]
        assert arr is not None, f"{key} is None in mode 4"
        n_finite = np.sum(np.isfinite(arr))
        assert n_finite > n_snps * 0.8, f"{key}: only {n_finite}/{n_snps} finite values"

    result_lrt = compute_lmm_chunk_numpy(lmm_mode=2, logl_H0=logl_H0, **common)
    assert result_lrt["lambdas_mle"] is not None
    assert result_lrt["p_lrts"] is not None
    assert result_lrt["lambdas_mle"].shape == (n_snps,)

    result_score = compute_lmm_chunk_numpy(
        lmm_mode=3, Hi_eval_null=Hi_eval_null, **common
    )
    assert result_score["p_scores"] is not None
    assert result_score["betas"] is not None
    assert result_score["p_scores"].shape == (n_snps,)


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_general_ncvt_openmp_deterministic(synthetic_covariate_data_ncvt2):
    """C-GEN-05: 1-thread vs N-thread produce identical results for n_cvt>1."""
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    data = _prepare_fused_general_data(synthetic_covariate_data_ncvt2)
    ws = _fused_general_workspace(data)

    r1 = compute_wald_fused_general_c_ws(ws, data["utg_t"], 1)
    rn = compute_wald_fused_general_c_ws(ws, data["utg_t"], n_threads)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            rn[key],
            r1[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: general MT vs ST mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="General C extension unavailable"
)
def test_general_ncvt_degenerate_snps(synthetic_covariate_data_ncvt2):
    """C-GEN-06: Constant genotypes produce NaN beta/se/p-value for n_cvt>1."""
    data = _prepare_fused_general_data(synthetic_covariate_data_ncvt2)

    # A constant genotype rotates to an all-zero UtG column, which drives xx to
    # zero and so P_XX to zero. Zeroing the row is how the fused kernel, which
    # builds Uab from UtG itself, is given a degenerate SNP.
    utg_t = data["utg_t"].copy()
    utg_t[[0, 2]] = 0.0

    ws = _fused_general_workspace(data)
    result = compute_wald_fused_general_c_ws(ws, utg_t, 1)

    for snp_idx in (0, 2):
        assert np.isnan(result["betas"][snp_idx]), f"SNP {snp_idx}: expected NaN beta"
        assert np.isnan(result["ses"][snp_idx]), f"SNP {snp_idx}: expected NaN se"
        assert np.isnan(result["pwalds"][snp_idx]), f"SNP {snp_idx}: expected NaN pwald"

    for snp_idx in (1, 3):
        assert np.isfinite(result["betas"][snp_idx]), (
            f"SNP {snp_idx}: expected finite beta"
        )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_general_ncvt_abi_version():
    """C-GEN-07: ABI version is 13 after the four n_cvt=1 creators became one."""
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION == 13, f"Expected ABI_VERSION=13, got {ABI_VERSION}"


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_existing_ncvt1_regression(synthetic_wald_data):
    """C-GEN-08: the n_cvt=1 fused workspace path still works after the general work.

    Ensures the general n_cvt additions (ABI bump, extra workspace types) did not
    regress the original n_cvt=1 path.
    """
    eigenvalues, Uab_batch, n_samples = synthetic_wald_data

    # The fused kernel builds Uab from w and UtG itself, so it is given the
    # invariant SoA plus the raw rotated vectors rather than a prebuilt Uab.
    # Column layout: 0=ww, 1=wx, 2=wy, 3=xx, 4=xy, 5=yy, and the fixture builds
    # every column from a positive w, so recovering w and Uty from it is exact.
    w = np.sqrt(Uab_batch[0, :, 0])
    Uty = Uab_batch[0, :, 2] / w
    utg_t = np.ascontiguousarray(Uab_batch[:, :, 1] / w)
    uab_inv_soa = np.stack(
        [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
    )

    ws = _c().create_workspace_ncvt1_c(
        eigenvalues, uab_inv_soa, w, Uty, n_samples, 1e-5, 1e5, 50, 20, lmm_mode=1
    )
    result = _c().compute_lmm_chunk_fused_c(ws, utg_t, 1)

    assert result["lambdas"].shape == (Uab_batch.shape[0],)
    assert result["betas"].shape == (Uab_batch.shape[0],)
    assert np.sum(np.isfinite(result["betas"])) > 0, (
        "No finite betas from the n_cvt=1 fused workspace"
    )
