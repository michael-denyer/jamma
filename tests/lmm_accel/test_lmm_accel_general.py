"""_lmm_accel C extension tests: general n_cvt Wald kernels.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from jamma.lmm.compute_numpy import (
    _C_ACCEL_AVAILABLE,
    _C_GENERAL_AVAILABLE,
    compute_lmm_chunk_numpy,
    compute_wald_general_c_ws,
    compute_wald_split_c_ws,
    create_lmm_workspace,
    create_lmm_workspace_general,
)
from jamma.lmm.schema import LmmConfig
from tests.lmm_accel._helpers import (
    _run_general_ncvt_c_vs_python,
)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_reml_wald_matches_python_ncvt2(
    synthetic_covariate_data_ncvt2,
):
    """C-GEN-01: C extension Wald results match Python for n_cvt=2."""
    _run_general_ncvt_c_vs_python(synthetic_covariate_data_ncvt2)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_reml_wald_ncvt4(
    synthetic_covariate_data_ncvt4,
):
    """C-GEN-01: C extension Wald results match Python for n_cvt=4."""
    _run_general_ncvt_c_vs_python(synthetic_covariate_data_ncvt4)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_workspace_lifecycle(synthetic_covariate_data_ncvt2):
    """C-GEN-02: Workspace create/compute/destroy cycle works for n_cvt>1."""
    from jamma.lmm.likelihood import classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    # a[0, :, list_idx] -> (n_inv, n_samples) due to numpy advanced indexing
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    # Create workspace
    ws = create_lmm_workspace_general(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        n_cvt,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    assert ws is not None

    # Compute first chunk
    mid = Uab_batch.shape[0] // 2
    r1 = compute_wald_general_c_ws(ws, uab_var_soa[:mid], 1)
    assert r1["lambdas"].shape == (mid,)

    # Reuse workspace for second chunk
    r2 = compute_wald_general_c_ws(ws, uab_var_soa[mid:], 1)
    assert r2["lambdas"].shape == (Uab_batch.shape[0] - mid,)

    # Full batch
    r_full = compute_wald_general_c_ws(ws, uab_var_soa, 1)
    combined = np.concatenate([r1["lambdas"], r2["lambdas"]])
    np.testing.assert_allclose(
        combined,
        r_full["lambdas"],
        rtol=1e-12,
        atol=1e-14,
        err_msg="Chunked vs full workspace mismatch",
    )

    # Destroy workspace (PyCapsule GC)
    del ws


@pytest.mark.tier1
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_gemma_covariate_match():
    """C-GEN-03: C extension Wald results match GEMMA reference with covariates.

    End-to-end test: loads gemma_synthetic PLINK data + covariates, runs the
    NumPy runner (which uses the general C workspace for n_cvt=2 Wald), and
    compares against GEMMA's covariate reference output.
    """
    from pathlib import Path

    from jamma.io import load_plink_binary
    from jamma.kinship.io import read_kinship_matrix
    from jamma.lmm.runner_numpy import run_lmm_association_numpy
    from jamma.validation import (
        ToleranceConfig,
        compare_assoc_results,
        load_gemma_assoc,
    )
    from tests.conftest import load_phenotypes_from_fam

    # parents[1] is tests/: this module lives one level down in lmm_accel/.
    fixture_root = Path(__file__).parents[1] / "fixtures"
    synthetic_dir = fixture_root / "gemma_synthetic"
    covariate_dir = fixture_root / "gemma_covariate"

    plink = load_plink_binary(synthetic_dir / "test")
    kinship = read_kinship_matrix(synthetic_dir / "gemma_kinship.cXX.txt")
    phenotypes = load_phenotypes_from_fam(synthetic_dir / "test.fam")
    covariates = np.loadtxt(covariate_dir / "covariates.txt")
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

    reference = load_gemma_assoc(covariate_dir / "gemma_covariate.assoc.txt")
    tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(results, reference, tolerances)
    assert comparison.passed, (
        f"C extension Wald+covariate vs GEMMA failed:\n{comparison}"
    )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_all_modes(synthetic_covariate_data_ncvt2):
    """C-GEN-04: All 4 LMM modes produce results with n_cvt=2 covariates.

    Verifies that compute_lmm_chunk_numpy with lmm_mode=4 produces non-None
    results for all output fields when covariates are present. Wald results
    use the C extension; LRT/Score use the Python fallback.
    """
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy
    from jamma.lmm.prepare_common import _compute_null_model_common

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    n_samples = data["n_samples"]
    UtW = data["UtW"]
    Uty = data["Uty"]
    UtG = data["UtG"]

    # Build Uab
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)
    n_snps = Uab_batch.shape[0]

    # Compute null model for LRT/Score
    logl_H0, _lambda_mle, Hi_eval_null = _compute_null_model_common(
        4, eigenvalues, UtW, Uty, n_cvt, False
    )

    # Mode 4 (All): exercises Wald (C ext), LRT (Python MLE), Score (Python)
    result = compute_lmm_chunk_numpy(
        lmm_mode=4,
        n_cvt=n_cvt,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        logl_H0=logl_H0,
        Hi_eval_null=Hi_eval_null,
        n_threads=1,
    )

    # All fields must be non-None and have correct shape
    all_keys = (
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "p_lrts",
        "p_scores",
    )
    for key in all_keys:
        arr = result[key]
        assert arr is not None, f"{key} is None in mode 4"
        assert arr.shape == (n_snps,), f"{key} shape mismatch: {arr.shape}"

    # Finite check (most values should be finite; allow NaN for degenerate SNPs)
    for key in ("betas", "ses", "pwalds"):
        arr = result[key]
        assert arr is not None, f"{key} is None in mode 4"
        n_finite = np.sum(np.isfinite(arr))
        assert n_finite > n_snps * 0.8, f"{key}: only {n_finite}/{n_snps} finite values"

    # Mode 2 (LRT only)
    result_lrt = compute_lmm_chunk_numpy(
        lmm_mode=2,
        n_cvt=n_cvt,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        logl_H0=logl_H0,
        n_threads=1,
    )
    assert result_lrt["lambdas_mle"] is not None
    assert result_lrt["p_lrts"] is not None
    assert result_lrt["lambdas_mle"].shape == (n_snps,)

    # Mode 3 (Score only)
    result_score = compute_lmm_chunk_numpy(
        lmm_mode=3,
        n_cvt=n_cvt,
        eigenvalues=eigenvalues,
        Uab_batch=Uab_batch,
        n_samples=n_samples,
        Hi_eval_null=Hi_eval_null,
        n_threads=1,
    )
    assert result_score["p_scores"] is not None
    assert result_score["betas"] is not None
    assert result_score["p_scores"].shape == (n_snps,)


@pytest.mark.tier0
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_openmp_deterministic(synthetic_covariate_data_ncvt2):
    """C-GEN-05: 1-thread vs N-thread produce identical results for n_cvt>1."""
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    from jamma.lmm.likelihood import classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"]
    n_samples = data["n_samples"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)
    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    ws = create_lmm_workspace_general(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        n_cvt,
        1e-5,
        1e5,
        50,
        20,
        1,
    )

    r1 = compute_wald_general_c_ws(ws, uab_var_soa, 1)
    rn = compute_wald_general_c_ws(ws, uab_var_soa, n_threads)

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
@pytest.mark.skipif(not _C_GENERAL_AVAILABLE, reason="General C extension unavailable")
def test_general_ncvt_degenerate_snps(synthetic_covariate_data_ncvt2):
    """C-GEN-06: Constant genotypes produce NaN beta/se/p-value for n_cvt>1."""
    from jamma.lmm.likelihood import classify_uab_columns

    data = synthetic_covariate_data_ncvt2
    n_cvt = data["n_cvt"]
    eigenvalues = data["eigenvalues"]
    Uab_batch = data["Uab_batch"].copy()
    n_samples = data["n_samples"]

    inv_indices, var_indices = classify_uab_columns(n_cvt)

    # Make SNPs 0 and 2 degenerate by zeroing all varying columns
    # (this makes xx=0, causing P_XX <= 0)
    for snp_idx in [0, 2]:
        for vi in var_indices:
            Uab_batch[snp_idx, :, vi] = 0.0

    uab_inv_soa = np.ascontiguousarray(Uab_batch[0, :, list(inv_indices)])
    uab_var_soa = np.ascontiguousarray(
        Uab_batch[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    ws = create_lmm_workspace_general(
        eigenvalues,
        uab_inv_soa,
        n_samples,
        n_cvt,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    result = compute_wald_general_c_ws(ws, uab_var_soa, 1)

    # Degenerate SNPs should have NaN
    for snp_idx in [0, 2]:
        assert np.isnan(result["betas"][snp_idx]), f"SNP {snp_idx}: expected NaN beta"
        assert np.isnan(result["ses"][snp_idx]), f"SNP {snp_idx}: expected NaN se"
        assert np.isnan(result["pwalds"][snp_idx]), f"SNP {snp_idx}: expected NaN pwald"

    # Non-degenerate SNPs should have valid results
    for snp_idx in [1, 3]:
        assert np.isfinite(result["betas"][snp_idx]), (
            f"SNP {snp_idx}: expected finite beta"
        )


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_general_ncvt_abi_version():
    """C-GEN-07: ABI version is 11 for persistent Score/LRT workspaces."""
    from jamma.lmm._lmm_accel import ABI_VERSION

    assert ABI_VERSION == 11, f"Expected ABI_VERSION=11, got {ABI_VERSION}"


@pytest.mark.tier0
@pytest.mark.skipif(not _C_ACCEL_AVAILABLE, reason="C extension not compiled")
def test_existing_ncvt1_regression(synthetic_wald_data):
    """C-GEN-08: Existing n_cvt=1 C extension path unchanged with ABI_VERSION=5.

    Ensures the general n_cvt additions (ABI_VERSION bump, new workspace types)
    did not regress the original n_cvt=1 split-Uab workspace path.
    """
    eigenvalues, Uab_batch, n_samples = synthetic_wald_data

    # Use the existing Uab directly for split components
    uab_varying_soa = np.stack(
        [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
    )
    uab_inv_soa_direct = np.stack(
        [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
    )

    # Create n_cvt=1 workspace and compute
    ws = create_lmm_workspace(
        eigenvalues, uab_inv_soa_direct, n_samples, 1e-5, 1e5, 50, 20, 1
    )
    result = compute_wald_split_c_ws(ws, uab_varying_soa, 1)

    # Basic sanity: shapes match, most values finite
    assert result["lambdas"].shape == (Uab_batch.shape[0],)
    assert result["betas"].shape == (Uab_batch.shape[0],)
    n_finite = np.sum(np.isfinite(result["betas"]))
    assert n_finite > 0, "No finite betas from n_cvt=1 workspace"
