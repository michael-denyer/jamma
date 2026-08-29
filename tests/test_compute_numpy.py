"""Parity tests for the general workspace's Score/LRT-only modes against the
full-Uab path.

The runner feeds the general workspace's compute directly rather than
reconstructing a full Uab per chunk. These check on real data that this does
not change the statistics.

Run at n_cvt=2, which is where ``DispatchPath.FUSED_GENERAL`` is selected for
every lmm_mode. modes 2 and 3 at n_cvt=1 take the n_cvt=1 fused workspace
kernels instead (``tests/lmm_accel/test_lmm_accel_workspace_score_lrt.py``).

The reference is the NumPy full-Uab path, not its C sibling. The C full-Uab
kernels compute_score_batch_general_c and compute_lrt_batch_general_c were
deleted, so pointing at them would have compared one dead kernel against
another the moment the live subject changed.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm import compute_numpy
from jamma.lmm.compute_numpy import (
    _c,
    _compute_lrt_numpy,
    _compute_score_numpy,
    compute_lmm_chunk_numpy,
)
from jamma.lmm.likelihood import build_pab_table_for_c, compute_null_model_mle
from jamma.lmm.uab import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)
from tests.builders import rotated_lmm_inputs
from tests.fixture_paths import MOUSE


@contextlib.contextmanager
def _numpy_only():
    """Hold the extension out, so the full-Uab helpers take their NumPy path.

    They consult compute_numpy._accel at call time and take a C branch when it
    is set, so the attribute has to be cleared rather than an argument changed.
    """
    orig = compute_numpy._accel
    compute_numpy._accel = None
    try:
        yield
    finally:
        compute_numpy._accel = orig


pytestmark = pytest.mark.tier0

# C against NumPy rather than C against C, so the bound is set by accumulation
# order over 1940 samples rather than by two orderings of the same code.
# Measured worst case on this fixture is 5.1e-9, on the Score p-values of the
# batch that carries the degenerate SNPs.
_C_VS_NUMPY_RTOL = 1e-7


@pytest.fixture(scope="module")
def mouse_data():
    """Load mouse_hs1940 fixture and prepare eigendecomposition + Uab arrays.

    Uses a synthetic phenotype with known signal to ensure a well-conditioned
    MLE null model (finite logl_H0). The mouse_hs1940 column-1 phenotype
    produces a degenerate MLE landscape (NaN logl_H0 at boundary lambda).
    """
    plink_data = load_plink_binary(MOUSE.bfile)
    genotypes = plink_data.genotypes
    K = read_kinship_matrix(MOUSE.kinship)

    n_samples = genotypes.shape[0]
    # n_cvt=2, so the general workspace (FUSED_GENERAL) is exercised.
    n_cvt = 2

    # Eigendecomposition
    eigenvalues, U = np.linalg.eigh(K)

    # Generate synthetic phenotype with genetic signal (ensures finite MLE null)
    rng = np.random.default_rng(42)
    # y = K @ beta + noise, where beta is random — ensures non-degenerate null
    phenotypes = K @ rng.standard_normal(n_samples) * 0.5 + rng.standard_normal(
        n_samples
    )

    # Intercept plus one covariate, so the general workspace has a genuine
    # invariant block to keep separate from the varying one.
    W = np.column_stack([np.ones(n_samples), rng.standard_normal(n_samples)])
    UtW = U.T @ W
    Uty = U.T @ phenotypes

    # Get a small subset of SNPs for parity tests (first 50)
    n_test_snps = min(50, genotypes.shape[1])
    geno_subset = genotypes[:, :n_test_snps].astype(np.float64)

    # Mean-impute
    col_means = np.nanmean(geno_subset, axis=0)
    missing = np.isnan(geno_subset)
    if missing.any():
        geno_subset = np.where(missing, col_means[None, :], geno_subset)

    # Rotate
    UtG = U.T @ geno_subset
    utg_t = np.ascontiguousarray(UtG.T)

    # Build full Uab
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG.T)
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)

    # Null model MLE for Score/LRT
    lambda_null_mle, logl_H0 = compute_null_model_mle(eigenvalues, UtW, Uty, n_cvt)
    Hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues + 1.0)

    return {
        "eigenvalues": eigenvalues,
        "n_samples": n_samples,
        "n_cvt": n_cvt,
        "Uab_batch": Uab_batch,
        "UtW": UtW,
        "Uty": Uty,
        "utg_t": utg_t,
        "uab_inv_soa": uab_inv_soa,
        "Hi_eval_null": Hi_eval_null,
        "logl_H0": logl_H0,
        "lambda_null_mle": lambda_null_mle,
    }


def _general_score_only_result(d):
    """The general workspace's lmm_mode=3 (Score only) compute for *d*."""
    pab_table = build_pab_table_for_c(d["n_cvt"])._asdict()
    ws = _c().create_workspace_general_c(
        d["eigenvalues"],
        d["uab_inv_soa"],
        d["UtW"],
        d["Uty"],
        d["n_samples"],
        1e-5,
        1e5,
        50,
        20,
        1,
        pab_table,
        lmm_mode=3,
        hi_eval_null=d["Hi_eval_null"],
    )
    return _c().compute_lmm_chunk_fused_general_c(ws, d["utg_t"], 1)


def _general_lrt_only_result(d, l_min=1e-5, l_max=1e5, n_grid=50, n_refine=20):
    """The general workspace's lmm_mode=2 (LRT only) compute for *d*."""
    pab_table = build_pab_table_for_c(d["n_cvt"])._asdict()
    ws = _c().create_workspace_general_c(
        d["eigenvalues"],
        d["uab_inv_soa"],
        d["UtW"],
        d["Uty"],
        d["n_samples"],
        l_min,
        l_max,
        n_grid,
        n_refine,
        1,
        pab_table,
        lmm_mode=2,
        logl_H0=d["logl_H0"],
    )
    return _c().compute_lmm_chunk_fused_general_c(ws, d["utg_t"], 1)


@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension unavailable")
class TestScoreSplitParity:
    """The general workspace's Score-only mode matches full-Uab Score."""

    def test_score_split_parity(self, mouse_data):
        """Score-only general workspace vs full-Uab C: betas, ses, p_scores match."""
        d = mouse_data
        # Full-Uab reference, via NumPy
        with _numpy_only():
            full_result = _compute_score_numpy(
                d["n_cvt"],
                d["eigenvalues"],
                d["Hi_eval_null"],
                d["Uab_batch"],
                d["n_samples"],
                n_threads=1,
            )

        general_result = _general_score_only_result(d)

        np.testing.assert_allclose(
            general_result["betas"],
            full_result["betas"],
            rtol=_C_VS_NUMPY_RTOL,
            err_msg="Score-only general workspace betas differ from full-Uab",
        )
        np.testing.assert_allclose(
            general_result["ses"],
            full_result["ses"],
            rtol=_C_VS_NUMPY_RTOL,
            err_msg="Score-only general workspace ses differ from full-Uab",
        )
        np.testing.assert_allclose(
            general_result["p_scores"],
            full_result["p_scores"],
            rtol=_C_VS_NUMPY_RTOL,
            err_msg="Score-only general workspace p_scores differ from full-Uab",
        )


@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension unavailable")
class TestLrtSplitParity:
    """The general workspace's LRT-only mode matches full-Uab LRT."""

    def test_lrt_split_parity(self, mouse_data):
        """LRT-only general workspace vs full-Uab C: lambdas_mle and p_lrts match."""
        d = mouse_data
        l_min, l_max = 1e-5, 1e5
        n_grid, n_refine = 50, 20

        # Full-Uab reference, via NumPy
        with _numpy_only():
            full_result = _compute_lrt_numpy(
                d["n_cvt"],
                d["eigenvalues"],
                d["Uab_batch"],
                l_min,
                l_max,
                n_grid,
                n_refine,
                d["logl_H0"],
                n_threads=1,
            )

        general_result = _general_lrt_only_result(d, l_min, l_max, n_grid, n_refine)

        np.testing.assert_allclose(
            general_result["lambdas_mle"],
            full_result["lambdas_mle"],
            rtol=5e-5,
            err_msg="LRT-only general workspace lambdas_mle differ from full-Uab",
        )
        np.testing.assert_allclose(
            general_result["p_lrts"],
            full_result["p_lrts"],
            rtol=5e-3,
            err_msg="LRT-only general workspace p_lrts differ from full-Uab",
        )


@pytest.fixture(scope="module")
def degenerate_data(mouse_data):
    """Extend mouse_data with constant-genotype (degenerate) SNP columns.

    Injects 3 constant columns (all-0, all-1, all-2) into the genotype
    matrix, rebuilds Uab. Degenerate SNPs have zero variance after covariate
    projection (P_xx <= 0), so beta/se/p-values must be NaN.
    """
    d = mouse_data
    n_samples = d["n_samples"]

    # Load the original rotated genotypes for the first 10 well-conditioned SNPs
    plink_data = load_plink_binary(MOUSE.bfile)
    genotypes = plink_data.genotypes
    K = read_kinship_matrix(MOUSE.kinship)
    eigenvalues, U = np.linalg.eigh(K)

    geno_subset = genotypes[:, :10].astype(np.float64)
    col_means = np.nanmean(geno_subset, axis=0)
    missing = np.isnan(geno_subset)
    if missing.any():
        geno_subset = np.where(missing, col_means[None, :], geno_subset)

    # Inject degenerate columns: all-zero genotype produces P_xx=0 exactly
    # (zero vector after eigenrotation). Also add a near-zero-variance column.
    const_cols = np.column_stack(
        [
            np.zeros(n_samples),  # zero vector → P_xx = 0 → NaN
            np.zeros(n_samples),  # second zero column for robustness
            np.zeros(n_samples),  # third zero column
        ]
    )
    geno_with_degen = np.column_stack([geno_subset, const_cols])

    rng = np.random.default_rng(42)
    W = np.column_stack([np.ones(n_samples), rng.standard_normal(n_samples)])
    UtW = U.T @ W
    Uty = U.T @ (
        K @ np.random.default_rng(42).standard_normal(n_samples) * 0.5
        + np.random.default_rng(42).standard_normal(n_samples)
    )
    UtG = U.T @ geno_with_degen
    utg_t = np.ascontiguousarray(UtG.T)

    n_cvt = 2
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG.T)
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty, n_cvt=n_cvt)
    lambda_null_mle, logl_H0 = compute_null_model_mle(eigenvalues, UtW, Uty, n_cvt)
    Hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues + 1.0)

    return {
        "eigenvalues": eigenvalues,
        "n_samples": n_samples,
        "n_cvt": n_cvt,
        "Uab_batch": Uab_batch,
        "UtW": UtW,
        "Uty": Uty,
        "utg_t": utg_t,
        "uab_inv_soa": uab_inv_soa,
        "Hi_eval_null": Hi_eval_null,
        "logl_H0": logl_H0,
        "n_normal": 10,  # first 10 are well-conditioned
        "n_degenerate": 3,  # last 3 are constant
    }


@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension unavailable")
class TestDegenerateSplitParity:
    """The general workspace and the full-Uab path agree on NaN for degenerate SNPs."""

    def test_score_degenerate_nan_parity(self, degenerate_data):
        """Score-only general workspace produces NaN for constant-genotype SNPs."""
        d = degenerate_data
        with _numpy_only():
            full_result = _compute_score_numpy(
                d["n_cvt"],
                d["eigenvalues"],
                d["Hi_eval_null"],
                d["Uab_batch"],
                d["n_samples"],
                n_threads=1,
            )
        general_result = _general_score_only_result(d)

        # Well-conditioned SNPs must match
        n = d["n_normal"]
        np.testing.assert_allclose(
            general_result["p_scores"][:n],
            full_result["p_scores"][:n],
            rtol=_C_VS_NUMPY_RTOL,
            err_msg="Score normal SNPs: general workspace vs full-Uab mismatch",
        )

        # Degenerate SNPs must be NaN in both paths
        assert np.all(np.isnan(full_result["p_scores"][n:])), (
            "Batch Score should return NaN for constant-genotype SNPs"
        )
        assert np.all(np.isnan(general_result["p_scores"][n:])), (
            "General workspace Score should return NaN for constant-genotype SNPs"
        )

    def test_lrt_degenerate_parity(self, degenerate_data):
        """LRT-only general workspace matches full-Uab for degenerate SNPs (p≈1)."""
        d = degenerate_data
        l_min, l_max = 1e-5, 1e5
        n_grid, n_refine = 50, 20

        with _numpy_only():
            full_result = _compute_lrt_numpy(
                d["n_cvt"],
                d["eigenvalues"],
                d["Uab_batch"],
                l_min,
                l_max,
                n_grid,
                n_refine,
                d["logl_H0"],
                n_threads=1,
            )
        general_result = _general_lrt_only_result(d, l_min, l_max, n_grid, n_refine)

        # Well-conditioned SNPs must match
        n = d["n_normal"]
        np.testing.assert_allclose(
            general_result["p_lrts"][:n],
            full_result["p_lrts"][:n],
            rtol=5e-3,
            err_msg="LRT normal SNPs: general workspace vs batch mismatch",
        )

        # Degenerate SNPs: LRT finds no signal (LR≈0, p≈1). Both paths
        # must agree — NaN pattern and finite values must match exactly.
        np.testing.assert_allclose(
            general_result["p_lrts"][n:],
            full_result["p_lrts"][n:],
            rtol=5e-3,
            equal_nan=True,
            err_msg="LRT degenerate SNPs: general workspace vs batch mismatch",
        )
        np.testing.assert_allclose(
            general_result["lambdas_mle"][n:],
            full_result["lambdas_mle"][n:],
            rtol=5e-5,
            equal_nan=True,
            err_msg="LRT degenerate lambdas_mle: general workspace vs batch mismatch",
        )


# ---------------------------------------------------------------------------
# Plan 53-03 Task 2: _compute_wald_numpy dispatch tests
# ---------------------------------------------------------------------------


@pytest.fixture
def compute_wald_data():
    """Synthetic data for _compute_wald_numpy dispatch tests.

    Returns:
        (eigenvalues, Uab_batch, n_samples) with n_samples=80, n_snps=30.
    """
    n_samples = 80
    d = rotated_lmm_inputs(n_samples, 30, seed=123)
    eigenvalues, UtW, Uty, UtG = d.eigenvalues, d.UtW, d.Uty, d.UtG

    from jamma.lmm.uab import batch_compute_uab_numpy

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    return eigenvalues, Uab_batch, n_samples


@pytest.mark.tier0
def test_compute_wald_numpy_dispatches_split_ncvt1(compute_wald_data):
    """_compute_wald_numpy with n_cvt=1 (C ext disabled) calls split optimizer."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy as cn
    from jamma.lmm.likelihood_numpy import (
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )

    eigenvalues, Uab_batch, n_samples = compute_wald_data

    call_log = []
    real_split_fn = golden_section_optimize_lambda_split_ncvt1_numpy

    def spy_split(*args, **kwargs):
        call_log.append("split")
        return real_split_fn(*args, **kwargs)

    split_generic_log = []
    real_generic_fn = cn.golden_section_optimize_lambda_numpy

    def spy_generic(*args, **kwargs):
        split_generic_log.append("generic")
        return real_generic_fn(*args, **kwargs)

    # allow-patch: dispatch spy. Which optimiser _compute_wald_numpy selects
    # for n_cvt is the contract; both spies forward to the real function.
    with (
        patch.object(cn, "_accel", None),
        patch.object(cn, "golden_section_optimize_lambda_split_ncvt1_numpy", spy_split),
        patch.object(cn, "golden_section_optimize_lambda_numpy", spy_generic),
    ):
        cn._compute_wald_numpy(1, eigenvalues, Uab_batch, n_samples, 1e-5, 1e5, 50, 20)

    assert len(call_log) == 1, (
        f"Split optimizer called {len(call_log)} times for n_cvt=1, expected 1"
    )
    assert len(split_generic_log) == 0, (
        "Generic optimizer should NOT be called for n_cvt=1 Python path"
    )

    # Also verify n_cvt=2 uses generic, not split
    n_samples2 = 80
    d2 = rotated_lmm_inputs(n_samples2, 10, n_cvt=2, seed=456)
    eigenvalues2, UtW2, Uty2, UtG2 = d2.eigenvalues, d2.UtW, d2.Uty, d2.UtG
    from jamma.lmm.uab import batch_compute_uab_numpy

    Uab_batch2 = batch_compute_uab_numpy(2, UtW2, Uty2, UtG2.T)

    call_log2 = []
    generic_log2 = []

    def spy_split2(*args, **kwargs):
        call_log2.append("split")
        return real_split_fn(*args, **kwargs)

    def spy_generic2(*args, **kwargs):
        generic_log2.append("generic")
        return real_generic_fn(*args, **kwargs)

    # allow-patch: dispatch spy, as above.
    with (
        patch.object(cn, "_accel", None),
        patch.object(
            cn, "golden_section_optimize_lambda_split_ncvt1_numpy", spy_split2
        ),
        patch.object(cn, "golden_section_optimize_lambda_numpy", spy_generic2),
    ):
        cn._compute_wald_numpy(
            2, eigenvalues2, Uab_batch2, n_samples2, 1e-5, 1e5, 50, 20
        )

    assert len(call_log2) == 0, "Split should NOT be called for n_cvt=2"
    assert len(generic_log2) == 1, "Generic should be called exactly once for n_cvt=2"


@pytest.mark.tier0
def test_compute_wald_numpy_split_matches_generic(compute_wald_data):
    """split path (n_cvt=1) in _compute_wald_numpy produces same results as generic."""
    from unittest.mock import patch

    from jamma.lmm import compute_numpy as cn
    from jamma.lmm.uab import batch_compute_iab_numpy

    eigenvalues, Uab_batch, n_samples = compute_wald_data
    n_cvt = 1
    Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)

    # Split path (n_cvt=1 Python branch)
    with patch.object(cn, "_accel", None):
        result_split = cn._compute_wald_numpy(
            n_cvt, eigenvalues, Uab_batch, n_samples, 1e-5, 1e5, 50, 20
        )

    # Generic path: bypass n_cvt==1 branch by calling generic optimizer directly
    import jamma.lmm.likelihood_numpy as ln
    from jamma.lmm import stats

    lambdas_gen, logls_gen, Pab_gen = ln.golden_section_optimize_lambda_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_iter=20,
    )
    betas_gen, ses_gen, pwalds_gen = stats.batch_calc_wald_stats_from_pab_numpy(
        n_cvt, Pab_gen, n_samples
    )

    np.testing.assert_allclose(
        result_split["lambdas"],
        lambdas_gen,
        rtol=1e-10,
        err_msg="lambdas: split path vs generic path",
    )
    np.testing.assert_allclose(
        result_split["betas"],
        betas_gen,
        rtol=1e-12,
        err_msg="betas: split path vs generic path",
    )
    np.testing.assert_allclose(
        result_split["ses"],
        ses_gen,
        rtol=1e-12,
        err_msg="ses: split path vs generic path",
    )
    np.testing.assert_allclose(
        result_split["pwalds"],
        pwalds_gen,
        rtol=1e-12,
        err_msg="pwalds: split path vs generic path",
    )


# ---------------------------------------------------------------------------
# Score/LRT C dispatch via general C path for n_cvt > 1 (Plan 70-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.parametrize("n_cvt", [2, 3, 4])
def test_vectorized_general_uab_parity(n_cvt):
    """Vectorized _batch_compute_uab_general_numpy matches reference per-SNP loop."""
    from jamma.lmm.likelihood import build_index_table
    from jamma.lmm.uab import _batch_compute_uab_general_numpy

    rng = np.random.default_rng(99)
    n_samples, n_snps = 60, 15
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Reference: per-SNP loop (the old implementation)
    table = build_index_table(n_cvt)
    n_index = table.n_index
    Uab_ref = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    vectors_base = np.column_stack([UtW, np.zeros(n_samples), Uty])
    for snp_idx in range(n_snps):
        vectors = vectors_base.copy()
        vectors[:, n_cvt] = UtG[:, snp_idx]
        for a_col, b_col, idx in table.uab_pairs:
            Uab_ref[snp_idx, :, idx] = vectors[:, a_col] * vectors[:, b_col]

    # Vectorized implementation
    Uab_vec = _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG.T)

    np.testing.assert_allclose(
        Uab_vec,
        Uab_ref,
        rtol=1e-14,
        atol=1e-14,
        err_msg=f"Vectorized general Uab (n_cvt={n_cvt}) does not match per-SNP loop",
    )


@pytest.mark.tier0
@pytest.mark.parametrize("n_cvt", [2, 3, 4])
def test_invariant_columns_constant_across_snps(n_cvt):
    """Uab columns classified as invariant are actually constant across SNPs."""
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.uab import _batch_compute_uab_general_numpy

    rng = np.random.default_rng(123)
    n_samples, n_snps = 40, 20
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    inv_indices, _var_indices = classify_uab_columns(n_cvt)
    Uab = _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG.T)

    for col_idx in inv_indices:
        first_snp = Uab[0, :, col_idx]
        for snp_i in range(1, n_snps):
            np.testing.assert_array_equal(
                Uab[snp_i, :, col_idx],
                first_snp,
                err_msg=(
                    f"Invariant column {col_idx} differs at SNP {snp_i} (n_cvt={n_cvt})"
                ),
            )


@pytest.mark.tier0
@pytest.mark.parametrize("n_cvt", [2, 3, 4])
def test_batch_compute_uab_varying_soa_rejects_ncvt_above_one(n_cvt):
    """batch_compute_uab_varying_soa_numpy is n_cvt=1 only.

    No production path builds this for n_cvt>1: the general dispatch path
    (``DispatchPath.FUSED_GENERAL``) forms its varying columns on the fly
    inside the C workspace instead.
    """
    rng = np.random.default_rng(55)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    with pytest.raises(ValueError, match="n_cvt must be 1"):
        batch_compute_uab_varying_soa_numpy(n_cvt, UtW, Uty, UtG.T)


@pytest.mark.tier0
def test_batch_compute_uab_numpy_rejects_wrong_layout():
    """batch_compute_uab_numpy raises ValueError when given (n_samples, n_snps)."""
    rng = np.random.default_rng(99)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))  # wrong layout for this fn

    with pytest.raises(ValueError, match="Pass \\(n_snps, n_samples\\)"):
        batch_compute_uab_numpy(1, UtW, Uty, UtG)


@pytest.mark.tier0
def test_batch_compute_uab_varying_soa_rejects_wrong_out_shape():
    """batch_compute_uab_varying_soa_numpy raises ValueError for wrong out= shape."""
    rng = np.random.default_rng(99)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    utg_t = rng.standard_normal((n_snps, n_samples))
    wrong_out = np.empty((n_snps + 1, 3, n_samples), dtype=np.float64)

    with pytest.raises(ValueError, match="out shape"):
        batch_compute_uab_varying_soa_numpy(1, UtW, Uty, utg_t, out=wrong_out)


@pytest.mark.tier0
def test_batch_compute_uab_varying_soa_ncvt1_rejects_wrong_out_dtype_and_layout():
    """The n_cvt=1 branch validates out= dtype and contiguity like the general one."""
    rng = np.random.default_rng(99)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    utg_t = rng.standard_normal((n_snps, n_samples))

    with pytest.raises(ValueError, match="out dtype"):
        batch_compute_uab_varying_soa_numpy(
            1, UtW, Uty, utg_t, out=np.empty((n_snps, 3, n_samples), dtype=np.float32)
        )
    fortran_out = np.asfortranarray(np.empty((n_snps, 3, n_samples), dtype=np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        batch_compute_uab_varying_soa_numpy(1, UtW, Uty, utg_t, out=fortran_out)


@pytest.mark.tier0
def test_batch_compute_uab_varying_soa_rejects_wrong_layout():
    """batch_compute_uab_varying_soa_numpy raises ValueError when given old layout."""
    rng = np.random.default_rng(99)
    n_samples, n_snps = 50, 10
    UtW = rng.standard_normal((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))  # wrong layout for this fn

    with pytest.raises(ValueError, match="Pass \\(n_snps, n_samples\\)"):
        batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG)


# ---------------------------------------------------------------------------
# Mode dispatch for compute_lmm_chunk_numpy
# ---------------------------------------------------------------------------


@pytest.fixture
def chunk_dispatch_data():
    """Small synthetic dataset for compute_lmm_chunk_numpy dispatch tests.

    Returns:
        (eigenvalues, UtW, Uty, UtG) with n_samples=50, n_snps=10.
    """
    d = rotated_lmm_inputs(50, 10, seed=42)
    return d.eigenvalues, d.UtW, d.Uty, d.UtG


def test_compute_lmm_chunk_numpy_all_modes(chunk_dispatch_data, monkeypatch):
    """compute_lmm_chunk_numpy must return non-None expected keys for each mode.

    The extension is cleared because this function is the full-Uab NumPy path,
    and the runner reaches it only on NUMPY_FALLBACK, which is selected only
    when the extension is absent.
    """
    monkeypatch.setattr(compute_numpy, "_accel", None)

    eigenvalues, UtW, Uty, UtG = chunk_dispatch_data
    n_samples = eigenvalues.shape[0]

    lambda_null = 0.1
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)
    logl_H0 = -25.0

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)

    # Mode 1: Wald — expects lambdas, logls, betas, ses, pwalds
    result1 = compute_lmm_chunk_numpy(1, 1, eigenvalues, Uab_batch, n_samples)
    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        assert result1[key] is not None, f"Mode 1: key '{key}' is None"
    assert result1["lambdas_mle"] is None
    assert result1["p_lrts"] is None
    assert result1["p_scores"] is None

    # Mode 2: LRT — expects lambdas_mle, p_lrts
    result2 = compute_lmm_chunk_numpy(
        2, 1, eigenvalues, Uab_batch, n_samples, logl_H0=logl_H0
    )
    for key in ("lambdas_mle", "p_lrts"):
        assert result2[key] is not None, f"Mode 2: key '{key}' is None"
    assert result2["lambdas"] is None
    assert result2["logls"] is None
    assert result2["betas"] is None
    assert result2["ses"] is None
    assert result2["pwalds"] is None
    assert result2["p_scores"] is None

    # Mode 3: Score — expects betas, ses, p_scores
    result3 = compute_lmm_chunk_numpy(
        3, 1, eigenvalues, Uab_batch, n_samples, Hi_eval_null=Hi_eval_null
    )
    for key in ("betas", "ses", "p_scores"):
        assert result3[key] is not None, f"Mode 3: key '{key}' is None"
    assert result3["lambdas"] is None
    assert result3["logls"] is None
    assert result3["pwalds"] is None
    assert result3["lambdas_mle"] is None
    assert result3["p_lrts"] is None

    # Mode 4: All — all keys non-None
    result4 = compute_lmm_chunk_numpy(
        4,
        1,
        eigenvalues,
        Uab_batch,
        n_samples,
        Hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
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
        assert result4[key] is not None, f"Mode 4: key '{key}' is None"


def test_compute_lmm_chunk_numpy_missing_args_raise(chunk_dispatch_data):
    """compute_lmm_chunk_numpy must raise ValueError when required args are absent."""
    eigenvalues, UtW, Uty, UtG = chunk_dispatch_data
    n_samples = eigenvalues.shape[0]
    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)

    with pytest.raises(ValueError, match="logl_H0 is required"):
        compute_lmm_chunk_numpy(2, 1, eigenvalues, Uab_batch, n_samples)

    with pytest.raises(ValueError, match="Hi_eval_null is required"):
        compute_lmm_chunk_numpy(3, 1, eigenvalues, Uab_batch, n_samples)

    # Mode 4 (All) requires both logl_H0 and Hi_eval_null.
    # Missing logl_H0 is checked first (line order in source).
    with pytest.raises(ValueError, match="logl_H0 is required"):
        compute_lmm_chunk_numpy(4, 1, eigenvalues, Uab_batch, n_samples)

    # Providing logl_H0 but omitting Hi_eval_null also raises.
    with pytest.raises(ValueError, match="Hi_eval_null is required"):
        compute_lmm_chunk_numpy(4, 1, eigenvalues, Uab_batch, n_samples, logl_H0=-50.0)
