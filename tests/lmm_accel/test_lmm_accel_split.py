"""_lmm_accel C extension tests: SoA split Uab/Iab kernels and the persistent workspace.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest

from jamma.lmm import compute_numpy
from jamma.lmm.compute_numpy import (
    _compute_wald_numpy,
    _compute_wald_split_c,
    compute_wald_split_c_ws,
    create_lmm_workspace,
)
from jamma.lmm.likelihood_numpy import (
    batch_compute_iab_numpy,
    batch_compute_iab_split_ncvt1,
    batch_compute_iab_split_ncvt1_soa,
    batch_compute_uab_split_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)
from jamma.lmm.schema import LmmConfig


@pytest.mark.tier0
def test_split_uab_matches_full_uab(split_wald_data):
    """Split Uab construction matches the full 6-column Uab."""
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    _, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    uab_var, uab_inv = batch_compute_uab_split_numpy(1, UtW, Uty, UtG)

    # Varying columns: wx(1), xx(3), xy(4) in full -> 0,1,2 in split
    np.testing.assert_allclose(
        uab_var[:, :, 0],
        full_uab[:, :, 1],
        rtol=1e-14,
        err_msg="wx column mismatch",
    )
    np.testing.assert_allclose(
        uab_var[:, :, 1],
        full_uab[:, :, 3],
        rtol=1e-14,
        err_msg="xx column mismatch",
    )
    np.testing.assert_allclose(
        uab_var[:, :, 2],
        full_uab[:, :, 4],
        rtol=1e-14,
        err_msg="xy column mismatch",
    )

    # Invariant columns: ww(0), wy(2), yy(5) in full -> 0,1,2 in inv
    # These should be identical across all SNPs in full_uab
    np.testing.assert_allclose(
        uab_inv[:, 0],
        full_uab[0, :, 0],
        rtol=1e-14,
        err_msg="ww column mismatch",
    )
    np.testing.assert_allclose(
        uab_inv[:, 1],
        full_uab[0, :, 2],
        rtol=1e-14,
        err_msg="wy column mismatch",
    )
    np.testing.assert_allclose(
        uab_inv[:, 2],
        full_uab[0, :, 5],
        rtol=1e-14,
        err_msg="yy column mismatch",
    )


@pytest.mark.tier0
def test_split_iab_matches_full_iab(split_wald_data):
    """Split Iab construction matches the full Iab."""
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    _, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    full_iab = batch_compute_iab_numpy(1, full_uab)

    uab_var, uab_inv = batch_compute_uab_split_numpy(1, UtW, Uty, UtG)
    split_iab = batch_compute_iab_split_ncvt1(uab_var, uab_inv)

    np.testing.assert_allclose(
        split_iab,
        full_iab,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Split Iab does not match full Iab",
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
def test_split_c_vs_full_c_parity(split_wald_data):
    """Split C extension matches full C extension within FP tolerance."""
    from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    # Full path
    full_uab = batch_compute_uab_numpy(1, UtW, Uty, UtG)
    full_iab = batch_compute_iab_numpy(1, full_uab)
    result_full = _compute_wald_numpy(
        n_cvt=1,
        eigenvalues=eigenvalues,
        Uab_batch=full_uab,
        n_samples=n_samples,
        l_min=1e-5,
        l_max=1e5,
        n_grid=50,
        n_refine=20,
        Iab_batch=full_iab,
        n_threads=1,
    )

    # Split path — use SoA layout (no per-call transpose since Task 1 changes)
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    split_iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)
    result_split = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        split_iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            result_split[key],
            result_full[key],
            rtol=1e-9,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: split vs full C mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
def test_split_c_degenerate_snps():
    """All-degenerate batch via split path produces all-NaN."""
    rng = np.random.default_rng(13)
    n_samples, n_snps = 50, 4
    eigenvalues = np.sort(rng.uniform(0.5, 1.5, n_samples))

    # Build SoA split arrays with xx=0 (degenerate)
    # SoA layout: (n_snps, 3, n_samples) — axis-1 rows [wx, xx, xy]
    uab_var_soa = rng.standard_normal((n_snps, 3, n_samples))
    uab_var_soa[:, 1, :] = 0.0  # xx row = 0 (row index 1 in SoA)
    uab_inv_soa = np.abs(rng.standard_normal((3, n_samples))) + 0.1
    iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)

    result = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )

    assert np.all(np.isnan(result["betas"])), "Expected all-NaN betas"
    assert np.all(np.isnan(result["ses"])), "Expected all-NaN ses"
    assert np.all(np.isnan(result["pwalds"])), "Expected all-NaN pwalds"


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
def test_split_c_multithreaded_parity(split_wald_data):
    """Multi-threaded split C matches single-threaded split C."""
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)

    r1 = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )
    rn = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        n_threads,
    )

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            rn[key],
            r1[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: MT vs ST split mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"]
)
def test_split_c_nonfinite_eigenvalues(bad_value):
    """Non-finite eigenvalues (NaN, Inf, -Inf) are rejected by the split C path."""
    rng = np.random.default_rng(11)
    n_samples, n_snps = 50, 3
    eigenvalues = rng.uniform(0.1, 2.0, n_samples)
    eigenvalues[10] = bad_value

    # SoA layout: (n_snps, 3, n_samples) for varying, (3, n_samples) for invariant
    uab_var_soa = rng.standard_normal((n_snps, 3, n_samples))
    uab_inv_soa = np.abs(rng.standard_normal((3, n_samples))) + 0.1
    iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)

    with pytest.raises(ValueError, match=r"eigenvalues.*not finite"):
        _compute_wald_split_c(
            eigenvalues,
            uab_var_soa,
            uab_inv_soa,
            iab,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
def test_workspace_api_matches_legacy_split(split_wald_data):
    """Workspace API (create + chunk) produces identical results to legacy split_c.

    Verifies that the per-run workspace path gives the same numerical output
    as the per-call _compute_wald_split_c (which uses Iab_batch). Both paths
    share the same golden section core — differences would indicate a bug in
    the internal Iab/logdet_iab computation.
    """
    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    iab = batch_compute_iab_split_ncvt1_soa(uab_var_soa, uab_inv_soa)

    # Legacy path (with Iab_batch passed explicitly)
    result_legacy = _compute_wald_split_c(
        eigenvalues,
        uab_var_soa,
        uab_inv_soa,
        iab,
        n_samples,
        1e-5,
        1e5,
        50,
        20,
        1,
    )

    # Workspace path (Iab computed internally from raw column sums)
    ws = create_lmm_workspace(eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1)
    result_ws = compute_wald_split_c_ws(ws, uab_var_soa, 1)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            result_ws[key],
            result_legacy[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: workspace vs legacy split mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
def test_workspace_reuse_across_chunks(split_wald_data):
    """Workspace created once can be reused across multiple chunk calls.

    Simulates the runner's cross-chunk reuse pattern: same workspace, different
    uab_varying_soa slices. Results must match per-call legacy path.
    """
    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data

    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

    # Create workspace once (before "chunk loop")
    ws = create_lmm_workspace(eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1)

    # Simulate two chunks by splitting the SNPs in half
    mid = n_snps // 2
    chunk1 = uab_var_soa[:mid]
    chunk2 = uab_var_soa[mid:]

    result_c1 = compute_wald_split_c_ws(ws, chunk1, 1)
    result_c2 = compute_wald_split_c_ws(ws, chunk2, 1)

    # Concatenate chunk results
    combined_lambdas = np.concatenate([result_c1["lambdas"], result_c2["lambdas"]])
    combined_betas = np.concatenate([result_c1["betas"], result_c2["betas"]])

    # Reference: single call with all SNPs
    result_full = compute_wald_split_c_ws(ws, uab_var_soa, 1)

    np.testing.assert_allclose(
        combined_lambdas,
        result_full["lambdas"],
        rtol=1e-12,
        atol=1e-14,
        err_msg="Chunked lambda mismatch vs single call",
    )
    np.testing.assert_allclose(
        combined_betas,
        result_full["betas"],
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
        err_msg="Chunked beta mismatch vs single call",
    )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
def test_workspace_multithreaded_parity(split_wald_data):
    """Workspace path: multi-threaded results match single-threaded results."""
    from jamma.core.threading import get_physical_core_count

    n_threads = get_physical_core_count()
    if n_threads < 2:
        pytest.skip("Need >=2 cores for multi-threaded test")

    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

    ws = create_lmm_workspace(eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1)
    r1 = compute_wald_split_c_ws(ws, uab_var_soa, 1)
    rn = compute_wald_split_c_ws(ws, uab_var_soa, n_threads)

    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        np.testing.assert_allclose(
            rn[key],
            r1[key],
            rtol=1e-10,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"{key}: workspace MT vs ST mismatch",
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
def test_workspace_invalid_inputs(split_wald_data):
    """Workspace creation and chunk compute reject invalid inputs cleanly."""
    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)

    # Wrong invariant shape
    with pytest.raises(ValueError, match="uab_invariant"):
        create_lmm_workspace(
            eigenvalues,
            uab_inv_soa.T,  # wrong shape: (n_samples, 3) instead of (3, n_samples)
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )


@pytest.mark.tier0
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"]
)
def test_workspace_nonfinite_eigenvalues(split_wald_data, bad_value):
    """Workspace creation rejects non-finite eigenvalues."""
    eigenvalues, UtW, Uty, UtG, n_samples, n_snps = split_wald_data
    uab_inv_soa = compute_uab_invariant_soa(UtW, Uty)
    bad_evals = eigenvalues.copy()
    bad_evals[0] = bad_value
    with pytest.raises(ValueError, match=r"eigenvalues.*not finite"):
        create_lmm_workspace(
            bad_evals,
            uab_inv_soa,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )

    # Wrong uab_varying shape for chunk compute
    ws = create_lmm_workspace(eigenvalues, uab_inv_soa, n_samples, 1e-5, 1e5, 50, 20, 1)
    uab_var_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    with pytest.raises(ValueError, match="uab_varying"):
        compute_wald_split_c_ws(ws, uab_var_soa.transpose(0, 2, 1), 1)


@pytest.mark.tier1
@pytest.mark.skipif(
    compute_numpy._accel is None, reason="Split C extension unavailable"
)
def test_pipeline_multi_chunk_correctness():
    """Pipeline path (multi-chunk) produces identical results to sequential path.

    Forces multi-chunk processing by using enough SNPs to exceed chunk_size,
    then compares pipeline results against sequential (non-pipeline) results.
    This catches off-by-one errors in the last-chunk handling, race conditions
    in buffer management, and write_offset accumulation bugs.
    """
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy

    rng = np.random.default_rng(42)
    n_samples = 100
    # Use enough SNPs that we get at least 3 chunks
    chunk_size = compute_chunk_size_numpy(
        n_samples,
        1000,
        n_cvt=1,
        mem_budget_bytes=int(2e9),
    )
    n_snps = chunk_size * 3 + 17  # non-aligned to catch last-chunk bugs
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))

    # Build realistic data: genotype matrix + covariates + phenotype
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.4, 0.4, 0.2])
    phenotypes = rng.standard_normal(n_samples)
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]

    # Stub out kinship — use pre-computed eigendecomposition
    U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]

    from jamma.lmm.runner_numpy import run_lmm_association_numpy

    # Run with pipeline enabled (multi-chunk, split C extension)
    run_result = run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=None,
        snp_info=snp_info,
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
    results_pipeline = run_result.associations

    # Verify we got results for all SNPs (none filtered at maf=0)
    # Some may be filtered by the internal variance check, but most should pass
    assert len(results_pipeline) > n_snps * 0.8, (
        f"Too many SNPs filtered: got {len(results_pipeline)} of {n_snps}"
    )

    # Run with pipeline disabled: force single chunk by using sequential path
    # We do this by monkeypatching the canonical dispatch flag to False.
    import jamma.lmm.compute_numpy as compute_mod

    orig_split = compute_mod._accel
    try:
        compute_mod._accel = None
        run_result = run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=None,
            snp_info=snp_info,
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
        results_sequential = run_result.associations
    finally:
        compute_mod._accel = orig_split

    # Same number of results
    assert len(results_pipeline) == len(results_sequential), (
        f"Pipeline: {len(results_pipeline)}, Sequential: {len(results_sequential)}"
    )

    # Compare numerical outputs
    for r_pipe, r_seq in zip(results_pipeline, results_sequential, strict=True):
        assert r_pipe.rs == r_seq.rs, f"SNP order mismatch: {r_pipe.rs} vs {r_seq.rs}"
        if r_pipe.p_wald is not None and r_seq.p_wald is not None:
            np.testing.assert_allclose(
                r_pipe.beta,
                r_seq.beta,
                rtol=1e-9,
                atol=1e-14,
                err_msg=f"beta mismatch for {r_pipe.rs}",
            )
            np.testing.assert_allclose(
                r_pipe.se,
                r_seq.se,
                rtol=1e-9,
                atol=1e-14,
                err_msg=f"se mismatch for {r_pipe.rs}",
            )
            np.testing.assert_allclose(
                r_pipe.p_wald,
                r_seq.p_wald,
                rtol=1e-8,
                atol=1e-14,
                err_msg=f"p_wald mismatch for {r_pipe.rs}",
            )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension not compiled")
def test_workspace_alignment():
    """Verify alloc_aligned_doubles returns 32-byte-aligned addresses."""
    from jamma.lmm._lmm_accel import _get_aligned_alloc_test_ptr

    # Test boundary sizes (n=1 minimum, n=4 exact 32-byte boundary) and larger
    for n in [1, 4, 100, 101, 200, 1400, 50001]:
        ptr = _get_aligned_alloc_test_ptr(n)
        assert ptr % 32 == 0, (
            f"alloc_aligned_doubles({n}) returned {ptr:#x}, not 32-byte aligned"
        )
