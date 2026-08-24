"""_lmm_accel C extension tests: SoA split Uab/Iab construction, and workspace guards.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.

The SoA Uab and Iab construction tested at the top is pure NumPy and still feeds
the live kernels. The C kernels this module used to drive, compute_lmm_batch_split_c
and the create_workspace_split_c workspace, are not reachable from any
DispatchPath, and the fused workspace has taken their place. Their parity,
degenerate-SNP and thread-determinism checks are covered on the fused kernel in
test_lmm_accel_fused.py. What did not exist there, and is kept here, is the input
validation: non-finite eigenvalues and wrong array shapes.
"""

import numpy as np
import pytest

from jamma.lmm import compute_numpy
from jamma.lmm.compute_numpy import (
    compute_wald_fused_c_ws,
    create_lmm_workspace_fused,
)
from jamma.lmm.likelihood_numpy import (
    batch_compute_iab_numpy,
    batch_compute_iab_split_ncvt1,
    batch_compute_uab_split_numpy,
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


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension unavailable")
@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"]
)
def test_fused_workspace_rejects_nonfinite_eigenvalues(fused_data, bad_value):
    """Workspace creation rejects NaN, Inf and -Inf eigenvalues."""
    eigenvalues, w, Uty, _, uab_inv_soa, _, n_samples = fused_data

    bad_evals = eigenvalues.copy()
    bad_evals[0] = bad_value

    with pytest.raises(ValueError, match=r"eigenvalues.*not finite"):
        create_lmm_workspace_fused(
            bad_evals, uab_inv_soa, w, Uty, n_samples, 1e-5, 1e5, 50, 20, 1
        )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension unavailable")
def test_fused_workspace_rejects_wrong_invariant_shape(fused_data):
    """Workspace creation rejects a transposed invariant SoA."""
    eigenvalues, w, Uty, _, uab_inv_soa, _, n_samples = fused_data

    with pytest.raises(ValueError, match="uab_invariant"):
        create_lmm_workspace_fused(
            eigenvalues,
            uab_inv_soa.T,
            w,
            Uty,
            n_samples,
            1e-5,
            1e5,
            50,
            20,
            1,
        )


@pytest.mark.tier0
@pytest.mark.skipif(compute_numpy._accel is None, reason="C extension unavailable")
def test_fused_workspace_reuse_across_chunks(fused_data):
    """One workspace reused across two chunks matches a single call over both.

    This is the runner's pattern: the workspace is built once before the chunk
    loop and fed successive genotype slices.
    """
    eigenvalues, w, Uty, utg_t, uab_inv_soa, _, n_samples = fused_data

    ws = create_lmm_workspace_fused(
        eigenvalues, uab_inv_soa, w, Uty, n_samples, 1e-5, 1e5, 50, 20, 1
    )

    mid = utg_t.shape[0] // 2
    first = compute_wald_fused_c_ws(ws, utg_t[:mid], 1)
    second = compute_wald_fused_c_ws(ws, utg_t[mid:], 1)
    full = compute_wald_fused_c_ws(ws, utg_t, 1)

    for key in ("lambdas", "betas"):
        np.testing.assert_allclose(
            np.concatenate([first[key], second[key]]),
            full[key],
            rtol=1e-12,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"chunked {key} differs from the single call",
        )
