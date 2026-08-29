"""_lmm_accel C extension tests: workspace guards and alignment.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.

The C kernels this module used to drive, compute_lmm_batch_split_c
and the create_workspace_split_c workspace, are not reachable from any
DispatchPath, and the fused workspace has taken their place. Their parity,
degenerate-SNP and thread-determinism checks are covered on the fused kernel in
test_lmm_accel_fused.py, and their input validation in test_lmm_accel_core.py.
What is kept here is the workspace reuse pattern the runner relies on.
"""

import numpy as np
import pytest

from jamma.lmm import accel
from jamma.lmm.schema import LmmConfig
from tests.conftest import requires_c


@pytest.mark.tier1
@requires_c
def test_pipeline_multi_chunk_correctness(monkeypatch):
    """Pipeline path (multi-chunk) produces identical results to sequential path.

    Forces multi-chunk processing by using enough SNPs to exceed chunk_size,
    then compares pipeline results against sequential (non-pipeline) results.
    This catches off-by-one errors in the last-chunk handling, race conditions
    in buffer management, and write_offset accumulation bugs.
    """
    from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
    from jamma.lmm.dispatch import DispatchPath

    rng = np.random.default_rng(42)
    n_samples = 100
    # Use enough SNPs that we get at least 3 chunks
    chunk_size = compute_chunk_size_numpy(
        n_samples,
        1000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
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
    monkeypatch.setattr(accel, "_accel", None)
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
@requires_c
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
@requires_c
def test_fused_workspace_reuse_across_chunks(fused_data):
    """One workspace reused across two chunks matches a single call over both.

    This is the runner's pattern: the workspace is built once before the chunk
    loop and fed successive genotype slices.
    """
    eigenvalues, w, Uty, utg_t, uab_inv_soa, _, n_samples = fused_data

    ws = accel.require().create_workspace_ncvt1_c(
        eigenvalues, uab_inv_soa, w, Uty, n_samples, 1e-5, 1e5, 50, 20, lmm_mode=1
    )

    mid = utg_t.shape[0] // 2
    first = accel.require().compute_lmm_chunk_ncvt1_c(ws, utg_t[:mid], 1)
    second = accel.require().compute_lmm_chunk_ncvt1_c(ws, utg_t[mid:], 1)
    full = accel.require().compute_lmm_chunk_ncvt1_c(ws, utg_t, 1)

    for key in ("lambdas", "betas"):
        np.testing.assert_allclose(
            np.concatenate([first[key], second[key]]),
            full[key],
            rtol=1e-12,
            atol=1e-14,
            equal_nan=True,
            err_msg=f"chunked {key} differs from the single call",
        )
