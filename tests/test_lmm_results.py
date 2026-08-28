"""What a run reports: lambda diagnostics, streamed output, and failure messages.

These read the result side of the runner rather than the compute side. The
boundary diagnostics count lambdas that converged at l_min or l_max, the
streaming tests check that an ``output_path`` run writes the same numbers it
would have returned, and the error tests check that a failing chunk names its
own kernel and offset.
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.lmm import compute_numpy
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from jamma.lmm.schema import LmmConfig, LmmMode
from jamma.validation import load_gemma_assoc
from tests.conftest import make_runner_synthetic_data

# ---------------------------------------------------------------------------
# Lambda boundary diagnostic tests (REGR-03)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestLambdaBoundaryDiagnostics:
    """Unit tests for count_lambda_boundary_hits and log_lambda_boundary_warning.

    Verifies that flat-optima SNPs with lambda converging at l_min or l_max
    are correctly counted, and that the boundary warning logger path does not crash.
    """

    def test_mode1_lower_bound_count(self):
        """Mode 1 (Wald): count 3 lambdas at l_min, 0 at l_max."""
        from jamma.lmm.results import count_lambda_boundary_hits

        arrays = {"lambdas": np.array([1e-5, 1e-5, 0.5, 1e-5, 2.0])}
        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode=1, arrays=arrays, l_min=1e-5, l_max=1e5
        )
        assert n_at_lmin == 3
        assert n_at_lmax == 0

    def test_mode2_upper_bound_count(self):
        """Mode 2 (LRT): count 1 at l_min, 2 at l_max using lambdas_mle."""
        from jamma.lmm.results import count_lambda_boundary_hits

        arrays = {"lambdas_mle": np.array([1e-5, 1e5, 1e5])}
        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode=2, arrays=arrays, l_min=1e-5, l_max=1e5
        )
        assert n_at_lmin == 1
        assert n_at_lmax == 2

    def test_mode4_combines_reml_and_mle(self):
        """Mode 4 (All): counts from both lambdas (REML) and lambdas_mle (MLE)."""
        from jamma.lmm.results import count_lambda_boundary_hits

        arrays = {
            "lambdas": np.array([1e-5, 0.5]),
            "lambdas_mle": np.array([1e5, 0.5]),
        }
        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode=4, arrays=arrays, l_min=1e-5, l_max=1e5
        )
        assert n_at_lmin == 1  # one REML lambda at l_min
        assert n_at_lmax == 1  # one MLE lambda at l_max

    def test_empty_array_returns_zeros(self):
        """Empty lambda arrays return (0, 0) without error."""
        from jamma.lmm.results import count_lambda_boundary_hits

        n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
            lmm_mode=1, arrays={"lambdas": np.array([])}, l_min=1e-5, l_max=1e5
        )
        assert n_at_lmin == 0
        assert n_at_lmax == 0

    def test_warning_lower_bound_does_not_crash(self):
        """log_lambda_boundary_warning with lower-bound hits does not raise."""
        from jamma.lmm.results import log_lambda_boundary_warning

        log_lambda_boundary_warning(3, 0, 1e-5, 1e5)  # should not raise

    def test_warning_upper_bound_does_not_crash(self):
        """log_lambda_boundary_warning with upper-bound hits does not raise."""
        from jamma.lmm.results import log_lambda_boundary_warning

        log_lambda_boundary_warning(0, 2, 1e-5, 1e5)  # should not raise

    def test_warning_no_hits_is_noop(self):
        """log_lambda_boundary_warning with zero counts is a no-op."""
        from jamma.lmm.results import log_lambda_boundary_warning

        log_lambda_boundary_warning(0, 0, 1e-5, 1e5)  # should not raise


# ---------------------------------------------------------------------------
# Output path streaming tests (65-03)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.parametrize("lmm_mode", [1, 2, 3, 4], ids=["wald", "lrt", "score", "all"])
def test_output_path_streaming_matches_inmemory(lmm_mode, tmp_path):
    """Streaming via output_path produces identical results to in-memory."""
    genotypes, phenotypes, kinship, snp_info = make_runner_synthetic_data()

    common_kwargs = {
        "genotypes": genotypes,
        "phenotypes": phenotypes,
        "snp_info": snp_info,
        "config": LmmConfig(
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=lmm_mode,
        ),
    }

    # In-memory run
    result_mem = run_lmm_association_numpy(kinship=kinship.copy(), **common_kwargs)

    # Streaming run
    output_file = tmp_path / f"streamed_mode{lmm_mode}.assoc.txt"
    result_disk = run_lmm_association_numpy(
        kinship=kinship.copy(), output_path=output_file, **common_kwargs
    )

    # Streaming result has empty associations but populated metadata
    assert result_disk.associations == [], (
        "Streaming mode should return empty associations"
    )
    assert result_disk.n_tested == len(result_mem.associations), (
        f"n_tested mismatch: {result_disk.n_tested} vs {len(result_mem.associations)}"
    )

    # PVE and PVE SE should match
    assert result_disk.pve is not None
    assert result_mem.pve is not None
    np.testing.assert_allclose(
        result_disk.pve,
        result_mem.pve,
        rtol=1e-10,
        err_msg="PVE mismatch between streaming and in-memory",
    )
    if result_mem.pve_se is not None:
        assert result_disk.pve_se is not None, (
            "in-memory run reported pve_se but streaming run did not"
        )
        np.testing.assert_allclose(
            result_disk.pve_se,
            result_mem.pve_se,
            rtol=1e-10,
            err_msg="PVE SE mismatch between streaming and in-memory",
        )

    # Load streamed file and compare p-values
    assert output_file.exists(), f"Streamed output file not created: {output_file}"
    disk_results = load_gemma_assoc(output_file)
    assert len(disk_results) == len(result_mem.associations), (
        f"Streamed file has {len(disk_results)} SNPs, "
        f"expected {len(result_mem.associations)}"
    )

    # Compare SNP identifiers and p-values.
    # Text serialization loses ~7 digits of precision (%.6g format), so
    # use rtol=1e-6 for file-round-tripped values.
    file_rtol = 1e-6
    for r_mem, r_disk in zip(result_mem.associations, disk_results, strict=True):
        assert r_mem.rs == r_disk.rs, f"SNP order mismatch: {r_mem.rs} vs {r_disk.rs}"
        if lmm_mode in (1, 3, 4):
            np.testing.assert_allclose(
                r_disk.beta,
                r_mem.beta,
                rtol=file_rtol,
                err_msg=f"beta mismatch for {r_mem.rs}",
            )
        if lmm_mode in (1, 4):
            assert r_disk.p_wald is not None, f"p_wald absent on disk for {r_mem.rs}"
            assert r_mem.p_wald is not None, f"p_wald absent in memory for {r_mem.rs}"
            np.testing.assert_allclose(
                r_disk.p_wald,
                r_mem.p_wald,
                rtol=file_rtol,
                err_msg=f"p_wald mismatch for {r_mem.rs}",
            )
        if lmm_mode in (2, 4):
            assert r_disk.p_lrt is not None, f"p_lrt absent on disk for {r_mem.rs}"
            assert r_mem.p_lrt is not None, f"p_lrt absent in memory for {r_mem.rs}"
            np.testing.assert_allclose(
                r_disk.p_lrt,
                r_mem.p_lrt,
                rtol=file_rtol,
                err_msg=f"p_lrt mismatch for {r_mem.rs}",
            )
        if lmm_mode in (3, 4):
            assert r_disk.p_score is not None, f"p_score absent on disk for {r_mem.rs}"
            assert r_mem.p_score is not None, f"p_score absent in memory for {r_mem.rs}"
            np.testing.assert_allclose(
                r_disk.p_score,
                r_mem.p_score,
                rtol=file_rtol,
                err_msg=f"p_score mismatch for {r_mem.rs}",
            )


@pytest.mark.tier1
def test_output_path_streaming_all_filtered(tmp_path):
    """Streaming with all SNPs filtered returns empty result, no file created."""
    genotypes, phenotypes, kinship, snp_info = make_runner_synthetic_data()
    output_file = tmp_path / "filtered.assoc.txt"

    # Constant genotypes fail the polymorphism check, so nothing survives.
    result = run_lmm_association_numpy(
        genotypes=np.full_like(genotypes, 2.0),
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
            lmm_mode=1,
        ),
        output_path=output_file,
    )

    assert result.associations == []
    assert result.pve is None, "PVE should be None when no SNPs pass filter"


# ---------------------------------------------------------------------------
# Error message differentiation tests (68-02)
# ---------------------------------------------------------------------------


def _tiny_invariants(n_cvt: int, lmm_mode: LmmMode, n_samples: int = 8):
    """Smallest RunInvariants every dispatch path will build a kernel from."""
    from jamma.lmm.chunk_kernel import RunInvariants
    from jamma.lmm.dispatch import select_dispatch_path

    return RunInvariants.build(
        dispatch=select_dispatch_path(n_cvt, lmm_mode, accel=True, log_choices=False),
        lmm_mode=lmm_mode,
        n_cvt=n_cvt,
        n_samples=n_samples,
        n_filtered=500,
        eigenvalues=np.linspace(0.1, 2.0, n_samples),
        UtW=np.ones((n_samples, n_cvt)) * np.arange(1, n_cvt + 1),
        Uty=np.linspace(-1.0, 1.0, n_samples),
        Hi_eval_null=np.ones(n_samples),
        logl_H0=-10.0,
        l_min=1e-5,
        l_max=1e5,
        n_grid=20,
        n_refine=20,
    )


@pytest.mark.tier0
class TestErrorMessageDifferentiation:
    """A failing chunk must say which kernel failed, and where.

    These used to call the wrapper with an operation label the test invented,
    then assert the message contained it. All three labels they checked
    ("Wald C workspace compute" and friends) appear nowhere in src and never
    did, so the assertions only ever proved that an f-string interpolates.
    The labels below come from ``make_kernel``, so a renamed or duplicated one
    fails here.
    """

    def _failing_kernel(self, n_cvt: int, lmm_mode: LmmMode, exc: Exception):
        """A real kernel for this path, with its call swapped for a raise."""
        from jamma.lmm.chunk_kernel import Kernel, make_kernel

        built = make_kernel(_tiny_invariants(n_cvt, lmm_mode), 1)

        def _boom(_chunk, _threads):
            raise exc

        return Kernel(label=built.label, n_filtered=built.n_filtered, call=_boom)

    def test_every_path_has_its_own_label(self):
        """Seven labels over eight (n_cvt, mode) shapes, and none repeat a path.

        Eight shapes, seven labels: SoA-split serves modes 2 and 3 with one
        kernel, so those two share. Every other shape is distinguishable,
        including mode 4 against Wald within each fused family.
        """
        from jamma.lmm.chunk_kernel import make_kernel

        if compute_numpy._accel is None:
            pytest.skip("kernel construction needs the C extension")

        labels = {
            (n_cvt, mode): make_kernel(_tiny_invariants(n_cvt, mode), 1).label
            for n_cvt in (1, 2)
            for mode in (1, 2, 3, 4)
        }
        assert len(set(labels.values())) == 7, labels
        assert labels[1, 4] != labels[1, 1], "mode 4 must not report as Wald"
        assert labels[2, 4] != labels[2, 1], "mode 4 must not report as Wald"
        assert labels[2, 2] == labels[2, 3], "both are the one SoA-split kernel"

    @pytest.mark.parametrize(
        ("n_cvt", "lmm_mode"), [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2)]
    )
    def test_wrapped_error_names_the_kernel_and_the_offset(self, n_cvt, lmm_mode):
        """A segfault-shaped failure reports its own label, offset, and total."""
        if compute_numpy._accel is None:
            pytest.skip("kernel construction needs the C extension")

        kernel = self._failing_kernel(n_cvt, lmm_mode, OSError("segfault"))
        with pytest.raises(RuntimeError) as exc_info:
            kernel.compute_chunk(np.zeros((1, 8)), 1, 300)

        message = str(exc_info.value)
        assert kernel.label in message
        assert "300/500" in message
        assert "300 SNPs before failure" in message

    @pytest.mark.parametrize(
        "exc",
        [
            MemoryError("out of memory"),
            ValueError("bad value"),
            TypeError("wrong type"),
            OverflowError("overflow"),
        ],
    )
    def test_diagnosable_exceptions_pass_through_unwrapped(self, exc):
        """These four say what went wrong already; wrapping would bury them."""
        if compute_numpy._accel is None:
            pytest.skip("kernel construction needs the C extension")

        kernel = self._failing_kernel(1, 1, exc)
        with pytest.raises(type(exc), match=str(exc)):
            kernel.compute_chunk(np.zeros((1, 8)), 1, 0)

    def test_exception_chaining_preserved(self):
        """The original exception is chained via 'from exc'."""
        if compute_numpy._accel is None:
            pytest.skip("kernel construction needs the C extension")

        kernel = self._failing_kernel(1, 1, OSError("root cause"))
        with pytest.raises(RuntimeError) as exc_info:
            kernel.compute_chunk(np.zeros((1, 8)), 1, 0)

        assert isinstance(exc_info.value.__cause__, OSError)
        assert "root cause" in str(exc_info.value.__cause__)

    def test_successful_call_returns_result_unwrapped(self):
        """A kernel that succeeds hands its dict straight back."""
        from jamma.lmm.chunk_kernel import Kernel

        expected = {"betas": [1.0], "ses": [0.1]}
        kernel = Kernel(
            label="Fused Uab dispatch",
            n_filtered=100,
            call=lambda _chunk, _threads: expected,
        )
        assert kernel.compute_chunk(np.zeros((1, 8)), 1, 0) is expected
