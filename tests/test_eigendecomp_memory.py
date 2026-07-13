"""Tests for eigendecomposition memory pre-flight check."""

from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.memory import (
    _dsyevd_inplace_peak_gb,
    _dsyevd_peak_gb,
    _dsyevd_workspace_gb,
    _dsyevr_peak_gb,
    _dsyevr_workspace_gb,
    check_memory_before_run,
    estimate_eigendecomp_memory,
    plan_eigen_driver,
)
from jamma.lmm.eigen import eigendecompose_kinship


@pytest.mark.tier0
class TestPlanEigenDriver:
    """Tests for plan_eigen_driver — the shared driver-selection decision.

    This pure function is the single source of truth for the
    DSYEVD-inplace -> DSYEVD -> DSYEVR -> numpy choice used by both the runtime
    path (eigendecompose_kinship) and the pre-flight estimator
    (check_memory_before_run), so the two cannot drift.
    """

    N = 100_000

    def test_inplace_when_ample_and_eligible(self):
        """Ample memory + in-place eligible K -> in-place DSYEVD."""
        plan = plan_eigen_driver(
            self.N,
            available_gb=1e6,
            has_dsyevd=True,
            has_dsyevr=True,
            no_vendor=False,
            inplace_eligible=True,
        )
        assert plan.driver == "DSYEVD-inplace"
        assert plan.use_inplace is True
        assert plan.use_dsyevr is False
        assert plan.no_vendor is False
        assert plan.required_gb == pytest.approx(_dsyevd_inplace_peak_gb(self.N))

    def test_non_inplace_when_not_eligible(self):
        """Ample memory but K not in-place eligible -> non-inplace DSYEVD."""
        plan = plan_eigen_driver(
            self.N,
            available_gb=1e6,
            has_dsyevd=True,
            has_dsyevr=True,
            no_vendor=False,
            inplace_eligible=False,
        )
        assert plan.driver == "DSYEVD"
        assert plan.use_inplace is False
        assert plan.required_gb == pytest.approx(_dsyevd_peak_gb(self.N))

    def test_dsyevr_fallback_when_inplace_wont_fit(self):
        """Memory below the in-place peak but above DSYEVR -> DSYEVR fallback."""
        # available sits between DSYEVR peak (+margin) and in-place peak (+margin).
        available = (_dsyevr_peak_gb(self.N) + _dsyevd_inplace_peak_gb(self.N)) / 2
        plan = plan_eigen_driver(
            self.N,
            available_gb=available,
            has_dsyevd=True,
            has_dsyevr=True,
            no_vendor=False,
            inplace_eligible=True,
        )
        assert plan.driver == "DSYEVR"
        assert plan.use_dsyevr is True
        assert plan.use_inplace is False
        assert plan.required_gb == pytest.approx(_dsyevr_peak_gb(self.N))
        # pre_fallback_gb records the in-place peak we fell back from.
        assert plan.pre_fallback_gb == pytest.approx(_dsyevd_inplace_peak_gb(self.N))

    def test_no_dsyevr_stays_on_dsyevd_when_tight(self):
        """Tight memory, no DSYEVR available -> stay on DSYEVD (no fallback)."""
        available = _dsyevr_peak_gb(self.N)  # below in-place peak
        plan = plan_eigen_driver(
            self.N,
            available_gb=available,
            has_dsyevd=True,
            has_dsyevr=False,
            no_vendor=False,
            inplace_eligible=True,
        )
        assert plan.driver == "DSYEVD-inplace"
        assert plan.use_dsyevr is False
        assert plan.use_inplace is True

    def test_no_vendor_forces_numpy(self):
        """no_vendor -> numpy fallback with conservative DSYEVD footprint."""
        plan = plan_eigen_driver(
            self.N,
            available_gb=1e6,
            has_dsyevd=True,
            has_dsyevr=True,
            no_vendor=True,
            inplace_eligible=True,
        )
        assert plan.driver == "numpy"
        assert plan.no_vendor is True
        assert plan.use_inplace is False
        assert plan.use_dsyevr is False
        assert plan.required_gb == pytest.approx(_dsyevd_peak_gb(self.N))

    def test_no_drivers_auto_numpy(self):
        """No vendor DSYEVD and no DSYEVR -> numpy fallback even if not forced."""
        plan = plan_eigen_driver(
            self.N,
            available_gb=1e6,
            has_dsyevd=False,
            has_dsyevr=False,
            no_vendor=False,
            inplace_eligible=True,
        )
        assert plan.driver == "numpy"
        assert plan.no_vendor is True

    def test_deterministic_no_drift(self):
        """Same inputs -> identical plan (pre-flight and runtime cannot diverge)."""
        args = {
            "has_dsyevd": True,
            "has_dsyevr": True,
            "no_vendor": False,
            "inplace_eligible": True,
        }
        assert plan_eigen_driver(self.N, 1e6, **args) == plan_eigen_driver(
            self.N, 1e6, **args
        )


@pytest.mark.tier0
class TestDsyevdWorkspaceFormula:
    """Tests for _dsyevd_workspace_gb (LWORK + LIWORK upper bound)."""

    def test_known_value_1000(self):
        """Spot-check: n=1000 workspace matches hand-computed value."""
        n = 1000
        lwork_bytes = (1 + 6 * n + 2 * n * n) * 8
        liwork_bytes = (3 + 5 * n) * 8
        expected_gb = (lwork_bytes + liwork_bytes) / 1e9
        assert _dsyevd_workspace_gb(n) == pytest.approx(expected_gb, rel=1e-12)

    def test_scales_quadratically(self):
        """Workspace is dominated by 2n^2 term, so 2x n -> ~4x workspace."""
        ws_10k = _dsyevd_workspace_gb(10_000)
        ws_20k = _dsyevd_workspace_gb(20_000)
        ratio = ws_20k / ws_10k
        assert 3.9 < ratio < 4.1

    def test_liwork_uses_8_bytes(self):
        """LIWORK uses 8-byte integers (ILP64 upper bound), not 4-byte."""
        n = 100_000
        # Compute with 4-byte integers (LP64) and 8-byte (ILP64)
        lwork_bytes = (1 + 6 * n + 2 * n * n) * 8
        liwork_4 = (3 + 5 * n) * 4
        liwork_8 = (3 + 5 * n) * 8
        ws_lp64 = (lwork_bytes + liwork_4) / 1e9
        ws_ilp64 = (lwork_bytes + liwork_8) / 1e9
        actual = _dsyevd_workspace_gb(n)
        assert actual == pytest.approx(ws_ilp64, rel=1e-12)
        assert actual > ws_lp64  # Must use the larger ILP64 estimate

    def test_zero_samples(self):
        """n=0 should return near-zero workspace."""
        ws = _dsyevd_workspace_gb(0)
        # (1+0+0)*8 + (3+0)*8 = 32 bytes
        assert ws == pytest.approx(32 / 1e9, rel=1e-12)


@pytest.mark.tier0
class TestDsyevrWorkspaceFormula:
    """Tests for _dsyevr_workspace_gb (linear O(N) workspace)."""

    def test_known_value_125k(self):
        """125k samples: DSYEVR workspace should be ~0.036 GB."""
        ws = _dsyevr_workspace_gb(125_000)
        # (26*125000 + 10*125000) * 8 / 1e9 = 0.036 GB
        assert 0.034 < ws < 0.038

    def test_scales_linearly(self):
        """Workspace is O(N): 2x N -> ~2x workspace."""
        ws_10k = _dsyevr_workspace_gb(10_000)
        ws_20k = _dsyevr_workspace_gb(20_000)
        ratio = ws_20k / ws_10k
        assert 1.9 < ratio < 2.1

    def test_zero_samples(self):
        """n=0 should return minimal workspace (max(1, 0) = 1 for both terms)."""
        ws = _dsyevr_workspace_gb(0)
        # max(1, 0) * 8 + max(1, 0) * 8 = 16 bytes = 1.6e-08 GB
        assert ws == pytest.approx(16 / 1e9, rel=1e-12)


@pytest.mark.tier0
class TestEigendecompMemoryEstimate:
    """Tests for memory estimation function."""

    def test_estimate_200k_samples(self):
        """200k samples: K + U + DSYEVD workspace ~1280GB."""
        n_samples = 200_000
        estimate = estimate_eigendecomp_memory(n_samples)
        # K (320GB) + U (320GB) + DSYEVD workspace (~640GB) = ~1280GB
        assert 1275 < estimate < 1285

    def test_estimate_100k_samples(self):
        """100k samples: K + U + DSYEVD workspace ~320GB."""
        n_samples = 100_000
        estimate = estimate_eigendecomp_memory(n_samples)
        # K (80GB) + U (80GB) + DSYEVD workspace (~160GB) = ~320GB
        assert 315 < estimate < 325

    def test_per_driver_peaks(self):
        """Per-driver peak helpers return correct values."""
        n = 200_000
        # DSYEVD: K (320GB) + U (320GB) + workspace (~640GB) = ~1280GB
        assert 1275 < _dsyevd_peak_gb(n) < 1285
        # DSYEVR: K (320GB) + U (320GB) + workspace (~0.06GB) = ~640GB
        assert 639 < _dsyevr_peak_gb(n) < 641

    def test_estimate_scales_quadratically(self):
        """Memory scales quadratically (kinship term dominates workspace)."""
        est_10k = estimate_eigendecomp_memory(10_000)
        est_20k = estimate_eigendecomp_memory(20_000)
        # 2x samples -> 4x memory (quadratic kinship dominates linear workspace)
        ratio = est_20k / est_10k
        assert 3.9 < ratio < 4.1


@pytest.mark.tier0
class TestEigendecompPreflightCheck:
    """Tests for pre-flight memory check in eigendecompose_kinship."""

    def test_raises_memory_error_when_insufficient(self):
        """Should raise MemoryError before LAPACK call when memory insufficient."""
        # Create small matrix (won't actually be decomposed if check fails)
        K = np.eye(100, dtype=np.float64)

        # 100x100 matrix needs ~0.0002GB, mock available memory well below that
        # With 10% safety margin, need < required_gb * 1.1 = 0.0002 * 1.1 = 0.00022GB
        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_process,
        ):
            mock_vm.return_value.available = 100  # 100 bytes (way too small)
            mock_vm.return_value.total = 100
            mock_process.return_value.memory_info.return_value.rss = 10
            mock_process.return_value.memory_info.return_value.vms = 20

            with pytest.raises(MemoryError) as exc_info:
                eigendecompose_kinship(K)

            # Verify error message is informative
            error_msg = str(exc_info.value)
            assert "Insufficient memory" in error_msg
            assert "100" in error_msg or "100x100" in error_msg

    def test_succeeds_when_memory_sufficient(self):
        """Should proceed when sufficient memory available."""
        # Create small matrix
        K = np.eye(100, dtype=np.float64)

        # Mock psutil to report ample memory (1TB)
        # Need to mock both virtual_memory and Process for log_memory_snapshot
        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_process,
        ):
            mock_vm.return_value.available = 1e12  # 1TB
            mock_vm.return_value.total = 1e12
            mock_process.return_value.memory_info.return_value.rss = 1e9  # 1GB
            mock_process.return_value.memory_info.return_value.vms = 2e9  # 2GB

            eigenvalues, eigenvectors = eigendecompose_kinship(K)

            assert eigenvalues.shape == (100,)
            assert eigenvectors.shape == (100, 100)

    def test_error_message_includes_required_and_available(self):
        """Error message should include required GB, available GB."""
        K = np.eye(1000, dtype=np.float64)

        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_process,
        ):
            mock_vm.return_value.available = 1e6  # 1MB (way too small)
            mock_vm.return_value.total = 1e6
            mock_process.return_value.memory_info.return_value.rss = 1e5
            mock_process.return_value.memory_info.return_value.vms = 2e5

            with pytest.raises(MemoryError) as exc_info:
                eigendecompose_kinship(K)

            error_msg = str(exc_info.value)
            assert "GB" in error_msg  # Should mention GB
            has_need = "Need" in error_msg or "required" in error_msg.lower()
            assert has_need

    def test_jlinalg_eigh_returns_correct_results(self):
        """jlinalg.eigh returns correct eigendecomp via eigendecompose_kinship."""
        n = 50
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        K_ref = K.copy()

        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = 1e12
            mock_vm.return_value.total = 1e12
            mock_proc.return_value.memory_info.return_value.rss = 1e9
            mock_proc.return_value.memory_info.return_value.vms = 2e9

            eigenvalues, eigenvectors = eigendecompose_kinship(K, check_memory=False)

        K_reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(K_ref, K_reconstructed, rtol=1e-10, atol=1e-14)

    def test_estimate_always_includes_eigenvector_allocation(self):
        """Memory estimate includes K + U + DSYEVD workspace."""
        est = estimate_eigendecomp_memory(200_000)
        # K (320GB) + U (320GB) + DSYEVD workspace (~640GB) = ~1280GB
        assert 1275 < est < 1285


@pytest.mark.tier0
class TestSymmetryThreshold:
    """Tests for EIGEN-04: symmetry check threshold is 1e-11."""

    def test_module_constant_value(self):
        """_SYMMETRY_ATOL is 1e-11."""
        from jamma.lmm.eigen import _SYMMETRY_ATOL

        assert _SYMMETRY_ATOL == 1e-11

    def test_passes_below_threshold(self):
        """Matrix with max asymmetry 5e-12 passes without warning."""
        n = 50
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        # Inject small asymmetry: 5e-12 < 1e-11 threshold
        K[0, 1] += 5e-12

        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = 1e12
            mock_vm.return_value.total = 1e12
            mock_proc.return_value.memory_info.return_value.rss = 1e9
            mock_proc.return_value.memory_info.return_value.vms = 2e9

            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                eigendecompose_kinship(K.copy(), check_memory=False)
            # No symmetry warning (negative eigenvalue warnings are fine)
            sym_warnings = [x for x in w if "not symmetric" in str(x.message)]
            assert len(sym_warnings) == 0

    def test_warns_above_threshold(self):
        """Matrix with max asymmetry 5e-11 triggers symmetry warning."""
        from loguru import logger

        n = 50
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        # Inject asymmetry: 5e-11 > 1e-11 threshold
        K[0, 1] += 5e-11

        # Capture loguru messages directly (compatible with pytest-xdist workers)
        captured_messages: list[str] = []
        handler_id = logger.add(
            lambda msg: captured_messages.append(msg),
            level="WARNING",
            format="{message}",
        )
        try:
            with (
                patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
                patch("jamma.core.memory.psutil.Process") as mock_proc,
            ):
                mock_vm.return_value.available = 1e12
                mock_vm.return_value.total = 1e12
                mock_proc.return_value.memory_info.return_value.rss = 1e9
                mock_proc.return_value.memory_info.return_value.vms = 2e9

                eigendecompose_kinship(K.copy(), check_memory=False)
        finally:
            logger.remove(handler_id)

        assert any("not symmetric" in m for m in captured_messages), (
            f"Expected 'not symmetric' warning from loguru, got: {captured_messages!r}"
        )


@pytest.mark.tier0
class TestDsyevdWorkspaceAccuracy:
    """Tests for EIGEN-05: DSYEVD workspace formula accuracy."""

    def test_formula_matches_lapack_documented_minimum(self):
        """_dsyevd_workspace_gb matches LAPACK DSYEVD documented LWORK for JOBZ='V'.

        LAPACK documentation: LWORK >= 2*N^2 + 6*N + 1 (float64)
                              LIWORK >= 3 + 5*N (integers)
        """
        for n in [100, 500, 1000, 5000]:
            lapack_lwork = 2 * n * n + 6 * n + 1
            lapack_liwork = 3 + 5 * n
            lapack_total_bytes = lapack_lwork * 8 + lapack_liwork * 8
            lapack_gb = lapack_total_bytes / 1e9
            actual_gb = _dsyevd_workspace_gb(n)
            assert actual_gb == pytest.approx(lapack_gb, rel=0.01), (
                f"n={n}: _dsyevd_workspace_gb={actual_gb:.6f}GB "
                f"vs LAPACK documented={lapack_gb:.6f}GB"
            )

    def test_peak_components_do_not_double_count(self):
        """_dsyevd_peak_gb = K + U + workspace. No overlap."""
        n = 10_000
        kinship_gb = n**2 * 8 / 1e9
        workspace_gb = _dsyevd_workspace_gb(n)
        # jlinalg.eigh: K + U + workspace
        peak = _dsyevd_peak_gb(n)
        assert peak == pytest.approx(2 * kinship_gb + workspace_gb, rel=0.01)

    @pytest.mark.parametrize("n", [100, 1000, 10_000])
    def test_workspace_is_within_10pct_of_formula(self, n):
        """Workspace formula is within 10% of 2*N^2*8 bytes (dominant term)."""
        dominant_term_gb = 2 * n * n * 8 / 1e9
        actual = _dsyevd_workspace_gb(n)
        assert actual >= dominant_term_gb, "Workspace must be >= dominant term"
        assert actual < dominant_term_gb * 1.1, (
            "Workspace must be < 110% of dominant term"
        )


@pytest.mark.tier0
class TestPreFlightDsyevrAware:
    """Tests for EIGEN-01: pre-flight check uses DSYEVR peak when appropriate."""

    def test_reports_dsyevr_peak_when_dsyevd_wont_fit(self):
        """When DSYEVR available and DSYEVD exceeds memory, report DSYEVR peak."""
        n_samples = 100_000
        n_snps = 10_000
        dsyevd_peak = _dsyevd_peak_gb(n_samples)
        dsyevr_peak = _dsyevr_peak_gb(n_samples)
        available_gb = (dsyevr_peak + dsyevd_peak) / 2
        with (
            patch("jamma.jlinalg.blas_has_dsyevr", 1),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = available_gb * 1e9
            mock_vm.return_value.total = available_gb * 1e9
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0
            result = check_memory_before_run(n_samples, n_snps)
            assert result is True

    def test_raises_when_neither_driver_fits(self):
        """When neither DSYEVD nor DSYEVR fits, MemoryError is raised."""
        n_samples = 100_000
        n_snps = 10_000
        dsyevr_peak = _dsyevr_peak_gb(n_samples)
        available_gb = dsyevr_peak * 0.5
        with (
            patch("jamma.jlinalg.blas_has_dsyevr", 1),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = available_gb * 1e9
            mock_vm.return_value.total = available_gb * 1e9
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0
            with pytest.raises(MemoryError):
                check_memory_before_run(n_samples, n_snps)

    def test_uses_dsyevd_peak_when_memory_ample(self):
        """When memory is ample, DSYEVD peak is reported (no DSYEVR switch)."""
        n_samples = 1_000
        n_snps = 1_000
        with (
            patch("jamma.jlinalg.blas_has_dsyevr", 1),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = 1e12
            mock_vm.return_value.total = 1e12
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0
            result = check_memory_before_run(n_samples, n_snps)
            assert result is True

    def test_no_import_error_when_jlinalg_unavailable(self):
        """Pre-flight check works even if jamma.jlinalg is not importable."""
        import sys

        n_samples = 100
        n_snps = 100
        with (
            patch.dict(sys.modules, {"jamma.jlinalg": None}),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = 1e12
            mock_vm.return_value.total = 1e12
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0
            result = check_memory_before_run(n_samples, n_snps)
            assert result is True
