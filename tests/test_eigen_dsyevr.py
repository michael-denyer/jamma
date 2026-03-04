"""Tests for DSYEVR eigendecomposition via C extension."""

from unittest.mock import patch

import numpy as np
import pytest

# Skip entire module if C extension not available
dsyevr_available = False
try:
    from jamma.lmm._eigen_accel import eigh_dsyevr

    dsyevr_available = True
except ImportError:
    pass

pytestmark = [
    pytest.mark.tier0,
    pytest.mark.skipif(not dsyevr_available, reason="DSYEVR C extension not compiled"),
]


class TestDsyevrCorrectness:
    """Eigenvalue/eigenvector accuracy against np.linalg.eigh (DSYEVD)."""

    def test_identity_matrix(self):
        """eigh_dsyevr(eye(n)) should return all eigenvalues == 1.0."""
        n = 50
        K = np.eye(n, dtype=np.float64)
        w, v = eigh_dsyevr(K)
        assert w.shape == (n,)
        assert v.shape == (n, n)
        np.testing.assert_allclose(w, np.ones(n), rtol=1e-14)

    def test_diagonal_matrix(self):
        """Known eigenvalues for diagonal matrix (eigenvalues == diagonal)."""
        diag_vals = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        K = np.diag(diag_vals)
        w, v = eigh_dsyevr(K)
        # DSYEVR returns eigenvalues ascending
        expected = np.sort(diag_vals)
        np.testing.assert_allclose(w, expected, rtol=1e-14)

    def test_random_spd_100x100(self):
        """Eigenvalues from DSYEVR match np.linalg.eigh to rtol=1e-12 (100x100)."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 100))
        K = (A @ A.T) / 100
        w_dsyevr, _ = eigh_dsyevr(K.copy())
        w_numpy, _ = np.linalg.eigh(K.copy())
        np.testing.assert_allclose(w_dsyevr, w_numpy, rtol=1e-12, atol=1e-14)

    def test_random_spd_1000x1000(self):
        """Eigenvalues from DSYEVR match np.linalg.eigh to rtol=1e-12 (1000x1000)."""
        rng = np.random.default_rng(123)
        A = rng.standard_normal((1000, 1000))
        K = (A @ A.T) / 1000
        w_dsyevr, _ = eigh_dsyevr(K.copy())
        w_numpy, _ = np.linalg.eigh(K.copy())
        np.testing.assert_allclose(w_dsyevr, w_numpy, rtol=1e-12, atol=1e-14)

    def test_reconstruction(self):
        """V @ diag(w) @ V.T reconstructs original matrix to rtol=1e-10."""
        rng = np.random.default_rng(42)
        n = 200
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        K_ref = K.copy()
        w, v = eigh_dsyevr(K.copy())
        K_recon = v @ np.diag(w) @ v.T
        np.testing.assert_allclose(K_recon, K_ref, rtol=1e-10, atol=1e-14)

    def test_eigenvalues_ascending(self):
        """DSYEVR returns eigenvalues in ascending order (RANGE='A')."""
        rng = np.random.default_rng(42)
        n = 150
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        w, _ = eigh_dsyevr(K.copy())
        assert np.all(np.diff(w) >= 0), "Eigenvalues should be sorted ascending"


class TestDsyevrSignConsistency:
    """Downstream invariants that hold regardless of eigenvector sign convention."""

    def test_utranspose_y_invariant(self):
        """(U.T @ y)**2 is sign-invariant: matches DSYEVR and DSYEVD."""
        rng = np.random.default_rng(42)
        n = 100
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        y = rng.standard_normal(n)

        w_dsyevr, v_dsyevr = eigh_dsyevr(K.copy())
        w_numpy, v_numpy = np.linalg.eigh(K.copy())

        # Squared projections are sign-invariant (tiny numerical diffs remain
        # from DSYEVR vs DSYEVD eigenvector precision, hence rtol=1e-10 not 1e-12)
        proj_dsyevr = (v_dsyevr.T @ y) ** 2
        proj_numpy = (v_numpy.T @ y) ** 2
        np.testing.assert_allclose(proj_dsyevr, proj_numpy, rtol=1e-10)

    def test_eigenvalue_order_matches_numpy(self):
        """Eigenvalues from DSYEVR match numpy's ascending order."""
        rng = np.random.default_rng(99)
        n = 80
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        w_dsyevr, _ = eigh_dsyevr(K.copy())
        w_numpy, _ = np.linalg.eigh(K.copy())
        np.testing.assert_allclose(w_dsyevr, w_numpy, rtol=1e-12, atol=1e-14)


class TestDsyevrEdgeCases:
    """Boundary conditions and error handling."""

    def test_n_equals_1(self):
        """1x1 matrix: single element is the eigenvalue."""
        K = np.array([[3.5]], dtype=np.float64)
        w, v = eigh_dsyevr(K.copy())
        assert w.shape == (1,)
        assert v.shape == (1, 1)
        np.testing.assert_allclose(w, [3.5], rtol=1e-14)

    def test_n_equals_2(self):
        """2x2 matrix: eigenvalues match analytical solution."""
        # [[2, 1], [1, 2]]: eigenvalues are 1 and 3
        K = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        w, v = eigh_dsyevr(K.copy())
        np.testing.assert_allclose(w, [1.0, 3.0], rtol=1e-14)
        # Reconstruct to verify eigenvectors
        K_recon = v @ np.diag(w) @ v.T
        np.testing.assert_allclose(K_recon, K, atol=1e-14)

    def test_near_singular(self):
        """Near-singular matrix (cond ~1e15): non-zero eigenvalues match rtol=1e-10."""
        rng = np.random.default_rng(42)
        n = 100
        A = rng.standard_normal((n, n))
        K_full = (A @ A.T) / n
        w_full, v_full = np.linalg.eigh(K_full)
        # Scale smallest eigenvalue way down (condition ~1e15)
        w_ill = w_full.copy()
        w_ill[0] = w_full[-1] * 1e-15
        K_ill = v_full @ np.diag(w_ill) @ v_full.T
        # Make symmetric (remove numerical asymmetry)
        K_ill = (K_ill + K_ill.T) / 2

        w_dsyevr, _ = eigh_dsyevr(K_ill.copy())
        w_numpy, _ = np.linalg.eigh(K_ill.copy())
        # Non-zero eigenvalues should still match to rtol=1e-10
        mask = np.abs(w_numpy) > 1e-12 * np.max(np.abs(w_numpy))
        np.testing.assert_allclose(w_dsyevr[mask], w_numpy[mask], rtol=1e-10)

    def test_zero_eigenvalues(self):
        """Rank-deficient matrix (outer product): should have (n-1) near-zero eigs."""
        rng = np.random.default_rng(42)
        n = 20
        v = rng.standard_normal(n)
        K = np.outer(v, v)  # rank 1: n-1 eigenvalues near 0
        w, _ = eigh_dsyevr(K.copy())
        # All but 1 should be near zero
        n_near_zero = int(np.sum(np.abs(w) < 1e-10 * np.max(np.abs(w) + 1e-30)))
        assert n_near_zero == n - 1, (
            f"Expected {n - 1} near-zero eigenvalues, got {n_near_zero}. "
            f"Eigenvalues: {w}"
        )

    def test_rejects_non_float64(self):
        """eigh_dsyevr raises TypeError for non-float64 input."""
        with pytest.raises(TypeError):
            eigh_dsyevr(np.eye(10, dtype=np.float32))

    def test_rejects_non_square(self):
        """eigh_dsyevr raises ValueError for non-square input."""
        with pytest.raises(ValueError):
            eigh_dsyevr(np.ones((3, 4), dtype=np.float64))

    def test_rejects_1d(self):
        """eigh_dsyevr raises ValueError for 1D input."""
        with pytest.raises(ValueError):
            eigh_dsyevr(np.ones(10, dtype=np.float64))

    def test_rejects_readonly(self):
        """eigh_dsyevr raises ValueError for read-only arrays."""
        K = np.eye(10, dtype=np.float64)
        K.flags.writeable = False
        with pytest.raises(ValueError, match="writeable"):
            eigh_dsyevr(K)

    def test_rejects_non_contiguous(self):
        """eigh_dsyevr raises ValueError for non-contiguous arrays."""
        K = np.eye(10, dtype=np.float64)
        # Slicing with step creates a non-contiguous view
        K_nc = K[::2, ::2]
        assert not K_nc.flags["C_CONTIGUOUS"]
        assert not K_nc.flags["F_CONTIGUOUS"]
        with pytest.raises(ValueError, match="contiguous"):
            eigh_dsyevr(K_nc)

    def test_f_contiguous_input(self):
        """eigh_dsyevr handles F-contiguous input correctly."""
        rng = np.random.default_rng(42)
        n = 50
        A = rng.standard_normal((n, n))
        K = np.asfortranarray((A @ A.T) / n)
        assert K.flags["F_CONTIGUOUS"]
        K_ref = K.copy()
        w, v = eigh_dsyevr(K)
        K_recon = v @ np.diag(w) @ v.T
        np.testing.assert_allclose(K_recon, K_ref, rtol=1e-10, atol=1e-14)

    def test_empty_matrix(self):
        """eigh_dsyevr handles 0x0 matrix (returns empty arrays)."""
        K = np.empty((0, 0), dtype=np.float64)
        w, v = eigh_dsyevr(K)
        assert w.shape == (0,)
        assert v.shape == (0, 0)

    def test_rejects_invalid_uplo(self):
        """eigh_dsyevr raises ValueError for invalid uplo argument."""
        K = np.eye(5, dtype=np.float64)
        with pytest.raises(ValueError, match="uplo"):
            eigh_dsyevr(K, uplo="X")


class TestDsyevrDispatch:
    """Memory-aware driver dispatch: prefer DSYEVD, fall back to DSYEVR."""

    def test_dsyevd_used_when_memory_sufficient(self):
        """With ample memory, eigendecompose_kinship uses DSYEVD (faster)."""
        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        K_ref = K.copy()

        # n=30 with real memory → DSYEVD (ample memory)
        eigenvalues, eigenvectors = eigendecompose_kinship(K.copy(), check_memory=False)

        K_recon = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(K_recon, K_ref, rtol=1e-10, atol=1e-14)

    def test_dsyevd_when_c_ext_unavailable(self):
        """Without C extension, DSYEVD is used regardless."""
        import jamma.lmm.eigen as eigen_mod
        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        K_ref = K.copy()

        with patch.object(eigen_mod, "_DSYEVR_AVAILABLE", False):
            eigenvalues, eigenvectors = eigendecompose_kinship(
                K.copy(), check_memory=False
            )

        K_recon = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(K_recon, K_ref, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("check_memory", [True, False])
    def test_dsyevr_when_dsyevd_wont_fit(self, check_memory):
        """DSYEVR used when DSYEVD workspace exceeds available memory."""
        import jamma.lmm.eigen as eigen_mod
        from jamma.lmm.eigen import _DSYEVR_AVAILABLE, eigendecompose_kinship

        if not _DSYEVR_AVAILABLE:
            pytest.skip("DSYEVR C extension not available")

        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n

        # Mock peaks to create a meaningful spread: DSYEVD=100GB, DSYEVR=50GB,
        # available=80GB. DSYEVD+margin(10GB) > 80 but DSYEVR+margin(5GB) < 80.
        call_count = []
        original = eigen_mod._eigh_dsyevr

        def tracking_wrapper(K_in):
            call_count.append(1)
            return original(K_in)

        with (
            patch.object(eigen_mod, "_eigh_dsyevr", tracking_wrapper),
            patch("jamma.lmm.eigen._dsyevd_peak_gb", return_value=100.0),
            patch("jamma.lmm.eigen._dsyevr_peak_gb", return_value=50.0),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = 80e9  # 80GB
            mock_vm.return_value.total = 80e9
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0

            eigendecompose_kinship(K.copy(), check_memory=check_memory)

        assert len(call_count) == 1, "Expected DSYEVR when DSYEVD won't fit"

    def test_dsyevd_preferred_with_ample_memory(self):
        """DSYEVD used even when DSYEVR available, if memory is ample."""
        import jamma.lmm.eigen as eigen_mod
        from jamma.lmm.eigen import _DSYEVR_AVAILABLE, eigendecompose_kinship

        if not _DSYEVR_AVAILABLE:
            pytest.skip("DSYEVR C extension not available")

        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n

        dsyevr_calls = []
        original = eigen_mod._eigh_dsyevr

        def tracking_wrapper(K_in):
            dsyevr_calls.append(1)
            return original(K_in)

        with (
            patch.object(eigen_mod, "_eigh_dsyevr", tracking_wrapper),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = 1e12  # 1TB
            mock_vm.return_value.total = 1e12
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0

            eigendecompose_kinship(K.copy(), check_memory=True)

        assert len(dsyevr_calls) == 0, "DSYEVD should be preferred when memory is ample"

    def test_neither_driver_fits_raises_memoryerror(self):
        """When neither DSYEVD nor DSYEVR fits, MemoryError reports DSYEVR peak."""
        from jamma.lmm.eigen import _DSYEVR_AVAILABLE, eigendecompose_kinship

        if not _DSYEVR_AVAILABLE:
            pytest.skip("DSYEVR C extension not available")

        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n

        from jamma.core.memory import _dsyevr_peak_gb

        dsyevr_peak = _dsyevr_peak_gb(n)
        # Set available memory below DSYEVR peak (both drivers fail)
        available = dsyevr_peak * 0.5

        with (
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = available * 1e9
            mock_vm.return_value.total = available * 1e9
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0

            with pytest.raises(MemoryError):
                eigendecompose_kinship(K.copy(), check_memory=True)

    def test_try_import_abi_mismatch_returns_false(self):
        """ABI mismatch returns (False, None) and logs warning."""
        from jamma.lmm.eigen import _try_import_dsyevr

        with patch("jamma.lmm.eigen._EXPECTED_EIGEN_ABI", 999):
            available, func = _try_import_dsyevr()
        assert available is False
        assert func is None


@pytest.mark.tier0
class TestDsyevrProbe:
    """Tests for EIGEN-03: DSYEVR probe runs once at import time."""

    # Override module-level skipif — these tests don't need the C extension
    pytestmark = [pytest.mark.tier0]

    def test_try_import_not_called_on_repeated_eigendecompose(self):
        """_try_import_dsyevr is NOT called by repeated eigendecompose_kinship calls."""
        import jamma.lmm.eigen as eigen_mod

        rng = np.random.default_rng(42)
        n = 20
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n

        # Mark recompile as already attempted (simulates normal post-init state)
        original_attempted = eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED
        eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED = True

        try:
            with patch.object(
                eigen_mod, "_try_import_dsyevr", wraps=eigen_mod._try_import_dsyevr
            ) as mock_probe:
                with (
                    patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
                    patch("jamma.core.memory.psutil.Process") as mock_proc,
                ):
                    mock_vm.return_value.available = 1e12
                    mock_vm.return_value.total = 1e12
                    mock_proc.return_value.memory_info.return_value.rss = 1e9
                    mock_proc.return_value.memory_info.return_value.vms = 2e9

                    # Call eigendecompose_kinship 3 times
                    for _ in range(3):
                        eigen_mod.eigendecompose_kinship(K.copy(), check_memory=False)

                # _try_import_dsyevr should NOT have been called by any of the 3 calls
                assert mock_probe.call_count == 0, (
                    f"_try_import_dsyevr called {mock_probe.call_count} times; "
                    "expected 0 (probe should only run at import time)"
                )
        finally:
            eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED = original_attempted

    def test_lazy_init_runs_recompile_once(self):
        """_lazy_init_dsyevr() recompiles exactly once when DSYEVR unavailable."""
        import jamma.lmm.eigen as eigen_mod

        # Save originals
        orig_available = eigen_mod._DSYEVR_AVAILABLE
        orig_attempted = eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED

        try:
            # Simulate: DSYEVR not available, recompile not yet attempted
            eigen_mod._DSYEVR_AVAILABLE = False
            eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED = False

            with patch.object(
                eigen_mod, "_auto_recompile_eigen", return_value=False
            ) as mock_recompile:
                eigen_mod._lazy_init_dsyevr()
                eigen_mod._lazy_init_dsyevr()  # Second call should be no-op
                eigen_mod._lazy_init_dsyevr()  # Third call should be no-op

            assert mock_recompile.call_count == 1, (
                f"_auto_recompile_eigen called {mock_recompile.call_count} times;"
                " expected 1"
            )
            assert eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED is True
        finally:
            eigen_mod._DSYEVR_AVAILABLE = orig_available
            eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED = orig_attempted


@pytest.mark.tier0
class TestDsyevrFallback:
    """Tests for EIGEN-02: DSYEVR fallback when DSYEVD raises MemoryError."""

    # Override module-level skipif — fallback test mocks DSYEVD
    pytestmark = [pytest.mark.tier0]

    def test_dsyevr_fallback_on_dsyevd_memory_error(self):
        """When DSYEVD raises MemoryError, falls back to DSYEVR if available."""
        import jamma.lmm.eigen as eigen_mod

        if not eigen_mod._DSYEVR_AVAILABLE:
            pytest.skip("DSYEVR C extension not available")

        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        K_ref = K.copy()

        # Track DSYEVR calls
        dsyevr_calls = []
        original_dsyevr = eigen_mod._eigh_dsyevr

        def tracking_dsyevr(K_in):
            dsyevr_calls.append(1)
            return original_dsyevr(K_in)

        def failing_dsyevd(K_in):
            """Simulates DSYEVD workspace malloc failure (before K is modified)."""
            raise MemoryError("simulated DSYEVD workspace allocation failure")

        with (
            patch.object(eigen_mod, "_eigh_inplace", failing_dsyevd),
            patch.object(eigen_mod, "_eigh_dsyevr", tracking_dsyevr),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            # Mock memory: enough that DSYEVD is selected (not DSYEVR)
            mock_vm.return_value.available = 1e12
            mock_vm.return_value.total = 1e12
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0

            eigenvalues, eigenvectors = eigen_mod.eigendecompose_kinship(
                K.copy(), check_memory=False
            )

        # DSYEVR was called as fallback
        assert len(dsyevr_calls) == 1, (
            "Expected DSYEVR fallback after DSYEVD MemoryError"
        )

        # Results are correct
        K_recon = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(K_recon, K_ref, rtol=1e-10, atol=1e-14)

    def test_no_fallback_when_dsyevr_unavailable(self):
        """When DSYEVR unavailable, DSYEVD MemoryError re-raises."""
        import jamma.lmm.eigen as eigen_mod

        rng = np.random.default_rng(42)
        n = 20
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n

        def failing_dsyevd(K_in):
            raise MemoryError("simulated DSYEVD workspace allocation failure")

        # Ensure _lazy_init_dsyevr() does not overwrite _DSYEVR_AVAILABLE=False
        # by suppressing the recompile path (simulate already-attempted state).
        orig_available = eigen_mod._DSYEVR_AVAILABLE
        orig_attempted = eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED
        try:
            eigen_mod._DSYEVR_AVAILABLE = False
            eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED = True

            with (
                patch.object(eigen_mod, "_eigh_inplace", failing_dsyevd),
                patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
                patch("jamma.core.memory.psutil.Process") as mock_proc,
            ):
                mock_vm.return_value.available = 1e12
                mock_vm.return_value.total = 1e12
                mock_proc.return_value.memory_info.return_value.rss = 0
                mock_proc.return_value.memory_info.return_value.vms = 0

                with pytest.raises(MemoryError, match="simulated"):
                    eigen_mod.eigendecompose_kinship(K.copy(), check_memory=False)
        finally:
            eigen_mod._DSYEVR_AVAILABLE = orig_available
            eigen_mod._DSYEVR_RECOMPILE_ATTEMPTED = orig_attempted

    def test_no_fallback_when_already_using_dsyevr(self):
        """When DSYEVR itself raises MemoryError, it re-raises (no double fallback)."""
        import jamma.lmm.eigen as eigen_mod

        if not eigen_mod._DSYEVR_AVAILABLE:
            pytest.skip("DSYEVR C extension not available")

        rng = np.random.default_rng(42)
        n = 20
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n

        def failing_dsyevr(K_in):
            raise MemoryError("simulated DSYEVR failure")

        with (
            patch.object(eigen_mod, "_eigh_dsyevr", failing_dsyevr),
            # Force DSYEVR path by mocking memory pressure
            patch("jamma.lmm.eigen._dsyevd_peak_gb", return_value=100.0),
            patch("jamma.lmm.eigen._dsyevr_peak_gb", return_value=50.0),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = 80e9
            mock_vm.return_value.total = 80e9
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0

            with pytest.raises(MemoryError):
                eigen_mod.eigendecompose_kinship(K.copy(), check_memory=False)


@pytest.mark.tier0
class TestEigendecompCrossVariant:
    """Cross-variant equivalence tests: DSYEVR vs DSYEVD through the wrapper."""

    def test_eigendecompose_kinship_dsyevr_vs_dsyevd_eigenvalue_equivalence(self):
        """DSYEVR and DSYEVD paths through eigendecompose_kinship match rtol=1e-12."""
        from jamma.lmm.eigen import _DSYEVR_AVAILABLE, eigendecompose_kinship

        if not _DSYEVR_AVAILABLE:
            pytest.skip("DSYEVR C extension not available")

        rng = np.random.default_rng(42)
        n = 50
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        K_ref = K.copy()

        # DSYEVD path: check_memory=False, no mocking (uses DSYEVD when memory ample)
        w_dsyevd, v_dsyevd = eigendecompose_kinship(K.copy(), check_memory=False)

        # DSYEVR path: mock memory to force DSYEVR selection
        with (
            patch("jamma.lmm.eigen._dsyevd_peak_gb", return_value=100.0),
            patch("jamma.lmm.eigen._dsyevr_peak_gb", return_value=50.0),
            patch("jamma.core.memory.psutil.virtual_memory") as mock_vm,
            patch("jamma.core.memory.psutil.Process") as mock_proc,
        ):
            mock_vm.return_value.available = 80e9  # 80GB — fits DSYEVR but not DSYEVD
            mock_vm.return_value.total = 80e9
            mock_proc.return_value.memory_info.return_value.rss = 0
            mock_proc.return_value.memory_info.return_value.vms = 0

            w_dsyevr, v_dsyevr = eigendecompose_kinship(K.copy(), check_memory=True)

        # Eigenvalues must match closely
        np.testing.assert_allclose(w_dsyevr, w_dsyevd, rtol=1e-12, atol=1e-14)

        # Eigenvectors must reconstruct the original K (sign convention may differ)
        K_recon_dsyevr = v_dsyevr @ np.diag(w_dsyevr) @ v_dsyevr.T
        np.testing.assert_allclose(K_recon_dsyevr, K_ref, rtol=1e-10, atol=1e-14)

    def test_eigendecomp_near_singular_matrix(self):
        """Near-singular kinship matrix is decomposed without exceptions."""
        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        n = 50
        A = rng.standard_normal((n, n))
        K_full = (A @ A.T) / n

        # Build a near-singular matrix with last eigenvalue ~1e-15
        w_full, v_full = np.linalg.eigh(K_full)
        w_ill = w_full.copy()
        w_ill[0] = w_full[-1] * 1e-15  # Make smallest eigenvalue near-zero
        K_ill = v_full @ np.diag(w_ill) @ v_full.T
        K_ill = (K_ill + K_ill.T) / 2  # Ensure exact symmetry

        # Should not raise
        eigenvalues, eigenvectors = eigendecompose_kinship(
            K_ill.copy(), check_memory=False
        )

        # Eigenvalues must be sorted ascending
        assert np.all(np.diff(eigenvalues) >= 0), "Eigenvalues must be sorted ascending"

        # All returned eigenvalues must be non-negative (zeroed if near-zero/negative)
        assert np.all(eigenvalues >= 0), (
            f"All eigenvalues >= 0 after thresholding, min={eigenvalues.min()}"
        )

        # Correct count returned
        assert eigenvalues.shape == (n,)

        # Reconstruct original matrix (looser tol for ill-conditioned)
        K_recon = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(
            K_recon,
            K_ill,
            rtol=1e-4,
            atol=1e-10,
            err_msg="Near-singular matrix reconstruction failed",
        )

    def test_eigendecomp_many_sub_threshold_eigenvalues(self):
        """Matrix with many sub-threshold eigenvalues returns all n values."""
        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        n = 50
        # Construct matrix with 40 tiny eigenvalues and 10 large ones
        diag_vals = np.array([1e-15] * 40 + [1.0] * 10)
        # Rotate by random orthogonal matrix
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        K = Q @ np.diag(diag_vals) @ Q.T
        K = (K + K.T) / 2  # Ensure exact symmetry

        # Should not raise; all 50 eigenvalues returned
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*rank.*deficien", category=UserWarning
            )
            eigenvalues, eigenvectors = eigendecompose_kinship(
                K.copy(), check_memory=False
            )

        assert eigenvalues.shape == (n,), (
            f"Expected {n} eigenvalues, got {eigenvalues.shape}"
        )
        assert eigenvectors.shape == (n, n), (
            f"Expected ({n},{n}) eigenvectors, got {eigenvectors.shape}"
        )

        # Sub-threshold eigenvalues (1e-15) should be zeroed by thresholding
        n_zeroed = int(np.sum(eigenvalues < 1e-10))
        assert n_zeroed >= 35, (
            f"Expected most of the 40 tiny eigenvalues to be zeroed, "
            f"got {n_zeroed} below threshold"
        )

        # Large eigenvalues (~1.0) should be preserved
        large_evals = eigenvalues[eigenvalues > 0.5]
        assert len(large_evals) == 10, (
            f"Expected 10 large eigenvalues, got {len(large_evals)}"
        )
        np.testing.assert_allclose(large_evals, 1.0, rtol=1e-6)


@pytest.mark.tier0
class TestDsyevrBoundsCheck:
    """Tests for EIGEN-06: workspace query bounds check in C extension."""

    # Override module-level skipif for this class
    pytestmark = [pytest.mark.tier0]

    @pytest.mark.skipif(not dsyevr_available, reason="DSYEVR C extension not compiled")
    def test_valid_workspace_on_normal_matrix(self):
        """Normal matrix workspace query returns valid (positive) sizes.

        This test exercises the happy path. The bounds check
        (lwork <= 0 || liwork <= 0) in _eigen_accel.c guards against
        LAPACK returning garbage workspace sizes. It cannot be triggered
        from Python without simulating LAPACK malfunction.
        """
        rng = np.random.default_rng(42)
        n = 100
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        # If workspace query returned invalid sizes, this would raise RuntimeError
        # with "workspace query returned invalid sizes" message
        w, v = eigh_dsyevr(K)
        assert w.shape == (n,)
        assert v.shape == (n, n)

    @pytest.mark.skipif(not dsyevr_available, reason="DSYEVR C extension not compiled")
    def test_small_matrices_have_valid_workspace(self):
        """Edge case matrices (n=1,2) still produce valid workspace queries."""
        for n in [1, 2, 3]:
            K = np.eye(n, dtype=np.float64)
            w, v = eigh_dsyevr(K.copy())
            assert w.shape == (n,)
            np.testing.assert_allclose(w, np.ones(n), rtol=1e-14)


@pytest.mark.tier0
class TestSelectEigenDriver:
    """Direct unit tests for _select_eigen_driver."""

    # Override module-level skipif — these tests don't need the C extension
    pytestmark = [pytest.mark.tier0]

    def test_no_dsyevr_always_returns_dsyevd(self):
        """_select_eigen_driver returns 'dsyevd' when dsyevr_available=False."""
        from jamma.lmm.eigen import _select_eigen_driver

        # Large n, ample memory, but DSYEVR unavailable
        assert _select_eigen_driver(50_000, 300.0, dsyevr_available=False) == "dsyevd"
        # Small n also returns dsyevd
        assert _select_eigen_driver(100, 300.0, dsyevr_available=False) == "dsyevd"
        # Even with minimal memory
        assert _select_eigen_driver(1000, 0.001, dsyevr_available=False) == "dsyevd"

    def test_small_matrix_returns_dsyevd(self):
        """Small matrix with ample memory returns 'dsyevd'."""
        from jamma.lmm.eigen import _select_eigen_driver

        # n=100 has trivially small DSYEVD peak — easily fits in 300GB
        result = _select_eigen_driver(100, 300.0, dsyevr_available=True)
        assert result == "dsyevd"

    def test_large_matrix_exceeding_budget_returns_dsyevr(self):
        """Large matrix where DSYEVD would exceed memory returns 'dsyevr'."""
        from jamma.core.memory import _dsyevd_peak_gb, _memory_margin_gb
        from jamma.lmm.eigen import _select_eigen_driver

        peak = _dsyevd_peak_gb(46_000)
        margin = _memory_margin_gb(peak)
        # Set available memory well below what DSYEVD needs
        assert _select_eigen_driver(46_000, 1.0, dsyevr_available=True) == "dsyevr"
        # Sanity check: the threshold is indeed larger than 1.0GB
        assert peak + margin > 1.0, (
            f"Expected DSYEVD peak+margin > 1.0 GB for n=46000, got {peak + margin}"
        )

    def test_boundary_threshold(self):
        """_select_eigen_driver uses <= for DSYEVD boundary check.

        At exactly peak+margin: DSYEVD selected (condition: peak+margin <= available).
        At peak+margin-0.001: DSYEVR selected (just below threshold).
        """
        from jamma.core.memory import _dsyevd_peak_gb, _memory_margin_gb
        from jamma.lmm.eigen import _select_eigen_driver

        n = 1000
        peak = _dsyevd_peak_gb(n)
        margin = _memory_margin_gb(peak)
        threshold = peak + margin

        # Exactly at threshold: DSYEVD fits (<=)
        assert _select_eigen_driver(n, threshold, dsyevr_available=True) == "dsyevd", (
            f"At exactly threshold={threshold:.6f} GB, expected 'dsyevd'"
        )
        # Just below threshold: DSYEVR selected
        below = threshold - 0.001
        assert _select_eigen_driver(n, below, dsyevr_available=True) == "dsyevr", (
            f"At below={below:.6f} GB, expected 'dsyevr'"
        )


@pytest.mark.tier0
class TestEigenvalueZeroingBoundary:
    """Boundary tests for eigenvalue zeroing in eigendecompose_kinship.

    The zeroing condition uses strict < (not <=), so eigenvalues at exactly
    the threshold (1e-10 by default) are NOT zeroed.
    """

    # Override module-level skipif — these tests don't need the C extension
    pytestmark = [pytest.mark.tier0]

    def test_eigenvalue_at_threshold_not_zeroed(self):
        """Eigenvalue at exactly 1e-10 is NOT zeroed (strict < threshold)."""
        from jamma.lmm.eigen import eigendecompose_kinship

        # Use diagonal matrix to get exact eigenvalues without QR perturbation
        n = 10
        target_eval = 1e-10
        K = np.diag([target_eval] + [1.0] * (n - 1))

        eigenvalues, _ = eigendecompose_kinship(K.copy(), check_memory=False)

        min_eval = np.min(eigenvalues)
        assert min_eval != 0.0, (
            f"Eigenvalue at threshold {target_eval} should NOT be zeroed, "
            f"got {min_eval}"
        )
        np.testing.assert_allclose(min_eval, target_eval, rtol=1e-12)

    def test_eigenvalue_below_threshold_zeroed(self):
        """Eigenvalue at 9e-11 IS zeroed (below strict < threshold of 1e-10)."""
        from jamma.lmm.eigen import eigendecompose_kinship

        n = 10
        target_eval = 9e-11  # Below 1e-10 threshold
        K = np.diag([target_eval] + [1.0] * (n - 1))

        eigenvalues, _ = eigendecompose_kinship(K.copy(), check_memory=False)

        min_eval = np.min(eigenvalues)
        assert min_eval == 0.0, (
            f"Eigenvalue {target_eval} should be zeroed (< threshold 1e-10), "
            f"got {min_eval}"
        )

    def test_negative_eigenvalue_zeroing_boundary(self):
        """Negative eigenvalue zeroing uses strict < -threshold.

        -5e-11: abs=5e-11 < 1e-10 -> zeroed by small_mask
        -1e-10: abs=1e-10, NOT < 1e-10 -> NOT zeroed; -1e-10 not < -1e-10 -> NOT zeroed
        -2e-10: -2e-10 < -1e-10 -> zeroed by negative eigenvalue path
        """
        import warnings

        from jamma.lmm.eigen import eigendecompose_kinship

        def decompose_diagonal(eval_val):
            n = 10
            K = np.diag([eval_val] + [1.0] * (n - 1))
            K = (K + K.T) / 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                evals, _ = eigendecompose_kinship(K.copy(), check_memory=False)
            return np.min(evals)

        # -5e-11: abs = 5e-11 < 1e-10 -> zeroed by small_mask path
        result_minus_5e11 = decompose_diagonal(-5e-11)
        assert result_minus_5e11 == 0.0, (
            f"Eigenvalue -5e-11 should be zeroed (|5e-11| < 1e-10), "
            f"got {result_minus_5e11}"
        )

        # -1e-10: abs = 1e-10 NOT < 1e-10; -1e-10 NOT < -1e-10 -> NOT zeroed
        result_minus_1e10 = decompose_diagonal(-1e-10)
        assert result_minus_1e10 != 0.0, (
            f"Eigenvalue -1e-10 should NOT be zeroed "
            f"(abs not < 1e-10, and not < -1e-10), got {result_minus_1e10}"
        )
        np.testing.assert_allclose(result_minus_1e10, -1e-10, rtol=1e-12)

        # -2e-10: -2e-10 < -1e-10 -> zeroed by negative eigenvalue path
        result_minus_2e10 = decompose_diagonal(-2e-10)
        assert result_minus_2e10 == 0.0, (
            f"Eigenvalue -2e-10 should be zeroed (-2e-10 < -1e-10), "
            f"got {result_minus_2e10}"
        )
