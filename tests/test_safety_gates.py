"""Tests for Phase 111 safety gates (SAFE-01, SAFE-02, SAFE-03)."""

import inspect
import warnings
from unittest.mock import MagicMock, patch

import numpy as np


class TestLP64OverflowWarning:
    """SAFE-01: LP64 overflow warning before eigendecomposition."""

    def test_lp64_warning_in_source(self):
        """eigendecompose_kinship source contains LP64 overflow guard."""
        from jamma.lmm.eigen import eigendecompose_kinship

        source = inspect.getsource(eigendecompose_kinship)
        assert "LP64 BLAS detected" in source
        assert "int32 overflow" in source
        assert "40_000" in source or "40000" in source

    def test_lp64_warning_after_no_vendor_resolution(self):
        """LP64 warning appears after no_vendor is resolved."""
        from jamma.lmm.eigen import eigendecompose_kinship

        source = inspect.getsource(eigendecompose_kinship)
        lp64_pos = source.index("LP64 BLAS detected")
        no_vendor_pos = source.index("No vendor LAPACK")
        assert lp64_pos > no_vendor_pos, (
            "LP64 warning must appear after no_vendor resolution"
        )

    def test_lp64_large_matrix_warns(self):
        """eigendecompose_kinship warns on LP64 with n_samples > 40k."""
        from jamma.lmm.eigen import eigendecompose_kinship

        n = 50_000
        K = np.eye(3, dtype=np.float64)  # small matrix, we patch shape
        fake_K = MagicMock(wraps=K)
        fake_K.shape = (n, n)
        fake_K.ndim = 2
        fake_K.dtype = np.float64
        fake_K.flags = K.flags

        mock_snapshot = MagicMock()
        mock_snapshot.available_gb = 1000.0

        with (
            patch("jamma.lmm.eigen.jlinalg") as mock_jl,
            patch("jamma.lmm.eigen._check_symmetry_sampled"),
            patch("jamma.lmm.eigen.log_memory_snapshot", return_value=mock_snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_jl.blas_is_ilp64 = 0
            try:
                eigendecompose_kinship(fake_K)
            except Exception:
                pass  # expected — mock doesn't support full pipeline

            lp64_warnings = [x for x in w if "LP64 BLAS detected" in str(x.message)]
            assert len(lp64_warnings) == 1
            assert lp64_warnings[0].category is RuntimeWarning
            assert "int32 overflow" in str(lp64_warnings[0].message)
            assert "50,000" in str(lp64_warnings[0].message)

    def test_ilp64_no_warning(self):
        """ILP64 BLAS does not trigger overflow warning."""
        from jamma.lmm.eigen import eigendecompose_kinship

        n = 50_000
        K = np.eye(3, dtype=np.float64)
        fake_K = MagicMock(wraps=K)
        fake_K.shape = (n, n)
        fake_K.ndim = 2
        fake_K.dtype = np.float64
        fake_K.flags = K.flags

        mock_snapshot = MagicMock()
        mock_snapshot.available_gb = 1000.0

        with (
            patch("jamma.lmm.eigen.jlinalg") as mock_jl,
            patch("jamma.lmm.eigen._check_symmetry_sampled"),
            patch("jamma.lmm.eigen.log_memory_snapshot", return_value=mock_snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_jl.blas_is_ilp64 = 1
            try:
                eigendecompose_kinship(fake_K)
            except Exception:
                pass

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0

    def test_small_matrix_no_warning(self):
        """n_samples <= 40k does not trigger warning even on LP64."""
        from jamma.lmm.eigen import eigendecompose_kinship

        n = 1_000
        K = np.eye(3, dtype=np.float64)
        fake_K = MagicMock(wraps=K)
        fake_K.shape = (n, n)
        fake_K.ndim = 2
        fake_K.dtype = np.float64
        fake_K.flags = K.flags

        with (
            patch("jamma.lmm.eigen.jlinalg") as mock_jl,
            patch("jamma.lmm.eigen._check_symmetry_sampled"),
            patch(
                "jamma.lmm.eigen.log_memory_snapshot",
                return_value=MagicMock(available_gb=1000.0),
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_jl.blas_is_ilp64 = 0
            try:
                eigendecompose_kinship(fake_K)
            except Exception:
                pass

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0

    def test_no_vendor_suppresses_lp64_warning(self):
        """LP64 warning does not fire when using np.linalg.eigh fallback."""
        from jamma.lmm.eigen import eigendecompose_kinship

        n = 50_000
        K = np.eye(3, dtype=np.float64)
        fake_K = MagicMock(wraps=K)
        fake_K.shape = (n, n)
        fake_K.ndim = 2
        fake_K.dtype = np.float64
        fake_K.flags = K.flags

        with (
            patch("jamma.lmm.eigen.jlinalg") as mock_jl,
            patch("jamma.lmm.eigen._check_symmetry_sampled"),
            patch(
                "jamma.lmm.eigen.log_memory_snapshot",
                return_value=MagicMock(available_gb=1000.0),
            ),
            patch.dict("os.environ", {"JLINALG_NO_VENDOR_LAPACK": "1"}),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_jl.blas_is_ilp64 = 0
            try:
                eigendecompose_kinship(fake_K)
            except Exception:
                pass

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0, (
                "LP64 warning should not fire when no_vendor forces np.linalg.eigh"
            )


class TestLOCOIteratorRuntimeError:
    """SAFE-02: LOCO iterator None raises RuntimeError, not AssertionError."""

    def test_no_assert_in_loco(self):
        """loco.py does not contain 'assert loco_iter is not None'."""
        from jamma.lmm import loco

        source = inspect.getsource(loco)
        assert "assert loco_iter is not None" not in source

    def test_has_runtime_error(self):
        """loco.py contains RuntimeError with descriptive message."""
        from jamma.lmm import loco

        source = inspect.getsource(loco)
        assert "LOCO kinship iterator was not initialized" in source
        assert "raise RuntimeError" in source

    def test_runtime_error_message(self):
        """RuntimeError message is descriptive and actionable."""
        from jamma.lmm.loco import run_lmm_loco

        source = inspect.getsource(run_lmm_loco)
        # Verify the error path exists and has the right exception type
        assert "raise RuntimeError" in source
        assert "LOCO kinship iterator was not initialized" in source
        assert "please report it" in source


class TestJlinalgABIValidation:
    """SAFE-03: _jlinalg ABI version validated at import."""

    def test_expected_abi_constant_exists(self):
        """jlinalg.__init__ defines _EXPECTED_JLINALG_ABI."""
        import jamma.jlinalg as jl

        assert hasattr(jl, "_EXPECTED_JLINALG_ABI")
        assert isinstance(jl._EXPECTED_JLINALG_ABI, int)

    def test_abi_matches_current(self):
        """Current ABI_VERSION matches expected."""
        from jamma import jlinalg

        if jlinalg.HAS_C_EXTENSION:
            assert jlinalg.ABI_VERSION == jlinalg._EXPECTED_JLINALG_ABI

    def test_abi_check_in_source(self):
        """jlinalg __init__.py contains ABI validation logic."""
        import jamma.jlinalg as jl

        source = inspect.getsource(jl)
        assert "ABI mismatch" in source
        assert "compiled=" in source
        assert "expected=" in source

    def test_has_c_extension_set_after_abi_check(self):
        """HAS_C_EXTENSION is set AFTER ABI validation, not before."""
        import jamma.jlinalg as jl

        source = inspect.getsource(jl)
        # In both import paths, HAS_C_EXTENSION = True must come after the ABI check
        # Find the primary path (first occurrence)
        abi_check_pos = source.index("ABI_VERSION != _EXPECTED_JLINALG_ABI")
        has_ext_pos = source.index("HAS_C_EXTENSION: bool = True")
        assert abi_check_pos < has_ext_pos, (
            "HAS_C_EXTENSION must be set AFTER ABI validation passes"
        )

    def test_abi_mismatch_prevents_has_c_extension(self):
        """ABI mismatch ImportError prevents HAS_C_EXTENSION from being set."""
        import jamma.jlinalg as jl

        source = inspect.getsource(jl)
        # Both paths: the raise ImportError for ABI mismatch must appear
        # before HAS_C_EXTENSION = True
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "HAS_C_EXTENSION: bool = True" in line:
                # Look backwards — there should be an ABI check before this
                preceding = "\n".join(lines[max(0, i - 10) : i])
                assert "ABI_VERSION != _EXPECTED_JLINALG_ABI" in preceding or i == 0
