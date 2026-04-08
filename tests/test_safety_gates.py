"""Tests for safety gates (SAFE-01, SAFE-02, SAFE-03).

Tests observable behavior (warnings emitted, exceptions raised, attribute values)
rather than inspecting source code strings. Source-inspection tests are brittle —
they break on refactors even when behavior is preserved.

Uses real types (MemorySnapshot, numpy arrays) instead of MagicMock. For the LP64
threshold tests, np.lib.stride_tricks.as_strided creates a 50k x 50k "view" backed
by a tiny 4x4 allocation — exercises the real shape-checking code path without
allocating 20 GB.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.memory import MemorySnapshot


def _make_memory_snapshot(available_gb: float = 1000.0) -> MemorySnapshot:
    """Build a real MemorySnapshot for test use."""
    return MemorySnapshot(
        rss_gb=1.0,
        vms_gb=2.0,
        available_gb=available_gb,
        total_gb=1024.0,
        percent_used=10.0,
    )


def _big_K_view(n: int = 50_000) -> np.ndarray:
    """Create a virtual n x n symmetric matrix backed by ~128 bytes.

    Uses stride_tricks with strides=(0,0) so every element reads the same
    memory.  The result is square, 2-D, float64, and symmetric (all
    elements equal) — satisfying eigendecompose_kinship's validation
    without a real allocation.
    """
    backing = np.ones((4, 4), dtype=np.float64)
    return np.lib.stride_tricks.as_strided(backing, shape=(n, n), strides=(0, 0))


class TestLP64OverflowWarning:
    """SAFE-01: LP64 overflow warning before eigendecomposition."""

    def test_lp64_large_matrix_warns(self):
        """eigendecompose_kinship warns on LP64 with n_samples > 40k."""
        from jamma.lmm import eigen

        big_K = _big_K_view(50_000)
        snapshot = _make_memory_snapshot()

        with (
            patch.object(eigen, "jlinalg") as mock_jl,
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_jl.blas_is_ilp64 = 0
            mock_jl.blas_has_dsyevd = True
            mock_jl.blas_has_dsyevr = False
            mock_jl.eigh.return_value = (np.ones(50_000), np.eye(3))

            try:
                eigen.eigendecompose_kinship(big_K)
            except Exception:
                pass  # eigh mock returns wrong shape, fine

            lp64_warnings = [x for x in w if "LP64 BLAS detected" in str(x.message)]
            assert len(lp64_warnings) == 1
            assert lp64_warnings[0].category is RuntimeWarning
            assert "int32 overflow" in str(lp64_warnings[0].message)
            assert "50,000" in str(lp64_warnings[0].message)

    def test_ilp64_no_warning(self):
        """ILP64 BLAS does not trigger overflow warning."""
        from jamma.lmm import eigen

        big_K = _big_K_view(50_000)
        snapshot = _make_memory_snapshot()

        with (
            patch.object(eigen, "jlinalg") as mock_jl,
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_jl.blas_is_ilp64 = 1
            mock_jl.blas_has_dsyevd = True
            mock_jl.blas_has_dsyevr = False
            mock_jl.eigh.return_value = (np.ones(50_000), np.eye(3))

            try:
                eigen.eigendecompose_kinship(big_K)
            except Exception:
                pass

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0

    def test_small_matrix_no_warning(self):
        """n_samples <= 40k does not trigger warning even on LP64."""
        from jamma.lmm import eigen
        from jamma.lmm.eigen import eigendecompose_kinship

        K = np.eye(100, dtype=np.float64)
        snapshot = _make_memory_snapshot()

        with (
            patch.object(eigen, "jlinalg") as mock_jl,
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_jl.blas_is_ilp64 = 0
            mock_jl.blas_has_dsyevd = True
            mock_jl.blas_has_dsyevr = False
            mock_jl.eigh.return_value = (np.ones(100), np.eye(100))

            eigendecompose_kinship(K)

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0

    def test_no_vendor_suppresses_lp64_warning(self):
        """LP64 warning does not fire when using np.linalg.eigh fallback."""
        from jamma.lmm import eigen
        from jamma.lmm.eigen import eigendecompose_kinship

        big_K = _big_K_view(50_000)
        snapshot = _make_memory_snapshot()

        with (
            patch.object(eigen, "jlinalg") as mock_jl,
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            patch.dict("os.environ", {"JLINALG_NO_VENDOR_LAPACK": "1"}),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_jl.blas_is_ilp64 = 0
            mock_jl.blas_has_dsyevd = False
            mock_jl.blas_has_dsyevr = False

            try:
                eigendecompose_kinship(big_K)
            except Exception:
                pass

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0, (
                "LP64 warning should not fire when no_vendor forces np.linalg.eigh"
            )


class TestLOCOIteratorRuntimeError:
    """SAFE-02: LOCO iterator None raises RuntimeError, not bare assert.

    The internal error path (loco_iter=None when eigen_cache=None) is deep in the
    LOCO pipeline and difficult to trigger without a full dataset. We read loco.py
    as text to verify the guard uses ``raise RuntimeError`` (not bare ``assert``),
    following the same pattern as test_lapack_no_ffast_math.
    """

    def test_loco_module_importable(self):
        """loco module imports without error."""
        from jamma.lmm import loco

        assert loco  # module loaded successfully

    def test_run_lmm_loco_requires_bed_path(self):
        """run_lmm_loco raises TypeError when required args missing."""
        from jamma.lmm.loco import run_lmm_loco

        with pytest.raises(TypeError):
            run_lmm_loco()  # type: ignore[call-arg]

    def test_loco_iter_none_raises_runtime_error(self):
        """loco.py guards loco_iter=None with RuntimeError, not bare assert.

        Bare ``assert`` is stripped by ``python -O``, which would cause a
        cryptic AttributeError downstream instead of a clear diagnostic.
        """
        import re
        from pathlib import Path

        loco_src = (
            Path(__file__).resolve().parent.parent / "src" / "jamma" / "lmm" / "loco.py"
        )
        source = loco_src.read_text()

        # Find the guard: ``if loco_iter is None:`` followed by ``raise RuntimeError``
        pattern = re.compile(
            r"if\s+loco_iter\s+is\s+None\s*:\s*\n\s+raise\s+RuntimeError\(",
        )
        assert pattern.search(source), (
            "loco.py must guard loco_iter=None with 'raise RuntimeError(...)' "
            "not bare assert — bare assert is stripped by python -O"
        )


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

    def test_abi_mismatch_raises_import_error(self):
        """ABI mismatch guard uses ``raise ImportError``, not silent fallback.

        The ABI check runs at import time and cannot be triggered via reload
        (the constant is re-initialised before the check). We read __init__.py
        as text and verify the guard structure, following the same pattern as
        test_lapack_no_ffast_math.
        """
        import re
        from pathlib import Path

        init_src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "jamma"
            / "jlinalg"
            / "__init__.py"
        )
        source = init_src.read_text()

        # Verify: ``if ABI_VERSION != _EXPECTED_JLINALG_ABI:`` followed by
        # ``raise ImportError(`` — the guard must exist and use ImportError
        pattern = re.compile(
            r"if\s+ABI_VERSION\s*!=\s*_EXPECTED_JLINALG_ABI\s*:\s*\n\s+raise\s+ImportError\(",
        )
        assert pattern.search(source), (
            "jlinalg/__init__.py must check ABI_VERSION != _EXPECTED_JLINALG_ABI "
            "and raise ImportError — this guard prevents silent ABI drift"
        )
