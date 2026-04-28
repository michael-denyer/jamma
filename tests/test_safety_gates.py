"""Tests for safety gates (SAFE-01, SAFE-02, SAFE-03).

Tests observable behavior (warnings emitted, exceptions raised, attribute values)
wherever possible. Uses real types (MemorySnapshot, numpy arrays) instead of
MagicMock. For the LP64 threshold tests, np.lib.stride_tricks.as_strided creates
a 50k x 50k "view" backed by a tiny 4x4 allocation — exercises the real
shape-checking code path without allocating 20 GB.

SAFE-02 and SAFE-03 use source-reading tests (read file as text + regex) because
the guards are unreachable at test time: SAFE-02's loco_iter=None is a dead-code
defensive check, and SAFE-03's ABI check runs at import time before test code
executes. These follow the same pattern as test_lapack_no_ffast_math (build/config
verification via text inspection).
"""

import contextlib
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.memory import MemorySnapshot

pytestmark = pytest.mark.tier0


def _make_memory_snapshot(available_gb: float = 1000.0) -> MemorySnapshot:
    """Build a real MemorySnapshot for test use."""
    return MemorySnapshot(
        rss_gb=1.0,
        vms_gb=2.0,
        available_gb=available_gb,
        total_gb=1024.0,
        percent_used=10.0,
    )


def _fake_eigh(
    return_value: tuple[np.ndarray, np.ndarray],
):
    """Build a fake eigh matching the real ``jlinalg.eigh(K, inplace=False)`` signature.

    Unlike MagicMock, this rejects unknown keyword arguments, catching
    call-site drift between tests and the real interface.
    """

    def _eigh(K: np.ndarray, inplace: bool = False) -> tuple[np.ndarray, np.ndarray]:
        return return_value

    return _eigh


def _fake_jlinalg(
    *,
    blas_is_ilp64: int = 0,
    blas_has_dsyevd: bool = True,
    blas_has_dsyevr: bool = False,
    blas_backend: str = "test",
    eigh_return: tuple | None = None,
) -> SimpleNamespace:
    """Build a fake jlinalg with only the attributes eigen.py reads.

    Using SimpleNamespace instead of MagicMock ensures that accessing an
    attribute not listed here raises AttributeError, catching drift between
    the test fake and the real module interface.
    """
    default_return = (np.ones(100), np.eye(100))
    return SimpleNamespace(
        blas_is_ilp64=blas_is_ilp64,
        blas_has_dsyevd=blas_has_dsyevd,
        blas_has_dsyevr=blas_has_dsyevr,
        blas_backend=blas_backend,
        eigh=_fake_eigh(eigh_return or default_return),
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
        fake_jl = _fake_jlinalg(
            blas_is_ilp64=0,
            eigh_return=(np.ones(50_000), np.eye(3)),
        )

        with (
            patch.object(eigen, "jlinalg", fake_jl),
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            # eigh fake returns wrong shape — we only care about the
            # LP64-detected warning triggered before the fake is invoked.
            with contextlib.suppress(ValueError, RuntimeError):
                eigen.eigendecompose_kinship(big_K, check_memory=False)

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
        fake_jl = _fake_jlinalg(
            blas_is_ilp64=1,
            eigh_return=(np.ones(50_000), np.eye(3)),
        )

        with (
            patch.object(eigen, "jlinalg", fake_jl),
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            with contextlib.suppress(ValueError, RuntimeError):
                eigen.eigendecompose_kinship(big_K, check_memory=False)

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0

    def test_small_matrix_no_warning(self):
        """n_samples <= 40k does not trigger warning even on LP64."""
        from jamma.lmm import eigen

        K = np.eye(100, dtype=np.float64)
        snapshot = _make_memory_snapshot()
        fake_jl = _fake_jlinalg(blas_is_ilp64=0)

        with (
            patch.object(eigen, "jlinalg", fake_jl),
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            eigen.eigendecompose_kinship(K)

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0

    def test_boundary_40k_no_warning(self):
        """Exactly 40,000 samples does not trigger LP64 warning."""
        from jamma.lmm import eigen

        K = _big_K_view(40_000)
        snapshot = _make_memory_snapshot()
        fake_jl = _fake_jlinalg(
            blas_is_ilp64=0,
            eigh_return=(np.ones(40_000), np.eye(3)),
        )

        with (
            patch.object(eigen, "jlinalg", fake_jl),
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            with contextlib.suppress(ValueError, RuntimeError):
                eigen.eigendecompose_kinship(K, check_memory=False)

            lp64_warnings = [x for x in w if "LP64" in str(x.message)]
            assert len(lp64_warnings) == 0

    def test_boundary_40001_warns(self):
        """40,001 samples triggers LP64 warning."""
        from jamma.lmm import eigen

        K = _big_K_view(40_001)
        snapshot = _make_memory_snapshot()
        fake_jl = _fake_jlinalg(
            blas_is_ilp64=0,
            eigh_return=(np.ones(40_001), np.eye(3)),
        )

        with (
            patch.object(eigen, "jlinalg", fake_jl),
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            with contextlib.suppress(ValueError, RuntimeError):
                eigen.eigendecompose_kinship(K, check_memory=False)

            lp64_warnings = [x for x in w if "LP64 BLAS detected" in str(x.message)]
            assert len(lp64_warnings) == 1

    def test_no_vendor_suppresses_lp64_warning(self):
        """LP64 warning does not fire when using np.linalg.eigh fallback."""
        from jamma.lmm import eigen

        big_K = _big_K_view(50_000)
        snapshot = _make_memory_snapshot()
        fake_jl = _fake_jlinalg(
            blas_is_ilp64=0,
            blas_has_dsyevd=False,
            blas_has_dsyevr=False,
        )

        # Patch eigh at the jamma import site (not numpy globally) and have
        # it short-circuit with a RuntimeError. The assertion only checks the
        # warning-routing branch, so the eigh return value is irrelevant —
        # short-circuiting is safer than fabricating a fake numerical result.
        with (
            patch.object(eigen, "jlinalg", fake_jl),
            patch.object(eigen, "log_memory_snapshot", return_value=snapshot),
            patch.dict("os.environ", {"JLINALG_NO_VENDOR_LAPACK": "1"}),
            patch.object(
                eigen.np.linalg, "eigh", side_effect=RuntimeError("test stub")
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            # Lock down two invariants:
            # 1. The RuntimeError from the patched np.linalg.eigh PROPAGATES
            #    to the caller (no silent catch returning a default result).
            # 2. The LP64 warning is NOT emitted on this routing branch.
            with pytest.raises(RuntimeError, match="test stub"):
                eigen.eigendecompose_kinship(big_K, check_memory=False)

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

        The ABI check runs at import time and cannot be re-triggered via
        importlib.reload (the constant is re-initialised before the check
        fires). We read __init__.py as text to verify the guard structure,
        following the same pattern as test_lapack_no_ffast_math.

        The behavioral counterpart is test_abi_matches_current above, which
        confirms the check passed at import (ABI_VERSION == expected).
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

        pattern = re.compile(
            r"if\s+ABI_VERSION\s*!=\s*_EXPECTED_JLINALG_ABI\s*:\s*\n\s+raise\s+ImportError\(",
        )
        assert pattern.search(source), (
            "jlinalg/__init__.py must check ABI_VERSION != _EXPECTED_JLINALG_ABI "
            "and raise ImportError — this guard prevents silent ABI drift"
        )
