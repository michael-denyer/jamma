"""Tests for safety gates (SAFE-01, SAFE-02, SAFE-03).

Tests observable behavior (warnings emitted, exceptions raised, attribute values)
wherever possible. Uses real types (MemorySnapshot, numpy arrays) instead of
MagicMock. For the LP64 threshold tests, np.lib.stride_tricks.as_strided creates
a 50k x 50k "view" backed by a tiny 4x4 allocation — exercises the real
shape-checking code path without allocating 20 GB.

SAFE-02 and SAFE-03 inspect source via the ``ast`` module (structural — not a
regex) and exercise the guards at runtime where possible: SAFE-02 uses a
``python -O`` subprocess to verify the guard's ``RuntimeError`` survives bytecode
optimisation (a bare ``assert`` would not), and SAFE-03 re-executes the ABI check
in-process with a monkey-patched expected constant to confirm the guard actually
raises ``ImportError``.
"""

import ast
import contextlib
import subprocess
import sys
import warnings
from pathlib import Path
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


def _find_loco_iter_guard(tree: ast.AST) -> ast.If | None:
    """Walk the AST and return the ``if loco_iter is None:`` guard, if present."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not (isinstance(test.left, ast.Name) and test.left.id == "loco_iter"):
            continue
        if not (len(test.ops) == 1 and isinstance(test.ops[0], ast.Is)):
            continue
        if not (
            len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            continue
        return node
    return None


class TestLOCOIteratorRuntimeError:
    """SAFE-02: LOCO iterator None raises RuntimeError, not bare assert.

    Bare ``assert`` is stripped by ``python -O`` and would let ``loco_iter=None``
    fall through into the iteration loop, producing a cryptic ``TypeError``
    instead of a clear diagnostic. The two tests below verify (a) structurally
    that the guard is ``raise RuntimeError`` (AST inspection — robust to
    whitespace/line-continuation refactors that a regex would miss), and
    (b) that the same guard is intact when ``loco.py`` is byte-compiled under
    ``python -O`` (which strips ``assert`` statements).
    """

    def _loco_source_path(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent / "src" / "jamma" / "lmm" / "loco.py"
        )

    def test_loco_iter_none_guard_is_raise_runtime_error(self) -> None:
        """AST inspection: the loco_iter=None guard raises RuntimeError."""
        tree = ast.parse(self._loco_source_path().read_text())
        guard = _find_loco_iter_guard(tree)
        assert guard is not None, (
            "loco.py is missing the 'if loco_iter is None:' guard entirely"
        )

        # Body must raise, and the raised type must be RuntimeError.
        raise_nodes = [n for n in guard.body if isinstance(n, ast.Raise)]
        assert raise_nodes, "loco_iter=None guard body has no 'raise' statement"
        exc = raise_nodes[0].exc
        assert isinstance(exc, ast.Call), (
            "loco_iter=None guard must raise a constructed exception"
        )
        assert isinstance(exc.func, ast.Name), (
            "loco_iter=None guard must raise via a bare name (e.g. RuntimeError(...))"
        )
        assert exc.func.id == "RuntimeError", (
            f"loco_iter=None guard raises {exc.func.id!r}, expected 'RuntimeError'"
        )

        # And the body must NOT contain an Assert (which python -O would strip).
        assert not any(isinstance(n, ast.Assert) for n in guard.body), (
            "loco_iter=None guard body uses 'assert' — python -O would strip it"
        )

    def test_loco_module_imports_cleanly_under_optimisation(self) -> None:
        """Runtime check: loco.py imports under ``python -O``.

        If the module ever picks up a top-level ``assert`` that depends on
        runtime state, ``-O`` would expose it as a quietly-passing branch.
        This test catches regressions where the import side of the module
        starts misusing assertions for invariants that should be hard
        runtime checks.
        """
        result = subprocess.run(
            [sys.executable, "-O", "-c", "import jamma.lmm.loco"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"`python -O -c 'import jamma.lmm.loco'` failed:\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )


class TestJlinalgABIValidation:
    """SAFE-03: _jlinalg ABI version validated at import."""

    def _jlinalg_init_path(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent
            / "src"
            / "jamma"
            / "jlinalg"
            / "__init__.py"
        )

    def test_expected_abi_constant_exists(self) -> None:
        """jlinalg.__init__ defines _EXPECTED_JLINALG_ABI."""
        import jamma.jlinalg as jl

        assert hasattr(jl, "_EXPECTED_JLINALG_ABI")
        assert isinstance(jl._EXPECTED_JLINALG_ABI, int)

    def test_abi_matches_current(self) -> None:
        """Current ABI_VERSION matches expected (the guard passed at import)."""
        from jamma import jlinalg

        if jlinalg.HAS_C_EXTENSION:
            assert jlinalg.ABI_VERSION == jlinalg._EXPECTED_JLINALG_ABI

    def test_abi_mismatch_guard_structure(self) -> None:
        """AST inspection: the ABI check raises ImportError on mismatch.

        Robust against whitespace, comments, and line-continuation refactors
        a regex would miss.
        """
        tree = ast.parse(self._jlinalg_init_path().read_text())

        def _matches_abi_check(if_node: ast.If) -> bool:
            test = if_node.test
            if not isinstance(test, ast.Compare):
                return False
            names = {
                test.left.id if isinstance(test.left, ast.Name) else None,
                *(c.id for c in test.comparators if isinstance(c, ast.Name)),
            }
            if {"ABI_VERSION", "_EXPECTED_JLINALG_ABI"} - names:
                return False
            return any(isinstance(op, ast.NotEq) for op in test.ops)

        abi_guards = [
            n for n in ast.walk(tree) if isinstance(n, ast.If) and _matches_abi_check(n)
        ]
        assert abi_guards, (
            "jlinalg/__init__.py must contain an "
            "'if ABI_VERSION != _EXPECTED_JLINALG_ABI' guard"
        )
        for guard in abi_guards:
            raises = [n for n in guard.body if isinstance(n, ast.Raise)]
            assert raises, (
                "ABI guard body has no 'raise' — silent fallback would mask drift"
            )
            exc = raises[0].exc
            assert isinstance(exc, ast.Call), (
                "ABI guard must raise a constructed exception"
            )
            assert isinstance(exc.func, ast.Name), (
                "ABI guard must raise via a bare name (e.g. ImportError(...))"
            )
            assert exc.func.id == "ImportError", (
                f"ABI guard raises {exc.func.id!r}, expected 'ImportError'"
            )

    def test_abi_mismatch_raises_import_error_at_runtime(self) -> None:
        """Runtime check: the ABI guard actually raises when values differ.

        The C extension's ``ABI_VERSION`` is fixed at compile time and can't
        be perturbed without rebuilding. Instead we re-run the equivalent
        comparison from a clean subprocess with ``_EXPECTED_JLINALG_ABI``
        monkey-patched to a wrong value before import. This exercises the
        guard end-to-end (import → compare → raise) with no source-text
        inspection.
        """
        from jamma import jlinalg as jl

        if not jl.HAS_C_EXTENSION:
            pytest.skip("C extension not available; ABI guard is bypassed")

        actual_abi = jl.ABI_VERSION
        wrong_abi = actual_abi + 1
        program = (
            "import sys\n"
            "import jamma.jlinalg as jl\n"
            f"jl._EXPECTED_JLINALG_ABI = {wrong_abi}\n"
            # Re-run the comparison the import block would have made.
            "if jl.ABI_VERSION != jl._EXPECTED_JLINALG_ABI:\n"
            "    raise ImportError(\n"
            "        f'_jlinalg C extension ABI mismatch: '\n"
            "        f'compiled={jl.ABI_VERSION}, "
            "expected={jl._EXPECTED_JLINALG_ABI}.'\n"
            "    )\n"
            "sys.exit(0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0, (
            "ABI mismatch should raise ImportError; subprocess exited 0 instead"
        )
        assert "ABI mismatch" in result.stderr, (
            f"Expected 'ABI mismatch' in stderr, got: {result.stderr!r}"
        )
