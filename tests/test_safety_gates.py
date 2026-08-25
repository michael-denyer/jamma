"""Tests for safety gates (SAFE-01, SAFE-02, SAFE-03).

Tests observable behavior (warnings emitted, exceptions raised, attribute values)
wherever possible. Uses real types (MemorySnapshot, numpy arrays) instead of
MagicMock. For the LP64 threshold tests, np.lib.stride_tricks.as_strided creates
a 50k x 50k "view" backed by a tiny 4x4 allocation — exercises the real
shape-checking code path without allocating 20 GB.

SAFE-02 and SAFE-03 exercise the guards at runtime. SAFE-02 uses a ``python -O``
subprocess to verify the guard's ``RuntimeError`` survives bytecode optimisation
(a bare ``assert`` would not). SAFE-03 checks the ABI guard where it now lives,
``recompile._import_and_validate`` (a wrong ABI or a missing required symbol
returns None), and stubs the C extension in ``sys.modules`` with a wrong
``ABI_VERSION`` *before* importing ``jamma.jlinalg`` so the production
import-time fallback fires; the loader is faked so the test observes the
warn-and-fallback path rather than a silent rebuild.
"""

import contextlib
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.memory_snapshot import MemorySnapshot

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

    CAVEAT: Tests in TestLP64OverflowWarning depend on
    ``eigendecompose_kinship`` not materialising the matrix before the
    LP64 overflow check fires (otherwise the strided view would be
    forced into a real ~20 GB allocation). If anyone adds a defensive
    ``np.allclose(K, K.T)`` symmetry check or copies ``K`` to ensure
    contiguity *before* the overflow guard, every test in this class
    will fail with an OOM-shaped traceback that looks unrelated. Move
    such validation steps after the overflow check.
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


LOCO_BFILE = Path(__file__).parent / "fixtures" / "gemma_loco" / "test"


class TestLOCOIteratorRuntimeError:
    """SAFE-02: LOCO must not rely on assertions stripped by ``python -O``.

    This once guarded a ``loco_iter=None`` fall-through with an explicit
    ``raise``, and was pinned by inspecting loco.py's AST for that ``if``.
    The eigen source is now chosen once and its iterator is built and consumed
    inside the same branch, so the variable never crosses a branch boundary and
    there is no None state left to guard. The structural test went with the
    structure it described; what remains is the behaviour that mattered, run
    under ``-O`` where any surviving ``assert`` is stripped.
    """

    def _loco_source_path(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent / "src" / "jamma" / "lmm" / "loco.py"
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

    def test_loco_runs_end_to_end_under_optimisation(self, tmp_path: Path) -> None:
        """A real LOCO run completes under ``-O``, where asserts are stripped.

        The import check alone would miss an assertion used as a runtime
        invariant inside ``run_lmm_loco``. Driving an actual two-chromosome
        analysis exercises the eigen-source setup and the chromosome loop with
        every ``assert`` removed, which is the failure SAFE-02 exists to catch.
        """
        script = (
            "from pathlib import Path\n"
            "import numpy as np\n"
            "from jamma.lmm.loco import run_lmm_loco\n"
            "from jamma.lmm.schema import LmmConfig\n"
            f"bfile = Path({str(LOCO_BFILE)!r})\n"
            "fam = np.loadtxt(bfile.with_suffix('.fam'), dtype=str, ndmin=2)\n"
            "pheno = fam[:, 5].astype(np.float64)\n"
            f"out = Path({str(tmp_path)!r}) / 'o.assoc.txt'\n"
            "r = run_lmm_loco(bed_path=bfile, phenotypes=pheno,\n"
            "                 output_path=out,\n"
            "                 config=LmmConfig(show_progress=False,\n"
            "                                  check_memory=False))\n"
            "assert r.n_tested > 0\n"
            "print('TESTED', r.n_tested)\n"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"LOCO run under `python -O` failed:\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr[-3000:]!r}"
        )
        assert "TESTED" in result.stdout


class TestJlinalgABIValidation:
    """SAFE-03: _jlinalg ABI version validated at import."""

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

    def test_abi_mismatch_rejected_by_loader(self, monkeypatch) -> None:
        """The ABI guard lives in recompile._import_and_validate now.

        A module whose ABI_VERSION differs from expected is rejected (returns
        None), as is one that is ABI-matched but missing a required symbol; a
        matching module with every required symbol is accepted. This asserts the
        observable behaviour of the guard rather than the source shape of the
        old inline ``raise ImportError``.
        """
        import types

        from jamma._build_support.compile_and_link import BuildSpec
        from jamma.core.recompile import _import_and_validate

        key = "jamma._fake_abi_probe"
        spec = BuildSpec(
            package_parts=(),
            source_parts=(),
            include_parts=(),
            sources=(),
            lapack_sources=(),
            output_stem="_fake_abi_probe",
            module_name="_fake_abi_probe",
            sys_module_key=key,
            required_attrs=("NEEDED",),
        )

        matched = types.ModuleType(key)
        matched.ABI_VERSION = 14  # type: ignore[attr-defined]
        matched.NEEDED = object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, key, matched)
        assert _import_and_validate(spec, 14) is matched

        # Right ABI is not enough — a missing required symbol is rejected too.
        wrong_abi = types.ModuleType(key)
        wrong_abi.ABI_VERSION = 13  # type: ignore[attr-defined]
        wrong_abi.NEEDED = object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, key, wrong_abi)
        assert _import_and_validate(spec, 14) is None

        missing_symbol = types.ModuleType(key)
        missing_symbol.ABI_VERSION = 14  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, key, missing_symbol)
        assert _import_and_validate(spec, 14) is None

    def test_abi_mismatch_triggers_production_guard_at_runtime(self) -> None:
        """Production guard fires when ``_jlinalg.ABI_VERSION`` is wrong.

        The C extension's ``ABI_VERSION`` is fixed at compile time and can't
        be perturbed without rebuilding. Instead, the subprocess installs a
        fake ``jamma.jlinalg._jlinalg`` module into ``sys.modules`` BEFORE
        importing ``jamma.jlinalg``, so the production import block in
        ``jlinalg/__init__.py`` reads the fake's wrong ABI_VERSION and the
        production guard fires.

        Auto-recompile is also stubbed out (returns False) — otherwise the
        production fallback path would try to rebuild the real extension
        and the test would observe the rebuilt-good module instead of the
        guard's drift-handling. With recompile disabled, the fallback
        emits a deterministic warning and ``HAS_C_EXTENSION`` becomes
        ``False``. We assert on those observable side-effects.
        """
        from jamma import jlinalg as jl

        if not jl.HAS_C_EXTENSION:
            pytest.skip("C extension not available; ABI guard is bypassed")

        wrong_abi = jl._EXPECTED_JLINALG_ABI - 1
        program = textwrap.dedent(
            f"""
            import sys
            import types
            import warnings
            from importlib.machinery import ModuleSpec

            # Stub the C extension before jamma.jlinalg is imported. Production
            # code at jamma/jlinalg/__init__.py reads ABI_VERSION from this
            # module; we plant the wrong value to drive the guard.
            fake = types.ModuleType("jamma.jlinalg._jlinalg")
            # ``importlib.util.find_spec`` requires a ModuleSpec on the
            # module object — production code calls ``find_spec`` early
            # to decide whether the .so exists.
            fake.__spec__ = ModuleSpec("jamma.jlinalg._jlinalg", loader=None)
            fake.ABI_VERSION = {wrong_abi}
            fake.HAS_OPENMP = False
            fake.blas_backend = "fake"
            fake.blas_has_dgeqrf = 0
            fake.blas_has_dgesvd = 0
            fake.blas_has_dsyevd = 0
            fake.blas_has_dsyevr = 0
            fake.blas_has_dsyrk = 0
            fake.blas_has_lapacke_dsyevd = 0
            fake.blas_is_ilp64 = 0
            fake.compute_snp_stats_chunk = lambda *a, **k: None
            fake.dgemm = lambda *a, **k: None
            fake.dsyrk = lambda *a, **k: None
            fake.eigh = lambda *a, **k: None
            fake.get_n_threads = lambda *a, **k: 1
            fake.jlinalg_isa = "fake"
            fake.qr = lambda *a, **k: None
            fake.set_n_threads = lambda *a, **k: None
            fake.svd = lambda *a, **k: None
            sys.modules["jamma.jlinalg._jlinalg"] = fake

            # Fully fake jamma.core.recompile BEFORE importing jamma, so no real
            # recompile can paper over the simulated ABI mismatch by rebuilding
            # (importing the real module would eager-load jamma.jlinalg and
            # rebuild). The fake _load_c_module runs the same ABI check the real
            # one does against the planted fake, so the wrong ABI drives None and
            # jamma.jlinalg falls back.
            recompile_mod = types.ModuleType("jamma.core.recompile")

            def _fake_load(spec, expected_abi):
                mod = sys.modules.get(spec.sys_module_key)
                if mod is None or getattr(mod, "ABI_VERSION", None) != expected_abi:
                    return None
                return mod

            recompile_mod._load_c_module = _fake_load
            sys.modules["jamma.core.recompile"] = recompile_mod

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                import jamma.jlinalg as jl

            assert jl.HAS_C_EXTENSION is False, (
                "ABI mismatch should drive HAS_C_EXTENSION to False; "
                f"got HAS_C_EXTENSION={{jl.HAS_C_EXTENSION!r}}"
            )
            assert jl.ABI_VERSION == 0, (
                "fallback should reset ABI_VERSION to 0; "
                f"got ABI_VERSION={{jl.ABI_VERSION!r}}"
            )
            warning_text = "\\n".join(str(w.message) for w in captured)
            assert "ABI mismatch" in warning_text, (
                "production fallback warning must mention 'ABI mismatch'; "
                f"got warnings: {{warning_text!r}}"
            )
            assert "Falling back to NumPy" in warning_text, (
                "production fallback warning must announce NumPy fallback; "
                f"got warnings: {{warning_text!r}}"
            )
            print("PASS")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"ABI guard runtime test failed (exit={result.returncode}):\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        assert "PASS" in result.stdout, (
            f"Subprocess did not print PASS; stdout={result.stdout!r}"
        )
