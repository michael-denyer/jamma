"""Tests for safety gates (SAFE-02, SAFE-03).

Tests observable behavior (warnings emitted, exceptions raised, attribute values)
wherever possible.

SAFE-02 and SAFE-03 exercise the guards at runtime. SAFE-02 uses a ``python -O``
subprocess to verify the guard's ``RuntimeError`` survives bytecode optimisation
(a bare ``assert`` would not). SAFE-03 checks the ABI guard where it now lives,
``recompile._import_and_validate`` (a wrong ABI or a missing required symbol
returns None), and stubs the C extension in ``sys.modules`` with a wrong
``ABI_VERSION`` *before* importing ``jamma.jlinalg`` so the production
import-time fallback fires; the loader is faked so the test observes the
warn-and-fallback path rather than a silent rebuild.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.fixture_paths import LOCO

pytestmark = pytest.mark.tier0

LOCO_BFILE = LOCO.bfile


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

        from jamma._build_support.build_models import BuildSpec
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
            fake.set_n_threads = lambda *a, **k: None
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
