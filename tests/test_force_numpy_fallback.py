"""Tests for the JAMMA_FORCE_NUMPY_FALLBACK env-var gate at the two C
extension import shims (jamma.jlinalg.__init__ and
jamma.core.recompile._load_c_module).

The gate is the load-bearing knob the ASAN/UBSAN
sanitizer workflow uses to skip the .so imports entirely — see
RESEARCH.md §"Pitfall 4" (ASAN + dlopen interaction can produce
false-positive heap-buffer-overflow reports inside dispatched BLAS calls).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
import warnings
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.tier0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_jlinalg():
    """Reload jamma.jlinalg so module-level env-var read takes effect."""
    import jamma.jlinalg as j

    return importlib.reload(j)


@pytest.fixture
def reload_jlinalg_after_test(monkeypatch):
    """Reload jlinalg with the env in its current state, and again at teardown
    with JAMMA_FORCE_NUMPY_FALLBACK explicitly cleared, so the post-test module
    state doesn't leak into other tests in the same process.
    """
    yield
    monkeypatch.delenv("JAMMA_FORCE_NUMPY_FALLBACK", raising=False)
    _reload_jlinalg()


# ---------------------------------------------------------------------------
# jlinalg gate
# ---------------------------------------------------------------------------


def test_force_numpy_default_not_set(monkeypatch, reload_jlinalg_after_test):
    """Unset env: blas_backend is NOT 'numpy-fallback-forced'."""
    monkeypatch.delenv("JAMMA_FORCE_NUMPY_FALLBACK", raising=False)
    j = _reload_jlinalg()
    assert j.blas_backend != "numpy-fallback-forced"


def test_force_numpy_set_to_1(monkeypatch, reload_jlinalg_after_test):
    """Truthy env: HAS_C_EXTENSION False, blas_backend forced, ABI_VERSION 0."""
    monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "1")
    j = _reload_jlinalg()
    assert j.HAS_C_EXTENSION is False
    assert j.blas_backend == "numpy-fallback-forced"
    assert j.blas_is_ilp64 == 0
    assert j.ABI_VERSION == 0


def test_force_numpy_set_to_zero_treated_as_off(monkeypatch, reload_jlinalg_after_test):
    """'0' is treated as off."""
    monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "0")
    j = _reload_jlinalg()
    assert j.blas_backend != "numpy-fallback-forced"


def test_force_numpy_empty_treated_as_off(monkeypatch, reload_jlinalg_after_test):
    """'' is treated as off."""
    monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "")
    j = _reload_jlinalg()
    assert j.blas_backend != "numpy-fallback-forced"


@pytest.mark.parametrize("value", ["true", "yes", "anything", "TRUE", " 1 "])
def test_force_numpy_other_truthy_values(monkeypatch, reload_jlinalg_after_test, value):
    """Anything not in {'', '0'} engages the gate (after .strip())."""
    monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", value)
    j = _reload_jlinalg()
    assert j.blas_backend == "numpy-fallback-forced", (
        f"value={value!r} did not engage the gate"
    )


def test_force_numpy_no_warning_emitted(monkeypatch, reload_jlinalg_after_test):
    """Forced fallback is a deliberate choice — no warnings.warn fires."""
    monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "1")
    # Drop any cached module so reload re-runs the module-level code.
    sys.modules.pop("jamma.jlinalg", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("jamma.jlinalg")
    # Filter out unrelated DeprecationWarning noise from the test infra.
    forced_warnings = [
        w
        for w in caught
        if "jlinalg" in str(w.message).lower() or "fallback" in str(w.message).lower()
    ]
    assert forced_warnings == [], [str(w.message) for w in forced_warnings]


def test_force_numpy_fallback_functions_callable(
    monkeypatch, reload_jlinalg_after_test
):
    """After forcing fallback, the module-level eigh/dgemm/dsyrk produce
    correct results on tiny inputs — proves the existing fallback block
    was actually reached.
    """
    monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "1")
    j = _reload_jlinalg()
    K = np.eye(4)
    w, v = j.eigh(K)
    assert w.shape == (4,)
    assert v.shape == (4, 4)
    np.testing.assert_allclose(w, np.ones(4))
    # dsyrk: K = X @ X.T
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K2 = j.dsyrk(X)
    np.testing.assert_allclose(K2, X @ X.T)
    out = np.eye(3)
    returned = j.dsyrk(X, out=out, beta=2.0)
    assert returned is out
    np.testing.assert_allclose(out, X @ X.T + 2.0 * np.eye(3))


def test_force_numpy_dsyrk_output_validation(monkeypatch, reload_jlinalg_after_test):
    """The NumPy DSYRK path enforces the same output contract as C."""
    monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "1")
    j = _reload_jlinalg()
    X = np.ones((3, 2))

    with pytest.raises(ValueError, match="2-D"):
        j.dsyrk(np.ones(3))
    with pytest.raises(ValueError, match="beta requires out"):
        j.dsyrk(X, beta=1.0)
    with pytest.raises(ValueError, match="shape"):
        j.dsyrk(X, out=np.empty((3, 4)))
    with pytest.raises(ValueError, match="float64"):
        j.dsyrk(X, out=np.empty((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        j.dsyrk(X, out=np.empty((3, 6))[:, ::2])

    readonly = np.empty((3, 3))
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        j.dsyrk(X, out=readonly)

    out = np.full((3, 3), np.nan)
    assert j.dsyrk(X, out=out) is out
    np.testing.assert_allclose(out, X @ X.T)


def test_natural_fallback_blas_backend_value(monkeypatch, reload_jlinalg_after_test):
    """When the gate is NOT engaged AND the .so import fails (forced via
    monkeypatching) AND the auto-recompile retry also fails, blas_backend
    == 'numpy-fallback' (not 'numpy-fallback-forced'). Regression guard for
    the globals().setdefault path: if line ~239 used `dir()` or naked
    assignment instead of setdefault, this would still pass — but if the
    forced branch had set 'numpy-fallback-forced' before the natural-
    fallback block ran, the natural-fallback block must NOT clobber it.
    The mirror-image regression (the natural path also being correct when
    forced is unset) is guarded here.
    """
    monkeypatch.delenv("JAMMA_FORCE_NUMPY_FALLBACK", raising=False)
    # Make find_spec report no .so, and pre-populate sys.modules with a stub
    # that will raise ImportError on attribute access during the from-import.
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "jamma.jlinalg._jlinalg":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    # Also block the auto-recompile retry — without this, the recompile
    # actually rebuilds the .so and the test sees HAS_C_EXTENSION=True.
    import jamma.core.recompile as recompile_mod

    monkeypatch.setattr(
        recompile_mod, "auto_recompile_c_extension", lambda *a, **kw: False
    )

    sys.modules.pop("jamma.jlinalg._jlinalg", None)
    sys.modules.pop("jamma.jlinalg", None)
    # Pre-populate _jlinalg as None so the `from ... import` raises ImportError.
    sys.modules["jamma.jlinalg._jlinalg"] = None  # type: ignore[assignment]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            j = importlib.import_module("jamma.jlinalg")
        assert j.HAS_C_EXTENSION is False
        assert j.blas_backend == "numpy-fallback", j.blas_backend
    finally:
        sys.modules.pop("jamma.jlinalg._jlinalg", None)
        sys.modules.pop("jamma.jlinalg", None)


# ---------------------------------------------------------------------------
# lmm._lmm_accel gate
# ---------------------------------------------------------------------------


class TestForceNumpyLmmAccel:
    """Tests for the JAMMA_FORCE_NUMPY_FALLBACK gate inside
    jamma.core.recompile._load_c_module for LMM_ACCEL_SPEC."""

    def test_returns_unavailable(self, monkeypatch):
        """Forced env: _load_c_module reports no extension."""
        monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "1")
        from jamma._build_support.compile_and_link import LMM_ACCEL_SPEC
        from jamma.core.recompile import _load_c_module
        from jamma.lmm.compute_numpy import _EXPECTED_ABI_VERSION

        assert _load_c_module(LMM_ACCEL_SPEC, _EXPECTED_ABI_VERSION) is None

    def test_no_so_import_attempted(self, monkeypatch):
        """Forced env: no .so import attempted — sys.modules stays clean."""
        monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "1")
        from jamma._build_support.compile_and_link import LMM_ACCEL_SPEC
        from jamma.core.recompile import _load_c_module
        from jamma.lmm.compute_numpy import _EXPECTED_ABI_VERSION

        sys.modules.pop("jamma.lmm._lmm_accel", None)
        _load_c_module(LMM_ACCEL_SPEC, _EXPECTED_ABI_VERSION)
        assert "jamma.lmm._lmm_accel" not in sys.modules

    def test_unset_preserves_behavior(self, monkeypatch):
        """Unset env: the result is the natural import outcome.

        Whether that is the module or None depends on the build, so pin the
        shape rather than the value.
        """
        monkeypatch.delenv("JAMMA_FORCE_NUMPY_FALLBACK", raising=False)
        from jamma._build_support.compile_and_link import LMM_ACCEL_SPEC
        from jamma.core.recompile import _load_c_module
        from jamma.lmm.compute_numpy import _EXPECTED_ABI_VERSION

        result = _load_c_module(LMM_ACCEL_SPEC, _EXPECTED_ABI_VERSION)
        assert result is None or isinstance(result, ModuleType)

    @pytest.mark.parametrize("value", ["", "0"])
    def test_off_values(self, monkeypatch, value):
        """'' and '0' are off — the gate does not short-circuit.

        Both an engaged gate and a natural import failure return None, so this
        is only decidable where the extension does import. Where it does, an
        off value must yield the module.
        """
        monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", value)
        from jamma._build_support.compile_and_link import LMM_ACCEL_SPEC
        from jamma.core.recompile import _load_c_module
        from jamma.lmm.compute_numpy import _EXPECTED_ABI_VERSION

        try:
            import jamma.lmm._lmm_accel  # noqa: F401
        except ImportError:
            pytest.skip("extension not built, so the gate is not observable here")
        assert _load_c_module(LMM_ACCEL_SPEC, _EXPECTED_ABI_VERSION) is not None, (
            f"value={value!r} engaged the gate"
        )

    def test_truthy_values_engage_gate(self, monkeypatch):
        """Multiple truthy values all engage the gate."""
        from jamma._build_support.compile_and_link import LMM_ACCEL_SPEC
        from jamma.core.recompile import _load_c_module
        from jamma.lmm.compute_numpy import _EXPECTED_ABI_VERSION

        for value in ["1", "true", "yes", "TRUE", " 1 "]:
            monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", value)
            assert _load_c_module(LMM_ACCEL_SPEC, _EXPECTED_ABI_VERSION) is None, (
                f"value={value!r} did not engage the gate"
            )


def test_gate_holds_across_the_whole_backend_selection_path():
    """The gate must hold for every consumer, not just the one loader call.

    ``test_no_so_import_attempted`` calls that one shim directly, so a second
    probe importing ``jamma.lmm._lmm_accel`` on its own left it green while the
    guarantee it names was already broken. ``sanitizers.yml`` sets this env var
    precisely so the .so is never dlopened under ASAN, so assert the property
    where the workflow relies on it: after driving backend selection and the
    banner. A fresh interpreter is required because the gate is read at import
    time.
    """
    code = textwrap.dedent("""
        import sys
        from jamma.lmm.runner import select_execution_mode
        from jamma.pipeline_banner import log_pipeline_banner

        log_pipeline_banner(select_execution_mode(1_000, 10_000, n_cvt=3))
        print("LOADED" if "jamma.lmm._lmm_accel" in sys.modules else "CLEAN")
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "JAMMA_FORCE_NUMPY_FALLBACK": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "CLEAN", proc.stdout
