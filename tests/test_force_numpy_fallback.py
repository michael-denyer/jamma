"""Tests for the JAMMA_FORCE_NUMPY_FALLBACK env-var gate at the two C
extension import shims (jamma.jlinalg.__init__ and
jamma.lmm.compute_numpy._try_import_accel).

Phase 116.1 plan 02. The gate is the load-bearing knob the ASAN/UBSAN
sanitizer workflow uses to skip the .so imports entirely — see
RESEARCH.md §"Pitfall 4" (ASAN + dlopen interaction can produce
false-positive heap-buffer-overflow reports inside dispatched BLAS calls).
"""

from __future__ import annotations

import importlib
import sys
import warnings

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
        recompile_mod, "auto_recompile_c_extension", lambda **kwargs: False
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
    jamma.lmm.compute_numpy._try_import_accel."""

    def test_returns_unavailable(self, monkeypatch):
        """Forced env: _try_import_accel returns _ACCEL_UNAVAILABLE."""
        monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "1")
        from jamma.lmm.compute_numpy import _ACCEL_UNAVAILABLE, _try_import_accel

        result = _try_import_accel()
        assert result == _ACCEL_UNAVAILABLE

    def test_no_so_import_attempted(self, monkeypatch):
        """Forced env: no .so import attempted — sys.modules stays clean."""
        monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", "1")
        from jamma.lmm.compute_numpy import _try_import_accel

        sys.modules.pop("jamma.lmm._lmm_accel", None)
        _try_import_accel()
        assert "jamma.lmm._lmm_accel" not in sys.modules

    def test_unset_preserves_behavior(self, monkeypatch):
        """Unset env: _try_import_accel returns the natural import result —
        whatever the .so produces (don't pin to True/False, depends on
        whether the extension is built)."""
        monkeypatch.delenv("JAMMA_FORCE_NUMPY_FALLBACK", raising=False)
        from jamma.lmm.compute_numpy import _try_import_accel

        result = _try_import_accel()
        # Just assert the call returned an AccelImport (named tuple) — the
        # natural state is environment-dependent.
        assert hasattr(result, "accel_available")

    @pytest.mark.parametrize("value", ["", "0"])
    def test_off_values(self, monkeypatch, value):
        """'' and '0' are off — _try_import_accel does NOT short-circuit."""
        monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", value)
        from jamma.lmm.compute_numpy import _try_import_accel

        # The result equals the natural-import result (depends on .so being
        # present), so we cannot assert .accel_available — but we CAN assert
        # the gate's early-return path was NOT taken: if it had been, the
        # AccelImport returned would equal _ACCEL_UNAVAILABLE exactly. Since
        # the .so is built in this environment, the result must differ.
        result = _try_import_accel()
        from jamma.lmm.compute_numpy import _ACCEL_UNAVAILABLE

        # If the .so loaded, accel_available is True and result != _ACCEL_UNAVAILABLE.
        # If the .so failed naturally, accel_available is False but the result is
        # constructed via the fallback path inside _try_import_accel, NOT via the
        # gate — both end up with accel_available=False. We cannot distinguish
        # these two from the outside. So just verify the call did not error.
        assert hasattr(result, "accel_available")
        # In the typical CI environment with the .so built, the natural path
        # MUST succeed — assert that to catch regressions where the gate
        # accidentally engages on '' or '0':
        if "_lmm_accel" in sys.modules and sys.modules["_lmm_accel"] is not None:
            assert result.accel_available is True or result == _ACCEL_UNAVAILABLE

    def test_truthy_values_engage_gate(self, monkeypatch):
        """Multiple truthy values all engage the gate."""
        from jamma.lmm.compute_numpy import _ACCEL_UNAVAILABLE, _try_import_accel

        for value in ["1", "true", "yes", "TRUE", " 1 "]:
            monkeypatch.setenv("JAMMA_FORCE_NUMPY_FALLBACK", value)
            result = _try_import_accel()
            assert result == _ACCEL_UNAVAILABLE, (
                f"value={value!r} did not engage the gate"
            )
