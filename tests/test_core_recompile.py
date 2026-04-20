"""Tests for jamma.core.recompile.auto_recompile_c_extension.

Covers the four observable outcomes of the runtime recompile shim:
  1. Compiler module import fails -> False + debug log
  2. Compiler module raises during compile -> False + warning log + fallback msg
  3. Compiler module returns False -> False + warning log + fallback msg
  4. Compiler module returns True -> True + sys.modules key evicted

The shim is a thin import-retry wrapper around the compile_extension()
entry points in jamma.lmm._compile_accel and jamma.jlinalg._compile_jlinalg.
Tests here run against the shim's public contract only; no real
subprocess / compilation ever fires. End-to-end wheel-install ->
auto_recompile_c_extension -> real compile coverage is tracked as a
follow-up (tier2, scratch venv) and is not yet implemented.
"""

from __future__ import annotations

import sys
import types

import pytest

from jamma.core.recompile import auto_recompile_c_extension


def _make_fake_compiler(module_name: str, *, outcome):
    """Build a fake compiler module exposing ``compile_extension(verbose=...)``.

    ``outcome`` controls behavior:
      - True  -> returns True
      - False -> returns False
      - Exception subclass -> raised when compile_extension is called
    """
    mod = types.ModuleType(module_name)

    def compile_extension(verbose: bool = False, on_retry=None) -> bool:
        del verbose, on_retry
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            raise outcome("fake failure")
        return bool(outcome)

    mod.compile_extension = compile_extension  # type: ignore[attr-defined]
    return mod


@pytest.mark.tier0
def test_compiler_module_missing_returns_false(monkeypatch):
    """Corrupted install case: the compiler module itself is missing."""
    module_name = "jamma._compiler_that_does_not_exist"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    result = auto_recompile_c_extension(
        module_name="_fake_ext",
        compiler_module=module_name,
        sys_module_key="jamma._fake_ext",
        label="fake",
    )

    assert result is False


@pytest.mark.tier0
def test_compiler_raises_returns_false_and_does_not_evict(monkeypatch):
    """Exception during compile -> False; stale module entry must remain.

    If the shim popped sys.modules on failure, a subsequent import would
    try to rebuild from scratch and likely re-hit the same crash loop.
    """
    compiler_name = "jamma._fake_compiler_raises"
    sys_key = "jamma._fake_ext_raises"

    monkeypatch.setitem(
        sys.modules,
        compiler_name,
        _make_fake_compiler(compiler_name, outcome=RuntimeError),
    )
    sentinel = types.ModuleType(sys_key)
    monkeypatch.setitem(sys.modules, sys_key, sentinel)

    result = auto_recompile_c_extension(
        module_name="_fake_ext_raises",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is False
    assert sys.modules.get(sys_key) is sentinel, (
        "stale module must NOT be evicted on failure — eviction would "
        "force re-import into the same broken path"
    )


@pytest.mark.tier0
def test_compiler_returns_false_does_not_evict(monkeypatch):
    """compile_extension returned False -> shim returns False, no eviction."""
    compiler_name = "jamma._fake_compiler_false"
    sys_key = "jamma._fake_ext_false"

    monkeypatch.setitem(
        sys.modules,
        compiler_name,
        _make_fake_compiler(compiler_name, outcome=False),
    )
    sentinel = types.ModuleType(sys_key)
    monkeypatch.setitem(sys.modules, sys_key, sentinel)

    result = auto_recompile_c_extension(
        module_name="_fake_ext_false",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is False
    assert sys.modules.get(sys_key) is sentinel


@pytest.mark.tier0
def test_successful_recompile_evicts_stale_module(monkeypatch):
    """Success path -> returns True AND pops the stale sys.modules entry so
    subsequent import picks up the freshly compiled .so."""
    compiler_name = "jamma._fake_compiler_success"
    sys_key = "jamma._fake_ext_success"

    monkeypatch.setitem(
        sys.modules,
        compiler_name,
        _make_fake_compiler(compiler_name, outcome=True),
    )
    stale = types.ModuleType(sys_key)
    monkeypatch.setitem(sys.modules, sys_key, stale)

    result = auto_recompile_c_extension(
        module_name="_fake_ext_success",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is True
    assert sys_key not in sys.modules, (
        "success must evict the stale module so re-import picks up the "
        "freshly compiled .so — a regression here would silently serve "
        "stale bytecode forever"
    )


@pytest.mark.tier0
def test_successful_recompile_with_no_prior_sys_modules_entry(monkeypatch):
    """pop(key, None) must not raise when the key was never present."""
    compiler_name = "jamma._fake_compiler_no_prior"
    sys_key = "jamma._fake_ext_no_prior"

    monkeypatch.setitem(
        sys.modules,
        compiler_name,
        _make_fake_compiler(compiler_name, outcome=True),
    )
    monkeypatch.delitem(sys.modules, sys_key, raising=False)

    result = auto_recompile_c_extension(
        module_name="_fake_ext_no_prior",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is True


@pytest.mark.tier0
def test_on_retry_callback_is_wired_and_emits_warning(monkeypatch, capsys):
    """The runtime recompile shim must pass a non-None on_retry callback
    to compile_extension AND invoking it must emit a warning the user
    can see. Without this, runtime recompile silently falls back to
    single-threaded with no user-visible signal — the exact gap this
    test guards.
    """
    from loguru import logger as _logger

    compiler_name = "jamma._fake_compiler_retry"
    sys_key = "jamma._fake_ext_retry"

    mod = types.ModuleType(compiler_name)
    captured_retry: list[object] = []

    def compile_extension(verbose: bool = False, on_retry=None) -> bool:
        del verbose
        captured_retry.append(on_retry)
        if on_retry is not None:
            on_retry("OpenMP compilation failed, retrying without OpenMP")
        return True

    mod.compile_extension = compile_extension  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, compiler_name, mod)
    monkeypatch.delitem(sys.modules, sys_key, raising=False)

    # Route loguru to stderr so capsys can observe it.
    sink_id = _logger.add(sys.stderr, level="WARNING")
    try:
        result = auto_recompile_c_extension(
            module_name="_fake_ext_retry",
            compiler_module=compiler_name,
            sys_module_key=sys_key,
            label="fake",
        )
    finally:
        _logger.remove(sink_id)

    assert result is True
    assert captured_retry, "compile_extension must be called"
    assert captured_retry[0] is not None, (
        "auto_recompile_c_extension must pass a non-None on_retry to "
        "compile_extension so OMP downgrade signals surface"
    )
    captured = capsys.readouterr()
    assert "OpenMP compilation failed" in captured.err, (
        "on_retry invocation must produce a user-visible warning — "
        "loguru must emit, not silently discard, retry notices"
    )


@pytest.mark.tier0
def test_partial_upgrade_fallback_when_compile_extension_lacks_on_retry(monkeypatch):
    """An older installed compile_extension (no on_retry kwarg) must not
    break the recompile shim — it falls back to the legacy call.
    """
    compiler_name = "jamma._fake_compiler_legacy"
    sys_key = "jamma._fake_ext_legacy"

    mod = types.ModuleType(compiler_name)

    def compile_extension(verbose: bool = False) -> bool:
        return True

    mod.compile_extension = compile_extension  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, compiler_name, mod)
    monkeypatch.delitem(sys.modules, sys_key, raising=False)

    result = auto_recompile_c_extension(
        module_name="_fake_ext_legacy",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is True
