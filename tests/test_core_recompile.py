"""Tests for jamma.core.recompile.auto_recompile_c_extension.

Covers the four observable outcomes of the runtime recompile shim:
  1. Compiler module missing (installed-wheel case) -> False + debug log
  2. Compiler module raises during compile -> False + warning log + fallback msg
  3. Compiler module returns False -> False + warning log + fallback msg
  4. Compiler module returns True -> True + sys.modules key evicted

This module is the runtime ABI-mismatch fallback on end-user wheels; it is
the highest-risk surface in phase 123 because build_support/ is NOT shipped
in the wheel. Tests run against the shim's public contract only; no real
subprocess / compilation ever fires.
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

    def compile_extension(verbose: bool = False) -> bool:
        del verbose
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            raise outcome("fake failure")
        return bool(outcome)

    mod.compile_extension = compile_extension  # type: ignore[attr-defined]
    return mod


@pytest.mark.tier0
def test_compiler_module_missing_returns_false(monkeypatch):
    """Installed-wheel case: build_support/ is absent, compiler import fails."""
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
