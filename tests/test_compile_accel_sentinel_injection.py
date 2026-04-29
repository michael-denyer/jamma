"""Tests for the JAMMA_SENTINEL_UB env-var injection in
jamma.lmm._compile_accel.compile_extension.

Phase 116.1, plan 04. The gate appends ``-DJAMMA_SENTINEL_UB`` to
``extra_cflags`` so the sanitizer workflow's sentinel-meta-test job can
rebuild ``_lmm_accel.so`` with the gated heap-OOB function exposed.

These tests monkeypatch ``compile_jlinalg`` so the real compiler never
runs — they only verify the wiring (kwargs reaching the helper).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from jamma._build_support.compile_and_link import CompileResult

pytestmark = pytest.mark.tier0


@pytest.fixture
def captured_compile_call(monkeypatch):
    """Replace compile_jlinalg with a recorder. Returns the captured kwargs.

    compile_extension invokes a post-link import probe that fails when no
    real .so was built; that's expected — these tests assert on the captured
    kwargs regardless of compile_extension's eventual return value.
    """
    captured: dict = {}

    def fake_compile_jlinalg(**kwargs):
        captured["kwargs"] = dict(kwargs)
        return CompileResult(
            success=True,
            used_openmp=False,
            used_openmp_link=False,
            output_path=kwargs["output"],
            obj_files=[],
        )

    # Patch the symbol AS IMPORTED into _compile_accel.py.
    monkeypatch.setattr(
        "jamma.lmm._compile_accel.compile_jlinalg",
        fake_compile_jlinalg,
    )
    return captured


def _run_compile_extension(monkeypatch, tmp_path: Path) -> None:
    """Invoke compile_extension. The post-link import probe will fail
    because we never wrote a real .so; the kwargs were already captured.
    """
    from jamma.lmm import _compile_accel

    # Don't mutate the real package directory — redirect ``out`` to tmp_path
    # by monkeypatching the source-file resolution. compile_extension uses
    # __file__ to locate _lmm_accel.c; we leave that intact (the file exists
    # in the repo) but the output .so path is computed from src.parent which
    # we cannot easily redirect. Instead, accept that compile_extension will
    # try and fail to import the (fake) .so post-link.
    # The fake compile_jlinalg records the kwargs successfully; any post-
    # success failure (import probe, etc.) is irrelevant to the wiring
    # assertion the test makes from `captured_compile_call`.
    with contextlib.suppress(ImportError, OSError, RuntimeError):
        _compile_accel.compile_extension(verbose=False)


# ---------------------------------------------------------------------------
# Wiring tests
# ---------------------------------------------------------------------------


def test_no_sentinel_no_injection(monkeypatch, captured_compile_call, tmp_path):
    """Unset env: extra_cflags must NOT contain -DJAMMA_SENTINEL_UB."""
    monkeypatch.delenv("JAMMA_SENTINEL_UB", raising=False)
    _run_compile_extension(monkeypatch, tmp_path)
    extra_cflags = captured_compile_call["kwargs"]["extra_cflags"]
    assert "-DJAMMA_SENTINEL_UB" not in extra_cflags, extra_cflags


def test_sentinel_set_to_1_injects_macro(monkeypatch, captured_compile_call, tmp_path):
    """JAMMA_SENTINEL_UB=1: -DJAMMA_SENTINEL_UB present, AFTER -march=native."""
    monkeypatch.setenv("JAMMA_SENTINEL_UB", "1")
    _run_compile_extension(monkeypatch, tmp_path)
    extra_cflags = captured_compile_call["kwargs"]["extra_cflags"]
    assert "-DJAMMA_SENTINEL_UB" in extra_cflags, extra_cflags
    # -DJAMMA_SENTINEL_UB appears after -march=native (consistent placement
    # at the end aids debuggability — defines have no ordering issues, but
    # uniform append-at-end is greppable in compile-command logs).
    march_idx = extra_cflags.index("-march=native")
    sentinel_idx = extra_cflags.index("-DJAMMA_SENTINEL_UB")
    assert sentinel_idx > march_idx, extra_cflags


@pytest.mark.parametrize("value", ["0", "", "  "])
def test_sentinel_off_values(monkeypatch, captured_compile_call, tmp_path, value):
    """'', '0', '  ' (after .strip()): no injection."""
    monkeypatch.setenv("JAMMA_SENTINEL_UB", value)
    _run_compile_extension(monkeypatch, tmp_path)
    extra_cflags = captured_compile_call["kwargs"]["extra_cflags"]
    assert "-DJAMMA_SENTINEL_UB" not in extra_cflags, (value, extra_cflags)


def test_sentinel_orthogonal_to_sanitize(monkeypatch, captured_compile_call, tmp_path):
    """JAMMA_SENTINEL_UB=1 AND JAMMA_SANITIZE=address,undefined:
    -DJAMMA_SENTINEL_UB present in extra_cflags; sanitizer flags are NOT
    here (they reach compile_jlinalg via apply_sanitizer_overrides at the
    helper level — see plan 05 — NOT through _compile_accel.py).
    """
    monkeypatch.setenv("JAMMA_SENTINEL_UB", "1")
    monkeypatch.setenv("JAMMA_SANITIZE", "address,undefined")
    _run_compile_extension(monkeypatch, tmp_path)
    extra_cflags = captured_compile_call["kwargs"]["extra_cflags"]
    assert "-DJAMMA_SENTINEL_UB" in extra_cflags, extra_cflags
    # Sanitizer flags must NOT have been spliced here — that wiring lands
    # in plan 05 via apply_sanitizer_overrides at the entry-point level.
    # (After plan 05, _compile_accel.py WILL forward sanitizer flags too,
    # but that goes via extra_lapack_cflags / extra_link_flags — at the
    # extra_cflags level, no -fsanitize=... should appear here yet.)
    assert "-fsanitize=address,undefined" not in extra_cflags, extra_cflags
    assert "-fno-omit-frame-pointer" not in extra_cflags, extra_cflags


def test_sentinel_truthy_values_engage(monkeypatch, captured_compile_call, tmp_path):
    """Various truthy values all engage the gate."""
    for value in ["1", "true", "yes", " 1 "]:
        monkeypatch.setenv("JAMMA_SENTINEL_UB", value)
        _run_compile_extension(monkeypatch, tmp_path)
        extra_cflags = captured_compile_call["kwargs"]["extra_cflags"]
        assert "-DJAMMA_SENTINEL_UB" in extra_cflags, (value, extra_cflags)
