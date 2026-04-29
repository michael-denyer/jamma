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
    BOTH flags reach compile_jlinalg's extra_cflags. The two env vars are
    truly orthogonal — setting one does not suppress the other, and the
    sentinel-define gate (plan 04) is independent of the sanitizer-override
    helper (plan 01 + plan 05).

    Post-plan-05 wiring: apply_sanitizer_overrides() runs inside
    compile_extension() and appends -fsanitize=..., -fno-omit-frame-pointer,
    -O1 to extra_cflags. The sentinel append happens BEFORE that helper, so
    -DJAMMA_SENTINEL_UB lands earlier in the list than the sanitizer flags.
    """
    monkeypatch.setenv("JAMMA_SENTINEL_UB", "1")
    monkeypatch.setenv("JAMMA_SANITIZE", "address,undefined")
    _run_compile_extension(monkeypatch, tmp_path)
    extra_cflags = captured_compile_call["kwargs"]["extra_cflags"]
    # Sentinel macro present (plan 04 wiring).
    assert "-DJAMMA_SENTINEL_UB" in extra_cflags, extra_cflags
    # Sanitizer flags also present (plan 05 wiring of apply_sanitizer_overrides).
    assert "-fsanitize=address,undefined" in extra_cflags, extra_cflags
    assert "-fno-omit-frame-pointer" in extra_cflags, extra_cflags
    # Sentinel must precede sanitizer flags (plan 04 appends BEFORE
    # apply_sanitizer_overrides runs).
    sentinel_idx = extra_cflags.index("-DJAMMA_SENTINEL_UB")
    sanitize_idx = extra_cflags.index("-fsanitize=address,undefined")
    assert sentinel_idx < sanitize_idx, extra_cflags
    # extra_lapack_cflags should ALSO carry the sanitizer flags (plan 05's
    # full wiring instruments LAPACK sources too — no LAPACK source in
    # _lmm_accel, so this is a defensive check that the wiring is uniform).
    extra_lapack = captured_compile_call["kwargs"]["extra_lapack_cflags"]
    assert "-fsanitize=address,undefined" in extra_lapack, extra_lapack


def test_sentinel_truthy_values_engage(monkeypatch, captured_compile_call, tmp_path):
    """Various truthy values all engage the gate."""
    for value in ["1", "true", "yes", " 1 "]:
        monkeypatch.setenv("JAMMA_SENTINEL_UB", value)
        _run_compile_extension(monkeypatch, tmp_path)
        extra_cflags = captured_compile_call["kwargs"]["extra_cflags"]
        assert "-DJAMMA_SENTINEL_UB" in extra_cflags, (value, extra_cflags)
