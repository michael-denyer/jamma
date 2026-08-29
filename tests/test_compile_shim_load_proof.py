"""Tests for the dev-mode compile shims' subprocess load proof.

``_compile_accel.py`` and ``_compile_jlinalg.py`` no longer prove the
freshly compiled ``.so`` imports in-process: F1 moved that proof into a
fresh subprocess (``_load_proof``) so the ``__main__`` entry point never
re-executes ``jamma.lmm``/``jamma.jlinalg``'s own import machinery, which is
what caused the #181 self-deadlock when the old in-process probe evicted the
parent package from ``sys.modules`` and re-imported it. These tests prove
``_load_proof`` fails loudly — non-zero exit, stderr populated — for a
subprocess that cannot import the extension, without touching a real
compiler or a real ``.so``.
"""

from __future__ import annotations

import pytest

from jamma.jlinalg import _compile_jlinalg
from jamma.lmm import _compile_accel

pytestmark = pytest.mark.tier0


def test_accel_load_proof_fails_loudly_on_broken_import(capsys):
    """A subprocess that cannot import the extension must return False and
    print the failure, not silently succeed."""
    result = _compile_accel._load_proof("import jamma.lmm._nonexistent_module")

    assert result is False
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "failed to import" in captured.err


def test_accel_load_proof_skipped_under_sanitize(monkeypatch, capsys):
    """JAMMA_SANITIZE set must skip the subprocess probe entirely — an
    ASan-instrumented .so cannot load without LD_PRELOAD, which this step
    does not set."""
    monkeypatch.setenv("JAMMA_SANITIZE", "address")

    result = _compile_accel._load_proof()

    assert result is True
    captured = capsys.readouterr()
    assert "skipping" in captured.err


def test_jlinalg_load_proof_fails_loudly_on_broken_import(capsys):
    """A subprocess that cannot import the extension must return False and
    print the failure, not silently succeed."""
    result = _compile_jlinalg._load_proof("import jamma.jlinalg._nonexistent_module")

    assert result is False
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "failed to import" in captured.err


def test_jlinalg_load_proof_skipped_under_sanitize(monkeypatch, capsys):
    """JAMMA_SANITIZE set must skip the subprocess probe entirely — an
    ASan-instrumented .so cannot load without LD_PRELOAD, which this step
    does not set."""
    monkeypatch.setenv("JAMMA_SANITIZE", "address")

    result = _compile_jlinalg._load_proof()

    assert result is True
    captured = capsys.readouterr()
    assert "skipping" in captured.err
