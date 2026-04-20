"""Tests for scripts/verify_compile_invocations_match.py.

The verifier enforces that all three compile entry points
(hatch_build.py, _compile_jlinalg.py, _compile_accel.py) route through
``jamma._build_support.compile_and_link.compile_jlinalg`` — the single
source of truth for compile flags and sources. A bug in the verifier
would silently bless divergence, defeating its purpose.

These tests build synthetic entry-point trees that should either pass
or fail the verifier, and assert the right outcome.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "verify_compile_invocations_match.py"


def _load_verifier():
    """Load the verifier module by file path (hyphens in filename block
    normal import). Using importlib mirrors how the verifier itself loads
    compile_and_link at runtime.
    """
    spec = importlib.util.spec_from_file_location(
        "_verify_module_under_test", str(_SCRIPT)
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_verify_module_under_test"] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


# Minimal valid compile_and_link.py for the verifier's import step.
# Only needs to expose the four constants the verifier prints.
_VALID_BUILD_SUPPORT = dedent(
    '''
    """Stub compile_and_link for verifier tests."""

    BASE_CFLAGS = ("-O3",)
    LAPACK_CFLAGS = ("-O2", "-fno-fast-math")
    BASELINE_SOURCES = ("platform.c",)
    LAPACK_SOURCES = ("eigh.c",)


    def compile_jlinalg(**kwargs):
        return True
    '''
).strip()


def _write_tree(
    tmp_path: Path,
    entry_point_contents: dict[str, str],
    build_support: str = _VALID_BUILD_SUPPORT,
) -> tuple[Path, list[Path]]:
    """Lay out a synthetic tree and return (build_support_path, entry_point_paths)."""
    bs_path = tmp_path / "_build_support.py"
    bs_path.write_text(build_support)
    entry_points: list[Path] = []
    for name, content in entry_point_contents.items():
        p = tmp_path / name
        p.write_text(content)
        entry_points.append(p)
    return bs_path, entry_points


# -------- _has_compile_jlinalg_call: AST-based detection --------


@pytest.mark.tier0
def test_ast_detects_bare_call():
    assert _VERIFIER._has_compile_jlinalg_call("compile_jlinalg(sources=[])")


@pytest.mark.tier0
def test_ast_detects_attribute_call():
    src = "from x import y\ncompile_and_link.compile_jlinalg(sources=[])"
    assert _VERIFIER._has_compile_jlinalg_call(src)


@pytest.mark.tier0
def test_ast_rejects_mention_in_comment():
    """A commented-out call must NOT satisfy the check."""
    src = "# compile_jlinalg(sources=[])\nreturn None"
    assert not _VERIFIER._has_compile_jlinalg_call(src)


@pytest.mark.tier0
def test_ast_rejects_mention_in_docstring():
    """A docstring mention must NOT satisfy the check — this is the
    exact weakness the old substring-match verifier had."""
    src = '"""Module doc: call compile_jlinalg(x) like this."""\nreturn None'
    assert not _VERIFIER._has_compile_jlinalg_call(src)


@pytest.mark.tier0
def test_ast_rejects_mention_in_string_literal():
    src = 'msg = "run compile_jlinalg(x) for help"'
    assert not _VERIFIER._has_compile_jlinalg_call(src)


@pytest.mark.tier0
def test_ast_accepts_local_definition_that_also_calls_itself():
    """Edge case: a local ``def compile_jlinalg`` that ALSO calls itself
    (or another compile_jlinalg) satisfies the check — the call is real,
    even though the definition shadows the imported helper. Worth
    documenting: the AST check is about call-site presence, not
    resolving WHICH compile_jlinalg is called. This is a deliberate
    trade-off: resolving symbol origin robustly requires a full type
    checker, and false positives here are caught by the compile-flag
    drift lint (since a local reimplementation would duplicate flags)."""
    src = dedent("""
        def compile_jlinalg(**kwargs):
            return compile_jlinalg(foo=1)  # recursion
    """)
    assert _VERIFIER._has_compile_jlinalg_call(src)


@pytest.mark.tier0
def test_ast_handles_syntax_error_gracefully():
    """Malformed source should return False, not crash. The verifier
    treating unparsable entry points as 'no call' causes a violation,
    which is the right outcome — the entry point is broken."""
    assert not _VERIFIER._has_compile_jlinalg_call("def broken(:")


# -------- check() end-to-end: synthetic trees --------


@pytest.mark.tier0
def test_valid_tree_passes(tmp_path):
    """All three entry points call compile_jlinalg properly and have no
    banned literals — verifier must return 0."""
    entry = "from x import y\ncompile_and_link.compile_jlinalg(sources=[])\n"
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": entry,
            "_compile_jlinalg.py": entry,
            "_compile_accel.py": entry,
        },
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 0, failures
    assert failures == []


@pytest.mark.tier0
def test_entry_point_with_no_call_fails(tmp_path):
    """An entry point that imports but never calls compile_jlinalg is a
    drift: someone might have deleted the call accidentally."""
    good = "compile_jlinalg(sources=[])\n"
    silent = "from jamma._build_support import compile_and_link\n# forgot to call\n"
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": good,
            "_compile_jlinalg.py": silent,
            "_compile_accel.py": good,
        },
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any("no compile_jlinalg call" in f for f in failures)


@pytest.mark.tier0
def test_commented_out_call_is_caught(tmp_path):
    """The KEY adversarial case: the old substring verifier accepted
    `# compile_jlinalg(` in a comment. The AST-based verifier must
    flag this as 'no real call'."""
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": "# TODO: re-enable compile_jlinalg(sources=[])\npass\n",
        },
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any("no compile_jlinalg call" in f for f in failures)


@pytest.mark.tier0
def test_docstring_mention_is_caught(tmp_path):
    """Same class as commented-out: a docstring mention looks like a
    call textually but isn't one."""
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": '"""See compile_jlinalg(sources) for usage."""\npass\n',
        },
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any("no compile_jlinalg call" in f for f in failures)


@pytest.mark.tier0
@pytest.mark.parametrize(
    "banned",
    ["'-O3'", '"-O3"', "'-fopenmp'", '"-fno-fast-math"'],
)
def test_banned_literal_in_entry_point_fails(tmp_path, banned):
    """Even with a valid call, a bare compile-flag literal in an entry
    point means drift from _build_support."""
    entry = f"compile_jlinalg(sources=[])\ncflags = [{banned}]\n"
    bs, eps = _write_tree(tmp_path, {"hatch_build.py": entry})
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any(banned in f for f in failures)


@pytest.mark.tier0
def test_missing_entry_point_is_flagged(tmp_path):
    """A deleted entry point is drift — maybe someone renamed a file
    without updating the verifier. Must surface as a failure, not
    silently pass."""
    bs, eps = _write_tree(tmp_path, {"hatch_build.py": "compile_jlinalg()\n"})
    # Append a non-existent entry point path.
    eps.append(tmp_path / "nonexistent.py")
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any("entry point missing" in f for f in failures)


@pytest.mark.tier0
def test_multiple_failures_all_reported(tmp_path):
    """Verifier reports every failure, not just the first."""
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": "# no call here\npass\n",
            "_compile_jlinalg.py": 'flags = ["-O3"]\ncompile_jlinalg()\n',
            "_compile_accel.py": 'flags = ["-fopenmp"]\n# no call\n',
        },
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    # hatch_build: missing call. jlinalg: banned -O3 literal. accel: missing
    # call AND banned -fopenmp literal.
    no_call_failures = [f for f in failures if "no compile_jlinalg call" in f]
    literal_failures = [f for f in failures if "banned literal" in f]
    assert len(no_call_failures) >= 2
    assert len(literal_failures) >= 2
