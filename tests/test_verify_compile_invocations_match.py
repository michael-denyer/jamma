"""Tests for scripts/verify_compile_invocations_match.py.

The verifier enforces that all three compile entry points
(hatch_build.py, _compile_jlinalg.py, _compile_accel.py) route through
``jamma._build_support.compile_and_link.run_build`` — the shared build driver.
A bug in the verifier would silently bless divergence, defeating its purpose.

The bare-flag-literal scan was retired (it duplicated
``check_compile_flag_literals.py``); these tests prove the remaining AST check
resolves calls back to the shared facade rather than trusting a callee's name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.tier0

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
# Only needs to expose ``run_build`` for the presence check.
_VALID_BUILD_SUPPORT = dedent(
    '''
    """Stub compile_and_link for verifier tests."""


    def run_build(**kwargs):
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


# -------- _has_run_build_call: AST-based detection --------


def test_ast_detects_bare_call():
    source = dedent("""
        from jamma._build_support.compile_and_link import run_build

        run_build(spec, pkg)
    """)
    assert _VERIFIER._has_run_build_call(source)


def test_ast_detects_attribute_call():
    src = dedent("""
        import jamma._build_support.compile_and_link as compile_and_link

        compile_and_link.run_build(spec, pkg)
    """)
    assert _VERIFIER._has_run_build_call(src)


def test_ast_detects_imported_compile_extension_alias():
    src = dedent("""
        from jamma._build_support.compile_and_link import compile_extension as _compile

        def compile_extension():
            return _compile(spec, pkg)
    """)
    assert _VERIFIER._has_run_build_call(src)


def test_ast_detects_isolated_build_facade_binding():
    src = dedent("""
        _cal = _load_build_support_module(
            "jamma_build_support.compile_and_link", "compile_and_link.py"
        )
        run_build = _cal.run_build
        run_build(spec, pkg)
    """)
    assert _VERIFIER._has_run_build_call(src)


def test_ast_rejects_mention_in_comment():
    """A commented-out call must NOT satisfy the check."""
    src = "# run_build(spec, pkg)\nreturn None"
    assert not _VERIFIER._has_run_build_call(src)


def test_ast_rejects_mention_in_docstring():
    """A docstring mention must NOT satisfy the check — this is the
    exact weakness the old substring-match verifier had."""
    src = '"""Module doc: call run_build(x) like this."""\nreturn None'
    assert not _VERIFIER._has_run_build_call(src)


def test_ast_rejects_mention_in_string_literal():
    src = 'msg = "run run_build(x) for help"'
    assert not _VERIFIER._has_run_build_call(src)


def test_ast_rejects_local_definition_that_calls_itself():
    """A same-named local function does not prove use of the shared facade."""
    src = dedent("""
        def run_build(**kwargs):
            return run_build(foo=1)  # recursion
    """)
    assert not _VERIFIER._has_run_build_call(src)


def test_ast_handles_syntax_error_gracefully():
    """Malformed source should return False, not crash. The verifier
    treating unparsable entry points as 'no call' causes a violation,
    which is the right outcome — the entry point is broken."""
    assert not _VERIFIER._has_run_build_call("def broken(:")


# -------- check() end-to-end: synthetic trees --------


def test_valid_tree_passes(tmp_path):
    """All three entry points call run_build properly — verifier returns 0."""
    entry = dedent("""
        from jamma._build_support.compile_and_link import run_build

        run_build(spec, pkg)
    """)
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


def test_entry_point_with_no_call_fails(tmp_path):
    """An entry point that imports but never calls run_build is a drift:
    someone might have deleted the call accidentally."""
    good = dedent("""
        from jamma._build_support.compile_and_link import run_build
        run_build(spec, pkg)
    """)
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
    assert any("no run_build call" in f for f in failures)


def test_commented_out_call_is_caught(tmp_path):
    """The KEY adversarial case: the old substring verifier accepted
    `# run_build(` in a comment. The AST-based verifier must flag this
    as 'no real call'."""
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": "# TODO: re-enable run_build(spec, pkg)\npass\n",
        },
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any("no run_build call" in f for f in failures)


def test_docstring_mention_is_caught(tmp_path):
    """Same class as commented-out: a docstring mention looks like a
    call textually but isn't one."""
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": '"""See run_build(spec) for usage."""\npass\n',
        },
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any("no run_build call" in f for f in failures)


def test_missing_run_build_in_helper_is_flagged(tmp_path):
    """If compile_and_link stops exporting run_build, the shared driver is
    gone and the guarantee is void — surface it rather than passing."""
    broken_support = '"""No driver here."""\n'
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": (
                "from jamma._build_support.compile_and_link import run_build\n"
                "run_build(spec, pkg)\n"
            )
        },
        build_support=broken_support,
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any("run_build not found" in f for f in failures)


def test_missing_entry_point_is_flagged(tmp_path):
    """A deleted entry point is drift — maybe someone renamed a file
    without updating the verifier. Must surface as a failure, not
    silently pass."""
    bs, eps = _write_tree(tmp_path, {"hatch_build.py": "run_build()\n"})
    # Append a non-existent entry point path.
    eps.append(tmp_path / "nonexistent.py")
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    assert any("entry point missing" in f for f in failures)


def test_multiple_failures_all_reported(tmp_path):
    """Verifier reports every failure, not just the first."""
    bs, eps = _write_tree(
        tmp_path,
        {
            "hatch_build.py": "# no call here\npass\n",
            "_compile_jlinalg.py": "# also no call\npass\n",
            "_compile_accel.py": (
                "from jamma._build_support.compile_and_link import run_build\n"
                "run_build()\n"
            ),
        },
    )
    rc, failures = _VERIFIER.check(bs, eps)
    assert rc == 1
    no_call_failures = [f for f in failures if "no run_build call" in f]
    assert len(no_call_failures) >= 2
