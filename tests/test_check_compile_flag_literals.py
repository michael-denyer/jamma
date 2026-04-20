"""Tests for scripts/check-compile-flag-literals.py.

The script's job is to block bare compile-flag literals (``"-O3"``,
``"-fopenmp"``, etc.) outside jamma._build_support. These tests exercise the
script against synthetic target trees to cover:

  (a) Expected positive cases — it must flag obvious drift.
  (b) Known false-negatives — documented so nobody assumes the lint is
      defense-in-depth. If these start passing in the future, that's a
      good thing and the xfail can be flipped.

The lint is a drift-catcher for honest copy-paste, not a sandbox escape.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check-compile-flag-literals.py"


def _run_with_targets(
    tmp_path: Path, files: dict[str, str]
) -> subprocess.CompletedProcess:
    """Copy the script to a temp repo root, stub target files, run it.

    The script computes ``repo_root = parents[1]`` and reads a hard-coded
    TARGETS list. We reproduce that layout under ``tmp_path``: script goes
    to ``tmp_path/scripts/``, target files to relative paths under tmp_path.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script_copy = scripts_dir / _SCRIPT.name
    shutil.copy2(_SCRIPT, script_copy)

    for rel_path, content in files.items():
        dst = tmp_path / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)

    return subprocess.run(
        [sys.executable, str(script_copy)],
        capture_output=True,
        text=True,
        check=False,
    )


# All four targets must exist or the script reports missing-file violations,
# so each test stubs every TARGETS entry.
_STUB_EMPTY_TARGETS: dict[str, str] = {
    "hatch_build.py": "# stub\n",
    "src/jamma/jlinalg/_compile_jlinalg.py": "# stub\n",
    "src/jamma/lmm/_compile_accel.py": "# stub\n",
    "src/jamma/core/recompile.py": "# stub\n",
}


@pytest.mark.tier0
def test_clean_tree_passes(tmp_path):
    """No flag literals anywhere -> rc=0."""
    result = _run_with_targets(tmp_path, _STUB_EMPTY_TARGETS)
    assert result.returncode == 0, result.stderr


@pytest.mark.tier0
def test_double_quoted_literal_is_detected(tmp_path):
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = dedent(
        """
        def make_flags():
            return ["-O3"]
        """
    ).strip()
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1
    assert "-O3" in result.stderr


@pytest.mark.tier0
def test_single_quoted_literal_is_detected(tmp_path):
    files = dict(_STUB_EMPTY_TARGETS)
    files["src/jamma/lmm/_compile_accel.py"] = "flags = ['-fopenmp']\n"
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1
    assert "-fopenmp" in result.stderr


@pytest.mark.tier0
def test_comment_only_line_is_ignored(tmp_path):
    """Flag literal inside a comment line should NOT trip the lint —
    documentation and rationale often mention -O3 in comments."""
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = '# rationale: we deliberately avoid "-O3" here\n'
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 0, result.stderr


@pytest.mark.tier0
def test_inline_comment_with_literal_on_code_line_still_flags(tmp_path):
    """A code line with a literal followed by an inline comment IS drift."""
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = 'cflags.append("-O3")  # noqa\n'
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1


@pytest.mark.tier0
@pytest.mark.parametrize(
    "flag",
    ["-march=native", "-mtune=native", "-std=c11", "-shared", "-pthread"],
)
def test_widened_flag_set_is_detected(tmp_path, flag):
    """Portability footguns and link-phase flags beyond the original
    -O/-f set must trip the lint — particularly -march=native which must
    stay dev-only per CLAUDE.md."""
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = f'cflags.append("{flag}")\n'
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1, result.stderr
    assert flag in result.stderr


@pytest.mark.tier0
def test_allow_compile_flag_literal_escape_hatch(tmp_path):
    """A line marked with `# allow-compile-flag-literal` opts out.
    Reserved for deliberate divergence (e.g. -march=native in _compile_accel.py
    dev builds) — wheels target the lowest common denominator, dev targets
    the local CPU, so the flag cannot migrate to BASE_CFLAGS."""
    files = dict(_STUB_EMPTY_TARGETS)
    files["src/jamma/lmm/_compile_accel.py"] = (
        'extra_cflags = ["-march=native"]  # allow-compile-flag-literal: dev-only\n'
    )
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 0, result.stderr


@pytest.mark.tier0
def test_allow_compile_flag_literal_on_preceding_line(tmp_path):
    """The escape hatch also accepts the marker on the immediately
    preceding comment line — needed when ruff-format splits an inline
    comment off a long line, which it does for ``extra_cflags`` with a
    long inline rationale."""
    files = dict(_STUB_EMPTY_TARGETS)
    files["src/jamma/lmm/_compile_accel.py"] = (
        "# allow-compile-flag-literal: dev-only, see rationale above\n"
        'extra_cflags = ["-march=native"]\n'
    )
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 0, result.stderr


@pytest.mark.tier0
def test_allow_marker_only_applies_to_next_line(tmp_path):
    """The preceding-line hatch covers exactly one line — a literal two
    lines below the marker is still flagged."""
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = (
        "# allow-compile-flag-literal\n"
        "x = 1\n"
        'cflags = ["-O3"]\n'  # two lines after marker — NOT suppressed
    )
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1
    assert "-O3" in result.stderr


@pytest.mark.tier0
def test_path_like_string_is_not_a_false_positive(tmp_path):
    """A string like "/usr/lib/-O3-foo" must NOT trip the lint — the
    regex requires `-` immediately after the opening quote."""
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = 'path = "/usr/lib/-O3-test/foo"\n'
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 0, result.stderr


@pytest.mark.tier0
def test_missing_target_file_is_reported(tmp_path):
    """If a target is absent entirely, that's a cleanup-went-wrong signal
    and must surface as a violation rather than passing silently."""
    files = dict(_STUB_EMPTY_TARGETS)
    del files["src/jamma/core/recompile.py"]
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1
    assert "recompile.py" in result.stderr


# ---------------------------------------------------------------------------
# Known false-negatives — documented bypasses. These pass the lint today.
# The regex matches only a single quoted literal per pair of delimiters.
# If any of these start FAILING the lint, the regex got smarter — flip the
# xfail to a plain assertion and update the docstring.
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.xfail(
    reason="Known false-negative: explicit string concat bypasses regex",
    strict=True,
)
def test_explicit_string_concat_bypass(tmp_path):
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = 'flag = "-O" + "3"\n'
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1


@pytest.mark.tier0
@pytest.mark.xfail(
    reason="Known false-negative: f-string interpolation bypasses regex",
    strict=True,
)
def test_fstring_interpolation_bypass(tmp_path):
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = 'level = 3\nflag = f"-O{level}"\n'
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1


@pytest.mark.tier0
@pytest.mark.xfail(
    reason="Known false-negative: implicit adjacent-string concat bypasses regex",
    strict=True,
)
def test_implicit_adjacent_string_concat_bypass(tmp_path):
    files = dict(_STUB_EMPTY_TARGETS)
    files["hatch_build.py"] = 'flag = "-O" "3"\n'
    result = _run_with_targets(tmp_path, files)
    assert result.returncode == 1
