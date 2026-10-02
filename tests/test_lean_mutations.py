"""The mutation CLI must distinguish checker failure from property rejection."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_LEAN = Path(__file__).resolve().parents[1] / "lean"


def _run(tmp_path: Path, checker_source: str) -> subprocess.CompletedProcess[str]:
    project = tmp_path / "lean"
    shutil.copytree(_LEAN, project, ignore=shutil.ignore_patterns(".lake"))
    checker = tmp_path / "checker.sh"
    checker.write_text(checker_source)
    return subprocess.run(
        [sys.executable, str(project / "check_mutations.py"), str(checker)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_unavailable_checker_is_not_reported_as_detecting_mutations(tmp_path):
    result = _run(
        tmp_path,
        "printf '%s\\n' 'error: toolchain unavailable; no model was checked' >&2\n"
        "exit 1\n",
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "DETECTED" not in result.stdout
    assert "toolchain unavailable" in result.stdout + result.stderr


def _run_checker(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    """Exercise the CLI with a file-based fake of the external Lean checker."""
    originals = {
        project: (_LEAN / project / "Model.lean").read_text()
        for project in ("MatrixBlocks", "ChunkSlots")
    }
    checker = tmp_path / "checker.py"
    checker.write_text(
        f"originals = {originals!r}\nmode = {mode!r}\n"
        """
import sys
from pathlib import Path

target = Path(sys.argv[1])
source = (target / 'Model.lean').read_text()
if source == originals[target.name]:
    if mode == 'second_baseline_error' and target.name == 'ChunkSlots':
        print('error: unknown module source path')
        sys.exit(1)
    print(f'PASS {target}: no sorry, no extra axioms')
    sys.exit(0)

if mode == 'always_success':
    sys.exit(0)
if mode == 'unrelated_error':
    print('error: toolchain unavailable; no model was checked')
    sys.exit(1)

# Emit actual Lean guard-rejection diagnostics for the bounded properties.
# Alternative modes exercise errors which do not establish sensitivity.
for line, text in enumerate(source.splitlines(), 1):
    if not text.startswith('#guard bad'):
        continue
    expression = text.removeprefix('#guard ')
    filename = 'Other.lean' if mode == 'wrong_file' else 'Model.lean'
    if mode == 'wrong_line':
        line += 1
    if mode == 'syntax_error':
        print(f"error: {filename}:{line}:0: unknown identifier 'isEmpty'")
    else:
        if mode == 'wrong_expression':
            expression = 'false'
        print(f'error: {filename}:{line}:0: Expression')
        print(f'  {expression}')
        print('did not evaluate to `true`')
sys.exit(0 if mode == 'zero_exit_rejection' else 1)
"""
    )
    return _run(
        tmp_path,
        f'exec {shlex.quote(sys.executable)} {shlex.quote(str(checker))} "$1"\n',
    )


@pytest.mark.parametrize(
    "mode",
    [
        "unrelated_error",
        "syntax_error",
        "wrong_file",
        "wrong_line",
        "wrong_expression",
        "zero_exit_rejection",
        "always_success",
    ],
)
def test_mutation_requires_its_evaluated_false_guard(tmp_path, mode):
    result = _run_checker(tmp_path, mode)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "BASELINE MatrixBlocks" in result.stdout
    assert "BASELINE ChunkSlots" in result.stdout
    assert "FAIL mutation MatrixBlocks/zero block rows" in result.stdout
    assert "DETECTED" not in result.stdout


def test_every_baseline_passes_before_any_mutation_is_accepted(tmp_path):
    result = _run_checker(tmp_path, "second_baseline_error")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL baseline ChunkSlots" in result.stdout
    assert "unknown module source path" in result.stdout
    assert "DETECTED" not in result.stdout


def test_expected_guard_rejections_detect_all_seven_mutations(tmp_path):
    result = _run_checker(tmp_path, "guard_rejection")

    assert result.returncode == 0, result.stdout + result.stderr
    detections = [
        line for line in result.stdout.splitlines() if line.startswith("DETECTED ")
    ]
    assert detections == [
        "DETECTED MatrixBlocks/zero block rows",
        "DETECTED MatrixBlocks/uncapped workers",
        "DETECTED MatrixBlocks/floor instead of ceiling",
        "DETECTED ChunkSlots/slot overflow",
        "DETECTED ChunkSlots/reuse consecutive slot",
        "DETECTED MatrixBlocks/missing advance",
        "DETECTED MatrixBlocks/undersized capacity",
    ]
