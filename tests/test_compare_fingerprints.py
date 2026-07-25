"""Behaviour tests for ``scripts/compare_fingerprints.py``.

This script decides whether a C change moved any bits, so it is the gate the
whole ``_lmm_accel.c`` refactoring programme leans on. A comparison tool that
silently passes is worse than no gate, because it looks like proof.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"

pytestmark = pytest.mark.tier0


@pytest.fixture(scope="module")
def compare_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import compare_fingerprints

        return compare_fingerprints
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)


def _write(path: Path, records: list[tuple[str, str, str]]) -> Path:
    path.write_text("".join(f"{a}\t{b}\t{c}\n" for a, b, c in records))
    return path


BASE = [
    ("compute_lmm_batch_c", "aaaa", "1111"),
    ("compute_lrt_batch_c", "bbbb", "2222"),
    ("compute_score_batch_c", "cccc", "raise:ValueError"),
]


#: The args digest cannot see inside a PyCapsule, so a workspace-taking entry
#: point records the same key twice: a value with a live workspace, a raise with
#: a spent one. Four such collisions exist in the real 139-record fingerprint.
COLLIDING = [
    ("compute_lrt_fused_ws_c", "eeee", "6666"),
    ("compute_lrt_fused_ws_c", "eeee", "raise:ValueError"),
]


def test_identical_runs_pass(compare_module, capsys, tmp_path):
    a = _write(tmp_path / "base.txt", BASE)
    b = _write(tmp_path / "head.txt", BASE)

    assert compare_module.main([str(a), str(b)]) == 0
    assert "bit-identical" in capsys.readouterr().out


def test_a_repeated_key_keeps_both_of_its_results(compare_module, capsys, tmp_path):
    """Two records sharing a key must not collapse into one."""
    a = _write(tmp_path / "base.txt", [*BASE, *COLLIDING])
    b = _write(tmp_path / "head.txt", [*BASE, *COLLIDING])

    assert compare_module.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "base: 5 records / 4 keys" in out, "a collapsed key would report 4 records"


def test_drift_on_a_repeated_key_is_caught(compare_module, capsys, tmp_path):
    """Keying by (name, args) alone would drop this record and pass."""
    a = _write(tmp_path / "base.txt", [*BASE, *COLLIDING])
    b = _write(
        tmp_path / "head.txt",
        [
            *BASE,
            ("compute_lrt_fused_ws_c", "eeee", "7777"),
            ("compute_lrt_fused_ws_c", "eeee", "raise:ValueError"),
        ],
    )

    assert compare_module.main([str(a), str(b)]) == 1
    assert "compute_lrt_fused_ws_c" in capsys.readouterr().err


def test_a_changed_result_digest_fails(compare_module, capsys, tmp_path):
    a = _write(tmp_path / "base.txt", BASE)
    b = _write(
        tmp_path / "head.txt",
        [*BASE[:1], ("compute_lrt_batch_c", "bbbb", "9999"), *BASE[2:]],
    )

    assert compare_module.main([str(a), str(b)]) == 1
    err = capsys.readouterr().err
    assert "1 of 3 shared keys CHANGED" in err
    assert "compute_lrt_batch_c" in err


def test_a_raise_becoming_a_value_fails(compare_module, capsys, tmp_path):
    """An entry point that stopped raising is drift, not a coverage change."""
    a = _write(tmp_path / "base.txt", BASE)
    b = _write(
        tmp_path / "head.txt", [*BASE[:2], ("compute_score_batch_c", "cccc", "3333")]
    )

    assert compare_module.main([str(a), str(b)]) == 1
    assert "compute_score_batch_c" in capsys.readouterr().err


def test_added_records_are_reported_but_do_not_fail(compare_module, capsys, tmp_path):
    """A new entry point or test case has nothing on the base side to compare."""
    a = _write(tmp_path / "base.txt", BASE)
    added = ("compute_new_thing_c", "dddd", "4444")
    b = _write(tmp_path / "head.txt", [*BASE, added])

    assert compare_module.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "1 record(s) added by this change" in out
    assert "All 3 shared keys are bit-identical." in out


def test_removed_records_are_reported_but_do_not_fail(compare_module, capsys, tmp_path):
    a = _write(tmp_path / "base.txt", BASE)
    b = _write(tmp_path / "head.txt", BASE[:2])

    assert compare_module.main([str(a), str(b)]) == 0
    assert "1 record(s) removed" in capsys.readouterr().out


def test_no_shared_records_fails_rather_than_passing_vacuously(
    compare_module, capsys, tmp_path
):
    """Nothing in common proves nothing, and must not read as success."""
    a = _write(tmp_path / "base.txt", BASE)
    b = _write(tmp_path / "head.txt", [("other_c", "zzzz", "5555")])

    assert compare_module.main([str(a), str(b)]) == 1
    assert "No shared records" in capsys.readouterr().err


def test_an_empty_head_file_fails(compare_module, capsys, tmp_path):
    """A run that produced nothing must not be mistaken for a clean run."""
    a = _write(tmp_path / "base.txt", BASE)
    b = _write(tmp_path / "head.txt", [])

    assert compare_module.main([str(a), str(b)]) == 1
    assert "No shared records" in capsys.readouterr().err


def test_a_malformed_record_is_an_error_not_a_silent_skip(compare_module, tmp_path):
    a = _write(tmp_path / "base.txt", BASE)
    b = tmp_path / "head.txt"
    b.write_text("compute_lmm_batch_c\tonly-two-fields\n")

    with pytest.raises(ValueError, match="expected 3 tab-separated fields"):
        compare_module.main([str(a), str(b)])
