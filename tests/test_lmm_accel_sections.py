"""Behaviour tests for ``scripts/lmm_accel_sections.py``.

The script's cross-section coupling count is the worklist that decides which
statics a ``_lmm_accel.c`` extraction must carry. A miscount sends real
refactoring work at functions that are not actually shared, so the parser gets
the same scrutiny as the code it measures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"

pytestmark = pytest.mark.tier0


@pytest.fixture(scope="module")
def sections_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import lmm_accel_sections

        return lmm_accel_sections
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)


def _analyse(module, capsys, source: Path) -> dict:
    assert module.main(["--json", str(source)]) == 0
    return json.loads(capsys.readouterr().out)


def _static(report: dict, name: str) -> dict:
    return next(s for s in report["statics"] if s["name"] == name)


BANNER = "/* " + "=" * 70

TWO_SECTIONS = f"""\
{BANNER}
 * SECTION ONE
{BANNER} */
static double helper(double x)
{{
    return x * 2.0;
}}

{BANNER}
 * SECTION TWO
{BANNER} */
static double caller(double x)
{{
    return helper(x) + 1.0;
}}
"""


def test_call_from_another_section_counts_as_crossing(
    sections_module, capsys, tmp_path
):
    source = tmp_path / "two_sections.c"
    source.write_text(TWO_SECTIONS)

    report = _analyse(sections_module, capsys, source)

    helper = _static(report, "helper")
    assert helper["crosses"] is True
    assert helper["ref_count"] == 1


def test_mention_inside_a_block_comment_is_not_a_reference(
    sections_module, capsys, tmp_path
):
    """A name written in prose inside ``/* ... */`` is documentation, not a call.

    Interior lines of a block comment carry no ``/*`` of their own, so a parser
    that strips comments line-by-line reads them as code and reports coupling
    that does not exist.
    """
    source = tmp_path / "commented.c"
    source.write_text(
        f"""\
/*
 * Overview of this translation unit.
 *
 * The entry points are helper and caller; helper is used by caller.
 */
{TWO_SECTIONS}"""
    )

    report = _analyse(sections_module, capsys, source)

    helper = _static(report, "helper")
    assert helper["ref_count"] == 1, "prose mentions were counted as calls"
    assert helper["ref_sections"] == [_static(report, "caller")["section"]]


def test_trailing_block_comment_on_a_code_line_is_stripped(
    sections_module, capsys, tmp_path
):
    source = tmp_path / "trailing.c"
    source.write_text(
        TWO_SECTIONS + "\nstatic double unused(double x) /* calls helper */\n"
        "{\n    return x;\n}\n"
    )

    report = _analyse(sections_module, capsys, source)

    assert _static(report, "helper")["ref_count"] == 1


def test_code_after_a_block_comment_closes_is_still_counted(
    sections_module, capsys, tmp_path
):
    """The state machine must resume counting once ``*/`` closes."""
    source = tmp_path / "reopened.c"
    source.write_text(
        TWO_SECTIONS + "\nstatic double later(double x)\n{\n"
        "    /* a note\n       spanning lines */\n"
        "    return helper(x);\n}\n"
    )

    report = _analyse(sections_module, capsys, source)

    assert _static(report, "helper")["ref_count"] == 2


def test_module_registration_block_is_not_coupling(sections_module, capsys, tmp_path):
    """A PyMethodDef entry is the module naming itself, not a family dependency.

    The table sits at the end of the file, so without its own banner it is
    attributed to whichever section precedes it and every entry point in the
    module reads as shared with that one section.
    """
    source = tmp_path / "registered.c"
    source.write_text(
        TWO_SECTIONS
        + f"""
{BANNER}
 * MODULE REGISTRATION — methods[]
{BANNER} */
static PyMethodDef methods[] = {{
    {{"helper", (PyCFunction)helper, METH_VARARGS, "doc"}},
    {{NULL, NULL, 0, NULL}},
}};
"""
    )

    report = _analyse(sections_module, capsys, source)

    helper = _static(report, "helper")
    assert helper["ref_count"] == 1, "method-table entries were counted as calls"
    assert helper["ref_sections"] == [_static(report, "caller")["section"]]


def test_line_comment_is_stripped(sections_module, capsys, tmp_path):
    source = tmp_path / "line_comment.c"
    source.write_text(TWO_SECTIONS + "\n// helper is documented here\n")

    report = _analyse(sections_module, capsys, source)

    assert _static(report, "helper")["ref_count"] == 1
