"""Every C unit that reaches the CPython API must include Python.h first.

CPython requires Python.h before any standard header. In this tree the
concrete casualty is ``M_PI``, which is not C11: glibc's ``<math.h>`` defines
it only under ``__USE_XOPEN``, which ``Python.h`` turns on via
``_XOPEN_SOURCE``. Let any other header reach ``<math.h>`` first and the
include guard blocks the later expansion, so every ``M_PI`` in
``_lmm_accel.c`` fails to compile.

The trap is that this is invisible on macOS, whose libc defines ``M_PI``
unconditionally. A local build and the ARM Mac CI job both pass while every
Linux job fails, which costs a full CI round trip to discover. That happened
when ``_lmm_stats.h`` was added, hence this guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_LMM_DIR = Path(__file__).resolve().parents[1] / "src/jamma/lmm"

#: The header that owns ``#include <Python.h>`` for the accelerator.
_PYTHON_OWNER = '#include "_lmm_support.h"'

_INCLUDE = re.compile(r'^\s*#\s*include\s+([<"][^>"]+[>"])', re.MULTILINE)

pytestmark = pytest.mark.tier0


def _first_include(source: Path) -> str:
    text = source.read_text()
    match = _INCLUDE.search(text)
    assert match is not None, f"{source.name} has no #include at all"
    return match.group(1)


def test_lmm_accel_includes_the_python_owner_first():
    """_lmm_accel.c uses M_PI, so nothing may reach <math.h> before Python.h."""
    source = _LMM_DIR / "_lmm_accel.c"
    assert "M_PI" in source.read_text(), (
        "guard is calibrated to M_PI; if that use is gone, re-check whether "
        "this ordering still needs enforcing"
    )
    assert _first_include(source) == '"_lmm_support.h"', (
        f"{_PYTHON_OWNER} must be the first include in _lmm_accel.c. It is what "
        "pulls in <Python.h>, and glibc only exposes M_PI from <math.h> once "
        "Python.h has set _XOPEN_SOURCE. macOS builds fine either way, so this "
        "only fails on Linux."
    )


def test_stats_unit_stays_free_of_the_python_api():
    """_lmm_stats is pure math; pulling in Python.h would recreate the trap."""
    for name in ("_lmm_stats.c", "_lmm_stats.h"):
        text = (_LMM_DIR / name).read_text()
        assert "Python.h" not in text, (
            f"{name} is the pure-arithmetic unit. It must not reach the CPython "
            "API, or its <math.h> include becomes order-sensitive again."
        )
