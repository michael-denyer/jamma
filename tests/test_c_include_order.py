"""Every LMM C unit that reaches the CPython API must include Python.h first.

CPython requires Python.h before any standard header. In this tree the
concrete casualty is ``M_PI``, which is not C11: glibc's ``<math.h>`` defines
it only under ``__USE_XOPEN``, which ``Python.h`` turns on via
``_XOPEN_SOURCE``. Let any other header reach ``<math.h>`` first and the
include guard blocks the later expansion, so every ``M_PI`` in
an accelerator family fails to compile.

The trap is that this is invisible on macOS, whose libc defines ``M_PI``
unconditionally. A local build and the ARM Mac CI job both pass while every
Linux job fails, which costs a full CI round trip to discover. That happened
when ``_lmm_stats.h`` was added, hence this guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jamma._build_support.build_models import LMM_ACCEL_SOURCES

_LMM_DIR = Path(__file__).resolve().parents[1] / "src/jamma/lmm"

#: The header that owns ``#include <Python.h>`` for the accelerator.
_PYTHON_OWNER = '#include "_lmm_support.h"'

#: The accelerator family units, read from the build manifest rather than
#: copied. A new ``_lmm_accel_*.c`` added to ``LMM_ACCEL_SOURCES`` is covered
#: by these guards the moment it can reach a build, so a unit cannot ship
#: outside the include-order and NumPy-C-API checks.
_ACCEL_UNITS: tuple[str, ...] = tuple(
    name for name in LMM_ACCEL_SOURCES if name.startswith("_lmm_accel")
)

#: The unit that registers the module and owns ``import_array()``.
_MODULE_UNIT = "_lmm_accel.c"

#: The family units that must borrow the module unit's C-API pointer.
_FAMILY_UNITS: tuple[str, ...] = tuple(n for n in _ACCEL_UNITS if n != _MODULE_UNIT)

_INCLUDE = re.compile(r'^\s*#\s*include\s+([<"][^>"]+[>"])', re.MULTILINE)

pytestmark = pytest.mark.tier0


def test_accel_units_derived_from_the_build_manifest():
    """The guards below must cover a real, non-trivial set of units.

    A typo in the prefix filter, or a manifest rename, would silently empty
    ``_ACCEL_UNITS`` and every loop below would pass by iterating nothing.
    """
    assert _MODULE_UNIT in _ACCEL_UNITS, (
        f"{_MODULE_UNIT} must be in LMM_ACCEL_SOURCES; got {_ACCEL_UNITS}"
    )
    assert _FAMILY_UNITS, (
        "no accelerator family units found in LMM_ACCEL_SOURCES — the guards "
        f"below would vacuously pass; got {_ACCEL_UNITS}"
    )


def _first_include(source: Path) -> str:
    text = source.read_text()
    match = _INCLUDE.search(text)
    assert match is not None, f"{source.name} has no #include at all"
    return match.group(1)


def test_accelerator_units_include_the_python_owner_first():
    """Nothing may reach <math.h> before the header that owns Python.h."""
    for name in _ACCEL_UNITS:
        source = _LMM_DIR / name
        assert _first_include(source) == '"_lmm_accel_internal.h"', (
            f"_lmm_accel_internal.h must be the first include in {name}; it "
            "pulls in Python.h before any standard header can reach math.h"
        )


def test_only_module_unit_owns_the_numpy_api_table():
    """Family units share the module unit's imported NumPy C-API pointer."""
    owner = (_LMM_DIR / _MODULE_UNIT).read_text()
    assert "#define NO_IMPORT_ARRAY" not in owner
    assert owner.count("import_array()") == 1

    for name in _FAMILY_UNITS:
        text = (_LMM_DIR / name).read_text()
        assert text.index("#define NO_IMPORT_ARRAY") < text.index(
            '#include "_lmm_accel_internal.h"'
        )
        assert "import_array()" not in text


def test_stats_unit_stays_free_of_the_python_api():
    """_lmm_stats is pure math; pulling in Python.h would recreate the trap."""
    for name in ("_lmm_stats.c", "_lmm_stats.h"):
        text = (_LMM_DIR / name).read_text()
        assert "Python.h" not in text, (
            f"{name} is the pure-arithmetic unit. It must not reach the CPython "
            "API, or its <math.h> include becomes order-sensitive again."
        )
