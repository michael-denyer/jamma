"""Every ``SYMS[]`` row in ``blas_dispatch.c`` resolves symbols of one calling
convention into a field of that same convention.

The resolver writes whatever ``dlsym`` returns into the field at the row's
offset, and ``blas_operations.c`` then calls the field with the signature its
type declares. A Fortran name (``dsyrk_64_``) stored in a CBLAS field
(``cblas_dsyrk_ilp64``) is therefore called with CBLAS enums where the routine
expects ``char *`` and dereferences the integer 101 as a pointer. MKL
``libmkl_rt`` segfaulted this way on every ``jlinalg.dsyrk`` call; Accelerate
hid it because its CBLAS name sits first in the list, and the OpenBLAS CI leg
because its symbols carry a ``scipy_`` prefix that no name in the list matches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DISPATCH_C = (
    Path(__file__).resolve().parents[1] / "src/jamma/jlinalg/src/blas_dispatch.c"
)

_NAME_LIST = re.compile(
    r"static const char \*const (?P<list>\w+)\[\]\s*=\s*\{(?P<body>.*?)\};", re.S
)
_SYMS_ROW = re.compile(
    r'\{"(?P<label>[^"]+)",\s*(?P<list>\w+),\s*offsetof\(blas_candidate_t,\s*(?P<field>\w+)\)'
)


def _symbol_table() -> list[tuple[str, str, list[str]]]:
    source = _DISPATCH_C.read_text()
    lists = {
        m["list"]: re.findall(r'"([^"]+)"', m["body"])
        for m in _NAME_LIST.finditer(source)
    }
    rows = [
        (m["label"], m["field"], lists[m["list"]]) for m in _SYMS_ROW.finditer(source)
    ]
    assert rows, "no SYMS[] rows found in blas_dispatch.c"
    return rows


@pytest.mark.tier0
def test_every_syms_row_matches_its_field_calling_convention():
    for label, field, names in _symbol_table():
        assert names, f"{label}: empty candidate list"
        field_is_cblas = field.startswith("cblas_")
        for name in names:
            name_is_cblas = name.startswith("cblas_")
            assert name_is_cblas == field_is_cblas, (
                f"{label}: {name!r} resolved into {field}, which is called with the "
                f"{'CBLAS' if field_is_cblas else 'Fortran'} signature"
            )
