"""Directory probing for vendor BLAS discovery.

blas_dispatch.c's step-4 numpy scan and its pip-mkl scan used to walk
``numpy.__file__`` and ``mkl.__file__`` through the CPython C API one
``pathlib`` call at a time (``PyObject_GetAttrString``, ``PyObject_CallMethod``,
``Py_DECREF`` for each path segment). That is Python's job: it only touches
importable modules and ``pathlib``, nothing dlopen needs C for. This module
returns the same candidate directories in the same order; ``blas_dispatch.c``
still does every ``opendir``/``dlopen``/``dlsym`` call, unchanged.

The C side calls ``probe_plan`` once per ``discover_*`` step through
``PyObject_CallMethod`` and gets back a list of ``(kind, path)`` pairs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _numpy_candidate_dirs() -> list[Path]:
    """Directories inside the installed numpy package that may hold vendor BLAS.

    Mirrors the previous C code's ``discover_system_blas`` step 4: numpy's own
    ``.libs``/``_core/.libs`` (delvewheel/auditwheel-repaired vendored
    libraries) and the sibling ``numpy.libs`` directory.
    """
    spec = importlib.util.find_spec("numpy")
    if spec is None or spec.origin is None:
        return []
    np_dir = Path(spec.origin).resolve().parent
    return [
        np_dir / ".libs",
        np_dir / "_core" / ".libs",
        np_dir.parent / "numpy.libs",
    ]


def _mkl_candidate_dirs() -> list[Path]:
    """Directories inside a pip-installed ``mkl`` package that may hold MKL.

    Mirrors the previous C code's ``discover_pip_mkl``: ``mkl.libs`` next to
    the ``mkl`` package, and next to its parent (covers both the flat and
    nested pip layouts seen in the wild).
    """
    spec = importlib.util.find_spec("mkl")
    if spec is None or spec.origin is None:
        return []
    mkl_dir = Path(spec.origin).resolve().parent
    return [
        mkl_dir / "mkl.libs",
        mkl_dir.parent / "mkl.libs",
    ]


def probe_plan() -> list[tuple[str, str]]:
    """Return candidate BLAS library directories to scan, in probe order.

    Each entry is ``(kind, path)`` where ``kind`` is ``"openblas_or_mkl"``
    (scan for ``*openblas*``/``*libmkl*`` ``.so``/``.dylib`` files, the
    system-BLAS discovery step) or ``"mkl"`` (scan for the ordered
    ``libmkl_core``/``libmkl_sequential``/``libmkl_intel_ilp64`` triple, the
    pip-MKL discovery step). C does not choose directories; it only opens the
    ones this returns, in this order, and dlopens what it finds inside them.

    Returns:
        Candidate ``(kind, path)`` pairs. A directory that does not exist is
        still included; the C-side ``opendir`` call is the existence check,
        same as before this seam moved to Python.
    """
    plan: list[tuple[str, str]] = []
    plan.extend(("openblas_or_mkl", str(d)) for d in _numpy_candidate_dirs())
    plan.extend(("mkl", str(d)) for d in _mkl_candidate_dirs())
    return plan
