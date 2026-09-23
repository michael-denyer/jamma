"""Every rejecting exit of the native dgemm, dsyrk, and eigh wrappers.

Each case calls the raw ``_jlinalg`` entry point with arguments that fail one
check, asserts the exception, and then repeats the call to show the reused
array arguments' refcounts do not drift and no allocation the call made
survives it. A missed release of a borrowed argument grows its refcount
monotonically, a missed release of a temporary (the WRITEBACKIFCOPY copy that
``inplace=True`` rejects) leaves a block behind on every call, and a double
release crashes the interpreter.
"""

from __future__ import annotations

import gc
import sys
import tracemalloc
from collections.abc import Callable

import numpy as np
import pytest

from jamma.jlinalg import HAS_C_EXTENSION

pytestmark = [
    pytest.mark.tier0,
    pytest.mark.skipif(not HAS_C_EXTENSION, reason="native _jlinalg required"),
]


def _unaligned(n_rows: int, n_cols: int) -> np.ndarray:
    storage = bytearray(4 + n_rows * n_cols * 8)
    out = np.frombuffer(storage, dtype=np.float64, count=n_rows * n_cols, offset=4)
    return out.reshape(n_rows, n_cols)


def _readonly(n_rows: int, n_cols: int) -> np.ndarray:
    out = np.zeros((n_rows, n_cols))
    out.flags.writeable = False
    return out


def _mod():
    from jamma.jlinalg import _jlinalg

    return _jlinalg


A = np.ones((3, 2))
B = np.ones((2, 4))
X = np.ones((3, 2))
K = np.eye(3)

DGEMM_REJECTIONS = {
    "B not convertible": ((A, [["x"]]), {}, ValueError, "could not convert"),
    "not 2-D": ((A, np.ones(2)), {}, ValueError, "must be 2-D"),
    "inner mismatch": ((A, np.ones((3, 4))), {}, ValueError, "inner dimensions"),
    "out not array": ((A, B), {"out": [0.0]}, TypeError, "numpy array"),
    "out dtype": ((A, B), {"out": np.zeros((3, 4), np.float32)}, ValueError, "float64"),
    "out strided": (
        (A, B),
        {"out": np.zeros((3, 8))[:, ::2]},
        ValueError,
        "C-contiguous",
    ),
    "out readonly": ((A, B), {"out": _readonly(3, 4)}, ValueError, "writeable"),
    "out unaligned": ((A, B), {"out": _unaligned(3, 4)}, ValueError, "aligned"),
    "out shape": ((A, B), {"out": np.zeros((4, 3))}, ValueError, "out shape"),
}

DSYRK_REJECTIONS = {
    "X not convertible": (([["x"]],), {}, ValueError, "could not convert"),
    "not 2-D": ((np.ones(3),), {}, ValueError, "must be a 2-D"),
    "out not array": ((X,), {"out": [0.0]}, TypeError, "numpy array"),
    "out dtype": ((X,), {"out": np.zeros((3, 3), np.float32)}, ValueError, "float64"),
    "out strided": (
        (X,),
        {"out": np.zeros((3, 6))[:, ::2]},
        ValueError,
        "C-contiguous",
    ),
    "out unaligned": ((X,), {"out": _unaligned(3, 3)}, ValueError, "aligned"),
    "out readonly": ((X,), {"out": _readonly(3, 3)}, ValueError, "writeable"),
    "out shape": ((X,), {"out": np.zeros((2, 2))}, ValueError, "out shape"),
}

EIGH_REJECTIONS = {
    "bad driver": ((K,), {"driver": "qr"}, ValueError, "driver must be"),
    "K not an array": (([[1.0]],), {}, TypeError, "WRITEBACKIFCOPY"),
    "not square": ((np.ones((2, 3)),), {}, ValueError, "2-D square"),
    "inplace on a copy": (
        (np.eye(3, dtype=np.float32),),
        {"inplace": True},
        ValueError,
        "temporary copy",
    ),
}


def _refcounts(args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Refcounts of the array-like arguments only.

    ``True`` and the other interpreter singletons are skipped: every thread in
    the process moves their count, and CPython 3.12+ makes them immortal.
    """
    return tuple(
        sys.getrefcount(v)
        for v in (*args, *kwargs.values())
        if isinstance(v, (np.ndarray, list))
    )


def _invoke(fn: Callable, args: tuple, kwargs: dict) -> object:
    return fn(*args, **kwargs)


_INVOKE_LINE = _invoke.__code__.co_firstlineno + 1


def _surviving_blocks(before: tracemalloc.Snapshot, after: tracemalloc.Snapshot) -> int:
    """Allocations made under ``_invoke`` that the native call never freed."""
    only_invoke = [tracemalloc.Filter(True, __file__, lineno=_INVOKE_LINE)]
    diffs = after.filter_traces(only_invoke).compare_to(
        before.filter_traces(only_invoke), "lineno"
    )
    return sum(diff.count_diff for diff in diffs)


def _assert_repeated_calls_do_not_drift(
    call: Callable[[], object], args: tuple, kwargs: dict
) -> None:
    for _ in range(3):
        call()
    before = _refcounts(args, kwargs)
    tracemalloc.start()
    try:
        baseline = tracemalloc.take_snapshot()
        for _ in range(50):
            call()
        gc.collect()
        after = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
    assert _refcounts(args, kwargs) == before, (
        "error path leaked or over-released a reference"
    )
    assert _surviving_blocks(baseline, after) == 0, "call leaked a temporary"


def _assert_rejects_without_leak(
    fn: Callable, args: tuple, kwargs: dict, exc: type[Exception], match: str
) -> None:
    def call() -> None:
        with pytest.raises(exc, match=match):
            _invoke(fn, args, kwargs)

    _assert_repeated_calls_do_not_drift(call, args, kwargs)


@pytest.mark.parametrize("case", DGEMM_REJECTIONS.values(), ids=DGEMM_REJECTIONS.keys())
def test_dgemm_rejection_releases_references(case) -> None:
    mod = _mod()
    if not mod.blas_has_dgemm:
        pytest.skip("vendor dgemm required for the native argument checks")
    _assert_rejects_without_leak(mod.dgemm, *case)


@pytest.mark.parametrize("case", DSYRK_REJECTIONS.values(), ids=DSYRK_REJECTIONS.keys())
def test_dsyrk_rejection_releases_references(case) -> None:
    mod = _mod()
    if not mod.blas_has_dsyrk:
        pytest.skip("vendor dsyrk required for the native argument checks")
    _assert_rejects_without_leak(mod.dsyrk, *case)


@pytest.mark.parametrize("case", EIGH_REJECTIONS.values(), ids=EIGH_REJECTIONS.keys())
def test_eigh_rejection_releases_references(case) -> None:
    _assert_rejects_without_leak(_mod().eigh, *case)


def test_eigh_inplace_rejection_leaves_input_untouched() -> None:
    K32 = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    snapshot = K32.copy()
    with pytest.raises(ValueError, match="temporary copy"):
        _mod().eigh(K32, inplace=True)  # type: ignore[bad-argument-type]
    np.testing.assert_array_equal(K32, snapshot)


@pytest.mark.parametrize("name", ["dgemm", "dsyrk", "eigh"])
def test_success_path_refcounts_are_stable(name: str) -> None:
    mod = _mod()
    if name == "dgemm" and not mod.blas_has_dgemm:
        pytest.skip("vendor dgemm required")
    if name == "dsyrk" and not mod.blas_has_dsyrk:
        pytest.skip("vendor dsyrk required")
    if name == "eigh" and not (mod.blas_has_dsyevd or mod.blas_has_dsyevr):
        pytest.skip("vendor LAPACK required")
    calls = {
        "dgemm": ((A, B), {"out": np.zeros((3, 4))}),
        "dsyrk": ((X,), {"out": np.zeros((3, 3))}),
        "eigh": ((np.eye(3),), {"inplace": True}),
    }
    args, kwargs = calls[name]
    fn = getattr(mod, name)
    _assert_repeated_calls_do_not_drift(lambda: _invoke(fn, args, kwargs), args, kwargs)
