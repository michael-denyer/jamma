"""Bit-exact fingerprint of every _lmm_accel C entry point, as a pytest plugin.

Splitting ``_lmm_accel.c`` must not move a single bit of any result. Tolerance
based tests cannot prove that: they pass just as happily when the last mantissa
bit drifts, which is exactly the failure mode a translation-unit split can
introduce (inlining changes FMA contraction and register allocation).

This plugin wraps every public callable in ``jamma.lmm._lmm_accel`` with a
recorder, lets the existing ``tests/lmm_accel/`` suite drive them with the
inputs the maintainers already care about, and writes one sorted line per
distinct ``(function, args digest, result digest)`` triple. Two runs across a
refactor that differ by one bit produce different files.

Usage::

    JAMMA_FINGERPRINT_OUT=/tmp/before.txt \\
      uv run pytest tests/lmm_accel/ -n0 -p no:randomly \\
      -p scripts.lmm_accel_fingerprint
    # ... make the change, rebuild the extension ...
    JAMMA_FINGERPRINT_OUT=/tmp/after.txt \\
      uv run pytest tests/lmm_accel/ -n0 -p no:randomly \\
      -p scripts.lmm_accel_fingerprint
    diff /tmp/before.txt /tmp/after.txt && echo "bit-identical"

``-n0`` is required: under xdist each worker would write over the same file.
``-p no:randomly`` is not required for correctness (the output is a sorted set)
but keeps a failing run's ordering reproducible.

Digests cover raw float64 bytes, so ``-0.0`` vs ``0.0`` and differing NaN
payloads both register as changes. That is deliberate. Judge such a diff on its
merits rather than widening the digest to hide it.
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

_records: set[str] = set()
_ENV_OUT = "JAMMA_FINGERPRINT_OUT"


def _feed(obj: Any, h: Any) -> None:
    """Fold one value into the running hash, byte-exactly for floats."""
    if isinstance(obj, np.ndarray):
        h.update(b"nd|")
        h.update(f"{obj.dtype}|{obj.shape}|".encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, np.generic):
        h.update(b"ns|")
        h.update(f"{obj.dtype}|".encode())
        h.update(np.asarray(obj).tobytes())
    elif isinstance(obj, bool):
        h.update(b"b|" + repr(obj).encode())
    elif isinstance(obj, int):
        h.update(b"i|" + repr(obj).encode())
    elif isinstance(obj, float):
        h.update(b"f|" + struct.pack("<d", obj))
    elif isinstance(obj, str | bytes):
        h.update(b"s|")
        h.update(obj.encode() if isinstance(obj, str) else obj)
    elif obj is None:
        h.update(b"none|")
    elif isinstance(obj, dict):
        h.update(b"{")
        for key in sorted(obj, key=repr):
            h.update(repr(key).encode() + b":")
            _feed(obj[key], h)
        h.update(b"}")
    elif isinstance(obj, list | tuple):
        h.update(b"[")
        for item in obj:
            _feed(item, h)
        h.update(b"]")
    else:
        # Opaque workspace capsules. Their contents are unreachable from
        # Python, but every result computed from one is still digested, so a
        # workspace that changed would surface downstream.
        h.update(b"opaque|" + type(obj).__name__.encode())


def _digest(*values: Any) -> str:
    h = hashlib.sha256()
    for value in values:
        _feed(value, h)
    return h.hexdigest()[:32]


def _wrap(name: str, fn: Any) -> Any:
    def recorder(*args: Any, **kwargs: Any) -> Any:
        args_digest = _digest(args, kwargs)
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            _records.add(f"{name}\t{args_digest}\traise:{type(exc).__name__}")
            raise
        _records.add(f"{name}\t{args_digest}\t{_digest(result)}")
        return result

    recorder.__name__ = getattr(fn, "__name__", name)
    recorder.__doc__ = getattr(fn, "__doc__", None)
    return recorder


def _install() -> None:
    """Wrap every public C callable on the extension.

    Callers reach each kernel as an attribute lookup on the extension at call
    time, so replacing the attribute here covers every call site.

    This used to need a second pass sweeping ``sys.modules`` by ``id()``:
    ``compute_numpy`` copied each C function into a module-level ``_compute_*``
    global at import, and those copies kept pointing at the unwrapped original.
    Those globals are gone, so the sweep matched nothing.
    ``test_no_jamma_module_holds_a_raw_c_callable`` fails if a copy is
    reintroduced, which is the condition that would bring the sweep back.
    """
    import jamma.lmm._lmm_accel as accel

    for name in dir(accel):
        if name.startswith("_"):
            continue
        attr = getattr(accel, name)
        if callable(attr) and not isinstance(attr, type):
            setattr(accel, name, _wrap(name, attr))


if os.environ.get(_ENV_OUT):
    _install()


def pytest_sessionfinish() -> None:
    out = os.environ.get(_ENV_OUT)
    if not out:
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(_records)) + "\n")
    print(f"\nfingerprint: {len(_records)} records -> {path}", flush=True)
