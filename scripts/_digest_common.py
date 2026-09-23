"""Shared value hashing and the ``--out``/``--diff`` CLI for the digest levers.

``assoc_digest.py`` and ``kinship_digest.py`` import this with a bare
``import _digest_common``, which resolves because ``python scripts/x.py``
puts ``scripts/`` on ``sys.path[0]``. The digest workflows stage this file
beside the head's script in the base checkout, so both sides hash with the
head's encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from jamma import jlinalg

REPO_ROOT = Path(__file__).resolve().parent.parent


def _feed(h: Any, value: Any) -> None:
    if value is None:
        h.update(b"N|")
    elif isinstance(value, bool):
        h.update(b"b|" + repr(value).encode())
    elif isinstance(value, int):
        h.update(b"i|" + repr(value).encode())
    elif isinstance(value, float):
        h.update(b"f|" + struct.pack("<d", value))
    elif isinstance(value, str):
        h.update(b"s|" + value.encode() + b"|")
    elif isinstance(value, bytes):
        h.update(b"B|" + value)
    else:
        raise TypeError(f"cannot digest {type(value).__name__}")


def digest_values(*values: Any) -> str:
    """Hash scalars with a type tag each, so ``1``, ``1.0`` and ``"1"`` differ."""
    h = hashlib.sha256()
    for value in values:
        _feed(h, value)
    return h.hexdigest()


def digest_array(arr: np.ndarray) -> str:
    """Shape-prefixed so a reshape cannot collide with a same-byte-count array."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _header() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = None
    return {
        "blas_backend": jlinalg.blas_backend,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "sha": sha,
    }


def _load(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload = json.loads(path.read_text())
    return payload["header"], payload["digests"]


def _diff(path_a: Path, path_b: Path) -> int:
    header_a, digests_a = _load(path_a)
    header_b, digests_b = _load(path_b)

    for field in ("blas_backend", "platform"):
        if header_a.get(field) != header_b.get(field):
            print(
                f"ERROR: {field} differs between runs "
                f"({header_a.get(field)!r} vs {header_b.get(field)!r}); "
                "a digest comparison across backends or platforms is meaningless.",
                file=sys.stderr,
            )
            return 2

    keys_a, keys_b = set(digests_a), set(digests_b)
    shared = keys_a & keys_b
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    differing = sorted(k for k in shared if digests_a[k] != digests_b[k])

    for label, keys in (("only in A", only_a), ("only in B", only_b)):
        if keys:
            print(f"{len(keys)} key(s) {label} (coverage change, not compared):")
            for key in keys:
                print(f"  {key}")

    if differing:
        print(f"{len(differing)} keys differ:", file=sys.stderr)
        for key in differing:
            print(f"  {key}  A={digests_a[key]}  B={digests_b[key]}", file=sys.stderr)
        return 1

    print(f"0 keys differ ({len(shared)} shared, {len(shared)} identical)")
    return 0


def run_cli(
    name: str,
    description: str | None,
    compute_all: Callable[[], dict[str, str]],
    argv: list[str] | None,
) -> int:
    """Parse ``--out FILE`` or ``--diff A B`` and run it.

    Args:
        name: Program name printed with the key count.
        description: Help text, normally the calling script's ``__doc__``.
        compute_all: Returns ``key -> sha256`` for the current checkout.
        argv: Arguments, or None for ``sys.argv[1:]``.

    Returns:
        0 when written or identical, 1 when keys differ, 2 when the two
        headers disagree on BLAS backend or platform.
    """
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, metavar="FILE", help="write digests to FILE")
    group.add_argument(
        "--diff",
        nargs=2,
        type=Path,
        metavar=("A", "B"),
        help="compare two digest files",
    )
    args = parser.parse_args(argv)

    if args.diff is not None:
        return _diff(*args.diff)
    payload = {"header": _header(), "digests": compute_all()}
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{name}: {len(payload['digests'])} keys -> {args.out}")
    return 0
