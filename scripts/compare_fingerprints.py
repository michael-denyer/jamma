"""Compare two ``lmm_accel_fingerprint`` outputs and report bit-level drift.

Each record is ``<entry point>\\t<args digest>\\t<result digest>``, so a record
is keyed by which C function was called with which inputs. Two runs of the same
test suite produce the same keys; a key present in both whose result digest
differs means the C code returned different bytes for identical inputs.

Keys present on only one side are coverage changes, not drift. A pull request
that adds an entry point or a test case adds keys, and one that removes a test
drops them. Those are reported and do not fail the comparison, because there is
nothing to compare them against.

Usage::

    uv run python scripts/compare_fingerprints.py BASE.txt HEAD.txt

Exit status is 1 when any shared key disagrees, which is the gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    """Read a fingerprint file into ``{(entry point, args digest): results}``.

    The value is a tuple, not a single digest, because the key is not unique.
    The args digest cannot see inside an opaque ``PyCapsule``, so the
    workspace-taking entry points record the same key twice: once returning a
    value with a live workspace, once raising with a spent one. Collapsing
    those to one result would silently drop records and hide drift on them.
    """
    grouped: dict[tuple[str, str], list[str]] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(
                f"{path}:{lineno}: expected 3 tab-separated fields, got {len(parts)}"
            )
        name, args_digest, result = parts
        grouped.setdefault((name, args_digest), []).append(result)
    return {key: tuple(sorted(results)) for key, results in grouped.items()}


def compare(
    base: dict[tuple[str, str], tuple[str, ...]],
    head: dict[tuple[str, str], tuple[str, ...]],
):
    """Return (drifted, added, removed) for two record sets."""
    shared = base.keys() & head.keys()
    drifted = sorted(k for k in shared if base[k] != head[k])
    return drifted, sorted(head.keys() - base.keys()), sorted(base.keys() - head.keys())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="fingerprint from the merge base")
    parser.add_argument("head", type=Path, help="fingerprint from the PR head")
    args = parser.parse_args(argv)

    base, head = load(args.base), load(args.head)
    drifted, added, removed = compare(base, head)
    shared = len(base.keys() & head.keys())
    base_n = sum(len(v) for v in base.values())
    head_n = sum(len(v) for v in head.values())

    print(
        f"base: {base_n} records / {len(base)} keys, "
        f"head: {head_n} records / {len(head)} keys, {shared} keys shared"
    )

    if not shared:
        print(
            "\nNo shared records. The two runs exercised nothing in common, so "
            "this proves nothing about the C code. Check that both sides built "
            "and that the test suite ran.",
            file=sys.stderr,
        )
        return 1

    for label, keys in (("added by this change", added), ("removed", removed)):
        if keys:
            print(f"\n{len(keys)} record(s) {label} (not compared):")
            for name, args_digest in keys[:10]:
                print(f"  {name}  {args_digest}")
            if len(keys) > 10:
                print(f"  ... and {len(keys) - 10} more")

    if not drifted:
        print(f"\nAll {shared} shared keys are bit-identical.")
        return 0

    print(f"\n{len(drifted)} of {shared} shared keys CHANGED:", file=sys.stderr)
    for name, args_digest in drifted[:20]:
        print(
            f"  {name}  args={args_digest}\n"
            f"    base={', '.join(base[(name, args_digest)])}\n"
            f"    head={', '.join(head[(name, args_digest)])}",
            file=sys.stderr,
        )
    if len(drifted) > 20:
        print(f"  ... and {len(drifted) - 20} more", file=sys.stderr)
    print(
        "\nThe same C entry point returned different bytes for identical inputs. "
        "If that is intended, say so in the pull request and explain why the "
        "numerics moved; tolerance-based tests will not catch drift this small.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
