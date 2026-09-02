"""Dump or diff the memory-ledger digest table.

``tests/test_memory_ledger_digest.py`` pins the table as one sha256, which
says *that* a gate decision changed but not *which*. This script writes the
rows to JSON and diffs two dumps row by row so a deliberate policy change
can list exactly the rows it moved.

Usage:
    uv run python scripts/dump_memory_ledger.py dump /tmp/before.json
    uv run python scripts/dump_memory_ledger.py diff /tmp/before.json /tmp/after.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from tests.test_memory_ledger_digest import ledger_digest, ledger_table


def _load(path: str) -> list[list]:
    return json.loads(Path(path).read_text())


def dump(path: str) -> None:
    rows = ledger_table()
    Path(path).write_text(json.dumps(rows, indent=0))
    print(f"rows={len(rows)} digest={ledger_digest(rows)} -> {path}")


def diff(before_path: str, after_path: str) -> int:
    before, after = _load(before_path), _load(after_path)
    if len(before) != len(after):
        print(f"row count differs: {len(before)} vs {len(after)}")
        return 1
    changed = 0
    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        if b != a:
            changed += 1
            pairs = enumerate(zip(b, a, strict=True))
            moved = [(j, x, y) for j, (x, y) in pairs if x != y]
            print(f"row {i} {b[0]}: key={b[1:8]} changed={moved}")
    print(f"{changed} of {len(before)} rows changed")
    return 0


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "dump":
        dump(*args)
    elif cmd == "diff":
        sys.exit(diff(*args))
    else:
        raise SystemExit(__doc__)
