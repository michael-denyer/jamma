"""Check the Lean models' sensitivity using isolated project copies."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MUTATIONS = (
    ("MatrixBlocks", "zero block rows", "max 1 (values / columns)", "values / columns"),
    (
        "MatrixBlocks",
        "uncapped workers",
        "min requested ((rows + block - 1) / block)",
        "max requested ((rows + block - 1) / block)",
    ),
    (
        "MatrixBlocks",
        "floor instead of ceiling",
        "min requested ((rows + block - 1) / block)",
        "min requested (rows / block)",
    ),
    (
        "ChunkSlots",
        "slot overflow",
        "def slot (counter buffers : Nat) : Nat := counter % buffers",
        "def slot (counter buffers : Nat) : Nat := counter % buffers + buffers",
    ),
    (
        "ChunkSlots",
        "reuse consecutive slot",
        "def slot (counter buffers : Nat) : Nat := counter % buffers",
        "def slot (counter buffers : Nat) : Nat := (counter / 2) % buffers",
    ),
    (
        "MatrixBlocks",
        "missing advance",
        "min rows (start + block)",
        "min rows (start + block - 1)",
    ),
    (
        "MatrixBlocks",
        "undersized capacity",
        "min rows block * columns * bytes",
        "(min rows block - 1) * columns * bytes",
    ),
)


def main() -> int:
    checker = Path(sys.argv[1]).resolve(strict=True)
    root = Path(__file__).resolve().parent
    for project, label, old, new in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="jamma-lean-mutation-") as temporary:
            target = Path(temporary) / project
            shutil.copytree(
                root / project, target, ignore=shutil.ignore_patterns(".lake")
            )
            model = target / "Model.lean"
            original = model.read_text()
            assert original.count(old) == 1, (project, old)
            model.write_text(original.replace(old, new))
            checked = subprocess.run(
                ["bash", str(checker), str(target)], capture_output=True, text=True
            )
            output = checked.stdout + checked.stderr
            # A rejected model must fail a property, not the setup or invocation.
            if checked.returncode == 0 or "error:" not in output:
                print(f"FAIL mutation {project}/{label}: {output}")
                return 1
            print(f"DETECTED {project}/{label}")
            for line in output.splitlines():
                if "info: Model.lean" in line or "error: Model.lean" in line:
                    print(line[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
