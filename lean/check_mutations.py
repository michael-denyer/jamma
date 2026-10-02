"""Check the Lean models' sensitivity using isolated project copies."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MUTATIONS = (
    (
        "MatrixBlocks",
        "zero block rows",
        "max 1 (values / columns)",
        "values / columns",
        "badBlockSizes.isEmpty",
    ),
    (
        "MatrixBlocks",
        "uncapped workers",
        "min requested ((rows + block - 1) / block)",
        "max requested ((rows + block - 1) / block)",
        "badWorkerCounts.isEmpty",
    ),
    (
        "MatrixBlocks",
        "floor instead of ceiling",
        "min requested ((rows + block - 1) / block)",
        "min requested (rows / block)",
        "badWorkerCounts.isEmpty",
    ),
    (
        "ChunkSlots",
        "slot overflow",
        "def slot (counter buffers : Nat) : Nat := counter % buffers",
        "def slot (counter buffers : Nat) : Nat := counter % buffers + buffers",
        "badPairs.isEmpty",
    ),
    (
        "ChunkSlots",
        "reuse consecutive slot",
        "def slot (counter buffers : Nat) : Nat := counter % buffers",
        "def slot (counter buffers : Nat) : Nat := (counter / 2) % buffers",
        "badPairs.isEmpty",
    ),
    (
        "MatrixBlocks",
        "missing advance",
        "min rows (start + block)",
        "min rows (start + block - 1)",
        "badSteps.isEmpty",
    ),
    (
        "MatrixBlocks",
        "undersized capacity",
        "min rows block * columns * bytes",
        "(min rows block - 1) * columns * bytes",
        "badSteps.isEmpty",
    ),
)


def main() -> int:
    checker = Path(sys.argv[1]).resolve(strict=True)
    root = Path(__file__).resolve().parent
    # Check every untouched project before accepting any mutation evidence.
    for project in dict.fromkeys(mutation[0] for mutation in MUTATIONS):
        with tempfile.TemporaryDirectory(prefix="jamma-lean-baseline-") as temporary:
            target = Path(temporary) / project
            shutil.copytree(
                root / project, target, ignore=shutil.ignore_patterns(".lake")
            )
            checked = subprocess.run(
                ["bash", str(checker), str(target)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if checked.returncode != 0:
                print(f"FAIL baseline {project}: {checked.stdout}{checked.stderr}")
                return 1
            print(f"BASELINE {project}")

    for project, label, old, new, guard in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="jamma-lean-mutation-") as temporary:
            target = Path(temporary) / project
            shutil.copytree(
                root / project, target, ignore=shutil.ignore_patterns(".lake")
            )
            model = target / "Model.lean"
            original = model.read_text()
            assert original.count(old) == 1, (project, old)
            mutated = original.replace(old, new)
            guard_line = mutated.splitlines().index(f"#guard {guard}") + 1
            model.write_text(mutated)
            checked = subprocess.run(
                ["bash", str(checker), str(target)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = checked.stdout + checked.stderr
            # Match Lean's evaluated-false guard diagnostic, not an import,
            # syntax, proof-elaboration or checker setup failure.
            rejection = (
                rf"^error: Model\.lean:{guard_line}:\d+: Expression\n"
                rf"  {re.escape(guard)}\ndid not evaluate to `true`$"
            )
            if checked.returncode == 0 or not re.search(
                rejection, output, re.MULTILINE
            ):
                print(f"FAIL mutation {project}/{label}: {output}")
                return 1
            print(f"DETECTED {project}/{label}")
            for line in output.splitlines():
                if "info: Model.lean" in line or "error: Model.lean" in line:
                    print(line[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
