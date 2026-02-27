#!/usr/bin/env python3
"""Check per-module coverage floors against coverage.json.

Usage:
    uv run python scripts/check_module_coverage.py [coverage.json]

Reads coverage.json (produced by pytest-cov with --cov-report=json) and checks
that each tracked module meets its configured minimum coverage floor. Exits 0 if
all modules pass, exits 1 if any module falls below its floor or if the file is
missing.

This script tracks modules with coverage below 75% — they are too low to be
caught by the global --cov-fail-under gate and need explicit floor enforcement to
prevent regressions from hiding under the high global average (~90%).

Floor values are set at current coverage minus ~5 percentage points, rounded down
to the nearest 5, giving headroom for minor test changes while catching real
regressions.
"""

import json
import sys
from pathlib import Path

# Modules with coverage below 75% that need explicit floor enforcement.
# Keys are source paths as they appear in coverage.json (relative to repo root).
# Values are integer minimum percentages.
MODULE_FLOORS: dict[str, int] = {
    "src/jamma/kinship/io.py": 50,
    "src/jamma/io/matrix_writer.py": 60,
    "src/jamma/core/hardware.py": 65,
}


def main(coverage_path: Path) -> int:
    """Check per-module coverage floors.

    Args:
        coverage_path: Path to coverage.json file.

    Returns:
        Exit code: 0 if all modules pass, 1 if any fail or file is missing.
    """
    if not coverage_path.exists():
        print(f"ERROR: {coverage_path} not found. Run pytest --cov-report=json first.")
        return 1

    data = json.loads(coverage_path.read_text())
    files = data.get("files", {})

    failures: list[str] = []

    for module_path, floor in MODULE_FLOORS.items():
        if module_path not in files:
            print(f"  WARN: {module_path} not in coverage.json (may not be imported)")
            continue

        actual = files[module_path]["summary"]["percent_covered"]
        if actual < floor:
            failures.append(f"  FAIL: {module_path}: {actual:.1f}% < {floor}% required")
        else:
            print(f"  PASS: {module_path}: {actual:.1f}% >= {floor}% required")

    if failures:
        print("\nPer-module coverage check FAILED:")
        for failure in failures:
            print(failure)
        return 1

    print(
        f"\nPer-module coverage checks passed ({len(MODULE_FLOORS)} modules checked)."
    )
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("coverage.json")
    sys.exit(main(path))
