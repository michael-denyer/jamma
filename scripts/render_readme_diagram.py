#!/usr/bin/env python3
"""Render the README architecture diagram to a PNG for PyPI.

PyPI's markdown renderer emits a ```mermaid fence as a literal <pre> block and
ships no mermaid JS, so the diagram arrives as raw source text. The sdist
metadata substitutes a PNG in its place (see the fancy-pypi-readme hook in
pyproject.toml); this script produces that PNG from the README fence so the two
never diverge.

README.md is the single source of truth. docs/architecture.mmd is an extracted
copy kept only so --check can detect drift without invoking mermaid.

Usage:
    scripts/render_readme_diagram.py          # regenerate .mmd and .png
    scripts/render_readme_diagram.py --check  # fail if the .mmd is stale
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from _lint_common import repo_root

REPO_ROOT = repo_root()
README = REPO_ROOT / "README.md"
MMD = REPO_ROOT / "docs" / "architecture.mmd"
PNG = REPO_ROOT / "docs" / "architecture.png"
CONFIG = REPO_ROOT / "docs" / "mermaid-render.json"

FENCE = re.compile(r"^```mermaid\n(.*?)^```", re.DOTALL | re.MULTILINE)

# The canvas behind the diagram. GitHub supplies its own in dark mode; a
# standalone PNG has to carry one, and PyPI's page is always light.
BACKGROUND = "#1a1a2e"

# 2x for legible text on high-DPI displays; keeps the PNG well under the
# repo's 500 KB check-added-large-files cap.
SCALE = "2"


def extract_fence() -> str:
    """Return the sole mermaid block in README.md.

    Raises:
        SystemExit: If README.md does not contain exactly one mermaid fence.
    """
    blocks = FENCE.findall(README.read_text())
    if len(blocks) != 1:
        sys.exit(f"expected exactly 1 mermaid fence in README.md, found {len(blocks)}")
    return blocks[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify docs/architecture.mmd matches the README fence; render nothing",
    )
    args = parser.parse_args()

    fence = extract_fence()

    if args.check:
        if not MMD.exists() or MMD.read_text() != fence:
            sys.exit(
                f"{MMD.relative_to(REPO_ROOT)} is stale — the README diagram changed.\n"
                "Run: scripts/render_readme_diagram.py"
            )
        return 0

    MMD.write_text(fence)
    subprocess.run(
        [
            "npx",
            "--yes",
            "@mermaid-js/mermaid-cli",
            "--input",
            str(MMD),
            "--output",
            str(PNG),
            "--backgroundColor",
            BACKGROUND,
            "--scale",
            SCALE,
            "--configFile",
            str(CONFIG),
        ],
        check=True,
    )
    print(f"wrote {MMD.relative_to(REPO_ROOT)} and {PNG.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
