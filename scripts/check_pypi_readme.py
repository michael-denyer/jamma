#!/usr/bin/env python3
"""Assert the PyPI long description renders the way the project page will.

PyPI runs the long description through readme_renderer + the nh3 sanitizer.
Two failure modes are invisible on GitHub and only show up on the live project
page, which is the worst place to find them:

  * a ```mermaid fence lands as a literal <pre lang="mermaid"> block of source
    text, because PyPI ships no mermaid JS (jamma 8.0.2 shipped like this);
  * relative links resolve against pypi.org/project/jamma/<version>/ and 404.

The fancy-pypi-readme substitutions in pyproject.toml fix both at build time.
This check proves they still fire — a substitution whose regex quietly stops
matching fails loudly here instead of on release day.

Usage:
    uv run python scripts/check_pypi_readme.py
"""

from __future__ import annotations

import re
import subprocess
import sys

from _lint_common import repo_root, report
from readme_renderer.markdown import render

# readme_renderer emits its own heading permalinks as "#user-content-<slug>",
# and those targets do exist. Any other fragment link is README-authored and
# dead, because the renderer prefixes every heading id it emits.
LIVE_ON_PYPI = ("http://", "https://", "mailto:", "#user-content-")


def pypi_long_description() -> str:
    """Return README.md with the pyproject.toml substitutions applied."""
    result = subprocess.run(
        [sys.executable, "-m", "hatch_fancy_pypi_readme"],
        cwd=repo_root(),
        # stdout only: hatch_fancy_pypi_readme's own error text has to reach
        # the terminal when check=True raises, or the failure is undiagnosable.
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout


def dead_links(html: str) -> list[str]:
    """Return the rendered hrefs that have no target on the PyPI project page."""
    hrefs = set(re.findall(r'href="([^"]+)"', html))
    return sorted(href for href in hrefs if not href.startswith(LIVE_ON_PYPI))


def main() -> int:
    html = render(pypi_long_description(), stream=None)
    if html is None:
        print("readme_renderer rejected the long description", file=sys.stderr)
        return 1

    failures = []
    if 'lang="mermaid"' in html:
        failures.append(
            "a ```mermaid fence survived — PyPI will show it as raw source text"
        )
    dead = dead_links(html)
    if dead:
        failures.append(f"links that will 404 on PyPI: {', '.join(dead)}")

    if failures:
        return report(
            "The PyPI long description will not render as the project page needs:",
            failures,
            "Fix the substitutions under "
            "[tool.hatch.metadata.hooks.fancy-pypi-readme] in\n"
            "pyproject.toml, then rerun scripts/render_readme_diagram.py if the "
            "diagram changed.",
        )

    print("pypi readme: renders clean, no mermaid fences, no relative links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
