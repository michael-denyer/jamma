"""Tests that the citation and registry metadata name the package version.

Nothing regenerates ``CITATION.cff`` or ``docs/biotools.json`` on a version
bump, and ``build-wheels.yml`` sends the bio.tools entry to the registry on
release, so a stale version there is published as written.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _package_version() -> str:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_citation_names_the_package_version():
    citation = (_REPO_ROOT / "CITATION.cff").read_text()
    match = re.search(r"^version: (.+)$", citation, flags=re.M)
    assert match, "CITATION.cff has no version line"
    assert match.group(1) == _package_version()


def test_biotools_entry_names_the_package_version():
    entry = json.loads((_REPO_ROOT / "docs" / "biotools.json").read_text())
    assert entry["version"] == [_package_version()]
