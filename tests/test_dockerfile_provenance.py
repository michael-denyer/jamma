"""Static provenance contract for the published container recipe."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
REQUIREMENTS = ROOT / "docker/requirements-container.txt"

pytestmark = pytest.mark.tier0


def test_base_images_are_immutable_and_versioned():
    from_lines = [
        line for line in DOCKERFILE.read_text().splitlines() if line.startswith("FROM ")
    ]
    assert len(from_lines) == 2
    assert all(
        re.fullmatch(
            r"FROM python:\d+\.\d+\.\d+-(?:slim-)?bookworm"
            r"@sha256:[0-9a-f]{64}(?: AS build)?",
            line,
        )
        for line in from_lines
    )


def test_container_requirements_are_exactly_pinned():
    assert REQUIREMENTS.is_file()
    requirements = [
        line.strip()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", item) for item in requirements)


def test_image_installs_this_checkout_not_published_jamma():
    source = DOCKERFILE.read_text()
    assert "COPY pyproject.toml README.md hatch_build.py ./" in source
    assert "COPY src ./src" in source
    assert "python -m pip install --no-cache-dir --no-deps ." in source
    assert not re.search(r"pip install[^\n]*\bjamma(?:[=<>!~]|\s|$)", source)


def test_mutable_install_inputs_are_absent():
    source = DOCKERFILE.read_text()
    assert ":latest" not in source
    assert "numpy<" not in source
    assert "numpy==2.4.6" in source
    assert "mkl-service==2.7.2" in source
    assert "--no-deps" in source
