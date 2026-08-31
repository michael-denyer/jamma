"""Static provenance contract for the published container recipe."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
REQUIREMENTS = ROOT / "docker/requirements-container.txt"
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"

pytestmark = pytest.mark.tier0


def _pins(text: str) -> dict[str, str]:
    """Map package name to version for every `name==version` line in text."""
    pins = {}
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^=\s]+)", entry)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _dockerfile_numpy_pin() -> str:
    match = re.search(r"\bnumpy==([^\s\\]+)", DOCKERFILE.read_text())
    assert match, "the Dockerfile must pin numpy to an exact version"
    return match.group(1)


def _build_backend_numpy_pin() -> str:
    requires = tomllib.loads(PYPROJECT.read_text())["build-system"]["requires"]
    pins = _pins("\n".join(requires))
    assert "numpy" in pins, "[build-system].requires must pin numpy exactly"
    return pins["numpy"]


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


def test_every_installed_package_is_pinned_or_this_checkout():
    """Every install argument is an exact pin, the requirements file, or `.`.

    Keyed on the shape an unpinned input has, not on the spelling of inputs
    previously removed: a range, a bare name, and a floating tag all fail here
    regardless of which comparison operator writes them.
    """
    source = DOCKERFILE.read_text().replace("\\\n", " ")
    install_lines = [
        line for line in source.splitlines() if re.search(r"\bpip install\b", line)
    ]
    assert install_lines, "the Dockerfile must install something"

    allowed = re.compile(
        r"[A-Za-z0-9_.-]+==[^=\s]+"  # exact pin
        r"|\."  # this checkout
        r"|-r|/tmp/requirements-container\.txt"  # the pinned requirements file
        r"|--[a-z-]+(?:=\S+)?"  # pip flags
        r"|https://\S+"  # --index-url value
        r"|RUN|python|-m|pip|install"
    )
    for line in install_lines:
        for token in line.split():
            assert allowed.fullmatch(token), (
                f"unpinned or unexpected install input {token!r} in: {line.strip()}"
            )


def test_dockerfile_numpy_matches_the_build_backend_pin():
    """The image's numpy equals the pin the extensions compile against.

    The repo rule is that the wheel never builds against newer headers than it
    runs on. Inside the image `pip install .` compiles under build isolation
    against the pyproject-pinned PyPI numpy, while the runtime numpy is the
    ILP64 wheel of the same version, so the two pins must agree.
    """
    assert _dockerfile_numpy_pin() == _build_backend_numpy_pin()


def test_container_runtime_pins_match_the_lockfile():
    """Shared packages agree between the container file and uv.lock.

    docker/generate-requirements.py derives these from the lockfile. This test
    is the gate that notices when the lockfile moves and nobody regenerated.
    """
    container = _pins(REQUIREMENTS.read_text())
    lock = tomllib.loads(LOCKFILE.read_text())
    locked = {
        package["name"].lower().replace("_", "-"): package["version"]
        for package in lock["package"]
        if "version" in package
    }

    shared = sorted(set(container) & set(locked))
    assert shared, "expected the container file to share packages with uv.lock"
    mismatched = {
        name: (container[name], locked[name])
        for name in shared
        if container[name] != locked[name]
    }
    assert not mismatched, (
        "container pins differ from uv.lock (container, lock): "
        f"{mismatched}. Regenerate with: "
        "uv run python docker/generate-requirements.py"
    )
