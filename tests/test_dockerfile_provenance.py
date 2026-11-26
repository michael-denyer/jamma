"""Static provenance contract for the published container recipe."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
REQUIREMENTS = ROOT / "docker/requirements-container.txt"
NUMPY_MKL_REQUIREMENTS = ROOT / "docker/requirements-numpy-mkl.txt"
HASHED_REQUIREMENTS = (REQUIREMENTS, NUMPY_MKL_REQUIREMENTS)
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"

pytestmark = pytest.mark.tier0


def _entries(text: str) -> list[tuple[str, list[str]]]:
    """Split requirements text into (requirement, hash options) per entry.

    Continuation lines are joined first, so a pin followed by its
    `--hash=sha256:...` lines is one entry.
    """
    entries = []
    for line in text.replace("\\\n", " ").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        requirement, *options = entry.split()
        entries.append((requirement, options))
    return entries


def _pins(text: str) -> dict[str, str]:
    """Map package name to version for every `name==version` entry in text."""
    pins = {}
    for requirement, _ in _entries(text):
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^=\s]+)", requirement)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _image_numpy_pin() -> str:
    pins = _pins(NUMPY_MKL_REQUIREMENTS.read_text())
    assert "numpy" in pins, "requirements-numpy-mkl.txt must pin numpy exactly"
    return pins["numpy"]


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


@pytest.mark.parametrize("path", HASHED_REQUIREMENTS, ids=lambda path: path.name)
def test_container_requirements_are_exactly_pinned_and_hashed(path):
    entries = _entries(path.read_text())
    assert entries
    for requirement, options in entries:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", requirement), requirement
        assert options, f"{requirement} in {path.name} has no --hash"
        assert all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", o) for o in options), (
            f"{requirement} in {path.name} has a non-sha256 option: {options}"
        )


def test_image_installs_this_checkout_not_published_jamma():
    source = DOCKERFILE.read_text()
    assert "COPY pyproject.toml README.md hatch_build.py ./" in source
    assert "COPY src ./src" in source
    assert "python -m pip install --no-cache-dir --no-deps ." in source
    assert not re.search(r"pip install[^\n]*\bjamma(?:[=<>!~]|\s|$)", source)


def _install_lines() -> list[str]:
    source = DOCKERFILE.read_text().replace("\\\n", " ")
    lines = [
        line for line in source.splitlines() if re.search(r"\bpip install\b", line)
    ]
    assert lines, "the Dockerfile must install something"
    return lines


def test_every_installed_package_is_pinned_or_this_checkout():
    """Every install argument is a hashed requirements file or `.`.

    Keyed on the shape an unpinned input has, not on the spelling of inputs
    previously removed: a range, a bare name, a floating tag, and an exact pin
    on the command line (which cannot carry a hash) all fail here.
    """
    allowed = re.compile(
        r"\."  # this checkout
        r"|-r|/tmp/requirements-(?:container|numpy-mkl)\.txt"  # hashed pin files
        r"|--[a-z-]+(?:=\S+)?"  # pip flags
        r"|https://\S+"  # --index-url value
        r"|RUN|python|-m|pip|install"
    )
    for line in _install_lines():
        for token in line.split():
            assert allowed.fullmatch(token), (
                f"unpinned or unexpected install input {token!r} in: {line.strip()}"
            )


def test_every_download_is_hash_checked():
    """Every install that fetches packages runs pip in hash-checking mode.

    Only the install of this checkout (`.`) is exempt: pip refuses
    `--require-hashes` for a directory.
    """
    for line in _install_lines():
        tokens = line.split()
        if "." in tokens:
            continue
        assert "--require-hashes" in tokens, f"no --require-hashes in: {line.strip()}"


def test_dockerfile_numpy_matches_the_build_backend_pin():
    """The image's numpy equals the pin the extensions compile against.

    The repo rule is that the wheel never builds against newer headers than it
    runs on. Inside the image `pip install .` compiles under build isolation
    against the pyproject-pinned PyPI numpy, while the runtime numpy is the
    ILP64 wheel of the same version, so the two pins must agree.
    """
    assert _image_numpy_pin() == _build_backend_numpy_pin()


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
