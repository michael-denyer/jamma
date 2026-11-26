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
BUILD_REQUIREMENTS = ROOT / "docker/requirements-build.txt"
HASHED_REQUIREMENTS = (REQUIREMENTS, NUMPY_MKL_REQUIREMENTS, BUILD_REQUIREMENTS)
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


def _build_system_pins() -> dict[str, str]:
    requires = tomllib.loads(PYPROJECT.read_text())["build-system"]["requires"]
    return _pins("\n".join(requires))


def _build_backend_numpy_pin() -> str:
    pins = _build_system_pins()
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


def _pip_commands() -> list[str]:
    """Return every `pip install` and `pip wheel` command, one per `&&` segment."""
    source = DOCKERFILE.read_text().replace("\\\n", " ")
    commands = [
        segment.strip()
        for line in source.splitlines()
        for segment in line.split("&&")
        if re.search(r"\bpip (?:install|wheel)\b", segment)
    ]
    assert commands, "the Dockerfile must install something"
    return commands


def test_image_installs_this_checkout_not_published_jamma():
    source = DOCKERFILE.read_text()
    assert "COPY pyproject.toml README.md hatch_build.py ./" in source
    assert "COPY src ./src" in source
    commands = _pip_commands()
    assert any(
        "pip wheel" in c and "--no-build-isolation" in c and c.endswith(" .")
        for c in commands
    ), "the image must build a wheel from this checkout"
    assert any(c.endswith(" /tmp/dist/*.whl") for c in commands), (
        "the image must install the wheel it built"
    )
    assert not re.search(r"pip install[^\n]*\bjamma(?:[=<>!~]|\s|$)", source)


def test_every_installed_package_is_pinned_or_this_checkout():
    """Every install argument is a hashed requirements file or this checkout.

    Keyed on the shape an unpinned input has, not on the spelling of inputs
    previously removed: a range, a bare name, a floating tag, and an exact pin
    on the command line (which cannot carry a hash) all fail here.
    """
    allowed = re.compile(
        r"\."  # this checkout
        r"|/tmp/dist/\*\.whl"  # the wheel built from this checkout
        r"|-r|/tmp/requirements-(?:container|numpy-mkl|build)\.txt"  # hashed files
        r"|--[a-z-]+(?:=\S+)?"  # pip flags
        r"|https://\S+"  # --index-url value
        r"|/tmp/build-backend|/tmp/dist"  # --target and --wheel-dir values
        r"|PYTHONPATH=/tmp/build-backend"
        r"|RUN|python|-m|pip|install|wheel"
    )
    for command in _pip_commands():
        for token in command.split():
            assert allowed.fullmatch(token), (
                f"unpinned or unexpected install input {token!r} in: {command}"
            )


def test_every_download_is_hash_checked():
    """Every pip command either checks hashes or cannot download at all.

    The checkout is built and installed with `--no-index`, since pip refuses
    `--require-hashes` for a directory; its build backend comes from the hashed
    `requirements-build.txt` instead of an isolated build environment.
    """
    commands = _pip_commands()
    for command in commands:
        tokens = command.split()
        assert "--require-hashes" in tokens or "--no-index" in tokens, (
            f"neither --require-hashes nor --no-index in: {command}"
        )
    assert not any(
        "pip wheel" in c and "--no-build-isolation" not in c for c in commands
    ), "an isolated build downloads its backend without hash checks"


def test_build_backend_pins_match_the_build_system():
    """The hashed build backend is exactly [build-system].requires without numpy.

    The docker-build dependency group is what uv.lock hashes, and
    requirements-build.txt is exported from it, so all three must agree.
    """
    expected = {n: v for n, v in _build_system_pins().items() if n != "numpy"}
    assert expected, "[build-system].requires must pin the build backend exactly"
    group = tomllib.loads(PYPROJECT.read_text())["dependency-groups"]["docker-build"]
    assert _pins("\n".join(group)) == expected
    exported = _pins(BUILD_REQUIREMENTS.read_text())
    assert {n: exported.get(n) for n in expected} == expected


def test_dockerfile_numpy_matches_the_build_backend_pin():
    """The image's numpy equals the pin the published wheels compile against.

    The repo rule is that the wheel never builds against newer headers than it
    runs on. The image builds without isolation against its own ILP64 numpy,
    while wheels built elsewhere compile against the pyproject-pinned PyPI
    numpy, so the two pins must agree for the image to match a published wheel.
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
