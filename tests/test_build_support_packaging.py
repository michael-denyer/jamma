"""Tests that jamma._build_support ships in both sdist and wheel.

Contract: jamma._build_support holds the canonical compile flags and the
compile+link driver. It MUST ship inside the installed wheel so that
jamma.core.recompile.auto_recompile_c_extension (the runtime ABI-mismatch
recompile path end users depend on) can reach the same helpers the wheel
was built with. A wheel that omits jamma._build_support makes
auto_recompile_c_extension dead code — every ABI mismatch silently falls
back to pure-Python.

Runs via ``uv build``. Marked tier2 because it shells out to the build
backend and takes several seconds; the hard contract is also checked
structurally in test_wheel_contains_build_support_structural below,
which is tier0 and runs in every CI job.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD_SUPPORT_MODULES = (
    "build_models.py",
    "build_execution.py",
    "compile_and_link.py",
    "load_proof.py",
)


@pytest.mark.tier0
def test_wheel_contains_build_support_structural():
    """pyproject.toml must list src/jamma in the wheel-target packages.

    jamma._build_support is a subpackage of jamma, so any config that
    includes ``src/jamma`` automatically ships it. But we check the config
    text explicitly: if someone splits the packages list to exclude
    _build_support, the other (tier2) test will catch it — but tier2 is
    gated behind ``-m slow`` and excluded from default runs. This tier0
    check fails instantly in CI so the regression cannot slip through.
    """
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    # Look for the wheel target. We want either packages=["src/jamma"]
    # (blanket) or an explicit jamma._build_support entry. Anything that
    # excludes _build_support will fail the literal-match check below.
    assert "[tool.hatch.build.targets.wheel]" in pyproject, (
        "pyproject.toml missing [tool.hatch.build.targets.wheel] section"
    )
    assert 'packages = ["src/jamma"]' in pyproject, (
        "pyproject.toml wheel target does not ship src/jamma as a blanket "
        "package — jamma._build_support may be excluded. If the packages "
        "list is split, ensure jamma._build_support is listed explicitly "
        "AND update this assertion."
    )


@pytest.mark.tier0
def test_build_support_responsibilities_are_separate_modules():
    """Models, compiler execution, and orchestration must not collapse again."""
    support_dir = _REPO_ROOT / "src/jamma/_build_support"
    missing = [
        name for name in _BUILD_SUPPORT_MODULES if not (support_dir / name).is_file()
    ]
    assert not missing, f"build-support modules missing: {missing}"


def _build(tmp_out: Path) -> tuple[Path, Path]:
    """Run ``uv build -o <tmp_out>`` and return (sdist_path, wheel_path)."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not available on PATH")

    result = subprocess.run(
        [uv, "build", "-o", str(tmp_out)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"uv build failed — packaging contract cannot be verified. "
            f"stderr: {result.stderr[-800:]}"
        )

    sdists = list(tmp_out.glob("*.tar.gz"))
    wheels = list(tmp_out.glob("*.whl"))
    assert sdists, f"uv build did not produce sdist: {tmp_out}"
    assert wheels, f"uv build did not produce wheel: {tmp_out}"
    return sdists[0], wheels[0]


@pytest.mark.tier2
@pytest.mark.slow
def test_build_support_ships_in_sdist_and_wheel(tmp_path):
    """jamma._build_support must be present in both distributions.

    sdist: hatch_build.py imports it at wheel-build time via sys.path+src.
    wheel: jamma.core.recompile calls compile_extension() which imports it
    as a regular package. Missing from either distribution is a regression.
    """
    sdist_path, wheel_path = _build(tmp_path)

    with tarfile.open(sdist_path) as tar:
        sdist_names = tar.getnames()
    with zipfile.ZipFile(wheel_path) as zf:
        wheel_names = zf.namelist()

    for module in _BUILD_SUPPORT_MODULES:
        target = f"jamma/_build_support/{module}"
        assert any(name.endswith(target) for name in sdist_names), (
            f"{target} missing from sdist {sdist_path.name}; isolated source "
            "builds need the complete build-support package"
        )
        assert any(name.endswith(target) for name in wheel_names), (
            f"{target} missing from wheel {wheel_path.name}; runtime ABI "
            "recompilation needs the complete build-support package"
        )

    # And every helper submodule must be present. A partial ship silently breaks
    # imports at the first submodule lookup.
    for submodule in (
        *_BUILD_SUPPORT_MODULES,
        "find_compiler.py",
        "openmp_detect.py",
        "__init__.py",
    ):
        target = f"jamma/_build_support/{submodule}"
        assert any(name.endswith(target) for name in wheel_names), (
            f"Wheel {wheel_path.name} missing {target}. jamma._build_support "
            "must ship complete — a partial ship breaks runtime recompile."
        )
